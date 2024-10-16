from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Final
import pandas as pd
from transformers import Seq2SeqTrainingArguments, WhisperTokenizer, TrainerCallback, pipeline, AutoProcessor
from tqdm import tqdm
from transformers.models.whisper.modeling_whisper import *
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from augmentations import generate_noise_dataset
from peftModification import create_peft, create_peft_model
from test_Whisper import suppress_specific_warnings, timing_decorator
import torchaudio
from pathlib import Path
import os
import datasets


@dataclass
@final
class RunDetails:
    dataset_name: str  #name of the dataset
    model_id: str  #name of the model
    version: str  # plain model or modifed ?
    environment: str  # laptop or cluster
    train_state: str  # training wanted ?
    date: str  # current date
    device: str  # cuda
    task: str  #classification or transciption or joint
    developer_mode: str  # small datasets?
    augmentation: str  # use of synthetic noise augmentation
    additional_tokens: str = field( default="N" )  # should additonal tokens be added
    checkpoint_path: str = field(default ="") # if a checkpoint is used for transcription what checkpoint should be loaded





@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad( input_features, return_tensors="pt" )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad( label_features, return_tensors="pt" )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill( labels_batch.attention_mask.ne( 1 ), -100 )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def generate_training_args(run_details):
    train_batch_size = 8
    per_device_eval_batch_size = 8
    max_steps = 300
    loggings_steps = 100
    save_steps = 200
    output_dir = f'trained_models/{run_details.task}/{run_details.dataset_name}/{run_details.version}/{run_details.model_id}'
    run_name = f'{run_details.task}_{run_details.dataset_name}_{run_details.version}_{run_details.model_id}'
    if run_details.version == "peft":
        training_args = Seq2SeqTrainingArguments(
            output_dir="reach-vb/test",  # change to a repo name of your choice
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-3,
            warmup_steps=50,
            num_train_epochs=1,
            evaluation_strategy="steps",
            fp16=True,
            per_device_eval_batch_size=8,
            generation_max_length=128,
            logging_steps=100,
            max_steps=100,  # only for testing purposes, remove this from your final run :)
            remove_unused_columns=False,
            # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            label_names=["labels"],  # same reason as above
            )
        return training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir='./logs',
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=0,
        max_steps=max_steps,  # 4000
        gradient_checkpointing=True,
        eval_strategy="steps",
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=100,
        fp16=True,
        logging_steps=loggings_steps,
        report_to='wandb',
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=True

        )
    return training_args


@suppress_specific_warnings
@timing_decorator
def transcribe_audio(eval_df, pipe, run_details):
    # transcription of the test_data
    for i in tqdm( range( eval_df.shape[0] ) ):

        audio, _ = torchaudio.load( eval_df['file_path'][i], frame_offset=eval_df['startframe'][i],
                                    num_frames=eval_df['num_frames'][i] )
        audio_data = audio.squeeze().numpy()
        print( audio_data.shape )
        if ("large") in run_details.model_id:
            result = pipe( audio_data, generate_kwargs={"language": "english"} )
        else:
            result = pipe( audio_data )

        eval_df.loc[i, 'results'] = result['text']

    return eval_df


class PrintTrainableParamsCallback( TrainerCallback ):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        model_parameters = filter( lambda p: p.requires_grad, model.parameters() )
        num_trainable_params = sum( [torch.tensor( p.numel() ) for p in model_parameters] )
        print( f"Number of trainable parameters: {num_trainable_params}" )


def freeze_all_layers_but_last(model):
    for param in model.parameters():
        param.requires_grad = False
        last_layer = list( model.children() )[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    return model


def get_parser():
    import argparse
    parser = argparse.ArgumentParser( description="RunDetails argument parser" )

    parser.add_argument( '--dataset_name', type=str, required=True, help='Name of the dataset' )
    parser.add_argument( '--model_id', type=str, required=True, help='Name of the model' )
    parser.add_argument( '--version', type=str, required=True, help='Model version (plain or modified)' )
    parser.add_argument( '--environment', type=str, choices=['laptop', 'cluster', 'bwcluster'], required=True,
                         help='Execution environment (laptop or cluster)' )
    parser.add_argument( '--train_state', type=str, choices=['T', 'NT'], required=True,
                         help='Is training wanted? (T(raining) / N(o)T(raining)' )
    parser.add_argument( '--date', type=str, default=datetime.now().strftime( '%Y-%m-%d' ),
                         help='Current date (default: today)' )
    parser.add_argument( '--device', type=str, required=True, help='Device to be used (e.g., cuda or cpu)' )
    parser.add_argument( '--task', type=str, choices=['classification', 'transcribe', 'joint'], required=True,
                         help='Task type (classification, transcription, or joint)' )
    parser.add_argument( '--developer_mode', type=str, choices=['Y', 'N'], required=True,
                         help='Developer mode (yes for small datasets, no for full training)' )
    parser.add_argument( '--augmentation', type=str, choices=['Y', 'N'], required=True,
                         help='Use synthetic noise augmentation can be Y(es) or N(o)' )
    parser.add_argument( '--additional_tokens', type=str, choices=['Y', 'N'], required=True,
                         help='Add additonal tokens of the dataset to the network can be Y(es) or N(o)' )

    return parser


def add_prediction_column(words, labels_trained, temp):
    if words == labels_trained:
        return temp
    else:
        print( "words " + words )
        print( "labels_trained " + labels_trained )
        return temp





def get_model_size(model_id):
    import re

    # Regex pattern splits on substrings "; " and ", "
    components = re.split( '-|/|.|', model_id )
    model_size = components[2]
    return model_size


def transcribe_raw(eval_df, model, processor, run_details, torch_dtype):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=run_details.device

        )
    model_size = get_model_size( run_details.model_id )
    transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
    if Path( transcription_csv_path ).is_file():
        print( "transcription csv already exists" )
        print( transcription_csv_path )
    else:
        eval_df = transcribe_audio( eval_df=eval_df, pipe=pipe, run_details=run_details )
        eval_df.to_csv( transcription_csv_path, index=False )

    return transcription_csv_path


def create_tokenizer_model_processor(run_details, torch_dtype):
    tokenizer = WhisperTokenizer.from_pretrained( run_details.model_id, task="transcribe", language="en" )
    if (run_details.checkpoint_path != ""):
        path_of_model = run_details.checkpoint_path
    else:
        path_of_model = run_details.model_id

    tokenizer.set_prefix_tokens( language="english" )
    model = WhisperForConditionalGeneration.from_pretrained(
        path_of_model, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch_dtype,
        )


    num_params = sum( p.numel() for p in model.parameters() if p.requires_grad )

    print( f"Number of trainable parameters: {num_params}" )
    processor = AutoProcessor.from_pretrained( run_details.model_id, language='en', task="transcribe" )
    if ("large" or "medium") in run_details.model_id:
        processor = AutoProcessor.from_pretrained( run_details.model_id, language='en', task="transcribe" )
        model.generation_config.language = "English"
        model.generation_config.task = "transcribe"
        model._set_language_and_task( language="en", task="transcribe", is_multilingual=True,
                                      generation_config=model.generation_config )
        # model.model_tags /
    else:
        processor = AutoProcessor.from_pretrained( run_details.model_id )
    model.generation_config.forced_decoder_ids = None
    if run_details.additional_tokens == "Y":
        # define new tokens to add to vocab
        new_tokens = ['[laugh]', '[unintelligible]', '[noise]', ]
        # check if the new tokens are already in the vocabulary
        new_tokens = set( new_tokens ) - set( tokenizer.vocab.keys() )
        # add the tokens to the tokenizer vocabulary
        tokenizer.add_tokens( list( new_tokens ) )
        # add new random embeddings for the appended tokens
        model.resize_token_embeddings( len( tokenizer ) )

    if run_details.version == 'peft':
        model = create_peft(run_details)
    elif run_details.version == "last-layer":
        model = freeze_all_layers_but_last( model )
    return tokenizer, model, processor


def generate_datasets(run_details, features, args, expanded_df, dev_df, eval_df):
    from preprocessing import generate_test_features
    from preprocessing import Hug_dataset_creation, generate_dataset_paths, mapped_dataset_exists, map_datasets
    train_dataset_path, eval_dataset_path, test_dataset_path = generate_dataset_paths(
        run_details=run_details )\

    eval_df.to_csv( "shuffled_test_dataframe.csv" )
    if run_details.developer_mode == "Y":
        eval_df = eval_df.head(100)
    eval_df.to_csv( "shuffled_test_dataframe.csv" )
    if not (mapped_dataset_exists( train_dataset_path )):
        print( "dataset not mapped yet" )
        dataset_paths = {"train": train_dataset_path, "eval": eval_dataset_path, "test": test_dataset_path}
        train_dataset = Hug_dataset_creation( expanded_df, run_details.developer_mode, features, test_dataset=False )
        eval_dataset = Hug_dataset_creation( dev_df, run_details.developer_mode, features, test_dataset=False )
        test_features = generate_test_features(run_details)
        test_dataset = Hug_dataset_creation( eval_df, run_details.developer_mode, test_features, test_dataset=True )
        map_datasets( run_details=run_details, train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      test_dataset=test_dataset, dataset_paths=dataset_paths )


    train_dataset = datasets.load_from_disk( train_dataset_path )
    # remove the unncessary columns

    def drop_columns(dataset):
        columns_to_be_dropped = [x for x in dataset.column_names if x not in ['input_features', 'labels']]
        dataset = dataset.remove_columns( columns_to_be_dropped )
        return dataset
    train_dataset = drop_columns(train_dataset)


    eval_dataset = datasets.load_from_disk( eval_dataset_path )
    eval_dataset = drop_columns(eval_dataset)

    test_dataset = datasets.load_from_disk( test_dataset_path )
    test_dataset = drop_columns(test_dataset)

    #tsne_sample_dataset = datasets.load_from_disk(tsne_dataset_path)

    if args.augmentation == "Y":
        noisy_train_dataset_path = "noise" + train_dataset_path
        if not (mapped_dataset_exists( noisy_train_dataset_path )):
            noisy_train_dataset_path = generate_noise_dataset( expanded_df=expanded_df, run_details=run_details,
                                                               features=features )
        train_dataset = datasets.load_from_disk( noisy_train_dataset_path )

    return train_dataset, eval_dataset, test_dataset
