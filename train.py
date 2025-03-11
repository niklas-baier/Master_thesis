from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Final
import pandas as pd
from transformers import Seq2SeqTrainingArguments, WhisperTokenizer, TrainerCallback, pipeline, AutoProcessor, EarlyStoppingCallback, AutoModelForSpeechSeq2Seq
from tqdm import tqdm
from transformers.models.whisper.modeling_whisper import *
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from augmentations import generate_noise_dataset
from peftModification import create_peft, create_peft_model
from decorators import suppress_specific_warnings, timing_decorator
import torchaudio
from pathlib import Path
import os
import datasets
import argparse
import torch
from datasets import Features
from functools import cache 

@dataclass(frozen=True)
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
    run_notes: str
    oversampling: int
    data_portion: str
    additional_tokens: str = field( default="N" )  # should additonal tokens be added
    checkpoint_path: str = field(default ="") # if a checkpoint is used for transcription what checkpoint should be loaded
    dataset_evaluation_part: str = field(default ="eval")
    beamforming: str = field(default ="N")
    num_trainable_parameters: int = 0
    SWAD: bool = False
    diffusion: str = field(default="N") 






from typing import Dict
@dataclass
class RunResults:
    # class that contains the runresults ID:133
    wer_per_session: Dict[str, float]
    wer_per_mictype: Dict[str, float]
    wer_per_special_token: Dict[str, float]
    wer_per_mic: Dict[str, float]



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


def generate_training_args(run_details: RunDetails)-> Seq2SeqTrainingArguments:
    #ID 169
    train_batch_size = 16
    per_device_eval_batch_size = 16
    if (run_details.environment == "bwcluster"):
        train_batch_size = 64
        per_device_eval_batch_size = 64
    max_steps = 100
    loggings_steps = 100
    save_steps = loggings_steps
    output_dir = f'trained_models/{run_details.task}/{run_details.dataset_name}/{run_details.version}/{run_details.model_id}'
    run_name = f'{run_details.task}_{run_details.dataset_name}_{run_details.version}_{run_details.model_id}'

    # Define base parameters
    base_args = {
        "output_dir": output_dir,
        "logging_dir": './logs',
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "evaluation_strategy": "no",
        "predict_with_generate": True,
        "save_steps": save_steps,
        "eval_steps": save_steps,
        "fp16": True,
        "logging_steps": loggings_steps,
        "remove_unused_columns": True,
        "eval_accumulation_steps": 4,
        "report_to": 'wandb',
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True
    }
    
    # Adjustments for PEFT version
    if run_details.version == "peft":
        peft_args = {   
            "per_device_train_batch_size": train_batch_size*4,
            "per_device_eval_batch_size": per_device_eval_batch_size*4,
            "learning_rate": 1e-4,
            "warmup_steps": 2,
            "max_steps": 200,
            "generation_max_length": 200,
            "torch_empty_cache_steps": 4,
            "label_names": ["labels"],

        }
        # Merge base and PEFT-specific arguments
        training_args = Seq2SeqTrainingArguments(**base_args, **peft_args)
    else:
        non_peft_args = {
            "per_device_train_batch_size": train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": 1e-5,
            "warmup_steps": 0,
            "max_steps": max_steps,
            "generation_max_length": 200,
            "run_name": run_name,
        }
        # Merge base and non-PEFT-specific arguments
        training_args = Seq2SeqTrainingArguments(**base_args, **non_peft_args)

    return training_args
  


@suppress_specific_warnings
@timing_decorator
def transcribe_audio(eval_df:pd.DataFrame, pipe, run_details):
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


def get_parser()-> argparse.ArgumentParser:
    # Implementation of ID:134
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
    parser.add_argument( '--run_notes', type=str, required=True,
                         help='Documentation of the run' )
    parser.add_argument('--dataset_evaluation_part', type=str, choices=['dev','eval'],required=False)
    parser.add_argument( '--oversampling_clean_data', type=int, choices=[1,2,3,4,5,6,7,8,9,10], required= True )
    parser.add_argument( '--checkpoint', type=str, required=False )
    parser.add_argument( '--data_portion', type=str, choices=["clean-only", "far-only", "all"], required=True )
    parser.add_argument( '--beamforming', type=str, choices=["Y","N"], required=False )
    parser.add_argument('--SWAD', type=bool, required=False)
    parser.add_argument('--diffusion', type=str, required=False, choices=['Y','N'])




    return parser

def add_prediction_column(words:pd.Series, labels_trained:pd.Series, temp:pd.Series)-> pd.Series:
    if words == labels_trained:
        return temp
    else:
        print( "words " + words )
        print( "labels_trained " + labels_trained )
        return temp





def get_model_size(model_id:str)-> str:
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

from functools import *
# partials to ensure that the right arguments are always provided ( same as a getter)
#ID157
get_tokenizer = partial(WhisperTokenizer.from_pretrained, language="English", task="transcribe", use_fast=True)
#ID159
get_Processor = partial(AutoProcessor.from_pretrained, language='en', task="transcribe",use_fast=True )
#ID158
get_plain_model = partial(AutoModelForSpeechSeq2Seq.from_pretrained,low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa")
_cached_tokenizer: Optional[WhisperTokenizer] = None
_cached_model: Optional[AutoModelForSpeechSeq2Seq] = None
_cached_processor: Optional[AutoProcessor] = None
@cache
def create_tokenizer_model_processor(run_details:RunDetails, torch_dtype:torch.dtype)-> Tuple[WhisperTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor]:
    #ID156
    from transformers import AutoConfig, AutoModel
    global _cached_tokenizer, _cached_model, _cached_processor
    if (run_details.checkpoint_path != ""):
        path_of_model = run_details.checkpoint_path
    else:
        path_of_model = run_details.model_id
    tokenizer = get_tokenizer(run_details.model_id)
    tokenizer.set_prefix_tokens( language="english" )
   
    model = WhisperForConditionalGeneration.from_pretrained(path_of_model)


    num_params = sum( p.numel() for p in model.parameters() if p.requires_grad )

    print( f"Number of trainable parameters: {num_params}" )
    processor = get_tokenizer(run_details.model_id)
    if ("large" or "medium") in run_details.model_id:
        processor = get_Processor(run_details.model_id)
        model.generation_config.language = "English"
        model.generation_config.forced_decoder_ids = None
        model.generation_config.task = "transcribe"
        model._set_language_and_task( language="en", task="transcribe", is_multilingual=True,
                                      generation_config=model.generation_config )
        model.config.apply_spec_augment = True
        # model.model_tags /
    else:
        processor = get_Processor(run_details.model_id)
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
        #model = create_peft(run_details)
        from peftModification import alterative_peft
        model = alterative_peft(run_details, model)
    elif run_details.version == "last-layer":
        model = freeze_all_layers_but_last( model )

  

        
    _cached_tokenizer, _cached_model, _cached_processor = tokenizer, model, processor
    return tokenizer, model, processor
def get_cached_components() -> Tuple[WhisperTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor]:
    """
    Returns the cached components. Raises  a error if called before components' initializiation.
    """
    if any(x is None for x in (_cached_tokenizer, _cached_model, _cached_processor)):
        raise RuntimeError("Components not yet initialized. Must call create_tokenizer_model_processor.")
    return _cached_tokenizer, _cached_model, _cached_processor
def get_cached_tokenizer() :
    return _cached_tokenizer
def generate_datasets(run_details:RunDetails, features:Features, args:argparse, expanded_df:pd.DataFrame, dev_df:pd.DataFrame, eval_df:pd.DataFrame)-> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    #ID 160
    from preprocessing import generate_test_features
    from preprocessing import Hug_dataset_creation, generate_dataset_paths, mapped_dataset_exists, map_datasets
    train_dataset_path, eval_dataset_path, test_dataset_path = generate_dataset_paths(
        run_details=run_details )\

    eval_df.to_csv( "shuffled_test_dataframe.csv" )
    if run_details.developer_mode == "Y":
        eval_df = eval_df.head(100)
    if not (mapped_dataset_exists( train_dataset_path )):
        # save the data from the dataframe in a csv fails if the file already exists
        eval_df.to_csv( "shuffled_test_dataframe.csv")
        print( "dataset not mapped yet" )
        dataset_paths = {"train": train_dataset_path, "eval": eval_dataset_path, "test": test_dataset_path}
        train_dataset = Hug_dataset_creation( expanded_df, run_details.developer_mode, features,test_dataset=False )
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
