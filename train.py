from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Final
import pandas as pd
from transformers import WhisperTokenizer, TrainerCallback, pipeline
from tqdm import tqdm
from test_Whisper import suppress_specific_warnings, timing_decorator
import torchaudio
from pathlib import Path
import os

@dataclass
@final
class RunDetails:
    dataset_name: str #name of the dataset
    model_id: str #name of the model
    version: str  # plain model or modifed ?
    environment: str # laptop or cluster
    train_state: str # training wanted ?
    date: str # current date
    device: str # cuda
    task: str #classification or transciption or joint
    developer_mode: str # small datasets?
    augmentation: str # use of synthetic noise augmentation
    additional_tokens : str = field(default="Y") # should additonal tokens be added

@dataclass
class DataDetails:
    num_speakers: int
    speakers: [str]
    num_origins: int
    origins: [str]
    num_locations: int
    locations: [str]
    ref_chimes: [str]
    num_ref_chimes: int
    ref_dipcos: [str]
    num_ref_dipcos: int
    session_ids: [str]
    num_session_ids: int
    genders : [str]
    num_genders: int
    nativitys: [str]
    num_nativitys: int
    mother_tongues: [str]
    num_mother_tongues: int


from transformers.models.whisper.modeling_whisper import *
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



def generate_training_args(run_details):
    train_batch_size = 16
    per_device_eval_batch_size = 16
    max_steps = 300
    loggings_steps = 100
    save_steps = 200
    output_dir = f'trained_models/{run_details.task}/{run_details.dataset_name}/{run_details.version}/{run_details.model_id}'
    run_name = f'{run_details.task}_{run_details.dataset_name}_{run_details.version}_{run_details.model_id}'
    if run_details.environment == 'cluster':
        max_steps = 4000
        if 'tiny' in run_details.model_id:
            train_batch_size = 64
            per_device_eval_batch_size = 64
    elif run_details.environment == 'bwcluster':
        train_batch_size = 64
        per_device_eval_batch_size = 64
        max_steps = 4000
    return train_batch_size, per_device_eval_batch_size, max_steps, loggings_steps, save_steps, output_dir,run_name

@suppress_specific_warnings
@timing_decorator
def transcribe_audio(eval_df, pipe, run_details):
    # transcription of the test_data
    for i in tqdm(range(eval_df.shape[0])):

        audio, _ = torchaudio.load(eval_df['file_path'][i], frame_offset=eval_df['startframe'][i],
                                   num_frames=eval_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        if ("openai/whisper-large") in run_details.model_id:
            result = pipe(audio_data, generate_kwargs={"language": "english"})
        else:
            result = pipe(audio_data)

        eval_df.loc[i, 'results'] = result['text']

    return eval_df




class PrintTrainableParamsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_trainable_params = sum([torch.tensor(p.numel()) for p in model_parameters])
        print(f"Number of trainable parameters: {num_trainable_params}")

def freeze_all_layers_but_last(model):
    for param in model.parameters():
        param.requires_grad = False
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    return model




def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="RunDetails argument parser")

    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_id', type=str, required=True, help='Name of the model')
    parser.add_argument('--version', type=str, required=True, help='Model version (plain or modified)')
    parser.add_argument('--environment', type=str, choices=['laptop', 'cluster','bwcluster'], required=True,
                        help='Execution environment (laptop or cluster)')
    parser.add_argument('--train_state', type=str, choices=['T', 'NT'], required=True,
                        help='Is training wanted? (T(raining) / N(o)T(raining)')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='Current date (default: today)')
    parser.add_argument('--device', type=str, required=True, help='Device to be used (e.g., cuda or cpu)')
    parser.add_argument('--task', type=str, choices=['classification', 'transcribe', 'joint'], required=True,
                        help='Task type (classification, transcription, or joint)')
    parser.add_argument('--developer_mode', type=str, choices=['Y', 'N'], required=True,
                        help='Developer mode (yes for small datasets, no for full training)')
    parser.add_argument('--augmentation', type=str, choices=['Y', 'N'], required=True,
                        help='Use synthetic noise augmentation can be Y(es) or N(o)')
    parser.add_argument('--additional_tokens', type=str, choices=['Y', 'N'], required=True,
                        help='Add additonal tokens of the dataset to the network can be Y(es) or N(o)')

    return parser
def add_prediction_column(words, labels_trained, temp):
    if words == labels_trained:
        return temp
    else:
        print("words " + words)
        print("labels_trained " + labels_trained)
        return temp

def transcribe_results(*, test_dataset, trainer, transcription_csv_path,run_details):
    trainer.evaluate(eval_dataset=test_dataset)
    results_directory = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}")
    file_path = os.path.join(results_directory, "results.json")
    results = pd.read_json(file_path)
    test_df = pd.read_csv("shuffled_test_dataframe.csv")
    assert results.shape[0] == test_df.shape[0]
    test_df['labels_trained'] = results['labels']
    test_df['temp'] = results['predictions']
    test_df['results'] = test_df.apply(lambda row: add_prediction_column(row['words'],row['labels_trained'], row['temp']), axis=1)
    test_df.drop(columns=['temp','labels_trained'])
    model_size = get_model_size(run_details.model_id)
    trained_path = f'{run_details.dataset_name}_eval_{model_size}_trained.csv'
    test_df.to_csv(trained_path, index=False)
    return trained_path


def get_model_size(model_id):
    import re

    # Regex pattern splits on substrings "; " and ", "
    components = re.split('-|/|.|', model_id)
    model_size = components[2]
    return model_size

def transcribe_raw(eval_df,model,processor, run_details,torch_dtype):

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
    model_size = get_model_size(run_details.model_id)
    transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
    if (Path(transcription_csv_path).is_file()):
        print("transcription csv already exists")
        print(transcription_csv_path)
    else:
        eval_df = transcribe_audio(eval_df=eval_df, pipe=pipe, run_details=run_details)
        eval_df.to_csv(transcription_csv_path, index=False)

    return transcription_csv_path


