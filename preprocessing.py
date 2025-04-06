from __future__ import annotations
import os
from absl import flags

import glob
from datetime import datetime
import re
import pandas as pd
from transformers import WhisperFeatureExtractor
import augmentations
from datasets import Dataset
pd.options.mode.copy_on_write = True
import torchaudio
import pprint
from typing import List, Any, Callable 
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import Features, Value
import numpy as np
from typing import Any
from argparse import Namespace
def dipco_paths(dataset_path:str)-> tuple[str, str, str, str, str]:

    dev_path = os.path.join(dataset_path, 'audio/dev')
    eval_path = os.path.join(dataset_path, 'audio/eval')
    transcript_dev_path = os.path.join(dataset_path, 'transcriptions/dev')
    transcript_eval_path = os.path.join(dataset_path, 'transcriptions/eval')
    return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path

def chime_paths(dataset_path:str, run_details:"RunDetails")-> tuple[str, str, str, str, str]:
    dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path)

    train_path = os.path.join(dataset_path, 'audio/train')
    transcript_train_path = os.path.join(dataset_path, 'transcriptions/train')
    if run_details.dataset_evaluation_part == "dev":
        dev_path, eval_path = eval_path, dev_path
        transcript_dev_path, transcript_eval_path = transcript_eval_path, transcript_dev_path
    return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path

def setup_paths(environment:str, dataset_name:str, run_details)-> tuple[str, str, str, str, str, str | None, str | None]:
    # This method sets the base paths of the dipco and Chim6Dataset for different environments ID:126
    dataset_path = "/project/data_asr/dipco/Dipco"
    bw_workplace_path = '/pfs/work7/workspace/scratch/uhicv-blah'
    if environment == 'cluster':
        if dataset_name == "Chime6":
            dataset_path = '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/'#'/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
            return chime_paths(dataset_path=dataset_path, run_details=run_details)



        else:
            dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path=dataset_path)
            if run_details.dataset_evaluation_part == "dev":
                dev_path, eval_path = eval_path,dev_path
                transcript_dev_path, transcript_eval_path = transcript_eval_path,transcript_dev_path
            return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path,'',''
    elif environment == 'bwcluster':

        if dataset_name == "Chime6":
            dataset_path = '/home/kit/stud/uhicv'  # '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
            dataset_path = os.path.join(bw_workplace_path, "Chime6")
            return chime_paths(dataset_path=dataset_path)



        else:
            dataset_path = f'{bw_workplace_path}/Dipco'
            dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(
                dataset_path=dataset_path)
            if run_details.run_notes =='facebook denoising':
                eval_path = '/pfs/work7/workspace/scratch/uhicv-blah/facebook_denoiser/data/eval/testable_results'
            if run_details.run_notes == 'noise reduce':
                eval_path = '/pfs/work7/workspace/scratch/uhicv-blah/noise_reduce/Dipco/eval'
            if run_details.run_notes == 'storm':
                eval_path = '/pfs/work7/workspace/scratch/uhicv-blah/storm/whisper_inference_wavs/audio/eval'
            if run_details.dataset_evaluation_part == "dev":
                dev_path, eval_path = eval_path,dev_path
                transcript_dev_path, transcript_eval_path = transcript_eval_path,transcript_dev_path
            return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, '', ''


    else:
        if dataset_name == "Chime6":
            dataset_path = "/home/niklas/Downloads/Master/espnet/egs2/chime7_task1/asr1/datasets/ChIME6/"
            return chime_paths(dataset_path)
        else:
            dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path=dataset_path)
            if run_details.dataset_evaluation_part == "dev":
                dev_path, eval_path = eval_path,dev_path
                transcript_dev_path, transcript_eval_path = transcript_eval_path,transcript_dev_path
            return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path ,'',''

def generate_dataset_paths(run_details:"RunDetails")-> tuple[str, str, str]:
    model_str = extract_letters(run_details.model_id)
    train_dataset_path = f"{model_str}_{run_details.dataset_name}_train.hf"  # TODO
    eval_dataset_path = f"{model_str}_{run_details.dataset_name}_eval.hf"
    test_dataset_path = f"{model_str}_{run_details.dataset_name}_test.hf"
    if run_details.environment == 'bwcluster':
        bw_workplace_path = '/pfs/work7/workspace/scratch/uhicv-blah'
        train_dataset_path = os.path.join(bw_workplace_path,train_dataset_path)
        eval_dataset_path = os.path.join(bw_workplace_path,eval_dataset_path)
        test_dataset_path = os.path.join(bw_workplace_path,test_dataset_path)
    return train_dataset_path, eval_dataset_path, test_dataset_path

def get_formated_date() -> str:
    return datetime.now().strftime("%m/%d/%Y")


def extract_prefix(file_path: str) -> str:
    pattern = r'^(.*)\.json$'
    match = re.search(pattern, file_path)
    if match:
        prefix = match.group(1)
        return prefix
    else:
        raise ValueError


def list_json_files(directory:str)-> list[str]:
    # Construct the file path pattern
    pattern = os.path.join(directory, '*.json')

    # Use glob to get a list of files matching the pattern
    json_files = glob.glob(pattern)

    return json_files


def load_and_concatenate_json_files(directory:str)-> pd.DataFrame:
    #implementation of ID136
    json_files = list_json_files(directory)
    # List to hold individual DataFrames
    data_frames = []

    for json_file in json_files:
        # Read the JSON file into a DataFrame
        df = pd.read_json(json_file)
        data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df


def expand_start_time(row: pd.Series) -> pd.DataFrame:
    start_time_dict = row['start_time']
    rows = []
    for key, time_str in start_time_dict.items():
        new_row = row.copy()
        new_row['audio'] = key
        new_row['start'] = time_str
        rows.append(new_row)
    return pd.DataFrame(rows)


# @time_to_seconds: Function to convert time string to seconds
def time_to_seconds(time_str:str)-> float:
    h, m, s = map(float, time_str.split(':'))

    return h * 3600 + m * 60 + s


def chime_get_seconds_from_time(time_obj):
    #ID150
    # Extract hours, minutes, and seconds from the Timestamp object
    h = time_obj.hour
    m = time_obj.minute
    s = time_obj.second
    ms = time_obj.microsecond // 1000
    # Convert the time to seconds
    return h * 3600 + m * 60 + s + ms / 1000


def get_corresponding_end_time(dict: dict, key: str)-> float:
    end_time = [v for k, v in dict if k == key]
    return end_time


def generate_microphone_paths(row:pd.Series,mode_path:str)-> List[str]:
    # Implementation of ID141

    paths = []
    if row['audio'] == 'close-talk':
        for i in range(1, 2):
            path = f"{mode_path}/{row['session_id']}_{row['speaker_id']}.wav"
            paths.append(path)
    else:
         for i in range(1,2):
             path = f"{mode_path}/{row['session_id']}_{row['audio']}.CH{i}.wav"
             paths.append(path)
    
    return paths


def chime_generate_microphone_paths(row:pd.Series, mode_path:str) -> List[str]:
    #ID151
    paths = []
    dataset_paths = Paths.get_instance()
    if mode_path == dataset_paths.train_path:
        for i in range(1, 2):
            for j in range(2,3):
                path = f"{mode_path}/{row['session_id']}_U0{j}.CH{i}.wav"
                if (j == 3 or j==4) :
                    paths.append(path)

                else:
                    paths.append(path)



    else:
        for i in range(1, 2):
            path = f"{mode_path}/{row['session_id']}_{row['ref']}.CH{i}.wav"
            paths.append(path)



    path = f"{mode_path}/{row['session_id']}_{row['speaker']}.wav"
    paths.append(path)
    return paths



def get_Frames(starting_second: float, sample_rate: int, end_second: float) -> List[int]:
    # change the seconds to frames
    # ID152
    return [int(starting_second * sample_rate), int(end_second * sample_rate)]


def validate_frames_column(frames_list):
    #ID 153
    return len(frames_list) == 2


def chime_parsing(dataframe:pd.DataFrame, run_details,mode_path:str)-> pd.DataFrame:
    dataframe['start'] = dataframe['start_time'].apply(chime_get_seconds_from_time)
    dataframe['end'] = dataframe['end_time'].apply(chime_get_seconds_from_time)
    dataframe['file_path'] = dataframe.apply(chime_generate_microphone_paths, axis=1,args=(mode_path,))
    dataframe = dataframe.explode('file_path').reset_index(drop=True)
    dataframe['frames'] = dataframe.apply(lambda row: get_Frames(row['start'], 16000, row['end']), axis=1)
    dataframe['duration'] = dataframe.apply(lambda row: row['end'] - row['start'], axis=1)
    if dataframe['frames'].isnull().any():
        raise ValueError("The 'frames' column contains null values.")
    if not dataframe['frames'].apply(validate_frames_column).all():
        raise ValueError(
            "Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")
    dataframe[['startframe', 'endframe']] = pd.DataFrame(dataframe['frames'].tolist(), index=dataframe.index)
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    #dataframe = dataframe.query('words !=""')

    #dataframe = remove_duplicates( dataframe ) this removes 2 values that are apparently the same
    dataframe = drop_chime_columns( dataframe, mode_path, run_details )
    if run_details.developer_mode == 'Y':
        return dataframe.sample(n=100, random_state=42)
    else:
        return dataframe


def drop_chime_columns(dataframe:pd.DataFrame, mode_path:str, run_details)-> pd.DataFrame:

    #ID154
    if run_details.task == 'classification':
        dataframe.drop(
            columns=['end_time', 'start_time', 'duration', 'frames', 'start', 'end', 'location', 'ref', 'endframe',
                     'session_id', 'words'], inplace=True )  # don't drop the speaker but wordss for the time being
    else:
        dataset_paths = Paths.get_instance()
        if mode_path == dataset_paths.train_path:
            dataframe.drop(
                columns=['end_time', 'start_time', 'endframe',
                         'session_id', 'speaker', 'duration', 'frames', 'start', 'end'], inplace=True )

        else:
            dataframe.drop(
                columns=['end_time', 'start_time', 'ref', 'endframe',
                         'session_id', 'speaker', 'duration', 'frames', 'start', 'end', 'location'],
                inplace=True )  # additonally drop location and ref
    dataframe.reset_index( drop=True, inplace=True )
    return dataframe


def dipco_parsing(dataframe:pd.DataFrame, run_details:"RunDetails", mode_path:str)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Implementation of ID139
    # Apply the function to each row and concatenate the results
    dataframe = pd.concat([expand_start_time(row) for _, row in dataframe.iterrows()], ignore_index=True)
    # Drop the original 'start_time' column
    dataframe = dataframe.drop(columns=['start_time'])
    dataframe['start'] = dataframe['start'].apply(time_to_seconds)
    dataframe['end'] = dataframe.apply(lambda row: row['end_time'][row['audio']], axis=1)
    dataframe['end'] = dataframe['end'].apply(time_to_seconds)
    dataframe = dataframe.drop(columns=['end_time'])
    # Apply the function to generate the paths for each row
    dataframe['file_path'] = dataframe.apply(generate_microphone_paths, axis=1, args=(mode_path,))
    # Expand the DataFrame to include the microphone paths
    dataframe = dataframe.explode('file_path').reset_index(drop=True) #doubles the size

    dataframe['frames'] = dataframe.apply(lambda row: get_Frames(row['start'], 16000, row['end']), axis=1) 
    
    # get the maximum speaking duration
    dataframe['duration'] = dataframe.apply(lambda row: row['end'] - row['start'], axis=1)
    if dataframe['frames'].isnull().any():
        raise ValueError("The 'frames' column contains null values.")
    if not dataframe['frames'].apply(validate_frames_column).all():
        raise ValueError(
            "Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")
    dataframe[['startframe', 'endframe']] = pd.DataFrame(dataframe['frames'].tolist(), index=dataframe.index)
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    dataframe = dataframe.rename(columns={'speaker_id':'speaker'}) # to give both datasets the same names
    # #dataframe['speaker_id_int'] = dataframe['speaker_id'].str.extract('(\d+)').astype(int) there are not the same persons in each dataset

    #train_dataframe,test_dataframe = train_test_split(dataframe, test_size=0.05, random_state=42)
    train_dataframe,test_dataframe = dipco_split_sessions(dataframe)
    train_dataframe = set_data_portion_of_training( run_details, train_dataframe )
    if run_details.augmentation == 'Y':
        # only take close micorphone samples with no background music
        #TODO only take close micorphone samples with no background music
        test_dataframe = add_noise_paths(test_dataframe)
    if run_details.beamforming == 'Y':
        beamformed_direc = os.path.join(os.getcwd(), 'beamforming')
        test_dataframe['file_path'] = test_dataframe['file_path'].apply(lambda x: os.path.join(beamformed_direc, os.path.basename(x)))
    if run_details.diffusion == 'Y':
        assert(run_details.beamforming != 'Y')
        beamformed_direc = os.path.join(os.getcwd(), 'outputsfromdiffusionmodel')
        test_dataframe['file_path'] = test_dataframe['file_path'].apply(lambda x: os.path.join(beamformed_direc, os.path.basename(x)))
        beamformed_direc = os.path.join(os.getcwd(), 'training_data_from_diffusion_model')
        train_dataframe['file_path'] = train_dataframe['file_path'].apply(lambda x: os.path.join(beamformed_direc, os.path.basename(x)))
    test_dataframe['try'] = test_dataframe.index
    test_dataframe = test_dataframe.sort_values(by=['file_path', 'try'], ascending=[True, True])
    test_dataframe = test_dataframe.drop(columns = ['try'])

    train_dataframe = drop_columns_dipco(train_dataframe,run_details)
    test_dataframe = drop_columns_dipco(test_dataframe, run_details)
    train_dataframe, eval_dataframe = train_test_split( train_dataframe, test_size=0.05, random_state=42 )
    eval_dataframe.reset_index( drop=True, inplace=True )
    train_dataframe.reset_index(drop=True, inplace=True)
    test_dataframe.reset_index(drop=True, inplace=True)
    if run_details.developer_mode == 'Y':
        return train_dataframe.sample(n=100,random_state=42),eval_dataframe.sample(n=100, random_state=42), test_dataframe.sample(n=100, random_state=42)
    else:
        return train_dataframe, eval_dataframe,test_dataframe


def set_data_portion_of_training(run_details:"RunDetails", train_dataframe:pd.DataFrame)-> pd.DataFrame:
    # ID144
    if run_details.data_portion == "clean-only":
        train_dataframe = augmentations.filter_p_audio( train_dataframe )
    if run_details.data_portion == "far-only":
        train_dataframe = augmentations.filter_far_audio( train_dataframe )
    return train_dataframe


def add_noise_paths(dataframe:pd.DataFrame)-> pd.DataFrame:
    #ID 146
    dataframe = get_clean_audio_without_music( dataframe )
    noise_paths = augmentations.get_noises()
    dataframe['noise_paths'] = noise_paths['file_path']
    np.random.seed( 42 )
    sampled_indices = np.random.randint( 0, len( noise_paths ), size=len( dataframe ) )
    dataframe['noise_path'] = noise_paths['file_path'].values[sampled_indices]
    return dataframe


def dipco_split_sessions(dataframe:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
    #Implementation of ID 143
    session_ids = dataframe['session_id'].unique()
    eval_session = session_ids[0]
    if 'S03' in session_ids:# necessary becase the unique function does not reqturn the same across devices: to compare with results on IAR-gpu 
        eval_session = 'S03'
    if 'S04' in session_ids:
        eval_session = 'S04'
    eval_dataframe = dataframe[dataframe['session_id'] == eval_session]
    train_dataframe = dataframe[dataframe['session_id'] != eval_session]
    return train_dataframe, eval_dataframe

def drop_columns_dipco(dataframe:pd.DataFrame, run_details:"RunDetails")-> pd.DataFrame:
    if run_details.task =='classification':
        dataframe.drop(
            columns=['endframe', 'session_id', 'gender', 'nativeness', 'mother_tongue', 'audio', 'start',
                     'end', 'endframe', 'duration', 'frames', 'ref', 'words'], inplace=True) # don't drop the speaker ID but drop words
    else:
        dataframe.drop(
            columns=['endframe', 'session_id', 'speaker', 'gender', 'nativeness', 'mother_tongue', 'audio', 'start',
                     'end', 'endframe', 'duration', 'frames', 'ref'], inplace=True)

    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def _build_basic_features(run_details:"RunDetails") -> dict:
    basic_features = {
        'file_path': Value( 'string' ),
        'startframe': Value( 'int64' ),
        'num_frames': Value( 'int64' )
        }
    if run_details.augmentation == 'Y':
        basic_features.update( {
            'noise_path': Value( 'string' ),
            'words': Value( 'string' )
            } )
    else:
        basic_features['words'] = Value( 'string' )

    return basic_features


def generate_features(run_details:"RunDetails") -> Features:
    basic_features = _build_basic_features( run_details )
    return Features( basic_features )


def generate_test_features(run_details:"RunDetails") -> Features:
    basic_features = _build_basic_features( run_details )
    basic_features['results'] = Value( 'string' )
    return Features( basic_features )


def Hug_dataset_creation(expanded_df:pd.DataFrame, developer_mode:str,features:Features,test_dataset:bool)-> Dataset:
    #ID 161
    # selects subset if developer mode is selected
    if expanded_df is None:
        return None
    expanded_df.reset_index(drop=True, inplace=True)






    dataset = Dataset.from_pandas(expanded_df, features=features)
    #TODO seems to change everytime
    shuffled_dataset = dataset


    if developer_mode == 'Y':
        selection_size = min(len(shuffled_dataset), 100)

        shuffled_dataset = shuffled_dataset.select(range(selection_size))
        if test_dataset:
            shuffled_test_dataframe = shuffled_dataset.to_pandas()
            shuffled_test_dataframe.to_csv("shuffled_test_dataframe.csv")
        return shuffled_dataset

    if test_dataset:

        shuffled_test_dataframe = shuffled_dataset.to_pandas()
        shuffled_test_dataframe.to_csv("shuffled_test_dataframe.csv")

    return shuffled_dataset
from functools import *
# partial to ensure that the feature extractor has the right arguments
get_Feature_extractor = partial(WhisperFeatureExtractor.from_pretrained, language='en', task="transcribe" )

def prepare_dataset_seq2seq(batch):
    # load and resample audio data from 48 to 16kHz

    from train import get_cached_tokenizer, get_cached_components
    tokenizer,_, processor= get_cached_components()

    feature_extractor = processor.feature_extractor


    feature_extractor = processor.feature_extractor
    waveform, sample_rate = torchaudio.load(batch["file_path"], frame_offset=batch["startframe"],
                                            num_frames=batch["num_frames"])
    input = waveform.squeeze().numpy()
    batch["input_features"] = feature_extractor(input, sampling_rate=sample_rate).input_features[0]

    # compute log-Mel input features from input audio array

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["words"]).input_ids
    return batch
def prepare_noisedataset_seq2seq(batch):
    # load and resample audio data from 48 to 16kHz


    from whisper_main import run_details, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run_details.model_id)

    waveform, sample_rate = torchaudio.load(batch["file_path"], frame_offset=batch["startframe"],
                                            num_frames=batch["num_frames"])
    #TODO overlay the audioforms

    waveform = augmentations.apply_noises(filepath_original_sound=batch["file_path"],filepath_synthetic_noise=batch["filepath_noise"])
    #TODO shape is 3,43453000

    input = waveform
    batch["input_features"] = feature_extractor(input, sampling_rate=sample_rate).input_features[0]

    # compute log-Mel input features from input audio array

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["words"]).input_ids

    return batch





def map_datasets(run_details:"RunDetails", train_dataset:Dataset,eval_dataset:Dataset, test_dataset:Dataset, dataset_paths:str)-> None:
    #ID 163
    if run_details.augmentation == "Y":
        mapping_function = prepare_noisedataset_seq2seq
    else:
        mapping_function = prepare_dataset_seq2seq


    map_and_store_datasets(run_details, train_dataset, eval_dataset, test_dataset, dataset_paths, mapping_function)





def map_and_store_datasets(run_details:"RunDetails", train_dataset:Dataset, eval_dataset:Dataset, test_dataset:Dataset, dataset_paths:dict, mapping_function:Callable[[Any], Dataset]) ->None:
    #ID 164
    if run_details.augmentation == "Y":
        mapping_function = prepare_noisedataset_seq2seq
    
    train_dataset = train_dataset.map(mapping_function)
    train_dataset.save_to_disk(dataset_paths['train'])
    mapping_function = prepare_dataset_seq2seq
    eval_dataset = eval_dataset.map(mapping_function)
    eval_dataset.save_to_disk(dataset_paths['eval'])
    test_dataset = test_dataset.map(mapping_function)
    test_dataset.save_to_disk(dataset_paths['test'])
    del test_dataset
    return


def mapped_dataset_exists(dataset_path:str)-> bool:
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        return True
    return False


def extract_special_token(label_string:str)->str:
    import re
    match = re.search(r'\[\w+\]', label_string)
    if match:
        return str(match.group(0))
    else:
        return "No token"

def extract_letters(input_string:str)->str:
    return ''.join([char for char in input_string if char.isalpha()])
def generate_dfs(args:Namespace, run_details:Any)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Implementation of ID135
    dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path = setup_paths(
        environment=args.environment, dataset_name=args.dataset_name, run_details=run_details)
    Paths.initialize(args.environment, args.dataset_name, run_details=run_details)
    df = load_and_concatenate_json_files(transcript_dev_path)


    if run_details.dataset_name == 'Chime6':
        assert(n:= df.shape[0] == 7437) # 3 samples different
        eval_df = load_and_concatenate_json_files(transcript_eval_path)
        assert (eval_n := eval_df.shape[0] == 11028)
        train_df = load_and_concatenate_json_files( transcript_train_path )
        assert(train_n := train_df.shape[0] == 79967) # 13 samples different
        dev_df = chime_parsing(df, run_details, dev_path)  # dev 14872

        eval_df = chime_parsing(eval_df, run_details, eval_path) # 22051
        expanded_df = chime_parsing(train_df, run_details, train_path) # 399735



    else:
        if run_details.dataset_name == "dipco":
            df = load_and_concatenate_json_files(transcript_eval_path)
            assert ((original_size := df.shape[0]) == 3673 or original_size ==3405)


        expanded_df, dev_df, eval_df = dipco_parsing(df, run_details, eval_path)
        if(run_details.developer_mode == "N"):
            if(run_details.data_portion == "all"):
                assert (original_size := df.shape[0]) in (3673, 3405)
                values = eval_df['file_path'].value_counts().values
                max_value = values.max() # the channels
                close_samples_value_counts = np.where(values < max_value, values, 0) 
                assert (np.sum(close_samples_value_counts) == max_value)




        # TODO Verify

    eval_df['results'] = eval_df['words']
    eval_df.reset_index(drop=True, inplace=True)
    if run_details.oversampling != 1:
        expanded_df = oversample_clean_audio( expanded_df, run_details )
    return expanded_df, dev_df, eval_df


def oversample_clean_audio(expanded_df:pd.DataFrame, run_details:"RunDetails")-> pd.DataFrame:
    #ID149
    import numpy as np
    from augmentations import filter_p_audio
    np.random.seed( 42 )
    near_person_samples = filter_p_audio( expanded_df=expanded_df )
    oversampled_sub_df = pd.concat( [near_person_samples] * (run_details.oversampling - 1), ignore_index=True )
    expanded_df = pd.concat( [expanded_df, oversampled_sub_df], ignore_index=True )
    expanded_df = expanded_df.iloc[np.random.permutation( len( expanded_df ) )]
    return expanded_df


class Paths:
    _instance = None

    def __init__(self, dataset_path=None, dev_path=None, eval_path=None,
                 transcript_dev_path=None, transcript_eval_path=None,
                 train_path=None, transcript_train_path=None, prediction_path = None):
        self.dataset_path = dataset_path
        self.dev_path = dev_path
        self.eval_path = eval_path
        self.transcript_dev_path = transcript_dev_path
        self.transcript_eval_path = transcript_eval_path
        self.train_path = train_path
        self.transcript_train_path = transcript_train_path
        self.prediction_path = prediction_path

    @classmethod
    def initialize(cls, environment, dataset_name, run_details):
        paths = setup_paths(environment, dataset_name, run_details=run_details)
        prediction_directory = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}")
        paths = paths + (prediction_directory,)
        cls._instance = cls(*paths)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("Paths have not been initialized.")
        return cls._instance

def generate_transcription_csv_path(run_details:"RunDetails")-> str:
    # ID 167
    from train import get_model_size
    model_size = get_model_size(run_details.model_id)
    transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
    return transcription_csv_path


def get_clean_audio_without_music(df:pd.DataFrame)-> pd.DataFrame:
    #This is the implementation for ID: 60

    # music noise starts playing at dipco at different timestamps more details in README of DIPCO dataset
    music_start = {"S01": "00:38:52", "S02": "00:19:30", "S03": "00:33:45","S04": "00:23:25", "S05": "00:31:15", "S06": "00:06:17","S07": "00:10:05", "S08": "00:01:02", "S09": "00:12:18","S10": "00:07:10"}
    music_start_seconds = {key: time_to_seconds(value) for key,value in music_start.items()}
    music_start_frames = {key: float(value)*16000 for key,value in music_start_seconds.items()}
    df["music_start"] = df['session_id'].map( lambda x, mapping=music_start_frames: mapping.get( x, None ) )
    no_background_music_samples = df.query("music_start > endframe")
    no_background_music_samples.drop(columns=['music_start'], inplace=True)
    return no_background_music_samples

def remove_duplicates(df:pd.DataFrame)-> pd.DataFrame:
    #Implementation of ID:142
    # Drop duplicates based on the combination of 'filepath', 'words', and 'startframe'
    unique_df = df.drop_duplicates( subset=['file_path', 'words', 'startframe'] )



    return unique_df


