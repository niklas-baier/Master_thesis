import os
import glob
from datetime import datetime
import re
import pandas as pd
from transformers import WhisperFeatureExtractor

import augmentations
from train import get_model_size

pd.options.mode.copy_on_write = True
import torchaudio
import pprint
from typing import List
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import Features, Value
def dipco_paths(dataset_path):

    dev_path = os.path.join(dataset_path, 'audio/dev')
    eval_path = os.path.join(dataset_path, 'audio/eval')
    transcript_dev_path = os.path.join(dataset_path, 'transcriptions/dev')
    transcript_eval_path = os.path.join(dataset_path, 'transcriptions/eval')
    return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path

def chime_paths(dataset_path):
    dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path)
    train_path = os.path.join(dataset_path, 'audio/train')
    transcript_train_path = os.path.join(dataset_path, 'transcriptions/train')
    return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path

def setup_paths(environment, dataset_name):
    dataset_path = "/project/data_asr/dipco/Dipco"
    bw_workplace_path = '/pfs/work7/workspace/scratch/uhicv-blah'
    if environment == 'cluster':
        if dataset_name == "Chime6":
            dataset_path = '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/'#'/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
            return chime_paths(dataset_path=dataset_path)



        else:
            dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path=dataset_path)
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
            return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, '', ''


    else:
        if dataset_name == "Chime6":
            dataset_path = "/home/niklas/Downloads/Master/espnet/egs2/chime7_task1/asr1/datasets/ChIME6/"
            return chime_paths(dataset_path)
        else:
            dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path = dipco_paths(dataset_path=dataset_path)
            return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path ,'',''

def generate_dataset_paths(run_details):
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


def list_json_files(directory):
    # Construct the file path pattern
    pattern = os.path.join(directory, '*.json')

    # Use glob to get a list of files matching the pattern
    json_files = glob.glob(pattern)

    return json_files


def load_and_concatenate_json_files(directory):
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


def expand_start_time(row):
    start_time_dict = row['start_time']
    rows = []
    for key, time_str in start_time_dict.items():
        new_row = row.copy()
        new_row['audio'] = key
        new_row['start'] = time_str
        rows.append(new_row)
    return pd.DataFrame(rows)


# @time_to_seconds: Function to convert time string to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))

    return h * 3600 + m * 60 + s


def chime_get_seconds_from_time(time_obj):
    # Extract hours, minutes, and seconds from the Timestamp object
    h = time_obj.hour
    m = time_obj.minute
    s = time_obj.second
    ms = time_obj.microsecond // 1000
    # Convert the time to seconds
    return h * 3600 + m * 60 + s + ms / 1000


def get_corresponding_end_time(dict: dict, key: str):
    end_time = [v for k, v in dict if k == key]
    return end_time


def generate_microphone_paths(row,mode_path):

    paths = []

    for i in range(1, 7):
        path = f"{mode_path}/{row['session_id']}_{row['audio']}.CH{i}.wav"
        paths.append(path)

    path = f"{mode_path}/{row['session_id']}_{row['speaker_id']}.wav"
    paths.append(path)
    return paths


def chime_generate_microphone_paths(row, mode_path):
    paths = []
    from Whisper import train_path
    if mode_path == train_path:
        for i in range(1, 5):
            for j in range(1,7):
                path = f"{mode_path}/{row['session_id']}_U0{j}.CH{i}.wav"
                if (j == 3 or j==4) :
                    pass
                else:
                    paths.append(path)



    else:
        for i in range(1, 5):
            path = f"{mode_path}/{row['session_id']}_{row['ref']}.CH{i}.wav"
            paths.append(path)



    path = f"{mode_path}/{row['session_id']}_{row['speaker']}.wav"
    paths.append(path)
    return paths


# change the seconds to frames
def get_Frames(starting_second: float, sample_rate: int, end_second: float) -> List[int]:
    return [int(starting_second * sample_rate), int(end_second * sample_rate)]


def validate_frames_column(frames_list):
    return len(frames_list) == 2


def chime_parsing(dataframe, run_details,mode_path):
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
    if run_details.task == 'classification':
        dataframe.drop(
            columns=['end_time', 'start_time', 'duration', 'frames', 'start', 'end', 'location', 'ref', 'endframe',
                     'session_id', 'words'], inplace=True) # don't drop the speaker but wordss for the time being
    else:
        from Whisper import train_path
        if mode_path == train_path:
            dataframe.drop(
                columns=['end_time', 'start_time', 'endframe',
                         'session_id', 'speaker', 'duration','frames','start','end'], inplace=True)

        else:
            dataframe.drop(
                columns=['end_time', 'start_time', 'ref', 'endframe',
                         'session_id', 'speaker', 'duration','frames','start','end','location'], inplace=True) # additonally drop location and ref
    dataframe.reset_index(drop=True, inplace=True)
    if run_details.developer_mode == 'Y':
        return dataframe.sample(n=100)
    else:
        return dataframe


def dipco_parsing(dataframe, run_details, mode_path):
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
    dataframe = dataframe.explode('file_path').reset_index(drop=True)
    dataframe['frames'] = dataframe.apply(lambda row: get_Frames(row['start'], 16000, row['end']), axis=1)
    dataframe = dataframe[dataframe['audio'] != 'close-talk']
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
    train_dataframe,test_dataframe = train_test_split(dataframe, test_size=0.05, random_state=42)
    train_dataframe = drop_columns_dipco(train_dataframe,run_details)
    test_dataframe = drop_columns_dipco(test_dataframe, run_details)
    train_dataframe.reset_index(drop=True, inplace=True)
    test_dataframe.reset_index(drop=True, inplace=True)
    if run_details.developer_mode == 'Y':
        return train_dataframe.sample(n=100,random_state=42), test_dataframe.sample(n=100, random_state=42)
    else:
        return train_dataframe, test_dataframe
def drop_columns_dipco(dataframe, run_details):
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

def generate_features(run_details):
    basic_features = {'file_path': Value('string'),
                'startframe': Value('int64'),
                'num_frames': Value('int64')}
    if run_details.augmentation == 'Y':
        basic_features['snr'] = Value('int64')
        basic_features['filepath_noise'] = Value('string')
        basic_features['words'] = Value('string')
        basic_features['file_name'] = Value('string')

        return Features(basic_features)
    else:
        basic_features['words'] = Value('string')
        return Features(basic_features)



def Hug_dataset_creation(expanded_df, developer_mode,features,test_dataset):
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


def prepare_dataset_seq2seq(batch):
    # load and resample audio data from 48 to 16kHz

    from Whisper import run_details, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run_details.model_id)

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


    from Whisper import run_details, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(run_details.model_id)

    waveform, sample_rate = torchaudio.load(batch["file_path"], frame_offset=batch["startframe"],
                                            num_frames=batch["num_frames"])
    breakpoint()
    # overlay the audioforms
    waveform = augmentations.apply_noises(filepath_original_sound=batch["file_path"],filepath_synthetic_noise=batch["filepath_noise"], snrs=batch['snr'])
    #TODO shape is 3,43453000

    input = waveform
    batch["input_features"] = feature_extractor(input, sampling_rate=sample_rate).input_features[0]

    # compute log-Mel input features from input audio array

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["words"]).input_ids

    return batch





def map_datasets(run_details, train_dataset,eval_dataset, test_dataset, dataset_paths):
    if run_details.augmentation == "Y":
        mapping_function = prepare_noisedataset_seq2seq
    else:
        mapping_function = prepare_dataset_seq2seq


    map_and_store_datasets(run_details, train_dataset, eval_dataset, test_dataset, dataset_paths, mapping_function)





def map_and_store_datasets(run_details, train_dataset, eval_dataset, test_dataset, dataset_paths, mapping_function):
    if run_details.augmentation == "Y":
        mapping_function = prepare_noisedataset_seq2seq
    if run_details.train_state == 'T':
        train_dataset = train_dataset.map(mapping_function)
        train_dataset.save_to_disk(dataset_paths['train'])
        del train_dataset
        mapping_function = prepare_dataset_seq2seq
        eval_dataset = eval_dataset.map(mapping_function)
        eval_dataset.save_to_disk(dataset_paths['eval'])
        del eval_dataset
    test_dataset = test_dataset.map(mapping_function)
    test_dataset.save_to_disk(dataset_paths['test'])
    del test_dataset
    return


def mapped_dataset_exists(dataset_path):
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        return True
    return False


def extract_special_token(label_string):
    import re
    match = re.search(r'\[\w+\]', label_string)
    if match:
        return str(match.group(0))
    else:
        return "No token"

def extract_letters(input_string):
    return ''.join([char for char in input_string if char.isalpha()])
def generate_dfs(args, run_details):
    dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path = setup_paths(
        environment=args.environment, dataset_name=args.dataset_name)
    df = load_and_concatenate_json_files(transcript_dev_path)
    eval_df = load_and_concatenate_json_files(transcript_eval_path)
    if run_details.dataset_name == 'Chime6':
        train_df = load_and_concatenate_json_files(transcript_train_path)
    transcriptions = df['words']
    if run_details.dataset_name == 'Chime6':
        dev_df = chime_parsing(df, run_details, dev_path)  # dev
        eval_df = chime_parsing(eval_df, run_details, eval_path)
        expanded_df = chime_parsing(train_df, run_details, train_path)

    else:
        expanded_df, dev_df = dipco_parsing(df, run_details, dev_path)
        # TODO Verify
        eval_df, eval_df2 = dipco_parsing(eval_df, run_details, eval_path)
        eval_df = pd.concat([eval_df, eval_df2])
    eval_df['results'] = eval_df['words']
    eval_df.reset_index(drop=True, inplace=True)
    return expanded_df, dev_df, eval_df

def generate_transcription_csv_path(run_details):
    model_size = get_model_size(run_details.model_id)
    transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
    return transcription_csv_path

