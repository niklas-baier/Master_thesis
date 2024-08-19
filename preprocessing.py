import os
import glob
import datetime
import re
import pandas as pd
import torchaudio
import pprint
from typing import List,Dict
import torch

from datasets import Dataset
from datasets import Features, Value
def dipco_paths(dataset_path):

    dev_path = os.path.join(dataset_path, 'audio/dev')
    eval_path = os.path.join(dataset_path, 'audio/eval')
    transcript_dev_path = os.path.join(dataset_path, 'transcriptions/dev')
    transcript_eval_path = os.path.join(dataset_path, 'transcriptions/eval')
    return dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path


def setup_paths(environment, dataset_name):
    dataset_path = "/project/data_asr/dipco/Dipco"
    if environment == 'cluster':
        if dataset_name == "Chime6":
            dataset_path = '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
            return dipco_paths(dataset_path)

        else:

            return dipco_paths(dataset_path=dataset_path)
    else:
        if dataset_name == "Chime6":
            dataset_path = "/home/niklas/Downloads/Master/espnet/egs2/chime7_task1/asr1/datasets/ChIME6/"
            return dipco_paths(dataset_path)
        else:
            return dipco_paths(dataset_path=dataset_path)


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
    print(json_files)
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


# Function to convert time string to seconds
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


def generate_microphone_paths(row):
    from Whisper import dev_path
    paths = []

    for i in range(1, 7):
        path = f"{dev_path}/{row['session_id']}_{row['audio']}.CH{i}.wav"
        paths.append(path)

    path = f"{dev_path}/{row['session_id']}_{row['speaker_id']}.wav"
    paths.append(path)
    return paths


def chime_generate_microphone_paths(row):
    from Whisper import dev_path
    paths = []

    for i in range(1, 7):
        path = f"{dev_path}/{row['session_id']}_{row['ref']}.CH{i}.wav"
        paths.append(path)

    path = f"{dev_path}/{row['session_id']}_{row['speaker']}.wav"
    paths.append(path)
    return paths


# change the seconds to frames
def get_Frames(starting_second: float, sample_rate: int, end_second: float) -> List[int]:
    return [int(starting_second * sample_rate), int(end_second * sample_rate)]


# columns_to_drop = ['mother_tongue', 'ref', 'nativeness', 'audio', 'session_id','speaker_id', 'gender']


# print(expanded_df['duration'].max()) yielded that the biggest in the dipco dataset was above 60 seconds for those an additional separation is required
# expanded_df = expanded_df.drop(columns=columns_to_drop)
# sorting for cache efficiency so far no speedup
def validate_frames_column(frames_list):
    return len(frames_list) == 2


# Drop the original 'frames' column if no longer needed
"""expanded_df = expanded_df.drop(columns=['frames'])

expanded_df = expanded_df.sort_values(by=['file_path','start'])
expanded_df = expanded_df.reset_index(drop=True)
grouped = expanded_df.groupby(['words'])
count_df = grouped.size().reset_index(name='counts')
first_group_key = list(grouped.groups.keys())[0]
first_group = grouped.get_group(first_group_key)
print(first_group)
print(first_group['file_path'].value_counts())
print(count_df)"""


# expanded_df['logmel'] = expanded_df.apply(lambda row: get_logmel(row['startframe'], row['endframe'], row['file_path']), axis=1)


# print(expanded_df)
# print(expanded_df.head(10))

def chime_parsing(dataframe):
    dataframe['start'] = dataframe['start_time'].apply(chime_get_seconds_from_time)
    dataframe['end'] = dataframe['end_time'].apply(chime_get_seconds_from_time)

    dataframe['file_path'] = dataframe.apply(chime_generate_microphone_paths, axis=1)
    dataframe['file_path'] = dataframe.apply(lambda row: row['file_path'][0], axis=1)
    dataframe['frames'] = dataframe.apply(lambda row: get_Frames(row['start'], 16000, row['end']), axis=1)
    dataframe['duration'] = dataframe.apply(lambda row: row['end'] - row['start'], axis=1)
    if dataframe['frames'].isnull().any():
        raise ValueError("The 'frames' column contains null values.")
    if not dataframe['frames'].apply(validate_frames_column).all():
        raise ValueError(
            "Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")
    dataframe[['startframe', 'endframe']] = pd.DataFrame(dataframe['frames'].tolist(), index=dataframe.index)
    print(dataframe.shape)
    pprint.pp(dataframe.head(10))
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    dataframe.drop(
        columns=['end_time', 'start_time', 'duration', 'frames', 'start', 'end', 'location', 'ref', 'endframe',
                 'session_id', 'speaker'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def dipco_parsing(dataframe):
    # Apply the function to each row and concatenate the results
    dataframe = pd.concat([expand_start_time(row) for _, row in dataframe.iterrows()], ignore_index=True)
    # Drop the original 'start_time' column
    dataframe = dataframe.drop(columns=['start_time'])
    dataframe['start'] = dataframe['start'].apply(time_to_seconds)
    dataframe['end'] = dataframe.apply(lambda row: row['end_time'][row['audio']], axis=1)
    dataframe['end'] = dataframe['end'].apply(time_to_seconds)
    dataframe = dataframe.drop(columns=['end_time'])
    # Apply the function to generate the paths for each row
    dataframe['file_path'] = dataframe.apply(generate_microphone_paths, axis=1)
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
    print(dataframe.shape)
    pprint.pp(dataframe.head(10))
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    # handle chime and dipco data differently
    # TODO
    # #dataframe['speaker_id_int'] = dataframe['speaker_id'].str.extract('(\d+)').astype(int) there are not the same persons in each dataset
    if 'nativeness' in dataframe.columns:
        dataframe.drop(
            columns=['endframe', 'session_id', 'speaker_id', 'gender', 'nativeness', 'mother_tongue', 'audio', 'start',
                     'end', 'endframe', 'duration', 'frames', 'ref'], inplace=True)
    else:
        dataframe.drop(columns=['end_time', 'start_time'], inplace=True)

    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def Hug_dataset_creation(expanded_df, mode,features):
    dataset = Dataset.from_pandas(expanded_df, features=features)
    shuffled_dataset = dataset.shuffle(seed=42)
    if mode == 'train':
        return shuffled_dataset.select(range(100))
    else:
        return shuffled_dataset.select(range(100))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    from Whisper import feature_extractor, tokenizer
    print(batch["file_path"])

    waveform, sample_rate = torchaudio.load(batch["file_path"], frame_offset=batch["startframe"],
                                            num_frames=batch["num_frames"])
    input = waveform.squeeze().numpy()
    batch["input_features"] = feature_extractor(input, sampling_rate=sample_rate).input_features[0]

    # compute log-Mel input features from input audio array

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["words"]).input_ids
    return batch





