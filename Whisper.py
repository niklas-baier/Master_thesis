#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import typing
#from functool import cache 
from datasets import load_dataset, Dataset, IterableDataset
from functools import reduce
chime_path = "/home/niklas/Downloads/Datasets/CHIME6/CHiME6_eval/CHiME6/audio/eval"
#dipco_path = "/home/niklas/Downloads/Datasets/Dipco/"

import os
from datasets import Dataset, Audio
import pandas as pd


'''def create_dipco_from_directory(directory_path, output_path):
    dev_path = os.path.join(dipco_path, 'audio/dev')
    transcript_dev_path = os.path.join(dipco_path, 'transcriptions/dev')
    #dev_audio_files = [os.path.join(directory_path, os.path.join(dev_path, file)) for file in os.listdir(dev_path) if file.endswith(('.wav', '.mp3', '.flac'))]
    dev_transcript_files = [os.path.join(transcript_dev_path, file) for file in os.listdir(transcript_dev_path) if file.endswith(('.json'))]
    print(dev_transcript_files[0])
    df_devs = [pd.read_json(jsonfile) for jsonfile in dev_transcript_files]
 
    final_dev = reduce(lambda left,right: pd.merge(left,right, on=['labels'], how='outer'), df_devs)
    
    #df = pd.DataFrame(dev_audio_files, columns=["file_path"])
    #dataset = Dataset.from_pandas(df)
    #dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
    
    #dataset.save_to_disk(output_path)
/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6
/project/data_asr/dipco/Dipco
directory_path = dipco_path
output_path = dipco_path
create_dipco_from_directory(directory_path, output_path)'''

    
"""
def create_audio_dataset_from_directory(directory_path, output_path):
    
    Create a Hugging Face dataset from a directory of audio files.

    Args:
        directory_path (str): Path to the directory containing audio files.
        output_path (str): Path to save the output dataset.
    
    # Get all audio file paths
    audio_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.wav', '.mp3', '.flac'))]

    # Create a dataframe with file paths
    df = pd.DataFrame(audio_files, columns=["file_path"])
 
    # Convert dataframe to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # Cast the 'file_path' column to 'audio' feature type
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

    # Save the dataset
    dataset.save_to_disk(output_path)

    print(f"Dataset saved to {output_path}")

# Usage
directory_path = chime_path
output_path = os.getcwd()
create_audio_dataset_from_directory(directory_path, output_path)

"""



# In[13]:


import os  
import pandas as pd 
import torchaudio 
import re 
from typing import List
import glob
dipco_path = "/project/data_asr/dipco/Dipco"    
dev_path = os.path.join(dipco_path, 'audio/dev')
transcript_dev_path = os.path.join(dipco_path, 'transcriptions/dev')


def extract_prefix(file_path:str) -> str:
    pattern = r'^(.*)\.json$'
    match = re.search(pattern, file_path)
    if match:
        prefix = match.group(1)
        return prefix
    else :
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


df = load_and_concatenate_json_files(transcript_dev_path)

#df = pd.read_json(full_path)
transcriptions = df['words']

print(df.columns)
print(df['start_time'].head(1))
#print(full_path)



# In[14]:


from transformers import WhisperFeatureExtractor
from typing import Dict
import pprint
import torch 
import matplotlib.pyplot as plt 
import multiprocessing
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
def expand_start_time(row):
    start_time_dict = row['start_time']
    rows = []
    for key, time_str in start_time_dict.items():
        new_row = row.copy()
        new_row['audio'] = key
        new_row['start'] = time_str
        rows.append(new_row)
    return pd.DataFrame(rows)


# Apply the function to each row and concatenate the results
expanded_df = pd.concat([expand_start_time(row) for _, row in df.iterrows()], ignore_index=True)

# Drop the original 'start_time' column
expanded_df = expanded_df.drop(columns=['start_time'])

# Function to convert time string to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    
    return h * 3600 + m * 60 + s

# Apply the conversion to the 'start' column
expanded_df['start'] = expanded_df['start'].apply(time_to_seconds)

def get_corresponding_end_time(dict:dict, key:str):
    end_time = [v for k,v in dict if k==key]
    return end_time
print(expanded_df.columns)
expanded_df['end'] = expanded_df.apply(lambda row: row['end_time'][row['audio']], axis=1)
expanded_df['end'] = expanded_df['end'].apply(time_to_seconds)
# removal of the end_time
expanded_df = expanded_df.drop(columns=['end_time'])
#expanded_df = expanded_df.drop(expanded_df['audio']=='close-talk')


# U01 - U05 & CH 1 - 7 

# Function to generate microphone paths
def generate_microphone_paths(row):
    paths = []
    for i in range(1, 7):
        path = f"{dev_path}/{row['session_id']}_{row['audio']}.CH{i}.wav"
        paths.append(path)

    path = f"{dev_path}/{row['session_id']}_{row['speaker_id']}.wav"
    paths.append(path)
    return paths

# Apply the function to generate the paths for each row
expanded_df['file_path'] = expanded_df.apply(generate_microphone_paths, axis=1)

# Expand the DataFrame to include the microphone paths
expanded_df = expanded_df.explode('file_path').reset_index(drop=True)


#change the seconds to frames
def get_Frames(starting_second:float, sample_rate:int, end_second:float )-> List[int] :
     return [int(starting_second*sample_rate), int(end_second*sample_rate)]

expanded_df['frames'] = expanded_df.apply(lambda row: get_Frames(row['start'], 16000, row['end']), axis=1)
expanded_df = expanded_df[expanded_df['audio'] != 'close-talk']
columns_to_drop = ['mother_tongue', 'ref', 'nativeness', 'audio', 'session_id','speaker_id', 'gender']

#get the maximum speaking duration 
expanded_df['duration'] = expanded_df.apply(lambda row: row['end'] - row['start'], axis=1)
# print(expanded_df['duration'].max()) yielded that the biggest in the dipco dataset was above 60 seconds for those an additional separation is required 
expanded_df = expanded_df.drop(columns=columns_to_drop)
# sorting for cache efficiency so far no speedup 
def validate_frames_column(frames_list):
    return len(frames_list) == 2
if expanded_df['frames'].isnull().any():
    raise ValueError("The 'frames' column contains null values.")
if not expanded_df['frames'].apply(validate_frames_column).all():
    raise ValueError("Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")

expanded_df[['startframe', 'endframe']] = pd.DataFrame(expanded_df['frames'].tolist(), index=expanded_df.index)

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

print(expanded_df.shape)
pprint.pp(expanded_df.head(10))

print(expanded_df.columns)
#expanded_df['logmel'] = expanded_df.apply(lambda row: get_logmel(row['startframe'], row['endframe'], row['file_path']), axis=1)
def get_logmel(startframe: int, endframe: int, filepath: str) -> Dict[str, torch.Tensor]:
    sliced_waveform = load_audio_segment(filepath=filepath, start_frame=startframe, end_frame=endframe)
    features = feature_extractor(sliced_waveform.numpy(), sampling_rate=16000, return_tensors='pt')
    return features


    
def load_audio_segment(filepath, start_frame, end_frame):
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform[:, start_frame:end_frame], sample_rate    
#print(expanded_df)
#print(expanded_df.head(10))




# In[16]:


import torch
import cProfile
import time 
import functools
from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


dataset = Dataset.from_pandas(expanded_df.head(15))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cast the 'file_path' column to 'audio' feature type
dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
#print(dataset.features)
def load_audio_segment(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate

def slice_audio_segment(waveform, start_frame, end_frame, sample_rate):
    return waveform[:, start_frame:end_frame], sample_rate
    
def profile_audio_loading(df):
    load_audio_segment(df['file_path'][0], df['startframe'][0], df['endframe'][0])
# slicing operation is fast 
#cProfile.run("load_audio_segment(filepath=expanded_df['file_path'][0], start_frame=expanded_df['startframe'][0], end_frame=expanded_df['endframe'][0])")
def extract_audio(batch):
    
    feature_list = []
    sample_rate_list = []
    waveform = load_audio_segment(batch['file_path'][0]['path'])
    
    for idx in range(len(batch['file_path'])):
        filepath = batch['file_path'][idx]['path']
        start_frame = batch['startframe'][idx]
        end_frame = batch['endframe'][idx]
        
        # Ensure filepath is a string
        if isinstance(filepath, dict):
            print(filepath)
        
        # Load audio segment from file path
        #waveform, sample_rate = load_audio_segment(filepath, start_frame, end_frame)
        waveform, sample_rate = slice_audio_segment(waveform, start_frame, end_frame,16000)
        features = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors='pt')

        
        # Append loaded waveform and sample rate to lists
        feature_list.append(features)
        sample_rate_list.append(sample_rate)
    
    # Assign audio and sample rate lists to batch
    batch['logmel'] = feature_list
    batch['sample_rate'] = sample_rate_list
    
    return batch
#cProfile.run("dataset.map(extract_audio, batched=True, batch_size=1, num_proc=1, load_from_cache_file=True)")

#dataset = dataset.map(extract_audio, batched=True, batch_size=1, num_proc=1, load_from_cache_file=True)
import multiprocessing
from datasets import load_dataset, DatasetDict, concatenate_datasets

# Load the dataset


# Define the function to process a chunk of the dataset
def process_chunk(dataset_chunk, process_id):
    return dataset_chunk.map(extract_audio, batched=True, batch_size=1, num_proc=1, load_from_cache_file=True)

# Function to split the dataset into n chunks
def split_dataset(dataset, n):
    total_len = len(dataset)
    chunk_size = total_len // n
    chunks = [dataset.select(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(n)]
    if total_len % n != 0:
        chunks.append(dataset.select(range(n * chunk_size, total_len)))
    return chunks
'''
# Split the dataset into 5 chunks
start_time = time.perf_counter()
num_chunks = 3
dataset_chunks = split_dataset(dataset, num_chunks)  # Adjust 'train' as necessary

# Create a multiprocessing pool
pool = multiprocessing.Pool(processes=num_chunks)

# Process each chunk in parallel
results = pool.starmap(process_chunk, [(chunk, i) for i, chunk in enumerate(dataset_chunks)])

# Close the pool and wait for the work to finish
pool.close()
pool.join()

# Concatenate the results back into a single dataset
processed_dataset = concatenate_datasets(results)
end_time = time.perf_counter()
print(f"Downloaded the tutorial in {end_time - start_time:0.4f} seconds")

# Save the processed dataset
processed_dataset.save_to_disk("./Downloads/processed_dataset")

# Save the processed dataset to the downloads folder
dataset.save_to_disk('./Downloads/processed_dataset')'''


# In[5]:


#numba gives 1 sec improvement over 17 sec per sample 
'''
from numba import jit
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


dataset = Dataset.from_pandas(expanded_df.head(15))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cast the 'file_path' column to 'audio' feature type
dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
dataset = dataset.map(extract_audio, batched=True, batch_size=1, num_proc=1, load_from_cache_file=True)
@jit
def load_audio_segment(filepath, start_frame, end_frame):
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform[:, start_frame:end_frame], sample_rate
@jit
def extract_audio(batch):
    
    feature_list = []
    sample_rate_list = []
    
    for idx in range(len(batch['file_path'])):
        filepath = batch['file_path'][idx]['path']
        start_frame = batch['startframe'][idx]
        end_frame = batch['endframe'][idx]
        
        # Ensure filepath is a string
        if isinstance(filepath, dict):
            print(file_path)
        
        # Load audio segment from file path
        waveform, sample_rate = load_audio_segment(filepath, start_frame, end_frame)
        features = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors='pt')

        
        # Append loaded waveform and sample rate to lists
        feature_list.append(features)
        sample_rate_list.append(sample_rate)
    
    # Assign audio and sample rate lists to batch
    batch['logmel'] = feature_list
    batch['sample_rate'] = sample_rate_list
    
    return batch
dataset = dataset.map(extract_audio, batched=True, batch_size=1, num_proc=1, load_from_cache_file=True)'''


# In[6]:


# c profiling whether it is faster to separate the audio so that not as much needs to be loaded

"""
import torchaudio
import torch
import os

# Load the waveform from a file
filename = expanded_df['file_path'][0]
waveform, sample_rate = torchaudio.load(filename)

# Get the number of samples and compute the segment length
num_samples = waveform.size(1)
segment_length = num_samples // 10

# Create an output directory if it doesn't exist
output_dir = "output_segments"
os.makedirs(output_dir, exist_ok=True)

# Split the waveform into 10 segments and save each to a file
for i in range(10):
    start = i * segment_length
    end = (i + 1) * segment_length
    segment = waveform[:, start:end]
    
    # Handle the last segment which might be slightly longer due to integer division
    if i == 9:
        segment = waveform[:, start:]

    output_filename = os.path.join(output_dir, f"segment_{i + 1}.wav")
    torchaudio.save(output_filename, segment, sample_rate)

print("Segments saved successfully.")


"""


# In[17]:


import inspect
import whisper
source_code = inspect.getsource(whisper.load_audio)
print(source_code)


# In[ ]:


from IPython.display import IFrame
import whisper
from tqdm import tqdm 
model = whisper.load_model("base")
expanded_df['results'] = ''
expanded_df.reset_index(drop=True, inplace=True)
print(expanded_df.shape)
# load audio and pad/trim it to fit 30 seconds

def transcribe_audio(expanded_df, model):
    
    for i in tqdm(range(expanded_df.shape[0])):
        #audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio,_ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i], num_frames=expanded_df['endframe'][i]-expanded_df['startframe'][i])
        audio = whisper.pad_or_trim(audio)
    
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    
        # detect the spoken language
        #_, probs = model.detect_language(mel)
        #print(f"Detected language: {max(probs, key=probs.get)}")
    
        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        expanded_df.loc[i,'results'] = result 
    
        # print the recognized text
        #print(result[0].text)
    return expanded_df
expanded_df=transcribe_audio(expanded_df, model)
expanded_df.to_csv('dipco_eval.csv', index=False)
    
#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')

# result the load audio function takes a quarter of the time when the snippets are cut into lenghts of 1:10th


# In[ ]:


import matplotlib.pyplot as plt
print(expanded_df.columns)
expanded_df['frame_diff'] = expanded_df['endframe'] - expanded_df['startframe'] 
print(expanded_df['duration'].nsmallest(20))
filtered_df = expanded_df[expanded_df['frame_diff'] < 0]
print(filtered_df)
# Plot the histogram with 20 bins
plt.hist(expanded_df['frame_diff'], bins=20, edgecolor='black')
plt.title('Histogram of Frame Differences')
plt.xlabel('Frame Difference (endframe - startframe)')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[ ]:


print(dir(model))


# In[ ]:


import meeteval
from meeteval.viz.visualize import AlignmentVisualization



# In[ ]:


import meeteval

# SISO WER
wer = meeteval.wer.wer.siso.siso_word_error_rate(
    reference='The quick brown fox jumps over the lazy dog',
    hypothesis='The qwick brown fox jump over lazy '
)
print(wer)
expanded_df = expanded_df.head(10)
print(dir(expanded_df['results'][0].__str__()))
print(expanded_df['results'][0].text)
print(expanded_df['words'][0])
print((expanded_df['results'].apply(type)))
def extract_text(result):
    # Assuming the DecodingResult object has a 'text' attribute
    return result.text

# Apply the extraction function to the 'results' column
expanded_df['results_text'] = expanded_df.apply(lambda row: row['results'].text, axis=1)
#expanded_df['duration'] = expanded_df.apply(lambda row: row['end'] - row['start'], axis=1)
print(type(expanded_df['results_text'][0]))
# Calculate WER using the extracted text
expanded_df['wer'] = expanded_df.apply(
    lambda row: meeteval.wer.wer.siso.siso_word_error_rate(
        reference=row['words'], 
        hypothesis=row['results_text']
    ), 
    axis=1
)
print(expanded_df['wer'])


# In[ ]:


import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm

class DIPCODataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, transform=None):
        self.root_dir = root_dir
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    if file.startswith('._'):
                        pass
                    else:
                        self.file_paths.append(os.path.join(subdir, file))
                        
                    
        self.target_sample_rate = target_sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if the sample rate is different from the target sample rate
        if sample_rate != self.target_sample_rate:
            waveform = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)

        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, file_path

def collate_fn(batch):
    waveforms, file_paths = zip(*batch)
    waveforms = [waveform.mean(dim=0, keepdim=True) for waveform in waveforms]  # Convert to mono
    return torch.cat(waveforms, dim=0), file_paths

# Parameters
root_dir = '/media/niklas/SSD/Dataset/Dipco/audio'
target_sample_rate = 16000
batch_size = 16
num_workers = 4

# Dataset and DataLoader
dataset = DIPCODataset(root_dir, target_sample_rate=target_sample_rate)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

# Example of iterating through the DataLoader


# Example of iterating through the DataLoader with a progress bar
'''
for waveforms, file_paths in tqdm(dataloader, total=len(dataloader), desc="Processing"):
    print(f'Batch size: {waveforms.size(0)}')
    for waveform, file_path in zip(waveforms, file_paths):
        print(f'Processed file: {file_path}')
    break  # Remove this line to iterate over the entire dataset
'''


# In[ ]:


import torch
from torch.utils.data import Dataset
import soundfile as sf

class CustomAudioDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        wav_file_path = row['file_path']
        start_frame = row['startframe']
        end_frame = row['endframe']
        label = row['words']
        
        # Load the audio file segment
        audio, _ = sf.read(wav_file_path, start=start_frame, stop=end_frame)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio)
        
        return audio_tensor, label


# In[ ]:


from torch.utils.data import DataLoader

# Create the dataset
audio_dataset = CustomAudioDataset(expanded_df)

# Create the dataloader
dataloader = DataLoader(audio_dataset, batch_size=2, shuffle=True)

# Example usage in a training loop
for batch in dataloader:
    inputs, labels = batch
    print(inputs, labels)
    # Your training code here


# In[ ]:


from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.train_batch_size, 
            shuffle=True
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset, 
            batch_size=self.args.eval_batch_size
        )


# In[ ]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame
train_df, eval_df = train_test_split(expanded_df, test_size=0.2)

train_dataset = CustomAudioDataset(train_df)
eval_dataset = CustomAudioDataset(eval_df)


# In[ ]:


from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")



# In[ ]:


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print(expanded_df.columns)


# In[ ]:



# In[ ]:


# untrained model just inference
import inspect
yesno_data = torchaudio.datasets.YESNO('.', download=True)
#print(dir(yesno_data))
source_code = inspect.getsource(yesno_data.__class_getitem__)
#print(source_code)
data_loader = torch.utils.data.DataLoader(
    yesno_data,
    batch_size=1,
    shuffle=True,   
    num_workers=4)
from datasets import Dataset

source_code = inspect.signature(Dataset.from_pandas)
import inspect
from transformers import WhisperFeatureExtractor

# Get the source code of the WhisperFeatureExtractor class
source_code = inspect.getsource(WhisperFeatureExtractor)
print(source_code)

