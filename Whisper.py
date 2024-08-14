#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import typing
#from functool import cache 
from datasets import load_dataset, Dataset, IterableDataset
from functools import reduce


train_state = 'NT'
version = "vanilla"
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
         
        

#dipco_path = "/home/niklas/Downloads/Datasets/Dipco/"
chime_path_cluster = '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
dataset_name = "Chime6"
environment = "laptop"
import os
from datasets import Dataset, Audio
import pandas as pd
dataset_path,dev_path,eval_path,transcript_dev_path,transcript_eval_path = setup_paths(environment=environment, dataset_name=dataset_name)
print(dataset_path)
print(dev_path)
print(eval_path)
print(transcript_dev_path)
print(transcript_eval_path)


# In[ ]:


import os  
import pandas as pd 
import torchaudio 
import re 
from typing import List
import glob
from datetime import datetime



def get_formated_date() -> str:
    return datetime.now().strftime("%m/%d/%Y")

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

print(transcript_dev_path)
df = load_and_concatenate_json_files(transcript_dev_path)
eval_df = load_and_concatenate_json_files(transcript_eval_path)
#df = pd.read_json(full_path)
transcriptions = df['words']

print(df.columns)
print(df['start_time'].head(1))
#print(full_path)



# In[ ]:


from transformers import WhisperFeatureExtractor
from typing import Dict
import pprint
import torch 
import matplotlib.pyplot as plt 
import multiprocessing
import inspect
from datasets import Features, Value
model_name = model_id = "openai/whisper-tiny.en"#"openai/whisper-large"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
print(inspect.signature(feature_extractor))
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
    return h * 3600 + m * 60 + s + ms/1000


def get_corresponding_end_time(dict:dict, key:str):
    end_time = [v for k,v in dict if k==key]
    return end_time



# removal of the end_time

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


def chime_generate_microphone_paths(row):
    paths = []
       
    for i in range(1, 7):
        path = f"{dev_path}/{row['session_id']}_{row['ref']}.CH{i}.wav"
        paths.append(path)

    path = f"{dev_path}/{row['session_id']}_{row['speaker']}.wav"
    paths.append(path)
    return paths
    


#change the seconds to frames
def get_Frames(starting_second:float, sample_rate:int, end_second:float )-> List[int] :
     return [int(starting_second*sample_rate), int(end_second*sample_rate)]


#columns_to_drop = ['mother_tongue', 'ref', 'nativeness', 'audio', 'session_id','speaker_id', 'gender']


# print(expanded_df['duration'].max()) yielded that the biggest in the dipco dataset was above 60 seconds for those an additional separation is required 
#expanded_df = expanded_df.drop(columns=columns_to_drop)
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
        raise ValueError("Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")
    dataframe[['startframe', 'endframe']] = pd.DataFrame(dataframe['frames'].tolist(), index=dataframe.index)
    print(dataframe.shape)
    pprint.pp(dataframe.head(10))
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    dataframe.drop(columns=['end_time','start_time','duration', 'frames', 'start','end', 'location','ref', 'endframe', 'session_id', 'speaker'], inplace=True)
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
    #get the maximum speaking duration 
    dataframe['duration'] = dataframe.apply(lambda row: row['end'] - row['start'], axis=1)
    if dataframe['frames'].isnull().any():
        raise ValueError("The 'frames' column contains null values.")
    if not dataframe['frames'].apply(validate_frames_column).all():
        raise ValueError("Each entry in the 'frames' column must be a list of exactly two elements [startframe, endframe].")
    dataframe[['startframe', 'endframe']] = pd.DataFrame(dataframe['frames'].tolist(), index=dataframe.index)
    print(dataframe.shape)
    pprint.pp(dataframe.head(10))
    dataframe['num_frames'] = dataframe['endframe'] - dataframe['startframe']
    # handle chime and dipco data differently
    #TODO
    # #dataframe['speaker_id_int'] = dataframe['speaker_id'].str.extract('(\d+)').astype(int) there are not the same persons in each dataset
    if 'nativeness' in dataframe.columns:
        dataframe.drop(columns=['endframe', 'session_id', 'speaker_id','gender', 'nativeness','mother_tongue','audio','start','end','endframe','duration','frames', 'ref'], inplace=True)
    else:
        dataframe.drop(columns=['end_time','start_time'], inplace=True)
        
        
        
    
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe 

features = Features({
    'file_path': Value('string'),
    'words': Value('string'),
     'startframe': Value('int64'),
    'num_frames': Value('int64'),    
})

if dataset_name == 'Chime6':
    expanded_df = chime_parsing(df)
    eval_df = chime_parsing(eval_df)
else:
    expanded_df = dipco_parsing(df)
    eval_df = dipco_parsing(eval_df)




    
print(expanded_df.head(10))








# In[ ]:


print(len(expanded_df['file_path'].head(1)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from transformers import WhisperTokenizer
from datasets import load_dataset
device = "cuda" 
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = model_name


tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
def Hug_dataset_creation(expanded_df, mode):
    dataset = Dataset.from_pandas(expanded_df, features=features)
    shuffled_dataset = dataset.shuffle(seed=42)  
    if mode == 'train':
        return shuffled_dataset.select(range(100))
    else:
          return shuffled_dataset.select(range(100))
        

    
train_dataset = Hug_dataset_creation(expanded_df, mode='train')
    
eval_dataset = Hug_dataset_creation(eval_df, mode='eval')


#dataset = dataset.to_iterable_dataset()
print(train_dataset[0])
import inspect
print(inspect.signature(WhisperTokenizer))



def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    print(batch["file_path"])
   
  
    waveform,sample_rate = torchaudio.load(batch["file_path"], frame_offset=batch["startframe"], num_frames=batch["num_frames"])
    input = waveform.squeeze().numpy()
    batch["input_features"]= feature_extractor(input, sampling_rate=sample_rate).input_features[0]
        
   
    
    

    # compute log-Mel input features from input audio array


    # encode target text to label ids
    batch["labels"] = tokenizer(batch["words"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset)
#TODO
eval_dataset = train_dataset
train_dataset.save_to_disk("train.hf")
eval_dataset.save_to_disk("eval.hf")
import os
from datasets import load_from_disk, Dataset

# Define the path to the dataset directory
dataset_path = "train.hf"

# Check if the directory exists
if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
    try:
        # Attempt to load the dataset
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded.")
    except Exception as e:
        print(f"error while loading the dataset: {e}")
else:
    print(f"The directory '{dataset_path}' does not exist or is not a directory.")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch_dtype,
)





# In[ ]:


import warnings
from tqdm import tqdm
import warnings
from functools import wraps
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper



# Define a decorator to suppress specific warnings
def suppress_specific_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # Suppress FutureWarning
            return func(*args, **kwargs)
    return wrapper


# In[ ]:


if ("openai/whisper-large") in model_id:
    processor = AutoProcessor.from_pretrained(model_id, language='en', task="transcribe")
    model.generation_config.language = "English"
    model.generation_config.task = "transcribe"

   

else:
    processor = AutoProcessor.from_pretrained(model_id)
    
model.generation_config.forced_decoder_ids = None



pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device

 
)

from tqdm import tqdm 

expanded_df['results'] = ''
expanded_df = expanded_df.head(10)
expanded_df.reset_index(drop=True, inplace=True)
print(expanded_df.shape)
# load audio and pad/trim it to fit 30 seconds
from numba import jit

@suppress_specific_warnings
@timing_decorator

def transcribe_audio(expanded_df):
    
    for i in tqdm(range(expanded_df.shape[0])):
        #audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio,_ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i], num_frames=expanded_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        if ("openai/whisper-large") in model_id:
               result = pipe(audio_data, generate_kwargs={"language": "english"})
        else:
            result = pipe(audio_data)
            
            
     

       
        expanded_df.loc[i,'results'] = result['text']
    
    
    return expanded_df

def transcribe_dataset(dataset):
    
    for i in tqdm(range(expanded_df.shape[0])):
        #audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio,_ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i], num_frames=expanded_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        if ("openai/whisper-large") in model_id:
               result = pipe(audio_data, generate_kwargs={"language": "english"})
        else:
            result = pipe(audio_data)
            
            
     

       
        expanded_df.loc[i,'results'] = result['text']
    
    
    return expanded_df
print(expanded_df.columns)
expanded_df=transcribe_audio(expanded_df)
dev = "dev"

import re

# Regex pattern splits on substrings "; " and ", "
components = re.split('-|/|', model_id)
model_size = components[2]
transcription_csv_path = f'{dataset_name}_{dev}_{model_size[:4]}_{train_state}.csv'
expanded_df.to_csv(transcription_csv_path, index=False)

#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')


#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')

# result the load audio function takes a quarter of the time when the snippets are cut into lenghts of 1:10th


# In[ ]:


"""@suppress_specific_warnings
@timing_decorator
def transcribe_audio_ds(dataset: Dataset, batch_size=8):
    results = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        for example in batch:
            audio, _ = torchaudio.load(
                example['file_path'], 
                frame_offset=example['startframe'], 
                num_frames=example['num_frames']
            )
            audio_data = audio.squeeze().numpy()
            
            if "openai/whisper-large" in model_id:
                result = pipe(audio_data, generate_kwargs={"language": "english"})
            else:
                result = pipe(audio_data)
            
            # Collect result for each item in the batch
            results.append(result['text'])

    # Add results to the dataset
    
    dataset = dataset.add_column("results", results)
    return dataset

# Example usage
# Assuming you have a Hugging Face dataset `ds` with columns 'file_path', 'startframe', and 'num_frames'
#expanded_df.drop(columns=['results'], inplace=True)
print(expanded_df.columns)

ds = Dataset.from_pandas(expanded_df)
ds = transcribe_audio_ds(ds)"""


# In[ ]:





# In[ ]:


# chime normalization
import jiwer
from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import TimeFormatConverter, normalize_text_chime6
def chime_normalisation(input:str) -> str:
    jiwer_chime6_scoring = jiwer.Compose(
    [
        RemoveKaldiNonWords(),
        jiwer.SubstituteRegexes({r"\"": " ", "^[ \t]+|[ \t]+$": "", r"\u2019": "'"}),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    jiwer_chime7_scoring = jiwer.Compose(
    [
        jiwer.SubstituteRegexes(
            {
                "(?:^|(?<= ))(hm|hmm|mhm|mmh|mmm)(?:(?= )|$)": "hmmm",
                "(?:^|(?<= ))(uhm|um|umm|umh|ummh)(?:(?= )|$)": "ummm",
                "(?:^|(?<= ))(uh|uhh)(?:(?= )|$)": "uhhh",
            }
        ),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    def chime6_norm_scoring(txt):
        return jiwer_chime6_scoring(normalize_text_chime6(txt, normalize="kaldi"))


# here we also normalize non-words sounds such as hmmm which are quite a lot !
# you are free to use whatever normalization you prefer for training but this
# normalization below will be used when we score your submissions.
    def chime7_norm_scoring(txt):
        return jiwer_chime7_scoring(
            jiwer_chime6_scoring(
                normalize_text_chime6(txt, normalize="kaldi")
            )  # noqa: E731
        )  # noqa: E731
    return chime7_norm_scoring(input)



# In[ ]:


# peft 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# Define LoRA Config
# Print out all the module names in the model
import bitsandbytes as bnb 
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


for name, module in model.named_modules():
    print(name)
linear_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
target_modules = [name for name in linear_modules if name.startswith('model.encoder.layers')]
# lowering alpha seems to be helpful when training although keeping r significantly higher than alpha does not seem to work either.
text_lora_config = LoraConfig(
    r=2,
    lora_alpha=2,
    init_lora_weights="gaussian",
    target_modules=["q_proj",  "v_proj", "k_proj","out_proj"],
    bias="none",
    task_type="Seq2Seq",
)
# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)
print ([module for module in model.modules()])
# add LoRA adaptor
model = get_peft_model(model, text_lora_config)
model.print_trainable_parameters()


# In[ ]:


# training of the model 
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
import torch
import json 
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
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


import evaluate
metric = evaluate.load("wer")
# 
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
def compute_chime_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    results = {"predictions": pred_str, "labels": label_str}
    results_directory = str(f"{model_id}_{dataset_name}_{version}_{get_formated_date()}")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    file_path = os.path.join(results_directory, "results.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

# Define the file path


# Write the evaluation results to the file
    # Example evaluation results
    chime_normalized_reference = [chime_normalisation(reference) for reference in label_str]
    chime_normalized_prediction = [chime_normalisation(pred) for pred in pred_str]

    #wer = 100 * metric.compute(predictions=chime_normalized_prediction, references=chime_normalized_reference)
    wer = jiwer.wer(list(chime_normalized_prediction), list(chime_normalized_reference))

    return {"wer": wer}
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=300,#4000
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,    
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
 
)
#english_feature_extractor = processor.feature_extractor(language='en')
print(inspect.signature(processor.feature_extractor))
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_chime_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)


# Print evaluation results
def plot_loss(trainer):
    df_log = pd.DataFrame(trainer.state.log_history)
# visualization of the loss during training 
    (df_log.dropna(subset=["eval_loss"]).reset_index()["eval_loss"].plot(label="Validation"))
    df_log.dropna(subset=["loss"]).reset_index()["loss"].plot(label="Train")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")
    
    filepath = f'Figures/Training/LOSS/{dataset_name}/{model_id}/{version}/{get_formated_date()}'
    try:
        os.makedirs(filepath)
    except FileExistsError:
        print("Directory already exists")
    finally:
        plt.savefig(filepath + "1", format='png')
        
def plot_WER(trainer):
    #print evaluation of WER over training 
    df_log = pd.DataFrame(trainer.state.log_history)
    # visualization of the loss during training 
    (df_log.dropna(subset=["eval_wer"]).reset_index()["eval_wer"].plot(label="WER"))
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.legend(loc="upper right")
    
    min_eval_wer = df_log['eval_wer'].min()
    def format_wer(wer):
        wer_str = f"{wer:.3f}"  # Format the WER to three decimal places
        return wer_str.replace(".", "_")
    min_eval_wer_str = format_wer(min_eval_wer)
    filepath = f'Figures/Training/WER/{dataset_name}/{min_eval_wer_str}/{model_id}/{version}/{get_formated_date()}'
    
    try:
        os.makedirs(filepath)
    except FileExistsError:
        print("Directory already exists")
    finally:
        plt.savefig(f'{filepath}/test.png', format='png')
    


    


# In[ ]:


if train_state == 'NT':
    pass
else:
    trainer.train()
    plot_loss(trainer)
    plot_WER(trainer)







# In[ ]:





# In[ ]:


# Chime Normalization of the results 
model_path = "./whisper-small-hi/checkpoint-101"

# Load the model from the safetensors file

tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
# Load the tokenizer (if necessary)



# Example input
print(expanded_df.columns)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
from tqdm import tqdm



# Iterate over the dataset with progress tracking
eval_temp = pd.DataFrame(columns=['results_trained'])
for i, example in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    outputs = model.generate(input_features)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)
   
    eval_temp.loc[i, "results_trained"]= transcription
    


   

   
    



print(inspect.signature(model))
# Generate the output

# Decode the output





# In[ ]:


## visualization of the layers 



# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt


def visualize_frames():
    print(expanded_df.columns)
    expanded_df['frame_diff'] = expanded_df['num_frames'] 
    print(expanded_df['num_frames'].nsmallest(20))
    filtered_df = expanded_df[expanded_df['frame_diff'] < 0]
    print(filtered_df)
    # Plot the histogram with 20 bins
    plt.hist(expanded_df['frame_diff'], bins=20, edgecolor='black')
    plt.title('Histogram of Frame Differences')
    plt.xlabel('Frame Difference (num_frames')
    plt.ylabel('Frequency')
    


# In[ ]:


print(dir(model))


# In[ ]:


import meeteval
from meeteval.viz.visualize import AlignmentVisualization

folder = r'https://raw.githubusercontent.com/fgnt/meeteval/main/'
av = AlignmentVisualization(
    meeteval.io.load(folder + 'example_files/ref.stm').groupby('filename')['recordingA'],
    meeteval.io.load(folder + 'example_files/hyp.stm').groupby('filename')['recordingA']
)
#display(av)  # Jupyter
av.dump('viz.html')  # Create standalone HTML file


# In[ ]:


import meeteval
import pandas as pd
import jiwer
from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import TimeFormatConverter, normalize_text_chime6

# SISO WER
wer = meeteval.wer.wer.siso.siso_word_error_rate(
    reference='The quick brown fox jumps over the lazy dog',
    hypothesis='The qwick brown fox jump over lazy '
)
print(wer)
"""
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
"""


from whisper.normalizers import EnglishTextNormalizer
data = pd.read_csv(transcription_csv_path)
normalizer = EnglishTextNormalizer()


def chime_normalisation(input:str) -> str:
    jiwer_chime6_scoring = jiwer.Compose(
    [
        RemoveKaldiNonWords(),
        jiwer.SubstituteRegexes({r"\"": " ", "^[ \t]+|[ \t]+$": "", r"\u2019": "'"}),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    jiwer_chime7_scoring = jiwer.Compose(
    [
        jiwer.SubstituteRegexes(
            {
                "(?:^|(?<= ))(hm|hmm|mhm|mmh|mmm)(?:(?= )|$)": "hmmm",
                "(?:^|(?<= ))(uhm|um|umm|umh|ummh)(?:(?= )|$)": "ummm",
                "(?:^|(?<= ))(uh|uhh)(?:(?= )|$)": "uhhh",
            }
        ),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    def chime6_norm_scoring(txt):
        return jiwer_chime6_scoring(normalize_text_chime6(txt, normalize="kaldi"))


# here we also normalize non-words sounds such as hmmm which are quite a lot !
# you are free to use whatever normalization you prefer for training but this
# normalization below will be used when we score your submissions.
    def chime7_norm_scoring(txt):
        return jiwer_chime7_scoring(
            jiwer_chime6_scoring(
                normalize_text_chime6(txt, normalize="kaldi")
            )  # noqa: E731
        )  # noqa: E731
    return chime7_norm_scoring(input)


print(data.head)
#dataset = dataset.map(lambda example: {'normalized_ref': chime_normalisation(example['words'])})
data['chime_ref'] =  [chime_normalisation(text) for text in data["words"]]
data['chime_hyp'] =  [chime_normalisation(text) for text in data["results"]]
data["hypothesis_clean"] = [normalizer(text) for text in data["results"]]
data["reference_clean"] = [normalizer(text) for text in data["words"]]
data['chime_ref2'] =  [normalizer(text) for text in data["chime_ref"]]
data['chime_hyp2'] =  [normalizer(text) for text in data["chime_hyp"]]
wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
# WER of the whisper normalizer
print(f"WER: {wer * 100:.2f} %")
wer = jiwer.wer(list(data["chime_ref"]), list(data["chime_hyp"]))
# WER of the whisper normalizer
print(f"WER: {wer * 100:.2f} %")
# combination of whisper normalizer and chime_normalizer
wer = jiwer.wer(list(data["chime_ref2"]), list(data["chime_hyp2"]))

print(f"WER2: {wer * 100:.2f} %")


# In[ ]:


print(data.sample(n=10))
data['wer'] = data.apply(
    lambda row: meeteval.wer.wer.siso.siso_word_error_rate(
        reference=row['reference_clean'], 
        hypothesis=row['hypothesis_clean']
    ), 
    axis=1
)


# In[ ]:


ascii_pattern = r'^[\x00-\x7F]*$'

# Step 3: Filter the DataFrame
print(data.shape)
df_ascii = data[data['hypothesis_clean'].str.contains(ascii_pattern, na=False)]
print (df_ascii.shape)
wer = jiwer.wer(list(df_ascii["reference_clean"]), list(df_ascii["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")


# In[ ]:


# looking at the results from the individual sessions 
import re 
def extract_session(file_path):
    match = re.search(r'/S(\d+)', file_path)
    if match:
        return int(match.group(1))
    else:
        return None
# the microphones on person vs not on person 
def extract_person(file_path):
    match = re.search(r'/S(\d+)_([PU])(\d+)', file_path)
    if match:
        return str(match.group(2))
    else:
        return None
# the people 
def extract_location(file_path):
    match = re.search(r'/S(\d+)_([PU])(\d+)', file_path)
    if match:
        return str(match.group(3))
    else:
        return None

def print_wer(grouped, type):
    for name, group in grouped:
    
        wer = jiwer.wer(list(group["reference_clean"]), list(group["hypothesis_clean"]))
        print(f"{type} {name}")
        print(f"wer {wer}")
        
        
    
data['session_number'] = data['file_path'].apply(extract_session)
data['mic_type'] = data['file_path'].apply(extract_person)
data['mic_number'] = data['file_path'].apply(extract_location)
grouped_ses = data.groupby('session_number')
print_wer(grouped_ses, "session")
grouped_mic_type = data.groupby('mic_type')
grouped_mic = data.groupby(['mic_type','mic_number'])
print_wer(grouped_mic, "mic_type")
print(wer)



    

    



# In[ ]:


# plot visualization of the different sessions and store the results
import ast
import re
import matplotlib.pyplot as plt
def visualize_wer(grouped, type):
    names = []
    wers = []
    for name, group in grouped:
    
        wer = jiwer.wer(list(group["reference_clean"]), list(group["hypothesis_clean"]))
        

        # Regular expression to check if the string is a tuple representation
        tuple_pattern = r"\(\s*'[^']*'\s*,\s*'[^']*'\s*\)"
        
        # Check if the input string matches the tuple pattern
        if re.fullmatch(tuple_pattern, str(name)):
            # Safely evaluate the string to get the tuple
            parsed_tuple = ast.literal_eval(str(name))
        
            # Concatenate the elements and convert to lowercase
            result_str = f"{parsed_tuple[0].lower()}{parsed_tuple[1]}"
            names.append(result_str)
           
        else:
            names.append(str(name))
            
        
        wers.append(wer)
    plt.figure(figsize=(8, 6))
    plt.bar(names, wers)
    plt.ylabel(f'Mean average WER per {type[0]}')
    model_name = type[2].rsplit('/', 1)[-1]
    plt.title(f'WER of {model_name} on the {(dataset_name:=(type[1]))} dataset')
   
    plt.savefig(f'Figures/{(partition_type:=(type[0]))} bar_plot.png', format='png')
    plt.show()
    
directory = "Figures"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")      
visualize_wer(grouped_ses, ["session", f"{dataset_name}", f"{model_name}"])
visualize_wer(grouped_mic_type, ["mic_type", f"{dataset_name}", f"{model_name}"])
visualize_wer(grouped_mic, ["mic", f"{dataset_name}", f"{model_name}"])



# In[ ]:


error_rates = data['wer'].apply(lambda x: x.error_rate)

# Calculate the mean of the error rates
mean_error_rate = error_rates.mean()
print(mean_error_rate)


# In[ ]:


import smtplib
import ssl
from email.message import EmailMessage


def send_email():
    # Define email sender and receiver
    email_sender = 'uhicv@student.kit.edu'
    email_password = '***REMOVED***'
    email_receiver = 'uhicv@student.kit.edu'
    
    # Set the subject and body of the email
    subject = 'Test has finished'
    body = """
    I've just published a new video on YouTube: https://youtu.be/2cZzP9DLlkg
    """
    
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    
    # Add SSL (layer of security)
    context = ssl.create_default_context()
    
    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:


from huggingface_hub import notebook_login
#***REMOVED***
#notebook_login()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




