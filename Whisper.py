#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import inspect
from datasets import load_from_disk


# In[ ]:


import os  
import pandas as pd 
import torchaudio 
import re 
from typing import List
import glob
dipco_path = "/project/data_asr/dipco/Dipco"  
dataset_name = "Dipco"
dev_path = os.path.join(dipco_path, 'audio/dev')
eval_path = os.path.join(dipco_path, 'audio/eval')
transcript_dev_path = os.path.join(dipco_path, 'transcriptions/dev')
transcript_eval_path = os.path.join(dipco_path, 'transcriptions/eval')

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
model_name = "openai/whisper-large-v3" # "openai/whisper-tiny" "distil-whisper/distil-large-v3"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, language='en')
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




def get_corresponding_end_time(dict:dict, key:str):
    end_time = [v for k,v in dict if k==key]
    return end_time



# removal of the end_time

#expanded_df = expanded_df.drop(expanded_df['audio']=='close-talk')


# U01 - U05 & CH 1 - 7 

# Function to generate microphone paths
def generate_microphone_paths(row,mode):
    paths = []
    for i in range(1, 7):
        path = f"{mode}/{row['session_id']}_{row['audio']}.CH{i}.wav"
        paths.append(path)
                
        


    path = f"{mode}/{row['session_id']}_{row['speaker_id']}.wav"
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


def string_parsing(dataframe, path):
    # Apply the function to each row and concatenate the results
    dataframe = pd.concat([expand_start_time(row) for _, row in dataframe.iterrows()], ignore_index=True)
    # Drop the original 'start_time' column
    dataframe = dataframe.drop(columns=['start_time'])
    dataframe['start'] = dataframe['start'].apply(time_to_seconds)
    dataframe['end'] = dataframe.apply(lambda row: row['end_time'][row['audio']], axis=1)
    dataframe['end'] = dataframe['end'].apply(time_to_seconds)
    dataframe = dataframe.drop(columns=['end_time'])
    # Apply the function to generate the paths for each row
    dataframe['file_path'] = dataframe.apply(lambda row: generate_microphone_paths(row,path), axis=1)
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
    dataframe.drop(columns=['endframe', 'session_id', 'speaker_id','gender', 'nativeness','mother_tongue','audio','start','end','endframe','duration','frames', 'ref'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe 

expanded_df = string_parsing(df,dev_path)
eval_df = string_parsing(eval_df, eval_path)

    









# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import Features, Value
from transformers import WhisperTokenizer
from datasets import load_dataset
device = "cuda" 
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16
model_id = model_name
features = Features({
    'file_path': Value('string'),
    'words': Value('string'),
     'startframe': Value('int64'),
    'num_frames': Value('int64'),
    
    
})
tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
def Hug_dataset_creation(expanded_df, mode):
    dataset = Dataset.from_pandas(expanded_df, features=features)
    shuffled_dataset = dataset.shuffle(seed=42)  
    if mode == 'train':
        return shuffled_dataset
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
train_dataset.save_to_disk("train3.hf")
eval_dataset = eval_dataset.map(prepare_dataset)
eval_dataset.save_to_disk("eval3.hf")
import datasets
train_dataset = datasets.load_from_disk('train3.hf')
eval_dataset = datasets.load_from_disk('eval3.hf')


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
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,  
)

model.to(device)



# In[ ]:


processor = AutoProcessor.from_pretrained(model_id, language='en')

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=1,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
 
)

def transcribe_audio_with_model(dataframe, dataset, pipe):
    for idx, example in enumerate(dataset):
        sample = dataset[idx]["audio"]
        result = pipe(sample)
        dataframe.loc[idx,'results'] = result['text']

    return dataframe
eval_df = eval_df.sample(frac=1, random_state=42).head(1000)
#eval_df = transcribe_audio_with_model(eval_df, eval_dataset,pipe)
from tqdm import tqdm 
expanded_df= eval_df
from tqdm import tqdm 

expanded_df['results'] = ''
expanded_df = expanded_df.head(10)
expanded_df.reset_index(drop=True, inplace=True)
print(expanded_df.shape)
# load audio and pad/trim it to fit 30 seconds

def transcribe_audio(expanded_df):
    
    for i in tqdm(range(expanded_df.shape[0])):
        #audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio,_ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i], num_frames=expanded_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        result = pipe(audio_data, generate_kwargs={"language": "english"})

       
        expanded_df.loc[i,'results'] = result['text']
    
    
    return expanded_df
print(expanded_df.columns)
expanded_df=transcribe_audio(expanded_df)
expanded_df.to_csv('dipco_dev.csv', index=False)

    
#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')


#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')

# result the load audio function takes a quarter of the time when the snippets are cut into lenghts of 1:10th


# In[ ]:


# training of the model 
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
import torch

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
        batch["input_features"] = batch["input_features"].to(torch.float16)

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

model.to(device)
# Prepare the input batch


# Decode the output




metric = evaluate.load("wer")
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
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=101,#4000
    gradient_checkpointing=True,
    fp16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
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

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)
#trainer.train()


# In[ ]:


# Chime Normalization of the results 
'''
model_path = "./whisper-small-hi/checkpoint-101"

# Load the model from the safetensors file
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, from_tf=False, config=model_path + "/config.json")
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

'''



# In[ ]:


## visualization of the layers 
import torch.nn as nn 
print([module for module in model.modules() if not isinstance(module, nn.Sequential)])

name_of_part_to_train = 'encoder'
part_to_train = getattr(model, name_of_part_to_train, None)

#for param in part_to_train.parameters():
  #  print(type(param))




# In[ ]:


#freezing parameters of the encoder


# In[ ]:


from torch import optim




# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
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

# Show the plot
plt.show()


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
data = pd.read_csv('dipco_dev.csv')
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




