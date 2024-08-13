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

def generate_microphone_paths(row):
    paths = []
    for i in range(1, 7):
        path = f"{dev_path}/{row['session_id']}_{row['audio']}.CH{i}.wav"
        paths.append(path)

    path = f"{dev_path}/{row['session_id']}_{row['speaker_id']}.wav"
    paths.append(path)
    return paths


# Function to convert time string to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    
    return h * 3600 + m * 60 + s
#change the seconds to frames
def get_Frames(starting_second:float, sample_rate:int, end_second:float )-> List[int] :
     return [int(starting_second*sample_rate), int(end_second*sample_rate)]


#columns_to_drop = ['mother_tongue', 'ref', 'nativeness', 'audio', 'session_id','speaker_id', 'gender']


# print(expanded_df['duration'].max()) yielded that the biggest in the dipco dataset was above 60 seconds for those an additional separation is required 
#expanded_df = expanded_df.drop(columns=columns_to_drop)
# sorting for cache efficiency so far no speedup 
def validate_frames_column(frames_list):
    return len(frames_list) == 2

def get_corresponding_end_time(dict:dict, key:str):
    end_time = [v for k,v in dict if k==key]
    return end_time

def string_parsing(dataframe):
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
    dataframe.drop(columns=['endframe', 'session_id', 'speaker_id','gender', 'nativeness','mother_tongue','audio','start','end','endframe','duration','frames', 'ref'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe 

expanded_df = string_parsing(df)


    












# removal of the end_time

#expanded_df = expanded_df.drop(expanded_df['audio']=='close-talk')


# U01 - U05 & CH 1 - 7 

# Function to generate microphone paths






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
torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32

model_id = model_name
features = Features({
    'file_path': Value('string'),
    'words': Value('string'),
     'startframe': Value('int64'),
    'num_frames': Value('int64'),
    
    
})

dataset = Dataset.from_pandas(expanded_df, features=features)
print(expanded_df.head(10))
#dataset = dataset.to_iterable_dataset()
print(dataset[0])
import inspect

tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
dataset = dataset.select(range(100))

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

dataset = dataset.map(prepare_dataset)
dataset.save_to_disk("testing.hf")
import os
from datasets import load_from_disk, Dataset

# Define the path to the dataset directory
dataset_path = "testing.hf"

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
print(dataset.features)


# In[ ]:


from torch.nn import CrossEntropyLoss
import transformers
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import _CONFIG_FOR_DOC
from transformers.models.whisper.modeling_whisper import *
from typing import Optional, Tuple, Union
import types
class CustomWhisperwithSpeakerAttribution(WhisperForConditionalGeneration):
     def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_outputsout(outputs[0])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
          

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )   
processor = AutoProcessor.from_pretrained(model_id, language="en")
print(inspect.getsource(Seq2SeqLMOutput))



# Update the model config with forced_decoder_ids


custom_model = CustomWhisperwithSpeakerAttribution(model.config)
custom_model.load_state_dict(model.state_dict(), strict=True)


#custom_model.generation_config.language = "<|en|>"

custom_model.to(device)   
print(model.__class__)
print(vars(custom_model.generation_config))
print(inspect.getsource(WhisperForConditionalGeneration))


# In[ ]:





# In[ ]:


class WhisperForConditionalGeneration2(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #self.num_speakers=4
        #self.classifier = nn.Linear(config.vocab_size,self.num_speakers)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])
        #speaker_logits = self.classifier(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels_clas = labels.to(lm_logits.device)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            #loss_classification = lm_logits.view(-1, self.num_speakers)
            #weighted_loss =  0.8 * loss + 0.2 * loss_classification

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1] :]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
        }


# In[ ]:


cu2_model = WhisperForConditionalGeneration2(model.config)
cu2_model.load_state_dict(model.state_dict(), strict=True)


pipe = pipeline(
    "automatic-speech-recognition",
    model=cu2_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
 
)

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
        
        result = pipe(audio_data)

       
        expanded_df.loc[i,'results'] = result['text']
        print(result['text'])
    
    
    return expanded_df
expanded_df=transcribe_audio(expanded_df)
expanded_df.to_csv('dipco_eval.csv', index=False)

    
#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')


#cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')

# result the load audio function takes a quarter of the time when the snippets are cut into lenghts of 1:10th
    


# In[ ]:


# save the dataset in zip format so that there is no conflict with data memory on cluster 
import tarfile
import os 
import shutil
# Open the tar.gz file
def delete_file(dataset_name):
    # Check if the item exists
    if os.path.exists(dataset_name):
        # Check if the item is a file
        if os.path.isfile(dataset_name):
            # Remove the file
            os.remove(dataset_name)
            print(f'{dataset_name} has been removed.')
        # Check if the item is a directory
        elif os.path.isdir(dataset_name):
            # Remove the directory and its contents
            shutil.rmtree(dataset_name)
            print(f'{dataset_name} directory and its contents have been removed.')
    else:
        print(f'{dataset_name} does not exist.')
        

def get_tarname_from_hf(dataset_name):
    dataset_name_without_suffix = dataset_name.split('.')[0]
    tar_gz_filename = f'{dataset_name_without_suffix}.tar.gz'
    return tar_gz_filename

def create_hf_from_tar_gz(dataset_name):
    tar_gz_filename = get_tarname_from_hf(dataset_name)
    with tarfile.open(tar_gz_filename, 'r:gz') as tar:
    # Check if the file exists in the tar archive
        if dataset_name in tar.getnames():
            # Extract the specified file
            tar.extract(dataset_name, path='.')
            print(f'{dataset_name} has been extracted from {tar_gz_filename}.')
        else:
            print(f'{dataset_name} does not exist in {tar_gz_filename}.')
    
        
def create_compression(dataset_name):
    tar_gz_filename = get_tarname_from_hf(dataset_name)
    with tarfile.open(tar_gz_filename, 'w:gz') as tar:
        tar.add(dataset_name, arcname=dataset_name)
        print(f'{dataset_name} has been added to {tar_gz_filename}')

def store_dataset(dataset_name):
    create_compression(dataset_name)
    delete_file(dataset_name)
    print('dataset has been stored')

def get_dataset_from_disk(dataset_name):
    tar_gz_filename = get_tarname_from_hf(dataset_name)
    create_hf_from_tar_gz(dataset_name)
    delete_file(tar_gz_filename)
dataset.save_to_disk('exp.hf')
store_dataset('exp.hf')
get_dataset_from_disk('exp.hf')
    
    






# In[ ]:


import inspect
import transformers
from transformers import WhisperForConditionalGeneration
transformers_path = os.path.dirname(transformers.__file__)


# In[ ]:





# In[ ]:





# In[ ]:


def are_models_equal(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if both state dicts have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Check if all tensors are equal
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

equal = are_models_equal(model, custom_model)
print(f"Models are equal: {equal}")


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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=400,#4000
    gradient_checkpointing=True,
    fp16=True,
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
    model=cu2_model,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)
trainer.train()


# In[ ]:





# In[ ]:


"""import torch.nn as nn 
class CustomLoss(nn.Module):
    def __init__(self, speaker_weight=1.0):
        super(CustomLoss, self).__init__()
        self.speaker_weight = speaker_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets, speaker_predictions, speaker_labels):
        # Standard cross-entropy loss for transcription errors
        transcription_loss = self.ce_loss(predictions, targets)
        
        # Additional loss for speaker attribution errors
        speaker_loss = self.ce_loss(speaker_predictions, speaker_labels)
        
        # Weighted combination
        total_loss = transcription_loss + self.speaker_weight * speaker_loss
        return total_loss

class CustomTrainer(Seq2SeqTrainer):
      def __init__(self, speaker_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = CustomLoss(speaker_weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        speaker_labels = inputs.get("speaker_labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        speaker_logits = outputs.get("speaker_logits")
        
        # Compute custom loss
        loss = self.custom_loss(logits, labels, speaker_logits, speaker_labels)
        
        return (loss, outputs) if return_outputs else loss

custom_trainer = CustomTrainer()"""


# In[ ]:


# Chime Normalization of the results 
model_path = "./whisper-small-hi/checkpoint-101"

# Load the model from the safetensors file
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, from_tf=False, config=model_path + "/config.json")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task="transcribe", language="en")
# Load the tokenizer (if necessary)



# Example input
print(expanded_df.columns)


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

print(inspect.signature(model))
# Generate the output
outputs = model.generate(input_features)
transcription = processor.batch_decode(outputs, skip_special_tokens=True)
# Decode the output

print(transcription)



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
data = pd.read_csv('/home/niklas/dipco_eval.csv')
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




