import meeteval
import wandb
import preprocessing
from peftModification import create_peft_model

from preprocessing import setup_paths, load_and_concatenate_json_files, chime_parsing, dipco_parsing, \
    Hug_dataset_creation, prepare_dataset_seq2seq
from evaluation import compute_chime_metrics, chime_normalisation
from test_Whisper import suppress_specific_warnings, timing_decorator, run_details_valid
from visualizations import plot_WER, plot_loss, visualize_wer, extract_person, extract_session, extract_location, \
    print_wer
from transformers import WhisperTokenizer, AutoModelForAudioClassification
from train import RunDetails
from notification import send_email
import os
os.environ['WANDB_PROJECT'] = 'exprmt'
os.environ['WAND_LOG_MODEL'] = 'true'
wandb.login(key ='37305846834e634f3640e818c42a90f5b26de39a')
train_state = 'T'  # ["T","NT"]
developer_mode = 'Y'  # ['Y','N']
version = "vanilla"  # ["vanilla","peft"]
task = 'transcribe'  # ["classification","joint","transcribe"]

# dipco_path = "/home/niklas/Downloads/Datasets/Dipco/"
chime_path_cluster = '/export/data2/nbaier/espnet/egs2/chime7_task1/asr1/dataset/ChiME6/audio/train'
dataset_name = "dipco"  # ["Chime6", "dipco"]
environment = "laptop"  # ["laptop","cluster"]
device = "cuda"  # ["cuda"]
model_name = model_id = "openai/whisper-tiny.en"  # "openai/whisper-large"
formated_date = preprocessing.get_formated_date()
dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path = setup_paths(
    environment=environment, dataset_name=dataset_name)

run_details = RunDetails(dataset_name=dataset_name, model_id=model_id, environment=environment,
                         train_state=train_state, date=formated_date, version=version, device=device, task=task,
                         developer_mode=developer_mode)
assert run_details_valid(run_details)

import pandas as pd
import torchaudio
from train import RunDetails, trained_model_transcription

df = load_and_concatenate_json_files(transcript_dev_path)
eval_df = load_and_concatenate_json_files(transcript_eval_path)
train_df = load_and_concatenate_json_files(transcript_train_path)
transcriptions = df['words']
print(eval_df.columns)
from transformers import WhisperFeatureExtractor

import inspect
from datasets import Features, Value

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

# expanded_df = expanded_df.drop(expanded_df['audio']=='close-talk')

features = preprocessing.generate_features(run_details)


if run_details.dataset_name == 'Chime6':
    dev_df = chime_parsing(df, run_details)  # dev
    eval_df = chime_parsing(eval_df, run_details)
    expanded_df = chime_parsing(train_df, run_details)

else:
    expanded_df, temp = dipco_parsing(df, run_details, dev_path)
    print(eval_df.columns)

    _, dev_df = dipco_parsing(eval_df, run_details, eval_path)
    eval_df = temp

print(expanded_df.head(10))

print(len(expanded_df['file_path'].head(1)))

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from transformers import WhisperTokenizer
from datasets import load_dataset

torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
model_id = model_name

from transformers import AutoConfig

print(AutoConfig.from_pretrained(model_id))

tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
dfs = [expanded_df, dev_df, eval_df]
dataset_names = ["train_dataset", "eval_dataset", "test_dataset"]
print(expanded_df.columns)
print(features)
train_dataset = Hug_dataset_creation(expanded_df,developer_mode,features)
eval_dataset = Hug_dataset_creation(dev_df,developer_mode,features)
print(eval_df.columns)
print(eval_df.head(10))
test_dataset = Hug_dataset_creation(eval_df,developer_mode,features)

datasets = {name: Hug_dataset_creation(df, developer_mode=run_details.developer_mode, features=features) for name, df in
            zip(dataset_names, dfs)}
train_dataset, eval_dataset, test_dataset = datasets.values()

# dataset = dataset.to_iterable_dataset()

print(train_dataset[0])
import inspect

train_dataset = train_dataset.map(prepare_dataset_seq2seq)
# TODO
def extract_letters(input_string):
    return ''.join([char for char in input_string if char.isalpha()])

# Example usage

model_str = extract_letters(model_name)
train_dataset_path = f"{model_str}_{dataset_name}_train.hf" #TODO
eval_dataset_path = f"{model_str}_{dataset_name}_eval.hf"
test_dataset_path = f"{model_str}_{dataset_name}_test.hf"
if preprocessing.mapped_dataset_exists(train_dataset_path):
    import datasets
    print("datasets alreaady mapped")

    train_dataset = datasets.load_from_disk(train_dataset_path)
    eval_dataset = datasets.load_from_disk(eval_dataset_path)
    test_dataset = datasets.load_from_disk(test_dataset_path)
else:
    train_dataset, eval_dataset, test_dataset = preprocessing.map_datasets(run_details=run_details, train_dataset=train_dataset,
                                                                           eval_dataset=eval_dataset,
                                                                           test_dataset=test_dataset)
    train_dataset.save_to_disk(train_dataset_path)
    eval_dataset.save_to_disk(eval_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)

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

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch_dtype,
)

processor = AutoProcessor.from_pretrained(model_id, language='en', task="transcribe")

if ("large") in model_id:
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
        # audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio, _ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i],
                                   num_frames=expanded_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        if ("openai/whisper-large") in model_id:
            result = pipe(audio_data, generate_kwargs={"language": "english"})
        else:
            result = pipe(audio_data)

        expanded_df.loc[i, 'results'] = result['text']

    return expanded_df


def transcribe_dataset(dataset):
    for i in tqdm(range(expanded_df.shape[0])):
        # audio = whisper.load_audio('output_segments/segment_' + str(i + 1) + '.wav')
        audio, _ = torchaudio.load(expanded_df['file_path'][i], frame_offset=expanded_df['startframe'][i],
                                   num_frames=expanded_df['num_frames'][i])
        audio_data = audio.squeeze().numpy()
        print(audio_data.shape)
        if ("openai/whisper-large") in model_id:
            result = pipe(audio_data, generate_kwargs={"language": "english"})
        else:
            result = pipe(audio_data)

        expanded_df.loc[i, 'results'] = result['text']

    return expanded_df


print(expanded_df.columns)
#expanded_df = transcribe_audio(expanded_df)
dev = "dev"

import re

# Regex pattern splits on substrings "; " and ", "
components = re.split('-|/|', model_id)
model_size = components[2]
transcription_csv_path = f'{dataset_name}_{dev}_{model_size[:4]}_{train_state}.csv'
expanded_df.to_csv(transcription_csv_path, index=False)

# cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')


# cProfile.run("transcribe_audio(expanded_df,model)", 'whisper_resultssmall.prof')

# result the load audio function takes a quarter of the time when the snippets are cut into lenghts of 1:10th





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

# chime normalization
import jiwer

# peft
if version == 'peft':
    model = create_peft_model(model)

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

if task == 'classification':
    metric = evaluate.load("accuracy")
    dataset = dataset.select_columns('filepath')
    label2id, id2label = dict(), dict()
    labels = dataset["train"].features["label"].names
    num_labels = 4
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = model = AutoModelForAudioClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,

    )

training_args = Seq2SeqTrainingArguments(
    output_dir=f'trained_models/{task}/{dataset_name}/{version}/{model_id}',
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=300,  # 4000
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to='wandb',
    run_name = f'{task}_{dataset_name}_{version}_{model_id}',
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

)
# english_feature_extractor = processor.feature_extractor(language='en')
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


if train_state == 'NT':
    pass
else:
    trainer.train()

    plot_loss(trainer)
    plot_WER(trainer, Run_details=run_details)

# Chime Normalization of the results
model_path = "./whisper-small-hi/checkpoint-101"

# Load the model from the safetensors file
# transcriptions = trained_model_transcription(model=model, eval_dataset=eval_dataset, Run_details=Run_details)


print(inspect.signature(model))
# Generate the output

# Decode the output


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

data = pd.read_csv(transcription_csv_path)
data = eval_df
print(data.head)
# dataset = dataset.map(lambda example: {'normalized_ref': chime_normalisation(example['words'])})
data['chime_ref'] = [chime_normalisation(text) for text in data["words"]]
data['chime_hyp'] = [chime_normalisation(text) for text in str(data["results"])]

wer = jiwer.wer(list(data["chime_ref"]), list(data["chime_hyp"]))
# WER of the whisper normalizer
print(f"WER: {wer * 100:.2f} %")

print(data.sample(n=10))
data['wer'] = data.apply(
    lambda row: meeteval.wer.wer.siso.siso_word_error_rate(
        reference=row['chime_ref'],
        hypothesis=row['chime_hyp']
    ),
    axis=1
)

ascii_pattern = r'^[\x00-\x7F]*$'
# Step 3: Filter the DataFrame
print(data.shape)
df_ascii = data[data['chime_hyp'].str.contains(ascii_pattern, na=False)]
print(df_ascii.shape)
wer = jiwer.wer(list(df_ascii["chime_ref"]), list(df_ascii["chime_hyp"]))

print(f"WER: {wer * 100:.2f} %")

data['session_number'] = data['file_path'].apply(extract_session)
data['mic_type'] = data['file_path'].apply(extract_person)
data['mic_number'] = data['file_path'].apply(extract_location)
grouped_ses = data.groupby('session_number')
print_wer(grouped_ses, "session")
grouped_mic_type = data.groupby('mic_type')
grouped_mic = data.groupby(['mic_type', 'mic_number'])
print_wer(grouped_mic, "mic_type")
print(wer)

# plot visualization of the different sessions and store the results

import re
import matplotlib.pyplot as plt

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

error_rates = data['wer'].apply(lambda x: x.error_rate)

# Calculate the mean of the error rates
mean_error_rate = error_rates.mean()
print(mean_error_rate)

raise ValueError()
from huggingface_hub import notebook_login
# ***REMOVED***
# notebook_login()


