import meeteval
import wandb
import preprocessing
from peftModification import create_peft_model

from preprocessing import setup_paths, load_and_concatenate_json_files, chime_parsing, dipco_parsing, \
    Hug_dataset_creation, prepare_dataset_seq2seq
from evaluation import compute_chime_metrics, chime_normalisation
from test_Whisper import suppress_specific_warnings, timing_decorator, run_details_valid
from visualizations import plot_WER, plot_loss, visualize_wer, extract_person, extract_session, extract_location, \
    print_wer, visualize_results
from transformers import WhisperTokenizer, AutoModelForAudioClassification
from train import RunDetails, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, transcribe_audio, \
    PrintTrainableParamsCallback, freeze_all_layers_but_last
from notification import send_email
import os
os.environ['WANDB_PROJECT'] = 'WHISPER'
os.environ['WAND_LOG_MODEL'] = 'true'
#wandb.login(key ='37305846834e634f3640e818c42a90f5b26de39a')
train_state = 'T'  # ["T","NT"]
developer_mode = 'Y'  # ['Y','N']
version = "last-layer"  # ["vanilla","peft", "last-layer"]
task = 'transcribe'  # ["classification","joint","transcribe"]

# dipco_path = "/home/niklas/Downloads/Datasets/Dipco/"

dataset_name = "dipco"  # ["Chime6", "dipco"]
environment = "laptop"  # ["laptop","cluster", "bwcluster"]
device = "cuda"  # ["cuda", "cpu"]
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
if run_details.dataset_name == 'Chime6':
    train_df = load_and_concatenate_json_files(transcript_train_path)

transcriptions = df['words']

from transformers import WhisperFeatureExtractor

import inspect
from datasets import Features, Value

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

# expanded_df = expanded_df.drop(expanded_df['audio']=='close-talk')

features = preprocessing.generate_features(run_details)
print(features)
print("hi")
# Example usage


if run_details.dataset_name == 'Chime6':
    dev_df = chime_parsing(df, run_details,dev_path)  # dev
    eval_df = chime_parsing(eval_df, run_details,eval_path)
    expanded_df = chime_parsing(train_df, run_details,train_path)

else:
    expanded_df, dev_df = dipco_parsing(df, run_details, dev_path)
    #TODO Verify


    eval_df, eval_df2 = dipco_parsing(eval_df, run_details, eval_path)
    eval_df = pd.concat([eval_df,eval_df2])



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

train_dataset = Hug_dataset_creation(expanded_df,developer_mode,features)
eval_dataset = Hug_dataset_creation(dev_df,developer_mode,features)

test_dataset = Hug_dataset_creation(eval_df,developer_mode,features)

'''datasets = {name: Hug_dataset_creation(df, developer_mode=run_details.developer_mode, features=features) for name, df in
            zip(dataset_names, dfs)}
train_dataset, eval_dataset, test_dataset = datasets.values()'''

# dataset = dataset.to_iterable_dataset()


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
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {num_params}")
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
elif version == "last-layer":
    model = freeze_all_layers_but_last(model)
# training of the model
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
import torch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

import evaluate

metric = evaluate.load("wer")


train_batch_size, per_device_eval_batch_size, max_steps, loggings_steps,save_steps, output_dir, run_name = generate_training_args(run_details)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=max_steps,  # 4000
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=per_device_eval_batch_size,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=save_steps,
    eval_steps=100,
    logging_steps=loggings_steps,
    report_to='wandb',
    run_name = run_name,
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
    callbacks= [PrintTrainableParamsCallback()]
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
model_path = output_dir
visualize_results(transcription_csv_path, eval_df, run_details)
raise ValueError()
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




from huggingface_hub import notebook_login
# ***REMOVED***
# notebook_login()


