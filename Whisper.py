
import meeteval
import datasets

import wandb
import pdb
import preprocessing
from logrun import log_run
from peftModification import create_peft_model
from pathlib import Path
from preprocessing import setup_paths, load_and_concatenate_json_files, chime_parsing, dipco_parsing, \
    Hug_dataset_creation, prepare_dataset_seq2seq
from evaluation import compute_chime_metrics, chime_normalisation
from test_Whisper import suppress_specific_warnings, timing_decorator, run_details_valid
from visualizations import plot_WER, plot_loss, visualize_wer, extract_person, extract_session, extract_location, \
    print_wer, visualize_results
from transformers import WhisperTokenizer, AutoModelForAudioClassification
from train import RunDetails, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, transcribe_audio, \
    PrintTrainableParamsCallback, freeze_all_layers_but_last, get_parser, transcribe_results
from notification import send_email
import os
os.environ['WANDB_PROJECT'] = 'WHISPER'
os.environ['WAND_LOG_MODEL'] = 'true'
#wandb.login(key ='37305846834e634f3640e818c42a90f5b26de39a')
# setting the run details
parser = get_parser()
args = parser.parse_args()

formated_date = preprocessing.get_formated_date()
dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path = setup_paths(
    environment=args.environment, dataset_name=args.dataset_name)

run_details = RunDetails(dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,
                         train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,
                         developer_mode=args.developer_mode, augmentation=args.augmentation)
assert run_details_valid(run_details)

import pandas as pd


df = load_and_concatenate_json_files(transcript_dev_path)
eval_df = load_and_concatenate_json_files(transcript_eval_path)
if run_details.dataset_name == 'Chime6':
    train_df = load_and_concatenate_json_files(transcript_train_path)

transcriptions = df['words']

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(run_details.model_id)


features = preprocessing.generate_features(run_details)



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
model_id = model_name = run_details.model_id



tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
dfs = [expanded_df, dev_df, eval_df]
dataset_names = ["train_dataset", "eval_dataset", "test_dataset"]

train_dataset = Hug_dataset_creation(expanded_df,run_details.developer_mode,features)
eval_dataset = Hug_dataset_creation(dev_df,run_details.developer_mode,features)

test_dataset = Hug_dataset_creation(eval_df,run_details.developer_mode,features)

'''datasets = {name: Hug_dataset_creation(df, developer_mode=run_details.developer_mode, features=features) for name, df in
            zip(dataset_names, dfs)}
train_dataset, eval_dataset, test_dataset = datasets.values()'''

# dataset = dataset.to_iterable_dataset()




# TODO
def extract_letters(input_string):
    return ''.join([char for char in input_string if char.isalpha()])

# Example usage


train_dataset_path,eval_dataset_path, test_dataset_path = preprocessing.generate_dataset_paths(run_details=run_details)
if not(preprocessing.mapped_dataset_exists(train_dataset_path)):
    print("dataset not mapped yet")
    dataset_paths = {"train": train_dataset_path, "eval":eval_dataset_path, "test":test_dataset_path}
    preprocessing.map_datasets(run_details=run_details, train_dataset=train_dataset,
                                                                           eval_dataset=eval_dataset,
                                                                           test_dataset=test_dataset,dataset_paths=dataset_paths)

train_dataset = datasets.load_from_disk(train_dataset_path)
eval_dataset = datasets.load_from_disk(eval_dataset_path)
test_dataset = datasets.load_from_disk(test_dataset_path)

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch_dtype,
)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {num_params}")
processor = AutoProcessor.from_pretrained(model_id, language='en', task="transcribe")

if ("large" or "medium") in model_id:
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
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=run_details.device

)


eval_df['results'] = ''

eval_df.reset_index(drop=True, inplace=True)





import re

# Regex pattern splits on substrings "; " and ", "
components = re.split('-|/|.|', model_id)
model_size = components[2]
transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
if(Path(transcription_csv_path).is_file()):
    print("transcription csv already exists")
    print(transcription_csv_path)
else:
    eval_df = transcribe_audio(eval_df=eval_df, pipe=pipe, run_details=run_details)
    eval_df.to_csv(transcription_csv_path, index=False)



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
if run_details.version == 'peft':
    model = create_peft_model(model)
elif run_details.version == "last-layer":
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


if run_details.train_state == 'NT':
    visualize_results(transcription_csv_path, run_details)
else:
    trainer.train()
    plot_loss(trainer, run_details=run_details)
    plot_WER(trainer, Run_details=run_details)
    log_run(run_details=run_details)
    model_path = output_dir
    #TODO take it from the model




transcribe_results(test_dataset=test_dataset,trainer=trainer)
visualize_results(transcription_csv_path, run_details)


raise ValueError()





from huggingface_hub import notebook_login
# ***REMOVED***
# notebook_login()


