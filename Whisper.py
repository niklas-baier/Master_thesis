
import datasets
import evaluate
import pandas as pd
from transformers import WhisperFeatureExtractor
import preprocessing
from augmentations import generate_noise_dataset
from logrun import log_run
from peftModification import create_peft_model
from preprocessing import setup_paths, load_and_concatenate_json_files, chime_parsing, dipco_parsing, \
    Hug_dataset_creation, prepare_dataset_seq2seq
from evaluation import compute_chime_metrics, chime_normalisation
from test_Whisper import suppress_specific_warnings, timing_decorator, run_details_valid
from visualizations import plot_WER, plot_loss, visualize_wer, extract_person, extract_session, extract_location, \
    print_wer, visualize_results
from train import RunDetails, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, transcribe_audio, \
    PrintTrainableParamsCallback, freeze_all_layers_but_last, get_parser, transcribe_results, get_model_size, \
    transcribe_raw
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments,Seq2SeqTrainer, WhisperTokenizer
from datasets import load_dataset
from huggingface_hub import notebook_login

os.environ['WANDB_PROJECT'] = 'WHISPER'
os.environ['WAND_LOG_MODEL'] = 'true'

parser = get_parser()
args = parser.parse_args()
formated_date = preprocessing.get_formated_date()
dataset_path, dev_path, eval_path, transcript_dev_path, transcript_eval_path, train_path, transcript_train_path = setup_paths(
    environment=args.environment, dataset_name=args.dataset_name)

run_details = RunDetails(dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,
                         train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,
                         developer_mode=args.developer_mode, augmentation=args.augmentation, additional_tokens=args.additional_tokens)
assert run_details_valid(run_details)
df = load_and_concatenate_json_files(transcript_dev_path)
eval_df = load_and_concatenate_json_files(transcript_eval_path)
if run_details.dataset_name == 'Chime6':
    train_df = load_and_concatenate_json_files(transcript_train_path)
transcriptions = df['words']
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

torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
model_id = model_name = run_details.model_id
tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe", language="en")
tokenizer.set_prefix_tokens(language="english")
dfs = [expanded_df, dev_df, eval_df]
dataset_names = ["train_dataset", "eval_dataset", "test_dataset"]
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
if run_details.additional_tokens=="Y":
    # define new tokens to add to vocab
    new_tokens = ['[laugh]', '[unintelligible]', '[noise]', ]
    # check if the new tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))
    # add new random embeddings for the appended tokens
    model.resize_token_embeddings(len(tokenizer))
train_dataset_path,eval_dataset_path, test_dataset_path = preprocessing.generate_dataset_paths(run_details=run_details)
if not(preprocessing.mapped_dataset_exists(train_dataset_path)):
    print("dataset not mapped yet")
    dataset_paths = {"train": train_dataset_path, "eval":eval_dataset_path, "test":test_dataset_path}
    train_dataset = Hug_dataset_creation(expanded_df, run_details.developer_mode, features, test_dataset=False)
    eval_dataset = Hug_dataset_creation(dev_df, run_details.developer_mode, features, test_dataset=False)
    test_dataset = Hug_dataset_creation(eval_df, run_details.developer_mode, features, test_dataset=True)
    preprocessing.map_datasets(run_details=run_details, train_dataset=train_dataset,
                                                                           eval_dataset=eval_dataset,
                                                                           test_dataset=test_dataset,dataset_paths=dataset_paths)

train_dataset = datasets.load_from_disk(train_dataset_path)
eval_dataset = datasets.load_from_disk(eval_dataset_path)
test_dataset = datasets.load_from_disk(test_dataset_path)

if args.augmentation == "Y":
    noisy_train_dataset_path = "noise"+ train_dataset_path
    if not(preprocessing.mapped_dataset_exists(noisy_train_dataset_path)) :
        noisy_train_dataset_path = generate_noise_dataset(expanded_df=expanded_df,run_details= run_details,features=features)
    train_dataset = datasets.load_from_disk(noisy_train_dataset_path)

eval_df['results'] = eval_df['words']
eval_df.reset_index(drop=True, inplace=True)
model_size = get_model_size(run_details.model_id)
transcription_csv_path = f'{run_details.dataset_name}_eval_{model_size}_{run_details.train_state}.csv'
eval_df.to_csv(transcription_csv_path, index=False)

if run_details.version == 'peft':
    model = create_peft_model(model)
elif run_details.version == "last-layer":
    model = freeze_all_layers_but_last(model)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
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
    remove_unused_columns=True

)

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
    transcription_csv_path = transcribe_raw(model=model, run_details=run_details, torch_dtype=torch_dtype,eval_df=eval_df, processor=processor)
    visualize_results(transcription_csv_path, run_details)
else:
    pass

# significantly faster than pandas dataframe
breakpoint()
transcription_csv_path_trained = transcribe_results(test_dataset=test_dataset,trainer=trainer,transcription_csv_path=transcription_csv_path, run_details=run_details)
visualize_results(transcription_csv_path_trained, run_details)
raise ValueError()



