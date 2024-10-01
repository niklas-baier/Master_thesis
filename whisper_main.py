import evaluate
import json

import jiwer

import evaluation
import preprocessing
from logrun import log_run
from test_Whisper import run_details_valid
from visualizations import visualize_results, plot_loss, plot_WER, plot_tsne
from train import RunDetails, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, get_parser, transcribe_results,transcribe_raw, create_tokenizer_model_processor, generate_datasets
import os
import torch
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer
def compute_chime_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    results = {"predictions": pred_str, "labels": label_str}
    results_directory = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    file_path = os.path.join(results_directory, "results.json")

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

# Define the file path


# Write the evaluation results to the file
    # Example evaluation results
    chime_normalized_reference = [evaluation.chime_normalisation( reference ) for reference in label_str]
    chime_normalized_prediction = [evaluation.chime_normalisation( pred ) for pred in pred_str]

    #wer = 100 * metric.compute(predictions=chime_normalized_prediction, references=chime_normalized_reference)
    wer = jiwer.wer(hypothesis = list(chime_normalized_prediction), reference = list(chime_normalized_reference))

    return {"wer": wer}


os.environ['WANDB_PROJECT'] = 'WHISPER'
os.environ['WAND_LOG_MODEL'] = 'true'
torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
parser = get_parser()
args = parser.parse_args()
formated_date = preprocessing.get_formated_date()
run_details = RunDetails(dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,
                         train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,
                         developer_mode=args.developer_mode, augmentation=args.augmentation, additional_tokens=args.additional_tokens)

assert run_details_valid(run_details)
features = preprocessing.generate_features(run_details)
expanded_df, dev_df, eval_df = preprocessing.generate_dfs(args=args, run_details=run_details)
tokenizer, model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
train_dataset, eval_dataset, test_dataset = generate_datasets(run_details=run_details, args=args, expanded_df=expanded_df,eval_df=eval_df, dev_df=dev_df, features=features)
transcription_csv_path = preprocessing.generate_transcription_csv_path(run_details)
eval_df.to_csv(transcription_csv_path, index=False)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
metric = evaluate.load("wer")
train_batch_size, per_device_eval_batch_size, max_steps, loggings_steps,save_steps, output_dir, run_name = generate_training_args(run_details)
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    logging_dir='./logs',
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
    fp16=True,
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
)
processor.save_pretrained(training_args.output_dir)
#plot_tsne(model=model, processor=processor, test_dataset=test_dataset, torch_dtype=torch_dtype, run_details=run_details)
if run_details.train_state == 'NT':
    transcription_csv_path = transcribe_raw(model=model, run_details=run_details, torch_dtype=torch_dtype,eval_df=eval_df, processor=processor)
    visualize_results(transcription_csv_path, run_details)
else:
    #plot_tsne(trainer=trainer, run_details=run_details,test_dataset=test_dataset, torch_dtype=torch_dtype,processor = processor)
    trainer.train()
    plot_loss(trainer, run_details=run_details)
    plot_WER( trainer, run_details=run_details )
    log_run(run_details=run_details)
    model_path = output_dir
    #TODO take it from the mode
    pass

# significantly faster than pandas dataframe
transcription_csv_path_trained = transcribe_results(test_dataset=test_dataset,trainer=trainer, run_details=run_details)
visualize_results(transcription_csv_path_trained, run_details)
raise ValueError()



