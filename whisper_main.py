import evaluate
from absl import flags, app
from torchsummary import summary
import numpy as np 
import json
import glob 
import re
import polars as pl
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from peft import PeftModel, PeftConfig
import jiwer
import evaluation
from test_Whisper import check_no_missing_values
import preprocessing
from datasets import Dataset
import pandas as pd
from logrun import log_run
from test_Whisper import run_details_valid
from visualizations import visualize_results, plot_loss, plot_WER, plot_tsne
from train import RunDetails, add_prediction_column, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, \
    get_model_size, transcribe_raw, create_tokenizer_model_processor, generate_datasets, get_cached_components
from config import get_parser
import os
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, WhisperForConditionalGeneration, EarlyStoppingCallback, Trainer
from torch.utils.data import DataLoader, Subset 
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction
import copy 
from typing import Optional
def main(argv):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the current working directory to the directory where whisper_main.py is located
    os.chdir(script_dir)
    os.environ['WANDB_PROJECT'] = 'WHISPER'
    os.environ['WAND_LOG_MODEL'] = 'true'
    torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
    parser = get_parser()
    args = parser.parse_args()
    formated_date = preprocessing.get_formated_date() 
    run_details = RunDetails(dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,
                            train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,
                            developer_mode=args.developer_mode, augmentation=args.augmentation, run_notes=args.run_notes, additional_tokens=args.additional_tokens, dataset_evaluation_part=args.dataset_evaluation_part,oversampling = args.oversampling_clean_data, checkpoint_path=args.checkpoint, data_portion=args.data_portion, beamforming=args.beamforming, SWAD=args.SWAD, diffusion=args.diffusion)

    assert run_details_valid(run_details)
    features = preprocessing.generate_features(run_details)
    expanded_df, dev_df, eval_df = preprocessing.generate_dfs(args=args, run_details=run_details)
    breakpoint()
    expanded_df['words'] = expanded_df['words'].apply(evaluation.chime_normalisation)
    dev_df['words'] = dev_df['words'].apply(evaluation.chime_normalisation)
    tokenizer, model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
    train_dataset, eval_dataset, test_dataset = generate_datasets(run_details=run_details, args=args, expanded_df=expanded_df,eval_df=eval_df, dev_df=dev_df, features=features)
    transcription_csv_path = preprocessing.generate_transcription_csv_path(run_details)
    eval_df.to_csv(transcription_csv_path, index=False)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    metric = evaluate.load("wer")
    #train_batch_size, per_device_eval_batch_size, max_steps, loggings_steps,save_steps, output_dir, run_name = generate_training_args(run_details)

    training_args = generate_training_args(run_details)
    trainer = get_trainer(run_details=run_details, training_args=training_args, data_collator= data_collator,train_dataset=train_dataset,eval_dataset=eval_dataset, model=model, processor=processor )


    processor.save_pretrained(training_args.output_dir)
    #plot_tsne(model=model, processor=processor, test_dataset=test_dataset, torch_dtype=torch_dtype, run_details=run_details)
    if run_details.train_state == 'NT':
        transcription_csv_path_trained = transcribe_results( test_dataset=test_dataset, trainer=trainer,
                                                            run_details=run_details )
        run_results = visualize_results(transcription_csv_path_trained, run_details)
        log_run( run_details=run_details, run_results=run_results )
    else:
        #plot_tsne(trainer=trainer, run_details=run_details,test_dataset=test_dataset, torch_dtype=torch_dtype,processor = processor)
        start_time = time.perf_counter()
        trainer.train()

        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    
        #model.push_to_hub(peft_model_id)
        transcription_csv_path_trained = transcribe_results( test_dataset=test_dataset, trainer=trainer,
                                                            run_details=run_details )
        run_results = visualize_results( transcription_csv_path_trained, run_details )

        plot_loss(trainer, run_details=run_details)
        plot_WER( trainer, run_details=run_details )
        log_run(run_details=run_details, run_results=run_results,training_time=elapsed_time)

        #TODO take it from the mode
    return 



  


def get_latest_checkpoint(path):
    # Find all directories matching the pattern checkpoint-{number}
    checkpoint_dirs = glob.glob(os.path.join(path, "checkpoint-*"))
    
    # Extract numbers and find the highest one
    numbers = []
    for dir_path in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)$', dir_path)
        if match:
            numbers.append(int(match.group(1)))
    
    if not numbers:
        return None
    
    # Get the highest number and construct the full path
    latest_checkpoint = f"checkpoint-{max(numbers)}"
    result_path = os.path.join(path, latest_checkpoint)
    
    return result_path
def average_weights(model_1, model_2, model_3):
    import copy
    averaged_model = copy.deepcopy(model_1)  # Create a new model to store averaged weights
    
    # Iterate over model parameters
    for param_name, param_1 in model_1.state_dict().items():
        param_2 = model_2.state_dict()[param_name]
        param_3 = model_3.state_dict()[param_name]
        
        # Average the weights
        averaged_param = (param_1 + param_2+param_3) / 3.0
        
        # Set the averaged weights in the new model
        averaged_model.state_dict()[param_name].copy_(averaged_param)
    
    return averaged_model

def get_peft_model(trainer,run_details):
    path = trainer.args.output_dir
    highest_checkpoint_path = get_latest_checkpoint(path)
    df = pl.read_json(os.path.join(highest_checkpoint_path, 'trainer_state.json'))
    best_checkpoint = df['best_model_checkpoint'].item()
    model = get_normal_model(run_details)
    adapter_path = best_checkpoint
    adapter_name = model.load_adapter(adapter_path)
    model.active_adapters = adapter_name
    if run_details.SWAD == True:
        second_best_path, third_best_path = get_top_lora_paths(adapter_path=adapter_path, df=df)
        model_1 = get_normal_model(run_details)
        model_2 = copy.deepcopy(model_1)
        model_2.load_adapter(second_best_path)
        model_3 = copy.deepcopy(model_1)
        model_3.load_adapter(third_best_path)
        model_1.load_adapter(adapter_path)
        assert(model_1.state_dict().keys() == model_2.state_dict().keys())

        averaged_model = average_weights(model_1=model_1, model_2=model_2, model_3=model_3)
        return averaged_model
    return model 

def get_top_lora_paths(adapter_path,df):
   eval_wers = [entry['eval_wer'] for log in df['log_history'] for entry in log if 'eval_wer' in entry]
   steps = [entry['step'] for log in df['log_history'] for entry in log if 'step' in entry]
   zipped = list(zip(eval_wers[1::2], steps[::2]))
   zipped.sort() # sorts after the first element
   steps = [ steps for (_,steps) in zipped[:3]] # get first 3 steps 
   second_best = adapter_path.replace(str(steps[0]),str(steps[1]))
   third_best = adapter_path.replace(str(steps[0]),str(steps[2]))
   breakpoint()
   return second_best, third_best


def get_normal_model(run_details):
    copy = run_details
    copy.version = "vanilla"
    _, model , _ = create_tokenizer_model_processor(copy, torch_dtype=torch.float32)
    return model
def compute_metrics(pred:EvalPrediction)->dict:
    from train import _cached_tokenizer
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    tokenizer = _cached_tokenizer
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    if len(pred_ids) == 2:
        first_value = pred_ids[0]
        pred_ids = np.argmax(first_value, axis=-1)  # Shape: (598, 81) Assuming those are logits

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True) # this approach is faster than a parallelized threadpoolexecutor approach
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def batch_decode_parallel(tokenizer, pred_ids, label_ids, skip_special_tokens=True):
    # Define a function to decode a batch
    def decode(ids):
        return tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)
    
    # Run decoding in parallel for predictions and labels
    with ThreadPoolExecutor(max_workers=2) as executor:
        pred_str, label_str = executor.map(decode, [pred_ids, label_ids])
    
    return list(pred_str), list(label_str)

def compute_chime_metrics(pred:EvalPrediction)->dict:
    from preprocessing import Paths
    paths = Paths.get_instance()
    tokenizer,_,_ = get_cached_components()
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    if len(pred_ids) == 2:
        first_value = pred_ids[0]
        pred_ids = np.argmax(first_value, axis=-1)  # Shape: (598, 81) Assuming those are logits
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    results = {"predictions": pred_str, "labels": label_str}
    results_directory = paths.prediction_path
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

        # wer = 100 * metric.compute(predictions=chime_normalized_prediction, references=chime_normalized_reference)
        #wer = jiwer.wer( hypothesis=list( chime_normalized_prediction ), reference=list( chime_normalized_reference ) )
        wer = jiwer.wer(hypothesis=chime_normalized_prediction, reference = chime_normalized_reference)
        # TODO different normalizers for eval and testing ?

        return {"wer": wer}

def transcribe_results(*, test_dataset:Dataset, trainer:Seq2SeqTrainer, run_details:RunDetails) -> str:
    #ID 172
   
    #results = trainer.evaluate( eval_dataset=test_dataset )
    if run_details.version == "peft":
       model = get_peft_model(trainer, run_details)
       trainer.args.model = model
       '''
       total_size = len(test_dataset)
       part_size = total_size // 20
       #create 20 slices
       lazy_parts = [test_dataset.select(range(i * part_size, (i + 1) * part_size))for i in range(20)]
       if total_size % 20 != 0:
           lazy_parts[-1] = test_dataset.select(range(19 * part_size, total_size))
       predictions = [predict_logits_and_get_strings_from_them(trainer,x) for x in lazy_parts]
       predictions = pl.concat(predictions, how='vertical')
       print("hi")'''
       predictions = predict(trainer=trainer, test_dataset=test_dataset)

       

    else:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        model_size = "distil-large-v3"
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        #breakpoint()
        #segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)
        #texts = [segment.text for segment in segments]
        #print(texts[0:10])
        start_time_transcription= time.perf_counter()
        predictions = predict( trainer=trainer, test_dataset=test_dataset )
        end_time_transcription = time.perf_counter()
        inference_time = end_time_transcription-start_time_transcription
        breakpoint()
        print(inference_time)
        
    path = save_evaluation_results_as_csv( run_details, results=predictions )
    return path

def predict_logits_and_get_strings_from_them(trainer:Seq2SeqTrainer, dataset_slice:Dataset) -> pl.DataFrame:
    tokenizer,_,_ = get_cached_components
    predict = trainer.predict(dataset_slice)
    predictions = predict.predictions[0]
    logits = np.argmax(predictions, axis=-1)
    result = tokenizer.batch_decode(logits, skip_special_tokens=True)
    map_to_strings = partial(tokenizer.batch_decode, skip_special_tokens=True)
    result_predictions = map_to_strings(logits)
    result_labels = map_to_strings(predict.label_ids)
    df = pl.DataFrame({'predictions':result_predictions, 'labels': result_labels})
    return df

def predict(trainer:Seq2SeqTrainer,test_dataset:Dataset) -> pl.DataFrame:
    tokenizer,_,processor = get_cached_components()
    import time
    start_time_transcription= time.perf_counter()
    if 1==1:
           trainer.model.config.output_hidden_states = True
           results = []
           collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
           test_dataloader= DataLoader(test_dataset, batch_size=16, collate_fn=collator)
           prediction_sentences = []
           labels_list = []
           model = trainer.model.eval()
           model = torch.compile(model)
           for batch in test_dataloader:  # Ensure you have a DataLoader for test_dataset
                with torch.no_grad():
                  
                    #inputs_features_of_batch = processor()
                    batch.to("cuda")
                    #outputs = model(**batch, output_hidden_states=True)  # simple forward pass is not sufficient for a acceptable WER
                    model.config.output_hidden_states = True # return also the hidden states
                    model.generation_config.return_dict_in_generate=True # set to true otherwise hidden_states are not returned
                    outputs = model.generate(input_features=batch["input_features"],output_hidden_states=True,return_dict_in_generate=True)
        
                    #logits = outputs.logits
                    #prediction_ids = torch.argmax(logits, dim=-1) ssimple forward pass not sufficient
                    predictions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                    labels = processor.batch_decode(batch["labels"], skip_special_tokens=True)
                    #hidden_states = outputs.encoder_last_layer_hidden_state
                    prediction_sentences.append(predictions)
                    labels_list.append(labels)
                    #hidden_states = outputs.hidden_states
                #results.append(outputs)
    #trainer.model.generation_config.cache_implementation = "static"
    #trainer.model.forward = torch.compile(trainer.model.forward, mode="reduce-overhead", fullgraph=True)
    #breakpoint()
    #result2 = trainer.predict(test_dataset)
    end_time_transcription = time.perf_counter()
    inference_time = end_time_transcription-start_time_transcription
    print(inference_time)
    breakpoint()
    predictions = result2.predictions
    labels = result2.label_ids
    decode = lambda data: [tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in data]
    decoded_sentences = decode(predictions)
    decoded_labels = decode(labels)
    df = create_polars_df(decoded_sentences,decoded_labels)
    return df



def create_polars_df(decoded_sentences:list,decoded_labels:list)->pl.DataFrame:
    df = pl.DataFrame({'predictions': decoded_sentences, 'labels': decoded_labels})
    return df
    



def save_evaluation_results_as_csv(run_details:RunDetails, results:pl.DataFrame) -> str:
    #ID 173
    results_directory = str( f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}" )
    test_df = pd.read_csv( "shuffled_test_dataframe.csv" )
    assert results.shape[0] == test_df.shape[0]
    test_df['labels_trained'] = results['labels']
    test_df['temp'] = results['predictions']
    check_no_missing_values( test_df, results )

    test_df['results'] = test_df.apply(
        lambda row: add_prediction_column( row['words'], row['labels_trained'], row['temp'] ), axis=1 )
    test_df.drop( columns=['labels_trained'] )
    model_size = get_model_size( run_details.model_id )
    trained_path = f'{run_details.dataset_name}_eval_{model_size}_trained.csv'
    test_df.to_csv( trained_path, index=False )
    return trained_path


class SavePeftModelCallback( TrainerCallback ):

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
            ):
        checkpoint_folder = os.path.join( args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}" )

        peft_model_path = os.path.join( checkpoint_folder, "adapter_model" )
        kwargs["model"].save_pretrained( peft_model_path )

        pytorch_model_path = os.path.join( checkpoint_folder, "pytorch_model.bin" )
        if os.path.exists( pytorch_model_path ):
            os.remove( pytorch_model_path )
        return control

def get_trainer(run_details:RunDetails, training_args:dict, data_collator,train_dataset:Dataset, eval_dataset:Dataset,model, processor)->Seq2SeqTrainer:
    #ID 170
    if run_details.version == "peft":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,

         
            )
        # silence the warnings. Please re-enable for inference!
    else:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics = compute_metrics,
            tokenizer=processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
    return trainer
# significantly faster than pandas dataframe








if __name__ == "__main__":
    app.run(main)
