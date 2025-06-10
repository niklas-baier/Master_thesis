import evaluate
import einops
from discriminator import setup_models, train_adversarial
from tqdm.auto import tqdm
from absl import flags, app
import numpy as np
import json
import glob
import re
import polars as pl
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import jiwer
import evaluation
from test_Whisper import check_no_missing_values
import preprocessing
from datasets import Dataset
import pandas as pd
from logrun import log_run
from test_Whisper import run_details_valid
from visualizations import visualize_results, plot_loss, plot_WER, plot_tsne, plot_validation_wer
from train import RunDetails,WhisperSeq2SeqTrainer, add_prediction_column, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, DataCollatorSpeechClassification, get_model_size, transcribe_raw, create_tokenizer_model_processor, generate_datasets, get_cached_components
from config import get_parser
import os
import torch
from torch import optim
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, WhisperForConditionalGeneration, EarlyStoppingCallback, Trainer
from torch.utils.data import DataLoader, Subset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction
import copy
import wandb
import torchaudio
from typing import Optional
from contrastive import train_infonce, train_improved_contrastive_aligned
from transcribe import transcribe_results, transcribe_helper, validate_results, predict_logits_and_get_strings_from_them, predict, get_hidden_states, flatten_list_once, create_polars_df, save_evaluation_results_as_csv
def main(argv):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the current working directory to the directory where whisper_main.py is located
    os.chdir(script_dir)
    wandb.login(key ='37305846834e634f3640e818c42a90f5b26de39a')
    wandb.init()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['WANDB_PROJECT'] = 'WHISPER'
    os.environ['WAND_LOG_MODEL'] = 'true'
    torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
    parser = get_parser()
    args = parser.parse_args()
    formated_date = preprocessing.get_formated_date()
    def generate_rundetails(args):
        run_details = RunDetails(precision = args.precision, dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,developer_mode=args.developer_mode, augmentation=args.augmentation, run_notes=args.run_notes, additional_tokens=args.additional_tokens, dataset_evaluation_part=args.dataset_evaluation_part,oversampling = args.oversampling_clean_data, checkpoint_path=args.checkpoint, data_portion=args.data_portion, beamforming=args.beamforming, SWAD=args.SWAD, diffusion=args.diffusion)
        assert run_details_valid(run_details )
        return run_details
    run_details= generate_rundetails(args)
    def generate_eval_df(args ,run_details):

        args_copy = args

        copy = RunDetails(precision = args.precision, dataset_name=args.dataset_name, model_id=args.model_id, environment=args.environment,train_state=args.train_state, date=formated_date, version=args.version, device=args.device, task=args.task,developer_mode=args.developer_mode, augmentation=args.augmentation, run_notes=args.run_notes, additional_tokens=args.additional_tokens, dataset_evaluation_part="eval",oversampling = args.oversampling_clean_data, checkpoint_path=args.checkpoint, data_portion=args.data_portion, beamforming=args.beamforming, SWAD=args.SWAD, diffusion=args.diffusion)
        args_copy.dataset_evaluation_part = "eval"
        features = preprocessing.generate_features(run_details)
        expanded_df, dev_df, eval_df = preprocessing.generate_dfs(args=args_copy, run_details=copy)
        expanded_df['words'] = expanded_df['words'].apply(evaluation.chime_normalisation)
        dev_df['words'] = dev_df['words'].apply(evaluation.chime_normalisation)
        eval_df = pd.concat([expanded_df, dev_df, eval_df])

        eval_df['results'] = eval_df['words']
        breakpoint()
        return eval_df


    def setup(run_details,args):

        features = preprocessing.generate_features(run_details)
        expanded_df, dev_df, eval_df = preprocessing.generate_dfs(args=args, run_details=run_details)
        expanded_df['words'] = expanded_df['words'].apply(evaluation.chime_normalisation)
        dev_df['words'] = dev_df['words'].apply(evaluation.chime_normalisation)
        #expanded_df.to_csv('expanded_df.csv', index=False) the same across multiple iterations

        #generate_audio_only(all_df)
        tokenizer, model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
        if run_details.dataset_evaluation_part == 'eval':
            all = [expanded_df, dev_df]
            breakpoint()
            expanded_df = pd.concat(all).reset_index(drop=True)
            eval_df['words'] = eval_df['words'].apply(evaluation.chime_normalisation)
            eval_df = eval_df.drop(columns = ['results'])
            dev_df = eval_df
            eval_df = generate_eval_df(args, run_details)




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
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
        processor.save_pretrained(training_args.output_dir)
        return trainer, train_dataset, eval_dataset, test_dataset
    trainer, train_dataset, eval_dataset, test_dataset = setup(run_details=run_details,args=args)
    breakpoint()
    #plot_tsne(model=model, processor=processor, test_dataset=test_dataset, torch_dtype=torch_dtype, run_details=run_details)
    def transcribe_visualize_log_results(test_dataset,trainer, run_details):
        transcription_csv_path_trained = transcribe_results( test_dataset=test_dataset, trainer=trainer,run_details=run_details )
        run_results = visualize_results(transcription_csv_path_trained, run_details)
        log_run( run_details=run_details, run_results=run_results, results_path=transcription_csv_path_trained )
    if run_details.train_state == 'NT':
        #TODO
        transcribe_visualize_log_results( test_dataset=test_dataset, trainer=trainer,run_details=run_details )
    else:
        if run_details.run_notes == 'contrastive':
            from evaluation import calculate_wer_on_dataset
            #trainer = get_trainer(run_details=run_details, training_args=training_args, data_collator= data_collator,train_dataset=train_dataset,eval_dataset=eval_dataset, model=model, processor=processor )
            BATCH_SIZE = 16 # Keep relatively small for demonstration; ensure > 1               # Ensure dataloader_A and dataloader_B use the SAME batch size
            if run_details.environment == 'bwcluster':
                BATCH_SIZE = 128
            NUM_EPOCHS = 20
            LEARNING_RATE = 5e-5 # Standard fine-tuning LR for Whisper can work
            WEIGHT_DECAY = 0.01
            INFONCE_WEIGHT = 0.1 # Weight for the contrastive loss term
            TEMPERATURE = 0.07 # Common temperature value for InfoNCE
            device = "cuda"
            _,model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
            collator = DataCollatorSpeechSeq2SeqWithPadding(processor,model.config.decoder_start_token_id )
            clean_dataloader= DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
            dirty_dataloader= DataLoader(train_dataset[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
            start_time = time.perf_counter()
            print("here")
            wandb.run.tags = wandb.run.tags + ("contrastive", "shuffled", "large batchsize")
            #wer = calculate_wer_on_dataset(dataset=train_dataset[0], model=model, processor=processor, device=device,run_details=run_details)
            contrastive_model = train_improved_contrastive_aligned(
                whisper_model=model,
                processor=processor,
                collator=collator,
                train_datasets=train_dataset,
                eval_dataset=eval_dataset,
                device=device,
                use_multi_positive=True,  # Use all far-field mics as positives
                temperature=0.07,         # Lower temp = harder negatives
                contrastive_weight=0.3,
                num_epochs = NUM_EPOCHS    # Balance ASR + contrastive loss
            )
            #contrastive_model = train_infonce(whisper_model = model, processor=processor, train_dataset=train_dataset,eval_dataset=eval_dataset,device="cuda", num_epochs=NUM_EPOCHS,BATCH_SIZE=BATCH_SIZE, lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,infonce_weight=INFONCE_WEIGHT, temperature=TEMPERATURE, collator = collator, trainer=trainer, run_details=run_details)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            args_copy = args
            args_copy.run_notes = 'ntxent evaluation'
            args_copy.train_state= 'NT'
            run_details = generate_rundetails(args_copy)
            #trainer, train_dataset, eval_dataset, test_dataset = setup(run_details=run_details, args=args)
            trainer.model = contrastive_model
            del train_dataset
            transcribe_visualize_log_results(test_dataset=test_dataset, trainer=trainer, run_details=run_details)

            transcribe_visualize_log_results(test_dataset= train_dataset[0], trainer=trainer, run_details=run_details)
            transcribe_visualize_log_results(test_dataset= train_dataset[1], trainer=trainer, run_details=run_details)


        elif run_details.run_notes == 'GAN':
            LAMBDA_DOMAIN = 0.1
            NUM_EPOCHS =20
            BATCH_SIZE = 1 # Adjust based on GPU memory
            NUM_EPOCHS = 5
            LEARNING_RATE = 1e-5
            WEIGHT_DECAY = 0.01
            LAMBDA_DOMAIN = 0.1 # How much to weight the adversarial loss
            _,model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
            collator = DataCollatorSpeechSeq2SeqWithPadding(processor,model.config.decoder_start_token_id )
            warmup_collator= DataCollatorSpeechClassification(processor, model.config.decoder_start_token_id)
            clean_dataloader= DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
            dirty_dataloader= DataLoader(train_dataset[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
            whisper_model, discriminator, grl, device = setup_models(run_details.model_id)
            from discriminator import train_adversarial
            train_adversarial(trainer, discriminator, grl,eval_dataset=eval_dataset, train_datasets = train_dataset, test_dataset=test_dataset, device=run_details.device,num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,lambda_domain_loss=LAMBDA_DOMAIN, BATCH_SIZE= BATCH_SIZE, collator=collator, run_details=run_details)
        else:
            tokenizer,model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
            num_epochs = 4
            tokenizer,_,processor = get_cached_components()
            hidden_states = []
            labels  = []
            predictions = []
            collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
            test_dataloader= DataLoader(test_dataset, batch_size=16, collate_fn=collator, num_workers=2 )
            scaler = torch.amp.GradScaler("cuda")
            if run_details.SWAD == 'Y':
                optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, momentum = 0.9)
                #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, mx_lr = 0.1)
                #from torchcontrib.optim import SWA
                #swad = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=5,swa_lr = 0.05)

                swa_model = torch.optim.swa_utils.AveragedModel(model)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
                swa_start = 5
                swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)

                for epoch in range(10):
                    model.to("cuda")
                    for batch in tqdm(test_dataloader, desc="training SWAD"):
                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            outputs = model(input_features=batch['input_features'].to("cuda"), labels = batch['labels'].to("cuda"))
                            loss = outputs.loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if epoch > swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                # Update bn statistics for the swa_model at the end
                torch.optim.swa_utils.update_bn(test_dataloader, swa_model)

                # Use swa_model to make predictions on test data
                model = swa_model
            else:
                trainer.evaluation_strategy="no"
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model = trainer.model
                model.model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
                trainer.model = model
                start_time = time.perf_counter()
                wers = []
                min_wer = 5
                counter_since_last_min = 0
                path_of_best_model = f'min_training.pth'
                num_epochs = 20
                clean_dataloader= DataLoader(train_dataset, batch_size=8, collate_fn=collator, num_workers=2 )
                original_peft_model_forward = model.forward # Store original reference

                def whisper_peft_forward_wrapper(
                    self, # 'self' refers to the PeftModelForSeq2SeqLM instance
                    *args,
                    **kwargs
                ):
                    #print("DEBUG: whisper_peft_forward_wrapper called!")
                    #print(f"DEBUG: wrapper received args: {len(args)}")
                    #print(f"DEBUG: wrapper received kwargs keys: {list(kwargs.keys())}")

                    # Remove input_ids from kwargs if present
                    if 'input_ids' in kwargs:
                        #print("DEBUG: Removing input_ids from kwargs")
                        del kwargs['input_ids']

                    # Store original base model forward
                    base_model = self.base_model
                    original_base_forward = base_model.forward

                    def patched_whisper_forward(**base_kwargs):
                        #print("DEBUG: patched_whisper_forward called!")
                        #print(f"DEBUG: Received kwargs: {list(base_kwargs.keys())}")

                        # Filter kwargs to only include parameters that Whisper actually accepts
                        whisper_params = {
                            'input_features', 'attention_mask', 'decoder_input_ids',
                            'decoder_attention_mask', 'head_mask', 'decoder_head_mask',
                            'cross_attn_head_mask', 'encoder_outputs', 'past_key_values',
                            'decoder_inputs_embeds', 'decoder_position_ids', 'labels',
                            'use_cache', 'output_attentions', 'output_hidden_states',
                            'return_dict', 'cache_position'
                        }

                        filtered_kwargs = {}
                        for key, value in base_kwargs.items():
                            if key in whisper_params:
                                filtered_kwargs[key] = value
                            else:
                                pass

                            #print(f"DEBUG: Filtering out unsupported parameter: {key}")


                        #print(f"DEBUG: Calling base Whisper with filtered params: {list(filtered_kwargs.keys())}")
                        return original_base_forward(**filtered_kwargs)

                    # Temporarily patch the base model
                    base_model.forward = patched_whisper_forward

                    try:
                        #print(f"DEBUG: Calling original PEFT forward with kwargs: {list(kwargs.keys())}")
                        result = original_peft_model_forward(self, *args, **kwargs)
                        #print(f"DEBUG: Got result type: {type(result)}")
                        return result
                    finally:
                        # Always restore the original forward method
                        base_model.forward = original_base_forward

                # Proper method binding
                import types
                model.forward = types.MethodType(whisper_peft_forward_wrapper, model)

                print(f"DEBUG: Model forward method is now: {model.forward}")
                print(f"DEBUG: Model type: {type(model)}")
                breakpoint()
                print("Using WhisperTraineri in training?", isinstance(trainer, WhisperSeq2SeqTrainer))
                for i in range(num_epochs):
                    print(i)
                    trainer.args.max_steps = trainer.train_dataset.shape[0]//trainer.args.per_device_train_batch_size
                    print("Sample from train_dataset:", train_dataset[0].keys())
                    print(trainer.args.max_steps)

                    trainer.train()
                    validation_results = validate_results(trainer=trainer, test_dataset=trainer.eval_dataset, run_details=run_details)
                    torch.cuda.empty_cache()
                    test=validation_results.with_columns(pl.col(["predictions","labels"]).map_elements(evaluation.chime_normalisation))
                    df = test.with_columns(pl.struct(["predictions", "labels"]).map_elements(lambda x: jiwer.wer(x["labels"], x["predictions"])).alias("wer"))
                    mean_wer = df['wer'].mean()
                    wandb.log({"validation_wer" : mean_wer})
                    print(mean_wer)
                    print(type(mean_wer))
                    wers.append(mean_wer)
                    if(mean_wer < min_wer):
                        torch.save(trainer.model.state_dict(), path_of_best_model)
                        counter_since_last_min = 0
                        torch.cuda.empty_cache()
                    else:
                        counter_since_last_min +=1
                    if counter_since_last_min >4:
                        break

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                best_dict = torch.load(path_of_best_model)
                best_model = trainer.model
                best_model.load_state_dict(best_dict)
                trainer.model = best_model

        #model.push_to_hub(peft_model_id)
        transcription_csv_path_trained = transcribe_results( test_dataset=test_dataset, trainer=trainer,
                                                            run_details=run_details )
        run_results = visualize_results( transcription_csv_path_trained, run_details )

        #plot_loss(trainer, run_details=run_details)
        #plot_WER( trainer, run_details=run_details )
        log_run(run_details=run_details, run_results=run_results,training_time=elapsed_time, results_path=transcription_csv_path_trained)

        #TODO take it from the mode
    return

def generate_audio_only(all_df):
    for i in range(all_df.shape[0]):
        base_path = '/pfs/work7/workspace/scratch/uhicv-blah/just_the_data/'
        dataset_path = os.path.join(base_path,run_details.dataset_name)
        target_directory = os.path.join(dataset_path, run_details.dataset_evaluation_part)
        filepath = all_df.iloc[i]['file_path']
        start_frame = all_df.iloc[i]['startframe']
        num_frames = all_df.iloc[i]['num_frames']
        wav,sr = torchaudio.load(filepath,frame_offset=start_frame,num_frames=num_frames)
        file_prefix = os.path.basename(filepath)[:4]
        target_path = os.path.join(target_directory, f'{file_prefix}{i}.wav')
        torchaudio.save(target_path, wav, sr)
        if i == 1:
            alldf_path = os.path.join(target_directory,'df.csv')
            #all_df.to_csv(alldf_path)
            old_df = pd.read_csv(alldf_path)
            for row_num in tqdm(range(all_df.shape[0])):
                if(all_df['file_path'][row_num] != old_df['file_path'][row_num]):
                    print('not same filepath')
                if(all_df['num_frames'][row_num] != old_df['num_frames'][row_num]):
                    print('not same num_frames')
                if(all_df['startframe'][row_num] != old_df['startframe'][row_num]):
                    print('not same startframe')
                if(all_df['words'][row_num] != old_df['words'][row_num]):
                    print('not same')
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
def get_trainer(run_details: RunDetails, training_args: dict, data_collator, train_dataset: Dataset, eval_dataset: Dataset, model, processor) -> Seq2SeqTrainer:
    from train import WhisperSeq2SeqTrainer
    #TrainerClass = WhisperSeq2SeqTrainer if run_details.version == "peft" else Seq2SeqTrainer
    TrainerClass = Seq2SeqTrainer
    print(type(TrainerClass)) # yields <class 'type'> is this normal ?
    trainer = TrainerClass(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # Still okay to keep for logging etc.
    )
    print("Using WhisperTrainer?", isinstance(trainer, WhisperSeq2SeqTrainer))
    return trainer



def generate_snippets(eval_df):
    import pandas as pd
    import torchaudio
    import os

    # Assuming eval_df is your DataFrame
    # Group the DataFrame by 'file_path'
    grouped_by_path = eval_df.groupby('file_path')

    # Directory where you want to save the snippets
    output_dir = '/media/niklas/SSD2/ind_beamforming'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each group
    for file_path, group in grouped_by_path:
        # Load the audio file
        audio_data, sample_rate = torchaudio.load(file_path)

        # Iterate over each row in the group
        for index, row in group.iterrows():
            start_frame = row['startframe']
            num_frames = row['num_frames']

            # Extract the snippet
            snippet = audio_data[:, start_frame:start_frame + num_frames]

            # Define the output file name
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{file_name}_{index}.wav")

            # Save the snippet
            torchaudio.save(output_file, snippet, sample_rate)

    print("Audio snippets have been saved successfully.")




if __name__ == "__main__":
    app.run(main)
