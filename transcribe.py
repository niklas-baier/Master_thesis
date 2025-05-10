import polars as pl
import time
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from test_Whisper import check_no_missing_values
from datasets import Dataset
from torch.utils.data import DataLoader, Subset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, WhisperForConditionalGeneration, EarlyStoppingCallback, Trainer

from train import RunDetails, add_prediction_column, generate_training_args, DataCollatorSpeechSeq2SeqWithPadding, DataCollatorSpeechClassification, get_model_size, transcribe_raw, create_tokenizer_model_processor, generate_datasets, get_cached_components
def transcribe_results(*, test_dataset:Dataset, trainer:Seq2SeqTrainer, run_details:RunDetails) -> str:

    #ID 172

    #results = trainer.evaluate( eval_dataset=test_dataset )
    if run_details.version == "pefti":
       #model = get_peft_model(trainer, run_details)
       predictions = predict(trainer=trainer,test_dataset=test_dataset, run_details=run_details)

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



    else:

        #breakpoint()
        #segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)
        #texts = [segment.text for segment in segments]
        #print(texts[0:10])
        hidden_states,predictions = get_hidden_states(trainer=trainer, test_dataset = test_dataset, run_details=run_details)
        np.save('hidden_states_encoder.npy', hidden_states)
        from visualizations import visualize_hidden_states
        visualize_hidden_states(hidden_states=hidden_states, run_details=run_details)
        start_time_transcription= time.perf_counter()
        #predictions = predict( trainer=trainer, test_dataset=test_dataset, run_details=run_details )
        end_time_transcription = time.perf_counter()
        inference_time = end_time_transcription-start_time_transcription
        print(inference_time)

    path = save_evaluation_results_as_csv( run_details, results=predictions )
    return path



def transcribe_helper(*, test_dataset:Dataset, trainer:Seq2SeqTrainer, run_details:RunDetails) :
    if run_details.version == "peft":
        predictions = predict(trainer=trainer, test_dataset=test_dataset, run_details=run_details)

    else:

        start_time_transcription = time.perf_counter()
        predictions = predict(trainer=trainer, test_dataset = test_dataset, run_details=run_details)
        end_time_transcription  = time.perf_counter()
        inference_time = end_time_transcription - start_time_transcription
        print(inference_time)
    return predictions

def validate_results(*, test_dataset:Dataset, trainer:Seq2SeqTrainer, run_details:RunDetails) :
    return transcribe_helper(test_dataset=test_dataset,trainer=trainer, run_details=run_details)

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
def get_hidden_states(trainer:Seq2SeqTrainer, test_dataset:Dataset, run_details) :
    tokenizer,_,processor = get_cached_components()
    hidden_states = []
    labels  = []
    predictions = []
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
    test_dataloader= DataLoader(test_dataset, batch_size=16, collate_fn=collator, num_workers=2 )
    model = trainer.model.eval()
    if run_details.precision =='half':
        model = model.half()
    model.config.output_hidden_states = True # return also the hidden states#
    model.generation_config.return_dict_in_generate=True # set to true otherwise hidden_states are not returned

    for batch in tqdm(test_dataloader, desc = "tsne-visualization"):
        with torch.no_grad():
            batch.to("cuda")
            if run_details.precision == 'half':
                outputs = model.generate(input_features=batch['input_features'].half(), output_hidden_states=True)
            else:
                outputs = model.generate(input_features=batch["input_features"], output_hidden_states=True)
            hidden_states.append(torch.mean(list(outputs.encoder_hidden_states)[-1], axis=1))
            prediction = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            label = processor.batch_decode(batch['labels'],skip_special_tokens=True)
            predictions.append(prediction)
            labels.append(label)
            del outputs
            #torch.cuda.memory._dump_snapshot('hidden_states.pickle')
    hidden_states_np = []
    for tensor in hidden_states:
        np_array = tensor.detach().cpu().numpy()
        hidden_states_np.append(np_array)
    hidden_states_np = np.vstack(hidden_states_np)
    predictions_flattened = flatten_list_once(predictions)
    labels_flattened = flatten_list_once(labels)
    df = create_polars_df(predictions_flattened,labels_flattened)
    return hidden_states_np,df


def predict(trainer:Seq2SeqTrainer,test_dataset:Dataset, run_details) -> pl.DataFrame:
    tokenizer,_,processor = get_cached_components()
    import time
    start_time_transcription= time.perf_counter()
    if 1==1:
           #trainer.model.config.output_hidden_states = True
           results = []
           collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
           test_dataloader= DataLoader(test_dataset, batch_size=16, collate_fn=collator, num_workers=2 )
           prediction_sentences = []
           labels_list = []
           last_states = []
           model = trainer.model.eval()
           if run_details.precision == 'half':
               model = model.half()
           for batch in tqdm(test_dataloader, desc= "Evaluating batches"):  # Ensure you have a DataLoader for test_dataset
                with torch.no_grad():
                    batch.to("cuda")
                    #inputs_features_of_batch = processor()
                    #outputs = model(**batch, output_hidden_states=True)  # simple forward pass is not sufficient for a acceptable WER
                    #model.config.output_hidden_states = True # return also the hidden states
                    #model.generation_config.return_dict_in_generate=True # set to true otherwise hidden_states are not returned
                    if run_details.precision == 'half':
                        outputs = model.generate(input_features = batch['input_features'].half())
                    else:
                        outputs = model.generate(input_features=batch["input_features"])

                    #logits = outputs.logits
                    #prediction_ids = torch.argmax(logits, dim=-1) ssimple forward pass not sufficient
                    predictions = processor.batch_decode(outputs, skip_special_tokens=True)
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
    predictions_flattened = flatten_list_once(prediction_sentences)
    labels_flattened = flatten_list_once(labels_list)

    #predictions = result2.predictions
    #labels = result2.label_ids
    #decode = lambda data: [tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in data]
    #decoded_sentences = decode(predictions)
    #decoded_labels = decode(labels)
    #df = create_polars_df(decoded_sentences,decoded_labels)
    df = create_polars_df(predictions_flattened, labels_flattened)
    return df

def flatten_list_once(list_to_be_flattened:list)-> list:
     return  [x for xs in list_to_be_flattened for x in xs]


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
