import jiwer
import librosa
import meeteval
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import ast
import torch
from transformers import pipeline

import wandb
import numpy as np

from augmentations import filter_p_audio, add_file_name
from evaluation import  analysis_special_tokens
from preprocessing import get_formated_date
from train import RunResults, transcribe_audio



def plot_loss(trainer, run_details):
    from preprocessing import get_formated_date
    df_log = pd.DataFrame( trainer.state.log_history )
    # visualization of the loss during training
    df_log.dropna( subset=["eval_loss"] ).reset_index()["eval_loss"].plot( label="Validation" )
    df_log.dropna( subset=["loss"] ).reset_index()["loss"].plot( label="Train" )
    plt.xlabel( "Epochs" )
    plt.legend( loc="upper right" )

    filepath = f'Figures/Training/LOSS/{run_details.dataset_name}/{run_details.model_id}/{run_details.version}/{get_formated_date()}'
    try:
        os.makedirs( filepath )
    except FileExistsError:
        print( "Directory already exists" )
    finally:
        plt.savefig( filepath + "1", format='png' )


def plot_WER(trainer, run_details):
    # print evaluation of WER over training
    df_log = pd.DataFrame( trainer.state.log_history )
    # visualization of the loss during training
    (df_log.dropna( subset=["eval_wer"] ).reset_index()["eval_wer"].plot( label="WER" ))
    plt.xlabel( "Epochs" )
    plt.ylabel( "WER" )
    plt.legend( loc="upper right" )

    min_eval_wer = df_log['eval_wer'].min()

    def format_wer(wer):
        wer_str = f"{wer:.3f}"  # Format the WER to three decimal places
        return wer_str.replace( ".", "_" )

    min_eval_wer_str = format_wer( min_eval_wer )
    filepath = f'Figures/Training/WER/{run_details.dataset_name}/{min_eval_wer_str}/{run_details.model_id}/{run_details.version}/{get_formated_date()}'

    try:
        os.makedirs( filepath )
    except FileExistsError:
        print( "Directory already exists" )
    finally:
        plt.savefig( f'{filepath}/test.png', format='png' )


def visualize_wer(grouped, model_type):
    names = []
    wers = []
    for name, group in grouped:

        wer = jiwer.wer( list( group["chime_ref"] ), list( group["chime_hyp"] ) )

        # Regular expression to check if the string is a tuple representation
        tuple_pattern = r"\(\s*'[^']*'\s*,\s*'[^']*'\s*\)"

        # Check if the input string matches the tuple pattern
        if re.fullmatch( tuple_pattern, str( name ) ):
            # Safely evaluate the string to get the tuple
            parsed_tuple = ast.literal_eval( str( name ) )

            # Concatenate the elements and convert to lowercase
            result_str = f"{parsed_tuple[0].lower()}{parsed_tuple[1]}"
            names.append( result_str )

        else:
            names.append( str( name ) )

        wers.append( wer )

    plt.figure( figsize=(8, 6) )
    plt.bar( names, wers )
    plt.ylabel( f'Mean average WER per {model_type[0]}' )
    model_name = model_type[2].rsplit( '/', 1 )[-1]
    plt.title( f'WER of {model_name} on the {(dataset_name := (model_type[1]))} dataset' )

    plt.savefig( f'Figures/{(partition_type := (model_type[0]))} bar_plot.png', format='png' )
    wandb.log( {f"{dataset_name}_{model_name}": wandb.Image( plt )} )
    dict_of_wers = {k: v for k, v in zip( names, wers )}
    return dict_of_wers


def extract_info(file_path, pattern, group_idx):
    match = re.search( pattern, file_path )
    if match:
        return match.group( group_idx )
    return None


def extract_session(file_path):
    return extract_info( file_path, r'/S(\d+)', 1 )


def extract_person(file_path):
    return extract_info( file_path, r'/S(\d+)_([PU])(\d+)', 2 )


def extract_location(file_path):
    return extract_info( file_path, r'/S(\d+)_([PU])(\d+)', 3 )


def print_wer(grouped, type):
    for name, group in grouped:
        wer = jiwer.wer( list( group["chime_ref"] ), list( group["chime_hyp"] ) )
        print( f"{type} {name}" )
        print( f"wer {wer}" )


#TODO meeteval and wandb
def plot_histograms(data, run_details):
    plt.figure( figsize=(10, 6) )
    metric = "wer"
    data['only'] = data.apply( lambda row: row[metric].error_rate, axis=1 )
    plt.hist( data['only'], bins=100, color='blue', alpha=0.7 )
    plt.title( 'Histogram of Word Error Rate (WER)' )
    plt.xlabel( 'WER' )
    plt.ylabel( 'Frequency' )
    plt.grid( True )
    hist_path = f'Figures/Training/histograms/{run_details.dataset_name}/{metric}.png'
    plt.savefig( hist_path, format='png' )
    metric = "cer"
    hist_path = f'Figures/Training/histograms/{run_details.dataset_name}/{metric}.png'
    plt.hist( data[metric], bins=100, color='yellow', alpha=0.7 )
    plt.title( 'Histogram of Character Error Rate (CER)' )
    plt.xlabel( 'CER' )
    plt.ylabel( 'Frequency' )
    plt.grid( True )
    plt.savefig( hist_path, format='png' )


def visualize_results(transcription_csv_path, run_details):
    from evaluation import chime_normalisation
    data = pd.read_csv( transcription_csv_path )
    # dataset = dataset.map(lambda example: {'normalized_ref': chime_normalisation(example['words'])})
    data['results'] = data['results'].astype( str )

    data['chime_ref'] = [chime_normalisation( text ) for text in data["words"]]
    data['chime_hyp'] = [chime_normalisation( text ) for text in data["results"]]

    wer = jiwer.wer( list( data["chime_ref"] ), list( data["chime_hyp"] ) )
    # WER of the whisper normalizer
    print( f"WER: {wer * 100:.2f} %" )

    print( data.sample( n=10 ) )
    data['wer'] = data.apply(
        lambda row: meeteval.wer.wer.siso.siso_word_error_rate(
            reference=row['chime_ref'],
            hypothesis=row['chime_hyp']
            ),
        axis=1
        )
    data['cer'] = data.apply( lambda row: jiwer.cer( reference=row['chime_ref'], hypothesis=row['chime_hyp'] ), axis=1 )

    ascii_pattern = r'^[\x00-\x7F]*$'
    # Step 3: Filter the DataFrame
    print( data.shape )
    df_ascii = data[data['chime_hyp'].str.contains( ascii_pattern, na=False )]
    print( df_ascii.shape )
    wer = jiwer.wer( list( df_ascii["chime_ref"] ), list( df_ascii["chime_hyp"] ) )

    print( f"WER: {wer * 100:.2f} %" )

    data['session_number'] = data['file_path'].apply( extract_session )
    data['mic_type'] = data['file_path'].apply( extract_person )
    data['mic_number'] = data['file_path'].apply( extract_location )
    grouped_ses = data.groupby( 'session_number' )
    print_wer( grouped_ses, "session" )
    grouped_mic_type = data.groupby( 'mic_type' )
    grouped_mic = data.groupby( ['mic_type', 'mic_number'] )
    print_wer( grouped_mic, "mic_type" )
    grouped_token = analysis_special_tokens( data )
    print( wer )

    # plot visualization of the different sessions and store the results

    directory = "Figures"

    # Create the directory if it doesn't exist
    if not os.path.exists( directory ):
        os.makedirs( directory )
        print( f"Directory '{directory}' created." )
    else:
        print( f"Directory '{directory}' already exists." )
    dict_of_special_tokens = visualize_wer( grouped_token, ["special_token", f"{run_details.dataset_name}", f"{run_details.model_id}"] )
    dict_of_wer_per_session = visualize_wer( grouped_ses, ["session", f"{run_details.dataset_name}", f"{run_details.model_id}"] )
    dict_wer_per_mic_type = visualize_wer( grouped_mic_type, ["mic_type", f"{run_details.dataset_name}", f"{run_details.model_id}"] )
    dict_wer_per_mic = visualize_wer( grouped_mic, ["mic", f"{run_details.dataset_name}", f"{run_details.model_id}"] )
    run_results = RunResults(wer_per_session=dict_of_wer_per_session, wer_per_special_token=dict_of_special_tokens, wer_per_mictype=dict_wer_per_mic_type, wer_per_mic= dict_wer_per_mic)
    plot_histograms( data, run_details=run_details )
    # TODO sort by WER and CER what percentage is close what percentage

    error_rates = data['wer'].apply( lambda x: x.error_rate )

    # Calculate the mean of the error rates
    mean_error_rate = error_rates.mean()
    print( mean_error_rate )
    return run_results


def plot_waveform(waveform, sample_rate):
    # Assume waveform is 1D (single channel)
    num_samples = waveform.shape[0]
    time_axis = np.linspace( 0, num_samples / sample_rate, num_samples )

    plt.figure()
    plt.plot( time_axis, waveform )
    plt.xlabel( 'Time [s]' )
    plt.ylabel( 'Amplitude' )
    plt.title( 'Waveform' )
    plt.show()


# Example usage

def plot_spec(ax, spec, title):
    ax.set_title( title )
    ax.imshow( librosa.amplitude_to_db( spec ), origin="lower", aspect="auto" )
    fig, axes = plt.subplots( 1, 1, sharex=True, sharey=True )
    plot_spec( axes[1], torch.abs( spec[0] ), title="Original" )
    fig.tight_layout()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots( 1, 1 )
    if title is not None:
        ax.set_title( title )
    ax.set_ylabel( ylabel )
    power_specgram = np.abs( specgram ) ** 2
    ax.imshow( librosa.power_to_db( power_specgram ), origin="lower", aspect="auto", interpolation="nearest" )


def plot_tsne(trainer, test_dataset, torch_dtype, run_details, processor):
    def custom_compute_metrics(eval_pred):
        breakpoint()
        logits, labels, hidden_states = eval_pred

        # Optionally, do something with hidden_states or logits
        last_hidden_state = hidden_states[-1]  # Extract last layer hidden state

        # Return empty or any other computed metrics
        return torch.mean( last_hidden_state )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Pass 'output_hidden_states=True' to the model's forward pass
        breakpoint()
        outputs = model( **inputs, output_hidden_states=True )

        # Outputs: (loss, logits, past_key_values, decoder_hidden_states, hidden_states, attentions, cross_attentions)
        loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]  # Get the last layer's hidden states

        # If return_outputs is True, return both loss and outputs (with hidden states)
        if return_outputs:
            return loss, (outputs.logits, last_hidden_state)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Move inputs to the correct device (GPU/CPU)
        inputs = self._prepare_inputs( inputs )

        with torch.no_grad():
            # Forward pass: get outputs from the model, including hidden states
            outputs = model( **inputs, output_hidden_states=True )

            # Get predictions (usually logits), hidden states, and labels
            loss = outputs.loss if prediction_loss_only else None
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]  # Get last hidden state

        # Return hidden states in addition to logits, inputs, and labels
        return loss, logits, hidden_states

    model = trainer.model
    trainer.compute_metrics = custom_compute_metrics
    trainer.compute_loss = compute_loss
    trainer.prediction_step = prediction_step
    model.eval()
    trainer.evaluate()
    example1 = test_dataset[0]
    tokenizer = processor.tokenizer
    # Tokenize input text
    inputs = processor( example1['input_features'], return_tensors="pt", sampling_rate=16000 )
    #batch = data_collator(inputs)
    # Forward pass through the model
    model.to( "cpu" )
    outputs, hidden = model.forward( input_features=inputs["input_features"] )
    # Extract hidden states from the model's output
    hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
    # Get the last hidden state (from the last layer)
    last_hidden_state = hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
    breakpoint()

    random_df = create_720_pairs( test_dataset[0] )
    with torch.no_grad():
        outputs = ...  #model.encoder(inputs["input_features"])

    bottleneck_features = outputs.last_hidden_state
    pooled_features = bottleneck_features.mean( dim=1 )


def create_720_pairs(eval_df):
    filtered_df = filter_p_audio( eval_df )
    # get 720 random samples

    p_samples = get_p_samples( filtered_df )
    breakpoint()
    # assert that there is a corresponding U value

    p_samples, u_samples = get_corresponding_U_values( p_samples )

    pass


def get_p_samples(filtered_df):
    if filtered_df.shape[0] < 720:
        return filtered_df.sample( n=720, random_state=42, replace=True )
    else:
        return filtered_df.sample( n=720, random_state=42 )


@add_file_name
def get_corresponding_U_values(p_samples):
    p_samples['file_name_u'] = p_samples['file_name'].apply( lambda x: convert_path( x ) )
    p_samples["file_directory"] = p_samples['file_path'].apply( lambda row: os.path.dirname( row ) )
    p_samples["noisy_path"] = p_samples.apply( lambda row: os.path.join( row['file_directory'], row['file_name'] ),
                                               axis=1 )
    p_samples['session_number'] = p_samples['file_path'].apply( extract_session )
    # set seed for reproducability
    np.random.seed( 42 )
    # generate random array
    random_U = np.random.randint( 1, 8, size=720 )
    random_channel = np.random.randint( 1, 5, size=720 )


def convert_path(p_path):
    directory, filename = ...
    modified_path = ...
    return modified_path
