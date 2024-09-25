import jiwer
import librosa
import meeteval
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import ast
import torch
import wandb
import numpy as np
from evaluation import chime_normalisation, analysis_special_tokens
from preprocessing import get_formated_date


def plot_loss(trainer, run_details):
    from preprocessing import get_formated_date
    df_log = pd.DataFrame(trainer.state.log_history)
    # visualization of the loss during training
    (df_log.dropna(subset=["eval_loss"]).reset_index()["eval_loss"].plot(label="Validation"))
    df_log.dropna(subset=["loss"]).reset_index()["loss"].plot(label="Train")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")

    filepath = f'Figures/Training/LOSS/{run_details.dataset_name}/{run_details.model_id}/{run_details.version}/{get_formated_date()}'
    try:
        os.makedirs(filepath)
    except FileExistsError:
        print("Directory already exists")
    finally:
        plt.savefig(filepath + "1", format='png')


def plot_WER(trainer,Run_details):
    # print evaluation of WER over training
    df_log = pd.DataFrame(trainer.state.log_history)
    # visualization of the loss during training
    (df_log.dropna(subset=["eval_wer"]).reset_index()["eval_wer"].plot(label="WER"))
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.legend(loc="upper right")

    min_eval_wer = df_log['eval_wer'].min()

    def format_wer(wer):
        wer_str = f"{wer:.3f}"  # Format the WER to three decimal places
        return wer_str.replace(".", "_")

    min_eval_wer_str = format_wer(min_eval_wer)
    filepath = f'Figures/Training/WER/{Run_details.dataset_name}/{min_eval_wer_str}/{Run_details.model_id}/{Run_details.version}/{get_formated_date()}'

    try:
        os.makedirs(filepath)
    except FileExistsError:
        print("Directory already exists")
    finally:
        plt.savefig(f'{filepath}/test.png', format='png')




def visualize_wer(grouped, type):
    names = []
    wers = []
    for name, group in grouped:

        wer = jiwer.wer(list(group["chime_ref"]), list(group["chime_hyp"]))

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
    plt.title(f'WER of {model_name} on the {(dataset_name := (type[1]))} dataset')

    plt.savefig(f'Figures/{(partition_type := (type[0]))} bar_plot.png', format='png')
    wandb.log({f"{dataset_name}_{model_name}": wandb.Image(plt)})




def extract_info(file_path, pattern, group_idx):
    match = re.search(pattern, file_path)
    if match:
        return match.group(group_idx)
    return None

def extract_session(file_path):
    return extract_info(file_path, r'/S(\d+)', 1)

def extract_person(file_path):
    return extract_info(file_path, r'/S(\d+)_([PU])(\d+)', 2)

def extract_location(file_path):
    return extract_info(file_path, r'/S(\d+)_([PU])(\d+)', 3)


def print_wer(grouped, type):
    for name, group in grouped:
        wer = jiwer.wer(list(group["chime_ref"]), list(group["chime_hyp"]))
        print(f"{type} {name}")
        print(f"wer {wer}")

#TODO meeteval and wandb
def plot_histograms(data, run_details):
    plt.figure(figsize=(10, 6))
    metric = "wer"
    data['only'] = data.apply(lambda row: row[metric].error_rate, axis=1)
    plt.hist(data['only'], bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of Word Error Rate (WER)')
    plt.xlabel('WER')
    plt.ylabel('Frequency')
    plt.grid(True)
    hist_path = f'Figures/Training/histograms/{run_details.dataset_name}/{metric}.png'
    plt.savefig(hist_path,format='png')
    metric = "cer"
    hist_path = f'Figures/Training/histograms/{run_details.dataset_name}/{metric}.png'
    plt.hist(data[metric], bins=100, color='yellow', alpha=0.7)
    plt.title('Histogram of Character Error Rate (CER)')
    plt.xlabel('CER')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(hist_path,format='png')


def visualize_results(transcription_csv_path, run_details):
    data = pd.read_csv(transcription_csv_path)
    # dataset = dataset.map(lambda example: {'normalized_ref': chime_normalisation(example['words'])})
    data['results'] = data['results'].astype(str)

    data['chime_ref'] = [chime_normalisation(text) for text in data["words"]]
    data['chime_hyp'] = [chime_normalisation(text) for text in data["results"]]

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
    data['cer'] = data.apply(lambda row: jiwer.cer(reference=row['chime_ref'], hypothesis=row['chime_hyp']), axis=1)


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
    grouped_token = analysis_special_tokens(data)
    print(wer)

    # plot visualization of the different sessions and store the results


    directory = "Figures"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    visualize_wer(grouped_token, ["special_token", f"{run_details.dataset_name}", f"{run_details.model_id}"])
    visualize_wer(grouped_ses, ["session", f"{run_details.dataset_name}", f"{run_details.model_id}"])
    visualize_wer(grouped_mic_type, ["mic_type", f"{run_details.dataset_name}", f"{run_details.model_id}"])
    visualize_wer(grouped_mic, ["mic", f"{run_details.dataset_name}", f"{run_details.model_id}"])
    plot_histograms(data,run_details=run_details)
    # TODO sort by WER and CER what percentage is close what percentage


    error_rates = data['wer'].apply(lambda x: x.error_rate)

    # Calculate the mean of the error rates
    mean_error_rate = error_rates.mean()
    print(mean_error_rate)


def plot_waveform(waveform, sample_rate):
    # Assume waveform is 1D (single channel)
    num_samples = waveform.shape[0]
    time_axis = np.linspace(0, num_samples / sample_rate, num_samples)

    plt.figure()
    plt.plot(time_axis, waveform)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

# Example usage

def plot_spec(ax, spec, title):
    ax.set_title(title)
    ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")
    fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    plot_spec(axes[1], torch.abs(spec[0]), title="Original")
    fig.tight_layout()



def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_specgram = np.abs(specgram)**2
    ax.imshow(librosa.power_to_db(power_specgram), origin="lower", aspect="auto", interpolation="nearest")