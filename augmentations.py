import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from IPython.display import Audio
import sounddevice as sd
import torchaudio.transforms as T
import torchaudio.functional as F
import preprocessing
from audioprocessing import get_spectrogram
from visualizations import plot_waveform


def get_noises():
    csv_file = pd.read_csv('syntheticdata/ESC-50/meta/esc50.csv')
    print(csv_file.head())
    grouped_by_name =  csv_file.groupby('category')
    path_noise = csv_file['filename'].iloc[0]

    print(os.getcwd())
    cwd = os.getcwd()

    filepath = f'{cwd}/syntheticdata/ESC-50/audio/{path_noise}'
    waveform,sample_rate = torchaudio.load(filepath)
    waveform = waveform.numpy().T
    metadata = torchaudio.info(filepath)
    print(metadata)

    # Play the sound using sounddevice
    return filepath

def create_augmentations():
    spec = get_spectrogram(power=None)
    stretch = T.TimeStretch()

    spec_12 = stretch(spec, overriding_rate=1.2)
    spec_09 = stretch(spec, overriding_rate=0.9)

get_noises()
def apply_noises(filepath_original_sound,filepath_synthetic_noise, snrs):
    speech, sr = torchaudio.load(filepath_original_sound, sample_rate=16000)
    noise, syn_sr = torchaudio.load(filepath_original_sound, sample_rate=16000)
    noise = noise[:, :min(speech.shape[1], noise.shape[1])]
    snr_dbs = torch.tensor([20, 10, 3])
    noise = torch.nn.functional.pad(noise, (0, speech.shape[1] - noise.shape[1]), "constant", 0)
    noisy_speeches = F.add_noise(speech, noise, snr_dbs)
    noisy_speeches = np.where(noisy_speeches != 0, noisy_speeches, 1e-10)
    return noisy_speeches



def get_noise_taxonomy(dataframe):
    sound_dict = sound_categories = {
    "Animals": [
        "dog",
        "crow",
        "cow",
        "frog",
        "cat",
        "hen",
        "pig",
        "sheep",
        "rooster",
        "insects"
    ],
    "Natural soundscapes & water sounds": [
        "chirping_birds",
        "thunderstorm",
        "water_drops",
        "wind",
        "crackling_fire",
        "crickets",
        "rain",
        "sea_waves",
        "pouring_water"
    ],
    "Human, non-speech sounds": [
        "crying_baby",
        "sneezing",
        "coughing",
        "laughing",
        "footsteps",
        "snoring",
        "clapping",
        "drinking_sipping",
        "breathing",
        "brushing_teeth"
    ],
    "Interior/domestic sounds": [
        "door_wood_knock",
        "can_opening",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "vacuum_cleaner",
        "washing_machine",
        "toilet_flush",
        "glass_breaking",
        "clock_tick",
        "clock_alarm"
    ],
    "Exterior/urban noises": [
        "helicopter",
        "chainsaw",
        "siren",
        "engine",
        "car_horn",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw"
    ]}
    return sound_dict
def generate_noise_dataset(expanded_df,run_details,features):
     train_dataset_path, _,_ = preprocessing.generate_dataset_paths(run_details=run_details)
     # filter out and take only the clean samples with p in filepath
     expanded_df["file_name"] = expanded_df['file_path'].apply(lambda row: os.path.basename(row))
     clean_expanded_df = expanded_df.query("file_name.str.contains(r'P\d{2}')", engine='python')
     # Define the new SNR values
     snrs = [20, 3, 10]

     # Create the new expanded DataFrame by repeating rows and adding 'snrs'
     df_expanded = pd.concat([clean_expanded_df.assign(snrs=snr) for snr in snrs], ignore_index=True)
     filepath_noise = get_noises()




     train_dataset = preprocessing.Hug_dataset_creation(clean_expanded_df, run_details.developer_mode, features, test_dataset=False)



     stored_dataset_path = "noise" + train_dataset_path
     return stored_dataset_path