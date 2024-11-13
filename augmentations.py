import os
import torch
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import preprocessing
from audioprocessing import get_spectrogram


def get_noises():
    csv_df = pd.read_csv( 'syntheticdata/ESC-50/meta/esc50.csv' )
    print( csv_df.head() )
    grouped_by_name = csv_df.groupby( 'category' )
    path_noise = csv_df['filename'].iloc[0]
    csv_df['file_path'] = csv_df['filename'].apply(get_path_noise)

    #waveform = waveform.numpy().T
    taxonomy_dict = get_noise_taxonomy()
    filtered_df = csv_df[csv_df['category'].isin(taxonomy_dict['Interior/domestic sounds'] )]
    return filtered_df
def get_path_noise(filename):
    cwd = os.getcwd()
    file_path = f'{cwd}/syntheticdata/ESC-50/audio/{filename}'
    return file_path

def create_augmentations():
    spec = get_spectrogram( power=None )
    stretch = T.TimeStretch()

    spec_12 = stretch( spec, overriding_rate=1.2 )
    spec_09 = stretch( spec, overriding_rate=0.9 )


def apply_noises(filepath_original_sound, filepath_synthetic_noise):
    speech, sr = torchaudio.load( filepath_original_sound )
    noise, syn_sr = torchaudio.load( filepath_synthetic_noise )
    assert sr == syn_sr, f"the sample rates of the speech {sr} and of the noise {syn_sr}are different and need to be resampled "
    noise = noise[:, :min( speech.shape[1], noise.shape[1] )]
    snr_dbs = torch.tensor( [20, 10, 3] )
    noise = torch.nn.functional.pad( noise, (0, speech.shape[1] - noise.shape[1]), "constant", 0 )
    noisy_speeches = F.add_noise( speech, noise, snr_dbs )
    noisy_speeches = np.where( noisy_speeches != 0, noisy_speeches, 1e-10 )
    return noisy_speeches


def get_noise_taxonomy():
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


def generate_noise_dataset(expanded_df, run_details, features):
    #TODO sollte zusammen sein wahrscheinlich
    train_dataset_path, _, _, tsne_dataset_path = preprocessing.generate_dataset_paths( run_details=run_details )
    # filter out and take only the clean samples with p in filepath
    clean_expanded_df = filter_p_audio( expanded_df=expanded_df )
    # Define the new SNR values
    snrs = [20, 3, 10]
    # Create the new expanded DataFrame by repeating rows and adding 'snrs'
    df_expanded = pd.concat( [clean_expanded_df.assign( snr=snr ) for snr in snrs], ignore_index=True )
    filepath_noise = get_noises()
    df_expanded['filepath_noise'] = filepath_noise
    train_dataset = preprocessing.Hug_dataset_creation( df_expanded, run_details.developer_mode, features,
                                                        test_dataset=False )
    train_dataset = train_dataset.map( preprocessing.prepare_noisedataset_seq2seq )
    stored_dataset_path = "noise" + train_dataset_path
    train_dataset.save_to_disk( dataset_path=stored_dataset_path )
    return stored_dataset_path


def add_file_name(func):
    def wrapper(expanded_df, *args, **kwargs):
        # Add the 'file_name' column based on 'file_path'
        expanded_df["file_name"] = expanded_df['file_path'].apply( lambda row: os.path.basename( row ) )
        # Call the original function with the modified DataFrame
        return func( expanded_df, *args, **kwargs )

    return wrapper


@add_file_name
def filter_p_audio(expanded_df):
    clean_expanded_df = expanded_df.query( "file_name.str.contains(r'P\d{2}')", engine='python' )
    clean_expanded_df.drop(columns='file_name', inplace=True)
    return clean_expanded_df
@add_file_name
def filter_far_audio(expanded_df):
    clean_expanded_df = expanded_df.query( "not file_name.str.contains(r'P\d{2}')", engine='python' )
    clean_expanded_df.drop(columns='file_name', inplace=True)
    return clean_expanded_df

