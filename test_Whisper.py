import warnings
import time
from train import RunDetails
from typing import Union
import pandas as pd 
import unittest
import warnings
import gc
import torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    import traceback
    traceback.print_stack()
    print(f"{filename}:{lineno}: {category.__name__}: {message}")



def get_tensor_gpu_memory():
    objects = gc.get_objects()
    gpu_tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]
    total_memory_usage = sum(tensor.numel() * tensor.element_size() for tensor in gpu_tensors)
    total_memory_usage_mb = total_memory_usage / (1024 ** 2)
    return total_memory_usage_mb
def run_details_valid(run_details:RunDetails) -> bool:
    valid_run_details = get_dict_of_acceptable_run_details()
    if run_details.train_state in valid_run_details["train_state"]:
        print(f"{run_details.train_state} as train_state valid")
        if run_details.version in  valid_run_details["version"]:
            print(f"{run_details.version} as version valid")
            if run_details.task in valid_run_details["task"]:
                print(f"{run_details.task} as task valid")
                if run_details.dataset_name in valid_run_details["dataset_name"]:
                    print(f"{run_details.dataset_name} as dataset_name valid")
                    if run_details.environment in valid_run_details["environment"]:
                        print(f"{run_details.environment} as environment valid")
                        if run_details.device in ["cuda","cpu"]:
                            print(f"{run_details.device} as device valid")
                            if run_details.model_id in valid_run_details["model_id"]:
                                print(f"{run_details.model_id} as model_name valid")
                                if run_details.developer_mode in ['Y','N']:
                                    print(f"{run_details.developer_mode} as development_mode valid")
                                    if run_details.augmentation in valid_run_details["augmentation"]:
                                        print(f"{run_details.augmentation} as augmentation mode valid")
                                        if run_details.dataset_evaluation_part in valid_run_details["dataset_evaluation_part"]:
                                            print(f"{run_details.dataset_evaluation_part} as dataset_evaluation valid")
                                            if run_details.beamforming == "Y":
                                                assert(run_details.data_portion == "far-only")
                                            if run_details.augmentation == ['Y']:
                                                assert run_details.train_state == "T", "augmentation only in training mode valid"
                                            return True

    return False

def get_dict_of_acceptable_run_details() -> dict:
    return {
        "train_state": ["T","NT"],
        "version": ["vanilla","peft","last-layer"],
        "task": ["classification","joint","transcribe"],
        "dataset_name": ["Chime6", "dipco"],
        "environment": ["laptop","cluster","bwcluster"],
        "device": ["cuda","cpu"],
        "model_id": ['openai/whisper-tiny.en','openai/whisper-tiny','openai/whisper-small','openai/whisper-medium','openai/whisper-large',"openai/whisper-large-v2","openai/whisper-large-v3","distil-whisper/distil-large-v3"],
        "developer_mode": ['Y','N'],
        "augmentation": ['Y', 'N'],
        "dataset_evaluation_part": ["eval", "dev"]
    }

def dipco_only_planned_special_tokens(expanded_df,eval_df):
    from preprocessing import extract_special_token
    pattern = r'\[\w+\]'

    # should contain noise unintelligible and laugh


    expanded_df['token'] = expanded_df.apply(lambda row: extract_special_token(row['words']), axis=1)
    eval_df['token'] = eval_df.apply(lambda row: extract_special_token(row['words']), axis=1)
    grouped_train = expanded_df.groupby(['token'])
    allowed_tokens = ["[noise]", "[laugh]", "[unintelligible]","No token", "[laughs]"]# laughs is the token in chime instead of dipco laugh
    train_all_in_array = expanded_df['token'].isin(allowed_tokens).all()
    eval_all_in_array = eval_df['token'].isin(allowed_tokens).all()
    return (train_all_in_array and eval_all_in_array)


def check_no_missing_values(test_df:pd.DataFrame, results:pd.DataFrame) -> None :
    words = test_df['words'].tolist()
    labels = list(results['labels'])
    labels = list(results['labels'])
    comparison = [col == word for col, word in zip(labels, words)]
    all_match = all(comparison)
    assert(all_match == True)


'''class Test_dfs(unittest.TestCase):
    
def check_dataframe_for_correct_modifications():
    import polars as pl
    pl = '''
