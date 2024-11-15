import warnings
from functools import wraps
import time




def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper



# Define a decorator to suppress specific warnings
def suppress_specific_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # Suppress FutureWarning
            return func(*args, **kwargs)
    return wrapper


def run_details_valid(run_details):
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
                                            if run_details.augmentation == ['Y']:
                                                assert run_details.train_state == "T", "augmentation only in training mode valid"
                                            return True

    return False

def get_dict_of_acceptable_run_details():
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


def check_no_missing_values(test_df, results):
    words = test_df['words'].tolist()
    labels = list(results['labels'])
    labels = list(results['labels'])
    comparison = [col == word for col, word in zip(labels, words)]
    all_match = all(comparison)
    assert(all_match == True)
