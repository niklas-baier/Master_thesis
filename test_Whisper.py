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
    if run_details.train_state in ["T","NT"]:
        print(f"{run_details.train_state} as train_state valid")
        if run_details.version in  ["vanilla","peft","last-layer"]:
            print(f"{run_details.version} as version valid")
            if run_details.task in ["classification","joint","transcribe"]:
                print(f"{run_details.task} as task valid")
                if run_details.dataset_name in ["Chime6", "dipco"]:
                    print(f"{run_details.dataset_name} as dataset_name valid")
                    if run_details.environment in ["laptop","cluster","bwcluster"]:
                        print(f"{run_details.environment} as environment valid")
                        if run_details.device in ["cuda","cpu"]:
                            print(f"{run_details.device} as device valid")
                            if run_details.model_id in ['openai/whisper-tiny.en','openai/whisper-tiny','openai/whisper-small','openai/whisper-medium','openai/whisper-large',"openai/whisper-large-v2","openai/whisper-large-v3","distil-whisper/distil-large-v3" ]:
                                print(f"{run_details.model_id} as model_name valid")
                                if run_details.developer_mode in ['Y','N']:
                                    print(f"{run_details.developer_mode} as development_mode valid")
                                    return True

    return False


import sys
import traceback

# Save the original print function
original_print = print
# print = custom_print
def custom_print(*args, **kwargs):
    # Print the stack trace
    traceback.print_stack(limit=5, file=sys.stdout)
    # Call the original print function
    original_print(*args, **kwargs)

# Override the print function



