import jiwer
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import ast

from preprocessing import get_formated_date


def plot_loss(trainer):
    from Whisper import dataset_name, model_id,version
    from preprocessing import get_formated_date
    df_log = pd.DataFrame(trainer.state.log_history)
    # visualization of the loss during training
    (df_log.dropna(subset=["eval_loss"]).reset_index()["eval_loss"].plot(label="Validation"))
    df_log.dropna(subset=["loss"]).reset_index()["loss"].plot(label="Train")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")

    filepath = f'Figures/Training/LOSS/{dataset_name}/{model_id}/{version}/{get_formated_date()}'
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
    plt.show()


# looking at the results from the individual sessions
import re


def extract_session(file_path):
    match = re.search(r'/S(\d+)', file_path)
    if match:
        return int(match.group(1))
    else:
        return None


# the microphones on person vs not on person
def extract_person(file_path):
    match = re.search(r'/S(\d+)_([PU])(\d+)', file_path)
    if match:
        return str(match.group(2))
    else:
        return None


# the people
def extract_location(file_path):
    match = re.search(r'/S(\d+)_([PU])(\d+)', file_path)
    if match:
        return str(match.group(3))
    else:
        return None


def print_wer(grouped, type):
    for name, group in grouped:
        wer = jiwer.wer(list(group["chime_ref"]), list(group["chime_hyp"]))
        print(f"{type} {name}")
        print(f"wer {wer}")


