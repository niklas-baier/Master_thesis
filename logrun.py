from pathlib import Path

import meeteval.wer.wer.siso
import pandas as pd
from jiwer import cer

from latex import create_latex_table, save_latex_csv
from preprocessing import get_formated_date


def log_run(run_details, run_results):
    results_path = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}/results.json")
    results = pd.read_json(results_path)
    #TODO
    results['wer'] = results.apply(lambda row: meeteval.wer.wer.siso.siso_word_error_rate(row['predictions'], row['labels']), axis=1)
    results['only'] = results.apply(lambda row: row['wer'].error_rate, axis=1)
    run_average_wer = results['only'].mean()
    results['cer'] = results.apply(lambda row: cer(reference=row['predictions'], hypothesis=row['labels']), axis=1)
    run_average_cer = results['cer'].mean()
    filepath = "run_logs.csv"
    commit_hash, commit_branch = log_current_commit()
    if(Path(filepath).is_file()):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=["model_name", "dataset", "date", "Training_version", "environment", "developer_mode", "wer", "cer","results_path","checkpoint-path", "notes"])

    new_row = {
        "model_name": run_details.model_id,
        "dataset": run_details.dataset_name,
        "date": run_details.date,  # Current date
        "Training_version": run_details.version,
        "environment": run_details.environment,
        "developer_mode": run_details.developer_mode,
        "wer": run_average_wer,
        "cer": run_average_cer,
        "results_path": results_path,
        "notes": "Initial run with custom dataset",
        "checkpoint-path": run_details.checkpoint_path,
        "Training" : run_details.train_state,
        "commit_hash": commit_hash,
        "dataset_evaluation_part": run_details.dataset_evaluation_part,
        "wer_per_mic_type": run_results.wer_per_mictype,
        "wer_per_session": run_results.wer_per_session ,
        "wer_per_special_token": run_results.wer_per_special_token ,
        "wer_per_mic": run_results.wer_per_mic,
        # wer far field microphones
        # wer close microphones
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(filepath, index=False)
    save_latex_csv(df)
    create_latex_table( df, "str" )


import subprocess


def log_current_commit():
    try:
        commit_message = subprocess.check_output( ["git", "log", "-1", "--pretty=%B"] ).strip().decode( "utf-8" )
        commit_hash = subprocess.check_output( ["git", "rev-parse", "HEAD"] ).strip().decode( "utf-8" )



        print( f"Current commit hash: {commit_hash}" )
        print( f"Current commit message: {commit_message}" )
        return commit_hash, commit_message

    except subprocess.CalledProcessError as e:
        print( f"An error occurred while logging the git information: {e}" )




