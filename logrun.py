from pathlib import Path

import meeteval.wer.wer.siso
import pandas as pd
from jiwer import cer

from latex_conversion_results import save_latex_csv
from preprocessing import get_formated_date


def log_run(run_details):
    results_path = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}/results.json")
    results = pd.read_json(results_path)
    #TODO
    results['wer'] = results.apply(lambda row: meeteval.wer.wer.siso.siso_word_error_rate(row['predictions'], row['labels']), axis=1)
    results['only'] = results.apply(lambda row: row['wer'].error_rate, axis=1)
    run_average_wer = results['only'].mean()
    results['cer'] = results.apply(lambda row: cer(reference=row['predictions'], hypothesis=row['labels']), axis=1)
    run_average_cer = results['cer'].mean()
    filepath = "run_logs.csv"
    if(Path(filepath).is_file()):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=["model_name", "dataset", "date", "Training_version", "environment", "developer_mode", "wer", "cer","results_path", "notes"])

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
        "notes": "Initial run with custom dataset"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(filepath, index=False)
    save_latex_csv(df)


