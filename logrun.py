from pathlib import Path

import meeteval.wer.wer.siso
import pandas as pd
from jiwer import cer
from train import RunDetails, RunResults
from latex import create_base_line_latex_tables, create_latex_table, save_latex_csv
from preprocessing import get_formated_date


def log_run(run_details:RunDetails, run_results:RunResults, training_time=0)-> None:
    #implementation of ID: 63
    results_path = str(f"{run_details.model_id}_{run_details.dataset_name}_{run_details.version}/results.json")
    results = pd.read_json(results_path)
    #TODO
    results['wer'] = results.apply(lambda row: meeteval.wer.wer.siso.siso_word_error_rate(row['predictions'], row['labels']), axis=1)
    results['only'] = results.apply(lambda row: row['wer'].error_rate, axis=1)
    run_average_wer = results['only'].mean()
    results['cer'] = results.apply(lambda row: cer(reference=row['predictions'], hypothesis=row['labels']), axis=1)
    run_average_cer = results['cer'].mean()
    if results.shape[0] > 100:
        boxplot_wer( results )
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
        "wer": (run_results.wer_per_mictype["P"]+run_results.wer_per_mictype["U"])/2,
        "cer": run_average_cer,
        "results_path": results_path,
        "notes": run_details.run_notes,
        "checkpoint-path": run_details.checkpoint_path,
        "Training" : run_details.train_state,
        "commit_hash": commit_hash,
        "dataset_evaluation_part": run_details.dataset_evaluation_part,
        "wer_per_mic_type": run_results.wer_per_mictype,
        "wer_per_session": run_results.wer_per_session ,
        "wer_per_special_token": run_results.wer_per_special_token ,
        "wer_per_mic": run_results.wer_per_mic,
        "oversampling": run_details.oversampling,
        "training_time": training_time,
        "data_portion in training" : run_details.data_portion,

        # wer far field microphones
        # wer close microphones
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(filepath, index=False)
    save_latex_csv(df)
    create_latex_table( df, "str" )
    create_base_line_latex_tables()


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

def boxplot_wer(data:pd.DataFrame) ->None:
    import matplotlib.pyplot as plt
    # Extract attributes from each ErrorRate object into separate columns
    data['error_rate'] = data['wer'].apply( lambda x: x.error_rate )
    data['errors'] = data['wer'].apply( lambda x: x.errors )
    data['length'] = data['wer'].apply( lambda x: x.length )
    data['insertions'] = data['wer'].apply( lambda x: x.insertions )
    data['deletions'] = data['wer'].apply( lambda x: x.deletions )
    data['substitutions'] = data['wer'].apply( lambda x: x.substitutions )

    # Optionally group by length (if you want to condition boxplots on length ranges)
    data['length_group'] = pd.cut( data['length'], bins=[0, 5, 10, 15], labels=['0-5', '6-10', '11-15'] )

    # Plot boxplots for each error component
    fig, axs = plt.subplots( 1, 5, figsize=(20, 6), sharey=True )
    components = ['error_rate', 'errors', 'insertions', 'deletions', 'substitutions']

    for i, component in enumerate( components ):
        if 'length_group' in data.columns:
            data.boxplot( column=component, by='length_group', ax=axs[i] )
            axs[i].set_title( f'{component} by length group' )
        else:
            data.boxplot( column=component, ax=axs[i] )
            axs[i].set_title( component )

    plt.suptitle( 'ErrorRate Components by Length Group' if 'length_group' in data.columns else 'ErrorRate Components' )
    plt.tight_layout()
    plt.savefig("boxplot")
    plt.show()



