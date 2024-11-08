import pandas as pd
def save_latex_csv(df):
    # select only the runs not on dev_mode
    df = df[df['developer_mode'] == "N"]
    # delete the dates
    df.drop(columns=['developer_mode', 'date', 'results_path', 'notes','environment', 'commit_hash'], inplace=True)
    path = "latex_results.csv"
    df.to_csv(path, index=False)
def rename_columns_for_latex(df):
    df = df.rename( columns=lambda x: x.replace( '_', r'\_' ) )
    return df


def create_latex_table(df, columns):
    # drop unnecessary_columns

    #rename columns if they are not latex_conform
    df = rename_columns_for_latex(df)
    #apply style on a specific row marking the minimum
    styled_df = df.style.apply(lambda x: ['cellcolor: [HTML]{b5ffb1} \\itshape \\bfseries' if v == x.min() else '' for v in x],subset=['wer'] )



    # save the latex table to a file
    with open( "latex_table.txt", "w" ) as file:
        file.write( styled_df.hide( axis="index" ).to_latex(
            position="h",
            position_float="centering",
            hrules=True,
            label="table:results",
            caption="Results of the baselines",
            ) )
    return


def create_dipco_baseline_latex_table(dipco_df):
    from ast import literal_eval
    breakpoint()
    dipco_df.dropna( subset=['wer_per_mic_type'], inplace=True ) # filter out the old values that are NaN
    dipco_df['wer_per_mic_type'] = dipco_df['wer_per_mic_type'].apply(literal_eval)
    baseline_table = pd.DataFrame( {
        'name of the model': dipco_df['model_name'],
        'evaluation part' :dipco_df['dataset_evaluation_part'],
        'close talk': dipco_df['wer_per_mic_type'].apply(lambda x: x['P']),
        'far field': dipco_df['wer_per_mic_type'].apply(lambda x: x['U']),
        } )
    breakpoint()
    table = baseline_table.pivot_table(
        index='evaluation part',
        values=['model_name','close talk', 'far field'],
        aggfunc='mean'
        ).round( 2 ).to_latex(
        caption='Baseline table ',
        label='tab:baseline',
        float_format="%.2f",
        index=True,
        header=True
        )
    print(table)
    # split into eval and dev part



def create_base_line_latex_tables():
    df = pd.read_csv( "run_logs.csv" )
    # baseline has no checkpoints and no training
    #df = df.query( 'Training == "NT"' ) TODO
    # only take runs that were run on the full ata
    #df = df[df['developer_mode'] == "N"] TODO
    # delete the dates
    df.drop( columns=['developer_mode', 'date', 'results_path', 'notes', 'environment', 'commit_hash'], inplace=True )
    dipco_df = df.query("dataset == 'dipco'")
    chime6_df = df.query("dataset == 'Chime6'")
    create_dipco_baseline_latex_table(dipco_df)
    #create_chime_baseline_latex_table(chime6_df)
    #TODO noise as additional information
    #create_noise_baseline_latex_table(df)

def create_comparison_table():
    # should contain published baseline column whisper untrained baseline column, training-methods as columns  dev/eval and close-field far field as index
    pass

