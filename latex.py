import pandas as pd
def save_latex_csv(df:pd.DataFrame)->None:
    # select only the runs not on dev_mode
    df = df[df['developer_mode'] == "N"]
    # delete the dates
    df.drop(columns=['developer_mode', 'date', 'results_path', 'notes','environment', 'commit_hash'], inplace=True)
    path = "latex_results.csv"
    df.to_csv(path, index=False)
def rename_columns_for_latex(df):
    df = df.rename( columns=lambda x: x.replace( '_', r'\_' ) )
    return df


def create_latex_table(df:pd.DataFrame, columns)->None:
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


def create_dipco_baseline_latex_table(dipco_df:pd.DataFrame)->None:
    select_relevant_baseline_rows(dipco_df)
    create_vanilla_baseline_latex_table(dipco_df)
    create_peft_latex_table(dipco_df)
    create_dipco_diffusion_latex_table(dipco_df)
    #create_denoising_latex_table(df)
    return
  

def create_peft_latex_table(df):
    #df = df.query( 'augmentation == "N"' ) #TODO
    #df = df.query( 'oversampling == 1' )
   
    vanilla = df.query( 'Training_version == "peft"' )

    #separate after each data portion in training
    far_away_df = vanilla.query('`data_portion in training` == "far-only"')
    close_df = vanilla.query('`data_portion in training` == "clean-only"')
    all_df = vanilla.query('`data_portion in training` == "all"')
    print("peft-far")
    print_baseline_table(far_away_df)
    print("peft-close")
    print_baseline_table(close_df)
    print("peft-all")
    print_baseline_table(all_df)
    return

def create_dipco_diffusion_latex_table(df):
    df = df.query( 'diffusion == "Y"')
    no_training_df = df.query( 'Training == "NT"')
    training_df = df.query( 'Training == "T"')
    print("diffusion_with training")
    print_baseline_table(training_df)
    print("diffusion_without training")
    print_baseline_table(no_training_df)
    return


def create_denoising_latex_table(df):
    facebook_df = df.query('run_notes == "facebook denoising"')
    noise_reduce_df = df.query('run_notes == "noise reduce"')
    print_baseline_table(facebooks_df)
    print("facebook df above and noisereduce below")
    print_baseline_table(noise_reduce_df)
    return 





def create_vanilla_baseline_latex_table(df):
    vanilla = df.query( 'Training == "T"' )
    #df = df.query( 'augmentation == "N"' ) #TODO
    #df = df.query( 'oversampling == 1' )
    vanilla = vanilla.query( 'beamforming != "Y"' )
    vanilla = vanilla.query( 'Training_version == "vanilla"' )
    #separate after each data portion in training
    far_away_df = vanilla.query('`data_portion in training` == "far-only"')
    close_df = vanilla.query('`data_portion in training` == "clean-only"')
    all_df = vanilla.query('`data_portion in training` == "all"')
    print("vanilla-far")
    print_baseline_table(far_away_df)
    print("vanilla-close")
    print_baseline_table(close_df)
    print("vanilla-all")
    print_baseline_table(all_df)
    return
def select_relevant_baseline_rows(dipco_df):
    from ast import literal_eval
    dipco_df.dropna(subset=['wer_per_mic_type'], inplace=True) # filter out the old values that are NaN
    dipco_df['wer_per_mic_type'] = dipco_df['wer_per_mic_type'].apply(literal_eval)
    filtered_df = dipco_df.query('oversampling.isna() or oversampling == 1')
    dipco_df = dipco_df.query('beamforming != "Y"') # TODO augmentation already filtered out ? is not in the columns
    dipco_df = dipco_df.query('Training== "NT"')
    print("baseline table")
    print_baseline_table(dipco_df)
    return 

def print_baseline_table(dipco_df):
    baseline_table = pd.DataFrame( {
        'name of the model': dipco_df['model_name'],
        'evaluation part' :dipco_df['dataset_evaluation_part'],
        'close talk': dipco_df['wer_per_mic_type'].apply(lambda x: x['P']),
        'far field': dipco_df['wer_per_mic_type'].apply(lambda x: x['U']),
        } )
    pivot_table = baseline_table.pivot_table(
        index='evaluation part',
        values=['name of the model','close talk', 'far field'],
        aggfunc='min'
        )
    print(pivot_table)
    latex_table = pivot_table.round( 2 ).to_latex(
        caption='Baseline table ',
        label='tab:baseline',
        float_format="%.2f",
        index=True,
        header=True
        )
    print(latex_table)
    # split into eval and dev part



def create_base_line_latex_tables():
    df = pd.read_csv( "run_logs.csv" )
    # baseline has no checkpoints and no training
    #df = df.query( 'Training == "NT"' ) TODO
    # only take runs that were run on the full ata
    df = df[df['developer_mode'] == "N"]
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

