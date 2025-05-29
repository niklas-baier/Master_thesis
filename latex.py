import pandas as pd
import ast
def save_latex_csv(df:pd.DataFrame)->None:
    # select only the runs not on dev_mode
    df = df[df['developer_mode'] == "N"]
    # delete the dates
    df.drop(columns=['developer_mode', 'date', 'results_path', 'notes','environment', 'commit_hash'], inplace=True)
    path = "latex_results.csv"
    df.to_csv(path, index=False)
def rename_columns_for_latex(df):
    df.rename(columns={'dataset evaluation part': 'eval part', 'data_portion in training': 'trainingsdata', 'training_version': 'finetuning method'}, inplace=True)
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
    dipco_df = select_relevant_baseline_rows(dipco_df)
    print("baseline table")
    print_baseline_table(dipco_df)
    far_away_df, close_df, all_df = create_vanilla_baseline_latex_table(dipco_df)
    vanilla = create_peft_latex_table(dipco_df)
    diffusion_df = create_dipco_diffusion_latex_table(dipco_df)
    #create_denoising_latex_table(df)
    #create_diffusion_latex_table(df)
    #create_contrastive_latex_table(df)
    #create_GAN_latex_table(df)
    return

def create_overview_latex_table(df):
    #TODO selectino
    print_overview_table(dipco_df=df)

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
    return vanilla

def create_dipco_diffusion_latex_table(df):
    df = df.query( 'diffusion == "Y"')
    no_training_df = df.query( 'Training == "NT"')
    training_df = df.query( 'Training == "T"')
    print("diffusion_with training")
    print_baseline_table(training_df)
    print("diffusion_without training")
    print_baseline_table(no_training_df)
    return df


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
    return far_away_df,close_df,all_df
def select_relevant_baseline_rows(dipco_df):
    dipco_df.dropna(subset=['wer_per_mic_type'], inplace=True) # filter out the old values that are NaN
    filtered_df = dipco_df.query('oversampling.isna() or oversampling == 1')
    dipco_df = dipco_df.query('beamforming != "Y"') # TODO augmentation already filtered out ? is not in the columns
    dipco_df = dipco_df.query('Training== "NT"')
    return dipco_df


def select_min_rows(dipco_df):

    dipco_df['precision'] = dipco_df['precision'].fillna('full')
    dipco_df.drop( columns=['developer_mode', 'date', 'results_path', 'environment', 'commit_hash', 'wer', 'cer', 'checkpoint-path', 'wer_per_mic','oversampling', 'wer_per_session', 'wer_per_special_token','diffusion', 'training_time', 'Unnamed: 0', 'num_trainable_parameters', 'beamforming','wer_per_mic_type', 'inference_time'], inplace=True )
    grouped_by_notes = dipco_df.groupby(['notes', 'Training', 'Training_version', 'precision', 'dataset_evaluation_part'])
    idx = grouped_by_notes['WER close mic'].idxmin()
    result = dipco_df.loc[idx]
    return result
def convert_to_dict(x):
    import ast
    # If the entry is already a dictionary, return it
    if isinstance(x, dict):
        return x
    # If the entry is a string, try to convert it to a dictionary
    elif isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # Return a default dictionary if conversion fails
            return {'P': None, 'U': None}
    else:
        # Return a default dictionary for other types
        return {'P': None, 'U': None}

def print_overview_table(dipco_df):
    from ast import literal_eval
    dipco_df.dropna(subset=['wer_per_mic_type'], inplace=True) # filter out the old values that are NaN
    import ast
    dipco_df['WER close mic'] = dipco_df['wer_per_mic_type'].apply(lambda x: ast.literal_eval(x)['P'])
    dipco_df['far talk mic'] = dipco_df['wer_per_mic_type'].apply(lambda x: ast.literal_eval(x)['U'])


    dipco_df = select_relevant_baseline_rows(dipco_df)
    keywords = ["beamforming", "storm", "contrastive", "nt-xent", "GAN", "peft", "vanilla","latent diffusion", "facebook denoising"]
    def check_keywords(note):
        if note is None:
            return "else"
        note_lower = note.lower()
        return "else" if not any(keyword.lower() in note_lower for keyword in keywords) else note
    dipco_df['notes'] = dipco_df['notes'].apply(check_keywords)
    dipco_df = dipco_df[dipco_df['notes'] != 'else']
    min_rows = select_min_rows(dipco_df)
    min_rows = rename_columns_for_latex(min_rows)
    min_rows['model\_name'] = min_rows['model\_name'].apply(lambda s: s.split('/', 1)[1] if '/' in s else s)
    min_rows.drop(columns=['dataset'], inplace=True)
    min_rows.rename(columns={'dataset\_evaluation\_part': 'eval part', 'data_portion in training': 'trainingsdata', 'Training\_version': 'finetuning method', 'far talk mic':'WER far mic'}, inplace=True)
    latex_str = min_rows.to_latex(index=False, float_format="%.4f", bold_rows=True, caption = "this is the overview table of the best runs on the dipco datase", label= "tbl:overview")
    print("latex_str" + latex_str)
    '''
    dev_df = dipco_df[dipco_df['dataset_evaluation_part'] == 'dev']
    eval_df = dipco_df[dipco_df['dataset_evaluation_part'] == 'eval']
    overview_table = pd.DataFrame( {
           'name of the model': dipco_df['model_name'],
           'evaluation part' :dipco_df['dataset_evaluation_part'],
           'close talk': dipco_df['wer_per_mic_type'].apply(lambda x: x['P']),
           'far field': dipco_df['wer_per_mic_type'].apply(lambda x: x['U']),
           'method used' :dipco_df['notes']
           } )
    pivot_table = overview_table.pivot_table(
           index='evaluation part',
           values=['name of the model','close talk', 'far field', 'method used'],

           aggfunc='min'
           )
    print(pivot_table)
    latex_table = pivot_table.round( 2 ).to_latex(
           caption='overview table ',
           label='tab:overview',
           float_format="%.2f",
           index=True,
           header=True
           )
    print(latex_table)'''

def print_baseline_table(dipco_df):
    baseline_table = pd.DataFrame( {
        'name of the model': dipco_df['model_name'],
        'evaluation part' :dipco_df['dataset_evaluation_part'],
        'close talk': dipco_df['wer_per_mic_type'].apply(lambda x: ast.literal_eval(x)['P']),
        'far field': dipco_df['wer_per_mic_type'].apply(lambda x: ast.literal_eval(x)['U']),
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
    create_overview_latex_table(df)
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
