def save_latex_csv(df):
    # select only the runs not on dev_mode
    df = df[df['developer_mode'] == "N"]
    # delete the dates
    df.drop(columns=['developer_mode', 'date', 'results_path', 'notes','environment'], inplace=True)
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





