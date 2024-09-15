def save_latex_csv(df):
    # select only the runs not on dev_mode
    df = df[df['developer_mode'] == "N"]
    # delete the dates
    df.drop(columns=['developer_mode', 'date', 'results_path', 'notes','environment'], inplace=True)
    path = "latex_results.csv"
    df.to_csv(path, index=False)




