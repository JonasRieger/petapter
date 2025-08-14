import pandas as pd
import numpy as np
from pathlib import Path


def mean_df(list_of_dfs):
    result_df = pd.DataFrame(index=list_of_dfs[0].index, columns=list_of_dfs[0].columns)
    for col in result_df.columns:
        for idx in result_df.index:
            values = [df.loc[idx, col] for df in list_of_dfs]
            mean = np.mean(values)
            std = np.std(values)
            result_df.loc[idx, col] = f"{mean:.2f} ± {std:.2f}"

    return result_df


def summarize_over_runs(experiment_path, suffix=None):
    if isinstance(experiment_path,str):
        experiment_path = Path(experiment_path)
    classification_reports = []
    for file_path in experiment_path.glob(f'**/classification_report{"_"+suffix if suffix else ""}.csv'):
        if file_path.is_file():
            classification_reports.append(pd.read_csv(file_path, index_col=0))
    mean = mean_df(classification_reports)
    mean.to_csv(experiment_path / f'mean_report{"_"+suffix if suffix else ""}.csv')
    return mean
