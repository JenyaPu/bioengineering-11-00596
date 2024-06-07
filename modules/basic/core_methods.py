import math
import os

import numpy as np
import pandas as pd


def read_dfs(dataset, record, signal_start, signal_end, settings):
    file_path = os.path.join("datasets", "processed", dataset, record)
    f = settings[dataset]["f"]
    df = pd.read_csv(file_path)
    df['Time'] = np.arange(0, len(df)) / f
    signal_start = max(signal_start, 0)
    signal_end = min(signal_end, math.floor(len(df) / f))
    df_fragment = df.iloc[int(min(signal_start, signal_end) * f):int(max(signal_start, signal_end) * f)]
    df_fragment = df_fragment.reset_index(drop=True)
    return df, df_fragment
