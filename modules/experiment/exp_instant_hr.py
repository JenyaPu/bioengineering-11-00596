import os

import neurokit2 as nk
import numpy as np
import pandas as pd

from modules.basic.consts import folder_datasets, settings, methods_detection, methods_cleanup
from modules.basic.core_methods import read_dfs
from modules.experiment.exp_pipeline import exp_pipeline

dataset = "cebsdb"
dataset_folder = os.path.join(folder_datasets, dataset)
patches = settings[dataset]["patches"]
f = settings[dataset]["f"]
detection_method = methods_detection[0]
cleanup_method = methods_cleanup[0]
patch = patches[0]

list_of_files = os.listdir(dataset_folder)
list_of_files = [file_name for file_name in list_of_files if
                 len(pd.read_csv(os.path.join(dataset_folder, file_name))) > 50 * f]
signal_int = [0, 60]

j_intervals_all = []
r_intervals_all = []
for file_name in list_of_files:
    print("File: ", file_name)
    _, df = read_dfs(dataset, file_name, signal_int[0], signal_int[1], settings)
    j_peaks, labels, signal, ecg_signal, r_peaks, elapsed_time, success \
        = exp_pipeline(df, patch, detection_method=detection_method,
                       cleanup_method=cleanup_method,
                       dataset=dataset)
    cleaned_signals = nk.ecg_clean(ecg_signal, sampling_rate=f)
    print("J-peaks: ", len(j_peaks), " R-peaks: ", len(r_peaks))
    # form pairs of j-peaks and r-peaks, put there only those that are 400 ms apart
    pairs = []
    for j_peak in j_peaks:
        for r_peak in r_peaks:
            if 0 < j_peak - r_peak < 300:
                pairs.append((j_peak, r_peak))
                break
    j_intervals = []
    r_intervals = []
    for i in range(1, len(pairs)):
        j_intervals.append(pairs[i][0] - pairs[i - 1][0])
        r_intervals.append(pairs[i][1] - pairs[i - 1][1])
    print("J-intervals: ", len(j_intervals), " R-intervals: ", len(r_intervals))
    # Identify indices where j_intervals are greater than 1200 ms
    indices_to_remove = [i for i, j_interval in enumerate(j_intervals) if j_interval > 1200]

    # Create new lists excluding the identified indices
    j_intervals = [j for i, j in enumerate(j_intervals) if i not in indices_to_remove]
    r_intervals = [r for i, r in enumerate(r_intervals) if i not in indices_to_remove]
    print("J-intervals: ", len(j_intervals), " R-intervals: ", len(r_intervals))
    # calculate mean difference between j-peaks and r-peaks intervals and sd
    j_intervals_all.extend(j_intervals)
    r_intervals_all.extend(r_intervals)
    j_intervals = np.array(j_intervals)
    r_intervals = np.array(r_intervals)
    mean_abs_diff = np.mean(np.abs(j_intervals - r_intervals))
    sd_abs_diff = np.std(np.abs(j_intervals - r_intervals))
    print("Mean abs diff: ", mean_abs_diff, " SD abs diff: ", sd_abs_diff)
j_intervals_all = np.array(j_intervals_all)
r_intervals_all = np.array(r_intervals_all)
mean_abs_diff = np.mean(np.abs(j_intervals_all - r_intervals_all))
sd_abs_diff = np.std(np.abs(j_intervals_all - r_intervals_all))
print("Mean r_intervals: ", np.mean(r_intervals_all), " SD r_intervals: ", np.std(r_intervals_all))
print("Mean j_intervals: ", np.mean(j_intervals_all), " SD j_intervals: ", np.std(j_intervals_all))
print("Mean abs diff: ", mean_abs_diff, " SD abs diff: ", sd_abs_diff)
