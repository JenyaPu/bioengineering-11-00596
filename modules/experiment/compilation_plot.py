import pandas as pd

from modules.basic.consts import *
from modules.basic.core_methods import read_dfs
from modules.basic.visualization import plot_detection_no_ecg
from modules.experiment.exp_pipeline import exp_pipeline

dataset = "experiment"
iteration = "averaged"
dataset_folder = os.path.join(folder_datasets, dataset)
f = settings[dataset]["f"]
methods_cleanup = 'hamilton2002'
methods_detection = 'nabian2018'

# Note: this function works properly with two or more files
list_of_files = os.listdir(dataset_folder)
list_of_files = [file_name for file_name in list_of_files if
                 len(pd.read_csv(os.path.join(dataset_folder, file_name))) > 50 * f]
signal_int = [0, 5]
df_all = pd.DataFrame()
j_peaks_all = []
r_peaks_all = []
cleaned_signals = []
for file_name in list_of_files:
    _, df = read_dfs(dataset, file_name, signal_int[0], signal_int[1], settings)
    df = df[["Time", "Patch0_z", "Sensor_ECG"]]
    df = df.rename({'Patch0_z': file_name[:5]}, axis='columns')
    patch = file_name[:5]
    j_peaks_temp, labels_temp, signal_temp, ecg_signal, r_peaks_temp, elapsed_time, success \
        = exp_pipeline(df, patch, detection_method=methods_detection,
                       cleanup_method=methods_cleanup,
                       dataset=dataset)
    df_all = pd.concat([df_all, df], axis=1)
    j_peaks_all.append(j_peaks_temp)
    r_peaks_all.append(r_peaks_temp)
    cleaned_signals.append(signal_temp)

df_all = df_all.drop_duplicates(subset="Time")
settings[dataset]['patches'] = [file_name[:5] for file_name in list_of_files]
plot_detection_no_ecg(df_all, cleaned_signals, j_peaks_all, r_peaks_all,
                      None,
                      None, dataset + "_" + iteration, [],
                      settings[dataset])
