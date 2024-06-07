import os
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd

from modules.basic.consts import folder_datasets, settings, df_metrics_columns, methods_detection, methods_cleanup
from modules.basic.core_methods import read_dfs
from modules.experiment.exp_pipeline import exp_pipeline, calc_stats
from modules.experiment.exp_visualization import prepare_data_and_plot

dataset = "experiment"
iteration = dataset + "_1"
make_plots = True
seed = 1
np.random.seed(seed)
dataset_folder = os.path.join(folder_datasets, dataset)
patches = settings[dataset]["patches"]
ecg_patch = settings[dataset]["ecg_patch"]
f = settings[dataset]["f"]
taken_intervals = ["several", "middle", "random", "several_random", "simple"]
n_intervals = 0
length_of_interval = 60
take_int = taken_intervals[4]
methods_detection = ['nabian2018', 'neurokit']
methods_cleanup = ['hamilton2002', 'neurokit']

# iterate through all the files and take only files that have at least 50 seconds of data
list_of_files = os.listdir(dataset_folder)
list_of_files = [file_name for file_name in list_of_files if
                 len(pd.read_csv(os.path.join(dataset_folder, file_name))) > 50 * f]

if __name__ == '__main__':
    df_metrics = pd.DataFrame(columns=df_metrics_columns)
    df_metrics_best = pd.DataFrame(columns=df_metrics_columns)
    df_hrv_bcg = pd.DataFrame()
    df_hrv_ecg = pd.DataFrame()
    for file_name in list_of_files:
        signal_length = int(len(pd.read_csv(os.path.join(dataset_folder, file_name))) / f)
        if take_int == "several":
            signal_intervals = \
                [[i, i + length_of_interval] for i in range(0,
                                                            int(signal_length) - length_of_interval,
                                                            length_of_interval)]
            signal_intervals = signal_intervals[:n_intervals]
        elif take_int == "middle":
            signal_intervals = [
                [signal_length // 2 - length_of_interval / 2, signal_length // 2 + length_of_interval / 2]]
        elif take_int == "random":
            # select a single random interval of length 60 seconds
            random_start = np.random.randint(0, signal_length - length_of_interval)
            signal_intervals = [[random_start, random_start + length_of_interval]]
        elif take_int == "several_random":
            random_starts = [np.random.randint(0, signal_length)]
            for _ in range(n_intervals - 1):
                new_number = np.random.randint(0, signal_length)

                while abs(new_number - random_starts[-1]) < length_of_interval:
                    new_number = np.random.randint(0, signal_length)
                random_starts.append(new_number)
            signal_intervals = [[i, i + length_of_interval] for i in random_starts]
        else:
            # simply take the first taken_intervals seconds
            signal_intervals = [[0, length_of_interval]]
        for signal_int in signal_intervals:
            print('')
            print(file_name, ' ', signal_int)
            # create the dataframe, in which all the metrics will be stored
            file_path = os.path.join(dataset_folder, file_name)
            _, df = read_dfs(dataset, file_name, int(signal_int[0]), int(signal_int[1]), settings)
            if len(df) < 1000:
                continue
            bcg_signals = []
            stats = {}
            for detection_method in methods_detection:
                print(detection_method, end=' ')
                for cleanup_method in methods_cleanup:
                    for patch in patches:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            j_peaks, labels, signal, ecg_signal, r_peaks, elapsed_time, success \
                                = exp_pipeline(df, patch, detection_method=detection_method,
                                               cleanup_method=cleanup_method,
                                               dataset=dataset)
                        if len(signal) != 0:
                            signal = (signal - min(signal)) / (max(signal) - min(signal))
                            bcg_signals.append(signal)
                        else:
                            bcg_signals.append(df[patch])
                        r_peaks, j_peaks, stats = calc_stats(r_peaks, j_peaks, f=f)
                        df_m_temp = pd.DataFrame([[file_name,
                                                   str(signal_int),
                                                   detection_method,
                                                   cleanup_method,
                                                   patch,
                                                   len(r_peaks),
                                                   len(j_peaks),
                                                   stats['hr_bcg'],
                                                   stats['hr_ecg'],
                                                   stats['hr_diff'],
                                                   stats['hr_bcg_nn'],
                                                   stats['hr_ecg_nn'],
                                                   stats['hr_bcg_sdnn'],
                                                   stats['hr_ecg_sdnn'],
                                                   stats['precision'],
                                                   stats['recall'],
                                                   stats['f1-score'],
                                                   stats['rj_std'],
                                                   elapsed_time,
                                                   success]],
                                                 columns=df_metrics_columns)
                        df_metrics = pd.concat([df_metrics, df_m_temp])

                        hrv_stats = pd.DataFrame()
                        hrv_stats['file_name'] = [file_name]
                        hrv_stats['signal_interval'] = [str(signal_int)]
                        hrv_stats['detection_method'] = [detection_method]
                        hrv_stats['cleanup_method'] = [cleanup_method]
                        hrv_stats['patch'] = [patch]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # noinspection PyBroadException
                            try:
                                hrv_add = nk.hrv(peaks=j_peaks, sampling_rate=f, show=False)
                                hrv_stats = hrv_stats.join(hrv_add)
                            except:
                                pass
                        df_hrv_bcg = pd.concat([df_hrv_bcg, hrv_stats])
                        hrv_stats = pd.DataFrame()
                        hrv_stats['file_name'] = [file_name]
                        hrv_stats['signal_interval'] = [str(signal_int)]
                        hrv_stats['detection_method'] = [detection_method]
                        hrv_stats['cleanup_method'] = [cleanup_method]
                        hrv_stats['patch'] = [patch]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # noinspection PyBroadException
                            try:
                                hrv_add = nk.hrv(peaks=r_peaks, sampling_rate=f, show=False)
                                hrv_stats = hrv_stats.join(hrv_add)
                            except:
                                pass
                        df_hrv_ecg = pd.concat([df_hrv_ecg, hrv_stats])

    df_metrics.to_csv(os.path.join("output", "detection", "full", "metrics_" + str(iteration) + ".csv"), index=False)
    df_hrv_bcg.to_csv(os.path.join("output", "detection", "full", "hrv_bcg_" + str(iteration) + ".csv"), index=False)
    df_hrv_ecg.to_csv(os.path.join("output", "detection", "full", "hrv_ecg_" + str(iteration) + ".csv"), index=False)

    if make_plots:
        # for each file, calculate the best detection method
        for file_name in list_of_files:
            df_m_temp = df_metrics[df_metrics['file_name'] == file_name]
            signal_intervals = \
                [[i, i + 60] for i in range(0,
                                            int(len(pd.read_csv(os.path.join(dataset_folder, file_name))) / f),
                                            60)]
            int_index = len(signal_intervals) // 2
            signal_intervals = [signal_intervals[int_index]]
            for signal_int in signal_intervals:
                df_m_temp_2 = df_m_temp[df_m_temp['signal_interval'] == str(signal_int)]
                df_m_temp_2.reset_index(inplace=True)
                df_m_temp_2 = df_m_temp_2[df_m_temp_2['f1-score'] == df_m_temp_2['f1-score'].max()]
                df_metrics_best = pd.concat([df_metrics_best, df_m_temp_2])

        # for each file, make a plot for the best detection method
        for file_name in list_of_files:
            df_m_temp = df_metrics[df_metrics['file_name'] == file_name]
            signal_length = int(len(pd.read_csv(os.path.join(dataset_folder, file_name))) / f)
            if take_int == "several":
                signal_intervals = \
                    [[i, i + length_of_interval] for i in range(0,
                                                                int(signal_length) - length_of_interval,
                                                                length_of_interval)]
                signal_intervals = signal_intervals[:n_intervals]
            elif take_int == "middle":
                signal_intervals = [
                    [signal_length // 2 - length_of_interval / 2, signal_length // 2 + length_of_interval / 2]]
            elif take_int == "random":
                # select a single random interval of selected length
                random_start = np.random.randint(0, signal_length - length_of_interval)
                signal_intervals = [[random_start, random_start + length_of_interval]]
            elif take_int == "several_random":
                random_starts = [np.random.randint(0, signal_length)]
                for _ in range(n_intervals - 1):
                    new_number = np.random.randint(0, signal_length)

                    while abs(new_number - random_starts[-1]) < length_of_interval:
                        new_number = np.random.randint(0, signal_length)
                    random_starts.append(new_number)
                signal_intervals = [[i, i + length_of_interval] for i in random_starts]
            else:
                signal_intervals = [[0, length_of_interval]]
            for signal_int in signal_intervals:
                df_m_temp_2 = df_m_temp[df_m_temp['signal_interval'] == str(signal_int)]
                df_m_temp_2.reset_index(inplace=True)
                selected_detection_method = df_m_temp_2['detection_method'].values[0]
                selected_cleanup_method = df_m_temp_2['cleanup_method'].values[0]

                # select the patch with the smallest hr_diff
                if "," in df_m_temp_2['hr_diff'].values:
                    df_m_temp_2['hr_diff'] = df_m_temp_2['hr_diff'].apply(lambda j: float(j.replace(",", "."))
                    if j != "na" else j)
                df_m_temp_2 = df_m_temp_2[df_m_temp_2['hr_diff'] == df_m_temp_2['hr_diff'].min()]
                selected_patch = df_m_temp_2['patch'].values[0]
                title_to_show = file_name + " " + str(signal_int) + '\n'
                title_to_show += "Detection: " + selected_detection_method
                title_to_show += " Cleanup: " + selected_cleanup_method
                title_to_show += " Best patch: " + selected_patch + '\n'
                title_to_show += "Full Precision: " + str(df_m_temp_2['precision'].values[0])
                title_to_show += " Recall: " + str(df_m_temp_2['recall'].values[0])
                title_to_show += " F1-score: " + str(df_m_temp_2['f1-score'].values[0])
                title_to_show += " HR: " + str(round(df_m_temp_2['hr_bcg'].values[0], 2))
                title_to_show += (" J/R: " + str(df_m_temp_2["j_peaks"].values[0])
                                  + "/" + str(df_m_temp_2["r_peaks"].values[0]))
                _, df = read_dfs(dataset, file_name, int(signal_int[0]), int(signal_int[1]), settings)
                j_peaks_all = []
                labels_all = [[] * len(patches)]
                cleaned_signals = []
                r_peaks_temp = []

                for patch in patches:
                    j_peaks_temp, labels_temp, signal_temp, ecg_signal, r_peaks_temp, elapsed_time, success \
                        = exp_pipeline(df, patch, detection_method=selected_detection_method,
                                       cleanup_method=selected_cleanup_method,
                                       dataset=dataset)
                    signal_temp = (signal_temp - min(signal_temp)) / (max(signal_temp) - min(signal_temp))
                    cleaned_signals.append(signal_temp)
                    j_peaks_all.append(j_peaks_temp)
                    labels_all.append(labels_temp)
                r_peaks = r_peaks_temp

                # form the 10-second cuts of the signal and plot each of them
                for i in range(0, int(signal_int[1]) - int(signal_int[0]), 10):
                    prepare_data_and_plot(df, cleaned_signals, j_peaks_all, r_peaks, df_m_temp_2,
                                          selected_patch, file_name, title_to_show, signal_int, f, dataset, i)
