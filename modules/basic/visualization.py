import os

import numpy as np
from matplotlib import pyplot as plt

peak_colors = ['g', 'b', 'y', 'm', 'c']
peak_ecg = ['P', 'Q', 'R', 'S', 'T']
left, width = .01, .98
bottom, height = .25, .7
right = left + width
top = bottom + height


def plot_detection(df, cleaned_signals, all_j_peaks, r_peaks, all_labels,
                   best_patch, file_name, title_to_show, settings):
    # make a plot with 6 subplots, where put all the peaks with labels
    settings['patches labels'] = ["Patch 1, axis X", "Patch 1, axis Y", "Patch 1, axis Z",
                                  "Patch 2, axis X", "Patch 2, axis Y", "Patch 2, axis Z"]
    all_j_peaks = [list(j_peaks) for j_peaks in all_j_peaks]
    num_subplots = len(settings['patches']) + 1
    plt.figure(figsize=(25, num_subplots * 5))

    for i, j_peaks in enumerate(all_j_peaks):
        signal = df[settings['patches'][i]]
        # normalize signal to be in [0, 1]
        signal = (signal - min(signal)) / (max(signal) - min(signal))
        plt.subplot(num_subplots, 1, i + 1)
        plot_signal_with_labels(all_labels, i, j_peaks, signal)

    # Plot cleaned up signals
    if cleaned_signals is not None:
        for i, j_peaks in enumerate(all_j_peaks):
            signal = cleaned_signals[i]
            # normalize signal to be in [0, 1]
            signal = (signal - min(signal)) / (max(signal) - min(signal))
            # move the signal to the right signal interval
            df['patches_с'] = signal
            signal = df['patches_с']
            plt.subplot(num_subplots, 1, i + 1)
            plt.plot(signal, color='orange', alpha=0.9)

    # Best patch is shown in a frame
    if best_patch is not None:
        best_patch_index = settings['patches'].index(best_patch)
        plt.subplot(num_subplots, 1, best_patch_index + 1)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)

    # ECG channel
    plt.subplot(num_subplots, 1, num_subplots)
    ecg_signal = df[settings['ecg_patch']]
    ecg_signal = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
    plt.plot(ecg_signal, color='black', alpha=0.9)
    plt.ylabel('ECG', fontsize=32)
    for r_peak in r_peaks:
        plt.plot([r_peak, r_peak], [0, 1], color='r')
    plt.xlabel('Time, s', fontsize=32)

    # show also R peaks on all subplots as thin red vertical lines
    for i, j_peaks in enumerate(all_j_peaks):
        plt.subplot(num_subplots, 1, i + 1)
        plt.ylabel(settings['patches labels'][i], fontsize=32)
        for r_peak in r_peaks:
            plt.plot([r_peak, r_peak], [0, 1], color='r')

    # add title to the whole plot with file name and signal interval
    plt.suptitle(title_to_show)
    # take the lag value from the df index
    lag = df.index[0]
    for i in range(num_subplots):
        plt.subplot(num_subplots, 1, i + 1)
        x_ticks = lag + np.arange(0, len(df) + 1, 1000)
        plt.xticks(x_ticks, [str(round(x / 1000, 2)) for x in x_ticks])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "detection", file_name + ".png"))
    plt.close()


def plot_signal_with_labels(all_labels, i, j_peaks, signal):
    plt.plot(signal, color='gray', alpha=0.3)
    for j_peak in j_peaks:
        if all_labels is not None:
            plt.plot([j_peak, j_peak], [0, 1], color=peak_colors[all_labels[i][j_peaks.index(j_peak)]])
            plt.text(j_peak, 0, all_labels[i][j_peaks.index(j_peak)], rotation=90)
        else:
            plt.plot([j_peak, j_peak], [0, 1], color='blue')
        # plot as text the interval between the peak and the new peak, in milliseconds
        if j_peaks.index(j_peak) != len(j_peaks) - 1:  # + 20
            plt.text(j_peak + 20, 0.8, str((j_peaks[j_peaks.index(j_peak) + 1] - j_peak)), rotation=90, size=24)
        # plot as text the interval between the peak and the previous peak, in milliseconds
        if j_peaks.index(j_peak) != 0:  # - 95
            plt.text(j_peak - 155, 0.2, str((j_peak - j_peaks[j_peaks.index(j_peak) - 1])), rotation=90, size=24)


def plot_detection_no_ecg(df, cleaned_signals, all_j_peaks, all_r_peaks, all_labels,
                          best_patch, file_name, title_to_show, settings):
    settings['patches labels'] = ["Subject 1, rest", "Subject 1, interference", "Subject 2, rest",
                                  "Subject 2, interference", "Subject 3, rest", "Subject 3, interference"]
    # make a plot with 6 subplots, where put all the peaks with labels
    all_j_peaks = [list(j_peaks) for j_peaks in all_j_peaks]
    num_subplots = len(settings['patches'])
    plt.figure(figsize=(25, num_subplots * 2.5))

    for i, j_peaks in enumerate(all_j_peaks):
        signal = df[settings['patches'][i]]
        # normalize signal to be in [0, 1]
        signal = (signal - min(signal)) / (max(signal) - min(signal))
        plt.subplot(3, 2, i + 1)
        plot_signal_with_labels(all_labels, i, j_peaks, signal)

    # Plot cleaned up signals
    if cleaned_signals is not None:
        for i, j_peaks in enumerate(all_j_peaks):
            signal = cleaned_signals[i]
            # normalize signal to be in [0, 1]
            signal = (signal - min(signal)) / (max(signal) - min(signal))
            # move the signal to the right signal interval
            df['patches_с'] = signal
            signal = df['patches_с']
            plt.subplot(3, 2, i + 1)
            plt.plot(signal, color='orange', alpha=0.9)

    # Best patch is shown in a frame
    if best_patch is not None:
        best_patch_index = settings['patches'].index(best_patch)
        plt.subplot(num_subplots, 1, best_patch_index + 1)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)

    # show also R peaks on all subplots as thin red vertical lines
    for i, j_peaks in enumerate(all_j_peaks):
        plt.subplot(3, 2, i + 1)
        for r_peak in all_r_peaks[i]:
            plt.plot([r_peak, r_peak], [0, 1], color='r')

    # add title to the whole plot with file name and signal interval
    plt.suptitle(title_to_show)
    # take the lag value from the df index
    lag = df.index[0]
    for i in range(num_subplots):
        plt.subplot(3, 2, i + 1)
        x_ticks = lag + np.arange(0, len(df) + 1, 1000)
        plt.xticks(x_ticks, [str(round(x / 1000, 2)) for x in x_ticks])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylabel(settings['patches labels'][i], fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "detection", file_name + ".png"))
    plt.close()
