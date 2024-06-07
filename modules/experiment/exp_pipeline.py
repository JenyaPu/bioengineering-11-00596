import time
import warnings

import neurokit2 as nk
import numpy as np
from scipy.signal import butter, sosfiltfilt

from modules.basic.consts import settings

# suppress warnings
warnings.simplefilter("ignore")


def eliminate_low_amplitude_peaks(r_peaks, signal):
    # Function to calculate the amplitude threshold (Amp-Thr)

    # Initialize amplitude threshold (Amp-Thr) as 1/3 times the maximum amplitude of the first 2 seconds
    sampling_rate = 1  # Assuming 1 sample per second
    first_2_seconds_index = min(len(signal), 2 * sampling_rate)
    amplitude_threshold = (1 / 3) * np.max(signal[:first_2_seconds_index])

    # Store indices of R peaks passing the amplitude threshold
    filtered_R_peaks = []

    # Iterate through the R peaks
    for r_peak_index in r_peaks:
        # Check if the amplitude of the R peak is above the threshold
        if signal[r_peak_index] >= amplitude_threshold:
            filtered_R_peaks.append(r_peak_index)
            # Update amplitude threshold if sufficient R peaks have been detected
            if len(filtered_R_peaks) >= 8:
                # take the last 8 R peaks
                last_8_r_peaks = filtered_R_peaks[-8:]
                # take the amplitudes of the last 8 R peaks
                last_8_r_peaks_amplitudes = [signal[r_peak_index] for r_peak_index in last_8_r_peaks]
                # calculate the mean amplitude of the last 8 R peaks
                amplitude_threshold = np.mean(last_8_r_peaks_amplitudes) * 0.75
    return filtered_R_peaks


def interpolate_missing_r_peaks(r_peaks, signal):
    # Function to calculate RR interval threshold (RR-Thr)
    def calculate_rr_interval_threshold(rr_intervals):
        return np.mean(rr_intervals[-8:]) * 1.66

    # Initialize RR interval threshold (RR-Thr) as 1.66 times the previous RR interval
    RR_intervals = np.diff(r_peaks)
    RR_threshold = RR_intervals[-1] * 1.66

    # Store interpolated R peaks
    interpolated_R_peaks = [r_peaks[0]]

    # Iterate through RR intervals and interpolate missing R peaks
    for i in range(len(RR_intervals)):
        # Check if RR interval is greater than RR threshold, implying a missing R peak
        if RR_intervals[i] > RR_threshold:
            # Calculate the time range for searching the new R peak
            search_start = int(r_peaks[i] + 0.2 * 1000)  # R1 + 200ms
            search_end = int(r_peaks[i + 1] - 0.2 * 1000)  # Ri+1 - 200ms

            # Find the index of the maximum amplitude within the time range
            new_R_peak_index = np.argmax(signal[search_start:search_end]) + search_start

            # Add the new R peak to the list of interpolated R peaks
            interpolated_R_peaks.append(new_R_peak_index)

        # Add the original R peak to the list of interpolated R peaks
        interpolated_R_peaks.append(r_peaks[i + 1])

        # Update RR threshold if enough RR intervals have been detected
        if len(RR_intervals) - i >= 9:
            RR_threshold = calculate_rr_interval_threshold(RR_intervals[i + 1:])

    return np.array(interpolated_R_peaks)


def ecg_clean_pustozerov(signal, sf):
    sos = butter(N=4, Wn=[5, 35], fs=sf, btype='bandpass', output='sos')
    signal = sosfiltfilt(sos, x=signal)
    return signal


def ecg_peaks_nabian_simple(signal, sf):
    """R peak detection method by Nabian et al. (2018) inspired by the Pan-Tompkins algorithm.

    - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., Ostadabbas, S. (2018).
      An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data.
      IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
    """
    window_size = int(0.2 * sf)

    peaks = np.zeros(len(signal))

    for i in range(1 + window_size, len(signal) - window_size):
        ecg_window = signal[i - window_size: i + window_size]
        peak = np.argmax(ecg_window)
        if i == (i - window_size - 1 + peak):
            peaks[i] = 1

    j_peaks = np.where(peaks == 1)[0]
    return j_peaks


def ecg_peaks_nabian_original(signal, sf):
    """R peak detection method by Nabian et al. (2018) inspired by the Pan-Tompkins algorithm.

    - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., Ostadabbas, S. (2018).
      An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data.
      IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
    """
    window_size = int(0.4 * sf)

    peaks = np.zeros(len(signal))

    for i in range(1 + window_size, len(signal) - window_size):
        ecg_window = signal[i - window_size: i + window_size]
        peak = np.argmax(ecg_window)
        if i == (i - window_size - 1 + peak):
            peaks[i] = 1

    j_peaks = np.where(peaks == 1)[0]
    print(len(j_peaks), end=' ')
    j_peaks = eliminate_low_amplitude_peaks(j_peaks, signal)
    print(len(j_peaks), end=' ')
    j_peaks = interpolate_missing_r_peaks(j_peaks, signal)
    print(len(j_peaks), end=' | ')

    return j_peaks


# noinspection PyBroadException
def define_peaks_advanced(df, patch, sf, method_detection, method_cleanup='none'):
    # calculate elapsed time
    start_time = time.perf_counter()
    try:
        if method_cleanup == 'none':
            signal = df[patch]
        elif method_cleanup == 'pustozerov2024':
            signal = ecg_clean_pustozerov(df[patch], sf=sf)
        else:
            signal = nk.ecg_clean(df[patch], sampling_rate=sf, method=method_cleanup)
        if method_detection == 'nabian2018original':
            r_peaks = ecg_peaks_nabian_original(signal, sf)
        elif method_detection == 'nabian2018simple':
            r_peaks = ecg_peaks_nabian_simple(signal, sf)
        else:
            _, results = nk.ecg_peaks(signal, sampling_rate=sf, method=method_detection)
            r_peaks = results["ECG_R_Peaks"]
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return r_peaks, signal, elapsed_time, 1
    except:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return [], [], elapsed_time, 0


# noinspection DuplicatedCode,PyBroadException,PyTypeChecker
def calc_stats(r_peaks, j_peaks, f):
    pairs = []
    # convert to lists
    r_peaks = list(r_peaks)
    j_peaks = list(j_peaks)

    # form pairs with the following condition: there is a j_peak within the interval between two r_peaks
    for r_peak in r_peaks:
        for j_peak in j_peaks:
            # if this is not the last r_peak
            if r_peaks.index(r_peak) < len(r_peaks) - 1:
                # if the j_peak is within the interval between two r_peaks
                if r_peak < j_peak < r_peaks[r_peaks.index(r_peak) + 1]:
                    pairs.append((r_peak, j_peak))
                    break
            # if this is the last r_peak
            else:
                # if the j_peak is within the interval between the last r_peak and the end of the signal
                if r_peak < j_peak < len(j_peaks):
                    pairs.append((r_peak, j_peak))
                    break

    # if the first r_peak is not paired, remove it
    if len(pairs) > 0 and pairs[0][0] != r_peaks[0]:
        r_peaks = r_peaks[1:]
    # if the last r_peak is not paired, remove it
    if len(pairs) > 0 and pairs[-1][0] != r_peaks[-1]:
        r_peaks = r_peaks[:-1]
    # if the first j_peak is not paired, remove it
    if len(pairs) > 0 and pairs[0][1] != j_peaks[0]:
        j_peaks = j_peaks[1:]
    # if the last j_peak is not paired, remove it
    if len(pairs) > 0 and pairs[-1][1] != j_peaks[-1]:
        j_peaks = j_peaks[:-1]

    # remove all the pairs that have the same r_peak
    pairs = [pair for pair in pairs if pair[0] not in [pair1[0] for pair1 in pairs if pair1 != pair]]
    # remove all the pairs that have the same j_peak
    pairs = [pair for pair in pairs if pair[1] not in [pair1[1] for pair1 in pairs if pair1 != pair]]
    # print(len(pairs), 'pairs', len(r_peaks_s), 'r_peaks_s', len(j_peaks_s), 'j_peaks_s')
    right_j_peaks_count = len(pairs)

    stats = {}
    # calculate the array of distances between two peaks in pairs
    distances = [pair[1] - pair[0] for pair in pairs]
    # calculate the standard deviation of the distances
    stats['rj_std'] = np.std(distances)

    if len(j_peaks) == 0 or len(r_peaks) == 0:
        stats['recall'] = 0
    else:
        stats['recall'] = round(right_j_peaks_count / len(r_peaks) * 100, 2)
    if len(r_peaks) == 0 or len(j_peaks) == 0:
        stats['precision'] = 0
    else:
        stats['precision'] = round(right_j_peaks_count / len(j_peaks) * 100, 2)
    # calculate f1-score
    if stats['precision'] == 0:
        stats['f1-score'] = 0
    elif stats['recall'] == 0:
        stats['f1-score'] = 0
    else:
        # noinspection PyUnresolvedReferences
        stats['f1-score'] = (
            round(2 * float(stats['precision']) * stats['recall'] / (stats['precision'] + stats['recall']), 2))
    try:
        stats['r_peaks'] = len(r_peaks)
        stats['j_peaks'] = len(j_peaks)
        if len(j_peaks) < 3 or len(r_peaks) < 3:
            stats['hr_bcg'] = "na"
            stats['hr_ecg'] = "na"
            stats['hr_bcg_nn'] = "na"
            stats['hr_ecg_nn'] = "na"
            stats['hr_diff'] = "na"
            stats['hr_bcg_sdnn'] = "na"
            stats['hr_ecg_sdnn'] = "na"
        else:
            # print(np.diff(j_peaks_s))
            stats['hr_bcg'] = 60 * f / np.mean(np.diff(j_peaks))
            stats['hr_ecg'] = 60 * f / np.mean(np.diff(r_peaks))
            try:
                stats['hr_bcg_nn'] = 60 * f / nk.hrv(j_peaks, sampling_rate=f, show=False)['HRV_MeanNN'][0]
            except:
                print('hr_bcg_nn exception')
                stats['hr_bcg_nn'] = "na"
            try:
                stats['hr_ecg_nn'] = 60 * f / nk.hrv(r_peaks, sampling_rate=f, show=False)['HRV_MeanNN'][0]
            except:
                stats['hr_ecg_nn'] = "na"
            stats['hr_diff'] = \
                abs(60 * f / np.mean(np.diff(j_peaks)) - 60 * f / np.mean(np.diff(r_peaks)))
            try:
                stats['hr_bcg_sdnn'] = nk.hrv(j_peaks, sampling_rate=f, show=False)['HRV_SDNN'][0]
            except:
                stats['hr_bcg_sdnn'] = "na"
            try:
                stats['hr_ecg_sdnn'] = nk.hrv(r_peaks, sampling_rate=f, show=False)['HRV_SDNN'][0]
            except:
                stats['hr_ecg_sdnn'] = "na"
    except RuntimeWarning:
        breakpoint()
    return r_peaks, j_peaks, stats


# noinspection DuplicatedCode
def exp_pipeline(df, patch, detection_method="neurokit", cleanup_method="neurokit", dataset='default'):
    # Preprocessing
    f = settings[dataset]['f']
    patch_ecg = settings[dataset]['ecg_patch']
    j_peaks, bcg_signal, elapsed_time, success \
        = define_peaks_advanced(df, patch, f, detection_method, cleanup_method)
    labels = ['S'] * len(j_peaks)

    # Evaluation
    ecg_signal = df[patch_ecg]
    if ecg_signal is not None:
        r_peaks_s, ecg_signal, _, _ = define_peaks_advanced(df, patch_ecg, f, detection_method, cleanup_method)
    else:
        r_peaks_s = []
    return j_peaks, labels, bcg_signal, ecg_signal, r_peaks_s, elapsed_time, success
