from modules.basic.consts import settings
from modules.basic.visualization import plot_detection
from modules.experiment.exp_pipeline import calc_stats

# a palette of 6 colours from completely light yellow to dark red
pal_colors = {'peak_power': ["#FFFF00", "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000"],
              'interpolated': ["#808080", "#FF0000"]}


def prepare_data_and_plot(df, cleaned_signals, j_peaks_all, r_peaks, df_m_temp_2, selected_patch,
                          file_name, title_to_show, signal_int, f, dataset, i):
    patches = settings[dataset]["patches"]
    signal_int_temp = [i, i + 10]
    df_temp = df.iloc[signal_int_temp[0] * f:signal_int_temp[1] * f]
    # select only the signals that are within the interval
    cleaned_signals_temp = \
        [signal[signal_int_temp[0] * f:signal_int_temp[1] * f] for signal in cleaned_signals]
    j_peaks_all_temp = [j_peaks[j_peaks >= signal_int_temp[0] * f] for j_peaks in j_peaks_all]
    j_peaks_all_temp = \
        [j_peaks_temp[j_peaks_temp < signal_int_temp[1] * f] for j_peaks_temp in j_peaks_all_temp]
    r_peaks_temp = r_peaks[r_peaks >= signal_int_temp[0] * f]
    r_peaks_temp = r_peaks_temp[r_peaks_temp < signal_int_temp[1] * f]
    file_name_temp = file_name + " " + str(signal_int) + " " + str(signal_int_temp)
    r_peaks_a, j_peaks_a, stats_a = calc_stats(r_peaks_temp,
                                               j_peaks_all_temp[patches.index(selected_patch)],
                                               f=f)
    if df_m_temp_2 is not None:
        title_to_show = title_to_show + "\n Signal interval " + str(signal_int_temp) + ': '
        title_to_show += " Precision: " + str(stats_a['precision'])
        title_to_show += " Recall: " + str(stats_a['recall'])
        title_to_show += " F1-score: " + str(stats_a['f1-score'])
        title_to_show += " HR: " + str(round(df_m_temp_2['hr_bcg'].values[0], 2))
        title_to_show += (" J/R: " + str(len(j_peaks_a)) + "/" + str(len(r_peaks_a)))
    title_to_show = ""
    plot_detection(df_temp, cleaned_signals_temp, j_peaks_all_temp, r_peaks_temp, None,
                   selected_patch, file_name_temp, title_to_show, settings[dataset])
