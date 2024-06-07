import pandas as pd

dataset = "cebsdb"
iteration = "1"
metric = "hr_diff"

df_metrics = pd.read_csv("output/detection/full/metrics_" + dataset + "_" + iteration + ".csv")
df_metrics_averaged = pd.DataFrame(columns=['file_name', 'cleanup_method', 'hr_diff', 'hr_diff_std', 'hr_bcg',
                                            'hr_ecg', 'hr_diff_str', 'p-value', 'precision', 'recall', 'f1-score',
                                            'hr_bcg_sdnn', 'hr_ecg_sdnn'])
# for each detection_method, cleanup_method combination, calculate the average value of the metric
for detection_method in df_metrics['detection_method'].unique():
    for cleanup_method in df_metrics['cleanup_method'].unique():
        df_m_temp = df_metrics[(df_metrics['detection_method'] == detection_method) &
                               (df_metrics['cleanup_method'] == cleanup_method)]
        hr_diff = df_m_temp['hr_diff']
        hr_diff = hr_diff[hr_diff != "na"]
        hr_diff_val = hr_diff.mean()
        hr_diff_std = hr_diff.std()
        df_m_temp = df_m_temp[df_m_temp['hr_bcg'] != "na"]
        df_m_temp = df_m_temp[df_m_temp['hr_ecg'] != "na"]
        hr_bcg = str(round(df_m_temp['hr_bcg'].mean(), 1)) + "±" + str(round(df_m_temp['hr_bcg'].std(), 1))
        hr_ecg = str(round(df_m_temp['hr_ecg'].mean(), 1)) + "±" + str(round(df_m_temp['hr_ecg'].std(), 1))
        hr_diff_str = str(round(hr_diff_val, 1)) + "±" + str(round(hr_diff_std, 1))
        p_value = "na"
        # add also precision, recall, f1-score
        precision = df_m_temp['precision'].mean()
        precision = str(round(precision, 1)) + "±" + str(round(df_m_temp['precision'].std(), 1))
        recall = df_m_temp['recall'].mean()
        recall = str(round(recall, 1)) + "±" + str(round(df_m_temp['recall'].std(), 1))
        f1_score = df_m_temp['f1-score'].mean()
        f1_score = str(round(f1_score / 100, 3)) + "±" + str(round(df_m_temp['f1-score'].std() / 100, 3))

        hr_bcg_sdnn = df_m_temp['hr_bcg_sdnn']
        hr_bcg_sdnn = hr_bcg_sdnn[hr_bcg_sdnn != "na"]
        hr_bcg_sdnn1 = hr_bcg_sdnn.mean()
        hr_bcg_sdnn1 = str(round(hr_bcg_sdnn1, 1)) + "±" + str(round(hr_bcg_sdnn.std(), 1))
        hr_ecg_sdnn = df_m_temp['hr_ecg_sdnn']
        hr_ecg_sdnn = hr_ecg_sdnn[hr_ecg_sdnn != "na"]
        hr_ecg_sdnn1 = hr_ecg_sdnn.mean()
        hr_ecg_sdnn1 = str(round(hr_ecg_sdnn1, 1)) + "±" + str(round(hr_ecg_sdnn.std(), 1))

        # add the average value to the new dataframe
        # noinspection PyProtectedMember
        df_metrics_averaged = df_metrics_averaged._append({'detection_method': detection_method,
                                                           'cleanup_method': cleanup_method,
                                                           'hr_diff': hr_diff_val, 'hr_diff_std': hr_diff_std,
                                                           'hr_bcg': hr_bcg, 'hr_ecg': hr_ecg,
                                                           'hr_diff_str': hr_diff_str, 'p-value': p_value,
                                                           'precision': precision, 'recall': recall,
                                                           'f1-score': f1_score, 'hr_bcg_sdnn': hr_bcg_sdnn1,
                                                           'hr_ecg_sdnn': hr_ecg_sdnn1},
                                                          ignore_index=True)
print(df_metrics_averaged)

# A.
# for each detection_method take only the best cleanup_method
df_metrics_averaged_2 = pd.DataFrame(columns=['detection_method', 'cleanup_method', 'hr_diff', 'hr_diff_std'])
for detection_method in df_metrics_averaged['detection_method'].unique():
    df_m_temp = df_metrics_averaged[df_metrics_averaged['detection_method'] == detection_method]
    df_m_temp = df_m_temp.sort_values(by=metric)
    df_m_temp = df_m_temp.head(1)
    df_metrics_averaged_2 = pd.concat([df_metrics_averaged_2, df_m_temp])
print(df_metrics_averaged_2)
# sort the values based on hr_diff
df_metrics_averaged_2 = df_metrics_averaged_2.sort_values(by=metric)
print(df_metrics_averaged_2)
# save the table to a csv file
df_metrics_averaged_2.to_csv("output/detection/derived/metrics_averaged_"
                             + iteration + "_" + dataset + ".csv", index=False)

# B.
# drop all the rows apart from nabian2018
df_metrics_averaged_3 = df_metrics_averaged[df_metrics_averaged['detection_method'] == "nabian2018"]
# sort values based on hr_diff
df_metrics_averaged_3 = df_metrics_averaged_3.sort_values(by=metric)
# save the table to a csv file
df_metrics_averaged_3.to_csv("output/detection/derived/metrics_averaged_"
                             + iteration + "_" + dataset + "_nabian2018.csv", index=False)
