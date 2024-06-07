import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.stats import ks_1samp

dataset = "experiment"
iteration = "1"
metric = "hr_diff"

df_metrics = pd.read_csv("output/detection/full/metrics_" + dataset + "_" + iteration + ".csv")
df_metrics_averaged = pd.DataFrame(columns=['file_name', 'patch', 'hr_diff', 'hr_diff_std', 'hr_bcg',
                                            'hr_ecg', 'hr_diff_str', 'p-value', 'precision', 'recall', 'f1-score',
                                            'hr_bcg_sdnn', 'hr_ecg_sdnn'])

best_patches = {'sample': 'Patch1_z'}
# from the dataframe leave only the best patch for each file_name
print(df_metrics)
df_metrics_filtered = pd.DataFrame(columns=df_metrics.columns)
for i in range(len(df_metrics)):
    current_file_name = df_metrics.iloc[i]['file_name']
    current_file_name = current_file_name[:-4]
    current_patch = df_metrics.iloc[i]['patch']
    if current_patch == best_patches[current_file_name]:
        df_metrics_filtered = pd.concat([df_metrics_filtered, df_metrics.iloc[[i]]])

df_m_temp = df_metrics_filtered
hr_diff = df_m_temp['hr_diff']
hr_diff = hr_diff[hr_diff != "na"]
if "," in hr_diff.values:
    hr_diff = hr_diff.apply(lambda j: float(j.replace(",", ".")))
hr_diff_val = hr_diff.mean()
hr_diff_std = hr_diff.std()
df_m_temp = df_m_temp[df_m_temp['hr_bcg'] != "na"]
df_m_temp = df_m_temp[df_m_temp['hr_ecg'] != "na"]
if "," in df_m_temp['hr_bcg'].values:
    df_m_temp['hr_bcg'] = df_m_temp['hr_bcg'].apply(lambda j: float(j.replace(",", ".")))
if "," in df_m_temp['hr_ecg'].values:
    df_m_temp['hr_ecg'] = df_m_temp['hr_ecg'].apply(lambda j: float(j.replace(",", ".")))
hr_bcg = str(round(df_m_temp['hr_bcg'].mean(), 1)) + "±" + str(round(df_m_temp['hr_bcg'].std(), 1))
hr_ecg = str(round(df_m_temp['hr_ecg'].mean(), 1)) + "±" + str(round(df_m_temp['hr_ecg'].std(), 1))
hr_diff_str = str(round(hr_diff_val, 1)) + "±" + str(round(hr_diff_std, 1))
if len(df_m_temp) == 0:
    p_value = "na"
else:
    t_test_res = pg.ttest(df_m_temp['hr_bcg'], df_m_temp['hr_ecg'], paired=True, alternative='two-sided')
    p_value = t_test_res['p-val'].values[0]
    p_value = round(p_value, 3)

precision = df_m_temp['precision'].mean()
precision = str(round(precision, 1)) + "±" + str(round(df_m_temp['precision'].std(), 1))
recall = df_m_temp['recall'].mean()
recall = str(round(recall, 1)) + "±" + str(round(df_m_temp['recall'].std(), 1))
f1_score = df_m_temp['f1-score'].mean()
f1_score = str(round(f1_score / 100, 3)) + "±" + str(round(df_m_temp['f1-score'].std() / 100, 3))

hr_bcg_sdnn = df_m_temp['hr_bcg_sdnn']
hr_bcg_sdnn = hr_bcg_sdnn[hr_bcg_sdnn != "na"]
if "," in hr_bcg_sdnn.values:
    hr_bcg_sdnn = hr_bcg_sdnn.apply(lambda j: float(j.replace(",", ".")))
hr_bcg_sdnn1 = hr_bcg_sdnn.mean()
hr_bcg_sdnn1 = str(round(hr_bcg_sdnn1, 1)) + "±" + str(round(hr_bcg_sdnn.std(), 1))
hr_ecg_sdnn = df_m_temp['hr_ecg_sdnn']
hr_ecg_sdnn = hr_ecg_sdnn[hr_ecg_sdnn != "na"]
if "," in hr_ecg_sdnn.values:
    hr_ecg_sdnn = hr_ecg_sdnn.apply(lambda j: float(j.replace(",", ".")))
hr_ecg_sdnn1 = hr_ecg_sdnn.mean()
hr_ecg_sdnn1 = str(round(hr_ecg_sdnn1, 1)) + "±" + str(round(hr_ecg_sdnn.std(), 1))

x = df_m_temp['hr_ecg'].values
if len(x) == 0:
    print("no values")
else:
    print(ks_1samp(x, stats.norm.cdf, alternative='two-sided'))

# noinspection PyProtectedMember
df_metrics_averaged = df_metrics_averaged._append({'hr_diff': hr_diff_val, 'hr_diff_std': hr_diff_std,
                                                   'hr_bcg': hr_bcg, 'hr_ecg': hr_ecg,
                                                   'hr_diff_str': hr_diff_str, 'p-value': p_value,
                                                   'precision': precision, 'recall': recall,
                                                   'f1-score': f1_score, 'hr_bcg_sdnn': hr_bcg_sdnn1,
                                                   'hr_ecg_sdnn': hr_ecg_sdnn1},
                                                  ignore_index=True)
print(df_metrics_averaged)
df_metrics_averaged.to_csv(
    "output/detection/derived/metrics_averaged_" + iteration + "_" + dataset + ".csv",
    index=False)
