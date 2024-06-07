import os

# folder paths
folder_datasets = os.path.join("datasets", "processed")
folder_protocols = os.path.join('protocols', 'saved')
folder_export = os.path.join('protocols', 'export')

# lists
default_patches = ['Patch0_x', 'Patch0_y', 'Patch0_z', 'Patch1_x', 'Patch1_y', 'Patch1_z']

# methods
methods_cleanup = ['none', 'neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012',
                   'pustozerov2024']
methods_detection = ['neurokit', 'pantompkins1985', 'hamilton2002', 'zong2003', 'martinez2004', 'christov2004',
                     'gamboa2008', 'elgendi2010', 'engzeemod2012', 'manikandan2012', 'kalidas2017', 'nabian2018',
                     'rodrigues2021', 'emrich2023', 'promac']

# df columns
df_metrics_columns = ['file_name', 'signal_interval', 'detection_method', 'cleanup_method', 'patch',
                      'r_peaks', 'j_peaks', 'hr_bcg', 'hr_ecg', 'hr_diff', 'hr_bcg_nn', 'hr_ecg_nn',
                      'hr_bcg_sdnn', 'hr_ecg_sdnn',
                      'precision', 'recall', 'f1-score', 'rj_std', 'elapsed_time', 'success']

# main dict
settings = {"default": {"patches": default_patches, "ecg_patch": "Sensor_ECG", "f": 1000},
            "cebsdb": {"patches": ["SCG"], "ecg_patch": "II", "f": 1000},
            "experiment": {"patches": default_patches, "ecg_patch": "Sensor_ECG", "f": 1000}}
