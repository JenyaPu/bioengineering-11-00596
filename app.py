import subprocess


def run_experiment_script():
    subprocess.run(["python", "modules/experiment/experiment.py"], check=True)
    subprocess.run(["python", "modules/experiment/calc_metrics_cebsdb.py"], check=True)
    subprocess.run(["python", "modules/experiment/calc_metrics_experiment.py"], check=True)
    subprocess.run(["python", "modules/experiment/calc_metrics_experiment_per_subject.py"], check=True)
    subprocess.run(["python", "modules/experiment/exp_instant_hr.py"], check=True)


run_experiment_script()
