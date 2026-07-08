import json
import subprocess
import os
from multiprocessing import Process

def run_config(config, config_path):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    subprocess.run(["python", "scripts/run.py", "--config", config_path])

import os

PROJECT_ROOT = "/home/arumota_pupils/Josue/2EMBC_Extention/project2.0_mmd"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

with open(CONFIG_PATH, "r") as f:
    base_config = json.load(f)

CONFIG_DIR = "/home/arumota_pupils/Josue/2EMBC_Extention/project2.0_mmd/config/sweep_configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# Sweep over different w_mmd values
w_mmd_values = [0.1, 1, 10, 20, 30, 50]
processes = []

for w_mmd in w_mmd_values:
    config = base_config.copy()
    config["w_mmd"] = w_mmd
    config["model_name"] = f"test_test_test_test{w_mmd}"
    config["csvname"] = f"results_wmmd_{w_mmd}"

    config_path = os.path.join(CONFIG_DIR, f"config_wmmd_{w_mmd}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    p = Process(target=run_config, args=(config, config_path))
    p.start()
    processes.append(p)

# Optional: wait for all to finish
for p in processes:
    p.join()