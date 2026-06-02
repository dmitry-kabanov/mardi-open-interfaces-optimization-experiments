import json
import os
from pathlib import Path

OUTDIR = "exp/03-2026-05-22-vbe-bfgs"

results = []
results_directories = os.listdir(OUTDIR)
results_directories.sort()

for d in results_directories:
    if os.path.islink(d):
        continue

    config_filename = os.path.join(OUTDIR, d, "config.json")
    log_filename = os.path.join(OUTDIR, d, "log.log")
    res = {}
    with open(config_filename, "r") as fh:
        config = json.load(fh)
        impl = config["impl"]
        opt = config["opt"]
        gtol = config["gtol"]
        split_key = config["split_key"]
        res["impl"] = impl
        res["gtol"] = f"{gtol:.0e}"
        res["split_key"] = "yes" if split_key else " no"

    with open(log_filename, "r") as fh:
        log_lines = fh.readlines()

    for lline_ in log_lines:
        line = lline_.strip()
        if line.startswith("nit"):
            res["nit"] = line.split(":")[1]
        if line.startswith("nfev"):
            res["nfev"] = line.split(":")[1]
        if line.startswith("njev"):
            res["njev"] = line.split(":")[1]
        if line.startswith("Final training loss"):
            loss_value = float(line.split(":")[1])
            res["final_loss"] = f"{loss_value:.2e}"
        if line.startswith("Elapsed time"):
            elapsed = line.split(":")[1].strip()
            res["elapsed"] = elapsed.split(" ")[0]

    results.append(res)

for res in results:
    print(" & ".join(res.values()) + r" \\")
