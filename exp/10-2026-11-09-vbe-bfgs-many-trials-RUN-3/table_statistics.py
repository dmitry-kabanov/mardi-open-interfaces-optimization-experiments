import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exact import compute_viscous_burgers_with_ic_and_periodic_bc_spectral
from scipy import stats

OUTPUT_DIR = Path("_output")
ASSETS_DIR = Path("_assets")


def process_single_trial(trialdir):
    results = []
    results_directories = os.listdir(trialdir)
    results_directories.sort()

    for d in results_directories:
        dirname = os.path.join(trialdir, d)
        if os.path.islink(dirname):
            continue
        if dirname.endswith(".inprogress"):
            print("Not finished, exiting")
            return

        config_filename = os.path.join(dirname, "config.json")
        log_filename = os.path.join(dirname, "log.log")
        res = {}
        with open(config_filename, "r") as fh:
            config = json.load(fh)
            impl = config["impl"]
            gtol = config["gtol"]
            split_key = config["split_key"]
            if "linesearch" in config and config["linesearch"] is not None:
                impl = impl + "-" + config["linesearch"]
            res["impl"] = impl
            res["gtol"] = gtol
            res["split_key"] = "yes" if split_key else " no"

        with open(log_filename, "r") as fh:
            log_lines = fh.readlines()

        for lline_ in log_lines:
            line = lline_.strip()
            if res["impl"].strip().startswith("scipy_optimize"):
                if line.startswith("nit"):
                    res["niterations"] = int(line.split(":")[1])
                if line.startswith("nfev"):
                    res["nfuncevals"] = int(line.split(":")[1])
                # if line.startswith("njev"):
                #    res["ngradevals"] = int(line.split(":")[1])
                if line.startswith("|∇f"):
                    res["final-grad-norm"] = float(line.split("=")[1])
            elif res["impl"].strip().startswith("optim_jl"):
                if line.startswith("result.iterations"):
                    res["niterations"] = int((line.split("=")[1]))
                if line.startswith("result.f_calls"):
                    res["nfuncevals"] = int(line.split("=")[1])
                # if line.startswith("result.g_calls"):
                #    res["ngradevals"] = int(line.split("=")[1])
                if line.startswith("result.g_residual"):
                    res["final-grad-norm"] = float(line.split("=")[1])
            else:
                raise ValueError("Unknown implementation '" + res["impl"] + "'")
            if line.startswith("Final training loss"):
                res["final-loss"] = float(line.split(":")[1])
            if line.startswith("Elapsed time"):
                elapsed = line.split(":")[1].strip()
                res["elapsed, sec."] = float(elapsed.split(" ")[0])

        x, dx_exact = np.linspace(0, 2, 256, endpoint=True, retstep=True)
        u0_exact = 0.5 - 0.25 * np.sin(np.pi * x)
        nu_exact = 0.01 / np.pi
        tfinal_exact = 2.0
        exact_sol = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
            u0_exact, x, dx_exact, tfinal_exact, nu_exact
        )

        pred = np.load(os.path.join(dirname, "predictions.npy")).squeeze()
        assert len(pred) == len(exact_sol), (
            f"Prediction array has different size ({len(pred)}) "
            f"than the exact-solution array ({len(exact_sol)})"
        )

        res["error"] = np.sqrt(np.sum((pred - exact_sol) ** 2) / len(pred))

        # print(res)
        results.append(res)

    # Create DataFrame
    df = pd.DataFrame(results)

    df_true = df[df["split_key"] == "yes"].drop(columns=["split_key"])
    df_true_pivoted = df_true.set_index(["impl", "gtol"])

    return df_true_pivoted


def _custom_format(x, width=5, use_e_notation=True):
    if pd.isna(x):
        return x

    if isinstance(x, str):
        return x

    if x >= 1.0:
        return f"{x:{width}.1f}"
    else:
        if use_e_notation:
            return f"{x:.1e}"
        else:
            return f"{x:{width}.1f}"


def _custom_format_two_digits(x, width=6, use_e_notation=True):
    if pd.isna(x):
        return x

    if isinstance(x, str):
        return x

    if x >= 1.0:
        return f"{x:{width}.2f}"
    else:
        if use_e_notation:
            return f"{x:.2e}"
        else:
            return f"{x:{width}.2f}"


def _align(df_mean, df_sem, colname, sig_digits=2):
    values = []

    for mean, sem in zip(df_mean[colname], df_sem[colname]):
        # sem has the smaller magnitude so it determines the rounding
        sem_exp = math.floor(math.log10(abs(sem)))

        # 3. Find the exponent of the rounded mean (this becomes the shared exponent)
        mean_exp = math.floor(math.log10(abs(mean)))

        # 4. Scale both numbers down by the mean's exponent
        scaled_mean = mean / (10**mean_exp)
        scaled_sem = sem / (10**mean_exp)

        value = (
            f"{scaled_mean:.{sig_digits}f} \\pm {scaled_sem:.{sig_digits}f}e{mean_exp}"
        )
        values.append(value)

    return values


def main():
    trials_data = []
    for trialdir in OUTPUT_DIR.iterdir():
        df = process_single_trial(trialdir)
        if df is None:
            continue
        trials_data.append(df)
        del df

    combined_df = pd.concat(trials_data, keys=range(len(trials_data)))

    df_mean = combined_df.groupby(level=["impl", "gtol"], sort=False).mean()
    df_sem = combined_df.groupby(level=["impl", "gtol"], sort=False).sem()

    # Make some columns integer.
    df_mean["niterations"] = df_mean["niterations"].map("{:.0f}".format)
    df_mean["nfuncevals"] = df_mean["nfuncevals"].map("{:.0f}".format)

    df_sem["niterations"] = df_sem["niterations"].map("{:.0f}".format)
    df_sem["nfuncevals"] = df_sem["nfuncevals"].map("{:.0f}".format)

    df_mean["elapsed, sec."] = df_mean["elapsed, sec."].map(
        _custom_format, width=5, use_e_notation=False
    )

    df_sem["elapsed, sec."] = df_sem["elapsed, sec."].map(
        _custom_format, width=5, use_e_notation=False
    )

    df_mean_formatted = df_mean.map(_custom_format, width=5)
    df_sem_formatted = df_sem.map(_custom_format, width=4, use_e_notation=True)
    df_sem_formatted["elapsed, sec."] = df_sem["elapsed, sec."].map(
        _custom_format, width=4, use_e_notation=False
    )

    df_final = df_mean_formatted + " \\pm " + df_sem_formatted

    df_final["final-grad-norm"] = _align(
        df_mean, df_sem, "final-grad-norm", sig_digits=1
    )
    df_final["final-loss"] = _align(df_mean, df_sem, "final-loss", sig_digits=1)
    df_final["error"] = _align(df_mean, df_sem, "error", sig_digits=2)

    df_final.rename(
        columns=lambda x: "{" + x + "}",
        inplace=True,
    )
    df_final.index.names = [f"{{{name}}}" for name in df_final.index.names]

    print("--- Final stats")
    print(df_final)

    print(df_final.to_latex())

    grid = np.linspace(0, 1500, num=501)

    df_1 = combined_df.query("impl == 'scipy_optimize' and gtol == '1e-5'")
    df_2 = combined_df.query("impl == 'optim_jl-HagerZhang' and gtol == '1e-5'")
    df_3 = combined_df.query("impl == 'optim_jl-StrongWolfe' and gtol == '1e-5'")
    df_4 = combined_df.query("impl == 'optim_jl-BackTracking' and gtol == '1e-5'")

    kde_1 = stats.gaussian_kde(df_1["niterations"].to_numpy())
    kde_2 = stats.gaussian_kde(df_2["niterations"].to_numpy())
    kde_3 = stats.gaussian_kde(df_3["niterations"].to_numpy())
    kde_4 = stats.gaussian_kde(df_4["niterations"].to_numpy())

    niter_1 = df_1["niterations"].to_numpy()
    niter_2 = df_2["niterations"].to_numpy()
    niter_3 = df_3["niterations"].to_numpy()
    niter_4 = df_4["niterations"].to_numpy()

    hist_1, bins_1 = np.histogram(niter_1, bins=7, range=(300, 1400), density=True)
    hist_2, bins_2 = np.histogram(niter_2, bins=7, range=(300, 1400), density=True)
    hist_3, bins_3 = np.histogram(niter_3, bins=7, range=(300, 1400), density=True)
    hist_4, bins_4 = np.histogram(niter_4, bins=7, range=(300, 1400), density=True)

    plt.figure()
    plt.plot(grid, kde_1(grid), "-", label="SciPy")
    plt.plot(grid, kde_2(grid), "-", label="Optim.jl, HagerZhang")
    plt.plot(grid, kde_3(grid), "-", label="Optim.jl, StrongWolfe")
    plt.plot(grid, kde_4(grid), "-", label="Optim.jl, BackTracking")
    # plt.stairs(hist_1, bins_1, fill=True, alpha=1.0, label="SciPy")
    # plt.stairs(hist_2, bins_2, fill=True, alpha=0.8, label="Optim.jl, HagerZhang")
    # plt.stairs(hist_3, bins_3, fill=True, alpha=0.6, label="Optim.jl, StrongWolfe")
    # plt.stairs(
    #     hist_4,
    #     bins_4,
    #     fill=False,
    #     alpha=0.4,
    #     label="Optim.jl, BackTracking",
    # )
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    # plt.show()
    plt.savefig(ASSETS_DIR / "kde-niterations.pdf")


if __name__ == "__main__":
    main()
