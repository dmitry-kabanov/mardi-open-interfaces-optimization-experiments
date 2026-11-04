#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from exact import compute_viscous_burgers_with_ic_and_periodic_bc_spectral

matplotlib.use("Agg")


OUTPUT_DIR = Path("_output")
ASSETS_DIR = Path("_assets")

LINESEARCH_HZ = "HagerZhang"
LINESEARCH_SW = "StrongWolfe"
LINESEARCH_BT = "BackTracking"

FIGSIZE = (2 * 6.90, 2 * 4.26)

# Color scheme and styles
exact_color = "#1C1C1C"
colors_by_tol = {
    1e-3: "#E76F51",  # Warm Coral
    1e-4: "#2A9D8F",  # Sage Teal
    1e-5: "#457B9D",  # Slate Blue
}

linestyles_by_tol = {
    1e-3: "--",
    1e-4: "-.",
    1e-5: ":",
}

labels_by_tol = {
    1e-3: r"PINN ($\mathrm{gtol}=10^{-3}$)",
    1e-4: r"PINN ($\mathrm{gtol}=10^{-4}$)",
    1e-5: r"PINN ($\mathrm{gtol}=10^{-5}$)",
}

# Tolerances to plot
tols = [1e-3, 1e-4, 1e-5]


def load_predictions(out_dir):
    """
    Finds directories in out_dir that have split_key=true and loads predictions.npy.
    Categorizes them by implementation (scipy_optimize vs optim_jl) and gtol.
    """
    scipy_preds = {}
    optim_hz_preds = {}
    optim_sw_preds = {}
    optim_bt_preds = {}

    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory '{out_dir}' does not exist.")

    for d in os.listdir(out_dir):
        dir_path = os.path.join(out_dir, d)
        if not os.path.isdir(dir_path) or os.path.islink(dir_path):
            continue

        config_path = os.path.join(dir_path, "config.json")
        pred_path = os.path.join(dir_path, "predictions.npy")

        if not os.path.exists(config_path) or not os.path.exists(pred_path):
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            continue

        impl = config.get("impl")
        split_key = config.get("split_key")
        gtol = config.get("gtol")
        linesearch = config.get("linesearch")

        # Check if split_key is true
        if split_key is True:
            try:
                # Load array and flatten to 1D
                pred_data = np.load(pred_path).squeeze()
                if impl == "scipy_optimize":
                    scipy_preds[gtol] = pred_data
                elif impl == "optim_jl" and linesearch == LINESEARCH_HZ:
                    optim_hz_preds[gtol] = pred_data
                elif impl == "optim_jl" and linesearch == LINESEARCH_SW:
                    optim_sw_preds[gtol] = pred_data
                elif impl == "optim_jl" and linesearch == LINESEARCH_BT:
                    optim_bt_preds[gtol] = pred_data
            except Exception as e:
                print(f"Warning: Failed to load predictions from {pred_path}: {e}")
                continue

    return scipy_preds, optim_hz_preds, optim_sw_preds, optim_bt_preds


def get_pred_by_tol(preds_dict, tol):
    """
    Helper to robustly find a prediction array in a dictionary by float tolerance key.
    """
    for k, v in preds_dict.items():
        if np.isclose(float(k), tol):
            return v
    return None


# Plotting helper
def plot_impl(ax, preds_dict, label, x_exact, exact_sol):
    # Plot exact solution
    ax.plot(
        x_exact,
        exact_sol,
        color=exact_color,
        linestyle="-",
        linewidth=2.0,
        label="Exact solution",
    )

    # Plot predictions for each gtol
    for tol in tols:
        pred_arr = get_pred_by_tol(preds_dict, tol)
        if pred_arr is not None:
            ax.plot(
                x_exact,
                pred_arr,
                color=colors_by_tol[tol],
                linestyle=linestyles_by_tol[tol],
                linewidth=1.8,
                label=labels_by_tol[tol],
            )
        else:
            print(f"Warning: No prediction found for tol={tol} in {label}")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#E5E5E5")

    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            boxstyle="round,pad=0.2",
        ),
    )

    # Clean look: remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#E5E5E5",
        framealpha=0.9,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trial")
    args = p.parse_args()
    TRIALS_DIR = OUTPUT_DIR / f"trial_{args.trial}"

    print("Loading predictions from output directory...")
    scipy_preds, optim_hz_preds, optim_sw_preds, optim_bt_preds = load_predictions(
        TRIALS_DIR
    )

    # 3. Compute exact solution of the Burgers' equation
    print("Computing exact solution...")
    x, dx_exact = np.linspace(0, 2, 256, endpoint=True, retstep=True)
    u0_exact = 0.5 - 0.25 * np.sin(np.pi * x)
    nu_exact = 0.01 / np.pi
    tfinal_exact = 2.0
    exact_sol = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
        u0_exact, x, dx_exact, tfinal_exact, nu_exact
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIGSIZE, sharex=True, sharey=True
    )

    # Plot top left: scipy_optimize
    plot_impl(ax1, scipy_preds, "A", x, exact_sol)
    ax1.set_ylabel(r"$u(t=2, x)$", fontsize=12, labelpad=8)

    # Plot top right: optim_jl HagerZhang
    plot_impl(ax2, optim_hz_preds, "B", x, exact_sol)

    # Plot bottom left: optim_jl StrongWolfe
    plot_impl(ax3, optim_sw_preds, "C", x, exact_sol)
    ax3.set_ylabel(r"$u(t=2, x)$", fontsize=12, labelpad=8)
    ax3.set_xlabel(r"$x$", fontsize=12, labelpad=8)

    # Plot bottom right: optim_jl BackTracking
    plot_impl(ax4, optim_bt_preds, "D", x, exact_sol)
    ax4.set_xlabel(r"$x$", fontsize=12, labelpad=8)

    plt.tight_layout()

    # 5. Save the figure as vbe-bfgs-predictions.pdf
    pdf_filename = ASSETS_DIR / f"vbe-bfgs-predictions-trial={args.trial}.pdf"
    plt.savefig(pdf_filename)
    print(f"Figure successfully saved to: {pdf_filename}")


if __name__ == "__main__":
    main()
