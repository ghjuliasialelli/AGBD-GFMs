#!/usr/bin/env python3
"""
AEF advantage over AGBD features as a function of the test sample's distance to the nearest AEF
pre-training location (reviewer leakage check, revision round 2).

Reads the binned, paired statistics written by experiments/aef_leakage/analyse.py (AEF nico_film
59620113-1 vs AGBD features 59620098-1, same 2,807,977 no-overlap test rows; 95% CIs from a
1000x tile-cluster bootstrap). Leakage would show as an advantage that grows near the patches.

Plots the relative advantage, 1 - RMSE_AEF / RMSE_AGBD (%): far samples have lower biomass, so
the absolute gap (Mg/ha) would shrink with distance for scale reasons alone.

Run (from the repo root, `run` env):
    python plotting/helper/plot_aef_leakage_distance.py
"""

###################################################################################################
# Imports

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###################################################################################################
# Constants

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "experiments", "aef_leakage", "data", "binned_gap.csv")
OUT = os.path.join(ROOT, "plotting", "img", "aef_leakage_distance")

# Paul Tol 'muted': AGBD #332288, AEF #117733, AEF+ #44AA99, other GFM #88CCEE, ESA CCI #CC6677
C_AEF, C_AXIS = "#117733", "0.3"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_AXIS, "pdf.fonttype": 42, "savefig.dpi": 300,
})

###################################################################################################
# Helpers

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Plot AEF advantage vs distance to AEF pre-training locations.")
    parser.add_argument("--csv", default = CSV, help = "Binned statistics from experiments/aef_leakage/analyse.py.")
    parser.add_argument("--group", default = "all", help = "Row group in the CSV (e.g. 'all', 'region_cla=4').")
    parser.add_argument("--out", default = OUT, help = "Output path without extension.")
    args = parser.parse_args()
    return args.csv, args.group, args.out


def load(csv, group):
    df = pd.read_csv(csv)
    df = df[df["group"] == group].copy()
    assert len(df) > 0, f"no rows for group {group!r} in {csv}"
    lo_hi = df["bin_km"].str.split("-", expand = True).astype(float)
    df["lo"], df["hi"] = lo_hi[0], lo_hi[1]
    df["x"] = np.sqrt(df["lo"] * df["hi"])               # geometric bin centre (log axis)
    return df.sort_values("lo")


def panel(ax, df, y, lo, hi, ylabel):
    ax.errorbar(df["x"], y, yerr = [y - lo, hi - y], xerr = [df["x"] - df["lo"], df["hi"] - df["x"]],
                fmt = "o", ms = 4.5, mfc = C_AEF, mec = "white", mew = 0.6, ecolor = C_AEF,
                elinewidth = 0.9, capsize = 0, zorder = 3)
    ax.set_xscale("log")
    ax.set_xlim(0.58, 22)
    ax.set_xticks([0.64, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.64", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.grid(axis = "y", color = "0.9", lw = 0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Distance to nearest AEF pre-training location (km)")
    ax.set_ylabel(ylabel)

###################################################################################################
# Code execution

def main():
    csv, group, out = parse_arguments()
    df = load(csv, group)

    fig, ax = plt.subplots(figsize = (3.6, 2.8), layout = "constrained")
    rel, rel_lo, rel_hi = (100 * (1 - df[c]) for c in ("ratio", "ratio_hi", "ratio_lo"))
    panel(ax, df, rel, rel_lo, rel_hi, "Relative AEF advantage (%)")
    ax.set_ylim(bottom = 0)

    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    print("wrote", out + ".pdf", out + ".png")


if __name__ == "__main__":
    main()
