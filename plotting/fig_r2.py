#!/usr/bin/env python3
"""
Agreement between test RMSE and R^2 for the GFM runs on the Lite dataset.

One point per LoRA run; RMSE axis inverted so that right = better.

Outputs: fig_r2.pdf and fig_r2.png.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# (label, test RMSE [Mg/ha], test R^2), run IDs as comments
RUNS = [
    ("CROMA",           62.83, 0.60510),   # croma_optical
    ("DOFA",            66.57, 0.55663),   # dofa
    ("GFM-Swin",        69.17, 0.52132),   # gfmswin
    ("Prithvi",         70.91, 0.49686),   # prithvi
    ("RemoteCLIP",      74.91, 0.43866),   # remoteclip
    ("SatlasNet",       64.96, 0.57786),   # satlasnet_si
    ("ScaleMAE",        76.85, 0.40918),   # scalemae
    ("SpectralGPT",     65.77, 0.56724),   # spectralgpt
    ("SSL4EO-MoCo",     63.05, 0.60231),   # ssl4eo_moco
    ("TerraMind",       64.66, 0.58165),   # terramind_optical_tiny
    ("Prithvi v2",      64.89, 0.57865),   # prithvi2_100m
    ("CROMA + SAR",     61.56, 0.62077),   # croma_joint
    ("DOFA + SAR",      65.58, 0.56969),   # dofa_joint
    ("TerraMind + SAR", 62.45, 0.60979),   # terramind_tiny
]
XLIM = (60, 78.5)    # RMSE range, drawn inverted (right = better)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "fig_r2")

# Labels: the points sit on a near-straight diagonal, so labels alternate
# between the upper-left and lower-right side of it (in order of R^2), are
# pushed apart vertically so they never overlap, and get a short leader line.
LABEL_DX = 1.8       # horizontal distance from the point, in RMSE units
LABEL_GAP = 0.0105   # minimum vertical spacing between labels, in R^2 units

C_GFM, C_AXIS, C_LABEL = "#009E73", "0.3", "0.2"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_AXIS, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def main():
    fig, ax = plt.subplots(figsize=(3.6, 3.2), layout="constrained")
    ax.set_xlim(XLIM[1], XLIM[0])                     # inverted: right = better
    ax.grid(color="0.9", lw=0.5)
    ax.set_axisbelow(True)

    order = sorted(RUNS, key=lambda r: -r[2])
    last = {"left": np.inf, "right": np.inf}
    for i, (label, x_, y_) in enumerate(order):
        ax.plot(x_, y_, "o", ms=5, mfc=C_GFM, mec="white", mew=0.6, zorder=3)
        side = "left" if i % 2 == 0 else "right"
        y_lab = min(y_, last[side] - LABEL_GAP)
        last[side] = y_lab
        # inverted x: "left" on screen = larger RMSE
        x_lab = x_ + LABEL_DX if side == "left" else x_ - LABEL_DX
        ax.annotate(label, (x_, y_), xytext=(x_lab, y_lab),
                    ha="right" if side == "left" else "left", va="center",
                    fontsize=6.2, color=C_LABEL,
                    arrowprops=dict(arrowstyle="-", color="0.7", lw=0.5,
                                    shrinkA=1.5, shrinkB=3))

    ax.set_xlabel("Test RMSE [Mg/ha] (↓)")
    ax.set_ylabel(r"Test $R^2$ (↑)")
    fig.savefig(OUT + ".png")
    print("wrote", OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()
