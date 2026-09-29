#!/usr/bin/env python3
"""
Adaptation cost vs accuracy of the GFMs on the Lite dataset.

x: GPU-hours until the best validation checkpoint (log scale).
y: test RMSE (inverted by default, so up = better, as in Fig. 1).
Marker area is proportional to the number of encoder parameters.
One point per LoRA run, labelled directly.

Edit the DATA block only. `None` = not available yet: the run is skipped.

Outputs: fig_cost.pdf (vector, for LaTeX) and fig_cost.png (preview).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

# =============================== DATA ===============================
# (label, GPU-hours to best checkpoint, test RMSE on Lite [Mg/ha],
#  encoder parameters [M], without decoder or LoRA adapters)
RUNS = [
    ("CROMA",           15.0, 62.83,  303.03),   # croma_optical
    ("DOFA",             8.0, 66.57,  111.31),   # dofa
    ("GFM-Swin",         2.0, 69.17,   86.73),   # gfmswin
    ("Prithvi",         14.5, 70.91,   86.39),   # prithvi
    ("RemoteCLIP",       4.5, 74.91,   87.45),   # remoteclip
    ("SatlasNet",        8.5, 64.96,   87.94),   # satlasnet_si
    ("ScaleMAE",         9.5, 76.85,  303.10),   # scalemae
    ("SpectralGPT",    110.0, 65.77,   85.40),   # spectralgpt
    ("SSL4EO-MoCo",      5.0, 63.05,   22.65),   # ssl4eo_moco
    ("TerraMind",       11.5, 64.66,    5.92),   # terramind_optical_tiny
    ("Prithvi v2",       4.5, 64.89,   86.24),   # prithvi2_100m
    ("CROMA + SAR",     32.0, 61.56,  655.77),   # croma_joint
    ("DOFA + SAR",      13.0, 65.58,  111.31),   # dofa_joint
    ("TerraMind + SAR",  2.5, 62.45,    6.02),   # terramind_tiny
]

YLIM = (59.5, 78.5)  # RMSE range
INVERT_Y = True      # True: lower RMSE at the top (up = better)
OUT = "fig_cost"

SIZE_K = 1.0         # marker area [pt^2] per million parameters
SIZE_REF = (10, 100, 300, 600)   # reference sizes shown in the legend [M]

# Label side relative to its bubble ("right" if not listed). The offset
# grows with the bubble radius. Re-check when points are added.
LABEL_SIDE = {
    "TerraMind + SAR": "above",
    "Prithvi v2":      "left",
    "SatlasNet":       "below",
    "SpectralGPT":     "left",
}

# =============================== STYLE ==============================
C_GFM, C_AXIS, C_LABEL = "#009E73", "0.3", "0.2"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C_AXIS,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def main():
    runs = [r for r in RUNS if None not in r[1:4]]
    best = min(r[2] for r in runs)
    xs = [r[1] for r in runs]
    xlim = (min(xs) / 1.45, max(xs) * 1.5)

    fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*(YLIM[::-1] if INVERT_Y else YLIM))
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200)
             if xlim[0] <= t <= xlim[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(color="0.9", lw=0.5)
    ax.set_axisbelow(True)

    # big bubbles first, so small ones stay visible on top
    for label, x, y, p in sorted(runs, key=lambda r: -r[3]):
        s = SIZE_K * p
        ax.scatter(x, y, s=s, color=C_GFM, alpha=0.75, edgecolor="white",
                   linewidth=0.6, zorder=3)
        r = s ** 0.5 / 2                          # radius in points
        side = LABEL_SIDE.get(label, "right")
        dx, dy, ha = {"right": (r + 3, 0, "left"),
                      "left": (-r - 3, 0, "right"),
                      "above": (0, r + 5, "center"),
                      "below": (0, -r - 5, "center")}[side]
        ax.annotate(label, (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=6.8, color=C_LABEL,
                    fontweight="bold" if y == best else "normal")

    # size legend: reference bubbles drawn with the same scale
    handles = [ax.scatter([], [], s=SIZE_K * v, color=C_GFM, alpha=0.75,
                          edgecolor="white", linewidth=0.6, label=f"{v:g}M")
               for v in SIZE_REF]
    ax.legend(handles=handles, title="Encoder\nparameters",
              loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              labelspacing=2.3, borderpad=0.8, handletextpad=1.0,
              fontsize=6.8, title_fontsize=7, scatterpoints=1,
              alignment="left")

    ax.set_xlabel("GPU-hours to best checkpoint")
    ax.set_ylabel("Test RMSE [Mg/ha] (↓)")
    if INVERT_Y:
        # small "better" arrow at the top of the y-label column
        ax.yaxis.set_label_coords(-0.1, 0.4)
        trans = ax.transAxes + ScaledTranslation(0, 0, fig.dpi_scale_trans)
        ax.annotate("better", xy=(-0.1, 1.0), xycoords=trans,
                    xytext=(-0.1, 0.8), textcoords=trans, rotation=90,
                    ha="center", va="top", fontsize=7.5, color=C_AXIS,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=0.9,
                                    mutation_scale=8, shrinkA=2, shrinkB=0,
                                    relpos=(0.5, 1)))

    fig.savefig(OUT + ".png")
    print("wrote", OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()