#!/usr/bin/env python3
"""
Benchmark figure replacing Table 1 (AGB estimation with GFMs).

(a) Lite dataset, every model with the same UPerNet decoder. AEF and TESSERA
    are treated as frozen backbones, like the other GFMs.
(b) Full dataset, AEF embeddings with heads of increasing capacity.
Both panels: supervised fcn_film baselines as vertical reference lines.
The RMSE axis is inverted (low RMSE on the right), so further right = better.
Values are printed for every model that beats the supervised SOTA; the best
one per panel is in bold.

Edit the DATA block only. `None` = run still pending: that marker (or the whole
row, if it has no value yet) is skipped, so the figure stays valid while the
xx.xx entries are being filled in. Rows are re-sorted automatically.

Outputs: fig_benchmark.pdf (vector, for LaTeX) and fig_benchmark.png (preview).
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

# =============================== DATA ===============================
# Test RMSE (Mg/ha).

# --- (a) Lite dataset, UPerNet decoder -----------------------------------
#   frozen / lora : frozen or LoRA-adapted backbone
#   sar / mt      : LoRA with SAR / multi-temporal input added
MODELS_A = {
    "AEF":         dict(frozen=53.70),
    "TESSERA":     dict(frozen=57.12),
    "SSL4EO-MoCo": dict(frozen=64.34, lora=63.05),
    "Prithvi":     dict(frozen=76.17, lora=70.91, mt=71.98),
    "Prithvi v2":  dict(frozen=66.43, lora=64.89, mt=64.08),
    "CROMA":       dict(frozen=66.57, lora=62.83, sar=61.56),
    "TerraMind":   dict(frozen=68.33, lora=64.66, sar=62.45),
    "SatlasNet":   dict(frozen=69.20, lora=64.96, mt=62.136),
    "SpectralGPT": dict(frozen=71.23, lora=65.77),
    "DOFA":        dict(frozen=75.24, lora=66.57, sar=65.58),
    "RemoteCLIP":  dict(frozen=78.17, lora=74.91),
    "GFM-Swin":    dict(frozen=78.52, lora=69.17),
    "ScaleMAE":    dict(frozen=87.39, lora=76.85),
}
# Supervised fcn_film trained from scratch: all AGBD features / Sentinel-2 only
SUP_A = {"all": 59.02, "s2": 66.51}

# --- (b) Full dataset, AEF with heads of increasing capacity (3 runs) -----
EMB_B = {"LP": 56.67, "MLP": 52.22, "fcn_film": 50.92}
# 58.57 is quoted in Sec. 5.1 / 6.1 / 6.2 but is not in Table 1
SUP_B = {"all": 53.73, "s2": 58.57}

XLIM = (46, 90)      # RMSE range shown; drawn inverted (90 left, 46 right)
LEGEND = "right"     # "right" (two titled groups beside panel a) or "top"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "fig_benchmark")

# =============================== STYLE ==============================
# Okabe-Ito palette: supervised baselines = blue, all GFMs (incl. AEF) = green
C_SUP, C_GFM = "#0072B2", "#009E73"
C_TEXT, C_BEST, C_KEY, C_AXIS = "0.35", "0.05", "0.35", "0.3"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": C_AXIS,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,          # keep text editable in Inkscape / Illustrator
    "savefig.dpi": 300,
})

FCN = r"$\mathtt{fcn\_film}$"
HEAD_LABEL = {"LP": "LP", "MLP": "MLP", "fcn_film": FCN}
KEYS = ("frozen", "lora", "sar", "mt")


def marker_style(key, c):
    return {
        "frozen": dict(marker="o", ms=5.5, mfc="white", mec=c, mew=1.4),
        "lora":   dict(marker="o", ms=5.5, mfc=c, mec=c, mew=1.2),
        "sar":    dict(marker="D", ms=4.8, mfc=c, mec="white", mew=0.7),
        "mt":     dict(marker="s", ms=5.0, mfc=c, mec="white", mew=0.7),
    }[key]


Z = {"frozen": 3, "lora": 4, "sar": 5, "mt": 5}
Y_REF = -1.0         # row holding the supervised values and the band note


# =============================== ROWS ===============================
def _vals(d):
    return [d[k] for k in KEYS if d.get(k) is not None]


def rows_panel_a():
    rows = [(name, d) for name, d in MODELS_A.items() if _vals(d)]
    return sorted(rows, key=lambda r: min(_vals(r[1])))


def rows_panel_b():
    return [(f"AEF + {HEAD_LABEL[h]}", dict(frozen=v))
            for h, v in EMB_B.items() if v is not None]


# =============================== PLOT ===============================
def better_arrow(ax, offset_pt=22):
    """'better ---->' drawn under the right (low-RMSE) end of the x-axis."""
    trans = (blended_transform_factory(ax.transData, ax.transAxes)
             + ScaledTranslation(0, -offset_pt / 72, ax.figure.dpi_scale_trans))
    ax.annotate("better", xy=(XLIM[0] + 0.2, 0), xycoords=trans,
                xytext=(XLIM[0] + 6.5, 0), textcoords=trans,
                ha="right", va="center", fontsize=7.5, color=C_AXIS,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=0.9,
                                mutation_scale=8, shrinkA=2, shrinkB=0,
                                relpos=(1, 0.5)))


def draw_panel(ax, rows, sup, title, band_note=False):
    n = len(rows)
    ax.set_xlim(XLIM[1], XLIM[0])          # inverted: lower RMSE to the right
    ax.set_ylim(n - 0.4, Y_REF - 0.6)

    # supervised baselines: shaded region = beats supervised SOTA
    ax.axvspan(XLIM[0], sup["all"], color=C_SUP, alpha=0.07, lw=0, zorder=0)
    ax.axvline(sup["all"], color=C_SUP, lw=1.3, zorder=1)
    ax.axvline(sup["s2"], color=C_SUP, lw=1.0, ls=(0, (2, 2)), zorder=1)

    # reference row: supervised values + band note, separated from the models.
    # Offsets are in points, so "left of the line" holds whatever the axis
    # direction: the values sit in the white area, the note in the shaded one.
    kw = dict(color=C_SUP, va="center", ha="right", fontsize=6.5,
              fontweight="bold", textcoords="offset points", xytext=(-4, 0))
    ax.annotate(f"{sup['all']:.2f}", (sup["all"], Y_REF), **kw)
    ax.annotate(f"{sup['s2']:.2f}", (sup["s2"], Y_REF), **kw)
    if band_note:
        ax.annotate("better than supervised SOTA", (XLIM[0], Y_REF),
                    xytext=(-4, 0), textcoords="offset points",
                    color=C_SUP, fontsize=6.5, style="italic",
                    ha="right", va="center")
    ax.axhline(Y_REF + 0.5, color="0.8", lw=0.6, zorder=0)

    winners = [min(_vals(d)) for _, d in rows if min(_vals(d)) < sup["all"]]
    best = min(winners) if winners else None

    for y, (label, d) in enumerate(rows):
        vals = _vals(d)
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color=C_GFM, alpha=0.45,
                    lw=1.2, zorder=2, solid_capstyle="butt")
        for k in KEYS:
            if d.get(k) is not None:
                ax.plot(d[k], y, ls="", zorder=Z[k],
                        **marker_style(k, C_GFM))
        v = min(vals)
        if v < sup["all"]:
            is_best = v == best
            # printed on the better side, away from the SOTA line
            ax.annotate(f"{v:.2f}", (v, y), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.5,
                        fontweight="bold" if is_best else "normal",
                        color=C_BEST if is_best else C_TEXT)

    ax.set_yticks([Y_REF] + list(range(n)))
    ax.set_yticklabels([FCN] + [r[0] for r in rows])
    ax.get_yticklabels()[0].set_color(C_SUP)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold")
    #better_arrow(ax)


def marker_handles(labels):
    return [Line2D([], [], ls="", label=lab, **marker_style(k, C_KEY))
            for k, lab in zip(KEYS, labels)]


def line_handles(labels):
    return [Line2D([], [], color=C_SUP, lw=1.3, label=labels[0]),
            Line2D([], [], color=C_SUP, lw=1.0, ls=(0, (2, 2)),
                   label=labels[1])]


def legend_top(fig):
    h = marker_handles(["Frozen backbone", "LoRA", "LoRA + SAR",
                        "LoRA + multi-temporal"])
    h += line_handles([f"Supervised {FCN}, all AGBD features",
                       f"Supervised {FCN}, Sentinel-2 only"])
    # blanks so that markers fill column 1 and lines column 2
    h += [Line2D([], [], ls="", label=" ")] * 2
    fig.legend(handles=h, loc="outside upper center", ncol=2,
               frameon=False, handlelength=2.2, columnspacing=2.5)


def legend_right(fig, ax):
    """Two titled legends stacked to the right of panel (a), top-aligned."""
    kw = dict(loc="upper left", frameon=False, alignment="left",
              borderaxespad=0, handlelength=2.2,
              title_fontproperties=dict(weight="bold", size=7.5))
    leg1 = ax.legend(handles=marker_handles(
        ["Frozen", "LoRA", "LoRA + SAR", "LoRA + multi-temporal"]),
        title="Backbone", bbox_to_anchor=(1.03, 1.0), **kw)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=line_handles(
        ["All AGBD features", "Sentinel-2 only"]),
        title="Supervised " + FCN, bbox_to_anchor=(1.03, 0.5), **kw)
    # run the layout once, then tuck the second group right under the first
    fig.canvas.draw()
    gap_px = 10 * fig.dpi / 72
    y = ax.transAxes.inverted().transform(
        (0, leg1.get_window_extent().y0 - gap_px))[1]
    leg2.set_bbox_to_anchor((1.03, y), transform=ax.transAxes)


def main():
    rows_a, rows_b = rows_panel_a(), rows_panel_b()
    span_a, span_b = len(rows_a) + 1.2, len(rows_b) + 1.2
    row_h = 0.19
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, sharex=True,
        figsize=(6.5, row_h * (span_a + span_b)
                 + (1.9 if LEGEND == "top" else 1.2)),
        gridspec_kw=dict(height_ratios=[span_a, span_b]),
        layout="constrained",
    )
    draw_panel(ax_a, rows_a, SUP_A,
               "(a) Lite dataset (∼600k samples), UPerNet decoder",
               band_note=False)
    draw_panel(ax_b, rows_b, SUP_B,
               "(b) Full dataset (∼16M samples), heads of increasing "
               "capacity")
    ax_a.tick_params(labelbottom=True)
    ax_b.set_xlabel("Test RMSE [Mg/ha] (↓)")
    fig.get_layout_engine().set(h_pad=0.06)
    if LEGEND == "top":
        legend_top(fig)
    else:
        legend_right(fig, ax_a)

    fig.savefig(OUT + ".png")
    print("wrote", OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()