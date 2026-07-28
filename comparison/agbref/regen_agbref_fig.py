"""
Regenerate the AGBref scatter+binned figure (manuscript Fig. 8 / imgs/agbref.png) from the CURRENT
per-plot CSV (All config, 666 plots). Replicates comparison.py L696-819 exactly (colours, bins,
metric boxes, dpi). Pure numpy+matplotlib. Writes plots/comparison_combined_regen.png.

    python regen_agbref_fig.py                 # full 2x2 figure (unchanged)
    python regen_agbref_fig.py --binned-only   # binned row only, grayscale, metric boxes kept
    python regen_agbref_fig.py --big-fonts     # same plots, all fonts scaled up
"""
import argparse
import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERPLOT = "results/perplot_notrain_skip_jap_mean_max500.csv"
rows = list(csv.DictReader(open(PERPLOT)))
agbref = np.array([float(r["agbref"]) for r in rows])
nico   = np.array([float(r["nico_mean"]) for r in rows])
cci    = np.array([float(r["cci_mean"]) for r in rows])
print(f"loaded {len(rows)} plots")

COLORS = {"nico": "#0084FF", "cci": "#C02BF2"}
BIN_EDGES = np.arange(0, 550, 25)
SIZE_THRESHOLDS = [10, 20, 50, 100, 200]
SIZE_VALUES = [10, 25, 50, 90, 140, 200]
def get_marker_size(c):
    for th, s in zip(SIZE_THRESHOLDS, SIZE_VALUES):
        if c < th: return s
    return SIZE_VALUES[-1]
lim = 400
col_configs = [(nico, "nico", "Ours"), (cci, "cci", "ESA CCI")]


def metric_text(p, t, ttl):
    """Metric box contents; also prints the numbers, as the full figure does."""
    bias = np.mean(p - t); rmse = np.sqrt(np.mean((p - t) ** 2)); mae = np.mean(np.abs(p - t))
    corr = np.corrcoef(p, t)[0, 1]
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
    print(f"{ttl}: r={corr:.3f} R2={r2:.3f} RMSE={rmse:.1f} ME={bias:.1f} MAE={mae:.1f}")
    return f"r={corr:.3f}\nR²={r2:.3f}\nRMSE={rmse:.1f}\nME={bias:.1f}\nMAE={mae:.1f}"


def make_figure(binned_only=False, font_scale=1.0):
    # Every font size is multiplied by `font_scale`, for when the figure has to
    # be shrunk in the manuscript.
    fs = lambda pts: pts * font_scale
    big = font_scale != 1.0

    if binned_only:
        # Binned row only, in grayscale, carrying the scatter row's metric boxes.
        fig, binned_axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
        scatter_axes = None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
        scatter_axes, binned_axes = axes[0, :], axes[1, :]

    # row 0 scatter
    if scatter_axes is not None:
        for ci, (pred, ckey, ttl) in enumerate(col_configs):
            ax = scatter_axes[ci]
            valid = ~np.isnan(pred) & ~np.isnan(agbref)
            p, t = pred[valid], agbref[valid]
            ax.scatter(t, p, alpha=0.5, s=10, color=COLORS[ckey])
            ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
            ax.set_title(ttl, fontsize=fs(14))
            ax.text(0.05, 0.95, metric_text(p, t, ttl),
                    transform=ax.transAxes, va="top", fontsize=fs(12),
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # row 1 binned
    for ci, (pred, ckey, ttl) in enumerate(col_configs):
        ax = binned_axes[ci]
        valid = ~np.isnan(pred) & ~np.isnan(agbref)
        p, t = pred[valid], agbref[valid]
        bc, bm, bs, bsz = [], [], [], []
        for j in range(len(BIN_EDGES) - 1):
            lo, hi = BIN_EDGES[j], BIN_EDGES[j + 1]
            m = (t >= lo) & (t < hi)
            if m.sum() == 0: continue
            bc.append((lo + hi) / 2); bm.append(np.mean(p[m])); bs.append(np.std(p[m])); bsz.append(m.sum())
        bc, bm, bs = np.array(bc), np.array(bm), np.array(bs)
        ms = [get_marker_size(s) for s in bsz]
        ax.errorbar(bc, bm, yerr=bs, fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=1, alpha=0.6)
        ax.scatter(bc, bm, s=ms, c="black", zorder=2)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        if binned_only:
            # The titles and metric boxes now have to live on this row. At the
            # default size the bottom right is free (all bins sit near the 1:1
            # line) and the legend owns the top left; enlarged text no longer
            # fits there, so the box moves to the top left and the legend moves
            # out of the panel (below).
            ax.set_title(ttl, fontsize=fs(14))
            pos = dict(x=0.05, y=0.95, ha="left", va="top") if big else \
                  dict(x=0.95, y=0.05, ha="right", va="bottom")
            ax.text(pos["x"], pos["y"], metric_text(p, t, ttl),
                    transform=ax.transAxes, ha=pos["ha"], va=pos["va"], fontsize=fs(12),
                    bbox=dict(boxstyle="round", facecolor="0.9", edgecolor="0.4", alpha=0.9))

    legend_labels = ["< 10", "10–20", "20–50", "50–100", "100–200", "> 200"]
    handles = [binned_axes[0].scatter([], [], s=s, c="gray", label=lab)
               for s, lab in zip(SIZE_VALUES, legend_labels)]
    if not big:
        binned_axes[0].legend(title="#/bin", loc="upper left", fontsize=fs(11),
                              title_fontsize=fs(12), framealpha=0.8, labelspacing=1.0)
    for ax in binned_axes:
        ax.set_xlabel("AGBRef (t/ha)", fontsize=fs(12))
    for ax in ([binned_axes[0]] if binned_only else [scatter_axes[0], binned_axes[0]]):
        ax.set_ylabel("Predicted AGB (t/ha)", fontsize=fs(12))
    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=fs(10))
    plt.tight_layout()

    if big:
        # Enlarged, an in-panel legend swallows the low-biomass bins: put it
        # under the panels in one row instead. Placed after tight_layout so the
        # frame can be pinned to the plotted area — left edge of the left
        # y-axis to right edge of the right panel, no wider.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        left, right = binned_axes[0].get_position().x0, binned_axes[-1].get_position().x1
        # Below the x tick labels and axis title, not just below the axes.
        bottom = min(ax.get_tightbbox(renderer).transformed(to_fig).y0 for ax in binned_axes)
        fig.legend(handles=handles, title="#/bin", loc="upper left",
                   bbox_to_anchor=(left, bottom - 0.02, right - left, 1e-3),
                   mode="expand", ncol=len(handles),
                   fontsize=fs(11), title_fontsize=fs(12), framealpha=0.8,
                   columnspacing=1.0, handletextpad=0.4, borderpad=0.5)

    suffix = "_bigfont" if font_scale != 1.0 else ""
    out = ("plots/comparison_binned_regen{}.png" if binned_only
           else "plots/comparison_combined_regen{}.png").format(suffix)
    plt.savefig(out, dpi=1200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binned-only", action="store_true",
                        help="Plot only the binned row, in grayscale, keeping the "
                             "metric boxes (r, R2, RMSE, ME, MAE) from the scatter row")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Enlarge every font size (for when the figure is shrunk)")
    parser.add_argument("--font-scale", type=float, default=1.6,
                        help="Font multiplier used by --big-fonts (default: 1.6)")
    args = parser.parse_args()
    make_figure(binned_only=args.binned_only,
                font_scale=args.font_scale if args.big_fonts else 1.0)
