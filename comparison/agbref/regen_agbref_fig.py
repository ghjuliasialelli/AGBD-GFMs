"""
Regenerate the AGBref scatter+binned figure (manuscript Fig. 8 / imgs/agbref.png) from the CURRENT
per-plot CSV (All config, 666 plots). Replicates comparison.py L696-819 exactly (colours, bins,
metric boxes, dpi). Pure numpy+matplotlib. Writes plots/comparison_combined_regen.png.
"""
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

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
# row 0 scatter
for ci, (pred, ckey, ttl) in enumerate(col_configs):
    ax = axes[0, ci]
    valid = ~np.isnan(pred) & ~np.isnan(agbref)
    p, t = pred[valid], agbref[valid]
    ax.scatter(t, p, alpha=0.5, s=10, color=COLORS[ckey])
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_title(ttl, fontsize=14)
    bias = np.mean(p - t); rmse = np.sqrt(np.mean((p - t) ** 2)); mae = np.mean(np.abs(p - t))
    corr = np.corrcoef(p, t)[0, 1]
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
    ax.text(0.05, 0.95, f"r={corr:.3f}\nR²={r2:.3f}\nRMSE={rmse:.1f}\nME={bias:.1f}\nMAE={mae:.1f}",
            transform=ax.transAxes, va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    print(f"{ttl}: r={corr:.3f} R2={r2:.3f} RMSE={rmse:.1f} ME={bias:.1f} MAE={mae:.1f}")
# row 1 binned
for ci, (pred, ckey, ttl) in enumerate(col_configs):
    ax = axes[1, ci]
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
legend_labels = ["< 10", "10–20", "20–50", "50–100", "100–200", "> 200"]
for s, lab in zip(SIZE_VALUES, legend_labels):
    axes[1, 0].scatter([], [], s=s, c="gray", label=lab)
axes[1, 0].legend(title="#/bin", loc="upper left", fontsize=11, title_fontsize=12,
                  framealpha=0.8, labelspacing=1.0)
for ax in axes[1, :]:
    ax.set_xlabel("AGBRef (t/ha)", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("Predicted AGB (t/ha)", fontsize=12)
for ax in axes.flatten():
    ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
out = "plots/comparison_combined_regen.png"
plt.savefig(out, dpi=1200, bbox_inches="tight")
print("saved", out)
