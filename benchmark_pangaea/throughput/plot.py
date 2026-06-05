"""
Pareto front: RMSE vs Throughput for geospatial foundation models.
Fine-tuning cost (GPU-hours) encoded as bubble size OR color.

Usage:
    python pareto_plot.py --mode bubble   # bubble size = GPU-hours
    python pareto_plot.py --mode color    # color = GPU-hours
    python pareto_plot.py --mode combined # both size + color
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── Data (RMSE from "results (Lite)" column) ─────────────────────────────────
models = {
    #                         RMSE(Lite) Throughput  GPU-hours
    "CROMA":                 (66.574,    70.3,       15.5),
    "DOFA":                  (75.235,   147.6,        4.5),
    "GFM-Swin":              (78.523,   349.7,        2.8),
    "Prithvi":               (78.254,   152.2,       14.0),
    "RemoteCLIP":            (78.171,   415.4,       11.5),
    "SatLAS-Net":            (69.203,   452.7,        4.0),
    "ScaleMAE":              (87.393,    82.7,       10.5),
    "SpectralGPT":           (71.232,    38.1,      140.0),
    "SSL4EO MoCo":           (64.34,    204.1,       15.5),
    "TerraMind":             (74.504,   236.3,       16.0),
    "Prithvi2":              (66.431,   150.9,        7.0),
    "Supervised":            (66.52,     61.7,      0.333)
}

names      = list(models.keys())
rmse       = np.array([v[0] for v in models.values()])
throughput = np.array([v[1] for v in models.values()])
gpu_hours  = np.array([v[2] for v in models.values()])

# ── Pareto front (lower RMSE + higher throughput = better) ────────────────────
def pareto_mask(rmse, throughput):
    """Returns boolean mask: True if the point is Pareto-optimal."""
    n = len(rmse)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good on both and strictly better on one
            if throughput[j] >= throughput[i] and rmse[j] <= rmse[i]:
                if throughput[j] > throughput[i] or rmse[j] < rmse[i]:
                    is_pareto[i] = False
                    break
    return is_pareto

mask = pareto_mask(rmse, throughput)
pareto_idx = np.where(mask)[0]
pareto_idx = pareto_idx[np.argsort(throughput[pareto_idx])]


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plot(mode="bubble", show_pareto=False):
    fig, ax = plt.subplots(figsize=(10, 7))

    # -- Pareto front line --
    if show_pareto:
        ax.plot(
            throughput[pareto_idx], rmse[pareto_idx],
            "--", color="gray", alpha=0.5, linewidth=1.2, zorder=1,
            label="Pareto front",
        )


    # -- Scatter --
    if mode == "bubble":
        # Size encodes GPU-hours, uniform color
        sizes = 40 + (gpu_hours / gpu_hours.max()) * 800
        sc = ax.scatter(
            throughput, rmse, s=sizes,
            c=np.where(mask, "#2563eb", "#9ca3af"),
            alpha=0.6, edgecolors="white", linewidths=0.8, zorder=2,
        )
        # Size legend
        for hours_val in [5, 20, 140]:
            ax.scatter([], [], s=40 + (hours_val / gpu_hours.max()) * 800,
                       c="gray", alpha=0.5, edgecolors="white",
                       label=f"{hours_val} GPU-h")

    elif mode == "color":
        # Color encodes GPU-hours, uniform size.
        # Clip colorbar at 20 GPU-h so the non-outlier models use the full
        # color range; SpectralGPT (140 h) gets the "over" color.
        CLIP = 20
        cmap = cm.get_cmap("viridis").copy()
        cmap.set_over("#ff6b35")          # distinct "over-range" color
        norm = mcolors.LogNorm(vmin=gpu_hours.min(), vmax=CLIP)
        sc = ax.scatter(
            throughput, rmse, s=120,
            c=gpu_hours, cmap=cmap, norm=norm,
            edgecolors="white", linewidths=0.8, zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, extend="max")
        cbar.set_label("GPU-hours (>20 = outlier)", fontsize=11)
        # Explicit ticks at intuitive values within range
        tick_vals = [0.5, 1, 2, 5, 10, 20]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([str(v) for v in tick_vals])

    elif mode == "combined":
        # Both size + color
        sizes = 40 + (gpu_hours / gpu_hours.max()) * 800
        norm = mcolors.LogNorm(vmin=gpu_hours.min(), vmax=gpu_hours.max())
        cmap = cm.get_cmap("viridis")
        sc = ax.scatter(
            throughput, rmse, s=sizes,
            c=gpu_hours, cmap=cmap, norm=norm,
            edgecolors="white", linewidths=0.8, alpha=0.75, zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("GPU-hours", fontsize=11)
        for hours_val in [5, 20, 140]:
            ax.scatter([], [], s=40 + (hours_val / gpu_hours.max()) * 800,
                       c="gray", alpha=0.5, edgecolors="white",
                       label=f"{hours_val} GPU-h")

    # -- Labels --
    # Per-label (dx, dy) offsets in points to avoid overlaps.
    # Clusters needing care:
    #   • CROMA / Supervised  — nearly same RMSE, close throughput
    #   • Prithvi2 / DOFA / Prithvi — all ~throughput 150
    label_offsets = {
        "CROMA":       ( 1,   8),
        "Supervised":  (0,  -14),
        "SpectralGPT": (  0.1,   10),
        "ScaleMAE":    (  0,   10),
        "SSL4EO MoCo": (  0,   10),
        "TerraMind":   (  0,   10),
        "Prithvi2":    (  0,    8),
        "DOFA":        (  0,  10),
        "Prithvi":     ( 0,    8),
        "GFM-Swin":    (  0,   10),
        "RemoteCLIP":  (  0,  10),
        "SatLAS-Net":  (  0,   10),
    }
    for i, name in enumerate(names):
        dx, dy = label_offsets.get(name, (0, 10))
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        ax.annotate(
            name, (throughput[i], rmse[i]),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, fontsize=10, fontweight=500,
            color="#333",
        )

    ax.set_xlabel("Throughput (samples/s)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"pareto_{mode}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: pareto_{mode}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bubble", "color", "combined"], default="bubble")
    parser.add_argument("--pareto", action="store_true", help="Show Pareto front line")
    args = parser.parse_args()
    make_plot(args.mode, show_pareto=args.pareto)
