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
import matplotlib.transforms as mtransforms

# ── Data (RMSE from "results (Lite)" column) ─────────────────────────────────
models = {
    #                         RMSE(Lite) Throughput  GPU-hours
    "CROMA":                 (66.574,    70.3,       15.5),
    "DOFA":                  (75.235,   147.6,        4.5),
    "GFM-Swin":              (78.523,   349.7,        2.8),
    "Prithvi":               (76.168,   152.2,       14.0),
    "RemoteCLIP":            (78.171,   415.4,       11.5),
    "SatLAS-Net":            (69.203,   452.7,        4.0),
    "ScaleMAE":              (87.393,    82.7,       10.5),
    "SpectralGPT":           (71.232,    38.1,      140.0),
    "SSL4EO MoCo":           (64.34,    204.1,       15.5),
    "TerraMind":             (68.327,   236.3,       16.0),
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


# ── Label de-overlap ──────────────────────────────────────────────────────────
def _pad(bbox, p):
    """Bbox grown by `p` pixels on every side (Bbox.padded needs mpl>=3.8)."""
    return mtransforms.Bbox.from_extents(bbox.x0 - p, bbox.y0 - p,
                                         bbox.x1 + p, bbox.y1 + p)


def _push(a, b):
    """Displacement (dx, dy) px that separates box `a` from box `b`.

    Moves along whichever axis needs the smaller shift; (0, 0) if disjoint.
    """
    ox = min(a.x1, b.x1) - max(a.x0, b.x0)
    oy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ox <= 0 or oy <= 0:
        return np.zeros(2)
    if ox < oy:
        sign = 1.0 if (a.x0 + a.x1) >= (b.x0 + b.x1) else -1.0
        return np.array([sign * ox, 0.0])
    sign = 1.0 if (a.y0 + a.y1) >= (b.y0 + b.y1) else -1.0
    return np.array([0.0, sign * oy])


def separate_labels(fig, ax, annotations, obstacles, pinned=None,
                    n_iter=600, pad=3.0, damping=0.5):
    """Nudge annotation offsets until no label overlaps another label, a marker,
    or the legend, and every label stays inside the axes.

    Annotations must use textcoords="offset points"; their offsets are updated
    in place, so the result is independent of `font_scale` — bigger text simply
    gets pushed further. Annotations flagged in `pinned` keep the placement they
    were given: they still repel the others, but never move themselves.
    """
    fig.canvas.draw()                      # positions/renderer must be current
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    limit = _pad(ax.get_window_extent(renderer), -pad)
    pinned = [False] * len(annotations) if pinned is None else list(pinned)

    for _ in range(n_iter):
        boxes = [_pad(a.get_window_extent(renderer), pad) for a in annotations]
        shifts = [np.zeros(2) for _ in annotations]

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                d = _push(boxes[i], boxes[j])
                # A pinned label makes its neighbour absorb the whole shift.
                shifts[i] += d if pinned[j] else d / 2
                shifts[j] -= d if pinned[i] else d / 2
            for obs in obstacles:           # markers, legend: immovable
                shifts[i] += _push(boxes[i], obs)

        # Keep labels inside the axes.
        for i, box in enumerate(boxes):
            shifts[i][0] += max(0.0, limit.x0 - box.x0) - max(0.0, box.x1 - limit.x1)
            shifts[i][1] += max(0.0, limit.y0 - box.y0) - max(0.0, box.y1 - limit.y1)

        for i, is_pinned in enumerate(pinned):
            if is_pinned:
                shifts[i][:] = 0.0

        if max(np.abs(s).max() for s in shifts) < 0.5:
            return True
        for ann, shift in zip(annotations, shifts):
            x, y = ann.get_position()       # offset in points
            ann.set_position((x + damping * shift[0] / px_per_pt,
                              y + damping * shift[1] / px_per_pt))

    return False


def add_leader_lines(fig, ax, annotations, x, y, sizes, gap_pt=6.0,
                     color="#b0b0b0", skip=None):
    """Join a label to its marker with a hairline when de-overlapping pushed it
    far enough away that the pairing is no longer obvious."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    centres = ax.transData.transform(np.column_stack([x, y]))
    radii = np.broadcast_to(np.sqrt(np.atleast_1d(sizes) / np.pi) * px_per_pt,
                            (len(centres),))
    to_data = ax.transData.inverted()
    skip = [False] * len(annotations) if skip is None else skip

    for ann, centre, r, skip_it in zip(annotations, centres, radii, skip):
        if skip_it:
            continue
        box = ann.get_window_extent(renderer)
        # Closest point of the label box to the marker centre.
        near = np.array([np.clip(centre[0], box.x0, box.x1),
                         np.clip(centre[1], box.y0, box.y1)])
        d = np.hypot(*(near - centre))
        if d - r <= gap_pt * px_per_pt:
            continue                        # already visually attached
        direction = (near - centre) / d
        start = centre + direction * (r + 2 * px_per_pt)
        end = near - direction * 1.5 * px_per_pt
        (x0, y0), (x1, y1) = to_data.transform([start, end])
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.8, zorder=1)


def marker_boxes(ax, x, y, sizes, pad=2.0):
    """Bounding boxes (display px) of scatter markers of area `sizes` pt²."""
    centres = ax.transData.transform(np.column_stack([x, y]))
    radii = np.sqrt(np.atleast_1d(sizes) / np.pi) * ax.figure.dpi / 72.0
    radii = np.broadcast_to(radii, (len(centres),))
    return [mtransforms.Bbox.from_extents(cx - r - pad, cy - r - pad,
                                          cx + r + pad, cy + r + pad)
            for (cx, cy), r in zip(centres, radii)]


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plot(mode="bubble", show_pareto=False, font_scale=1.0):
    # All font sizes are multiplied by `font_scale` so the figure stays legible
    # when it is shrunk down in the paper.
    fs = lambda pts: pts * font_scale

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.tick_params(axis="both", labelsize=fs(10))

    # -- Pareto front line --
    if show_pareto:
        ax.plot(
            throughput[pareto_idx], rmse[pareto_idx],
            "--", color="gray", alpha=0.5, linewidth=1.2, zorder=1,
            label="Pareto front",
        )


    # -- Scatter --
    sizes = 120
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
        cbar.set_label("GPU-hours (>20 = outlier)", fontsize=fs(11))
        # Explicit ticks at intuitive values within range
        tick_vals = [0.5, 1, 2, 5, 10, 20]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([str(v) for v in tick_vals])
        cbar.ax.tick_params(labelsize=fs(10))

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
        cbar.set_label("GPU-hours", fontsize=fs(11))
        cbar.ax.tick_params(labelsize=fs(10))
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
        "TerraMind":   (  0,   10),
        "Prithvi2":    (  0,    8),
        "GFM-Swin":    (  0,   10),
        "RemoteCLIP":  (  0,  10),
        "SatLAS-Net":  (  0,   10),
    }
    # Labels set beside their marker instead of above it: the ~throughput 150
    # pair reads better split left/right than stacked, and they are pinned so
    # the de-overlap pass never drags them back on top of each other.
    side_labels = {"DOFA": "left", "Prithvi": "right", "SSL4EO MoCo": "right"}

    marker_radii = np.broadcast_to(np.sqrt(np.atleast_1d(sizes) / np.pi), (len(names),))
    annotations, pinned = [], []
    for i, name in enumerate(names):
        side = side_labels.get(name)
        if side:
            # Clear the marker edge, then a gap that grows with the text.
            gap = marker_radii[i] + fs(4)
            dx, dy = (gap if side == "right" else -gap), 0
            ha, va = ("left" if side == "right" else "right"), "center"
        else:
            dx, dy = label_offsets.get(name, (0, 10))
            # Offsets are in points, so scale them with the text to keep the
            # same visual gap between marker and label. They are only a
            # starting guess — separate_labels() resolves what still collides.
            dx, dy = fs(dx), fs(dy)
            ha, va = ("left" if dx > 0 else ("right" if dx < 0 else "center")), "baseline"
        pinned.append(bool(side))
        annotations.append(ax.annotate(
            name, (throughput[i], rmse[i]),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va=va, fontsize=fs(10), fontweight=500,
            color="#333",
        ))

    ax.set_xlabel("Throughput (samples/s)", fontsize=fs(12))
    ax.set_ylabel("RMSE", fontsize=fs(12))
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(loc="upper right", fontsize=fs(9), framealpha=0.9) if handles else None
    ax.grid(True, alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    # Room for the labels to move into, scaled with the text.
    ax.margins(x=0.05 + 0.04 * font_scale, y=0.05 + 0.04 * font_scale)

    plt.tight_layout()

    # -- Push labels apart (from each other, the markers and the legend) --
    fig.canvas.draw()                      # transforms must reflect tight_layout
    obstacles = marker_boxes(ax, throughput, rmse, sizes)
    if legend is not None:
        obstacles.append(legend.get_window_extent(fig.canvas.get_renderer()))
    if not separate_labels(fig, ax, annotations, obstacles, pinned=pinned):
        print("Warning: labels still overlap after de-overlap pass")
    # Pinned labels sit beside their marker, so they never need a leader.
    add_leader_lines(fig, ax, annotations, throughput, rmse, sizes,
                     gap_pt=6.0 * font_scale, skip=pinned)
    suffix = "_bigfont" if font_scale != 1.0 else ""
    out = f"pareto_{mode}{suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bubble", "color", "combined"], default="color")
    parser.add_argument("--pareto", action="store_true", help="Show Pareto front line")
    parser.add_argument("--big-fonts", action="store_true",
                        help="Enlarge every font size (for when the figure is shrunk)")
    parser.add_argument("--font-scale", type=float, default=1.6,
                        help="Font multiplier used by --big-fonts (default: 1.6)")
    args = parser.parse_args()
    make_plot(args.mode, show_pareto=args.pareto,
              font_scale=args.font_scale if args.big_fonts else 1.0)
