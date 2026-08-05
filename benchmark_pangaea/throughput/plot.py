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
    # S2 / TESSERA / AEF are the same model on different inputs, so they
    # share its throughput and fine-tuning cost.
    "S2":                    (66.52,     61.7,      0.333),
    "TESSERA":               (56.43,     61.7,      0.333),
    "AEF":                   (53.70,     61.7,      0.333),
}

# The same fcn_film architecture, trained on three different inputs.
FILM_VARIANTS = ("S2", "TESSERA", "AEF")

names      = list(models.keys())
rmse       = np.array([v[0] for v in models.values()])
throughput = np.array([v[1] for v in models.values()])
gpu_hours  = np.array([v[2] for v in models.values()])

# ── Broken y axis ─────────────────────────────────────────────────────────────
# AEF (53.7) and TESSERA (56.4) are ~8 RMSE clear of the field (next best:
# SSL4EO MoCo at 64.3), so the axis is cut and the empty band dropped. Both
# panels keep the same RMSE-per-inch — see height_ratios in make_plot().
Y_HI = (63.0, 89.5)          # every other model
Y_LO = (52.5, 58.0)          # AEF, TESSERA
LO_TICKS = [54, 56]

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

    # The y axis is broken: AEF/TESSERA sit ~8 RMSE below the next model, and
    # the empty band between them and SSL4EO MoCo would otherwise dominate the
    # figure. `height_ratios` matches the two spans exactly, so the RMSE scale
    # is identical above and below the break — only the gap is removed.
    fig, (ax, ax_lo) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[Y_HI[1] - Y_HI[0], Y_LO[1] - Y_LO[0]],
                         hspace=0.06),
    )
    axes = (ax, ax_lo)
    # Every point is drawn on both panels; the y limits decide which is visible.
    ax.set_ylim(*Y_HI)
    ax_lo.set_ylim(*Y_LO)
    ax_lo.set_yticks(LO_TICKS)
    for a in axes:
        a.tick_params(axis="both", labelsize=fs(10))

    # -- Pareto front line --
    if show_pareto:
        for a in axes:
            a.plot(
                throughput[pareto_idx], rmse[pareto_idx],
                "--", color="gray", alpha=0.5, linewidth=1.2, zorder=1,
                label="Pareto front" if a is ax else None,
            )


    # -- Scatter --
    sizes = 120
    if mode == "bubble":
        # Size encodes GPU-hours, uniform color
        sizes = 40 + (gpu_hours / gpu_hours.max()) * 800
        for a in axes:
            sc = a.scatter(
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
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_over("#ff6b35")          # distinct "over-range" color
        norm = mcolors.LogNorm(vmin=gpu_hours.min(), vmax=CLIP)
        for a in axes:
            sc = a.scatter(
                throughput, rmse, s=120,
                c=gpu_hours, cmap=cmap, norm=norm,
                edgecolors="white", linewidths=0.8, zorder=2,
            )
        cbar = fig.colorbar(sc, ax=list(axes), shrink=0.7, pad=0.02, extend="max")
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
        cmap = plt.get_cmap("viridis")
        for a in axes:
            sc = a.scatter(
                throughput, rmse, s=sizes,
                c=gpu_hours, cmap=cmap, norm=norm,
                edgecolors="white", linewidths=0.8, alpha=0.75, zorder=2,
            )
        cbar = fig.colorbar(sc, ax=list(axes), shrink=0.7, pad=0.02)
        cbar.set_label("GPU-hours", fontsize=fs(11))
        cbar.ax.tick_params(labelsize=fs(10))
        for hours_val in [5, 20, 140]:
            ax.scatter([], [], s=40 + (hours_val / gpu_hours.max()) * 800,
                       c="gray", alpha=0.5, edgecolors="white",
                       label=f"{hours_val} GPU-h")

    # -- Mark the three fcn_film runs --
    # S2/TESSERA/AEF are one architecture on three inputs; a cross inside the
    # marker flags them as a family (the caption says which input is which).
    film_idx = [names.index(n) for n in FILM_VARIANTS]
    cross_s = 0.5 * np.broadcast_to(sizes, (len(names),))[film_idx]
    for a in axes:
        a.scatter(throughput[film_idx], rmse[film_idx], marker="+", s=cross_s,
                  c="white", linewidths=1.2 * font_scale, zorder=3)

    # -- Labels --
    # Per-label (dx, dy) offsets in points to avoid overlaps.
    # Clusters needing care:
    #   • CROMA / S2  — nearly same RMSE, close throughput
    #   • Prithvi2 / DOFA / Prithvi — all ~throughput 150
    #   • S2 / TESSERA / AEF — identical throughput, stacked vertically
    label_offsets = {
        # Directly above its own marker (the right one of the near-coincident
        # CROMA/S2 pair), with S2 directly below the left one.
        "CROMA":       (  0,  10),
        "S2":          (  0, -18),
        "SpectralGPT": (  0.1,   10),
        "ScaleMAE":    (  0,   10),
        "TerraMind":   (  0,   10),
        "Prithvi2":    (  0,    8),
        "GFM-Swin":    (  0,   10),
        "RemoteCLIP":  ( 18,  10),
        "SatLAS-Net":  (  0,   10),
    }
    # Labels set beside their marker instead of above it: the ~throughput 150
    # pair reads better split left/right than stacked, and they are pinned so
    # the de-overlap pass never drags them back on top of each other.
    # TESSERA/AEF share S2's throughput, so they are labelled to the right
    # of their markers rather than stacked above them.
    side_labels = {"DOFA": "left", "Prithvi": "right", "SSL4EO MoCo": "right",
                   "TESSERA": "right", "AEF": "right"}
    # Hand-placed labels the de-overlap pass must leave exactly where they are.
    fixed_labels = {"CROMA", "S2"}

    marker_radii = np.broadcast_to(np.sqrt(np.atleast_1d(sizes) / np.pi), (len(names),))
    annotations, pinned, no_leader, is_fixed = [], [], [], []
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
        pinned.append(bool(side) or name in fixed_labels)
        # Side labels touch their marker, so they never get a leader line; the
        # hand-placed ones always do (see the second pass below), because their
        # markers have a near neighbour and the pairing must be spelled out.
        no_leader.append(bool(side) or name in fixed_labels)
        is_fixed.append(name in fixed_labels)
        # Each label lives on the panel its marker is visible in.
        host = ax_lo if rmse[i] < Y_LO[1] else ax
        annotations.append(host.annotate(
            name, (throughput[i], rmse[i]),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va=va, fontsize=fs(10), fontweight=500,
            color="#333",
        ))

    ax_lo.set_xlabel("Throughput (samples/s)", fontsize=fs(12))
    fig.supylabel("RMSE", fontsize=fs(12))
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(loc="upper right", fontsize=fs(9), framealpha=0.9) if handles else None
    for a in axes:
        a.grid(True, alpha=0.15)
        a.spines[["top", "right"]].set_visible(False)
    # Horizontal room for the labels to move into, scaled with the text. The y
    # limits are fixed by the break, so no y margin.
    ax.margins(x=0.05 + 0.04 * font_scale)

    # -- Break marks: hide the facing spines, draw the slanted cut --
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax_lo.spines["top"].set_visible(False)
    d = 0.4 * font_scale
    kw = dict(marker=[(-1, -d), (1, d)], markersize=10 * font_scale,
              linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **kw)
    ax_lo.plot([0], [1], transform=ax_lo.transAxes, **kw)

    plt.tight_layout()

    # -- Push labels apart (from each other, the markers and the legend) --
    # Run per panel: the two are disjoint in display space, so labels only ever
    # need to be resolved against the ones sharing their axes.
    fig.canvas.draw()                      # transforms must reflect tight_layout
    for a in axes:
        idx = [i for i, ann in enumerate(annotations) if ann.axes is a]
        if not idx:
            continue
        obstacles = marker_boxes(a, throughput, rmse, sizes)
        if legend is not None and a is ax:
            obstacles.append(legend.get_window_extent(fig.canvas.get_renderer()))
        sub = [annotations[i] for i in idx]
        if not separate_labels(fig, a, sub, obstacles,
                               pinned=[pinned[i] for i in idx]):
            print(f"Warning: labels still overlap after de-overlap pass ({a})")
        # Side labels sit beside their marker, so they never need a leader.
        sizes_i = np.broadcast_to(sizes, (len(names),))[idx]
        add_leader_lines(fig, a, sub, throughput[idx], rmse[idx], sizes_i,
                         gap_pt=6.0 * font_scale,
                         skip=[no_leader[i] for i in idx])
        # CROMA and S2 have near-coincident markers, so they get a
        # leader whatever the gap (gap_pt=0) to make the pairing unambiguous.
        add_leader_lines(fig, a, sub, throughput[idx], rmse[idx], sizes_i,
                         gap_pt=0.0,
                         skip=[not is_fixed[i] for i in idx])
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
