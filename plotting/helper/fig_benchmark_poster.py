#!/usr/bin/env python3
"""
Poster version of the benchmark figure (fig_benchmark.py): the Lite panel only.

Lite dataset, every GFM with the same UPerNet decoder. AEF and TESSERA are
frozen embeddings: their row shows the head progression LP -> MLP -> UPerNet,
the way GFM rows show frozen -> LoRA -> +SAR/MT. Compared with the paper
figure, it adds OlmoEarth and the embedding-concatenation rows
(AEF + OlmoEarth, AEF + TESSERA, OlmoEarth + TESSERA; fcn_film head only)
and drops the full-dataset panel. fig_benchmark.py and its outputs are left
untouched.
Supervised fcn_film baselines are drawn as vertical reference lines.
The RMSE axis is inverted (low RMSE on the right), so further right = better.
Values are printed for every model that beats the supervised SOTA; the best
one is in bold.

Edit the DATA block only. `None` = run still pending: that marker (or the whole
row, if it has no value yet) is skipped. Rows are re-sorted automatically.

Outputs: fig_benchmark_poster.pdf (vector) and fig_benchmark_poster.png (preview).

Run: python plotting/helper/fig_benchmark_poster.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

# =============================== DATA ===============================
# Test RMSE (Mg/ha).

# --- (a) Lite dataset, UPerNet decoder -----------------------------------
#   frozen / lora : frozen or LoRA-adapted backbone
#   sar / mt      : LoRA with SAR / multi-temporal input added
#   lp / mlp / film : linear probe / MLP / fcn_film head on frozen embeddings
#                   (AEF, TESSERA); their UPerNet result goes under `frozen`
MODELS_A = {
    "AEF":         dict(lp=57.34, mlp=53.91, frozen=53.71, film=53.70),
    "TESSERA":     dict(lp=66.86, mlp=59.79, frozen=57.12, film=56.43),
    # OlmoEarth v1.3 embeddings (128-d) and embedding concatenations,
    # fcn_film head only, single run each, Lite test (n=119,991, same labels
    # as the other files). RMSE recomputed from model/eval/results/
    # nico_film_<job>-1_..._2019-2020_lite.h5 (uploaded 2026-11-12):
    #   OlmoEarth 19819522, AEF+OlmoEarth 20004081,
    #   AEF+TESSERA 20420538, OlmoEarth+TESSERA 20420834
    # OlmoEarth LP 59.33 +- 0.02, MLP 54.14 +- 0.09 (user-reported 2026-11-12,
    # mean over runs; only means are drawn)
    "OlmoEarth":            dict(lp=59.33, mlp=54.14, film=53.68),
    # OlmoEarth Matryoshka-64 (first 64 of 128 dims, --olmo_dims 64), job
    # 19870583, fcn_film, Lite test; user-reported 2026-11-12
    "OlmoEarth (64-d)":     dict(film=53.94),
    "AEF + OlmoEarth":      dict(film=51.71),
    "AEF + TESSERA":        dict(film=53.09),
    "OlmoEarth + TESSERA":  dict(film=53.58),
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

XLIM = (46.5, 88.5)   # RMSE range shown; drawn inverted (low = right)
LEGEND = "right"     # "right" (two titled groups beside panel a) or "top"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", "fig_benchmark_poster")

# =============================== STYLE ==============================
# Okabe-Ito palette: supervised baselines = blue, all GFMs (incl. AEF) = green.
# (Paul Tol 'muted' tried 2026-10-07 and rejected: #88CCEE too light to read.)
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
# Encoding: fill = backbone (hollow frozen, filled LoRA), shape = decoder
# (triangle LP, square MLP, circle UPerNet, diamond fcn_film); plus / X =
# LoRA + SAR / multi-temporal input, both with UPerNet. MT is an X, not a
# square, so it cannot be read as "LoRA + MLP".
KEYS = ("lp", "mlp", "film", "frozen", "lora", "sar", "mt")


def marker_style(key, c):
    return {
        "lp":     dict(marker="^", ms=5.5, mfc="white", mec=c, mew=1.3),
        "mlp":    dict(marker="s", ms=4.8, mfc="white", mec=c, mew=1.3),
        "frozen": dict(marker="o", ms=5.5, mfc="white", mec=c, mew=1.4),
        "lora":   dict(marker="o", ms=5.5, mfc=c, mec=c, mew=1.2),
        "sar":    dict(marker="P", ms=6.0, mfc=c, mec="white", mew=0.7),
        "mt":     dict(marker="X", ms=6.0, mfc=c, mec="white", mew=0.7),
        "film":   dict(marker="D", ms=4.8, mfc="white", mec=c, mew=1.3),
    }[key]


Z = {"lp": 3, "mlp": 3, "film": 4, "frozen": 4, "lora": 4, "sar": 5, "mt": 5}
SUP_ROW_PT = 16.5   # pt below the x axis: the row of blue supervised values
OVERLAP = 1.0       # Mg/ha; markers closer than this are drawn see-through



# =============================== ROWS ===============================
def _vals(d):
    return [d[k] for k in KEYS if d.get(k) is not None]


def rows_panel_a():
    # one list, best first: no embedding/GFM split, since LP and MLP heads
    # could be put on any backbone
    rows = [(name, d) for name, d in MODELS_A.items() if _vals(d)]
    return sorted(rows, key=lambda r: min(_vals(r[1])))


# =============================== PLOT ===============================
# Inset magnifying adjacent rows whose markers sit on top of each other
# (OlmoEarth: MLP 54.14, fcn_film 53.68; AEF: MLP 53.91, UPerNet 53.71,
# fcn_film 53.70). 0.01 Mg/ha cannot be separated at any true scale, so the
# inset is not to scale: one line per row, in RMSE order (worse left, better
# right, like the panel), on ONE shared rank axis built from the union of the
# rows' values, so markers stay in value order across rows too. Consecutive
# values are spaced by their gap clipped to [ZOOM_GAP_MIN, 1] (exact ties
# share a position). The inset lines are in the same order as the zoomed
# rows, so they carry no row names. No values are printed in the inset: like
# every other row, each zoomed row's best value is printed in the main panel.
ZOOM_ROWS = ("OlmoEarth", "AEF")  # must be adjacent once the rows are sorted
ZOOM_KEYS = ("mlp", "frozen", "film")
ZOOM_BOX_X = (52.8, 54.8)     # grey box on the main rows: Mg/ha span ...
ZOOM_BOX_H = 0.5              # ... and padding in rows above/below; covers the markers
ZOOM_GAP_UNIT = 0.20          # Mg/ha drawn as one unit of spacing in the inset
ZOOM_GAP_MIN = 0.4            # near-ties stay apart; labels alternate, so they fit
ZOOM_ROW_SEP = 1.4            # inset row spacing, in inset data units
ZOOM_RIGHT_PAD = 0.5          # inset x room right of the last marker
FIG_W = 7.0                   # inches; wider than the paper figure (6.5) for the inset
# Inset in axes fraction of the panel, in the legend column (legends are
# stacked below it): (x0, width, height); y is set so the inset is centred on
# the zoomed rows, which makes the two connectors fan out at the same angle
ZOOM_BOUNDS = (1.04, 0.22, 0.13)


def draw_zoom(ax, rows):
    names = [r[0] for r in rows]
    idx = [names.index(r) for r in ZOOM_ROWS if r in names]
    if not idx:
        return None, 1.0
    idx.sort()
    assert idx == list(range(idx[0], idx[-1] + 1)), "zoomed rows not adjacent"
    y_top, y_bot = idx[0], idx[-1]
    # shared rank axis over the union of zoomed values, worse (left) -> better
    vals = sorted({rows[i][1][k] for i in idx for k in ZOOM_KEYS
                   if rows[i][1].get(k) is not None}, reverse=True)
    pos = {vals[0]: 0.0}
    for a, b in zip(vals, vals[1:]):
        pos[b] = pos[a] + min(1.0, max(ZOOM_GAP_MIN, (a - b) / ZOOM_GAP_UNIT))
    x0, w, h = ZOOM_BOUNDS
    yc = ax.transAxes.inverted().transform(
        ax.transData.transform((0, (y_top + y_bot) / 2)))[1]
    axin = ax.inset_axes((x0, yc - h / 2, w, h))
    xmax = max(pos.values())
    for j, i in enumerate(idx):
        d = rows[i][1]
        yi = -j * ZOOM_ROW_SEP
        keys = sorted((k for k in ZOOM_KEYS if d.get(k) is not None),
                      key=lambda k: -d[k])
        xs = [pos[d[k]] for k in keys]
        axin.plot(xs, [yi] * len(xs), color=C_GFM, alpha=0.45,
                  lw=1.2, zorder=2, solid_capstyle="butt")
        for x, k in zip(xs, keys):
            axin.plot(x, yi, ls="", zorder=3, **marker_style(k, C_GFM))
    axin.set_xlim(-0.5, xmax + ZOOM_RIGHT_PAD)
    axin.set_ylim(-(len(idx) - 1) * ZOOM_ROW_SEP - 0.6, 0.6)
    # no ticks: the shapes name the heads (Decoder legend); values are in the panel
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_facecolor("white")
    for side in ("top", "right", "left", "bottom"):
        axin.spines[side].set_visible(True)
        axin.spines[side].set_color("0.6")
        axin.spines[side].set_linewidth(0.6)
    # outline the magnified region on the main rows, and join its right-hand
    # corners to the inset's left-hand ones (the built-in connectors pick
    # corners that cross the rows' value labels)
    lo, hi = ZOOM_BOX_X
    b_top, b_bot = y_top - ZOOM_BOX_H, y_bot + ZOOM_BOX_H
    _, connectors = ax.indicate_inset(
        (lo, b_top, hi - lo, b_bot - b_top),
        inset_ax=axin, edgecolor="0.5", lw=0.6, alpha=1)
    for c in connectors:
        c.set_visible(False)
    # x inverted: lo is the box's right edge; y inverted: b_top is its top
    for y_box, y_in in ((b_top, 1), (b_bot, 0)):
        ax.add_artist(ConnectionPatch(
            xyA=(lo, y_box), coordsA=ax.transData,
            xyB=(0, y_in), coordsB=axin.transAxes, color="0.5", lw=0.6,
            ls=(0, (3, 2))))
    return axin, yc - h / 2


def better_arrow(ax, offset_pt=22):
    """'better ---->' drawn under the right (low-RMSE) end of the x-axis."""
    trans = (blended_transform_factory(ax.transData, ax.transAxes)
             + ScaledTranslation(0, -offset_pt / 72, ax.figure.dpi_scale_trans))
    lo = min(ax.get_xlim())
    ax.annotate("better", xy=(lo + 0.2, 0), xycoords=trans,
                xytext=(lo + 6.5, 0), textcoords=trans,
                ha="right", va="center", fontsize=7.5, color=C_AXIS,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=0.9,
                                mutation_scale=8, shrinkA=2, shrinkB=0,
                                relpos=(1, 0.5)))


def draw_panel(ax, rows, sup, title, xlim, zoom_rows=()):
    n = len(rows)
    ax.set_xlim(xlim[1], xlim[0])          # inverted: lower RMSE to the right
    ax.set_ylim(n - 0.4, -0.6)

    # supervised baselines: shaded region = beats supervised SOTA
    ax.axvspan(xlim[0], sup["all"], color=C_SUP, alpha=0.07, lw=0, zorder=0)
    ax.axvline(sup["all"], color=C_SUP, lw=1.3, zorder=1)
    ax.axvline(sup["s2"], color=C_SUP, lw=1.0, ls=(0, (2, 2)), zorder=1)

    # supervised values: a blue tick where each line meets the x axis, and
    # the value in a second row under the grey tick labels (left untouched)
    tick = mpl.rcParams["xtick.major.size"]
    for v, lw in ((sup["all"], 1.3), (sup["s2"], 1.0)):
        ax.annotate("", (v, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -tick), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=C_SUP, lw=lw,
                                    shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        ax.annotate(f"{v:.2f}", (v, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -SUP_ROW_PT), textcoords="offset points",
                    ha="center", va="top", color=C_SUP, fontweight="bold",
                    fontsize=6, annotation_clip=False)

    winners = [min(_vals(d)) for _, d in rows if min(_vals(d)) < sup["all"]]
    best = min(winners) if winners else None

    for y, (label, d) in enumerate(rows):
        vals = _vals(d)
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color=C_GFM, alpha=0.45,
                    lw=1.2, zorder=2, solid_capstyle="butt")
        for k in KEYS:
            if d.get(k) is None:
                continue
            st = marker_style(k, C_GFM)
            # head markers on an embedding row closer than OVERLAP (AEF MLP
            # 53.91 vs UPerNet 53.70) would hide each other: see-through
            # outlines. Only hollow head markers: a filled LoRA marker made
            # hollow would read as "frozen".
            if "lp" in d and k in HEAD_KEYS and any(
                    abs(d[k] - d[o]) < OVERLAP for o in HEAD_KEYS
                    if o != k and d.get(o) is not None):
                st.update(mfc="none", alpha=0.75)
                # white underlay below every outline but above the row line,
                # so the line does not show through the see-through markers
                ax.plot(d[k], y, ls="", zorder=2.5,
                        **dict(st, mfc="white", mec="white", alpha=1))
            ax.plot(d[k], y, ls="", zorder=Z[k], **st)
        v = min(vals)
        if v < sup["all"]:
            is_best = v == best
            # printed on the better side, away from the SOTA line
            # the zoomed rows' labels clear the grey zoom box around them
            pad = 11 if label in zoom_rows else 6
            ax.annotate(f"{v:.2f}", (v, y), xytext=(pad, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.5,
                        fontweight="bold" if is_best else "normal",
                        color=C_BEST if is_best else C_TEXT)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold")
    #better_arrow(ax)


def marker_handles(keys, labels):
    return [Line2D([], [], ls="", label=lab, **marker_style(k, C_KEY))
            for k, lab in zip(keys, labels)]


GFM_KEYS = ("frozen", "lora", "sar", "mt")
HEAD_KEYS = ("lp", "mlp", "frozen", "film")
DECODER_KEYS = ("lp", "mlp", "frozen", "film")
DECODER_LABELS = ["Linear probe", "MLP", "UPerNet", FCN]



def line_handles(labels):
    return [Line2D([], [], color=C_SUP, lw=1.3, label=labels[0]),
            Line2D([], [], color=C_SUP, lw=1.0, ls=(0, (2, 2)),
                   label=labels[1])]


def legend_top(fig):
    blank = Line2D([], [], ls="", label=" ")
    h = marker_handles(GFM_KEYS, ["Frozen backbone", "LoRA", "LoRA + SAR",
                                  "LoRA + multi-temporal"])
    h += marker_handles(DECODER_KEYS, DECODER_LABELS)
    h += line_handles([f"Supervised {FCN}, all AGBD features",
                       f"Supervised {FCN}, Sentinel-2 only"])
    # blanks so that each column holds one group
    h += [blank] * 2
    fig.legend(handles=h, loc="outside upper center", ncol=3,
               frameon=False, handlelength=2.2, columnspacing=2.5)


def legend_right(fig, ax, top=1.0):
    """Four titled legends stacked to the right of panel (a), top-aligned.
    Supervised first: its lines are the first thing the eye lands on."""
    kw = dict(loc="upper left", frameon=False, alignment="left",
              borderaxespad=0, handlelength=2.2,
              title_fontproperties=dict(weight="bold", size=7.5))
    legs = [
        ax.legend(handles=line_handles(
            ["All AGBD features", "Sentinel-2 only"]),
            title="Supervised baseline", bbox_to_anchor=(1.03, top), **kw),
        # one group per thing that varies, so each shape has one meaning:
        # fill = backbone, shape = decoder, except the extra-input markers,
        # which are all LoRA + UPerNet
        ax.legend(handles=marker_handles(("frozen", "lora"),
                                         ["Frozen", "LoRA"]),
                  title="Backbone", bbox_to_anchor=(1.03, 0.7), **kw),
        ax.legend(handles=marker_handles(DECODER_KEYS, DECODER_LABELS),
                  title="Decoder", bbox_to_anchor=(1.03, 0.5), **kw),
        ax.legend(handles=marker_handles(("sar", "mt"),
                                         ["LoRA + SAR",
                                          "LoRA + multi-temporal"]),
                  title="Extra input",
                  bbox_to_anchor=(1.03, 0.3), **kw),
    ]
    for leg in legs[:-1]:
        ax.add_artist(leg)
    # run the layout once, then tuck each group right under the previous one
    gap_px = 10 * fig.dpi / 72
    for above, leg in zip(legs, legs[1:]):
        fig.canvas.draw()
        y = ax.transAxes.inverted().transform(
            (0, above.get_window_extent().y0 - gap_px))[1]
        leg.set_bbox_to_anchor((1.03, y), transform=ax.transAxes)
    # outside the layout: they may run down beside panel (b) instead of
    # squeezing panel (a); savefig lists them so the tight bbox keeps them
    for leg in legs:
        leg.set_in_layout(False)
    return legs


def main():
    rows_a = rows_panel_a()
    span_a = len(rows_a) + 0.2
    row_h = 0.19
    fig, ax_a = plt.subplots(
        figsize=(FIG_W, row_h * span_a + (1.9 if LEGEND == "top" else 1.2)),
        layout="constrained",
    )
    draw_panel(ax_a, rows_a, SUP_A, xlim=XLIM, zoom_rows=ZOOM_ROWS, title=
               "")
    axin, zoom_bottom = draw_zoom(ax_a, rows_a)
    ax_a.set_xlabel("Test RMSE (Mg/ha) (↓)", labelpad=12)  # under blue row
    legs = []
    if LEGEND == "top":
        legend_top(fig)
    else:
        # below the zoom inset, if there is one
        top = zoom_bottom - 0.05 if axin is not None else 1.0
        legs = legend_right(fig, ax_a, top=top)

    # the stacked legends sit outside the axes: list them so the tight bbox
    # keeps them instead of clipping at the figure edge
    for ext in ("pdf", "png"):
        fig.savefig(OUT + "." + ext, bbox_inches="tight", pad_inches=0.03,
                    bbox_extra_artists=legs)
    print("wrote", OUT + ".pdf", OUT + ".png")


if __name__ == "__main__":
    main()