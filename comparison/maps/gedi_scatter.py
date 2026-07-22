"""GEDI-vs-prediction density scatter for the four map-figure AGB maps.

A companion to make_map_figure.py: the map shows WHERE each model is right; this shows HOW right,
against GEDI L4A. One hexbin per (model, tile), rows = models, columns = tiles, so it lines up with
the map figure's orientation. Sampling is IDENTICAL to gedi_per_tile_eval/metrics_4model.py Method A
(one median GEDI value per 10 m S2 cell, every map sampled at the cell centre, AEF-bounds,
water-masked, paired across all four maps), so the RMSE printed here equals the number on the map
panel to the decimal.

This is an OPTIONAL diagnostic panel -- kept separate and written to its own file so it can be
dropped without touching the main figure.

Run in the `agbd` env with PROJ_LIB set per-command:
    PROJ_LIB=/scratch2/gsialelli/miniconda3/envs/agbd/share/proj \
        /scratch2/gsialelli/miniconda3/envs/agbd/bin/python gedi_scatter.py
"""
import argparse
import json
from os.path import join, dirname, abspath
from os import makedirs

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- paths / conventions, kept in lockstep with metrics_4model.py -----------------------------
GEDI_DIR = "/scratch3/gsialelli/gedi_per_tile_eval"
WC  = "/scratch3/gsialelli/WorldCover/S2/ESA_WorldCover_10m_2020_v100_{}.tif"
CCI = "/scratch3/gsialelli/CCI/maps/{}_CCI.tif"
SSL = "/scratch3/gsialelli/ssl4eo_maps/preds/{}.tif"
NODATA = -9999.0
WATER_CLASS = 80
EPOCH = np.datetime64("2019-04-17")
DAY0 = int((np.datetime64("2020-01-01") - EPOCH).astype(int))
DAY1 = int((np.datetime64("2020-12-31") - EPOCH).astype(int))

# Columns = tiles (region order matches the map figure); rows = the four AGB maps.
TILE_ORDER = [("59GPM", "Australasia"), ("32TPT", "Europe"), ("49SBT", "Asia")]

# Per-tile display crop (must match make_map_figure.py TILES "crop" and metrics_4model.py CROP), so
# the scatter is scored over exactly the pixels the map panel shows. None = whole AEF window.
CROP = {
    "59GPM": (637460.0, 5137000.0, 666170.0, 5165710.0),
    "32TPT": None,
    "49SBT": (234297.1, 3724522.1, 275379.7, 3765694.6),
}
MODELS = [("aef", "AEF"), ("agbd", "AGBD features"),
          ("ssl4eo", "SSL4EO-MoCo (30 m)"), ("cci", "ESA CCI v6.0")]

# make_map_figure.py conventions.
VMAX = 400
CMAP = "viridis"


def sample1(path, xy) :
    with rasterio.open(path) as src :
        return np.array([v[0] for v in src.sample(xy, indexes=[1])], dtype="float64")


def paired_samples(tile) :
    """Return (ref, {model_key: pred}) over the AEF-bounds, water-masked, all-four-valid cells."""
    b = json.load(open(f"{GEDI_DIR}/bboxes.json"))[tile]
    crs = b["crs"]
    win = CROP[tile] if CROP[tile] is not None else b["aef_utm"]   # display scope, matches the map
    paths = {"aef": b["aef_path"], "agbd": b["agbd_path"],
             "ssl4eo": SSL.format(tile), "cci": CCI.format(tile)}

    df = pd.read_csv(f"{GEDI_DIR}/gedi_{tile}.csv")
    df = df[(df.date >= DAY0) & (df.date <= DAY1)]

    with rasterio.open(paths["agbd"]) as g :
        gt, gh, gw = g.transform, g.height, g.width
    xs, ys = warp_transform("EPSG:4326", crs, df.lon.values, df.lat.values)
    xs, ys = np.asarray(xs), np.asarray(ys)

    # one median GEDI value per 10 m S2 cell (the AGBD grid), at the cell centre
    col = np.floor((xs - gt.c) / gt.a).astype(np.int64)
    row = np.floor((ys - gt.f) / gt.e).astype(np.int64)
    inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
    grp = pd.DataFrame({"row": row[inside], "col": col[inside],
                        "v": df.agbd.values.astype("float64")[inside]}
                       ).groupby(["row", "col"])["v"].median().reset_index()
    cx = gt.c + (grp["col"].values + 0.5) * gt.a
    cy = gt.f + (grp["row"].values + 0.5) * gt.e
    ref = grp["v"].values.astype("float64")
    xy = list(zip(cx, cy))

    pred = {k: sample1(p, xy) for k, p in paths.items()}
    water = sample1(WC.format(tile), xy) == WATER_CLASS
    in_win = (cx >= win[0]) & (cx <= win[2]) & (cy >= win[1]) & (cy <= win[3])
    ok = {k: (pred[k] != NODATA) & np.isfinite(pred[k]) for k in pred}
    keep = in_win & (~water) & np.isfinite(ref)
    for k in ok : keep &= ok[k]
    return ref[keep], {k: pred[k][keep] for k in pred}


def main() :
    p = argparse.ArgumentParser()
    p.add_argument("--out", default = join(dirname(abspath(__file__)), "plots", "gedi_scatter"))
    p.add_argument("--dpi", type = int, default = 200)
    args = p.parse_args()

    data = {tile: paired_samples(tile) for tile, _ in TILE_ORDER}

    nrows, ncols = len(MODELS), len(TILE_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize = (3.5 * ncols, 3.5 * nrows),
                             sharex = True, sharey = True)

    for i, (mkey, mlabel) in enumerate(MODELS) :
        for j, (tile, region) in enumerate(TILE_ORDER) :
            ax = axes[i, j]
            ref, preds = data[tile]
            y = preds[mkey]

            ax.plot([0, VMAX], [0, VMAX], color = "0.5", lw = 1, ls = "--", zorder = 1)
            hb = ax.hexbin(ref, y, gridsize = 42, extent = (0, VMAX, 0, VMAX),
                           bins = "log", cmap = CMAP, mincnt = 1, linewidths = 0, zorder = 2)

            e = y - ref
            rmse = float(np.sqrt(np.mean(e ** 2)))
            bias = float(np.mean(e))
            r = float(np.corrcoef(ref, y)[0, 1])
            ax.text(0.04, 0.96,
                    f"RMSE {rmse:.0f}\nbias {bias:+.0f}\nr {r:.2f}\nn {len(ref):,}",
                    transform = ax.transAxes, ha = "left", va = "top", fontsize = 8.5,
                    color = "0.15", linespacing = 1.3,
                    bbox = dict(boxstyle = "round,pad=0.3", fc = "white", ec = "0.8", alpha = 0.85))

            ax.set_xlim(0, VMAX) ; ax.set_ylim(0, VMAX)
            ax.set_aspect("equal")
            if i == 0 : ax.set_title(f"{region}  ({tile})", fontsize = 11, fontweight = "bold")
            if j == 0 : ax.set_ylabel(f"{mlabel}\npredicted t/ha", fontsize = 10, fontweight = "bold")
            if i == nrows - 1 : ax.set_xlabel("GEDI L4A AGBD [t/ha]", fontsize = 10)

    fig.suptitle("Prediction vs GEDI L4A -- each panel's shown window, water-masked, common footprints",
                 fontsize = 12, fontweight = "bold", y = 0.995)
    fig.tight_layout(rect = (0, 0, 1, 0.985))

    makedirs(dirname(abspath(args.out)), exist_ok = True)
    for ext in ("pdf", "png") :
        fig.savefig(f"{args.out}.{ext}", dpi = args.dpi, bbox_inches = "tight")
        print(f"Saved {args.out}.{ext}")
    plt.close(fig)


if __name__ == "__main__" :
    main()
