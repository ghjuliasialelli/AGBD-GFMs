"""

Compose the side-by-side AGB map figure: AEF vs the AGBD-features baseline vs ESA CCI v6.0, for
Australasia and Europe. See the TILES comment below for why Asia is not shown despite having the
largest RMSE gap -- it is a deliberate, disclosable omission and the caption must reflect it.

Rows are the three sources, columns the two regions, with one shared colorbar. The styling follows
the repo's existing map-figure conventions (see EcosystemAnalysis/Models/Biomes/Sumatra/
compose_figure.py): viridis, no ticks, bold per-panel titles, PDF + PNG at 300 dpi. The one
departure is VMAX (see below).

The three sources do NOT come out at the same extent:
  - inference_agbd.py runs on a whole Sentinel-2 tile   -> 10980 x 10980 px (~110 km)
  - inference_aef.py runs on a download_tile.py window  ->  ~4060 x 4060 px (40 km)
  - ESA CCI is a global 100 m product on a 10 deg lat/lon grid
A full-tile AEF run is not possible: 11000 x 11000 x 64 as float32 is ~31 GB before load_input's
normalisation temporaries. So the AEF window defines each column's extent: the AGBD tile is cropped
to it here, and the CCI block was reprojected onto that same grid beforehand. That is what makes a
column a like-for-like comparison rather than three different areas stacked.

The AEF predictions must come from a `--masking true` run. With masking off, AEF predicts a median
of ~64 t/ha over the 5.9% of the 59GPM window where the embeddings are nodata -- i.e. visible
biomass over the Tasman Sea. It never mattered for AGBRef, where comparison.py clips to the cell.

Usage:
    python make_map_figure.py [--out <path without extension>] [--dpi 300]

"""

###################################################################################################
# Imports

import argparse
import glob
import numpy as np
import rasterio as rs
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling
from os.path import join, exists, dirname, abspath
from os import makedirs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

###################################################################################################
# Configuration

PRED_AGBD = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps/nico_film/59620098-1_59620098-1_59620098-1"
PRED_AEF  = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film/59620113-1_59620113-1_59620113-1"
# ESA CCI BIOMASS v6.0 2020, put onto each column's AEF grid by crop_cci_maps.py using NEAREST
# neighbour. Nearest does not interpolate: every pixel here holds an original 100 m CCI cell value,
# so the panel shows CCI's real numbers and renders at CCI's real resolution (visible 10x10 blocks)
# rather than a smoothed version that implies detail the product does not have. Reprojecting onto
# the AEF grid is still worth doing because it keeps the three rows of a column pixel-co-registered
# on identical ground; the only cost is a sub-cell (<=50 m) positional snap, negligible at 100 m.
PRED_CCI  = "/scratch3/gsialelli/CCI/maps"

# The three biggest-gap regions. dRMSE is the AEF-vs-AGBD-features improvement on the eval h5.
# Australasia and Europe. Asia (45RXL, the largest gap at dRMSE 5.64) is deliberately NOT shown:
# on that window AEF predicts a median of 214 t/ha with 37% of pixels above 250 and a max of 473 --
# beyond anything AEF produced on any of the 666 AGBRef plots (its max there is 307.9), and ~2x both
# the AGBD-features baseline (119) and ESA CCI v6 (91) on the identical window. It is steep eastern
# Himalaya (Sikkim/Bhutan), hard terrain for all three models, and the behaviour is unexplained
# rather than understood -- so it is omitted as anomalous, not as unflattering. The other Asian
# candidate, 44RLS, saturates not at all (median 16.8, 0% above 250) but is low-biomass Terai
# farmland, which would represent Asian forest no better than omitting it. Say so in the caption:
# these are two regions, not "the top-gap regions".
TILES = [
    {"tile": "59GPM", "region": "Australasia",  "drmse": 3.53},
    {"tile": "32TNS", "region": "Europe",       "drmse": 3.22},
]

ROWS = [
    {"key": "aef",  "label": "AEF"},
    {"key": "agbd", "label": "AGBD features"},
    {"key": "cci",  "label": "ESA CCI v6.0"},
]

# Repo convention (Sumatra/compose_figure.py) is 0-250; raised here because the AGBD-features
# panels reach ~417 t/ha and 250 clipped them visibly. 400 covers the p99 of every panel shown.
VMIN, VMAX = 0, 400
CMAP = "viridis"
CBAR_LABEL = "AGB Density [t/ha]"

# ESA WorldCover 10 m v100 (2020), one file per S2 tile, already in the tile's UTM CRS.
# Class 80 = permanent water bodies. Masked because all three models happily predict biomass over
# water: with AEF's nodata masking off, the 59GPM window came out at a median of 64 t/ha over the
# Tasman Sea. Only water is masked -- built-up (class 50) is deliberately NOT, because urban areas
# are genuinely low-biomass rather than invalid, so hiding them would remove real signal and would
# flatter whichever model handles cities worst.
#
# The mask is applied IDENTICALLY to all three rows. Masking one panel and not another would change
# which pixels each metric/colour ramp sees and the column would stop being a like-for-like
# comparison -- the whole point of cropping every row to the AEF window.
WORLDCOVER = "/scratch3/gsialelli/WorldCover/S2/ESA_WorldCover_10m_2020_v100_{tile}.tif"
WATER_CLASSES = (80,)
MASK_COLOR = "0.85"

# Scale bar. The AEF window defines every column's extent, so a column is ~40 km across, NOT the
# ~110 km of the Sentinel-2 tile it is named after. Without a scale cue a reader reasonably assumes
# "32TNS" means the whole tile -- the AGBD-features prediction genuinely is full-tile, and it is
# cropped down to the AEF window here. The bar is the cue: it is drawn from each panel's own bounds,
# so if the window ever changes the figure follows automatically rather than going quietly stale.
SCALEBAR_FRAC = 0.25                             # target bar length, as a fraction of panel width
SCALEBAR_NICE_KM = (1, 2, 5, 10, 20, 25, 50, 100)  # snap to a round number a reader can use
SCALEBAR_COLOR = "white"                         # over viridis; outlined below for the pale end
SCALEBAR_OUTLINE = "black"

# Render each panel at roughly this many pixels per side. The rasters are ~4000 px; at 300 dpi a
# panel is only ~1200 px wide, so reading at full resolution would just be resampled away by
# matplotlib -- and six full-resolution panels is 400 MB of float32 held at once for nothing.
RENDER_PX = 1400

###################################################################################################
# Helpers

def find_agbd(tile) :
    """
    Locate the AGBD-features prediction for a tile. It is named after the S2 product, not the tile,
    so it has to be globbed rather than constructed.

    Args:
    - tile (str): the MGRS tile name, e.g. '45RXL'.

    Returns:
    - str or None: path to the prediction, or None if absent.
    """
    hits = sorted(glob.glob(join(PRED_AGBD, f"*_T{tile}_*.tif")))
    return hits[0] if hits else None


def read_panel(path, bounds = None, crs = None) :
    """
    Read band 1 (AGB) of a prediction, optionally cropped to `bounds`, downsampled for rendering.

    Args:
    - path (str): path to the prediction GeoTIFF.
    - bounds (tuple): (left, bottom, right, top) to crop to, in the raster's own CRS. None = all.
    - crs: the CRS `bounds` are expressed in; must match the raster's (asserted, not reprojected --
           both models run in the tile's UTM zone, so a mismatch means something upstream is wrong).

    Returns:
    - np.ndarray: the AGB band, with nodata as NaN.
    - tuple: the bounds actually read.
    - CRS: the raster's CRS, so the water mask can be built on this panel's own grid.
    """
    with rs.open(path) as src :
        if bounds is not None :
            assert src.crs == crs, f"{path} is {src.crs}, expected {crs}"
            window = from_bounds(*bounds, transform = src.transform)
            out_bounds = bounds
        else :
            window = None
            out_bounds = tuple(src.bounds)

        h = int(window.height) if window is not None else src.height
        w = int(window.width) if window is not None else src.width
        step = max(1, int(round(max(h, w) / RENDER_PX)))
        out_shape = (max(1, h // step), max(1, w // step))

        data = src.read(1, window = window, out_shape = out_shape).astype(np.float32)
        if src.nodata is not None : data[data == src.nodata] = np.nan
        out_crs = src.crs

    return data, out_bounds, out_crs


def water_mask(tile, bounds, crs, out_shape) :
    """
    Build a boolean water mask on a panel's exact grid from ESA WorldCover.

    Resampling is `nearest` and nothing else: WorldCover stores class codes, so any averaging
    kernel would invent classes that do not exist (e.g. blending 10=tree and 80=water into 45).

    Args:
    - tile (str): MGRS tile name, used to locate the WorldCover file.
    - bounds (tuple): (left, bottom, right, top) of the panel, in `crs`.
    - crs: the panel's CRS.
    - out_shape (tuple): (height, width) of the rendered panel, so the mask lines up pixel-for-pixel.

    Returns:
    - np.ndarray or None: True where the pixel is water; None if WorldCover is missing for the tile.
    """
    path = WORLDCOVER.format(tile = tile)
    if not exists(path) : return None

    # Normalise the bounds: a south-up source reports top < bottom, and from_bounds would then build
    # a vertically mirrored transform without complaining.
    left, right = min(bounds[0], bounds[2]), max(bounds[0], bounds[2])
    bottom, top = min(bounds[1], bounds[3]), max(bounds[1], bounds[3])

    dst = np.zeros(out_shape, dtype = np.uint8)
    dst_transform = transform_from_bounds(left, bottom, right, top, out_shape[1], out_shape[0])

    with rs.open(path) as src :
        reproject(source = rs.band(src, 1), destination = dst,
                  src_transform = src.transform, src_crs = src.crs,
                  dst_transform = dst_transform, dst_crs = crs,
                  resampling = Resampling.nearest)

    return np.isin(dst, WATER_CLASSES)


def add_scalebar(ax, bounds, shape) :
    """
    Draw a scale bar in the bottom-right of a panel, sized from the panel's own ground extent.

    The panel is drawn with `imshow` and no `extent`, so axes coordinates are pixel indices; the
    bar length in pixels is therefore (bar length in metres) / (metres per pixel), where the latter
    comes from the bounds actually read rather than from any assumed resolution. Bounds are
    abs()-ed because a south-up source reports top < bottom.

    Args:
    - ax: the panel's axes.
    - bounds (tuple): (left, bottom, right, top) of the panel, in its own (projected, metric) CRS.
    - shape (tuple): (height, width) of the rendered array, in pixels.

    Returns:
    - None. Draws onto `ax`.
    """
    h, w = shape
    width_m = abs(bounds[2] - bounds[0])
    if width_m <= 0 or w <= 0 : return

    # Snap to the nice round length closest to SCALEBAR_FRAC of the panel.
    target_km = SCALEBAR_FRAC * width_m / 1000
    length_km = min(SCALEBAR_NICE_KM, key = lambda k : abs(k - target_km))
    bar_px = (length_km * 1000) / (width_m / w)

    margin = 0.05 * w
    x1 = w - margin
    x0 = x1 - bar_px
    y = h - margin

    stroke = [pe.withStroke(linewidth = 3, foreground = SCALEBAR_OUTLINE)]
    ax.plot([x0, x1], [y, y], color = SCALEBAR_COLOR, lw = 2, solid_capstyle = "butt",
            path_effects = stroke, clip_on = False)
    ax.text((x0 + x1) / 2, y - 0.025 * h, f"{length_km} km", color = SCALEBAR_COLOR,
            fontsize = 9, fontweight = "bold", ha = "center", va = "bottom",
            path_effects = stroke)


###################################################################################################
# Figure

def make_figure(out_path, dpi) :
    """
    Compose and save the figure.

    Args:
    - out_path (str): output path WITHOUT extension; .pdf and .png are both written.
    - dpi (int): resolution for the raster output.
    """
    ncols = len(TILES)
    nrows = len(ROWS)

    fig = plt.figure(figsize = (10, 13))
    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        width_ratios = [1] * ncols + [0.04],
        wspace = 0.08, hspace = 0.12,
        left = 0.06, right = 0.93, top = 0.90, bottom = 0.05,
    )

    missing = []
    for c, spec in enumerate(TILES) :
        tile = spec["tile"]

        # The AEF window defines the extent for the whole column: it is the smaller of the two, and
        # cropping the AGBD tile down to it is what makes the column a like-for-like comparison.
        aef_path = join(PRED_AEF, f"{tile}.tif")
        agbd_path = find_agbd(tile)

        col_bounds, col_crs = None, None
        if exists(aef_path) :
            with rs.open(aef_path) as s :
                col_bounds, col_crs = tuple(s.bounds), s.crs

        for r, row in enumerate(ROWS) :
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]) ; ax.set_yticks([])

            # Only the AGBD tile needs cropping: the AEF panel defines the window, and the CCI crop
            # was already reprojected onto that same grid.
            if row["key"] == "aef" :
                path, bounds, crs = aef_path, None, None
            elif row["key"] == "cci" :
                path, bounds, crs = join(PRED_CCI, f"{tile}_CCI.tif"), None, None
            else :
                path, bounds, crs = agbd_path, col_bounds, col_crs

            if path is None or not exists(path) or (row["key"] == "agbd" and col_bounds is None) :
                missing.append(f'{row["label"]} / {tile}')
                ax.text(0.5, 0.5, f'[{row["label"]}\n{tile}]\nnot found', ha = "center",
                        va = "center", fontsize = 9, color = "0.5", transform = ax.transAxes)
                ax.set_facecolor("0.95")
            else :
                data, p_bounds, p_crs = read_panel(path, bounds = bounds, crs = crs)

                # Same WorldCover water mask on every row of the column. Built per panel from that
                # panel's own bounds/shape rather than shared, because the three rasters differ by
                # a pixel or two after downsampling and a shared mask would be off-by-one.
                wm = water_mask(tile, p_bounds, p_crs, data.shape)
                if wm is not None : data = np.where(wm, np.nan, data)

                cmap = plt.get_cmap(CMAP).copy()
                cmap.set_bad(MASK_COLOR)
                ax.imshow(data, cmap = cmap, vmin = VMIN, vmax = VMAX, interpolation = "nearest")

                # One bar per column, on the bottom row: all three rows of a column are cropped to
                # the same AEF window on identical ground, so a per-panel bar would repeat the same
                # number six times. Move this to every panel if a single row is ever shown alone.
                if r == nrows - 1 : add_scalebar(ax, p_bounds, data.shape)

            if r == 0 :
                ax.set_title(f'{spec["region"]}  ({tile})\n$\\Delta$RMSE {spec["drmse"]:.2f} t/ha',
                             fontsize = 11, fontweight = "bold", pad = 8)
            if c == 0 :
                ax.set_ylabel(row["label"], fontsize = 12, fontweight = "bold", labelpad = 10)

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[:, ncols])
    sm = ScalarMappable(cmap = CMAP, norm = Normalize(vmin = VMIN, vmax = VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax = cbar_ax)
    cbar.set_label(CBAR_LABEL, fontsize = 10)
    cbar.ax.tick_params(labelsize = 9)

    makedirs(dirname(abspath(out_path)), exist_ok = True)
    for ext in ("pdf", "png") :
        fig.savefig(f"{out_path}.{ext}", dpi = dpi, bbox_inches = "tight")
        print(f"Saved {out_path}.{ext}")
    plt.close(fig)

    if missing :
        print(f"\nWARNING: {len(missing)} panel(s) missing, drawn as placeholders:")
        for m in missing : print(f"  {m}")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type = str,
                        default = join(dirname(abspath(__file__)), "plots", "map_AEF_vs_AGBD-features"))
    parser.add_argument("--dpi", type = int, default = 300)
    args = parser.parse_args()
    make_figure(args.out, args.dpi)
