"""

Compose the side-by-side AGB map figure: AEF vs the AGBD-features baseline vs the SSL4EO-MoCo
benchmark vs ESA CCI v6.0, for Australasia, Europe and Asia. See the TILES comment below for which
tile represents each region and why -- two Asian candidates are rejected there, and the caption must
reflect that the tile shown is 49SBT rather than the largest-gap one.

The SSL4EO-MoCo row is NOT like-for-like on two counts the caption MUST state:
  - It is 30 m, not 10 m. The model predicts only the CENTRE pixel of each 25x25 AGBD patch (624/625
    outputs never receive a gradient), so a dense output is invalid -- see model/inference_ssl4eo.py
    and memory/pangaea-agbd-centre-pixel-only.md. Every output pixel here is one true centre-pixel
    forward pass at a 3-pixel (30 m) stride; the coarser grid is inherent, not a downsample choice.
  - Its cloud/shadow exposure differs slightly: it masks SCL 0/1/6/11 at inference (the
    inference_agbd.py convention) whereas the AGBD/AEF rows here are masked only for WorldCover water
    at figure time. In practice the scenes are near cloud-free on the shown windows, but say it.

Rows are the Sentinel-2 input, the GEDI L4A reference, and the four AGB sources; columns are the
three regions, with one shared colorbar spanning every row that is in t/ha (all but Sentinel-2).

The GEDI row shows the reference footprints each panel's RMSE is computed against -- the identical
selection, taken from gedi_scatter.paired_samples and count-asserted against the "rmse" entries
below, so the figure cannot show one footprint set while reporting another. Two things the caption
must state: the markers are NOT to scale (a 25 m footprint is ~0.15 pt on a 41 km panel, so they are
drawn oversized, and only their location and value are meaningful), and the columns differ enormously
in coverage -- 51,995 cells in Europe against ~5-6k in the other two -- which is why the marker size
is fixed rather than per-column.

The styling follows
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
import sys
import numpy as np
import rasterio as rs
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling
from os.path import join, exists, dirname, abspath, basename
from os import makedirs
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

# The GEDI reference row reuses gedi_scatter.paired_samples VERBATIM rather than re-implementing the
# footprint selection here. That selection (2020 footprints, one median value per 10 m S2 cell,
# water-masked, valid in all four maps, inside the display crop) is precisely what the panel RMSEs
# are computed over in gedi_per_tile_eval/metrics_4model.py -- a second copy of it would be free to
# drift, and a figure that plots a different footprint set than it scores is worse than no row.
sys.path.insert(0, dirname(abspath(__file__)))
from gedi_scatter import paired_samples

###################################################################################################
# Configuration

PRED_AGBD = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps/nico_film/59620098-1_59620098-1_59620098-1"
PRED_AEF  = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film/59620113-1_59620113-1_59620113-1"
# SSL4EO-MoCo (ViT-S/16) + RegUPerNet, fine-tuned on AGBD full, run centre-pixel-only at a 30 m
# stride (see model/inference_ssl4eo.py). One file per tile, named `<tile>.tif`, already on the
# AEF window in the tile's UTM CRS -- so it reads through the same from_bounds crop as every other
# AGB row, at its own (coarser) grid. Provenance for these rasters is SSL4EO_MAPS_TODO.md.
PRED_SSL4EO = "/scratch3/gsialelli/ssl4eo_maps/preds"
# ESA CCI BIOMASS v6.0 2020, put onto each column's AEF grid by crop_cci_maps.py using NEAREST
# neighbour. Nearest does not interpolate: every pixel here holds an original 100 m CCI cell value,
# so the panel shows CCI's real numbers and renders at CCI's real resolution (visible 10x10 blocks)
# rather than a smoothed version that implies detail the product does not have. Reprojecting onto
# the AEF grid is still worth doing because it keeps the three rows of a column pixel-co-registered
# on identical ground; the only cost is a sub-cell (<=50 m) positional snap, negligible at 100 m.
PRED_CCI  = "/scratch3/gsialelli/CCI/maps"

# Three regions: Australasia, Europe, Asia. dRMSE is the AEF-vs-AGBD-features improvement on the
# eval h5 -- it is the SELECTION criterion, not a displayed number (see the "gedi" note below).
#
# Asia is represented by 49SBT (Qinling, Shaanxi), NOT by the largest-gap tile. Two other Asian
# candidates were rejected, and the caption must say so rather than implying these are simply the
# top-gap regions:
#   45RXL - largest gap of all (dRMSE 5.64) but anomalous: AEF predicts a median of 214 t/ha with
#           37% of pixels above 250 and a max of 473 -- beyond anything AEF produced on any of the
#           666 AGBRef plots (max there 307.9), and ~2x both the AGBD-features baseline (119) and
#           ESA CCI v6 (91) on the identical window. Steep eastern Himalaya (Sikkim/Bhutan), hard
#           terrain for all three models. Omitted as unexplained, not as unflattering.
#   44RLS - saturates not at all (median 16.8, 0% above 250) but is low-biomass Terai farmland,
#           which would represent Asian forest no better than omitting the region entirely.
# 49SBT was chosen because it is genuine mixed temperate forest AND it has by far the best GEDI
# validation coverage of any candidate window (6381 cells in 2020 vs 6178 for 59GPM and ZERO for
# 32TNS), so its panel header can carry a real, independently-sourced number.
#
# "drmse" is the tile SELECTION criterion only and is deliberately no longer displayed: it is a
# dataset-wide eval-h5 number, so printing it on a per-tile map panel asserted something the panel
# does not show. Each AGB PANEL now carries its own per-tile GEDI L4A RMSE instead (see "rmse"
# below and gedi_per_tile_eval/metrics_4model.py: AEF-window scope, 2020 footprints, water-masked,
# paired across all four maps).
#
# "rmse" carries per-tile GEDI L4A RMSE (t/ha) for EACH of the four AGB maps, keyed by ROW key, so
# every panel can label its own accuracy -- this replaced the old two-model AEF-vs-AGBD header line
# (2026-07-22). Numbers are Method A "paired4" from gedi_per_tile_eval/metrics_4model.py at scope
# = each panel's DISPLAY crop (so the RMSE describes exactly the pixels shown, not a wider window --
# only 59GPM crops; 49SBT/32TPT display == full AEF window), water-masked, and paired across all
# four maps so every model is scored on an IDENTICAL set of GEDI footprints (n below). SSL4EO (30 m)
# and CCI (100 m) carry a small resolution penalty
# under this common-footprint scoring; evaluating each on its own native grid (Method B) moves them
# by <=1 t/ha (SSL4EO) and ~4-5 t/ha (CCI), i.e. the ranking is not an artefact of resolution -- the
# caption should say so. 32TNS is gone; 32TPT replaced it precisely because 32TNS has ZERO GEDI
# footprints in its AEF window in any year.
#
# These numbers describe the CURRENT production rasters, which since 2026-07-21 are the
# BILINEAR-DEM re-runs (see predictions_maps/.../nearest_dem_20260721/README.md). That choice was
# made on appearance -- the bilinear maps have visibly less of the fine-scale striping -- and it
# costs accuracy, so the caption must not present these AGBD numbers as the model's best:
#         tile     AGBD RMSE  bilinear (production) / nearest (archived)
#         59GPM        67.5 / 67.5     (wash; MAE 40.4 vs 41.5)
#         32TPT        87.3 / 86.8     (wash; MAE 62.4 vs 63.3)
#         49SBT       121.5 / 115.9    (WORSE: MAE 96.1 vs 91.4, R2 -0.425, bias -73.5 t/ha)
# AEF is byte-identical between the two runs (it uses no DEM), which is the control confirming the
# difference is the DEM channel and not pipeline drift.
#
# The underlying cause of the striping is NOT the DEM: it is topographic-illumination bias in the
# AGBD-features model, which reads terrain shading as biomass -- r(prediction, cos solar incidence)
# = -0.18 to -0.21, negative in 318 of 318 windows across all three tiles, with the sign holding
# even though 59GPM's sun azimuth is ~100 deg from the other two. Bilinear DEMs suppress the
# speckle largely by shifting the whole prediction DOWN (-8.2, -7.1, -0.3 t/ha tile means), which
# is cosmetic. The model was also TRAINED on nearest-resampled DEMs, so this is additionally a
# train/inference mismatch. Both facts must be disclosed if these maps are published.
#
# "crop" (left, bottom, right, top) in the tile's own UTM CRS, or None for the whole AEF window.
# It is applied IDENTICALLY to all three rows, so a column stays a like-for-like comparison. Chosen
# per tile by measurement, not by eye -- rationale inline below so it can be re-derived or retuned.
TILES = [
    # 59GPM: a SQUARE 28.71 km window on Banks Peninsula, full north extent, ocean trimmed.
    # Vertical: the peninsula's southern tip is y = 5137240 and the AEF window ran to y = 5124580,
    # so the bottom 12.6 km was pure Pacific -- greyed by the water mask, carrying no information.
    # Horizontal: cut to match that height, 9.3 km off the right and 3.0 km off the left, which
    # centres Akaroa Harbour. A sweep of every horizontal offset put land at 79-80% across the
    # board (max 80.0%), so the exact offset is a framing choice, not an optimisation.
    # Net: land 46.2% -> 80.0%, and the panel stays square like the other two columns.
    # Do NOT try to extend further north: land occupies row 0 of the AEF raster, so the window is
    # already truncated at its northern edge. More north needs a new AEF download, not a new crop.
    {"tile": "59GPM", "region": "Australasia",  "drmse": 3.53,
     "rmse": {"aef": 64.28, "agbd": 66.81, "ssl4eo": 72.49, "cci": 90.32, "n": 5086},
     "crop": (637460.0, 5137000.0, 666170.0, 5165710.0)},
    # 32TPT (Tyrol, AUSTRIA -- 10.77-11.33 E, 47.15-47.53 N), swapped in 2026-07-21 to replace
    # 32TNS. 32TNS had to go: its AEF window sits in Graubuenden/Valtellina (Switzerland/Italy)
    # despite the "Austria" label, and it contains ZERO GEDI footprints in any year, so its panel
    # header could never carry an honest number. 32TNS also ranked LAST of 28 Austrian tiles by
    # footprint count (1,447 vs 285,988 for 32TPT). Its S2 scene was snow-dominated (21 May) too.
    # No crop needed: this AEF mosaic is clean -- 0.00% nodata, exact 10.0 m, no edge frame, and
    # the CCI block is already on its grid.
    {"tile": "32TPT", "region": "Europe",       "drmse": 3.22,
     "rmse": {"aef": 75.54, "agbd": 87.27, "ssl4eo": 103.77, "cci": 117.11, "n": 51995},
     "crop": None},
    # drmse here is Asia's REGION-level eval-h5 gap (5.64), the same number the original comment
    # attached to Asia when 45RXL was the chosen representative. It is a per-region figure, so it
    # does not change when the representative tile changes. Not displayed; selection criterion only.
    #
    # 49SBT: the AEF mosaic has a nodata frame -- 131 fully-empty rows on top, 111 on the bottom,
    # 122 cols on the left (8.27% of the raster, and ALL of its nodata). Cause: the GEE export left
    # three shards, two native EPSG:32649 (which tile together exactly) and one duplicate of the
    # same ground in EPSG:32648. Warping that cross-zone shard into 32649 turns its rectangle into a
    # skewed quadrilateral, so its BOUNDING BOX enlarged the mosaic by exactly those bands while
    # contributing zero valid pixels. Verified: the mosaic's valid-data rectangle equals the union
    # of the two native shards to within one pixel. This crop is that rectangle.
    # The real fix is upstream -- re-mosaic 49SBT from the two `xma4334trh30x9qrf` shards ONLY and
    # drop `x1q02j0mrxcvdop5j`; that also removes a needless warp and restores an exact 10.0 m pixel
    # (the mosaic is currently 9.99576 m because of it). That needs an AEF inference re-run, so this
    # crop is the no-recompute stand-in, not a substitute for fixing the mosaic.
    {"tile": "49SBT", "region": "Asia",         "drmse": 5.64,
     "rmse": {"aef": 95.57, "agbd": 121.46, "ssl4eo": 137.87, "cci": 175.59, "n": 5748},
     "crop": (234297.1, 3724522.1, 275379.7, 3765694.6)},
]

# The Sentinel-2 row goes FIRST: it is the input the AGBD-features model actually consumed, so the
# figure reads input -> two model outputs -> reference product. Its "key" is special-cased in the
# draw loop (3 bands, own stretch, no colorbar) -- everything else is a single AGB band.
# SSL4EO-MoCo sits with the other two model outputs (input -> models -> reference), after the
# AGBD-features baseline it is being compared against. Its label carries "(30 m)" because it is the
# one row not at 10 m -- the resolution difference is real and must not be silent on the figure.
#
# The GEDI row goes SECOND, between the input and the maps: it is the reference every RMSE on this
# figure is measured against, and putting it above the model rows lets a reader see how much of each
# panel is actually constrained by validation data before reading the numbers. Its cells sit on the
# same t/ha colour scale as the maps below, so a footprint's colour is directly comparable to the
# pixel underneath it. Like "s2" it is special-cased in the draw loop (a scatter, not a raster).
ROWS = [
    {"key": "s2",     "label": "Sentinel-2 RGB"},
    {"key": "gedi",   "label": "GEDI L4A footprints"},
    {"key": "aef",    "label": "AEF"},
    {"key": "agbd",   "label": "AGBD features"},
    {"key": "ssl4eo", "label": "SSL4EO-MoCo (30 m)"},
    {"key": "cci",    "label": "ESA CCI v6.0"},
]

# Directory of unzipped .SAFE products. The specific product is NOT globbed by tile -- it is derived
# from the AGBD prediction filename (see find_s2), because this directory holds more than one
# acquisition for the same tile (59GPM has both 20200223 and 20200418) and picking the wrong one
# would show a scene the models never saw, on a date months away, with no error raised.
S2_DIR = "/scratch3/gsialelli/S2_L2A"

# True-colour stretch percentiles, applied ONCE across all three bands of a panel (see read_rgb --
# per-band stretching produces false colour casts). Percentile rather than a fixed range because
# the three scenes are radically different (alpine, temperate forest, a peninsula in late summer)
# and one fixed range either blows out or muddies at least one of them. This row is CONTEXT, not a
# measurement: brightness is not comparable between panels -- say so in the caption.
S2_PCT = (1, 99)

# Display gamma applied AFTER the shared stretch. Sentinel-2 surface reflectance is bunched at the
# dark end, so a purely linear stretch renders vegetated scenes muddy and forces the contrast up
# until they look garish. Gamma < 1 lifts the midtones only.
#
# It is applied to all three channels EQUALLY, so like the shared stretch it changes brightness and
# contrast but NOT hue -- x**g is monotonic and identical per channel, so the band ordering that
# defines colour survives. This is the whole difference between this and per-band stretching:
# tuning appearance is fine, fabricating colour is not.
S2_GAMMA = 0.65

# GEDI row. Marker size is in points^2 and is NOT to scale: a 25 m footprint on a 41 km panel is
# ~0.15 pt across, i.e. invisible, so the markers are deliberately oversized and the caption must
# say the row shows footprint LOCATIONS and values, not footprint extents. 32TPT carries 51,995
# cells against ~5-6k for the other two columns, so a single size either speckles Europe into a
# solid block or loses the other columns entirely -- the size is therefore derived from each
# column's own count, not fixed.
#
# The marker size is the SAME in every column on purpose, even though 32TPT carries 51,995 cells
# against ~5-6k for the other two: the wildly different GEDI coverage between columns is real
# information about how well each number is supported, and per-column autoscaling would hide it.
GEDI_BG = "0.95"                 # panel background where there is no footprint (not nodata: no data)
GEDI_MARKER_PT = 0.8             # marker diameter in points (~3 px at 300 dpi), not to ground scale

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

# Cap on rendered pixels per panel side. None = NATIVE RESOLUTION (the default): every 10 m
# prediction pixel is drawn, which is the point of the figure.
#
# This was previously a hard-coded 1400, which silently decimated each ~4100 px panel by step 3 --
# nearest neighbour, so 8 of every 9 pixels were thrown away and the panels were effectively 30 m.
# That undersells a 10 m product and, worse, aliases away exactly the fine-scale striping the
# AGBD-features maps are being audited for. A figure must not quietly resample the thing it is
# evidence about. Kept as a FLAG (--render-px) rather than a constant so a quick preview is still
# cheap, per the "thresholds should be flags" rule.
#
# Note the downsampling path uses nearest, NOT averaging, and deliberately so: averaging would mix
# the -9999 nodata sentinel into its neighbours, since nodata is only converted to NaN after the
# read. So --render-px is a PREVIEW mode; published output should run at native resolution.
RENDER_PX = None

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


def find_s2(agbd_path) :
    """
    Locate the R10m B04/B03/B02 JP2s of the S2 product that produced a given AGBD prediction.

    Derived from the prediction filename rather than globbed by tile ON PURPOSE: S2_DIR holds
    several acquisitions of the same tile (59GPM has 20200223 AND 20200418), so a tile glob would
    happily return a scene from a different month that the model never saw. The prediction is named
    `<product>_NA.tif`, which pins the acquisition exactly.

    Args:
    - agbd_path (str or None): path to the AGBD-features prediction for this tile.

    Returns:
    - list or None: [B04, B03, B02] paths (red, green, blue), or None if anything is missing.
    """
    if agbd_path is None : return None

    product = basename(agbd_path)
    for suffix in ("_NA.tif", ".tif") :
        if product.endswith(suffix) :
            product = product[: -len(suffix)]
            break

    safe = join(S2_DIR, f"{product}.SAFE")
    if not exists(safe) : return None

    # Both extensions are real: the standard products ship JP2, but the reprocessed N9999 32TNS
    # product carries GeoTIFF bands instead. Globbing only *.jp2 silently lost that whole column.
    paths = []
    for band in ("B04", "B03", "B02") :
        hits = []
        for ext in ("jp2", "tif") :
            hits += sorted(glob.glob(
                join(safe, "GRANULE", "*", "IMG_DATA", "R10m", f"*_{band}_10m.{ext}")))
        if not hits : return None
        paths.append(hits[0])
    return paths


def boa_offset(jp2_path) :
    """
    Read BOA_ADD_OFFSET from a product's MTD_MSIL2A.xml.

    Baseline >= 04.00 products store reflectance shifted by -1000; ignoring it does not error, it
    just washes the image out. Parsed from the product's own metadata rather than inferred from the
    baseline string, because the reprocessed N9999 product carries the offset too.

    Args:
    - jp2_path (str): path to any band JP2 inside the .SAFE.

    Returns:
    - float: the offset to ADD to raw DN (0.0 if the product declares none).
    """
    safe = jp2_path
    while safe and not safe.endswith(".SAFE") : safe = dirname(safe)
    mtd = join(safe, "MTD_MSIL2A.xml")
    if not exists(mtd) : return 0.0

    with open(mtd) as fh : text = fh.read()
    m = re.search(r'BOA_ADD_OFFSET[^>]*>(-?\d+)', text)
    return float(m.group(1)) if m else 0.0


def read_rgb(paths, bounds, crs) :
    """
    Read a true-colour composite, cropped to `bounds` and downsampled for rendering.

    Args:
    - paths (list): [red, green, blue] band paths.
    - bounds (tuple): (left, bottom, right, top) in the rasters' CRS, or None for the full scene.
    - crs: the CRS `bounds` are in; asserted against each band.

    Returns:
    - np.ndarray: (H, W, 3) float in [0, 1], percentile-stretched.
    - tuple: the bounds actually read.
    - CRS: the rasters' CRS.
    """
    chans, out_bounds, out_crs = [], None, None
    for p in paths :
        with rs.open(p) as src :
            if bounds is not None :
                assert src.crs == crs, f"{p} is {src.crs}, expected {crs}"
                window = from_bounds(*bounds, transform = src.transform)
                out_bounds = bounds
            else :
                window = None
                out_bounds = tuple(src.bounds)

            h = int(window.height) if window is not None else src.height
            w = int(window.width) if window is not None else src.width
            step = 1 if RENDER_PX is None else max(1, int(round(max(h, w) / RENDER_PX)))
            out_shape = (max(1, h // step), max(1, w // step))

            band = src.read(1, window = window, out_shape = out_shape).astype(np.float32)
            out_crs = src.crs

        chans.append((band + boa_offset(p)) / 10000.0)

    rgb = np.dstack(chans)

    # ONE stretch shared by all three bands -- NOT per-band. Stretching each band to its own
    # percentiles rescales the band RATIOS, and those ratios are precisely what colour is. On 49SBT
    # the three bands have visibly different spreads (p2/p98 of 0.015-0.064 red, 0.025-0.083 green,
    # 0.011-0.049 blue), so an independent stretch pushed near-neutral pixels to R 0.73 / G 0.56 /
    # B 0.71 -- a magenta cast over 1.08% of the panel that looked like a sensor artefact in the
    # scene. A shared low/high preserves the ratios and just sets overall brightness/contrast.
    # Non-positive values are the fill/nodata sentinel and would drag the low end down, so they are
    # excluded from the percentile computation but still clipped into range afterwards.
    finite = rgb[np.isfinite(rgb) & (rgb > 0)]
    if finite.size :
        lo, hi = np.percentile(finite, S2_PCT)
        if hi > lo : rgb = (rgb - lo) / (hi - lo)

    # Clip BEFORE the gamma: x ** 0.65 on a negative x is NaN, and the low tail goes negative by
    # construction after subtracting the 1st percentile.
    return np.clip(rgb, 0, 1) ** S2_GAMMA, out_bounds, out_crs


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
        step = 1 if RENDER_PX is None else max(1, int(round(max(h, w) / RENDER_PX)))
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


def draw_gedi(ax, spec, bounds) :
    """
    Draw the GEDI L4A reference cells of a column: one marker per cell, coloured by its GEDI AGBD on
    the same viridis ramp as the map rows.

    These are exactly the cells the four RMSEs on this column are computed over -- same 2020 filter,
    same per-10 m-cell median, same water mask, same all-four-maps-valid pairing, same display crop
    -- because they come from gedi_scatter.paired_samples, which metrics_4model.py Method A mirrors.
    The count is ASSERTED against spec["rmse"]["n"], the n those published numbers were computed
    with: if the two ever diverge the figure would be plotting a different reference set than it
    reports, silently, so this fails loudly instead.

    Unlike the raster rows this panel is drawn in MAP coordinates (the tile's UTM metres) rather than
    pixel indices, so the axes get explicit limits and an equal aspect; the ground covered, and hence
    the alignment with the panels above and below, is identical.

    Args:
    - ax: the panel's axes.
    - spec (dict): a TILES entry.
    - bounds (tuple): (left, bottom, right, top) of the column window, in the tile's UTM CRS.

    Returns:
    - int: the number of cells drawn.
    """
    cx, cy, ref, _ = paired_samples(spec["tile"])

    n_expected = spec.get("rmse", {}).get("n")
    assert n_expected is None or len(ref) == n_expected, \
        f'{spec["tile"]}: {len(ref)} GEDI cells drawn but the panel RMSEs report n = {n_expected}'

    left, right = min(bounds[0], bounds[2]), max(bounds[0], bounds[2])
    bottom, top = min(bounds[1], bounds[3]), max(bounds[1], bounds[3])
    # paired_samples already restricts to the display window, so this is a bounds check, not a crop.
    assert (cx >= left).all() and (cx <= right).all() and (cy >= bottom).all() and (cy <= top).all(), \
        f'{spec["tile"]}: GEDI cells fall outside the display window {bounds}'

    ax.set_facecolor(GEDI_BG)
    ax.scatter(cx, cy, c = ref, cmap = CMAP, vmin = VMIN, vmax = VMAX,
               s = GEDI_MARKER_PT ** 2, marker = "s", linewidths = 0)
    ax.set_xlim(left, right) ; ax.set_ylim(bottom, top)
    ax.set_aspect("equal")

    # n on the panel, in the same slot and style as the maps' RMSE labels, because n is what makes
    # those RMSEs readable -- 5,086 footprints and 51,995 footprints do not support equal confidence.
    # Same slot and font as rmse_label, but a more opaque box: that one sits over viridis, this one
    # over a pale background, where alpha 0.55 washes the box out to light grey.
    ax.text(0.035, 0.04, f"n = {len(ref):,} cells", transform = ax.transAxes,
            ha = "left", va = "bottom", fontsize = 8.5, color = "white", fontweight = "bold",
            bbox = dict(boxstyle = "round,pad=0.25", fc = "black", ec = "none", alpha = 0.8))
    return len(ref)


def rmse_label(ax, spec, row_key) :
    """Annotate an AGB panel with this map's per-tile GEDI RMSE, bottom-LEFT (the scale bar is
    bottom-right). White text on a translucent dark box so it stays legible over any part of the
    viridis ramp. Silently draws nothing if this row has no number (e.g. the S2 row, or a missing
    entry) rather than printing a placeholder onto the map.

    Args:
        ax: the panel's axes.
        spec (dict): a TILES entry; spec["rmse"] maps row key -> RMSE (t/ha), plus "n".
        row_key (str): the ROWS key of this panel ("aef"/"agbd"/"ssl4eo"/"cci").
    """
    r = spec.get("rmse", {}).get(row_key)
    if r is None : return
    ax.text(0.035, 0.04, f"RMSE {r:.0f} t/ha", transform = ax.transAxes,
            ha = "left", va = "bottom", fontsize = 8.5, color = "white", fontweight = "bold",
            bbox = dict(boxstyle = "round,pad=0.25", fc = "black", ec = "none", alpha = 0.55))


####################################################################################################

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


def column_window(spec) :
    """
    Resolve a column's display window: the AEF footprint, or its "crop" override when set.

    Args:
    - spec (dict): a TILES entry.

    Returns:
    - tuple: (left, bottom, right, top) in the tile's UTM CRS, or None if the AEF panel is missing.
    - CRS: the AEF raster's CRS, or None.
    """
    aef_path = join(PRED_AEF, f'{spec["tile"]}.tif')
    if not exists(aef_path) : return None, None

    with rs.open(aef_path) as s :
        ab = s.bounds
        crs = s.crs

    if spec.get("crop") is None :
        return tuple(ab), crs

    cb = tuple(spec["crop"])
    # Normalise both before comparing: AEF sources are south-up in places, so rasterio can report
    # bounds.top < bounds.bottom and a raw comparison would silently invert.
    a_x0, a_x1 = min(ab.left, ab.right), max(ab.left, ab.right)
    a_y0, a_y1 = min(ab.bottom, ab.top), max(ab.bottom, ab.top)
    c_y0, c_y1 = min(cb[1], cb[3]), max(cb[1], cb[3])
    assert cb[0] >= a_x0 and cb[2] <= a_x1 and c_y0 >= a_y0 and c_y1 <= a_y1, \
        f'{spec["tile"]}: crop {cb} falls outside the AEF window {tuple(ab)}'
    return cb, crs


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

    # Column windows are resolved up front because the grid geometry depends on them: a "crop" need
    # not be square (59GPM is 41.0 x 28.7 km), and with equal-width cells a non-square panel gets
    # letterboxed inside its cell -- the column visibly shrinks and its title floats away from it.
    # Giving each column a width_ratio equal to its own aspect makes every panel the SAME HEIGHT
    # with differing widths, so titles line up and no panel is padded. Ground scale still differs
    # between columns, which is exactly what the per-column scale bar is there to state.
    windows = [column_window(spec) for spec in TILES]
    ratios = []
    for (cb, _) in windows :
        if cb is None :
            ratios.append(1.0)
        else :
            w = abs(cb[2] - cb[0]) ; h = abs(cb[3] - cb[1])
            ratios.append(w / h if h > 0 else 1.0)

    all_ratios = ratios + [0.04 * sum(ratios) / max(1, ncols)]
    wspace, hspace = 0.08, 0.12
    left, right, top, bottom = 0.06, 0.93, 0.90, 0.05

    # Derive the figure HEIGHT from the column aspects instead of hard-coding it. Panels are all
    # the same height, so once the widths are fixed the height follows; hard-coding it left several
    # inches of dead white space below the bottom row as soon as a column stopped being square.
    # matplotlib measures wspace/hspace as a fraction of the AVERAGE cell size, hence the
    # (1 + space * (n - 1) / n) terms.
    fig_w = 10.0
    n_cells = len(all_ratios)
    axes_w = (fig_w * (right - left)) / (1 + wspace * (n_cells - 1) / n_cells)
    panel_h = (axes_w * all_ratios[0] / sum(all_ratios)) / ratios[0]
    fig_h = (panel_h * nrows * (1 + hspace * (nrows - 1) / nrows)) / (top - bottom)

    fig = plt.figure(figsize = (fig_w, fig_h))
    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        width_ratios = all_ratios,
        wspace = wspace, hspace = hspace,
        left = left, right = right, top = top, bottom = bottom,
    )

    missing = []
    for c, spec in enumerate(TILES) :
        tile = spec["tile"]

        # The AEF window defines the extent for the whole column: it is the smaller of the two, and
        # cropping the AGBD tile down to it is what makes the column a like-for-like comparison.
        aef_path = join(PRED_AEF, f"{tile}.tif")
        agbd_path = find_agbd(tile)

        # The column window, already resolved (and bounds-checked) above for the grid geometry.
        col_bounds, col_crs = windows[c]

        for r, row in enumerate(ROWS) :
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]) ; ax.set_yticks([])

            # Every row is cropped to the SAME column window. Previously only the AGBD row was,
            # because the window was by definition the whole AEF panel; now that "crop" can shrink
            # it, the AEF and CCI rows must follow or a column would stop being like-for-like.
            # The CCI crop already sits on the AEF grid, so the same bounds apply unchanged.
            if row["key"] == "s2" :
                path, bounds, crs = find_s2(agbd_path), col_bounds, col_crs
            elif row["key"] == "gedi" :
                # No raster: the footprints are read from the GEDI extract by draw_gedi. The column
                # window is still required, since it is what the reference set is restricted to.
                path, bounds, crs = col_bounds, col_bounds, col_crs
            elif row["key"] == "aef" :
                path, bounds, crs = aef_path, col_bounds, col_crs
            elif row["key"] == "cci" :
                path, bounds, crs = join(PRED_CCI, f"{tile}_CCI.tif"), col_bounds, col_crs
            elif row["key"] == "ssl4eo" :
                path, bounds, crs = join(PRED_SSL4EO, f"{tile}.tif"), col_bounds, col_crs
            else :
                path, bounds, crs = agbd_path, col_bounds, col_crs

            # find_s2 returns a LIST of band paths, so exists() cannot be applied to it directly.
            # ssl4eo joins agbd/s2 in the col_bounds guard: it too must be cropped to the column
            # window (two tiles carry a "crop"), so a None window means the column cannot be drawn.
            gone = (path is None
                    or (row["key"] not in ("s2", "gedi") and not exists(path))
                    or (row["key"] in ("agbd", "s2", "ssl4eo", "gedi") and col_bounds is None))
            if gone :
                missing.append(f'{row["label"]} / {tile}')
                ax.text(0.5, 0.5, f'[{row["label"]}\n{tile}]\nnot found', ha = "center",
                        va = "center", fontsize = 9, color = "0.5", transform = ax.transAxes)
                ax.set_facecolor("0.95")
            elif row["key"] == "s2" :
                # True colour is NOT water-masked, unlike the AGB rows. The mask exists there
                # because an AGB estimate over water is meaningless; a photograph of water is not.
                # Greying the harbours here would delete the very context this row was added for.
                data, p_bounds, p_crs = read_rgb(path, bounds, crs)
                ax.imshow(data, interpolation = "nearest")
                # Scale bar lives on the Sentinel-2 row (the top row): the AEF window makes each
                # column ~40 km across, not the ~110 km tile it is named after, and the S2 panel is
                # the natural place to carry that cue. Every row of a column is cropped to the same
                # window on identical ground, so one bar per column (here) suffices; the white bar
                # is outlined in black in add_scalebar so it stays legible over the imagery too.
                add_scalebar(ax, p_bounds, data.shape[:2])
            elif row["key"] == "gedi" :
                # This panel is in map coordinates, so re-blank the ticks after the limits are set.
                draw_gedi(ax, spec, col_bounds)
                ax.set_xticks([]) ; ax.set_yticks([])
            else :
                data, p_bounds, p_crs = read_panel(path, bounds = bounds, crs = crs)

                # Same WorldCover water mask on every AGB row of the column. Built per panel from
                # that panel's own bounds/shape rather than shared, because the rasters differ by
                # a pixel or two after downsampling and a shared mask would be off-by-one.
                wm = water_mask(tile, p_bounds, p_crs, data.shape)
                if wm is not None : data = np.where(wm, np.nan, data)

                cmap = plt.get_cmap(CMAP).copy()
                cmap.set_bad(MASK_COLOR)
                ax.imshow(data, cmap = cmap, vmin = VMIN, vmax = VMAX, interpolation = "nearest")

                # Per-panel GEDI RMSE, bottom-left. This is where the old two-model header line went:
                # with four maps a header can't hold them all legibly, and a number belongs on the
                # panel it describes anyway.
                rmse_label(ax, spec, row["key"])

            if r == 0 :
                # Column header is just region (tile) now -- per-model GEDI RMSE moved onto each
                # panel (rmse_label), so the old two-line header with its GEDI subtitle is gone and
                # the title pad shrinks back to a normal gap.
                ax.set_title(f'{spec["region"]}  ({tile})',
                             fontsize = 11, fontweight = "bold", pad = 8)
            if c == 0 :
                ax.set_ylabel(row["label"], fontsize = 12, fontweight = "bold", labelpad = 10)

    # Shared colorbar, spanning ONLY the rows it actually describes. The Sentinel-2 row is a true
    # colour composite on no such scale, so running the bar past it would imply t/ha applies there.
    agb_rows = [i for i, row in enumerate(ROWS) if row["key"] != "s2"]
    cbar_ax = fig.add_subplot(gs[min(agb_rows) : max(agb_rows) + 1, ncols])
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
    parser.add_argument("--render-px", type = int, default = None,
                        help = "Downsample panels to about this many px per side. PREVIEW ONLY -- "
                               "it decimates with nearest neighbour and can alias out fine-scale "
                               "striping. Omit for native 10 m resolution, which is the default.")
    args = parser.parse_args()

    if args.render_px is not None :
        RENDER_PX = args.render_px
        print(f"WARNING: rendering at ~{RENDER_PX} px/panel (preview). "
              f"Published output should omit --render-px so panels stay at native 10 m.")

    make_figure(args.out, args.dpi)
