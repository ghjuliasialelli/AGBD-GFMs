"""

This script audits how much of each AGBRef plot is actually covered, by the AEF mosaics and by the
AGB prediction rasters derived from them. It exists because the AEF downloads were silently
truncated (see AEF_COVERAGE_TODO.md): download_aoi.py dropped files on transient S3 errors, so most
mosaics - and therefore most predictions - only span part of their 10km plot. A plot value averaged
over a fraction of the plot is a noisy estimate of the plot mean, which costs accuracy.

Run it before and after re-downloading, to check that the re-download actually repaired the gaps.
Each AGBRef plot is a 10km x 10km square, downloaded with a 5100m half-width buffer, so a complete
mosaic spans ~10.2km (~1024 pixels at 10m) and covers ~104 km2.

env: dwn (needs rasterio)

Usage: python check_coverage.py [--aef_dir <dir>] [--pred_dir <dir>] [--n_plots <n>] [--list_worst <n>]

e.g.  python check_coverage.py
      python check_coverage.py --list_worst 20

"""

###################################################################################################
# Imports

import numpy as np
import rasterio
import argparse
from pathlib import Path

###################################################################################################
# Helper functions and global variables

AEF_DIR = '/scratch3/gsialelli/AEF'
PRED_DIR = '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions/nico_film/59620113-1_59620113-1_59620113-1'

# An AGBRef plot is 10km x 10km; the AEF crops are built with a 5100m half-width buffer
PLOT_AREA_KM2 = 10.0 * 10.0
FULL_AREA_KM2 = 10.2 * 10.2


def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Audit the AEF / prediction coverage of the AGBRef plots.')
    parser.add_argument('--aef_dir', type = str, default = AEF_DIR, help = 'Directory holding one sub-folder of AEF files per AGBRef plot.')
    parser.add_argument('--pred_dir', type = str, default = PRED_DIR, help = 'Directory holding the per-plot AGB prediction rasters.')
    parser.add_argument('--n_plots', type = int, default = 780, help = 'Number of AGBRef plots.')
    parser.add_argument('--list_worst', type = int, default = 10, help = 'How many of the worst-covered plots to list.')
    args = parser.parse_args()
    return args.aef_dir, args.pred_dir, args.n_plots, args.list_worst


def raster_area_km2(path):
    """
    This function returns the ground area spanned by a raster, in km2, along with its shape. Note
    that this is the raster's extent, not its valid-pixel count: the AEF truncation removes extent,
    not interior pixels.

    Args:
    - path (Path): the path to the raster.

    Returns:
    - area (float): the area spanned by the raster, in km2, or np.nan if it could not be read.
    - shape (tuple): the (height, width) of the raster, or None.
    """
    try:
        with rasterio.open(path) as src:
            h, w = src.shape
            res_x, res_y = abs(src.res[0]), abs(src.res[1])
            return (w * res_x / 1000) * (h * res_y / 1000), (h, w)
    except Exception:
        return np.nan, None


def summarize(name, areas, list_worst):
    """
    This function prints a summary of the coverage of a set of rasters.

    Args:
    - name (str): the name of the set of rasters.
    - areas (np.ndarray): the area spanned by each raster, in km2.
    - list_worst (int): how many of the worst-covered plots to list.
    """
    present = np.isfinite(areas)
    n = len(areas)
    print(f'\n--- {name} ---')
    print(f'  present: {present.sum()} / {n}   missing: {(~present).sum()}')
    if present.sum() == 0: return

    vals = areas[present]
    print(f'  area spanned ($km^2$), a complete plot is ~{FULL_AREA_KM2:.0f}:')
    for q in [0, 5, 25, 50, 75, 100]:
        print(f'     {q:3d}th pct: {np.percentile(vals, q):7.2f}')
    print(f'     mean {vals.mean():.2f}   min {vals.min():.2f}   max {vals.max():.2f}')
    print(f'  complete (>= {PLOT_AREA_KM2:.0f} km2)  : {(vals >= PLOT_AREA_KM2).sum():4d} / {present.sum()}')
    print(f'  >= 90% of the plot     : {(vals >= 0.9 * PLOT_AREA_KM2).sum():4d} / {present.sum()}')
    print(f'  <  50% of the plot     : {(vals < 0.5 * PLOT_AREA_KM2).sum():4d} / {present.sum()}')
    print(f'  <  25% of the plot     : {(vals < 0.25 * PLOT_AREA_KM2).sum():4d} / {present.sum()}')

    if list_worst > 0:
        order = np.argsort(np.where(present, areas, np.inf))[:list_worst]
        print(f'  worst {list_worst} plots: ' + ', '.join(f'{int(i)} ({areas[i]:.1f})' for i in order))


###################################################################################################
# Code execution

if __name__ == '__main__':

    # Parse arguments and setup variables
    aef_dir, pred_dir, n_plots, list_worst = parse_arguments()
    aef_dir, pred_dir = Path(aef_dir), Path(pred_dir)

    # Measure the AEF mosaics. Only <aoi>/<aoi>.tiff is a mosaic; the other files in the folder are
    # the raw per-file downloads. This is also the only file that inference_aef.py picks up, via its
    # `f.stem == f.parent.name` filter.
    mosaic_areas, pred_areas = np.full(n_plots, np.nan), np.full(n_plots, np.nan)
    n_raw = np.zeros(n_plots, dtype = int)
    for i in range(n_plots):
        mosaic = aef_dir / str(i) / f'{i}.tiff'
        if mosaic.exists(): mosaic_areas[i], _ = raster_area_km2(mosaic)
        folder = aef_dir / str(i)
        if folder.is_dir(): n_raw[i] = len([f for f in folder.glob('*.tiff') if f.stem != f.parent.name])

        pred = pred_dir / f'{i}.tif'
        if pred.exists(): pred_areas[i], _ = raster_area_km2(pred)

    summarize('AEF mosaics  (<aoi>/<aoi>.tiff)', mosaic_areas, list_worst)
    summarize('AGB predictions (<i>.tif)', pred_areas, list_worst)

    # The predictions are derived from the mosaics, so a truncated mosaic explains a truncated
    # prediction. Flag any plot where they disagree, as that points at something else (e.g. the
    # Window(0, 0, 1024, 1024) read in inference_aef.py truncating an oversized mosaic).
    both = np.isfinite(mosaic_areas) & np.isfinite(pred_areas)
    if both.sum():
        print('\n--- mosaic vs prediction ---')
        capped = both & (mosaic_areas > FULL_AREA_KM2 * 1.02) & (pred_areas < mosaic_areas * 0.98)
        print(f'  plots whose mosaic is larger than the 1024x1024 read window: {int(capped.sum())}')
        print(f'  correlation(mosaic area, prediction area): {np.corrcoef(mosaic_areas[both], pred_areas[both])[0, 1]:.3f}')

    print(f'\n  plots with no raw AEF files left on disk: {int((n_raw == 0).sum())} / {n_plots}'
          f'  (they are removed once mosaicked)')
