"""

Stage A of the SSL4EO-MoCo map pipeline: extract the Sentinel-2 reflectance window that
`inference_ssl4eo.py` runs on, and cache it to disk.

This exists as a separate process for one reason: `process_S2_tile` (via `inference_helper.py`)
needs `skimage`, which the `pangaea-bench` conda env does not have, while the model needs `torch
2.4` + `hydra` + `timm`, which the `agbd` env does not have. Rather than install into either env
and risk breaking a third thing, the pipeline is split at this cache file. Run this one under
`agbd`, then `inference_ssl4eo.py` under `pangaea-bench`.

The cache is deliberately the RAW surface-reflectance window, not a normalised one: the model's
preprocessing chain resizes 25x25 -> 224x224 *before* normalising, so normalisation cannot be
hoisted out to tile level without changing what the model sees.

Reflectance conversion mirrors `pangaea/datasets/agbd.py` (not `inference_agbd.py`), because that
is what the SSL4EO-MoCo model was trained on:
    sr = (DN - boa_offset * 1000) / 10000 ;  sr[DN == 0] = 0 ;  sr[sr < 0] = 0

Band extraction itself goes through `process_S2_tile`, i.e. the exact same code path that produced
the AGBD-features panel, so the two panels differ by the model and nothing else.

Usage (agbd env):
    python -u model/cache_s2_window.py --tile_name 49SBT \
        --product_name S2A_MSIL2A_20200826T032541_N0500_R018_T49SBT_20230418T063839 \
        --out /scratch3/gsialelli/ssl4eo_maps/cache/49SBT.npz

"""

###################################################################################################
# Imports

import argparse
import json
import os
import sys
import time
from os import makedirs
from os.path import abspath, dirname, exists, join

import numpy as np
import rasterio as rs
from rasterio.windows import Window, from_bounds

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'model'))
from config import get_paths
from inference_helper import process_S2_tile

###################################################################################################
# Configuration

# The AEF predictions define the 40 km window every column of the map figure is cropped to; see
# make_map_figure.py. Taking the window from here (rather than re-deriving it) is what keeps the
# SSL4EO panel on the same ground as the AEF and AGBD-features panels.
PRED_AEF = ('/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/'
            'nico_film/59620113-1_59620113-1_59620113-1')

# Sentinel-2 band naming. process_S2_tile keys the bands the way the .SAFE product does (B01, B09);
# the pangaea AGBD dataset config names them B1, B9. Same bands, different spelling. B10 is absent
# from L2A entirely -- the encoder wants 13 bands and BandPadding zero-fills the missing one, so it
# is deliberately NOT listed here.
PANGAEA_TO_SAFE = {
    'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
    'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B9': 'B09', 'B11': 'B11', 'B12': 'B12',
}

# Band order the model expects, i.e. dataset.bands.optical in the run config. Order matters: the
# preprocessor's BandFilter/BandPadding index by position, so a permutation here would silently
# feed the model the wrong spectra.
BAND_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

# Scene classification classes treated as invalid, matching inference_agbd.py:
# 0 = no data, 1 = saturated/defective, 6 = water, 11 = snow/ice.
SCL_INVALID = (0, 1, 6, 11)

###################################################################################################
# Helpers

def parse_args() :
    """
    Returns the parsed command-line arguments.
    """

    p = argparse.ArgumentParser(description = __doc__,
                                formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tile_name', type = str, required = True,
                   help = 'MGRS tile, e.g. 49SBT.')
    p.add_argument('--product_name', type = str, default = '',
                   help = 'Sentinel-2 L2A product. Empty = take the least-cloudy product for '
                          '--year from mapping_2019-2020-v2.pkl. Pass this explicitly to match '
                          'the product the AGBD-features panel used.')
    p.add_argument('--year', type = int, default = 2020,
                   help = 'Year, used only when --product_name is empty.')
    p.add_argument('--out', type = str, required = True,
                   help = 'Output .npz path.')
    p.add_argument('--bounds', type = float, nargs = 4, default = None,
                   metavar = ('LEFT', 'BOTTOM', 'RIGHT', 'TOP'),
                   help = 'Window in the tile CRS. Default: the AEF prediction window for the '
                          'tile, which is what the map figure crops every panel to.')
    p.add_argument('--margin_px', type = int, default = 12,
                   help = 'Context margin in pixels on each side. The model reads a 25x25 patch '
                          'per output pixel, so every output pixel needs 12 px of context to have '
                          'a full patch; without it the edge would need padding the model never '
                          'saw in training. 12 = (25 - 1) // 2.')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'Recompute even if the output already exists.')
    return p.parse_args()


def resolve_product(tile_name, product_name, year, path_tiles) :
    """
    Resolve which Sentinel-2 product to read, and check it is actually on disk.

    The mapping pickle records products under the N9999 processing baseline while the downloads are
    often N0500. Those are the same acquisition reprocessed, so the match is made on the
    acquisition timestamp rather than the whole product name.

    Args:
    - tile_name (str): MGRS tile.
    - product_name (str): explicit product, or '' to look one up.
    - year (int): year, used only for the lookup.
    - path_tiles (str): directory holding the .SAFE / .zip products.

    Returns:
    - str: the product name as it exists on disk.
    """

    import glob, pickle

    if product_name == '' :
        with open(join(path_tiles, 'mapping_2019-2020-v2.pkl'), 'rb') as f :
            product_name = pickle.load(f)[year][tile_name]
        print(f'Least-cloudy product for {year}/{tile_name}: {product_name}')

    if exists(join(path_tiles, product_name + '.SAFE')) or exists(join(path_tiles, product_name + '.zip')) :
        return product_name

    # Fall back to matching on the acquisition timestamp, which is stable across baselines.
    stamp = product_name.split('_')[2]
    hits = sorted(glob.glob(join(path_tiles, f'*_{stamp}_*_T{tile_name}_*.SAFE')) +
                  glob.glob(join(path_tiles, f'*_{stamp}_*_T{tile_name}_*.zip')))
    hits = sorted({os.path.basename(h).rsplit('.', 1)[0] for h in hits})
    assert hits, (f'No product on disk for {tile_name} at acquisition {stamp}. Looked for '
                  f'{product_name} and for any baseline of the same acquisition in {path_tiles}.')
    if len(hits) > 1 :
        print(f'WARNING: {len(hits)} baselines for acquisition {stamp}: {hits}. Taking {hits[0]}.')
    print(f'{product_name} is not on disk; using same-acquisition product {hits[0]}.')
    return hits[0]


def window_from_bounds(bounds, transform, width, height, margin_px) :
    """
    Convert target bounds into an integer pixel window on the tile grid, grown by `margin_px`.

    The window is rounded OUTWARD so the requested bounds are fully covered, then clipped to the
    tile. The clip is reported rather than silently applied: a window that hits the tile edge has
    less context than it asked for, and the caller needs to know.

    Args:
    - bounds (tuple): (left, bottom, right, top) in the tile CRS.
    - transform (Affine): the tile's transform.
    - width (int): tile width in px.
    - height (int): tile height in px.
    - margin_px (int): context margin to add on each side.

    Returns:
    - tuple: (col_off, row_off, win_w, win_h) of the margin-inclusive window.
    - tuple: (left_margin, top_margin, right_margin, bottom_margin) actually obtained, in px.
    """

    win = from_bounds(*bounds, transform = transform)
    col0 = int(np.floor(win.col_off))
    row0 = int(np.floor(win.row_off))
    col1 = int(np.ceil(win.col_off + win.width))
    row1 = int(np.ceil(win.row_off + win.height))

    # Grow by the margin, then clip to the tile. The margin actually obtained is the gap between
    # the clipped window and the un-grown target window, which is <= margin_px at a tile edge.
    c_col0, c_row0 = max(0, col0 - margin_px), max(0, row0 - margin_px)
    c_col1, c_row1 = min(width, col1 + margin_px), min(height, row1 + margin_px)

    margins = (col0 - c_col0, row0 - c_row0, c_col1 - col1, c_row1 - row1)

    return (c_col0, c_row0, c_col1 - c_col0, c_row1 - c_row0), margins


###################################################################################################
# Main

def main() :

    args = parse_args()

    if exists(args.out) and not args.overwrite :
        print(f'{args.out} already exists; pass --overwrite to recompute. Nothing to do.')
        return

    paths = get_paths(local = True)
    t0 = time.time()

    # Which product ------------------------------------------------------------------------------
    product = resolve_product(args.tile_name, args.product_name, args.year, paths['tiles'])

    # Which window -------------------------------------------------------------------------------
    if args.bounds is not None :
        target_bounds = tuple(args.bounds)
        bounds_src = 'command line'
    else :
        aef_path = join(PRED_AEF, f'{args.tile_name}.tif')
        assert exists(aef_path), (f'No AEF prediction at {aef_path}, so there is no window to '
                                  f'match. Pass --bounds explicitly, or run the AEF inference first.')
        with rs.open(aef_path) as src :
            b = src.bounds
            # South-up rasters report top < bottom; normalise before using the values.
            target_bounds = (min(b.left, b.right), min(b.top, b.bottom),
                             max(b.left, b.right), max(b.top, b.bottom))
            aef_crs = src.crs
        bounds_src = f'AEF window {aef_path}'
    print(f'Target window from {bounds_src}: {target_bounds}')

    # Read the Sentinel-2 tile -------------------------------------------------------------------
    print(f'Processing {product} ...')
    (transform, upsampling_shape, s2_bands, crs, bounds,
     boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta) = process_S2_tile(product, paths['tiles'])
    del lat_cos, lat_sin, lon_cos, lon_sin
    height, width = upsampling_shape[0], upsampling_shape[1]
    print(f'Tile: {width} x {height} px, {crs}, boa_offset = {boa_offset}')

    if args.bounds is None :
        assert crs == aef_crs, (f'AEF window is in {aef_crs} but the S2 tile is in {crs}. Bounds '
                                f'would be compared numerically across CRSs and land ~hundreds of '
                                f'km away. Reproject the bounds before passing them via --bounds.')

    (col_off, row_off, win_w, win_h), margins = window_from_bounds(
        target_bounds, transform, width, height, args.margin_px)
    print(f'Window: col_off = {col_off}, row_off = {row_off}, {win_w} x {win_h} px')
    print(f'Context margins obtained (l, t, r, b): {margins} px (requested {args.margin_px})')
    if min(margins) < args.margin_px :
        print('WARNING: the window hits the tile edge, so some output pixels will have less than '
              'the full 25x25 context. inference_ssl4eo.py will mark those invalid.')

    # Crop and convert to surface reflectance ----------------------------------------------------
    scl = s2_bands.pop('SCL')[row_off : row_off + win_h, col_off : col_off + win_w]

    # Written band-by-band into a preallocated float32 array, and each full-tile band is POPPED
    # from the dict as it is consumed. process_S2_tile hands back all 13 bands at full tile size
    # (~3.1 GB as uint16) while only a ~40 km window of each is wanted, so holding the whole dict
    # to the end wastes several GB for no reason. Popping frees ~240 MB per band as we go.
    #
    # Note there is no concatenate here and nothing is float64: the crop happens BEFORE the cast,
    # so the float32 target is the only full-size float allocation. (inference_agbd.py had to be
    # fixed for exactly the opposite pattern -- a mixed-dtype np.concatenate that promoted a whole
    # 31-channel tile to float64 and OOM-killed the box.)
    sr = np.empty((len(BAND_ORDER), win_h, win_w), dtype = np.float32)
    for i, band in enumerate(BAND_ORDER) :
        full = s2_bands.pop(PANGAEA_TO_SAFE[band])
        dn = full[row_off : row_off + win_h, col_off : col_off + win_w].astype(np.float32)
        del full
        # agbd.py:294-296 -- the exact conversion the model was trained on.
        v = (dn - boa_offset * 1000) / 10000
        v[dn == 0] = 0
        v[v < 0] = 0
        sr[i] = v
    del s2_bands

    # Validity. Nodata is decided on the RAW DN (a zero in every band), before any conversion could
    # erase the sentinel; SCL adds water / snow / saturated.
    valid = ~np.isin(scl, SCL_INVALID)
    print(f'Valid: {100 * valid.mean():.2f}% of the window '
          f'({100 * (~np.isin(scl, (0, 1))).mean():.2f}% excluding water/snow)')

    win_transform = rs.windows.transform(Window(col_off, row_off, win_w, win_h), transform)

    # Save ---------------------------------------------------------------------------------------
    makedirs(dirname(abspath(args.out)), exist_ok = True)
    tmp = args.out + '.tmp.npz'
    np.savez(tmp, sr = sr, scl = scl.astype(np.uint8), valid = valid)
    os.replace(tmp, args.out)

    side = {
        'tile': args.tile_name,
        'product': product,
        'crs': str(crs),
        'transform': list(win_transform)[:6],
        'bounds': list(rs.windows.bounds(Window(col_off, row_off, win_w, win_h), transform)),
        'target_bounds': list(target_bounds),
        'bounds_source': bounds_src,
        'window': {'col_off': col_off, 'row_off': row_off, 'width': win_w, 'height': win_h},
        'margin_px': args.margin_px,
        'margins_obtained': list(margins),
        'band_order': BAND_ORDER,
        'boa_offset': int(boa_offset),
        'tile_shape': [height, width],
        'valid_frac': float(valid.mean()),
    }
    with open(args.out.replace('.npz', '.json'), 'w') as f :
        json.dump(side, f, indent = 2)

    print(f'Wrote {args.out} ({sr.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f} s')


if __name__ == '__main__' :
    main()
