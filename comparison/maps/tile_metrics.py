"""

Per-tile evaluation of the map-figure rows (AEF / AGBD-features / ESA CCI v6.0) against the GEDI
footprints of the tile -- and, in particular, against the footprints that fall inside the window
each column of `make_map_figure.py` actually displays.

Why this exists: the column titles in `make_map_figure.py` carry a dataset-wide dRMSE taken from a
whole-region eval h5. That number is not measured on the tile the reader is looking at, which makes
it misleading. This script measures the same quantity on the tile, and on the displayed window.

There are two independent code paths here, and they are meant to disagree slightly:

  MODE 'eval' (default, and the one to quote)
    Re-uses the repo's own evaluation output. `model/eval.py` writes one row per GEDI footprint to
    <plots>/nico_film_<run>_..._2019-2020_nooverlap.h5, in the exact order that
    `dataset.initialize_index` + `init_ranges_for_chunk` enumerate (file, tile) pairs. Those two
    functions are imported here -- not reimplemented -- so the per-tile slice is the same one
    eval.py wrote. The reconstruction is asserted twice: the total length must equal the eval h5's,
    and the eval h5's stored `labels` must equal the h5 `agbd` of the rows the index maps to.
    Nothing is re-predicted; these are eval.py's predictions.

  MODE 'raster'
    Samples the published prediction GeoTIFFs (the ones the figure draws) at the footprint
    coordinates. Slower to trust -- a map pixel is a windowed/median-aggregated prediction, not the
    patch-centre prediction eval.py scores -- but it is the only way to score ESA CCI, and it is a
    useful cross-check that the map rasters agree with the eval predictions.

Common to both:
  - The AEF prediction raster defines the displayed window (it is the smallest of the three; the
    figure crops the other two to it). `--scope window` keeps only footprints strictly inside it;
    `--scope tile` uses the whole S2 tile.
  - Footprint coordinates are reconstructed exactly as `dataset.py` (lines 875-878) does it:
        lat = sign(lat_decimal) * (abs(lat_decimal) + lat_offset)
    NOT `offset + decimal`, which disagrees in the southern and western hemispheres (the sign is
    carried by `lat_decimal`; `lat_offset` is an unsigned uint8 magnitude). `--audit` asserts that 100% of a tile's reconstructed
    footprints land inside that tile's own prediction raster, over the whole population.
  - Coordinates are transformed from EPSG:4326 into each raster's own CRS before sampling, and the
    rasters are asserted north-up, so a mirrored raster cannot be sampled silently. Sampling is
    pointwise (`rasterio.sample`); no full tile is ever read (a 10980^2 x 2 float32 tile is ~960 MB).
  - ESA CCI is sampled from the *source* 100 m block in EPSG:4326, not from the reprojected crop in
    CCI/maps, because that crop does not exist for every tile. `--check_cci_crop` reports how far
    the two disagree where both exist.
  - Metrics follow `comparison/agbref/comparison.py` verbatim: bias = mean(pred - truth), RMSE, MAE,
    Pearson r, R^2 = 1 - SS_res/SS_tot about the truth mean. No clipping. Units t/ha.

Usage:
    PROJ_LIB=... python -u tile_metrics.py --mode eval   --scope window --check_cci_crop --audit
    PROJ_LIB=... python -u tile_metrics.py --mode raster --scope window

"""

###################################################################################################
# Imports

import argparse
import glob
import h5py
import numpy as np
import pickle
import sys
import rasterio as rs
from pyproj import Transformer
from rasterio.warp import transform_bounds
from os.path import join, exists, dirname, abspath

# dataset.py lives in model/ and is imported for its index reconstruction. Importing it rather than
# copying it is the whole point: a reimplementation could drift from what eval.py actually wrote.
sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "model"))
from dataset import initialize_index, init_ranges_for_chunk

###################################################################################################
# Configuration

RUN_AEF  = "59620113-1"
RUN_AGBD = "59620098-1"
ARCH     = "nico_film"

# eval.py output, one row per GEDI footprint of the test split, years 2019-2020, drop_overlap=True.
EVAL_DIR = "/scratch3/gsialelli/AGBD-GFMs/data/agbd_lite/eval/results"
EVAL_H5  = "{arch}_{run}_{run}_{run}_2019-2020_nooverlap.h5"
EVAL_YEARS = [2019, 2020]
EVAL_MODE  = "test"
EVAL_DROP_OVERLAPS = True

PRED_AGBD = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps/nico_film/59620098-1_59620098-1_59620098-1"
PRED_AEF  = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film/59620113-1_59620113-1_59620113-1"
CCI_CROPS = "/scratch3/gsialelli/CCI/maps"
CCI_DIR   = "/scratch3/gsialelli/CCI"
CCI_BLOCK = "{block}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv6.0.tif"

PATCHES = "/scratch3/gsialelli/patches"
H5_GLOB = "data_subset-{year}-v4_*.h5"
SPLITS  = join(PATCHES, "biomes_splits_to_name.pkl")

# ESA WorldCover 10 m v100 (2020). `make_map_figure.py` blanks class 80 (permanent water) on every
# row, so a footprint over water is not shown in the figure and is dropped here too.
WORLDCOVER = "/scratch3/gsialelli/WorldCover/S2/ESA_WorldCover_10m_2020_v100_{tile}.tif"
WATER_CLASSES = (80,)

###################################################################################################
# Helpers

def find_agbd(tile) :
    """
    Locate the AGBD-features prediction raster for a tile. It is named after the S2 product, not the
    tile, so it has to be globbed rather than constructed.

    Args:
    - tile (str): the MGRS tile name, e.g. '59GPM'.

    Returns:
    - str or None: path to the prediction, or None if absent.
    """
    hits = sorted(glob.glob(join(PRED_AGBD, f"*_T{tile}_*.tif")))
    return hits[0] if hits else None


def cci_block_for(lat, lon) :
    """
    Name the 10 deg ESA CCI block containing a point. Blocks are named by their NORTH-WEST corner,
    so the latitude rounds up and the longitude down.

    Args:
    - lat (float), lon (float): a point in EPSG:4326.

    Returns:
    - str: the block name, e.g. 'S40E170'.
    """
    top  = int(np.ceil(lat / 10.0) * 10)
    left = int(np.floor(lon / 10.0) * 10)
    ns = f"N{top:02d}" if top >= 0 else f"S{-top:02d}"
    ew = f"E{left:03d}" if left >= 0 else f"W{-left:03d}"
    return ns + ew


def latlon_of(g, idx = None) :
    """
    Reconstruct footprint coordinates from a GEDI group, exactly as dataset.py does.

    Args:
    - g (h5py.Group): the tile's 'GEDI' group.
    - idx (np.ndarray or None): rows to read; None = all.

    Returns:
    - np.ndarray, np.ndarray: latitude and longitude in EPSG:4326.
    """
    sl = slice(None) if idx is None else idx
    lat_d, lat_o = g['lat_decimal'][sl], g['lat_offset'][sl]
    lon_d, lon_o = g['lon_decimal'][sl], g['lon_offset'][sl]
    lat = np.sign(lat_d) * (np.abs(lat_d) + lat_o)
    lon = np.sign(lon_d) * (np.abs(lon_d) + lon_o)
    return lat.astype(np.float64), lon.astype(np.float64)


def eval_ranges(years = EVAL_YEARS, mode = EVAL_MODE, drop_overlaps = EVAL_DROP_OVERLAPS) :
    """
    Rebuild the (start, end, fname, tile) layout of the eval h5 using dataset.py's own functions.

    Args:
    - years (list), mode (str), drop_overlaps (bool): must match the eval.py invocation.

    Returns:
    - dict: the index; list: the ranges; int: the total length.
    """
    fnames = [f'data_subset-{year}-v4_{i}-20.h5' for i in range(20) for year in years]
    index, total = initialize_index(fnames, mode, 1, PATCHES, PATCHES, False, None,
                                    None, None, False, drop_overlaps)
    ranges = init_ranges_for_chunk(index, total, oversampling = False, drop_overlaps = drop_overlaps)
    return index, ranges, total


def gather_eval(tile, index, ranges, preds, min_year = None) :
    """
    Pull every eval-h5 row belonging to a tile, together with the footprint it came from.

    Args:
    - tile (str): the MGRS tile name.
    - index (dict), ranges (list): from eval_ranges().
    - preds (dict): {name: np.ndarray} of eval-h5 prediction columns, plus 'labels'.
    - min_year (int or None): if set, keep only footprints from this year onwards.

    Returns:
    - dict: 'lat', 'lon', 'truth', 'year' and one key per entry of `preds`, all length-n arrays.
    """
    out = {k : [] for k in list(preds.keys()) + ['lat', 'lon', 'truth', 'year']}

    for start, end, fname, tname in ranges :
        if tname != tile : continue
        year = int(fname.split('-')[1])
        if min_year is not None and year < min_year : continue

        tile_data = index[fname][tname]
        rows = np.setdiff1d(np.arange(tile_data['n_total']), tile_data['indices_to_skip'])
        assert len(rows) == end - start, f"{fname}/{tname}: {len(rows)} kept rows vs {end - start} eval rows"

        with h5py.File(join(PATCHES, fname), 'r') as f :
            g = f[tname]['GEDI']
            lat, lon = latlon_of(g, rows)
            truth = g['agbd'][:][rows].astype(np.float64)

        # The reconstruction is only trustworthy if the labels eval.py stored match the h5 rows the
        # index says those eval rows came from. Assert it over every row, not a sample.
        stored = preds['labels'][start:end].astype(np.float64)
        assert np.allclose(stored, truth, atol = 1e-3), \
            f"{fname}/{tname}: eval labels do not match the h5 agbd of the mapped rows -- index reconstruction is wrong"

        out['lat'].append(lat) ; out['lon'].append(lon)
        out['truth'].append(truth) ; out['year'].append(np.full(len(rows), year))
        for k, v in preds.items() : out[k].append(v[start:end].astype(np.float64))

    if len(out['lat']) == 0 : return {k : np.array([]) for k in out}
    return {k : np.concatenate(v) for k, v in out.items()}


def load_gedi_raw(tile, years) :
    """
    Load every GEDI footprint of a tile straight from the patch h5 files (no eval involved).

    Args:
    - tile (str): the MGRS tile name.
    - years (list): the years to load.

    Returns:
    - dict: 'truth', 'lat', 'lon', 'year' as 1-D arrays of equal length.
    """
    truth, lat, lon, year_of = [], [], [], []
    for year in years :
        for fname in sorted(glob.glob(join(PATCHES, H5_GLOB.format(year = year)))) :
            with h5py.File(fname, 'r') as f :
                if tile not in f : continue
                g = f[tile]['GEDI']
                n = len(g['agbd'])
                if n == 0 : continue
                la, lo = latlon_of(g)
                lat.append(la) ; lon.append(lo)
                truth.append(g['agbd'][:].astype(np.float64))
                year_of.append(np.full(n, year))
    if len(truth) == 0 :
        return {k : np.array([]) for k in ('truth', 'lat', 'lon', 'year')}
    return {'truth' : np.concatenate(truth), 'lat' : np.concatenate(lat),
            'lon' : np.concatenate(lon), 'year' : np.concatenate(year_of)}


def sample_raster(path, lat, lon, band = 1, north_up_only = True) :
    """
    Sample one band of a raster at a set of lat/lon points, pointwise.

    The points are transformed into the raster's own CRS -- never the other way round, and never
    assumed to match. Points outside the raster, or on nodata/NaN, come back as NaN.

    Args:
    - path (str): the raster.
    - lat (np.ndarray), lon (np.ndarray): points in EPSG:4326.
    - band (int): 1-based band index.
    - north_up_only (bool): assert transform.e < 0. A south-up raster sampled as north-up yields
      plausible but vertically mirrored values, so this is asserted rather than silently handled.

    Returns:
    - np.ndarray: values as float64, NaN where invalid.
    - np.ndarray: boolean, True where the point fell inside the raster's extent at all.
    """
    out = np.full(len(lat), np.nan)
    inside = np.zeros(len(lat), dtype = bool)
    if len(lat) == 0 : return out, inside

    with rs.open(path) as src :
        if north_up_only :
            assert src.transform.e < 0, f"{path} is south-up (e={src.transform.e}); refusing to sample"

        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy = True)
        x, y = tr.transform(lon, lat)

        # Normalise the bounds before comparing: a south-up raster reports top < bottom and the
        # comparison would silently invert.
        left, right = min(src.bounds.left, src.bounds.right), max(src.bounds.left, src.bounds.right)
        bottom, top = min(src.bounds.bottom, src.bounds.top), max(src.bounds.bottom, src.bounds.top)
        inside = (x >= left) & (x < right) & (y > bottom) & (y <= top)

        if inside.sum() > 0 :
            vals = np.array([v[0] for v in src.sample(zip(x[inside], y[inside]), indexes = [band])],
                            dtype = np.float64)
            if src.nodata is not None : vals[vals == src.nodata] = np.nan
            out[inside] = vals

    return out, inside


def metrics(pred, truth) :
    """
    The metric set of comparison/agbref/comparison.py, verbatim. No clipping.

    Args:
    - pred (np.ndarray), truth (np.ndarray): matched values in t/ha.

    Returns:
    - dict or None: n, bias, rmse, mae, r, r2; None if fewer than two valid pairs.
    """
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    p, t = pred[valid], truth[valid]
    if len(p) < 2 : return None
    return {'n' : int(len(p)),
            'bias' : np.mean(p - t),
            'rmse' : np.sqrt(np.mean((p - t) ** 2)),
            'mae' : np.mean(np.abs(p - t)),
            'r' : np.corrcoef(p, t)[0, 1],
            'r2' : 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)}


###################################################################################################
# Code execution

def run(tiles, mode, scope, years, min_n, check_cci_crop, mask_water, audit) :
    """
    Compute and print the per-tile table.

    Args:
    - tiles (list): MGRS tile names.
    - mode (str): 'eval' (re-use eval.py's predictions) or 'raster' (sample the map GeoTIFFs).
    - scope (str): 'window' (the extent the figure displays) or 'tile' (the whole S2 tile).
    - years (list): GEDI years to score.
    - min_n (int): warn below this many footprints.
    - check_cci_crop (bool): also sample the reprojected CCI crop and report the disagreement.
    - mask_water (bool): drop footprints on WorldCover class 80, as the figure does.
    - audit (bool): assert every footprint of every tile lands inside that tile's own raster.
    """
    with open(SPLITS, 'rb') as f : splits = pickle.load(f)
    split_of = {t : m for m, names in splits.items() for t in names}

    if mode == 'eval' :
        index, ranges, total = eval_ranges()
        preds = {}
        for name, run_id in (('AEF', RUN_AEF), ('AGBD-features', RUN_AGBD)) :
            path = join(EVAL_DIR, EVAL_H5.format(arch = ARCH, run = run_id))
            with h5py.File(path, 'r') as f :
                assert len(f['predictions']) == total, \
                    f"{path} has {len(f['predictions'])} rows, index rebuilds to {total} -- refusing to slice"
                preds[name] = f['predictions'][:]
                if 'labels' not in preds : preds['labels'] = f['labels'][:]
        print(f"eval h5 length {total} matches the rebuilt index for both runs.\n")

    rows = []
    for tile in tiles :

        print("=" * 100)
        print(f"{tile}   split = {split_of.get(tile, 'UNKNOWN')}   mode = {mode}   scope = {scope}")

        aef_ras = join(PRED_AEF, f"{tile}.tif")
        agbd_ras = find_agbd(tile)

        # ---- gather footprints + model predictions -------------------------------------------
        if mode == 'eval' :
            if split_of.get(tile) != EVAL_MODE :
                print(f"  BLOCKED: {tile} is in the '{split_of.get(tile)}' split, so it has no rows in "
                      f"the '{EVAL_MODE}' eval h5. Use --mode raster (and note the numbers are then "
                      f"training-set numbers).")
                continue
            d = gather_eval(tile, index, ranges, preds, min_year = min(years))
            d = {k : v for k, v in d.items() if k != 'labels'}
            keep_year = np.isin(d['year'], years)
            d = {k : v[keep_year] for k, v in d.items()}
            print(f"  GEDI footprints of this tile in the eval h5 (years {years}): {len(d['truth'])}")
        else :
            d = load_gedi_raw(tile, years)
            print(f"  GEDI footprints in the whole S2 tile (years {years}): {len(d['truth'])}")

        if len(d['truth']) == 0 :
            print("  BLOCKED: no GEDI footprints for this tile in these years")
            continue

        # ---- audit: every footprint must land inside the tile's own raster --------------------
        if audit and agbd_ras is not None :
            _, ins = sample_raster(agbd_ras, d['lat'], d['lon'])
            frac = ins.mean()
            print(f"  AUDIT: {int(ins.sum())}/{len(ins)} ({100 * frac:.2f}%) footprints inside the S2 tile raster")
            assert frac == 1.0, f"{tile}: {int((~ins).sum())} footprints fall outside their own tile -- coordinates are wrong"

        # ---- restrict to the displayed window -------------------------------------------------
        if scope == 'window' :
            if not exists(aef_ras) :
                print(f"  BLOCKED: no AEF prediction raster at {aef_ras}, so the displayed window is undefined")
                continue
            _, inside = sample_raster(aef_ras, d['lat'], d['lon'])
            print(f"  ... inside the displayed AEF window: {int(inside.sum())}")
            if inside.sum() == 0 :
                with rs.open(aef_ras) as s : wb = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
                print(f"  BLOCKED: no GEDI footprint falls inside the displayed window.")
                print(f"           window  lon {wb[0]:.4f}..{wb[2]:.4f}  lat {wb[1]:.4f}..{wb[3]:.4f}")
                print(f"           GEDI    lon {d['lon'].min():.4f}..{d['lon'].max():.4f}  "
                      f"lat {d['lat'].min():.4f}..{d['lat'].max():.4f}")
                continue
            d = {k : v[inside] for k, v in d.items()}

        # ---- water mask, identical to the figure's --------------------------------------------
        wc_path = WORLDCOVER.format(tile = tile)
        if mask_water and exists(wc_path) :
            wc, _ = sample_raster(wc_path, d['lat'], d['lon'])
            water = np.isin(wc, WATER_CLASSES)
            print(f"  ... on WorldCover water (class 80, blanked in the figure): {int(water.sum())}")
            d = {k : v[~water] for k, v in d.items()}
        elif mask_water :
            print(f"  NOTE: no WorldCover for {tile}; water not masked")

        # ---- predictions ----------------------------------------------------------------------
        if mode == 'raster' :
            if not exists(aef_ras) or agbd_ras is None :
                print("  BLOCKED: a prediction raster is missing")
                continue
            d['AEF'], _ = sample_raster(aef_ras, d['lat'], d['lon'])
            d['AGBD-features'], _ = sample_raster(agbd_ras, d['lat'], d['lon'])

        block = cci_block_for(float(np.median(d['lat'])), float(np.median(d['lon'])))
        cci_path = join(CCI_DIR, CCI_BLOCK.format(block = block))
        if exists(cci_path) :
            d['ESA CCI v6.0'], _ = sample_raster(cci_path, d['lat'], d['lon'])
        else :
            print(f"  NOTE: no CCI v6.0 block {block} at {cci_path}; CCI not evaluated")
            d['ESA CCI v6.0'] = np.full(len(d['truth']), np.nan)

        if check_cci_crop :
            crop = join(CCI_CROPS, f"{tile}_CCI.tif")
            if exists(crop) :
                cci_c, _ = sample_raster(crop, d['lat'], d['lon'])
                both = ~np.isnan(d['ESA CCI v6.0']) & ~np.isnan(cci_c)
                if both.sum() > 0 :
                    diff = np.abs(d['ESA CCI v6.0'][both] - cci_c[both])
                    print(f"  CCI source block vs the displayed crop on {int(both.sum())} pts: "
                          f"mean|diff| = {diff.mean():.3f}, max|diff| = {diff.max():.3f} t/ha")
            else :
                print(f"  NOTE: no displayed CCI crop for {tile} ({crop}); cannot cross-check")

        # ---- like-for-like scoring ------------------------------------------------------------
        # Every source is scored on the SAME footprints. Scoring each on its own valid subset would
        # put the three rows on different populations and the column would stop comparing.
        sources = ['AEF', 'AGBD-features', 'ESA CCI v6.0']
        common = ~np.isnan(d['truth'])
        for s in sources : common = common & ~np.isnan(d[s])
        n_common = int(common.sum())
        no_cci = ~np.isnan(d['truth']) & ~np.isnan(d['AEF']) & ~np.isnan(d['AGBD-features'])

        print(f"  scored footprints: {n_common} (all three sources valid); "
              f"{int(no_cci.sum())} with the two models only")
        if n_common < min_n :
            print(f"  *** WARNING: n = {n_common} < {min_n}. Too few footprints for a stable RMSE -- "
                  f"report as indicative only, or not at all. ***")
        if n_common == 0 : continue

        t = d['truth'][common]
        print(f"  GEDI reference on that subset: mean {t.mean():.1f}, median {np.median(t):.1f}, "
              f"p90 {np.percentile(t, 90):.1f}, max {t.max():.1f} t/ha")
        print()
        print(f"  {'source':16s} {'n':>6s} {'RMSE':>8s} {'MAE':>8s} {'bias':>8s} {'r':>7s} {'R2':>8s}")
        res = {}
        for s in sources :
            res[s] = metrics(d[s][common], t)
            m = res[s]
            if m is None : print(f"  {s:16s} {'--':>6s}   (not computable)") ; continue
            print(f"  {s:16s} {m['n']:6d} {m['rmse']:8.2f} {m['mae']:8.2f} {m['bias']:8.2f} "
                  f"{m['r']:7.3f} {m['r2']:8.3f}")

        drmse = res['AGBD-features']['rmse'] - res['AEF']['rmse']
        print(f"\n  dRMSE (AGBD-features minus AEF; positive = AEF better) = {drmse:+.2f} t/ha  on n = {n_common}")
        rows.append((tile, split_of.get(tile, '?'), res, drmse))
        print()

    print("=" * 100)
    print(f"SUMMARY -- mode = {mode}, scope = {scope}, years = {years}")
    print(f"{'tile':8s} {'split':6s} {'n':>7s} {'RMSE AEF':>9s} {'RMSE AGBD':>10s} {'RMSE CCI':>9s} {'dRMSE':>8s}")
    for tile, split, res, drmse in rows :
        cci = f"{res['ESA CCI v6.0']['rmse']:9.2f}" if res['ESA CCI v6.0'] else f"{'--':>9s}"
        print(f"{tile:8s} {split:6s} {res['AEF']['n']:7d} {res['AEF']['rmse']:9.2f} "
              f"{res['AGBD-features']['rmse']:10.2f} {cci} {drmse:+8.2f}")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type = str, nargs = "+", default = ["59GPM", "32TNS", "49SBT"])
    parser.add_argument("--mode", type = str, default = "eval", choices = ["eval", "raster"])
    parser.add_argument("--scope", type = str, default = "window", choices = ["window", "tile"])
    parser.add_argument("--years", type = int, nargs = "+", default = [2019, 2020])
    parser.add_argument("--min_n", type = int, default = 100)
    parser.add_argument("--check_cci_crop", action = "store_true")
    parser.add_argument("--audit", action = "store_true")
    parser.add_argument("--mask_water", action = "store_true", default = True)
    parser.add_argument("--no_mask_water", dest = "mask_water", action = "store_false")
    args = parser.parse_args()
    run(args.tiles, args.mode, args.scope, args.years, args.min_n,
        args.check_cci_crop, args.mask_water, args.audit)
