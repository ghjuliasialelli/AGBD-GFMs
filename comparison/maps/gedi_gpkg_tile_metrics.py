"""

Per-tile validation of the map-figure prediction rasters against the GEDI L4A footprints held in the
authoritative GeoPackage, i.e. the reference the maps in `make_map_figure.py` should be judged on.

Why this exists, and how it differs from `tile_metrics.py`
---------------------------------------------------------
`tile_metrics.py` scores against the patch h5 files (`/scratch3/gsialelli/patches/data_subset-*.h5`)
and against `model/eval.py`'s test-split output. Both of those are *derived, subsampled* products:
the patch h5 keeps only the footprints that fell inside a sampled patch, and the eval h5 keeps only
the test split with train overlap removed. They therefore under-count the footprints available on a
tile by a large factor and must not be used to answer "how good is this map here?".

This script goes back to the source instead:

    /scratch3/gsialelli/GEDI/L4A_<regions>-indexed.gpkg   (layer 'main', EPSG:4326, 85 559 440 pts)

which is the full, spatially indexed GEDI L4A point set for the paper's AOIs. It is 25 GB, so it is
never read whole: every read is an r-tree bounding-box read of one tile's extent.

Everything the maps are compared to:
  - AEF            : predictions_maps_aef/nico_film/59620113-1_.../<tile>.tif, band 1, t/ha
  - AGBD-features  : predictions_maps/nico_film/59620098-1_.../*_T<tile>_*.tif, band 1, t/ha
  - ESA CCI v6.0   : the *source* 100 m block CCI/<block>_ESACCI-...-2020-fv6.0.tif -- NOT the
                     bilinearly resampled display crop in CCI/maps/, which differs by up to 77 t/ha.

Correctness notes (each of these has burned this project before)
---------------------------------------------------------------
  - CRS: the gpkg is asserted to be EPSG:4326 and every raster's CRS is read from the raster; points
    are transformed into the raster's CRS, never the reverse, never assumed equal.
  - Rasters are asserted north-up (transform.e < 0) before sampling, so a south-up raster cannot be
    read silently mirrored.
  - Sampling is done at PIXEL CENTRES: the query point is snapped to the centre of the pixel that
    contains it before `src.sample()` is called. Sampling on a pixel boundary breaks ties in
    opposite directions for north-up and south-up grids.
  - nodata (-9999, and each raster's declared nodata) is masked BEFORE any statistic is computed,
    and the number of footprints dropped for it is reported per tile per source.
  - ESA CCI 0 is a real measurement of 0 Mg/ha. Zeros are NOT dropped.
  - Reads are pointwise (`rasterio.sample`); no full 10980 x 10980 raster is ever loaded.
  - Counts are verified over the whole population of the bbox read, never a sample.

GEDI quality filtering
----------------------
None is applied here, because the gpkg is already filtered at creation time by
`BiomassDatasetCreation/GEDI/h5_to_csv_to_shp.py` + `GEDI_settings.py`:
    l4_quality_flag == 1  AND  degrade_flag == 0  AND  sensitivity > 0.95
followed by a clip to the AOI polygons. `--assert_sensitivity` re-checks the last one over the whole
population read. The `l4_quality_flag`/`degrade_flag` columns were dropped at creation, so they
cannot be re-applied; nothing further is filtered.

Usage:
    PROJ_LIB=/scratch2/gsialelli/miniconda3/envs/dwn/share/proj \
    python -u comparison/maps/gedi_gpkg_tile_metrics.py --scope window --scope tile

"""

###################################################################################################
# Imports

import argparse
import glob
import numpy as np
import geopandas as gpd
import rasterio as rs
from pyproj import Transformer
from rasterio.warp import transform_bounds
from os.path import join, exists

###################################################################################################
# Configuration

GPKG  = ("/scratch3/gsialelli/GEDI/L4A_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_"
         "Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana-indexed.gpkg")
LAYER = "main"
AGBD_COL = "agbd"

# The GEDI mission epoch the gpkg's integer `date` column counts days from. Established by matching
# `date` against the acquisition date encoded in the `pattern` (granule) string; the year is read
# from `pattern` directly rather than from this constant, which is kept only for reporting.
GEDI_EPOCH = "2019-04-16"

PRED_AEF  = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film/59620113-1_59620113-1_59620113-1"
PRED_AGBD = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps/nico_film/59620098-1_59620098-1_59620098-1"
CCI_DIR   = "/scratch3/gsialelli/CCI"
CCI_BLOCK = "{block}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv6.0.tif"
WORLDCOVER = "/scratch3/gsialelli/WorldCover/S2/ESA_WorldCover_10m_2020_v100_{tile}.tif"
WATER_CLASSES = (80,)

SOURCES = ["AEF", "AGBD-features", "ESA CCI v6.0"]

###################################################################################################
# Helpers

def find_agbd(tile) :
    """
    Locate the AGBD-features prediction raster of a tile. It is named after the S2 product rather
    than the tile, so it has to be globbed.

    Args:
    - tile (str): the MGRS tile name, e.g. '49SBT'.

    Returns:
    - str or None: the path, or None if absent.
    """
    hits = sorted(glob.glob(join(PRED_AGBD, f"*_T{tile}_*.tif")))
    assert len(hits) <= 1, f"{tile}: {len(hits)} AGBD-features rasters, expected 1: {hits}"
    return hits[0] if hits else None


def cci_block_for(lat, lon) :
    """
    Name the 10 deg ESA CCI block containing a point. Blocks are named by their north-west corner,
    so the latitude rounds up and the longitude down.

    Args:
    - lat (float), lon (float): a point in EPSG:4326.

    Returns:
    - str: e.g. 'S40E170'.
    """
    top  = int(np.ceil(lat / 10.0) * 10)
    left = int(np.floor(lon / 10.0) * 10)
    ns = f"N{top:02d}" if top >= 0 else f"S{-top:02d}"
    ew = f"E{left:03d}" if left >= 0 else f"W{-left:03d}"
    return ns + ew


def raster_bounds_4326(path) :
    """
    The bounds of a raster expressed in EPSG:4326, min/max normalised.

    Args:
    - path (str): the raster.

    Returns:
    - tuple: (west, south, east, north).
    """
    with rs.open(path) as src :
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return (min(w, e), min(s, n), max(w, e), max(s, n))


def read_gedi(bbox) :
    """
    Read every GEDI footprint inside a lat/lon bounding box from the gpkg, using its r-tree index.

    Args:
    - bbox (tuple): (west, south, east, north) in EPSG:4326.

    Returns:
    - dict: 'lat', 'lon', 'agbd', 'year', 'sensitivity' as equal-length arrays.
    """
    g = gpd.read_file(GPKG, layer = LAYER, bbox = bbox, engine = "fiona")
    assert str(g.crs).lower() in ("epsg:4326",), f"gpkg CRS is {g.crs}, expected EPSG:4326"
    if len(g) == 0 :
        return {k : np.array([]) for k in ('lat', 'lon', 'agbd', 'year', 'sensitivity')}

    lon = g.geometry.x.values.astype(np.float64)
    lat = g.geometry.y.values.astype(np.float64)

    # The r-tree read is an *overlap* query; clip to the box exactly, over the whole population.
    keep = (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])

    # The year comes from the granule name in `pattern` (YYYYDDDHHMMSS_...), which is the acquisition
    # itself, not from the derived integer `date` column.
    year = np.array([int(p[:4]) for p in g['pattern'].values], dtype = int)

    return {'lat' : lat[keep], 'lon' : lon[keep],
            'agbd' : g[AGBD_COL].values.astype(np.float64)[keep],
            'year' : year[keep],
            'sensitivity' : g['sensitivit'].values.astype(np.float64)[keep]}


def sample_raster(path, lat, lon, band = 1) :
    """
    Sample one band of a raster at lat/lon points, pointwise and at PIXEL CENTRES.

    The point is transformed into the raster's own CRS, the containing pixel is found, and the
    pixel's centre is what is handed to `src.sample()`. Points outside the raster, or landing on
    nodata / NaN, come back as NaN.

    Args:
    - path (str): the raster.
    - lat (np.ndarray), lon (np.ndarray): points in EPSG:4326.
    - band (int): 1-based band index.

    Returns:
    - np.ndarray: float64 values, NaN where invalid.
    - np.ndarray: bool, True where the point fell inside the raster extent.
    - int: how many in-extent points were dropped for being nodata/NaN.
    """
    out = np.full(len(lat), np.nan)
    inside = np.zeros(len(lat), dtype = bool)
    if len(lat) == 0 : return out, inside, 0

    with rs.open(path) as src :
        assert src.transform.e < 0, f"{path} is south-up (e={src.transform.e}); refusing to sample"
        assert src.transform.b == 0 and src.transform.d == 0, f"{path} is rotated; refusing to sample"

        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy = True)
        x, y = tr.transform(lon, lat)

        left, right = min(src.bounds.left, src.bounds.right), max(src.bounds.left, src.bounds.right)
        bottom, top = min(src.bounds.bottom, src.bounds.top), max(src.bounds.bottom, src.bounds.top)
        inside = (x >= left) & (x < right) & (y > bottom) & (y <= top)
        if inside.sum() == 0 : return out, inside, 0

        # Snap to pixel centres. rasterio's own index() is used so the convention matches the file's
        # transform exactly; the centre is then rebuilt with xy(..., offset='center').
        rows, cols = rs.transform.rowcol(src.transform, x[inside], y[inside])
        rows = np.asarray(rows) ; cols = np.asarray(cols)
        assert rows.min() >= 0 and cols.min() >= 0 and rows.max() < src.height and cols.max() < src.width, \
            f"{path}: in-bounds points mapped outside the grid -- bounds/transform disagree"
        cx, cy = rs.transform.xy(src.transform, rows, cols, offset = 'center')

        vals = np.array([v[0] for v in src.sample(zip(np.asarray(cx), np.asarray(cy)),
                                                  indexes = [band])], dtype = np.float64)
        # Mask nodata BEFORE anything else. -9999 is masked whether or not the file declares it.
        bad = ~np.isfinite(vals) | (vals == -9999)
        if src.nodata is not None : bad = bad | (vals == src.nodata)
        n_nodata = int(bad.sum())
        vals[bad] = np.nan
        out[inside] = vals

    return out, inside, n_nodata


def metrics(pred, truth) :
    """
    The metric set of comparison/agbref/comparison.py, verbatim. No clipping. Units t/ha.

    Args:
    - pred (np.ndarray), truth (np.ndarray): matched values.

    Returns:
    - dict or None: n, bias, rmse, mae, r, r2; None if fewer than two valid pairs.
    """
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    p, t = pred[valid], truth[valid]
    if len(p) < 2 : return None
    return {'n' : int(len(p)),
            'bias' : float(np.mean(p - t)),
            'rmse' : float(np.sqrt(np.mean((p - t) ** 2))),
            'mae' : float(np.mean(np.abs(p - t))),
            'r' : float(np.corrcoef(p, t)[0, 1]),
            'r2' : float(1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2))}


def print_table(title, d, mask, note = "") :
    """
    Print one RMSE/MAE/bias/r/R2/n block, both per-source (each source on its own valid footprints)
    and on the common subset where all three sources are valid.

    Args:
    - title (str): a header.
    - d (dict): must hold 'agbd' and one key per source.
    - mask (np.ndarray): bool, which footprints this block scores.
    - note (str): appended to the header.
    """
    print(f"\n  --- {title} {note}")
    if mask.sum() == 0 :
        print("      no footprints; nothing to score")
        return None

    t_all = d['agbd'][mask]
    print(f"      GEDI reference: n = {mask.sum()}, mean {t_all.mean():.1f}, median "
          f"{np.median(t_all):.1f}, p90 {np.percentile(t_all, 90):.1f}, max {t_all.max():.1f} t/ha")
    print(f"      {'source':16s} {'n':>7s} {'RMSE':>8s} {'MAE':>8s} {'bias':>8s} {'r':>7s} {'R2':>8s}")
    res = {}
    for s in SOURCES :
        res[s] = metrics(d[s][mask], t_all)
        m = res[s]
        if m is None :
            print(f"      {s:16s} {'--':>7s}   (not computable: no valid pairs)")
            continue
        print(f"      {s:16s} {m['n']:7d} {m['rmse']:8.2f} {m['mae']:8.2f} {m['bias']:8.2f} "
              f"{m['r']:7.3f} {m['r2']:8.3f}")

    common = mask.copy()
    for s in SOURCES : common = common & ~np.isnan(d[s])
    if common.sum() >= 2 and common.sum() != mask.sum() :
        tc = d['agbd'][common]
        print(f"      -- same-footprints subset (all three sources valid), n = {int(common.sum())}")
        for s in SOURCES :
            m = metrics(d[s][common], tc)
            if m is None : continue
            print(f"      {s:16s} {m['n']:7d} {m['rmse']:8.2f} {m['mae']:8.2f} {m['bias']:8.2f} "
                  f"{m['r']:7.3f} {m['r2']:8.3f}")
    return res


###################################################################################################
# Code execution

def run(tiles, scopes, years, assert_sensitivity) :
    """
    Compute and print the per-tile, per-model table for every requested scope.

    Args:
    - tiles (list): MGRS tile names.
    - scopes (list): any of 'window' (the AEF raster's extent, i.e. what the figure displays) and
      'tile' (the whole S2 tile, i.e. the AGBD-features raster's extent).
    - years (list or None): GEDI acquisition years to keep; None = all.
    - assert_sensitivity (bool): re-check the >0.95 sensitivity filter over the whole population.
    """
    print(f"gpkg   : {GPKG}")
    print(f"layer  : {LAYER}   AGBD column: {AGBD_COL}   CRS: EPSG:4326")
    print(f"filter : none applied here; the gpkg is pre-filtered l4_quality_flag==1, degrade_flag==0,"
          f" sensitivity>0.95")
    print(f"years  : {'all' if years is None else years}\n")

    summary = []

    for tile in tiles :

        aef_ras  = join(PRED_AEF, f"{tile}.tif")
        agbd_ras = find_agbd(tile)
        print("=" * 104)
        print(f"TILE {tile}")
        if not exists(aef_ras) :
            print(f"  BLOCKED: no AEF raster at {aef_ras}") ; continue
        if agbd_ras is None :
            print(f"  BLOCKED: no AGBD-features raster in {PRED_AGBD}") ; continue

        win_bb  = raster_bounds_4326(aef_ras)
        tile_bb = raster_bounds_4326(agbd_ras)
        print(f"  AEF window (displayed) : lon {win_bb[0]:.5f}..{win_bb[2]:.5f}  lat {win_bb[1]:.5f}..{win_bb[3]:.5f}")
        print(f"  S2 tile (AGBD-features): lon {tile_bb[0]:.5f}..{tile_bb[2]:.5f}  lat {tile_bb[1]:.5f}..{tile_bb[3]:.5f}")

        for scope in scopes :

            bb = win_bb if scope == 'window' else tile_bb
            d = read_gedi(bb)
            n_read = len(d['agbd'])
            print(f"\n  ### scope = {scope}   GEDI footprints in the box: {n_read}")
            if n_read == 0 :
                print("      BLOCKED: the gpkg holds no footprint in this box, so nothing can be scored here.")
                continue

            if assert_sensitivity :
                assert d['sensitivity'].min() > 0.95, \
                    f"{tile}: sensitivity min {d['sensitivity'].min()} <= 0.95 -- gpkg is not filtered as assumed"
                print(f"      sensitivity over the whole population: min {d['sensitivity'].min():.4f} (>0.95, as expected)")

            yr, cnt = np.unique(d['year'], return_counts = True)
            print(f"      acquisition years: {dict(zip(yr.tolist(), cnt.tolist()))}")

            if years is not None :
                keep = np.isin(d['year'], years)
                d = {k : v[keep] for k, v in d.items()}
                print(f"      kept years {years}: {len(d['agbd'])}")
                if len(d['agbd']) == 0 :
                    print("      BLOCKED: no footprint left after the year filter.") ; continue

            # ---- sample every source ---------------------------------------------------------
            d['AEF'], in_aef, nd_aef = sample_raster(aef_ras, d['lat'], d['lon'])
            d['AGBD-features'], in_agbd, nd_agbd = sample_raster(agbd_ras, d['lat'], d['lon'])

            block = cci_block_for(float(np.median(d['lat'])), float(np.median(d['lon'])))
            cci_path = join(CCI_DIR, CCI_BLOCK.format(block = block))
            if exists(cci_path) :
                d['ESA CCI v6.0'], in_cci, nd_cci = sample_raster(cci_path, d['lat'], d['lon'])
            else :
                print(f"      NOTE: CCI block {block} absent at {cci_path}")
                d['ESA CCI v6.0'] = np.full(len(d['agbd']), np.nan) ; in_cci = np.zeros(len(d['agbd']), bool) ; nd_cci = 0

            n = len(d['agbd'])
            print(f"      coverage / nodata (of {n} footprints):")
            for s, ins, ndv in (('AEF', in_aef, nd_aef), ('AGBD-features', in_agbd, nd_agbd),
                                ('ESA CCI v6.0', in_cci, nd_cci)) :
                out_ext = int((~ins).sum())
                print(f"        {s:16s} outside extent {out_ext:6d}   on nodata {ndv:6d}   "
                      f"valid {int(np.sum(~np.isnan(d[s]))):6d}")

            # ---- water --------------------------------------------------------------------------
            wc_path = WORLDCOVER.format(tile = tile)
            water = np.zeros(n, dtype = bool)
            if exists(wc_path) :
                wc, _, _ = sample_raster(wc_path, d['lat'], d['lon'])
                water = np.isin(wc, WATER_CLASSES)
                print(f"      on ESA WorldCover permanent water (class 80): {int(water.sum())}")
            else :
                print(f"      NOTE: no WorldCover for {tile}; the water-masked block cannot be produced")

            all_pts = np.ones(n, dtype = bool)
            res_nw = print_table(f"{tile} / {scope} / NO water masking", d, all_pts)
            res_w  = print_table(f"{tile} / {scope} / water (WC class 80) MASKED OUT", d, ~water) \
                     if exists(wc_path) else None

            summary.append((tile, scope, res_nw, res_w))

    # ---- summary --------------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("SUMMARY -- absolute RMSE / MAE per tile per model (t/ha). 'nw' = no water mask, 'w' = water masked.")
    print(f"{'tile':8s} {'scope':7s} {'wm':3s} {'n(AEF)':>8s} {'RMSE_AEF':>9s} {'MAE_AEF':>8s} "
          f"{'n(AGBD)':>8s} {'RMSE_AGBD':>10s} {'MAE_AGBD':>9s} {'RMSE_CCI':>9s} {'MAE_CCI':>8s}")
    for tile, scope, res_nw, res_w in summary :
        for tag, res in (('nw', res_nw), ('w', res_w)) :
            if res is None : continue
            def f(s, k, w) :
                return f"{res[s][k]:{w}.2f}" if res.get(s) else f"{'--':>{w.split('.')[0]}s}"
            def ni(s) :
                return f"{res[s]['n']:8d}" if res.get(s) else f"{'--':>8s}"
            print(f"{tile:8s} {scope:7s} {tag:3s} {ni('AEF')} {f('AEF','rmse','9')} {f('AEF','mae','8')} "
                  f"{ni('AGBD-features')} {f('AGBD-features','rmse','10')} {f('AGBD-features','mae','9')} "
                  f"{f('ESA CCI v6.0','rmse','9')} {f('ESA CCI v6.0','mae','8')}")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type = str, nargs = "+", default = ["32TNS", "49SBT", "59GPM"])
    parser.add_argument("--scopes", type = str, nargs = "+", default = ["window", "tile"],
                        choices = ["window", "tile"])
    parser.add_argument("--years", type = int, nargs = "+", default = None,
                        help = "GEDI acquisition years to keep; omit for all years in the gpkg.")
    parser.add_argument("--assert_sensitivity", action = "store_true", default = True)
    parser.add_argument("--no_assert_sensitivity", dest = "assert_sensitivity", action = "store_false")
    args = parser.parse_args()
    run(args.tiles, args.scopes, args.years, args.assert_sensitivity)
