"""

This script builds a multi-temporal version of AGBD-Lite: instead of a single Sentinel-2 L2A patch per
GEDI footprint, it extracts `--num_timesteps` 25x25 patches at the *same* location, from different dates.

The footprint set, the train/val/test split, and every non-Sentinel feature are identical to AGBD-Lite;
only the Sentinel-2 datasets gain a temporal axis:

    S2_bands                        (N, T, 25, 25, 12)  uint16
    Sentinel_metadata/S2_date       (N, T)              int16
    Sentinel_metadata/S2_boa_offset (N, T)              uint8

Timestep selection, per footprint:
    - the `anchor`, i.e. the product AGBD itself paired with that footprint (recorded in S2_date). It is
      the one that passed AGBD's SCL quality gate, so it is kept no matter how cloudy the alternatives are.
    - the remaining slots are filled with the least-cloudy other products available for the tile, ranked
      on the SCL statistics computed by audit_s2_products.py.
    - if the tile has fewer than T products, the anchor patch is replicated to fill the remaining slots.
      The *dates* are never faked: a replicated slot repeats the anchor's true date, so padded samples are
      identifiable after the fact with `len(unique(S2_date[i])) < T`.
    - the T slots are ordered by acquisition date.

Geometry, resampling and radiometry replicate BiomassDatasetCreation/patches/create_patches.py exactly,
so that the anchor slot reproduces the AGBD patch bit-for-bit. That equality is asserted for every sample
whose anchor product is still on disk at the processing baseline AGBD used (S2_pbn); see --min_exact_frac.

Execution:

    python -u gen_multitemporal.py --mode train --num_timesteps 3 --workers 6

env: agbd

"""

############################################################################################################################
# IMPORTS

import os
import re
import sys
import csv
import glob
import json
import pickle
import zipfile
import argparse
import datetime as dt
import traceback
from os.path import join, basename, isfile, isdir, exists
from collections import defaultdict, Counter
from multiprocessing import Pool

import h5py
import numpy as np
import rasterio as rs
from rasterio.transform import rowcol
from pyproj import Transformer
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt

############################################################################################################################
# CONSTANTS

# Paths. The patches directory holds the parent AGBD shards, the split mapping, and the AGBD-Lite indices.
PATH_S2 = join('/scratch3', 'gsialelli', 'S2_L2A')
HERE = os.path.dirname(os.path.abspath(__file__))

# The original extraction code is imported and CALLED, not re-implemented. Re-implementing it is how a
# missing fill_nan_with_nearest silently changed every patch bordering nodata.
PATH_BIOMASS = join('/scratch3', 'gsialelli', 'BiomassDatasetCreation')
sys.path.insert(0, join(PATH_BIOMASS, 'patches'))
sys.path.insert(0, join(PATH_BIOMASS, 'Sentinel'))
import create_patches as cp

# As taken from BiomassDatasetCreation/patches/create_patches.py
S2_L2A_BANDS = {'10m': ['B02', 'B03', 'B04', 'B08'],
                '20m': ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
                '60m': ['B01', 'B09']}
BAND_ORDER = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
BAND_RES = {b: res for res, bands in S2_L2A_BANDS.items() for b in bands}
NODATAVAL_S2 = 0
PATCH_SIZE = (25, 25)

# From Sentinel_settings.py. Note that S2 products acquired *before* this date yield a negative S2_date,
# which is why the field is int16 and not uint16.
GEDI_START_MISSION = '2019-04-17'

# Products that are on disk but truncated (verified unreadable). Both tiles have >= 4 readable alternatives.
BAD_PRODUCTS = {
    'S2A_MSIL2A_20200111T160641_N0500_R097_T17QPD_20230430T053825',
    'S2B_MSIL2A_20190702T073619_N0500_R092_T35MQR_20230706T145414',
}

PRODUCT_RE = re.compile(r'^(S2[ABC]_MSIL2A_(\d{8})T\d{6}_N(\d{4})_R\d{3}_T(\w{5})_\d{8}T\d{6})\.(zip|SAFE)$')

############################################################################################################################
# Helper functions


def _parser() :
    """
    This function parses command-line arguments for the multi-temporal extraction script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, required = True, choices = ['train', 'val', 'test'],
                        help = 'Set for which to generate the multi-temporal dataset.')
    parser.add_argument('--num_timesteps', type = int, default = 3,
                        help = 'Number of Sentinel-2 timesteps per footprint.')
    parser.add_argument('--path_patches', type = str, default = join('/scratch3', 'gsialelli', 'patches'),
                        help = 'Directory holding the parent AGBD shards and the index/split pickles.')
    parser.add_argument('--path_s2', type = str, default = PATH_S2,
                        help = 'Directory holding the Sentinel-2 L2A products.')
    parser.add_argument('--audit', type = str, default = join(HERE, 'scl_audit.csv'),
                        help = 'CSV of per-product SCL statistics, as written by audit_s2_products.py.')
    parser.add_argument('--output_dir', type = str, default = '',
                        help = 'Output directory. Defaults to <path_patches>/AGBD-Lite-MT.')
    parser.add_argument('--workers', type = int, default = 6,
                        help = 'Number of tiles to extract in parallel. Each worker peaks around 1 GB.')
    parser.add_argument('--max_nodata', type = float, default = 0.95,
                        help = 'Reject a candidate product whose SCL nodata fraction exceeds this. This is '
                               'deliberately loose: whether a product covers a given footprint is decided '
                               'per footprint, on its actual window, not on this tile-wide fraction.')
    parser.add_argument('--max_candidates', type = int, default = 8,
                        help = 'How many of the least-cloudy candidate products to test per tile.')
    parser.add_argument('--min_valid_frac', type = float, default = 0.5,
                        help = 'A candidate covers a footprint if at least this fraction of its 25x25 '
                               'window is non-nodata.')
    parser.add_argument('--check_per_date', type = int, default = 64,
                        help = 'How many anchors to re-extract and compare per (tile, date).')
    parser.add_argument('--max_cloud', type = float, default = 0.80,
                        help = 'Reject a candidate product whose cloud+shadow fraction exceeds this.')
    parser.add_argument('--max_padded_frac', type = float, default = 0.05,
                        help = 'Fail the build if more than this fraction of samples had to be padded.')
    parser.add_argument('--min_checked_frac', type = float, default = 0.5,
                        help = 'Warn if the bit-for-bit anchor check covers less than this fraction of '
                               'anchors. It is a warning, not a gate: on this machine most of the '
                               'products AGBD used are simply no longer on disk.')
    parser.add_argument('--min_exact_frac', type = float, default = 1.0,
                        help = 'Fail the build if fewer than this fraction of reproducible anchors match '
                               'the parent AGBD patch bit-for-bit. Only samples whose anchor product is on '
                               'disk at the baseline AGBD recorded (S2_pbn) count as reproducible.')
    parser.add_argument('--path_dem', type = str, default = join('/scratch3', 'gsialelli', 'ALOS'),
                        help = 'Directory holding DEM_<tile>.tif, used as the registration key.')
    parser.add_argument('--path_lc', type = str, default = join('/scratch3', 'gsialelli', 'LC'),
                        help = 'Directory holding LC_<tile>_<year>.tif, the tie-breaker.')
    parser.add_argument('--max_unresolved_frac', type = float, default = 0.01,
                        help = 'Fail the build if more than this fraction of windows cannot be pinned.')
    parser.add_argument('--tiles', type = str, nargs = '*', default = None,
                        help = 'Restrict the build to these tiles (debugging).')
    parser.add_argument('--overwrite', action = 'store_true',
                        help = 'Re-extract tiles whose shard already exists.')
    return parser.parse_args()


def decode_latlon(decimal, offset) :
    """
    This function decodes the GEDI coordinates, which are stored as a *signed* decimal part and an
    *unsigned* integer offset. Getting this wrong flips the hemisphere for every footprint with a
    negative coordinate, silently.

    Args:
    - decimal: array of float32, signed fractional part of the coordinate.
    - offset: array of uint8, unsigned integer part of the coordinate.

    Returns:
    - array of float64, the decoded coordinate.
    """
    return np.sign(decimal) * (np.abs(decimal) + offset)


def days_to_date(num_days) :
    """
    This function converts a number of days since the start of the GEDI mission into a date.

    Args:
    - num_days: int, number of days since GEDI_START_MISSION.

    Returns:
    - datetime.date
    """
    return dt.datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d').date() + dt.timedelta(days = int(num_days))


def date_to_days(date) :
    """
    This function converts a date into the number of days since the start of the GEDI mission.

    Args:
    - date: datetime.date

    Returns:
    - int, number of days since GEDI_START_MISSION. Negative for products predating the mission.
    """
    return (date - dt.datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d').date()).days


def list_products(path_s2) :
    """
    This function indexes the Sentinel-2 L2A products on disk. Products come in four shapes on this
    machine -- zipped or unzipped, with .jp2 or .tif band files -- and the directory also holds
    ancillary rasters (*_CLOUDS.tif, *_MASK.tif) whose names begin with a product name; matching the
    full product name against a strict pattern is what keeps those out.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - products: dict, {tile: {date (YYYYMMDD): [(baseline, product_name), ...]}}.
    """
    products = defaultdict(lambda: defaultdict(list))
    for entry in os.listdir(path_s2) :
        match = PRODUCT_RE.match(entry)
        if match is None : continue
        name, date, baseline, tile, _ = match.groups()
        if name in BAD_PRODUCTS : continue
        products[tile][date].append((baseline, name))
    return products


def band_path(path_s2, product, band) :
    """
    This function resolves the path to a band of a Sentinel-2 L2A product, and returns a path that
    rasterio can open. It tolerates every layout present on disk: unzipped .SAFE directories, zipped
    products, .jp2 and .tif band files, and zips that store full absolute paths internally (so that
    the .SAFE directory is not at the root of the archive).

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product (without extension).
    - band: string, name of the band (e.g. 'B02').

    Returns:
    - string path openable by rasterio, or None if the band could not be found.
    """
    res = BAND_RES[band]
    safe = join(path_s2, product + '.SAFE')
    if isdir(safe) :
        hits = glob.glob(join(safe, 'GRANULE', '*', 'IMG_DATA', f'R{res}', f'*_{band}_{res}.*'))
        return hits[0] if hits else None
    archive = join(path_s2, product + '.zip')
    if isfile(archive) :
        with zipfile.ZipFile(archive) as zf :
            names = [n for n in zf.namelist()
                     if f'/R{res}/' in n and n.endswith((f'_{band}_{res}.jp2', f'_{band}_{res}.tif'))]
        return f'/vsizip/{archive}/{names[0]}' if names else None
    return None


def upsampling_with_nans(image, upsampling_shape, nan_value, order) :
    """
    This function upsamples the image to the `upsampling_shape`, keeping the nodata pixels nodata.
    It is a verbatim port of BiomassDatasetCreation/patches/create_patches.py; the `order` values used
    there are 1 (bilinear) for the reflectance bands -- despite what that script's docstring says -- and
    0 (nearest) for the scene classification mask. Changing it breaks bit-for-bit equality with AGBD.

    Args:
    - image: 2d array, image to upsample.
    - upsampling_shape: tuple of ints, shape to upsample to.
    - nan_value: scalar, the nodata value.
    - order: int, interpolation order.

    Returns:
    - 2d array, the upsampled image.
    """
    assert not np.isinf(image).any(), 'There are inf values in the data.'
    nan_mask = np.isnan(image) if np.isnan(nan_value) else (image == nan_value)
    if np.count_nonzero(nan_mask) == 0 :
        return resize(image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)
    # The nodata pixels are filled with their nearest valid neighbour *before* resizing. Skipping this
    # lets the interpolation pull nodata into the pixels bordering a nodata region, which differs from
    # AGBD by a few DN in a one-pixel band along every swath edge -- invisible in a plot, and caught
    # only by the bit-for-bit anchor check.
    indices = distance_transform_edt(nan_mask, return_distances = False, return_indices = True)
    non_nan_image = image[tuple(indices)]
    upsampled_image = resize(non_nan_image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)
    upsampled_nan_mask = resize(nan_mask.astype(float), upsampling_shape, order = 0, mode = 'edge') > 0.5
    return np.where(upsampled_nan_mask, nan_value, upsampled_image)


def load_band(path_s2, product, band, reference_shape) :
    """
    This function reads one band of a Sentinel-2 L2A product and returns it at 10m resolution, upsampling
    the 20m and 60m bands to the shape of the 10m bands.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.
    - band: string, name of the band.
    - reference_shape: tuple of ints, shape of the 10m bands of this product.

    Returns:
    - 2d array of uint16, the band at 10m resolution.
    """
    path = band_path(path_s2, product, band)
    if path is None : raise FileNotFoundError(f'band {band} not found for {product}')
    with rs.open(path) as src :
        data = src.read(1)
    if data.ndim == 3 : data = data[0]
    if BAND_RES[band] != '10m' :
        data = upsampling_with_nans(data.astype(np.float32), reference_shape, NODATAVAL_S2, 1)
    return data.astype(np.uint16)


def product_grid(path_s2, product) :
    """
    This function returns the georeferencing of a product, taken from its B02 band, which is the
    reference all other bands are upsampled to.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.

    Returns:
    - (transform, crs, shape) of the 10m grid.
    """
    path = band_path(path_s2, product, 'B02')
    if path is None : raise FileNotFoundError(f'B02 not found for {product}')
    with rs.open(path) as src :
        return src.transform, src.crs, (src.height, src.width)


def read_audit(fname) :
    """
    This function loads the per-product SCL statistics written by audit_s2_products.py, used to rank
    candidate products by cloudiness.

    Args:
    - fname: string, path to the audit CSV.

    Returns:
    - stats: dict, {(tile, date): (nodata_frac, cloud_and_shadow_frac)}.
    """
    if not isfile(fname) :
        raise FileNotFoundError(f'no SCL audit at {fname}; run audit_s2_products.py first')
    stats = {}
    with open(fname) as fh :
        for row in csv.DictReader(fh) :
            if row['err'] : continue
            stats[(row['tile'], row['date'])] = (float(row['nodata']),
                                                 float(row['cloud']) + float(row['shadow']))
    return stats


def rank_candidates(tile, products, stats, max_nodata, max_cloud) :
    """
    This function ranks a tile's candidate products from least to most cloudy, dropping those that are
    mostly outside the swath or mostly cloud. A product with no SCL statistics is dropped rather than
    assumed clear.

    Args:
    - tile: string, name of the Sentinel-2 tile.
    - products: dict, {date: [(baseline, product_name), ...]} for this tile.
    - stats: dict, the SCL statistics.
    - max_nodata, max_cloud: floats, rejection thresholds.

    Returns:
    - list of (date, product_name), least cloudy first.
    """
    ranked = []
    for date, entries in products.items() :
        key = (tile, date)
        if key not in stats : continue
        nodata, cloud = stats[key]
        if nodata > max_nodata or cloud > max_cloud : continue
        # Prefer the baseline AGBD used, when both are on disk for the same acquisition.
        entries = sorted(entries, key = lambda e: (e[0] != '9999', e[1]))
        ranked.append((cloud, date, entries[0][1]))
    ranked.sort()
    return [(date, name) for _, date, name in ranked]


def resolve_windows(tile, transform, shape, rows, cols, dem_stored, lc_stored, path_dem, path_lc) :
    """
    This function recovers the exact pixel window that AGBD used for each footprint.

    It is needed because rowcol() floors, and the coordinates stored in AGBD (a float32 decimal plus an
    unsigned integer offset) carry only ~40 cm of precision -- not the full-precision GEDI coordinate the
    original pipeline used. Footprints within ~0.25 m of a pixel boundary therefore land one pixel away,
    which is about 3.8% of them. Left uncorrected, the extra timesteps sit one pixel off the anchor.

    The fix uses the DEM patch stored alongside every footprint: it was cut from the same window, on the
    same grid, and it does not depend on which Sentinel-2 products still exist. Matching it against the
    DEM on the tile grid pins the window exactly, for every footprint. Land cover breaks the rare tie.

    Args:
    - tile: string, name of the Sentinel-2 tile.
    - transform, shape: the 10m grid of the tile.
    - rows, cols: arrays of ints, the nominal (uncorrected) pixel indices.
    - dem_stored, lc_stored: the DEM and LC patches stored in the parent, per footprint.
    - path_dem, path_lc: strings, paths to the DEM and land cover directories.

    Returns:
    - rows, cols: arrays of ints, the corrected pixel indices.
    - stats: dict, how many footprints were exact, shifted, ambiguous or unmatched.
    """
    offset = (PATCH_SIZE[0] - 1) // 2
    dem = cp.get_tile(cp.load_DEM_data(path_dem, tile), transform, shape, 'DEM', cp.DEM_attrs)['dem']
    lc = None
    rows, cols = rows.copy(), cols.copy()
    stats = Counter()
    candidates = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    def window(data, r, c) :
        return data[r - offset : r + offset + 1, c - offset : c + offset + 1]

    for i in range(len(rows)) :
        r, c = int(rows[i]), int(cols[i])
        if r < offset + 1 or c < offset + 1 or r + offset + 2 > shape[0] or c + offset + 2 > shape[1] :
            stats['edge'] += 1
            continue
        hits = [o for o in candidates if np.array_equal(window(dem, r + o[0], c + o[1]), dem_stored[i])]
        if len(hits) > 1 :
            if lc is None :
                lc = cp.get_tile(cp.load_LC_data(path_lc, tile), transform, shape, 'LC', cp.LC_attrs)
            hits = [o for o in hits
                    if np.array_equal(window(lc['lc'], r + o[0], c + o[1]), lc_stored[i][..., 0])]
        if len(hits) != 1 :
            stats['ambiguous' if len(hits) > 1 else 'unmatched'] += 1
            continue
        dr, dc = hits[0]
        rows[i], cols[i] = r + dr, c + dc
        stats['exact' if (dr, dc) == (0, 0) else 'shifted'] += 1
    return rows, cols, stats


def pick_timesteps(anchor_date, ranked, num_timesteps) :
    """
    This function picks the acquisition dates for one footprint: the anchor first, then the least-cloudy
    other products available for the tile, then -- only if the tile has nothing else to offer -- the
    anchor again, as padding.

    Args:
    - anchor_date: string, YYYYMMDD of the product AGBD paired with this footprint.
    - ranked: list of (date, product_name), least cloudy first.
    - num_timesteps: int, number of slots to fill.

    Returns:
    - dates: list of `num_timesteps` dates (YYYYMMDD), the first of which is the anchor.
    - n_padded: int, how many slots are replications of the anchor.
    """
    dates = [anchor_date]
    for date, _ in ranked :
        if len(dates) == num_timesteps : break
        if date != anchor_date : dates.append(date)
    n_padded = num_timesteps - len(dates)
    dates += [anchor_date] * n_padded
    return dates, n_padded


def extract_tile(task) :
    """
    This function builds the multi-temporal patches for a single Sentinel-2 tile, and writes them to a
    per-tile shard. The anchor patch is copied straight from the parent AGBD file, so it is identical to
    AGBD-Lite by construction; the extra timesteps are extracted from the L2A products on disk, using the
    same geometry and resampling. Where the anchor product is still on disk at the baseline AGBD recorded,
    the anchor patch is *also* re-extracted and compared bit-for-bit, which is what validates the code
    path that produced the extra timesteps.

    Candidate products are accepted or rejected per *footprint*, not per tile: a product that is half
    outside the swath still covers the footprints that fall inside it, and rejecting it on a tile-level
    nodata fraction would pad those footprints for no reason. Validity is therefore decided by reading
    B02 and measuring the actual 25x25 window.

    Args:
    - task: tuple, (tile, idxs, parent_fname, shard_fname, args, stats, region).

    Returns:
    - report: dict, per-tile counts for the build report.
    """
    tile, idxs, parent_fname, shard_fname, args, stats, region = task
    T = args.num_timesteps
    idxs = np.sort(np.asarray(idxs))
    n = len(idxs)
    report = dict(tile = str(tile), n = n, padded = 0, checked = 0, exact = 0, substituted = 0,
                  missing_extra = 0, products = 0, outside = 0, absent = 0, empty_slots = 0,
                  win_exact = 0, win_shifted = 0, win_unresolved = 0, error = '')

    try :
        products = list_products(args.path_s2).get(tile, {})
        ranked = rank_candidates(tile, products, stats, args.max_nodata, args.max_cloud)

        with h5py.File(parent_fname, 'r') as f :
            grp = f[tile]
            anchor_days = grp['Sentinel_metadata']['S2_date'][:][idxs]
            anchor_pbn = grp['Sentinel_metadata']['S2_pbn'][:][idxs]
            boa_offset = grp['Sentinel_metadata']['S2_boa_offset'][:][idxs]
            lat = decode_latlon(grp['GEDI']['lat_decimal'][:][idxs], grp['GEDI']['lat_offset'][:][idxs])
            lon = decode_latlon(grp['GEDI']['lon_decimal'][:][idxs], grp['GEDI']['lon_offset'][:][idxs])
            anchor_patch = grp['S2_bands'][:][idxs]                      # (n, 25, 25, 12), the AGBD patch
            passthrough = {'ALOS_bands': grp['ALOS_bands'][:][idxs],
                           'DEM': grp['DEM'][:][idxs],
                           'LC': grp['LC'][:][idxs]}
            dem_stored, lc_stored = passthrough['DEM'], passthrough['LC']
            gedi = {k: grp['GEDI'][k][:][idxs] for k in
                    ['agbd', 'lat_decimal', 'lat_offset', 'lon_decimal', 'lon_offset',
                     'pft_class', 'region_cla', 'rh98']}

        anchor_dates = np.array([days_to_date(d).strftime('%Y%m%d') for d in anchor_days])

        # Resolve a product for every date we may need, preferring the baseline AGBD used.
        date_to_product = {date: name for date, name in ranked}
        for date in set(anchor_dates.tolist()) :
            if date in products :
                entries = sorted(products[date], key = lambda e: (e[0] != '9999', e[1]))
                date_to_product.setdefault(date, entries[0][1])
        if not date_to_product :
            report['error'] = 'no readable product for this tile'
            return report

        # Every product of an MGRS tile shares one grid, so the footprint windows are computed once.
        # That is asserted rather than assumed: a mismatch would silently shift the extra timesteps.
        reference = None
        for date, product in sorted(date_to_product.items()) :
            grid = product_grid(args.path_s2, product)
            if reference is None : reference = grid
            assert grid == reference, f'{tile}: grid mismatch at {date} -- windows would be misaligned'
        transform, crs, shape = reference

        transformer = Transformer.from_crs('EPSG:4326', crs, always_xy = True)
        xs, ys = transformer.transform(lon.astype(np.float64), lat.astype(np.float64))
        rows, cols = rowcol(transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        offset = (PATCH_SIZE[0] - 1) // 2

        # Pin each footprint's window to the one AGBD actually used, before any extraction: every
        # timestep is then cut from the same pixels as the stored anchor patch.
        rows, cols, wstats = resolve_windows(tile, transform, shape, rows, cols,
                                             dem_stored, lc_stored, args.path_dem, args.path_lc)
        report['win_exact'] = wstats['exact']
        report['win_shifted'] = wstats['shifted']
        report['win_unresolved'] = wstats['ambiguous'] + wstats['unmatched'] + wstats['edge']

        inside = ((rows - offset >= 0) & (cols - offset >= 0) &
                  (rows + offset + 1 <= shape[0]) & (cols + offset + 1 <= shape[1]))
        report['outside'] = int((~inside).sum())

        def window(data, i) :
            return data[rows[i] - offset : rows[i] + offset + 1, cols[i] - offset : cols[i] + offset + 1]

        # Per-footprint validity of each candidate, measured on B02 rather than on tile statistics.
        # Note that a date which is one footprint's anchor is a perfectly good extra timestep for a
        # different footprint, so no date is excluded from the pool here; each footprint excludes only
        # its own anchor, below. Excluding them globally starves tiles whose every product is someone's
        # anchor, which silently costs those footprints a real timestep.
        candidate_valid = {}
        for date, product in ranked[:args.max_candidates] :
            data = load_band(args.path_s2, product, 'B02', shape)
            valid = np.zeros(n, dtype = bool)
            for i in range(n) :
                if inside[i] :
                    valid[i] = (window(data, i) != NODATAVAL_S2).mean() >= args.min_valid_frac
            candidate_valid[date] = valid
            del data

        # Slot dates, per footprint: anchor first, then the least-cloudy candidates that actually cover
        # this footprint, then the anchor again as padding if the tile has nothing else to offer.
        slot_dates = np.empty((n, T), dtype = '<U8')
        candidate_order = [d for d, _ in ranked[:args.max_candidates] if d in candidate_valid]
        for i in range(n) :
            chosen = [anchor_dates[i]]
            for date in candidate_order :
                if len(chosen) == T : break
                if date == anchor_dates[i] : continue
                if candidate_valid[date][i] : chosen.append(date)
            n_padded = T - len(chosen)
            slot_dates[i] = chosen + [anchor_dates[i]] * n_padded
            report['padded'] += (n_padded > 0)
            report['missing_extra'] += n_padded

        s2 = np.zeros((n, T, PATCH_SIZE[0], PATCH_SIZE[1], len(BAND_ORDER)), dtype = np.uint16)
        dates_out = np.zeros((n, T), dtype = np.int16)
        boa_out = np.zeros((n, T), dtype = np.uint8)

        # Slot 0 is the anchor, copied from the parent: identical to AGBD-Lite by construction.
        s2[:, 0] = anchor_patch
        for i in range(n) :
            for t in range(T) :
                dates_out[i, t] = date_to_days(dt.datetime.strptime(slot_dates[i, t], '%Y%m%d').date())
            boa_out[i, :] = boa_offset[i]
        for t in range(1, T) :
            same = (slot_dates[:, t] == anchor_dates)
            s2[same, t] = anchor_patch[same]

        # Extra timesteps, read band by band so that a worker never holds more than one full band.
        needed = sorted(set(slot_dates[:, 1:].ravel().tolist()))
        for date in needed :
            # Only footprints whose own anchor differs need extracting here; the rest were filled from
            # the parent patch above.
            users = [(i, t) for i in range(n) for t in range(1, T)
                     if slot_dates[i, t] == date and inside[i] and anchor_dates[i] != date]
            if not users : continue
            report['products'] += 1
            for b, band in enumerate(BAND_ORDER) :
                data = load_band(args.path_s2, date_to_product[date], band, shape)
                for i, t in users :
                    s2[i, t, :, :, b] = window(data, i)
                del data

        # Verification: re-extract the anchor and require it to match the parent, bit for bit, wherever
        # the anchor product is on disk at the baseline AGBD used. Elsewhere the product on disk is a
        # different processing of the same acquisition, so a mismatch is expected and is only counted.
        for date in sorted(set(anchor_dates.tolist())) :
            if date not in date_to_product :
                # The product AGBD used is not on this machine at all. Count it: an anchor that could
                # not even be attempted must not be indistinguishable from one that passed.
                report['absent'] += int(((anchor_dates == date) & inside).sum())
                continue
            baseline = PRODUCT_RE.match(date_to_product[date] + '.zip').group(3)
            members = np.flatnonzero((anchor_dates == date) & inside)
            if len(members) == 0 : continue
            reproducible = np.array([str(int(anchor_pbn[i])).zfill(4) == baseline for i in members])
            if not reproducible.any() :
                report['substituted'] += len(members)
                continue
            members = members[reproducible][:args.check_per_date]
            check = np.zeros((len(members), PATCH_SIZE[0], PATCH_SIZE[1], len(BAND_ORDER)), dtype = np.uint16)
            for b, band in enumerate(BAND_ORDER) :
                data = load_band(args.path_s2, date_to_product[date], band, shape)
                for k, i in enumerate(members) :
                    check[k, :, :, b] = window(data, i)
                del data
            for k, i in enumerate(members) :
                report['checked'] += 1
                report['exact'] += int(np.array_equal(check[k], anchor_patch[i]))

        # A slot that is entirely nodata is a silent extraction failure: it looks like a patch, trains
        # like a patch, and means nothing. Assert it here rather than discovering it downstream.
        empty = (s2.reshape(n, T, -1) == NODATAVAL_S2).all(axis = 2)
        report['empty_slots'] = int(empty.sum())
        assert not empty.any(), (f'{tile}: {int(empty.sum())} all-nodata timesteps -- '
                                 f'dates {sorted(set(slot_dates[empty.any(axis=1)][:, 1:].ravel().tolist()))}')

        # Duplicate dates within a sample are legitimate only as padding, and padding is counted.
        dup = np.array([len(set(row.tolist())) < T for row in slot_dates])
        assert int(dup.sum()) == report['padded'], \
            f'{tile}: {int(dup.sum())} samples have duplicate dates but padding was counted as {report["padded"]}'

        # Order the slots by acquisition date.
        order = np.argsort(dates_out, axis = 1, kind = 'stable')
        s2 = np.take_along_axis(s2, order[:, :, None, None, None], axis = 1)
        dates_out = np.take_along_axis(dates_out, order, axis = 1)
        boa_out = np.take_along_axis(boa_out, order, axis = 1)

        # Write the shard atomically, so that an interrupted run cannot leave a half-written shard that
        # a resumed run would mistake for a finished one.
        tmp = shard_fname + '.tmp'
        with h5py.File(tmp, 'w') as out :
            out.create_dataset('S2_bands', data = s2, compression = 'gzip',
                               chunks = (min(32, n),) + s2.shape[1:])
            out.create_dataset('DEM', data = passthrough['DEM'], compression = 'gzip')
            out.create_dataset('ALOS_bands', data = passthrough['ALOS_bands'], compression = 'gzip')
            out.create_dataset('LC', data = passthrough['LC'], compression = 'gzip')
            out.create_dataset('region', data = np.full(n, region, dtype = np.uint8), compression = 'gzip')
            meta = out.create_group('Sentinel_metadata')
            meta.create_dataset('S2_date', data = dates_out, compression = 'gzip')
            meta.create_dataset('S2_boa_offset', data = boa_out, compression = 'gzip')
            g = out.create_group('GEDI')
            for k, v in gedi.items() : g.create_dataset(k, data = v, compression = 'gzip')
            out.attrs['tile'] = str(tile)
            out.attrs['num_timesteps'] = T
        os.replace(tmp, shard_fname)

    except Exception as e :
        report['error'] = f'{type(e).__name__}: {e}'
        report['traceback'] = traceback.format_exc()
    return report


def get_tile_to_file(path_patches) :
    """
    This function maps each tile to the parent AGBD shard that holds it, for the 2020 products, which
    are the ones AGBD-Lite is built from.

    Args:
    - path_patches: string, directory holding the parent AGBD shards.

    Returns:
    - dict, {tile: shard filename}.
    """
    mapping = {}
    for i in range(20) :
        fname = join(path_patches, f'data_subset-2020-v4_{i}-20.h5')
        if not isfile(fname) : continue
        with h5py.File(fname, 'r') as f :
            for tile in f : mapping.setdefault(tile, fname)
    return mapping


def merge_shards(shard_files, output_fname, num_timesteps, chunk_size = 32, seed = 42) :
    """
    This function concatenates the per-tile shards into a single .h5 file, mirroring the AGBD-Lite
    layout with a temporal axis on the Sentinel-2 datasets. Tiles are written in a shuffled order, and
    rows are shuffled within each tile, so that sequential reads are not ordered by geography. This is a
    block shuffle, not the global shuffle gen_subsample.py performs: use shuffle = True in the loader.

    Args:
    - shard_files: list of strings, the per-tile shards to merge.
    - output_fname: string, path to the output file.
    - num_timesteps: int, number of timesteps.
    - chunk_size: int, chunk size along the sample axis.
    - seed: int, seed for the shuffle.

    Returns:
    - total: int, number of samples written.
    """
    rng = np.random.default_rng(seed)
    shard_files = list(shard_files)
    rng.shuffle(shard_files)

    sizes = []
    for fname in shard_files :
        with h5py.File(fname, 'r') as f : sizes.append(f['GEDI']['agbd'].shape[0])
    total = int(sum(sizes))

    tmp = output_fname + '.tmp'
    with h5py.File(tmp, 'w') as out :
        specs = {
            'S2_bands':   ((num_timesteps, 25, 25, 12), np.uint16),
            'ALOS_bands': ((25, 25, 2), np.uint16),
            'DEM':        ((25, 25), np.int16),
            'LC':         ((25, 25, 2), np.uint8),
            'region':     ((), np.uint8),
        }
        for name, (shape, dtype) in specs.items() :
            out.create_dataset(name, shape = (total,) + shape, dtype = dtype,
                               chunks = (chunk_size,) + shape, compression = 'gzip')
        meta = out.create_group('Sentinel_metadata')
        meta.create_dataset('S2_date', shape = (total, num_timesteps), dtype = np.int16,
                            chunks = (chunk_size, num_timesteps), compression = 'gzip')
        meta.create_dataset('S2_boa_offset', shape = (total, num_timesteps), dtype = np.uint8,
                            chunks = (chunk_size, num_timesteps), compression = 'gzip')
        gedi = out.create_group('GEDI')
        gedi_specs = {'agbd': np.float32, 'lat_decimal': np.float32, 'lat_offset': np.uint8,
                      'lon_decimal': np.float32, 'lon_offset': np.uint8, 'pft_class': np.uint8,
                      'region_cla': np.uint8, 'rh98': np.float32}
        for name, dtype in gedi_specs.items() :
            gedi.create_dataset(name, shape = (total,), dtype = dtype,
                                chunks = (chunk_size,), compression = 'gzip')
        out['S2_bands'].attrs['order'] = BAND_ORDER
        out['ALOS_bands'].attrs['order'] = ['HH', 'HV']
        out['S2_bands'].attrs['axes'] = ['sample', 'timestep', 'row', 'col', 'band']
        out.attrs['num_timesteps'] = num_timesteps
        out.attrs['boa_offset_convention'] = (
            'Reflectances are stored exactly as AGBD stores them: the BOA_ADD_OFFSET of -1000 carried '
            'by reprocessed products is NOT subtracted, and S2_boa_offset is 0 throughout. This mirrors '
            'AGBD and AGBD-Lite so that models trained on them remain comparable.')
        out.attrs['padding_convention'] = (
            'Where a tile offered fewer than num_timesteps products, the anchor patch is replicated. The '
            'dates are not: a padded sample repeats the anchor date, so padded samples are exactly those '
            'with len(unique(S2_date[i])) < num_timesteps.')

        start = 0
        for fname, size in zip(shard_files, sizes) :
            order = rng.permutation(size)
            with h5py.File(fname, 'r') as f :
                sl = slice(start, start + size)
                for name in specs :
                    out[name][sl] = f[name][:][order]
                for name in ['S2_date', 'S2_boa_offset'] :
                    out['Sentinel_metadata'][name][sl] = f['Sentinel_metadata'][name][:][order]
                for name in gedi_specs :
                    out['GEDI'][name][sl] = f['GEDI'][name][:][order]
            start += size
        assert start == total, f'wrote {start} rows, expected {total}'
    os.replace(tmp, output_fname)
    return total


def main() :

    args = _parser()
    output_dir = args.output_dir if args.output_dir else join(args.path_patches, 'AGBD-Lite-MT')
    shard_dir = join(output_dir, f'shards-{args.mode}')
    os.makedirs(shard_dir, exist_ok = True)

    # Same footprints, same split, same region mapping as AGBD-Lite.
    with open(join(args.path_patches, 'subsampled_indices.pkl'), 'rb') as f : indices = pickle.load(f)
    with open(join(args.path_patches, 'biomes_splits_to_name.pkl'), 'rb') as f :
        tiles_in_mode = pickle.load(f)[args.mode]
    with open(join(HERE, 'tile_to_region.pkl'), 'rb') as f : tile_to_region = pickle.load(f)
    tile_to_file = get_tile_to_file(args.path_patches)
    stats = read_audit(args.audit)

    tiles = [t for t in indices if t in tiles_in_mode and t in tile_to_file]
    if args.tiles : tiles = [t for t in tiles if t in args.tiles]
    expected = sum(len(indices[t]) for t in tiles)
    print(f'{args.mode}: {len(tiles)} tiles, {expected} footprints, {args.num_timesteps} timesteps')

    tasks = []
    for tile in sorted(tiles) :
        shard = join(shard_dir, f'{tile}.h5')
        if isfile(shard) and not args.overwrite : continue
        tasks.append((tile, indices[tile], tile_to_file[tile], shard, args, stats,
                      tile_to_region.get(tile, 0)))
    print(f'{len(tasks)} tiles to extract ({len(tiles) - len(tasks)} shards already present)')

    reports = []
    if tasks :
        with Pool(args.workers) as pool :
            for i, report in enumerate(pool.imap_unordered(extract_tile, tasks), 1) :
                reports.append(report)
                flag = 'ERROR ' + report['error'] if report['error'] else \
                       f"n={report['n']:6d} padded={report['padded']:6d} exact={report['exact']}/{report['checked']}"
                print(f"[{i}/{len(tasks)}] {report['tile']}  {flag}", flush = True)

    ############################################################################################################################
    # Build report and gates. These are asserted over every sample, not a sample of them.

    failed = [r for r in reports if r['error']]
    if failed :
        print(f'\n{len(failed)} tiles failed:')
        for r in failed[:20] :
            print(f"  {r['tile']}: {r['error']}")
            if r.get('traceback') : print(r['traceback'])
        raise SystemExit(f'{len(failed)} tiles failed; refusing to merge a partial dataset')

    if reports :
        n_total = sum(r['n'] for r in reports)
        padded = sum(r['padded'] for r in reports)
        checked = sum(r['checked'] for r in reports)
        exact = sum(r['exact'] for r in reports)
        substituted = sum(r['substituted'] for r in reports)
        print(f'\nsamples                  {n_total}')
        print(f'padded (fewer than T)    {padded} ({100 * padded / max(n_total, 1):.2f}%)')
        outside = sum(r['outside'] for r in reports)
        absent = sum(r['absent'] for r in reports)
        w_exact = sum(r['win_exact'] for r in reports)
        w_shift = sum(r['win_shifted'] for r in reports)
        w_unres = sum(r['win_unresolved'] for r in reports)
        print(f'anchors re-checked       {checked}, of which exact {exact}')
        print(f'anchors not reproducible {substituted} (product on disk is a different baseline)')
        print(f'anchors not on disk      {absent} (the product AGBD used is absent from this machine)')
        print(f'footprints outside grid  {outside}')
        print(f'windows pinned via DEM   {w_exact + w_shift} '
              f'(nominal {w_exact}, shifted by 1 px {w_shift}), unresolved {w_unres}')
        # A verification that covers nothing must say so: silence here would read as success.
        coverage = (checked + substituted + absent)
        if coverage and checked / coverage < args.min_checked_frac :
            print(f'WARNING: the bit-for-bit anchor check covered only {checked}/{coverage} '
                  f'({100 * checked / coverage:.2f}%) of anchors -- the rest have no on-disk product at '
                  f'the baseline AGBD used, so this build is largely UNVERIFIED against AGBD.')
        elif checked == 0 :
            print('WARNING: the bit-for-bit anchor check covered NO samples; this build is UNVERIFIED.')

        by_region = defaultdict(lambda: [0, 0])
        for r in reports :
            by_region[r['tile'][:2]][0] += r['n']
            by_region[r['tile'][:2]][1] += r['padded']
        worst = sorted(by_region.items(), key = lambda kv: -kv[1][1] / max(kv[1][0], 1))[:5]
        if padded :
            print('padding concentration (top MGRS zones):')
            for zone, (tot, pad) in worst :
                print(f'  {zone}: {pad}/{tot} = {100 * pad / max(tot, 1):.1f}%')

        if checked and exact / checked < args.min_exact_frac :
            raise SystemExit(f'anchor check failed: {exact}/{checked} exact, '
                             f'below --min_exact_frac {args.min_exact_frac}')
        if n_total and w_unres / n_total > args.max_unresolved_frac :
            raise SystemExit(f'{w_unres}/{n_total} windows could not be pinned against the stored DEM, '
                             f'above --max_unresolved_frac {args.max_unresolved_frac}')
        if n_total and padded / n_total > args.max_padded_frac :
            raise SystemExit(f'padding {100 * padded / n_total:.2f}% exceeds '
                             f'--max_padded_frac {100 * args.max_padded_frac:.2f}%')

    shards = sorted(glob.glob(join(shard_dir, '*.h5')))
    print(f'\nmerging {len(shards)} shards')
    output_fname = join(output_dir, f'AGBD-Lite-MT{args.num_timesteps}-{args.mode}.h5')
    total = merge_shards(shards, output_fname, args.num_timesteps)
    assert total == expected, f'merged {total} samples, expected {expected}'
    print(f'wrote {output_fname}: {total} samples, '
          f'{os.path.getsize(output_fname) / 1e9:.1f} GB')


if __name__ == '__main__' :
    main()
