"""

This script downloads cropped subsets of AEF GeoTIFF files from S3, for a set of AOIs
provided as a GeoDataFrame. Instead of downloading whole files, it opens them remotely
via GDAL's /vsicurl/ and only saves the AOI extent. Works best with COG files.

env: awsenv

Usage: python download_aoi.py [--aois <path>] [--output_dir <dir>] [--download <bool>]
         [--verify <bool>] [--force <bool>] [--num_workers <n>] [--buffer_m <m>]

e.g.  python download_aoi.py --download true                 # first download of every AOI
      python download_aoi.py --verify true                   # fetch whatever is missing
      python download_aoi.py --verify true --force true      # re-fetch everything, even if present

"""

###################################################################################################
# Imports

import numpy as np
import time
import random
import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import rasterio
from rasterio.mask import mask as rio_mask
from pyproj import CRS, Transformer
from os.path import join, basename, exists
from os import makedirs
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import geopandas as gpd
from shapely.geometry import Point, box as shapely_box, mapping

###################################################################################################
# Helper functions and global variables

AOIS = '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/AGBRef/data/AGBref.gpkg'
OUTPUT_DIR = '/scratch3/gsialelli/AEF'


def _latlon_to_zone_path(lat, lon) :
    """
    This function returns the UTM zone directory name used in the S3 bucket (e.g. '10N').

    Args:
    - lat (float): Latitude in decimal degrees.
    - lon (float): Longitude in decimal degrees.

    Returns:
    - zone_path (str): UTM zone path string, e.g. '10N' or '32S'.
    """
    zone_num = int((lon + 180) / 6) + 1

    # Special zones for Norway (32V) and Svalbard (31X, 33X, 35X, 37X)
    if 56 <= lat < 64 and 3 <= lon < 12 : zone_num = 32
    elif 72 <= lat < 84 :
        if   0 <= lon <  9 : zone_num = 31
        elif 9 <= lon < 21 : zone_num = 33
        elif 21 <= lon < 33 : zone_num = 35
        elif 33 <= lon < 42 : zone_num = 37

    return f"{zone_num}{'N' if lat >= 0 else 'S'}"


def _list_zone_uris(year, zone_path, s3) :
    """
    This function lists all AEF S3 URIs for a given year and UTM zone.

    Args:
    - year (int): AEF year.
    - zone_path (str): UTM zone path string, e.g. '10N'.
    - s3: boto3 S3 client.

    Returns:
    - uris (list): List of S3 URIs for all .tiff files in the zone/year.
    """
    prefix = f"aef/v1/annual/{year}/{zone_path}/"
    paginator = s3.get_paginator('list_objects_v2')
    uris = []
    for page in paginator.paginate(Bucket='tge-labs', Prefix=prefix) :
        for obj in page.get('Contents', []) :
            if obj['Key'].endswith('.tiff') :
                uris.append(f"s3://us-west-2.opendata.source.coop/tge-labs/{obj['Key']}")
    return uris


def _parse_pixel_offsets(uri) :
    """
    This function extracts the pixel offsets from an AEF filename.

    AEF filenames follow the pattern: <hash>-<row_offset>-<col_offset>.tiff,
    e.g. xelkfa3ezwmt3ytsy-0000008192-0000000000.tiff.

    Args:
    - uri (str): S3 URI of the AEF file.

    Returns:
    - hash_prefix (str): The dataset hash prefix, shared by all tiles of the same dataset.
    - row_off (int): Row pixel offset within the dataset raster.
    - col_off (int): Column pixel offset within the dataset raster.
    """
    parts = basename(uri).replace('.tiff', '').split('-')
    return parts[0], int(parts[-2]), int(parts[-1])


def _build_zone_index(uris, num_workers=8, num_retries=4, strict=True) :
    """
    This function builds a spatial index mapping each URI to its bounding box in UTM.

    Files in a zone are grouped by their hash prefix (each prefix is an independent dataset).
    One tile per prefix is opened to get its spatial transform; bounds for all other tiles of
    that prefix are then derived from their pixel offsets, avoiding redundant file opens.
    Note: y_res may be positive (south-up) or negative (north-up), both are handled correctly.

    S3 throws transient errors when several tiles are opened at once, and a dropped prefix drops
    every file it holds from the index, i.e. it silently truncates the AOIs that those files cover.
    Each open is therefore retried with exponential backoff, and a prefix that still fails is
    raised rather than skipped: a truncated index is worse than a failed run, because nothing
    downstream can tell that it is truncated.

    Args:
    - uris (list): List of S3 URIs for all .tiff files in a zone/year.
    - num_workers (int): Number of parallel threads for metadata reads.
    - num_retries (int): Number of attempts per prefix before giving up.
    - strict (bool): If True, raise when a prefix cannot be read; if False, warn and skip it,
                     which yields a silently incomplete index. Default True.

    Returns:
    - index (list): List of (uri, (minx, miny, maxx, maxy)) tuples in the zone's UTM CRS.
    """
    # GDAL_HTTP_MAX_RETRY/RETRY_DELAY make GDAL retry the HTTP request itself, which is what
    # download_aoi() already does and what this function was missing: without them a throttled
    # request fails outright, and /vsicurl/ reports it as "does not exist in the file system"
    # rather than as the transient error it is. CPL_VSIL_CURL_NON_CACHED stops GDAL from caching
    # that bogus negative result, which would otherwise make every retry replay the failure.
    gdal_env = {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
                'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tiff',
                'CPL_VSIL_CURL_NON_CACHED': '/vsicurl/https://data.source.coop',
                'GDAL_HTTP_MAX_RETRY': '5',
                'GDAL_HTTP_RETRY_DELAY': '2'}

    # Group URIs by hash prefix
    by_prefix = defaultdict(list)
    for uri in uris :
        hash_prefix, row_off, col_off = _parse_pixel_offsets(uri)
        by_prefix[hash_prefix].append((uri, row_off, col_off))

    def _read_prefix_transform(path_part) :
        """
        Open one tile and return (tile_h, tile_w, x_res, y_res, x_origin_term, y_origin_term),
        retrying on the transient S3 errors that concurrent /vsicurl/ opens provoke.
        """
        last_error = None
        for attempt in range(num_retries) :
            try :
                with rasterio.Env(**gdal_env) :
                    with rasterio.open(f"/vsicurl/https://data.source.coop/{path_part}") as src :
                        return src.transform, src.height, src.width
            except Exception as e :
                last_error = e
                # Back off with jitter, so that retries of concurrent failures do not re-collide
                if attempt < num_retries - 1 :
                    time.sleep((2 ** attempt) * (1 + random.random()))
        raise RuntimeError(f"Could not read {path_part} after {num_retries} attempts: {last_error}")

    def _get_prefix_bounds(item) :
        hash_prefix, tiles = item
        # Open the first tile of this prefix to get its spatial transform and tile size
        uri0, row_off0, col_off0 = tiles[0]
        path_part = '/'.join(uri0.split('/')[3:])
        try :
            t, tile_h, tile_w = _read_prefix_transform(path_part)
        except RuntimeError as e :
            if strict : raise
            print(f"  WARNING: dropping prefix {hash_prefix} ({len(tiles)} file(s)): {e}")
            return []

        x_res, y_res = t.a, t.e
        # Back-calculate the dataset origin from this tile's transform and its offsets
        x_origin = t.c - col_off0 * x_res
        y_origin = t.f - row_off0 * y_res

        # Compute bounds for all tiles of this prefix using pixel offsets
        results = []
        for uri, row_off, col_off in tiles :
            x_start = x_origin + col_off * x_res
            y_start = y_origin + row_off * y_res
            x_end   = x_start + tile_w * x_res
            y_end   = y_start + tile_h * y_res
            bounds  = (min(x_start, x_end), min(y_start, y_end),
                       max(x_start, x_end), max(y_start, y_end))
            results.append((uri, bounds))
        return results

    index = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor :
        # executor.map re-raises inside this loop, so a strict failure aborts the whole index
        for result_list in executor.map(_get_prefix_bounds, by_prefix.items()) :
            index.extend(result_list)

    return index


def find_uris_for_aoi(geometry, year, s3, zone_index_cache=None, buffer_m=5100, strict=True) :
    """
    This function finds the S3 URIs of AEF files that overlap with the given AOI.

    For each UTM zone covering the AOI, it queries the zone index (building it from S3 if
    not already cached) and returns all files whose spatial footprint intersects the AOI.

    Args:
    - geometry: shapely geometry in EPSG:4326, or a (lon, lat) tuple. If a Point or tuple
                is given, a bounding box of half-width buffer_m is built around it.
    - year (int): AEF year to retrieve.
    - s3: boto3 S3 client (with endpoint_url='https://data.source.coop').
    - zone_index_cache (dict): Optional dict mapping (year, zone_path) -> zone index, used to
                               avoid rebuilding the index for repeated calls on the same zone.
                               Modified in-place when new zones are fetched.
    - buffer_m (float): Half-width in metres used when geometry is a point. Default 5100m
                        gives a ~10.2 km x 10.2 km box.
    - strict (bool): Passed to _build_zone_index. If True (default), raise when a zone index
                     cannot be built in full, rather than returning a partial file list.

    Returns:
    - uris (list): Unique S3 URIs for AEF files covering the AOI (may span multiple zones).
    """
    if zone_index_cache is None : zone_index_cache = {}

    if isinstance(geometry, tuple) :
        geometry = Point(geometry[0], geometry[1])

    if geometry.geom_type == 'Point' :
        lon, lat = geometry.x, geometry.y
        delta_lat = buffer_m / 111_000
        delta_lon = buffer_m / (111_000 * np.cos(np.radians(lat)))
        minx, maxx = lon - delta_lon, lon + delta_lon
        miny, maxy = lat - delta_lat, lat + delta_lat
    else :
        minx, miny, maxx, maxy = geometry.bounds

    # Sample 5 points to detect UTM zone crossings
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    sample_points = [(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy), (cx, cy)]

    zone_paths = set()
    for lon, lat in sample_points :
        zone_paths.add(_latlon_to_zone_path(lat, lon))

    all_uris = []
    for zone_path in zone_paths :
        key = (year, zone_path)

        # Build and cache the zone index if not done yet
        if key not in zone_index_cache :
            uris = _list_zone_uris(year, zone_path, s3)
            zone_index_cache[key] = _build_zone_index(uris, strict=strict) if uris else []

        zone_index = zone_index_cache[key]
        if not zone_index : continue

        # Reproject AOI bounds to this zone's UTM CRS
        zone_num = int(zone_path[:-1])
        is_south = zone_path[-1] == 'S'
        crs_utm = CRS.from_dict({'proj': 'utm', 'zone': zone_num, 'south': is_south})
        tfm = Transformer.from_crs('EPSG:4326', crs_utm, always_xy=True)
        xs, ys = tfm.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = min(xs), min(ys), max(xs), max(ys)

        # Filter by bounding box overlap
        for uri, (fminx, fminy, fmaxx, fmaxy) in zone_index :
            if fmaxx > aoi_minx and fminx < aoi_maxx and fmaxy > aoi_miny and fminy < aoi_maxy :
                all_uris.append(uri)

    return list(set(all_uris))


def download_aoi(uri, output_dir, geometry, aoi_id=None, force=False) :
    """
    This function downloads and crops an AEF GeoTIFF from S3 to the given AOI extent.

    The file is opened remotely via GDAL's /vsicurl/. If the file is a COG, only the
    relevant byte ranges are fetched. The output is a GeoTIFF cropped to the AOI bounding
    box, saved in the file's original CRS.

    Args:
    - uri (str): S3 URI of the AEF file.
    - output_dir (str): Local directory where the cropped file will be saved.
    - geometry: shapely geometry in EPSG:4326, or a (lon, lat) tuple. The bounding box
                of this geometry is used as the crop window.
    - aoi_id (str): Optional identifier prepended to the output filename.
    - force (bool): If True, re-download even if the file is already on disk. The skip check
                    only tests for existence, so a file left truncated by an interrupted write
                    would otherwise be kept forever; pass force=True to overwrite it.

    Returns:
    - status (str): A message indicating the download status.
    """
    gdal_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tiff',
        'GDAL_HTTP_MAX_RETRY': '3',
        'GDAL_HTTP_RETRY_DELAY': '1',
    }

    if isinstance(geometry, tuple) :
        geometry = Point(geometry[0], geometry[1])

    # Build output filename and check if already downloaded
    fname = basename(uri)
    if aoi_id is not None : fname = f"{aoi_id}_{fname}"
    local_path = join(output_dir, fname)
    if exists(local_path) and not force : return f"Skipped (exists): {fname}"

    # Convert S3 URI to HTTPS URL for /vsicurl/
    # s3://us-west-2.opendata.source.coop/tge-labs/... -> https://data.source.coop/tge-labs/...
    path_part = '/'.join(uri.split('/')[3:])
    vsicurl_path = f"/vsicurl/https://data.source.coop/{path_part}"

    try :
        with rasterio.Env(**gdal_env) :
            with rasterio.open(vsicurl_path) as src :
                file_crs = src.crs

                # Reproject the AOI bounding box to the file's CRS
                if file_crs.to_epsg() != 4326 :
                    tfm = Transformer.from_crs('EPSG:4326', file_crs, always_xy=True)
                    b = geometry.bounds
                    xs, ys = tfm.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
                    geom_proj = shapely_box(min(xs), min(ys), max(xs), max(ys))
                else :
                    geom_proj = geometry

                out_image, out_transform = rio_mask(src, [mapping(geom_proj)], crop=True)

                if out_image.size == 0 : return f"Skipped (no overlap): {fname}"

                out_meta = src.meta.copy()
                out_meta.update({
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': out_transform,
                })

        makedirs(output_dir, exist_ok=True)
        with rasterio.open(local_path, 'w', **out_meta) as dst :
            dst.write(out_image)

        return f"Downloaded: {fname}"

    except ValueError as e :
        # rio_mask raises ValueError when the AOI misses the file entirely. That is an expected
        # outcome (a tile is covered by several AEF files, and a small --window_km hits only one or
        # two of them), not a failure -- report it as a skip so it is not counted as an error.
        if 'do not overlap' in str(e).lower() : return f"Skipped (no overlap): {fname}"
        return f"Error {uri}: {e}"

    except Exception as e :
        return f"Error {uri}: {e}"


###################################################################################################
# Code execution

def download_aois_batch(aois_gdf, year_column, s3, output_dir, id_column=None,
                        num_workers=8, buffer_m=5100, force=False) :
    """
    This function downloads AOI-cropped AEF files for every row in a GeoDataFrame.

    For each AOI, find_uris_for_aoi() identifies the relevant AEF files by querying S3
    directly, and download_aoi() fetches and crops them. The zone spatial index is built
    once per unique (year, zone) combination and reused across all AOIs. Downloads are
    parallelised across files. Each AOI gets its own sub-folder under output_dir.
    AOIs in 2017 are mapped to 2018 (earliest available AEF year).

    Args:
    - aois_gdf (GeoDataFrame): AOI geometries (Point or polygon) in any CRS; reprojected
                               to EPSG:4326 automatically if needed.
    - year_column (str): Name of the column in aois_gdf containing the AEF year per AOI.
                        Can be fractional (e.g. 2018.038); the integer part is used for lookup.
                        Values in 2017 are mapped to 2018, the earliest available AEF year.
    - s3: boto3 S3 client (with endpoint_url='https://data.source.coop').
    - output_dir (str): Root output directory; one sub-folder per AOI is created.
    - id_column (str): Column name to use as AOI identifier in filenames (default: row index).
    - num_workers (int): Number of parallel download threads.
    - buffer_m (float): Buffer radius in metres around Point geometries (default 5100m).
    - force (bool): If True, re-download files that are already on disk (default: False).

    Returns:
    - results (list): List of (aoi_id, status) tuples.
    """
    if aois_gdf.crs and aois_gdf.crs.to_epsg() != 4326 :
        aois_gdf = aois_gdf.to_crs('EPSG:4326')

    # Shared zone index cache: (year, zone_path) -> list of (uri, bounds)
    zone_index_cache = {}

    # Build the list of (uri, output_dir, geom, aoi_id) tasks
    print(f"Building task list for {len(aois_gdf)} AOIs...")
    tasks = []
    for idx, row in aois_gdf.iterrows() :
        year = max(int(row[year_column]), 2018)  # 2018 is the earliest available AEF year
        aoi_id = str(row[id_column]) if id_column else str(idx)
        geom = row.geometry
        uris = find_uris_for_aoi(geom, year, s3, zone_index_cache=zone_index_cache, buffer_m=buffer_m)
        print(f"  AOI {aoi_id} (year {year}): {len(uris)} file(s) found")
        for uri in uris :
            tasks.append((uri, join(output_dir, aoi_id), geom, aoi_id))

    if not tasks :
        print("No matching URIs found for any AOI.")
        return []

    print(f"Downloading {len(tasks)} file(s) across {len(set(t[3] for t in tasks))} AOIs...")

    def _run(task) :
        uri, out_dir, geom, aoi_id = task
        status = download_aoi(uri, out_dir, geom, aoi_id=aoi_id, force=force)
        print(f"  {status}")
        return (aoi_id, status)

    with ThreadPoolExecutor(max_workers=num_workers) as executor :
        results = list(executor.map(_run, tasks))

    success_count = sum(1 for _, s in results if s.startswith('Downloaded'))
    error_count = sum(1 for _, s in results if s.startswith('Error'))
    print(f"Download completed: {success_count} files downloaded, {error_count} errors.")

    return results


def verify_downloads(aois_gdf, year_column, s3, output_dir, id_column=None,
                     buffer_m=5100, redownload=False, num_workers=8, force=False) :
    """
    This function verifies that all expected AEF files were downloaded for a set of AOIs.

    It rebuilds the expected task list using the same logic as download_aois_batch and checks
    whether each expected output file exists on disk. Useful to detect incomplete runs caused
    by crashes or network errors.

    Note that the expected list is rebuilt with find_uris_for_aoi, so this function is only as
    trustworthy as the resolver: it can only flag a file as missing if the resolver names it.
    That is why find_uris_for_aoi defaults to strict=True.

    Args:
    - aois_gdf (GeoDataFrame): Same GeoDataFrame used during the original download.
    - year_column (str): Same year column used during the original download.
    - s3: boto3 S3 client (with endpoint_url='https://data.source.coop').
    - output_dir (str): Same root output directory used during the original download.
    - id_column (str): Same id column used during the original download (default: row index).
    - buffer_m (float): Same buffer radius used during the original download (default 5100m).
    - redownload (bool): If True, re-download any missing files (default: False).
    - num_workers (int): Number of parallel download threads (only used if redownload=True).
    - force (bool): If True, re-download every expected file rather than only the absent ones
                    (default: False). Only meaningful together with redownload=True.

    Returns:
    - missing (list): List of local file paths that were expected but are absent.
    """
    if aois_gdf.crs and aois_gdf.crs.to_epsg() != 4326 :
        aois_gdf = aois_gdf.to_crs('EPSG:4326')

    zone_index_cache = {}

    print(f"Building expected file list for {len(aois_gdf)} AOIs...")
    tasks = []
    for idx, row in aois_gdf.iterrows() :
        year = max(int(row[year_column]), 2018)
        aoi_id = str(row[id_column]) if id_column else str(idx)
        geom = row.geometry
        uris = find_uris_for_aoi(geom, year, s3, zone_index_cache=zone_index_cache, buffer_m=buffer_m)
        for uri in uris :
            tasks.append((uri, join(output_dir, aoi_id), geom, aoi_id))

    total = len(tasks)
    missing_tasks = []
    for uri, out_dir, geom, aoi_id in tasks :
        fname = f"{aoi_id}_{basename(uri)}"
        local_path = join(out_dir, fname)
        if not exists(local_path) :
            missing_tasks.append((uri, out_dir, geom, aoi_id))

    missing_paths = [join(t[1], f"{t[3]}_{basename(t[0])}") for t in missing_tasks]
    print(f"Verification complete: {total - len(missing_tasks)}/{total} files present, "
          f"{len(missing_tasks)} missing.")

    for p in missing_paths :
        print(f"  Missing: {p}")

    # With force, every expected file is fetched again, not just the absent ones
    todo = tasks if force else missing_tasks

    if todo and redownload :
        print(f"{'Re-downloading all' if force else 'Re-downloading'} {len(todo)} file(s)...")

        def _run(task) :
            uri, out_dir, geom, aoi_id = task
            status = download_aoi(uri, out_dir, geom, aoi_id=aoi_id, force=force)
            print(f"  {status}")
            return status

        with ThreadPoolExecutor(max_workers=num_workers) as executor :
            results = list(executor.map(_run, todo))

        errors = [r for r in results if r.startswith('Error')]
        print(f"Re-download completed: {len(results) - len(errors)} ok, {len(errors)} errors.")
        for e in errors : print(f"  {e}")

    return missing_paths


def str2bool(value) :
    """
    Convert a string to a boolean.
    """
    return str(value).lower() in ('true', 't', 'yes', '1')


def parse_arguments() :
    """
    Parse the command-line arguments for downloading and verifying the AOI-cropped AEF files.
    """
    parser = argparse.ArgumentParser(description = 'Download AOI-cropped AEF files from S3.')
    parser.add_argument('--aois', type = str, default = AOIS, help = 'Path to the AOIs vector file.')
    parser.add_argument('--output_dir', type = str, default = OUTPUT_DIR, help = 'Root output directory; one sub-folder per AOI.')
    parser.add_argument('--year_column', type = str, default = 'AVG_YEAR', help = 'Name of the column holding the AEF year of each AOI.')
    parser.add_argument('--download', type = str2bool, default = False, help = 'Whether to run the initial download of all AOIs.')
    parser.add_argument('--verify', type = str2bool, default = True, help = 'Whether to verify the downloads, and fetch whatever is missing.')
    parser.add_argument('--force', type = str2bool, default = False, help = 'Whether to re-download files that are already on disk. Use this when the files on disk may be truncated, as the skip check only tests for existence.')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'Number of parallel download threads.')
    parser.add_argument('--buffer_m', type = float, default = 5100, help = 'Half-width in metres of the box built around Point AOIs.')
    args = parser.parse_args()
    return args.aois, args.output_dir, args.year_column, args.download, args.verify, args.force, args.num_workers, args.buffer_m


if __name__ == '__main__' :

    aois_path, output_dir, year_column, download, verify, force, num_workers, buffer_m = parse_arguments()

    s3 = boto3.client('s3', endpoint_url='https://data.source.coop', config=Config(signature_version=UNSIGNED))
    aois = gpd.read_file(aois_path)

    if download :
        download_aois_batch(aois, year_column=year_column, s3=s3, output_dir=output_dir,
                            num_workers=num_workers, buffer_m=buffer_m, force=force)

    if verify :
        verify_downloads(aois, year_column=year_column, s3=s3, output_dir=output_dir, id_column=None,
                         buffer_m=buffer_m, redownload=True, num_workers=num_workers, force=force)