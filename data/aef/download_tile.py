"""

This script downloads the AEF embedding files covering a given Sentinel-2 (MGRS) tile, e.g. 44RLS.
It resolves the AEF files for each tile in one of two ways:

1. Fast path: the tile is listed in tile_to_aefiles.pkl (470 tiles, the AGBD coverage), which maps
   each tile to a year and each year to the AEF files covering it. No S3 listing is needed.
2. Fallback: the tile is not listed, so its geometry is read from the Sentinel-2 index shapefile
   and the AEF files are resolved by querying S3, via find_uris_for_aoi() from download_aoi.py.

By default the files are cropped to the tile extent (an AEF file spans 8192x8192 pixels at 10m,
i.e. ~82km, so a 110km Sentinel-2 tile is covered by several of them, each only partly relevant).
Pass --crop false to download the whole AEF files instead.

The files are saved as <output_dir>/<tile>/<tile>_<aef_file>.tiff, which is the layout that
inference_aef.py expects (it globs '*/*.tiff' under paths['aef']).

env: awsenv

Usage: python download_tile.py --tiles <tile> [<tile> ...] --year <year> --output_dir <output_directory>
         [--tiles_file <path>] [--crop <bool>] [--num_workers <n>] [--path_pkl <path>] [--s2_index <path>]

e.g.  python download_tile.py --tiles 44RLS 44RLT --year 2020
      python download_tile.py --tiles_file tiles.txt --year 2020 --crop false

"""

###################################################################################################
# Imports

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import argparse
import pickle
import geopandas as gpd
from shapely.geometry import box as shapely_box
from math import cos, radians
from os.path import join, exists, basename, getsize, dirname, abspath
from os import makedirs

from download_aoi import find_uris_for_aoi, download_aoi

###################################################################################################
# Helper functions and global variables

BASE_DIR = dirname(abspath(__file__))
S2_INDEX = '/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp'


def str2bool(value):
    """
    Convert a string to a boolean.
    """
    return str(value).lower() in ('true', 't', 'yes', '1')


def parse_args():
    """
    Parse command-line arguments for downloading the AEF files covering a Sentinel-2 tile.
    """
    parser = argparse.ArgumentParser(description = "Download the AEF files covering a Sentinel-2 tile.")
    parser.add_argument('--tiles', type = str, nargs = '+', default = None, help = 'Sentinel-2 tile name(s) to download AEF files for, e.g. 44RLS.')
    parser.add_argument('--tiles_file', type = str, default = None, help = 'Path to a text file with one Sentinel-2 tile name per line.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the AEF files to download.')
    parser.add_argument('--output_dir', type = str, default = '/scratch3/gsialelli/AEF', help = 'Output directory to save downloaded files.')
    parser.add_argument('--crop', type = str2bool, default = True, help = 'Whether to crop the AEF files to the tile extent.')
    parser.add_argument('--window_km', type = float, default = None, help = 'Instead of the whole ~110km tile, crop to a window of this many km centred on the tile centre (e.g. 20). Cuts a ~2GB tile to ~20MB, which is all a map figure needs. Implies --crop true.')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'Number of parallel download workers.')
    parser.add_argument('--path_pkl', type = str, default = BASE_DIR, help = 'Path to the directory containing tile_to_aefiles.pkl.')
    parser.add_argument('--s2_index', type = str, default = S2_INDEX, help = 'Path to the Sentinel-2 index shapefile (must have a "Name" column).')
    args = parser.parse_args()

    if (args.tiles is None) == (args.tiles_file is None):
        parser.error('Pass exactly one of --tiles or --tiles_file.')

    if args.tiles_file is not None:
        with open(args.tiles_file, 'r') as f:
            args.tiles = [line.strip() for line in f if line.strip()]

    # A window is a crop; asking for a window with --crop false is contradictory.
    if args.window_km is not None:
        if args.window_km <= 0: parser.error('--window_km must be positive.')
        if not args.crop: parser.error('--window_km implies cropping; drop --crop false.')

    return args.tiles, args.year, args.output_dir, args.crop, args.num_workers, args.path_pkl, args.s2_index, args.window_km


def window_geometry(geometry, window_km):
    """
    Return a box of side window_km centred on the geometry's centroid, in EPSG:4326.

    A whole Sentinel-2 tile is ~110km, so cropping the AEF files to it yields ~2GB per tile. For a
    map figure a small window around the tile centre is enough (~20km -> ~20MB).

    The half-width is converted to degrees per axis using true metres-per-degree, so the window is
    window_km on the ground at any latitude. Do NOT do this by buffering in EPSG:3857: Web Mercator
    metres are true only at the equator, so a fixed buffer there shrinks the ground box by cos(lat)
    (this is exactly the bug that produced the shrunk AGBRef polygons).

    Args:
    - geometry: the tile geometry, in EPSG:4326.
    - window_km (float): the side of the window, in kilometres.

    Returns:
    - box: the window geometry, in EPSG:4326.
    """
    c = geometry.centroid
    half_m = window_km * 1000 / 2
    dlat = half_m / 110540
    dlon = half_m / (111320 * cos(radians(c.y)))
    return shapely_box(c.x - dlon, c.y - dlat, c.x + dlon, c.y + dlat)


def get_tile_to_aefiles(path_pkl):
    """
    This function loads the mapping from Sentinel-2 tile names, to years, to AEF files.

    Args:
    - path_pkl (str): the path to the directory containing tile_to_aefiles.pkl.

    Returns:
    - mapping (dict): maps a tile name to a year, and a year to the list of AEF S3 URIs covering it.
    """
    with open(join(path_pkl, 'tile_to_aefiles.pkl'), 'rb') as f:
        return pickle.load(f)


def get_tiles_geometries(tiles, s2_index):
    """
    This function looks up the geometries of the given Sentinel-2 tiles, in the Sentinel-2 index
    shapefile. The whole index is read once, and only the requested tiles are kept.

    Args:
    - tiles (list): the Sentinel-2 tile names.
    - s2_index (str): the path to the Sentinel-2 index shapefile.

    Returns:
    - geometries (dict): maps each tile name to its shapely geometry, in EPSG:4326.
    """
    gdf = gpd.read_file(s2_index, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
    gdf = gdf[gdf['Name'].isin(tiles)]
    if gdf.crs and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs('EPSG:4326')

    geometries = {row['Name']: row.geometry for _, row in gdf.iterrows()}
    missing = [t for t in tiles if t not in geometries]
    if missing: print(f'  WARNING: {len(missing)} tile(s) not found in the Sentinel-2 index: {missing}')

    return geometries


def get_tile_uris(tile, year, tile_to_aefiles, geometry, s3, zone_index_cache):
    """
    This function resolves the AEF S3 URIs covering a Sentinel-2 tile. It first looks the tile up in
    tile_to_aefiles.pkl; if the tile (or the year) is absent, it falls back to querying S3 for the
    files overlapping the tile geometry.

    Args:
    - tile (str): the Sentinel-2 tile name.
    - year (int): the AEF year.
    - tile_to_aefiles (dict): the mapping loaded from tile_to_aefiles.pkl.
    - geometry: the tile geometry in EPSG:4326, or None if it could not be resolved.
    - s3: boto3 S3 client.
    - zone_index_cache (dict): cache of the per-zone spatial indices, modified in place.

    Returns:
    - uris (list): the S3 URIs of the AEF files covering the tile.
    """
    if tile in tile_to_aefiles and year in tile_to_aefiles[tile]:
        return tile_to_aefiles[tile][year]

    if geometry is None:
        print(f'  WARNING: {tile} is not in tile_to_aefiles.pkl and has no geometry; skipping')
        return []

    print(f'  {tile} not in tile_to_aefiles.pkl for {year}; resolving from S3')
    return find_uris_for_aoi(geometry, year, s3, zone_index_cache = zone_index_cache)


def download_whole_file(uri, output_dir, tile, s3):
    """
    This function downloads a whole AEF file from S3, without cropping it. It mirrors the logic of
    download.py, but saves the file under a per-tile sub-directory.

    Args:
    - uri (str): the S3 URI of the AEF file.
    - output_dir (str): the directory in which to save the file.
    - tile (str): the Sentinel-2 tile name, prepended to the file name.
    - s3: boto3 S3 client.

    Returns:
    - status (str): a message indicating the download status.
    """
    key = '/'.join(uri.split('/')[4:])
    fname = f'{tile}_{basename(key)}'
    local_path = join(output_dir, fname)

    # Check if the file was already (successfully) downloaded
    if exists(local_path):
        info = s3.head_object(Bucket = 'tge-labs', Key = key)
        if getsize(local_path) == info['ContentLength']: return f'Skipped (exists): {fname}'

    try:
        makedirs(output_dir, exist_ok = True)
        s3.download_file('tge-labs', key, local_path)
        return f'Downloaded: {fname}'
    except Exception as e:
        return f'Error {uri}: {e}'


###################################################################################################
# Code execution

if __name__ == '__main__':

    # Setup
    tiles, year, output_dir, crop, num_workers, path_pkl, s2_index, window_km = parse_args()
    makedirs(output_dir, exist_ok = True)
    s3 = boto3.client('s3', endpoint_url = 'https://data.source.coop', config = Config(signature_version = UNSIGNED))
    tile_to_aefiles = get_tile_to_aefiles(path_pkl)

    # The geometries are needed to crop the files, and to resolve the tiles absent from the mapping
    need_geometries = crop or any((t not in tile_to_aefiles) or (year not in tile_to_aefiles.get(t, {})) for t in tiles)
    geometries = get_tiles_geometries(tiles, s2_index) if need_geometries else {}

    # Shrink each tile to a centred window, when asked. The AEF files covering the whole tile are
    # still resolved (that is what the mapping stores), but each is cropped to the window, and the
    # ones that miss it entirely are skipped by download_aoi.
    if window_km is not None:
        geometries = {t: window_geometry(g, window_km) for t, g in geometries.items()}
        print(f'Cropping to a {window_km:g}km window centred on each tile.')

    # Build the list of (uri, tile) tasks
    print(f'Building task list for {len(tiles)} tile(s), year {year}...')
    zone_index_cache, tasks = {}, []
    for tile in tiles:
        uris = get_tile_uris(tile, year, tile_to_aefiles, geometries.get(tile), s3, zone_index_cache)
        print(f'  {tile}: {len(uris)} file(s) found')
        for uri in uris: tasks.append((uri, tile))

    if not tasks:
        print('No matching AEF files found for any tile.')
        exit(0)

    # Launch the parallel downloads
    print(f"Downloading {len(tasks)} file(s) ({'cropped to the tile extent' if crop else 'whole files'})...")

    def _run(task):
        uri, tile = task
        tile_dir = join(output_dir, tile)
        if crop: status = download_aoi(uri, tile_dir, geometries[tile], aoi_id = tile)
        else: status = download_whole_file(uri, tile_dir, tile, s3)
        print(f'  {status}')
        return status

    with ThreadPoolExecutor(max_workers = num_workers) as executor:
        results = list(executor.map(_run, tasks))

    # Check how many files were downloaded successfully, and how many failed
    success_count = sum(1 for r in results if r.startswith('Downloaded'))
    skip_count = sum(1 for r in results if r.startswith('Skipped'))
    error_count = sum(1 for r in results if r.startswith('Error'))
    print(f'Download completed: {success_count} files downloaded, {skip_count} skipped, {error_count} errors.')

    # Write to a log file the ones that failed
    if error_count > 0:
        makedirs('dwn_errors', exist_ok = True)
        with open(join('dwn_errors', f'tiles_{year}_download_errors.log'), 'w') as log_file:
            for r in results:
                if r.startswith('Error'):
                    uri = r.split('Error ')[1].split(' ')[0].rstrip(':')
                    log_file.write(f'{uri}\n')
