"""

This script processes the ESA CCI tiles. It takes as input a list of Sentinel-2 tiles, and for each of them, it lists the CCI
tiles that intersect it, mosaics them, crops the resulting tile to the bounds of the Sentinel-2 tile, and saves the resulting
tile to a GeoTIFF file.

Execution:
    python process_tiles.py --tilenames /path/to/tile_names.txt 
                            --output_path /path/to/output_folder
                            --path_shp /path/to/Sentinel-2_index_shapefile.shp
                            --AGBRef (optional)
                            --year 2019
                            --version (optional, default: 4.0)
                            --i 0
                            --N 10

To process the AGBRef tiles:
    python process_tiles.py --output_path /scratch3/gsialelli/CCI/
            --path_shp /scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/AGBRef/AGBRef.geojson
            --AGBRef 
            --version 6.0

"""

############################################################################################################################
# IMPORTS

from os.path import join, exists
import geopandas as gpd
import numpy as np
from list_tiles import get_tiles_from_coordinates, get_true_bounds, round_tilename
from rasterio.merge import merge
from rasterio.mask import mask
import rasterio as rs
import argparse

local_path_shp = join('/scratch3', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'download_Sentinel', 'sentinel_2_index_shapefile.shp')

############################################################################################################################
# Helper functions

def setup_parser():
    """ 
    Set up the parser for the command line arguments.
    """

    parser = argparse.ArgumentParser()

    # Paths arguments
    parser.add_argument("--tilenames", type = str,
                   help = "Path to the .txt file listing the tiles to consider.") 
    parser.add_argument('--AOI', type = str, nargs = '*', default = [],
                        help = 'The AOI(s) for which to list the available granules')
    parser.add_argument('--path_shp', help = 'Path to the Sentinel-2 index shapefile.', default = local_path_shp)
    parser.add_argument("--output_path", type = str, required = True, 
                   help = "Path to the folder where the tiles will be downloaded.")
    parser.add_argument('--version', type = str, default = '4.0',
                   help = "Version of the ESA CCI dataset to process (default: 4.0).")
    parser.add_argument('--AGBRef', action = 'store_true', help = 'Whether to process the tiles at the AGBRef plots level.')
    parser.add_argument('--force', action = 'store_true', help = 'Re-crop tiles even if the output file already exists (e.g. after the plot geometries changed).')

    # Arguments for the procedure
    parser.add_argument("--year", type = int, help = "Year for which to download the tiles.")
    parser.add_argument('--i', help = 'Process split i/N.', type = int, default = 0)
    parser.add_argument('--N', help = 'Total number of splits.', type = int, default = 1)
    
    args = parser.parse_args()

    if not args.AGBRef : 
        if args.AOI != [] : 
            tilenames = f"/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_{'_'.join(args.AOI)}.txt"
        else: 
            tilenames = args.tilenames
            # Check that the tilenames argument is a .txt file
            if tilenames is not None: 
                if not tilenames.endswith('.txt'): 
                    raise ValueError('Please provide a .txt file for the --tilenames argument.')
        assert args.year is not None, "Please provide a year for which to process the tiles using the --year argument."
    else: tilenames, year = None, None

    return tilenames, args.output_path, args.year, args.path_shp, args.i, args.N, args.version, args.AGBRef, args.force


def list_s2_tiles(tilenames, grid_df) :
    """
    This function performs two tasks: 1) return the list of Sentinel-2 tile names for which we want to extract patches (this
    is done either reading the tiles listed in a .txt file); and 2) return the geometries of those tiles, from the Sentinel-2
    grid shapefile.

    Args:
    - tilenames: string, path to a .txt file listing the Sentinel-2 tiles to consider.
    - grid_df: geopandas dataframe, Sentinel-2 grid shapefile.

    Returns:
    - tile_names: list of strings, names of the Sentinel-2 tiles to consider.
    - tile_geoms: list of shapely geometries, geometries of the Sentinel-2 tiles to consider.
    """
    
    # List the tiles from the .txt file
    with open(tilenames) as f: 
        tile_names = [tile_name.strip() for tile_name in f.readlines()]
    
    # Get the geometries from the Sentinel-2 grid shapefile
    tile_geoms = [grid_df[grid_df['Name'] == tile_name]['geometry'].values[0] for tile_name in tile_names]

    return tile_names, tile_geoms


def mosaic_tiles(tiles, band, year, path_CCI_data, version) :
    """
    Mosaic the CCI tiles.

    Args:
    - tiles: list of str, names of the CCI tiles
    - band: str, name of the band to process (AGB or AGB_SD)
    - year: int, year for which to process the CCI tiles
    - path_CCI_data: str, path to the folder where the CCI data is stored

    Returns:
    - numpy.ndarray, the mosaic of the CCI tiles
    """
    
    # Open the tiles' files (they need to be open for the merge operation)
    to_mosaic = []
    for tile in tiles :
        src = rs.open(join(path_CCI_data, f'{tile}_ESACCI-BIOMASS-L4-{band}-MERGED-100m-{year}-fv{version}.tif'))
        to_mosaic.append(src)
        mosaic_crs = src.crs
    
    # Merge the tiles
    mosaic, mosaic_trans = merge(to_mosaic)

    # Close the tiles' files
    for src in to_mosaic : src.close()
        
    return mosaic, mosaic_trans, mosaic_crs


def save_tile(s2_tname, year, bands_data, output_path) :
    """
    Save the CCI tile to a GeoTIFF file.

    Args:
    - bands_data: dict, keys are the names of the bands, and values are the data of the bands
    - output_path: str, path to the folder where the processed tiles will be stored

    Returns:
    - None
    """
    
    # Extract the bands' data
    bands_values = list(bands_data.values())

    # Check that the bands have the same shape, transform and CRS
    shapes = [val['mosaic'].shape for val in bands_values]
    assert len(set(shapes)) == 1, "The shapes of the bands are not the same"
    transforms = [val['transform'] for val in bands_values]
    assert len(set(transforms)) == 1, "The transforms of the bands are not the same"
    crs = [val['crs'] for val in bands_values]
    assert len(set(crs)) == 1, "The CRS of the bands are not the same"


    CCI_meta = {'driver': 'GTiff', 'height': shapes[0][1], 'width': shapes[0][2], \
                'transform': transforms[0], 'crs': crs[0], 'count' : 2, 'nodata' : 65535, \
                'compress': 'lzw', 'dtype': 'uint16'} 
    
    # Write the bands to the file
    fname = f'CCI_{s2_tname}_{str(year)[-2:]}.tif'
    with rs.open(join(output_path, fname), 'w', **CCI_meta) as dst:
        for band_id, (band_name, band_data) in enumerate(bands_data.items()):
            dst.write(band_data['mosaic'][0, :, :], band_id + 1)
            dst.set_band_description(band_id + 1, band_name)


############################################################################################################################
# Main function

def process_CCI_tiles(s2_tname, s2_geom, year, path_CCI_data, version) :
    """
    For a given Sentinel-2 tile, list the ESA CCI tiles that intersect it, mosaic them, and crop the resulting tile to the
    bounds of the Sentinel-2 tile. Then save the resulting tile to a GeoTIFF file.

    Args:
    - args: tuple, containing the following elements:
        - s2_tile_data: tuple, (int, pandas.core.series.Series), the index and the geometry of the Sentinel-2 tile
        - year: int, year for which to process the CCI tiles
        - path_CCI_data: str, path to the folder where the CCI data is stored

    Returns:
    - bool, whether the ESA CCI tile was successfully processed
    """
    
    try:

        # Get the corresponding CCI tiles
        lon_min, lat_min, lon_max, lat_max = s2_geom.bounds
        if abs(lon_min - lon_max) > 180 :
            meridian_flag = True
            lon_min, lat_min, lon_max, lat_max = get_true_bounds(s2_geom)
        else: meridian_flag = False
        CCI_tiles = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)
        CCI_tiles = np.unique([round_tilename(tname) for tname in CCI_tiles])
        print(f'>> Got {len(CCI_tiles)} CCI tiles.')

        # Iterate over the CCI bands of interest (AGB and AGB_SD)
        bands_data = {}
        for band in ['AGB', 'AGB_SD'] :

            # Mosaic the CCI tiles
            mosaic, mosaic_trans, mosaic_crs = mosaic_tiles(CCI_tiles, band, year, path_CCI_data, version)

            # Crop to the Sentinel-2 tile (use a Memory File because `mask` requires an dataset open in 'r' mode)
            with rs.MemoryFile() as memfile:
                with memfile.open(driver = 'GTiff', height = mosaic.shape[1], width = mosaic.shape[2], count = 1, 
                                    dtype = mosaic.dtype, crs = mosaic_crs, transform = mosaic_trans) as mosaic_vrt:
                    mosaic_vrt.write(np.squeeze(mosaic, axis = 0), 1)
                with memfile.open() as src:
                    mosaic, mosaic_trans = mask(src, shapes = [s2_geom], crop = True)
            
            # Store the data
            bands_data[band] = {'mosaic': mosaic, 'transform': mosaic_trans, 'crs': mosaic_crs}
        
        # Save the CCI tile
        save_tile(s2_tname, year, bands_data, path_CCI_data)

    except Exception as e:
        print('Error', e)
        return f'Error processing {s2_tname}: {e}'


############################################################################################################################
# Execute

if __name__ == "__main__":

    tilenames, path_CCI_data, year, path_shp, i, N, version, AGBRef, force = setup_parser()

    if not AGBRef:
        # Read the Sentinel-2 grid shapefile
        grid_df = gpd.read_file(path_shp, engine = 'pyogrio')
        # List all S2 tiles and their geometries
        tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df)
        assert len(tile_names) == len(tile_geoms), "The number of tile names and geometries is not the same."
        years = [year] * len(tile_names)
    
    else:
        all_plots = gpd.read_file(path_shp)
        tile_names, years, tile_geoms = all_plots.index.values.tolist(), all_plots['AVG_YEAR'].tolist(), all_plots['geometry'].tolist()
        tile_names = [f'AGBRef_{tile_name}' for tile_name in tile_names]
        years = [max(2018, year) for year in years]
        assert len(tile_names) == len(tile_geoms) == len(years), "The number of tile names, geometries and years is not the same."


    # Split into N, and process the i-th split
    print(f'split {i}/{N}')
    tile_names = np.array_split(tile_names, N)[i]
    tile_geoms = np.array_split(tile_geoms, N)[i]
    years = np.array_split(years, N)[i]
    assert len(tile_names) == len(tile_geoms), "The number of tile names and geometries is not the same."

    tiles_num = len(tile_names)
    print(f'Processing {tiles_num} tile(s)...')

    for tile_idx, (tile_name, tile_geom, year) in enumerate(zip(tile_names, tile_geoms, years)) :

        # Check if the file has already been processed
        print(f'({tile_idx + 1}/{tiles_num}) Extracting for tile {tile_name}...')
    
        if not force and exists(join(path_CCI_data, f'CCI_{tile_name}_{str(year)[-2:]}.tif')) :
            print('already processed')
            continue

        print(f'Processing tile {tile_name}...')
        process_CCI_tiles(tile_name, tile_geom, year, path_CCI_data, version)


"""
python process_tiles.py --tilenames /scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_Ghana.txt --output_path /scratch3/gsialelli/CCI/ --year 2019

N = 10
for year in [2018, 2019, 2020, 2021, 2022] :
for i in range(N) :
    print(f'nohup python process_tiles.py --tilenames /scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.txt --output_path /scratch3/gsialelli/CCI/ --year {year} --i {i} --N 10 > logs/mosaicing-{year}-{i}-{N}.txt 2>&1 &')

year = 2019
AOIs = []
Ns = []
assert len(AOIs) == len(Ns), "The number of AOIs and Ns is not the same."
for AOI, N in zip(AOIs, Ns) :
    for i in range(N) :
        print(f'nohup python process_tiles.py --tilenames /scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_{AOI}.txt --output_path /scratch3/gsialelli/CCI/ --year {year} --i {i} --N {N} > logs/mosaicing-{AOI}-{year}-{i}-{N}.txt 2>&1 &')
    
    
"""