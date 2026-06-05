"""

This script finds the indices of the patches in the AGBD test dataset that are included in an AEF
training tile. 

It saves the indices in a dictionary of the form:
{
    'data_subset-2019-v4_0-20.h5': {
        '32TNN': [0, 1, 2, 3, 4, 5],
        '32TMM': [10, 11, 12],
        ...
    },
    'data_subset-2019-v4_1-20.h5': {
        '32TNN': [100, 101, 102],
        ...
    },
    ...
}

This will be used in the GEDIDataset() class to drop the overlapping samples.

env: dwn

"""

###################################################################################################
# Imports

import pandas as pd
import h5py
import numpy as np
import pickle
import geopandas as gpd
from rasterio.crs import CRS
from os.path import join, isfile

###################################################################################################
# Helper functions

def get_S2_bounds(tile_name, grid_df) :
    """
    Get the bounds of a Sentinel-2 tile from its name.

    Args:
    - tile_name: str, name of the Sentinel-2 tile.
    - path_shp: str, path to the shapefile containing the Sentinel-2 grid.

    Returns:
    - tile_geom: shapely.geometry.Polygon, the geometry of the Sentinel-2 tile.
    """

    # Get the geometry of the tile
    tile_geom = grid_df[grid_df['Name'] == tile_name]['geometry'].values[0]

    return tile_geom

def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return CRS.from_string(crs)


###################################################################################################
# Code execution

if __name__ == '__main__':

    lite = True

    # AGBD .h5 files
    path_h5 = '/scratch3/gsialelli/patches'

    # Load the subsampled indices
    if lite :
        years = [2020]
        with open(join(path_h5, 'subsampled_indices.pkl'), 'rb') as f: 
            subsampled_indices = pickle.load(f)
    else: years = [2019, 2020]

    h5_files = [f'data_subset-{year}-v4_{i}-20.h5' for i in range(20) for year in years]
    output_fname = f"AEF{'-Lite' if lite else ''}_overlaps.pkl"

    if isfile(output_fname) :
        print(f"{output_fname} already exists.")
        exit(1)

    # Mapping from mode to tiles
    with open('split_to_tiles.pkl', 'rb') as f: split_to_tiles = pickle.load(f)
    test_tiles = split_to_tiles['test']

    # Load the geometries of the Sentinel-2 tiles
    path_shp = "/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
    grid_s2_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # Get the AEF training points
    df = pd.read_csv("distribution/training_coordinates.csv")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    num_overlaps = {}

    for fname in h5_files:
        print(fname)
        with h5py.File(join(path_h5, fname), 'r') as f:

            tiles = list(f.keys())
            overlap = np.intersect1d(tiles, test_tiles)
            if len(overlap) == 0: continue
            num_tiles = len(overlap)

            for i, tile in enumerate(overlap):
                print('    processing tile', tile, f'({i+1}/{num_tiles})')

                f_tile = f[tile]['GEDI']

                # Get the geometry of the tile
                tile_geom = get_S2_bounds(tile, grid_s2_df)
                tile_crs = get_CRS_from_S2_tilename(tile)

                # Get the intersection between the S2 tile geometry and the AEF geometries
                gdf_within_tile = gdf[gdf.within(tile_geom)]
                if len(gdf_within_tile) == 0 : continue
                gdf_within_tile = gdf_within_tile.to_crs(tile_crs)
                gdf_within_tile['geometry'] = gdf_within_tile.buffer(640, cap_style = 3) # create a 1.28km x 1.28km patch around each point
                gdf_within_tile = gdf_within_tile.to_crs("EPSG:4326")

                # Get the coordinates of the patches in the tile
                lat_offset, lat_decimal = f_tile['lat_offset'][:], f_tile['lat_decimal'][:]
                lon_offset, lon_decimal = f_tile['lon_offset'][:], f_tile['lon_decimal'][:]
                lats = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
                lons = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
                idxs = np.arange(len(lats))
                
                # Only keep the subsampled indices if in lite mode
                if lite:
                    if tile not in subsampled_indices: continue
                    tile_indices = subsampled_indices[tile]
                    idxs = idxs[tile_indices]
                    lats = lats[tile_indices]
                    lons = lons[tile_indices]
                
                # Make a GeoDataFrame of the coordinates
                coords_df = pd.DataFrame({'latitude': lats, 'longitude': lons, 'idx': idxs})
                coords_gdf = gpd.GeoDataFrame(coords_df, geometry=gpd.points_from_xy(coords_df.longitude, coords_df.latitude), crs="EPSG:4326")

                if fname not in num_overlaps: num_overlaps[fname] = {}
                if tile not in num_overlaps[fname]: num_overlaps[fname][tile] = 0
                overlaps = gpd.sjoin(coords_gdf, gdf_within_tile, how='inner', predicate='within')
                num_overlaps[fname][tile] = overlaps['idx'].unique().tolist()
                        
    
    # Save
    with open(output_fname, 'wb') as f: pickle.dump(num_overlaps, f)