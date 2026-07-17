"""

This script lists all the ESA CCI tiles for given AOIs, or globally.

Execution:
    python list_tiles.py    --AOI California Cuba Paraguay UnitedRepublicofTanzania Ghana Austria Greece Nepal
                                ShaanxiProvince NewZealand FrenchGuiana 
                            --path_geojson {path_geojson} (optional)
                            --path_output /scratch3/gsialelli/BiomassDatasetCreation/CCI/txt_tiles
                            --tilenames {path_to_tilenames} (optional)


To run with AGBRef:
    --AOI AGBRef
    --path_geojson /scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/AGBRef/AGBRef.geojson

"""

############################################################################################################################
# IMPORTS 

import math
import tqdm
import requests
import argparse
import numpy as np
import geopandas as gpd
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle

############################################################################################################################
# Helper functions

def setup_args_parser() :
    """ 
    Setup the arguments parser for the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_output', type = str, required = True,
                        help = 'Directory in which to write the `CCI_<AOI>.txt` file.')
    parser.add_argument('--path_geojson', type = str,
                        help = 'Path to the .geojson file to consider for the geometries.',
                        default = join('/scratch3', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson'))
    parser.add_argument('--tilenames', type = str, required = False, help = 'Path to a .txt file listing the S2 tiles to consider.')
    parser.add_argument('--AOI', type = str, nargs = '*', default = 'global',
                        help = 'The AOI(s) for which to list the available granules')
    args = parser.parse_args()
    return args.path_output, args.path_geojson, args.AOI, args.tilenames


def rounddown(x, integ = 1):
    """
    Round down to the nearest multiple of `integ`. We return positive values only.
    """
    return math.floor(abs(x) / integ) * integ


def roundup(x, integ = 1):
    """
    Round up to the nearest multiple of `integ`. We return positive values only.
    """
    return int(np.sign(x) * int(math.ceil(abs(x) / integ)) * integ)


def define_filename(AOI) :
    """
    For given AOI(s), this function defines the name of the file that will contain the list of available granules.
    
    Args:
    - AOI (str, or str list) : the AOI(s) for which to list the available granules

    Returns:
    - filename (str) : the name of the file containing the list of available granules
    """
    if AOI != 'global': AOI = '_'.join(AOI)
    return f"CCI_{AOI}.txt"


############################################################################################################################
# Main functions

def list_all_CCI_tiles() :
    """
    TODO
    """

    url = 'https://data.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v4.0/geotiff/2019?json'
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        all_tiles = []
        for tile_data in json_data['items']:
            tile_name = tile_data['path'].split('/')[-1].split('-')[0].split('_')[0]
            all_tiles.append(tile_name)
        unique_tiles = np.unique(all_tiles)
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
    
    return unique_tiles


def get_tile_from_xy(lat, lon) :
    """
    For a given latitude and longitude, this function identifies the ALOS PALSAR-2 tile unit (1x1 degree) tile that
    contains the point. The naming convention for the tiles is <lat_letter><lat_number>_<lon_letter><lon_number>,
    where the letters are N or S for the latitude, and E or W for the longitude.
    
    We identify the tile by the following rules:
        .for latitude:
            if lat < 0 : N<roundup(abs(lat))>
            elif lat in [0;1] : N00
            else: S<rounddown(lat)>
        . for longitude:
            if lon < 0 : W<roundup(abs(lon))>
            else: E<rounddown(lon)>

    Args:
    - (lat, lon) (float) : the latitude and longitude of the point (in degrees

    Returns:
    - lat_letter (str) : the letter for the latitude
    - lat_number (int) : the number for the latitude
    - lon_letter (str) : the letter for the longitude
    - lon_number (int) : the number for the longitude
    """
    
    if lat > 0 :
        lat_letter = 'N'
        lat_number = roundup(lat)
    elif -1 < lat <= 0 : 
        lat_letter = 'N'
        lat_number = 0
    else:
        lat_letter = 'S'
        lat_number = rounddown(abs(lat))
    if lon < 0 :
        lon_letter = 'W'
        lon_number = roundup(abs(lon))
    else:
        lon_letter = 'E'
        lon_number = rounddown(lon)
    
    return lat_letter, lat_number, lon_letter, lon_number


def get_lat_lon_range_from_tile(lat_letter, lat_number, lon_letter, lon_number) :
    """
    For a given ALOS PALSAR-2 tile unit (1x1 degree) tile, this function returns the range of latitudes and longitudes
    that it spans. 

    Args:
    - lat_letter (str) : the letter for the latitude
    - lat_number (int) : the number for the latitude
    - lon_letter (str) : the letter for the longitude
    - lon_number (int) : the number for the longitude

    Returns:
    - latitudes (tuple of int) : the range of latitudes that the tile spans
    - longitudes (tuple of int) : the range of longitudes that the tile spans
    """

    # For the latitudes
    if lat_letter == 'S' : lat_number = -lat_number
    latitudes = (lat_number - 1, lat_number)

    # For the longitudes
    if lon_letter == 'W' : lon_number = -lon_number
    longitudes = (lon_number, lon_number + 1)

    return latitudes, longitudes


def get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag = False):
    """
    Given `lat_min, lat_max, lon_min, lon_max` this function returns all of the ALOS PALSAR-2 unit (1x1 degree)
    tiles that intersect the bounding box defined by these coordinates. The `meridian_flag` is set to True when
    the bounding box spans the 180th meridian. In this case, need to consider the tiles that span the meridian.

    Args:
    - (lat_min, lat_max, lon_min, lon_max) (float) : the bounding box coordinates
    - meridian_flag (bool) : whether the bounding box spans the 180th meridian

    Returns:
    - tiles (list of str) : the list of ALOS PALSAR-2 unit tiles that intersect the bounding box
    """

    # Identify the 1x1 degree tiles that contain the corners of the bounding box
    start_lat_letter, start_lat_number, start_lon_letter, start_lon_number = get_tile_from_xy(lat_min, lon_min)
    end_lat_letter, end_lat_number, end_lon_letter, end_lon_number = get_tile_from_xy(lat_max, lon_max)
    
    # Fill in the gap between the start and end tiles
    
    latitudes = []

    # If the latitudes are in the same hemisphere, simply get the tiles between the start and end tiles
    if start_lat_letter == end_lat_letter :

        # When in the negative range, we need to reverse the order of the tiles
        if start_lat_letter == 'S' : start_lat_number, end_lat_number = end_lat_number, start_lat_number
        for i in range(start_lat_number, end_lat_number + 1) :
            latitudes.append(f"{start_lat_letter}{i:02n}")
    
    # Otherwise, get the tiles from the start tile to the equator, and from the equator to the end tile
    else: 

        # Will be S, so from 1 to start_lat_number
        for i in range(1, start_lat_number + 1) :
            latitudes.append(f"{start_lat_letter}{i:02n}")
        # Will be N, so from 0 to end_lat_number
        for i in range(0, end_lat_number + 1) :
            latitudes.append(f"{end_lat_letter}{i:02n}")

    longitudes = []

    # If the longitudes are in the same hemisphere, simply get the tiles between the start and end tiles
    if start_lon_letter == end_lon_letter :

        # When in the negative range, we need to reverse the order of the tiles
        if start_lon_letter == 'W' : start_lon_number, end_lon_number = end_lon_number, start_lon_number
        for i in range(start_lon_number, end_lon_number + 1) :
            longitudes.append(f"{start_lon_letter}{i:03n}")
    
    # Otherwise, get the tiles from the start tile to the 0th meridian, and from the 0th meridian to the end tile
    else:
        # Corner case when the bounding box spans the 180th meridian
        if meridian_flag :
            # Go from the start tile to E179
            for i in range(start_lon_number, 180) :
                longitudes.append(f"{start_lon_letter}{i:03n}")
            # And from the end tile to W180
            for i in range(end_lon_number, 180 + 1) :
                longitudes.append(f"{end_lon_letter}{i:03n}")
        else:
            # Will be W, so from 1 to start_lon_number
            for i in range(1, start_lon_number + 1) :
                longitudes.append(f"{start_lon_letter}{i:03n}")
            # Will be E, so from 0 to end_lon_number
            for i in range(0, end_lon_number + 1) :
                longitudes.append(f"{end_lon_letter}{i:03n}")

    # Sanity check for the results
    results = [f"{lat}{lon}" for lat in latitudes for lon in longitudes]
    assert len(results) > 0, "No match found."

    return results


def round_tilename(name) :
    """
    Given a unit (1x1 degree) ALOS PALSAR-2 tile name, this function returns the name of the 10 x 10 degree tile that
    contains it. 

    Args:
    - name (str) : the name of the unit (1x1 degree) tile, in the format <lat_letter>xx<lon_letter>xxx

    Returns:
    - name (str) : the name of the 10x10 degree tile that contains the unit tile (<lat_letter>xx<lon_letter>xxx)
    """

    # Parse the unit tile name name
    lat_letter, lat_abs, lon_letter, lon_abs = name[0], int(name[1:3]), name[3], int(name[4:7])

    # Corner case: N00 spans from N00 to S09
    if lat_letter == 'S' and lat_abs <= 9 : lat_letter, lat_abs = 'N', 0
    # Round, e.g., N44 to N50
    if lat_letter == 'N' : lat = roundup(lat_abs, integ = 10)
    # and S11 to S10
    else : lat = rounddown(lat_abs, integ = 10)
    # Round, e.g., W72 to W80
    if lon_letter == 'W' : lon = roundup(lon_abs, integ = 10)
    # and E03 to E00
    else: lon = rounddown(lon_abs, integ = 10)

    return "{}{:02n}{}{:03n}".format(lat_letter, lat, lon_letter, lon)


def get_true_bounds(geometry) :
    """
    For a given geometry, this function returns the "true" min/max lat/lon values. This is necessary when the geometry
    spans the 180th meridian, as the `bounds` method of the geometry object blindly returns the min/max lat/lon values,
    disregarding the mixture of positive and negative values. We need to separate the positive and negative longitudes,
    and define lon_min, lat_min, lon_max, lat_max = min(pos_lon), max(latitudes), max(neg_lon), min(latitudes). Note
    that the use of the `.geoms` attribute is necessary, cf. https://stackoverflow.com/a/76493457.

    Args:
    - geometry (geopandas.geoseries.GeoSeries) : the geometry for which to get the true bounds

    Returns:
    - lon_min, lat_min, lon_max, lat_max (float) : the true min/max lat/lon values    
    """

    points = []
    for polygon in geometry.values[0].geoms :
        points.extend(polygon.exterior.coords[:-1])
    longitudes, latitudes = [point[0] for point in points], [point[1] for point in points]
    pos_lon = [l for l in longitudes if l > 0]
    neg_lon = [l for l in longitudes if l <= 0]

    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(pos_lon), max(neg_lon)

    return lon_min, lat_min, lon_max, lat_max


def visual_inspection(tiles) :
    """
    This function allows for a visual inspection of the tiles on a map. Given a list of ESA CCI tiles (in their
    official format, i.e. <10x10 degree tile name>), this function plots the tiles on a map using the Basemap
    library.

    Args:
    - tiles (list of str) : the list of ESA CCI tiles to plot

    Returns:
    - None
    """

    # Setup the figure
    fig, ax = plt.subplots()
    m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='l')
    m.drawcoastlines()

    # Iterate over the tiles
    min_x1, max_x2, min_y1, max_y2 = np.inf, -np.inf, np.inf, -np.inf
    for tile in tiles :
        lat_letter, lat_number, lon_letter, lon_number = tile[0], int(tile[1:3]), tile[3], int(tile[4:7])
        (min_lat, max_lat), (min_lon, max_lon) = get_lat_lon_range_from_tile(lat_letter, lat_number, lon_letter, lon_number)

        # Plot the tile on the map
        x1, y1 = m(min_lon, min_lat)
        x2, y2 = m(max_lon, max_lat)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        rect = Rectangle((x1, y1), width, height, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)

        # Update the min/max lat/lon values
        min_x1, max_x2 = min(min_x1, x1), max(max_x2, x2)
        min_y1, max_y2 = min(min_y1, y1), max(max_y2, y2)

    # Set the aspect of the plot to be equal so that the square is not distorted
    ax.set_xlim(min_x1 - 1000000, max_x2 + 1000000)
    ax.set_ylim(min_y1 - 1000000, max_y2 + 1000000)
    ax.set_aspect('equal')
    plt.savefig('visualization.png')


############################################################################################################################
# Execute

if __name__ == '__main__' :

    path_output, path_geojson, AOI, tilenames = setup_args_parser()
    all_tiles = []

    # Case 1 : we list the tiles from a file
    if tilenames is not None:
        print('Listing tiles from a file...')

        # Read the list of tiles from a file
        with open(tilenames, 'r') as f: tiles = [t.strip() for t in f.readlines()]
        grid_df = gpd.read_file(join('/scratch3', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'download_Sentinel', 'sentinel_2_index_shapefile.shp'), engine = 'pyogrio')
        
        for tile in tqdm.tqdm(tiles) :
            country = grid_df[grid_df.Name == tile]
            lon_min, lat_min, lon_max, lat_max = country.geometry.values[0].bounds
            
            # Corner case when a tile spans the 180th meridian
            if abs(lon_min - lon_max) > 180 :
                meridian_flag = True
                lon_min, lat_min, lon_max, lat_max = get_true_bounds(country.geometry)
            else: meridian_flag = False
            
            tnames = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)
            prefixes = [round_tilename(tname) for tname in tnames]
            all_tiles.extend(prefixes)
    
    # Case 2 : we list the tiles from an AOI
    else:
        print('Listing tiles for an AOI...')

        # Get the tiles for the whole CCI coverage
        if AOI == 'global' : all_tiles = list_all_CCI_tiles()

        else:

            for aoi in AOI:

                # Extract for the AGBD AOIs
                if aoi in ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana'] :

                    print(f'Extracting tiles for {aoi}...')
                    countries = gpd.read_file(path_geojson)
                    country = countries[countries['name'] == aoi]
                    lon_min, lat_min, lon_max, lat_max = country.geometry.values[0].bounds

                    # Corner case when an AOI spans the 180th meridian
                    if abs(lon_min - lon_max) > 180 :
                        meridian_flag = True
                        lon_min, lat_min, lon_max, lat_max = get_true_bounds(country.geometry)
                    else: meridian_flag = False

                    tnames = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)
                    prefixes = [round_tilename(tname) for tname in tnames]
                    all_tiles.extend(prefixes)

                elif aoi == 'AGBRef' :

                    print(f'Extracting tiles for {aoi}...')
                    all_regions = gpd.read_file(path_geojson)
                    
                    # For each geomtry in the AGBRef geojson, get the tiles that intersect it

                    for idx, row in tqdm.tqdm(all_regions.iterrows(), total = len(all_regions)) :

                        geometry = row.geometry
                        lon_min, lat_min, lon_max, lat_max = geometry.bounds

                        year = max(row.AVG_YEAR, 2018) # map 2017 to 2018

                        # Corner case when an AOI spans the 180th meridian
                        if abs(lon_min - lon_max) > 180 :
                            meridian_flag = True
                            lon_min, lat_min, lon_max, lat_max = get_true_bounds(geometry)
                        else: meridian_flag = False

                        tnames = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)
                        prefixes = [f'{round_tilename(tname)}_{year}' for tname in tnames]
                        all_tiles.extend(prefixes)

                    pass

                else: 
                    raise ValueError(f"AOI {aoi} not recognized. Please choose from California, Cuba, Paraguay, \
                                     UnitedRepublicofTanzania, Ghana, Austria, Greece, Nepal, ShaanxiProvince, \
                                     NewZealand, FrenchGuiana, or AGBRef.")

    
    all_tiles = np.unique(all_tiles)

    # Write the list of tiles to a file
    fname = define_filename(AOI)
    with open(join(path_output, fname), 'w') as f:
        # We append an extra \n character, otherwise using `wc -l` will not count the last line
	    f.write('\n'.join(all_tiles) + '\n')
