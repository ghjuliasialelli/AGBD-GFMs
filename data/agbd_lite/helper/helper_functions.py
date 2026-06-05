"""

Helper functions for the dataset.py script.

"""

###################################################################################################
# Imports

import numpy as np
from scipy.ndimage import distance_transform_edt
import random
random.seed(3)
np.seterr(divide = 'ignore')

###################################################################################################
# Helper functions

NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}


# -------------------------------------------------------------------------------------------------
# For the latitute and longitude
# -------------------------------------------------------------------------------------------------

def encode_lat_lon(lat, lon) :
    """
    Encode the latitude and longitude into sin/cosine values. We use a simple WRAP positional encoding, as 
    Mac Aodha et al. (2019).

    Args:
    - lat (float): the latitude
    - lon (float): the longitude

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Now we put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin


def encode_coords(central_lat, central_lon, patch_size, resolution = 10) :
    """ 
    This function computes the latitude and longitude of a patch, from the latitude and longitude of its central pixel.
    It then encodes these values into sin/cosine values, and scales the results to [0,1].

    Args:
    - central_lat (float): the latitude of the central pixel
    - central_lon (float): the longitude of the central pixel
    - patch_size (tuple): the size of the patch
    - resolution (int): the resolution of the patch

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # Initialize arrays to store latitude and longitude coordinates
    i_indices, j_indices = np.indices(patch_size)

    # Calculate the distance offset in meters for each pixel
    offset_lat = (i_indices - patch_size[0] // 2) * resolution
    offset_lat = offset_lat[np.newaxis, :, :]
    offset_lon = (j_indices - patch_size[1] // 2) * resolution
    offset_lon = offset_lon[np.newaxis, :, :]

    # Calculate the latitude and longitude for each pixel
    latitudes = central_lat[:, np.newaxis, np.newaxis] + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon[:, np.newaxis, np.newaxis] + (offset_lon / 6371000) * (180 / np.pi) / np.cos(central_lat[:, np.newaxis, np.newaxis] * np.pi / 180)

    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]



# -------------------------------------------------------------------------------------------------
# For the dates (Sentinel-2, GEDI)
# -------------------------------------------------------------------------------------------------

def get_doy(num_days, patch_size, GEDI_START_MISSION = '2019-04-17') :
    """
    For a given number of days before/since the start of the GEDI mission, this function calculates
    the day of year (number between 1 and 365) and encodes it into sin/cosine values.

    Args:
    - num_days (int): the number of days before/since the start of the GEDI mission
    - GEDI_START_MISSION (str): the start date of the GEDI mission

    Returns:
    - (doy_cos, doy_sin) (tuple): the sin/cosine values for the day of year (doy_cos, doy_sin)
    """

    # Get the date of acquisition and day of year
    days_delta = np.array(num_days, dtype='timedelta64[D]')
    start_date = np.datetime64(GEDI_START_MISSION)
    target_dates = start_date + days_delta
    years = target_dates.astype('datetime64[Y]')
    doy = (target_dates - years).astype('timedelta64[D]').astype(int) + 1

    # Get the doy_cos and doy_sin
    doy_cos = np.cos(2 * np.pi * doy / 365)
    doy_sin = np.sin(2 * np.pi * doy / 365)

    # Now we put everything in the [0,1] range
    doy_cos, doy_sin = (doy_cos + 1) / 2, (doy_sin + 1) / 2

    # Patchify
    grid_cos = np.full((len(num_days), *patch_size, 1), doy_cos[:, np.newaxis, np.newaxis, np.newaxis])
    grid_sin = np.full((len(num_days), *patch_size, 1), doy_sin[:, np.newaxis, np.newaxis, np.newaxis])

    return grid_cos, grid_sin


# -------------------------------------------------------------------------------------------------
# For the topography (DEM, slope, aspect)
# -------------------------------------------------------------------------------------------------

def func_slope(px, py) :
    return np.sqrt(px ** 2 + py ** 2)

def func_aspect(px, py) :
    aspect = np.pi / 2 - np.arctan2(py, px)
    return np.where(aspect < 0, aspect + 2 * np.pi, aspect)

def get_topology(dem) :
    """
    This function computes the slope and aspect of the DEM.
    
    Resources: 
    . https://www.spatialanalysisonline.com/HTML/gradient__slope_and_aspect.htm
    . https://gis.stackexchange.com/questions/361837/calculating-slope-of-numpy-array-using-gdal-demprocessing
    . https://math.stackexchange.com/a/3923660

    Args:
    - dem (np.array, shape batch_size, patch_size, patch_size): the DEM

    Returns:
    - slope (np.array): the slope of the DEM
    - aspect_cos (np.array): the cosine of the aspect of the DEM
    - aspect_sin (np.array): the sine of the aspect of the DEM
    """

    # Where the DEM is not available, we take the nearest one available
    if np.any(dem == NODATAVALS['DEM']) :
        mask = (dem == NODATAVALS['DEM'])
        _, indices = distance_transform_edt(mask, return_indices = True) # Calculate the distance to the nearest non-invalid cell
        dem = dem[tuple(indices)]
    
    # Get the partial derivatives
    px, py = np.gradient(dem, 10, axis = (1, 2))
    # Get the slope, in [0,1]
    slope = np.sqrt(px ** 2 + py ** 2)
    # Get the aspect, in [0,2pi]
    aspect = np.pi / 2 - np.arctan2(py, px)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    # Encode and scale the aspect, in [0,1]
    aspect_cos = (np.cos(aspect) + 1) / 2
    aspect_sin = (np.sin(aspect) + 1) / 2
    
    return slope, aspect_cos, aspect_sin


# -------------------------------------------------------------------------------------------------
# For the biome / Land Classification
# -------------------------------------------------------------------------------------------------

REF_BIOMES = {'20': 'Shrubs', '30': 'Herbaceous vegetation', '40': 'Cultivated', 
        '90': 'Herbaceous wetland', '111': 'Closed-ENL', '112': 'Closed-EBL', '114': 'Closed-DBL', 
        '115': 'Closed-mixed', '116': 'Closed-other', '121': 'Open-ENL', '122': 'Open-EBL', 
        '124': 'Open-DBL', '125': 'Open-mixed', '126': 'Open-other'}


def encode_lc(lc_data) :

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]

    # Encode the LC classes with sin/cosine values and scale the data to [0,1]
    lc_cos = np.where(lc_map == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * lc_map / 100) + 1) / 2)
    lc_sin = np.where(lc_map == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * lc_map / 100) + 1) / 2)

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_cos, lc_sin, lc_prob


def embed_lc(lc_data, embeddings) :
    """
    Embed the land cover classes using the cat2vec embeddings.

    Args:
    - lc_data (np.array): the land cover data
    - embeddings (dict): the cat2vec embeddings

    Returns:
    - lc_map (np.array): the embedded land cover classes
    - lc_prob (np.array): the land cover class probabilities
    """

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]
    lc_map = np.vectorize(lambda x: embeddings.get(x, embeddings.get(0)), signature = '()->(n)')(lc_map).astype(np.float32)

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_map, lc_prob


_biome_values_mapping = {int(v): i for i, v in enumerate(REF_BIOMES.keys())}
def one_hot_encode(data, dtype) :
    """
    One-hot encode the data.

    Args:
    - data (np.array): the data to one-hot encode
    - dtype (str): the data type

    Returns:
    - one_hot_data (np.array): the one-hot encoded data
    """

    # Define the number of classes and the values mapping
    if dtype == 'region_cla' :
        num_classes = 8
        values_mapping = {i:i for i in range(num_classes)}
    elif dtype == 'lc' : 
        num_classes = 14
        values_mapping = _biome_values_mapping
    else: raise ValueError(f'Data `{dtype}` is not eligible for one-hot encoding.')

    # Actually perform the one-hot encoding
    def one_hot(x) :
        one_hot = np.zeros(num_classes)
        one_hot[values_mapping.get(x, 0)] = 1
        return one_hot
    
    one_hot_data = np.vectorize(one_hot, signature = '() -> (n)')(data).astype(np.float32)

    return one_hot_data


_ref_biome_values = [int(v) for v in REF_BIOMES.keys()]
def biome_distribution(patch_lc) :
    """
    This function computes the distribution of biomes in a patch.

    Args:
    - patch_lc (np.array): the land cover classes in the patch, of size (patch_size, patch_size)

    Returns:
    - biome_emb (np.array): the biome distribution, of size (num_classes,)
    """
    # Number of pixels in the patch
    num_pixels = patch_lc.size
    # Percentage of each biome in the patch
    counts = {value: np.count_nonzero(patch_lc == value) / num_pixels for value in _ref_biome_values}
    return np.array(list(counts.values())).astype(np.float32)
