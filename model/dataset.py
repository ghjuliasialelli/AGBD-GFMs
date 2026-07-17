"""

This script defines the dataset class for the GEDI dataset.

"""

############################################################################################################################
# IMPORTS

import time
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join, isdir
import pickle
from os.path import join
from datetime import datetime, timedelta
import argparse
import pandas as pd
from biomes import REF_BIOMES
from scipy.ndimage import distance_transform_edt
import random
random.seed(3)
np.seterr(divide = 'ignore')

# Define the nodata values for each data source
NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

continent_to_region = {'NorthAmerica': ['California', 'Cuba'], 'SouthAmerica': ['Paraguay', 'FrenchGuiana'],
    'Africa': ['UnitedRepublicofTanzania', 'Ghana'], 'Europe': ['Austria', 'Greece'],
    'SouthAsia': ['Nepal', 'ShaanxiProvince'], 'Australasia': ['NewZealand']}

############################################################################################################################
# Helper functions

def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5, inference, tile_name, oversampling_factors = None, hold_out_region = None, keep_region = False, drop_overlaps = False) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.

    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - chunk_size (int): the size of the chunks
    - path_mapping (str): the path to the file mapping each mode to its tiles
    - path_h5 (str): the path to the h5 files
    - inference (bool): whether to run in inference mode
    - tile_name (str): the name of the tile to use in inference mode
    - oversampling_factors (list): the oversampling factors for each AGB bin

    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)
    
    # If running in inference mode, only keep the required tile_name in the mapping
    if inference: tile_mapping = {mode: [tile_name]}

    # Skip the tiles in the region to hold out, if specified (only for train and val)
    if hold_out_region :
        # Mapping from e.g. New Zealand to the S2 tiles it contains
        with open(join(path_mapping, 'tiles_per_region.pkl'), 'rb') as f: tiles_per_region = pickle.load(f)
        # Mapping from world region (e.g. North America) to the regions in it (e.g. California, Cuba)
        subregions = continent_to_region.get(hold_out_region)
        hold_out_tiles = []
        for region in subregions : hold_out_tiles.extend(tiles_per_region[region])
    else : hold_out_tiles = []

    # If need to drop the test patches that overlap with the AEF train set
    if drop_overlaps:
        with open(join(path_mapping, 'AEF_overlaps.pkl'), 'rb') as f:
            overlap = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames :
        idx[fname] = {}
        
        with h5py.File(join(path_h5, fname), 'r') as f:
            
            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])
            
            # Iterate over the tiles
            for tile in tiles :

                if (keep_region and len(hold_out_tiles) > 0):
                    if tile not in hold_out_tiles : 
                        continue
                else: 
                    if tile in hold_out_tiles : continue

                # Get the number of patches in the tile
                if oversampling_factors is not None:
                    total = len(f[tile]['GEDI']['agbd'])
                    tile_factors = []
                    for i in range(0, total, 10000) :
                        agbd_values = f[tile]['GEDI']['agbd'][i : i + 10000]
                        bins = agbd_values // 50
                        factors = [oversampling_factors[int(b)] for b in bins]
                        tile_factors.extend(factors)
                    n_patches = sum(tile_factors)
                    idx[fname][tile] = {'n_patches' : n_patches // chunk_size, 'factors' : tile_factors}
                elif drop_overlaps :
                    if fname in overlap and tile in overlap[fname] : indices_to_skip = overlap[fname][tile]
                    else: indices_to_skip = []
                    n_total = len(f[tile]['GEDI']['agbd'])
                    n_patches = n_total - len(indices_to_skip)
                    idx[fname][tile] = {'n_patches' : n_patches, 'n_total' : n_total, 'indices_to_skip' : indices_to_skip}
                else:
                    n_patches = len(f[tile]['GEDI']['agbd'])
                    idx[fname][tile] = n_patches // chunk_size
    
    if (oversampling_factors is not None) or drop_overlaps : total_length = sum(sum(d['n_patches'] for d in idx[fname].values()) for fname in idx.keys())
    else: total_length = sum(sum(v for v in d.values()) for d in idx.values())

    return idx, total_length


def initialize_index_lite(fname, chunk_size, path_h5, mode = None) :
    """
    This function returns the total number of chunks in the AGBD-Lite dataset.

    Args:
    - fname (str): the name of the file
    - chunk_size (int): the size of the chunks
    - path_h5 (str): the path to the h5 files

    Returns:
    - total_length (int): the total number of chunks in the dataset
    """

    with h5py.File(join(path_h5, fname), 'r') as f:
        total_length = (len(f['GEDI']['agbd']) // chunk_size) + (1 if (len(f['GEDI']['agbd']) % chunk_size != 0) else 0)
    # AGBD-Lite uses a fixed key derived from `mode`; otherwise key by filename.
    index = {f'AGBD-Lite-{mode}.h5': None} if mode is not None else {fname: None}

    return index, total_length


def initialize_mapping_aef(fnames, path_h5) :
    """
    This function creates a mapping from Sentinel-2 tile names, to year, to the AEF file in which it's contained.

    Args:
    - fnames (list): list of AEF file names
    - path_h5 (str): the path to the AEF .h5 files

    Returns:
    - mapping (dict): dictionary mapping the tile names to the years and the years to the AEF files
    """

    mapping = {}
    for fname in fnames :
        year = int(fname.rstrip('.h5').split('_')[1])
        with h5py.File(join(path_h5, fname), 'r') as f:
            for tile in f.keys() :
                if tile not in mapping : mapping[tile] = {}
                mapping[tile][year] = fname
    return mapping


def init_ranges_for_chunk(index, total_length, oversampling = False, drop_overlaps = False):
    """
    This function creates a list of tuples (start_idx, end_idx, fname, tname) for each tile in the index, where
    start_idx and end_idx are the indices of the first and last chunk of the tile in the dataset. This will allow us to
    quickly find the file, tile, and row index corresponding to a given chunk index.

    Args:
    - index (dict): the index of the dataset, mapping file names to tile names and tile names to number of chunks
    - total_length (int): the total number of chunks in the dataset
    - oversampling (bool): whether to use oversampling or not
    - drop_overlaps (bool): whether to drop overlapping patches or not

    Returns:
    - ranges (list): list of tuples (start_idx, end_idx, fname, tname) for each tile in the index
    """

    ranges = []
    start_idx = 0
    for fname, file_data in index.items() :
        for tname, tile_data in file_data.items() :
            num_patches = tile_data['n_patches'] if oversampling or drop_overlaps else tile_data
            end_idx = start_idx + num_patches
            assert end_idx <= total_length, f"Index out of bounds: {end_idx} > {total_length}"
            ranges.append((start_idx, end_idx, fname, tname))
            start_idx = end_idx

    return ranges


def find_index_for_chunk(index, ranges, n, total_length, chunk_size, oversampling = False, lite = False, drop_overlaps = False) :
    """
    For a given `index`, `ranges`, and `n`-th chunk, find the file, tile, and row index corresponding to this chunk.
    
    Args:
    - index (dict): dictionary mapping the files to the tiles and the tiles to the chunks
    - ranges (list): list of tuples (start_idx, end_idx, fname, tname) for each tile in the index
    - n (int): the n-th chunk
    - total_length (int): the total number of chunks in the dataset
    - chunk_size (int): the size of the chunks
    - oversampling (bool): whether to use oversampling or not
    - lite (bool): whether to use the lite version of the dataset
    - drop_overlaps (bool): whether to drop overlapping patches or not

    Returns:
    - file_name (str): the name of the file
    - tile_name (str): the name of the tile
    - chunk_within_tile (int): the chunk index within the tile
    """

    # Check that the chunk index is within bounds
    assert n < total_length, "The chunk index is out of bounds"

    # If AGBD-Lite or AEF-Lite
    if lite : return None, None, n * chunk_size

    for start, end, fname, tname in ranges :
        if start <= n < end :
            chunk_within_tile = n - start

            if oversampling :
                factors = index[fname][tname]['factors']
                tile_cum_sum = 0
                for i in range(len(factors)) :
                    tile_cum_sum += factors[i]
                    if tile_cum_sum > chunk_within_tile :
                        chunk_within_tile = i
                        break
            
            elif drop_overlaps :
                tile_data = index[fname][tname]
                indices_to_skip = tile_data['indices_to_skip']
                n_total = tile_data['n_total']
                indices_to_keep = np.setdiff1d(np.arange(n_total), indices_to_skip)
                chunk_within_tile = indices_to_keep[chunk_within_tile]

            return fname, tname, chunk_within_tile


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
    offset_lon = (j_indices - patch_size[1] // 2) * resolution

    # Calculate the latitude and longitude for each pixel
    latitudes = central_lat + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon + (offset_lon / 6371000) * (180 / np.pi) / np.cos(central_lat * np.pi / 180)

    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos, lat_sin, lon_cos, lon_sin



def get_doy(num_days, patch_size, GEDI_START_MISSION = '2019-04-17') :
    """
    For a given number of days before/since the start of the GEDI mission, this function calculates
    the day of year (number between 1 and 365) and encodes it into sin/cosine values.

    Args:
    - num_days (int): the number of days before/since the start of the GEDI mission
    - GEDI_START_MISSION (str): the start date of the GEDI mission

    Returns:
    - (doy_cos, doy_sin) (tuple): the sin/cosine values for the day of year (doy_cos, doy_sin
    """

    # Get the date of acquisition and day of year
    start_date = datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d')
    target_date = start_date + timedelta(days = int(num_days))
    doy = target_date.timetuple().tm_yday

    # Get the doy_cos and doy_sin
    doy_cos = np.cos(2 * np.pi * doy / 365)
    doy_sin = np.sin(2 * np.pi * doy / 365)

    # Now we put everything in the [0,1] range
    doy_cos, doy_sin = (doy_cos + 1) / 2, (doy_sin + 1) / 2

    return np.full((patch_size[0], patch_size[1]), doy_cos), np.full((patch_size[0], patch_size[1]), doy_sin)


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
    px, py = np.gradient(dem, 10,)
    # Get the slope, in [0,1]
    slope = np.sqrt(px ** 2 + py ** 2)
    # Get the aspect, in [0,2pi]
    aspect = np.pi / 2 - np.arctan2(py, px)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    # Encode and scale the aspect, in [0,1]
    aspect_cos = (np.cos(aspect) + 1) / 2
    aspect_sin = (np.sin(aspect) + 1) / 2
    
    return slope, aspect_cos, aspect_sin


def normalize_data(data, norm_values, norm_strat, nodata_value = None, clip = True) :
    """
    Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
    - data (np.array): the data to normalize
    - norm_values (dict): the normalization values
    - norm_strat (str): the normalization strategy

    Returns:
    - normalized_data (np.array): the normalized data
    """

    if norm_strat == 'mean_std' :
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct' :
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else :
            data = (data - p1) / (p99 - p1)
        if clip : data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    
    else: 
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data


def normalize_bands(bands_data, norm_values, order, norm_strat, nodata_value = None) :
    """
    This function normalizes the bands data using the normalization values and strategy.

    Args:
    - bands_data (np.array): the bands data to normalize
    - norm_values (dict): the normalization values
    - order (list): the order of the bands
    - norm_strat (str): the normalization strategy
    - nodata_value (int/float): the nodata value

    Returns:
    - bands_data (np.array): the normalized bands data
    """
    
    for i, band in enumerate(order) :
        band_norm = norm_values[band]
        bands_data[:, :, i] = normalize_data(bands_data[:, :, i], band_norm, norm_strat, nodata_value)
    
    return bands_data


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


def offsetted_coords(lat, lon, min_offset = 0, max_offset = 0) :
    """
    This function randomly offsets the latitude and longitude values by a maximum of max_offset kilometers in each direction.

    Args:
    - lat (float) : the latitude (-90 to 90)
    - lon (float) : the longitude (-180 to 180)
    - max_offset (int) : the maximum offset in meters

    Returns:
    - lat (float) : the offsetted latitude
    - lon (float) : the offsetted longitude
    """

    # Get the random offset
    lat_offset = random.choice([1, -1]) * random.uniform(min_offset, max_offset) * 1000
    lon_offset = random.choice([1, -1]) * random.uniform(min_offset, max_offset) * 1000

    # Apply the offset
    # cf. https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-latitude-longitude-by-some-amount-of-meters
    R = 6371000 # Earth's radius
    dLat = lat_offset / R
    dLon = lon_offset / (R * np.cos(np.pi * lat / 180))

    # Add the offset
    lat = (lat + dLat * 180 / np.pi)
    lon = (lon + dLon * 180 / np.pi)

    # Cast the values to the valid range
    lat = (lat + 90) % 180 - 90
    lon = (lon + 180) % 360 - 180

    return lat, lon


class GEDIDataset(Dataset):

    def __init__(self, paths, years, chunk_size, mode, args, version = 4, debug = False, film = False, inference = False, tile_name = None, offset = False, return_region = False, min_offset = 0, max_offset = 0, mask_s2 = False):

        # Get the parameters
        self.h5_path, self.norm_path, self.mapping, self.embed_path = paths['h5'], paths['norm'], paths['map'], paths['embeddings']
        self.aef_h5_path, self.aef_norm_path = paths['aef_h5'], paths['aef_norm']
        self.tessera_h5_path, self.tessera_norm_path = paths['tessera_h5'], paths['tessera_norm']
        self.mode = mode
        self.chunk_size = chunk_size
        self.years = years
        self.film = film
        self.inference = inference
        self.offset = offset
        self.return_region = return_region
        self.min_offset, self.max_offset = min_offset, max_offset
        self.lite = args.lite
        self.lite_eval_big = args.lite_eval_big
        self.lite_chunk_size = args.lite_chunk_size
        self.lite_and_test = (self.lite and self.lite_eval_big and self.mode == 'test')
        self.hold_out_region = args.hold_out_region
        self.keep_region = args.keep_region
        self.stats_hold_out_region = args.stats_hold_out_region
        self.stats_keep_region = args.stats_keep_region
        self.oversampling = args.oversampling if self.mode == 'train' else False
        self.drop_overlaps = args.drop_overlaps if self.mode == 'test' else False
        if self.drop_overlaps:
            assert chunk_size == 1, "Dropping overlaps is only available for chunk_size = 1."
            assert not self.oversampling, "Dropping overlaps is not compatible with oversampling."
            if self.lite : assert self.lite_eval_big, "Dropping overlaps is only available for the full AGBD dataset."
        self.ensemble, self.n_members = args.ensemble, args.n_members

        # Get the files
        N = 2 if debug else 20
        if (not self.lite) or self.lite_and_test : # use the original .h5 files
            self.fnames = [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(N) for year in self.years]
        else: # use the AGBD-Lite .h5 files
            self.fnames = [f'AGBD-Lite-{self.mode}.h5']
            if isdir(join(self.h5_path, 'AGBD-Lite')) : self.h5_path = join(self.h5_path, 'AGBD-Lite')

        # Paths
        if self.lite :
            if isdir(join(self.norm_path, 'AGBD-Lite')): self.norm_path = join(self.norm_path, 'AGBD-Lite')

        # Whether to over-sampling from the minority AGB bins
        if self.oversampling : # get the over-sampling factors
            oversampling_factors = [1, 1, 1, 1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 10.0]
        else: oversampling_factors = None

        # Initialize the index
        if (not self.lite) or self.lite_and_test :
            self.index, self.length = initialize_index(self.fnames, self.mode, self.chunk_size, self.mapping, self.h5_path, inference, tile_name, oversampling_factors, self.hold_out_region, self.keep_region, self.drop_overlaps)
            self.ranges = init_ranges_for_chunk(self.index, self.length, oversampling = self.oversampling, drop_overlaps = self.drop_overlaps)
        else:
            self.index, self.length = initialize_index_lite(self.fnames[0], self.lite_chunk_size, self.h5_path)
            self.ranges = None
        
        
        # Define the data to use
        self.latlon = args.latlon
        self.bands = args.bands
        self.ch = args.ch
        self.s1 = args.s1
        self.alos = args.alos
        self.lc = args.lc
        self.dem = args.dem
        self.topo = args.topo
        self.aspect = args.aspect
        self.slope = args.slope
        self.gedi_dates = args.gedi_dates
        self.s2_dates = args.s2_dates
        self.s2_day = args.s2_day
        self.s2_doy = args.s2_doy
        self.region = args.region
        self.biome = args.biome
        self.patch_size = args.patch_size
        self.crop = args.crop
        self.aef = args.aef
        self.tessera = args.tessera
        self.predict = args.predict

        # Whether to mask the Sentinel-2 data for this mode
        self.mask_s2 = mask_s2

        # Whether to log transform the AGB values
        self.log_transform = args.log_transform

        # Input features flags
        self.ft_onehot = args.ft_onehot
        self.ft_cat2vec = args.ft_cat2vec
        self.ft_sincos = args.ft_sincos

        # Embeddings flags
        self.emb_cat2vec = args.emb_cat2vec
        self.emb_onehot = args.emb_onehot
        self.emb_dist = args.emb_dist
        self.emb_sincos = args.emb_sincos

        # Residuals flags
        self.residuals = args.residuals
        self.res_norm = args.res_norm
        self.res_film = args.res_film
        self.res_in = args.res_in
        self.res_in_central = args.res_in_central
        self.res_in_patch = args.res_in_patch

        # Flags for distance similarity optimization
        self.sim_dist = args.sim_dist

        # FiLM RH98
        self.rh98_film = args.rh98_film

        # Define the learning procedure
        self.norm_strat = args.norm_strat
        self.norm_target = args.norm
        self.prob_norm = args.prob_norm

        # Temporal ablation experiments
        self.temp_ablation = args.temp_ablation
        self.trained_years = args.trained_years
        self.years_stats = args.years_stats

        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'"

        # Load the normalization values
        if not self.lite:
            if self.stats_hold_out_region : # geographical ablations
                if self.stats_keep_region : region_str = self.stats_hold_out_region
                else:
                    default_str = "Europe-SouthAsia-Australasia-Africa-NorthAmerica-SouthAmerica"
                    region_str = default_str.replace(f"{self.stats_hold_out_region}-", '').replace(f"-{self.stats_hold_out_region}", '')
                statistics_fname = f"AGBD_statistics_2019-2020_{region_str}.pkl"
                print(f'Using file: {statistics_fname} for normalization values.')
                with open(join(self.norm_path, statistics_fname), mode = 'rb') as f:
                    self.norm_values = pickle.load(f)
            else: # standard setting
                if self.years_stats is not None: years_str = self.years_stats
                else:
                    if self.temp_ablation : years_str = '-'.join(str(year) for year in self.trained_years)
                    else: years_str = '-'.join(str(year) for year in self.years)
                print(f'Using file: AGBD_statistics_{years_str}_global.pkl for normalization values.')
                with open(join(self.norm_path, f"AGBD_statistics_{years_str}_global.pkl"), mode = 'rb') as f:
                    self.norm_values = pickle.load(f)
        else:
            print(f'Using file: AGBD-Lite-statistics.pkl for normalization values.')
            with open(join(self.norm_path, f"AGBD-Lite-statistics.pkl"), mode = 'rb') as f:
                self.norm_values = pickle.load(f)

        # "Global" AGB residuals flags
        self.agb_residuals = args.agb_residuals
        self.agb_residuals_film = args.agb_residuals_film
        self.agb_residuals_file = args.agb_residuals_file
        self.agb_res_all = args.agb_res_all
        self.agb_res_one = args.agb_res_one
        if self.agb_residuals or self.agb_residuals_film:
            with open(join(self.h5_path, self.agb_residuals_file), 'rb') as f:
                self.agb_res_stats = pickle.load(f)
            self.norm_values['agb_residuals'] = self.agb_res_stats['stats']
        if self.agb_res_all : self.agb_res_keys = ['min', 'max', 'mean', 'median', 'std']

        # Open the file handles
        self.handles = {fname: h5py.File(join(self.h5_path, fname), 'r') for fname in self.index.keys()}

        # Define the window size
        assert self.patch_size[0] == self.patch_size[1], "The patch size must be square"
        if self.crop:
            # min and max range for the center pixel value of the cropped patch
            self.minrange, self.maxrange = self.patch_size[0] // 2, 25 - self.patch_size[0] // 2 - 1
        else: 
            self.center_x, self.center_y = 12, 12 # because the patch size is 25x25 in the .h5 files
        self.window_size = self.patch_size[0] // 2
        
        # Get the cat2vec LC embeddings
        if self.emb_cat2vec or self.ft_cat2vec :
            embeddings = pd.read_csv(join(self.embed_path, f"embeddings_train{'_lite' if self.lite else ''}.csv"))
            embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
            self.embeddings = embeddings
        
        # Prepare for using AEF embeddings
        self.get_og_idx = False
        if self.aef :
            aef_fnames = ['California_2019.h5', 'UnitedRepublicofTanzania_2019_1-2.h5', 'California_2020.h5', 'Paraguay_2020.h5', 'UnitedRepublicofTanzania_2019_2-2.h5', 'Cuba_2020.h5', 'Nepal_2020.h5', 'NewZealand_2020.h5', 'Greece_2019.h5', 'Austria_2019.h5', 'ShaanxiProvince_2019.h5', 'FrenchGuiana_2020.h5', 'Cuba_2019.h5', 'Austria_2020.h5', 'Nepal_2019.h5', 'UnitedRepublicofTanzania_2020_1-2.h5', 'FrenchGuiana_2019.h5', 'Ghana_2020.h5', 'UnitedRepublicofTanzania_2020_2-2.h5', 'Paraguay_2019.h5', 'ShaanxiProvince_2020.h5', 'Ghana_2019.h5', 'Greece_2020.h5', 'NewZealand_2019.h5']
            aef_fnames = [f for f in aef_fnames for year in self.years if str(year) in f]
            self.aef_mapping = initialize_mapping_aef(aef_fnames, self.aef_h5_path)
            if self.lite: aef_stats_fname = "AEF_statistics_2020_global_lite.pkl"
            elif self.stats_hold_out_region:
                if self.stats_keep_region: aef_region_str = self.stats_hold_out_region
                else:
                    aef_default_str = "Europe-SouthAsia-Australasia-Africa-NorthAmerica-SouthAmerica"
                    aef_region_str = aef_default_str.replace(f"{self.stats_hold_out_region}-", '').replace(f"-{self.stats_hold_out_region}", '')
                aef_stats_fname = f"AEF_statistics_2019-2020_{aef_region_str}.pkl"
            else:
                if self.years_stats is not None: aef_years_str = self.years_stats
                elif self.temp_ablation: aef_years_str = '-'.join(str(year) for year in self.trained_years)
                else: aef_years_str = '-'.join(str(year) for year in self.years)
                aef_stats_fname = f"AEF_statistics_{aef_years_str}_global.pkl"
            print(f'Using file: {aef_stats_fname} for AEF normalization values.')
            with open(join(self.aef_norm_path, aef_stats_fname), mode = 'rb') as f:
                self.norm_values['AEF'] = pickle.load(f)
            self.aef_handles = {fname: h5py.File(join(self.aef_h5_path, fname), 'r') for fname in aef_fnames}
            if self.lite:
                if not (self.lite_eval_big and self.mode == 'test') :
                    self.get_og_idx = True
                    with open(join(self.aef_norm_path, 'mapping_lite_to_og.pkl'), 'rb') as f:
                        self.lite_to_og_mapping = pickle.load(f)[self.mode]

        # Prepare for using TESSERA embeddings
        if self.tessera :
            assert self.lite, "TESSERA embeddings are currently only supported with the Lite dataset."
            assert not (self.lite_eval_big and self.mode == 'test'), "TESSERA embeddings are not available for the non-Lite test set."
            tessera_fname = f'TESSERA-Lite-{self.mode}.h5'
            tessera_stats_fname = "TESSERA_statistics_2020_global_lite.pkl"
            print(f'Using file: {tessera_stats_fname} for TESSERA normalization values.')
            with open(join(self.tessera_norm_path, tessera_stats_fname), mode = 'rb') as f:
                self.norm_values['TESSERA'] = pickle.load(f)
            self.tessera_handle = h5py.File(join(self.tessera_h5_path, tessera_fname), 'r')


    def __len__(self):
        return int(self.length)
    
    def __getitem__(self, n):
            
        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx = find_index_for_chunk(self.index, self.ranges, n, self.length, self.lite_chunk_size, self.oversampling, self.lite and not self.lite_and_test, self.drop_overlaps)
        
        # Get the file handle
        if tile_name is not None: f = self.handles[file_name][tile_name]
        else: 
            file_name = self.fnames[0]
            f = self.handles[file_name]

        # Set the order for the Sentinel-1 bands
        if self.s1 and not hasattr(self, 's1_order') : self.s1_order = f['S1_bands'].attrs['order']

        # Set the order for the ALOS bands
        if self.alos and not hasattr(self, 'alos_order') : self.alos_order = f['ALOS_bands'].attrs['order']

        # If necessary, define the cropped patch
        if self.crop :
            self.center_x, self.center_y = random.randint(self.minrange, self.maxrange), random.randint(self.minrange, self.maxrange)
            # And store the indices of the ground-truth pixel in the newly cropped patch
            gt_x, gt_y = 12 - (self.center_x - self.window_size), 12 - (self.center_y - self.window_size)

        data = []

        # Sentinel-2 bands
        if self.bands != [] :

            # Set the order and indices for the Sentinel-2 bands
            if not hasattr(self, 's2_order') : self.s2_order = list(f['S2_bands'].attrs['order'])
            if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.bands]

            if self.mask_s2 and random.randint(0,1) :

                bogus_s2 = np.zeros((self.patch_size[0], self.patch_size[1], len(self.s2_indices)), dtype = np.float32)
                data.extend([bogus_s2])
            
            else : 
            
                # Get the bands
                s2_bands = f['S2_bands'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)
                
                # Get the BOA offset, if it exists
                if 'S2_boa_offset' in f['Sentinel_metadata'].keys() : 
                    s2_boa_offset = float(f['Sentinel_metadata']['S2_boa_offset'][idx])
                else: s2_boa_offset = 0

                # Get the surface reflectance values
                sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
                sr_bands[s2_bands == 0] = 0
                sr_bands[sr_bands < 0] = 0
                s2_bands = sr_bands

                # Normalize the bands
                s2_bands = normalize_bands(s2_bands, self.norm_values['S2_bands'], self.s2_order, self.norm_strat, NODATAVALS['S2_bands'])
                s2_bands = s2_bands[:, :, self.s2_indices]
                
                data.extend([s2_bands])
                
            if self.s2_dates : 
                s2_num_days = f['Sentinel_metadata']['S2_date'][idx]
                s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, self.patch_size)
                s2_num_days = np.full((self.patch_size[0], self.patch_size[1]), s2_num_days).astype(np.float32)
                s2_num_days = normalize_data(s2_num_days, self.norm_values['Sentinel_metadata']['S2_date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
                if self.s2_day:
                    data.extend([s2_num_days[..., np.newaxis]])
                if self.s2_doy:
                    data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
                            

        # Sentinel-1 bands
        if self.s1:
            s1_bands = f['S1_bands'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)
            s1_bands = normalize_bands(s1_bands, self.norm_values['S1_bands'], self.s1_order, self.norm_strat)
            
            s1_num_days = f['Sentinel_metadata']['S1_date'][idx, :]
            s1_doy_cos, s1_doy_sin = get_doy(s1_num_days, self.patch_size)
            s1_num_days = np.full((self.patch_size[0], self.patch_size[1]), s1_num_days).astype(np.float32)
            s1_num_days = normalize_data(s1_num_days, self.norm_values['Sentinel_metadata']['S1_date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
            
            data.extend([s1_bands, s1_num_days[..., np.newaxis], s1_doy_cos[..., np.newaxis], s1_doy_sin[..., np.newaxis]])
        
        # Latitude and longitude data
        if self.latlon :
            lat_offset, lat_decimal = f['GEDI']['lat_offset'][idx], f['GEDI']['lat_decimal'][idx]
            lon_offset, lon_decimal = f['GEDI']['lon_offset'][idx], f['GEDI']['lon_decimal'][idx]
            lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
            lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
            if self.offset : lat, lon = offsetted_coords(lat, lon, self.min_offset, self.max_offset)
            lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat, lon, self.patch_size)
            data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
        
        # GEDI dates
        if self.gedi_dates :
            gedi_num_days = f['GEDI']['date'][idx]
            gedi_doy_cos, gedi_doy_sin = get_doy(gedi_num_days, self.patch_size)
            gedi_num_days = np.full((self.patch_size[0], self.patch_size[1]), gedi_num_days).astype(np.float32)
            gedi_num_days = normalize_data(gedi_num_days, self.norm_values['GEDI']['date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
            data.extend([gedi_num_days[..., np.newaxis], gedi_doy_cos[..., np.newaxis], gedi_doy_sin[..., np.newaxis]])

        # ALOS bands
        if self.alos:

            # Get the bands
            alos_bands = f['ALOS_bands'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)

            # Get the gamma naught values
            alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)

            # Normalize the bands
            alos_bands = normalize_bands(alos_bands, self.norm_values['ALOS_bands'], self.alos_order, self.norm_strat, -9999.0)

            data.extend([alos_bands])
        
        # CH data
        if self.ch :
            ch = f['CH']['ch'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1]
            ch = normalize_data(ch, self.norm_values['CH']['ch'], self.norm_strat, NODATAVALS['CH'])
            
            ch_std = f['CH']['std'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1]
            ch_std = normalize_data(ch_std, self.norm_values['CH']['std'], self.norm_strat, NODATAVALS['CH'])

            data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])

        # LC data
        lc_data = f['LC'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :]
        lc_map, lc_prob = lc_data[:, :, 0], lc_data[:, :, 1]
        biome = lc_data[self.patch_size[0] // 2, self.patch_size[1] // 2, 0] # get the biome of the central pixel

        # For the LC input feature
        if self.lc :

            # In any case, calculate lc_prob (scale the class probabilities to [0,1])
            lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)
            if self.prob_norm: lc_prob = normalize_data(lc_prob, self.norm_values['LC']['lc_prob'], self.norm_strat, NODATAVALS['LC'])

            if self.ft_onehot : # get the one-hot encoding of the biome
                lc = one_hot_encode(lc_map, 'lc').astype(np.float32)
                data.extend([lc, lc_prob[..., np.newaxis]])
            elif self.ft_cat2vec : # get the cat2vec embedding of the biome
                lc = np.vectorize(lambda x: self.embeddings.get(x, self.embeddings.get(0)), signature = '()->(n)')(lc_map).astype(np.float32)
                data.extend([lc, lc_prob[..., np.newaxis]])
            elif self.ft_sincos : # get the sin/cosine encoding of the biome
                lc_cos = np.where(lc_map == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * lc_map / 100) + 1) / 2)
                lc_sin = np.where(lc_map == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * lc_map / 100) + 1) / 2)
                data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
            else: raise ValueError('No biome encoding strategy selected.')
        
        # For the FiLM biome embeddings
        if self.biome:
            if self.emb_cat2vec:
                biome_emb = self.embeddings.get(biome, self.embeddings.get(0)).astype(np.float32)
            elif self.emb_onehot:
                biome_emb = one_hot_encode(biome, 'lc').astype(np.float32)
            elif self.emb_dist:
                biome_emb = biome_distribution(lc_map)
            elif self.emb_sincos:
                if biome == NODATAVALS['LC'] : 
                    biome_emb = np.zeros(2).astype(np.float32)
                else:
                    lc_cos = (np.cos(2 * np.pi * biome / 100) + 1) / 2
                    lc_sin = (np.sin(2 * np.pi * biome / 100) + 1) / 2
                    biome_emb = np.array([lc_cos, lc_sin]).astype(np.float32)
        elif self.ensemble :
            biome_emb = np.zeros(self.n_members).astype(np.float32)
            biome_emb[np.random.randint(0, self.n_members)] = 1.0
        else: biome_emb = None

        # DEM data
        if self.topo :
            dem = f['DEM'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1]
            if self.slope or self.aspect : # Get the slope and aspect
                slope, aspect_cos, aspect_sin = get_topology(dem)
                if self.slope :
                    if self.prob_norm : slope = normalize_data(slope, self.norm_values['topo']['slope'], self.norm_strat, NODATAVALS['DEM'])
                    data.extend([slope[..., np.newaxis]])
                if self.aspect: data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])
            if self.dem :
                dem = normalize_data(dem, self.norm_values['DEM'], self.norm_strat, NODATAVALS['DEM'])
                data.extend([dem[..., np.newaxis]])

        # Get the GEDI region class (0=Water, 1=Europe, 2=North Asia, 3=Australasia, 4=Africa, 5=South Asia, 6=South America, 7=North America)
        region = f['GEDI']['region_cla'][idx]
        if self.region :
            region_cla = one_hot_encode(region, 'region_cla').astype(np.float32)
            if (biome_emb is not None) : biome_emb = np.concatenate([biome_emb, region_cla], axis = 0)
            else: biome_emb = region_cla

        # Compute the CHM - GEDI RH98 residuals
        if self.residuals :

            # Get the CH data
            ch = f['CH']['ch'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1]
            
            # Get the RH98 data
            rh98 = f['GEDI']['rh98'][idx]
            rh98 = np.where(rh98 < 0, 0, rh98) # remove negative values
            
            if not self.res_norm :
                ch = normalize_data(ch, self.norm_values['CH']['ch'], self.norm_strat, NODATAVALS['CH'], False) # no clipping
                rh98 = normalize_data(rh98, self.norm_values['GEDI']['rh98'], self.norm_strat, None, False) # no clipping

            # Calculate the residuals
            if self.res_film : # give the central pixel offset to the FiLM layers for conditioning
                central_residual = (ch[self.patch_size[0] // 2, self.patch_size[1] // 2].astype(np.float32) - rh98.astype(np.float32))[..., np.newaxis]
                if self.res_norm : central_residual = normalize_data(central_residual, self.norm_values['CH']['residuals'], self.norm_strat, None, False)
                if (biome_emb is not None) : biome_emb = np.concatenate([biome_emb, central_residual], axis = 0)
                else: biome_emb = central_residual
            
            elif self.res_in : # give the patch residuals to the model as input
                
                if self.res_in_central : # compute the patch's residuals as CH_central - RH98_central
                    patch_residual = ch[self.patch_size[0] // 2, self.patch_size[1] // 2] - rh98
                    patch_residual = np.full((self.patch_size[0], self.patch_size[1]), patch_residual).astype(np.float32)
                
                elif self.res_in_patch : # compute the patch's residuals as CH_patch - RH98_central
                    patch_residual = ch - rh98
                
                if self.res_norm : patch_residual = normalize_data(patch_residual, self.norm_values['CH']['residuals'], self.norm_strat, None, False)
                data.extend([patch_residual[..., np.newaxis]])
            
            else:
                raise ValueError('--residuals enabled but --res_in and --res_film both False.')
        
        # RH98 FiLM
        if self.rh98_film :
            rh98 = f['GEDI']['rh98'][idx]
            rh98 = np.where(rh98 < 0, 0, rh98)
            rh98 = normalize_data(rh98, self.norm_values['GEDI']['rh98'], self.norm_strat, None, True)[..., np.newaxis] # no clipping
            if (biome_emb is not None) : biome_emb = np.concatenate([biome_emb, rh98], axis = 0)
            else: biome_emb = rh98

        # AGB residuals stats
        if self.agb_residuals :
            try: stats = self.agb_res_stats[(region, biome)]
            except: # if not a (region,biome) we're interested in, fill with zeros
                if self.agb_res_all : res_stats = np.zeros((self.patch_size[0], self.patch_size[1], len(self.agb_res_keys))).astype(np.float32)
                else: res_stats = np.zeros((self.patch_size[0], self.patch_size[1], 1)).astype(np.float32)
            else:
                if self.agb_res_all : 
                    res_stats = np.array([stats[key] for key in self.agb_res_keys]).astype(np.float32)[:, np.newaxis] # (5,1)
                    res_stats = np.repeat(res_stats, self.patch_size[0] * self.patch_size[1], axis = 1) # (5, patch_size[0] * patch_size[1])
                    res_stats = res_stats.reshape(len(self.agb_res_keys), self.patch_size[0], self.patch_size[1]) # (5, patch_size[0], patch_size[1])
                    res_stats = res_stats.swapaxes(0, -1) # (patch_size[0], patch_size[1], 5)
                    res_stats = normalize_bands(res_stats, self.norm_values['agb_residuals'], self.agb_res_keys, self.norm_strat)
                else :
                    res_stats = np.full((self.patch_size[0], self.patch_size[1]), stats[self.agb_res_one]).astype(np.float32)
                    res_stats = normalize_data(res_stats, self.norm_values['agb_residuals'][self.agb_res_one], self.norm_strat, None, False)[..., np.newaxis]
            data.extend([res_stats])
        elif self.agb_residuals_film :
            try: stats = self.agb_res_stats[(region, biome)]
            except:
                if self.agb_res_all : res_stats = np.zeros(len(self.agb_res_keys),).astype(np.float32)
                else: res_stats = np.zeros(1,).astype(np.float32)
            else:
                if self.agb_res_all : 
                    res_stats = np.array([stats[key] for key in self.agb_res_keys]).astype(np.float32)[:, np.newaxis, np.newaxis].swapaxes(0, -1)
                    res_stats = normalize_bands(res_stats, self.norm_values['agb_residuals'], self.agb_res_keys, self.norm_strat).squeeze(0).squeeze(0)
                else : 
                    res_stats = np.array([stats[self.agb_res_one]]).astype(np.float32)[:, np.newaxis].swapaxes(0, -1)
                    res_stats = normalize_data(res_stats, self.norm_values['agb_residuals'][self.agb_res_one], self.norm_strat, None, False).squeeze(0)
            if (biome_emb is not None) : biome_emb = np.concatenate([biome_emb, res_stats], axis = 0)
            else: biome_emb = res_stats
            
        # Distribution similarity
        if self.inference or self.sim_dist : # get the CH patches
            ch = f['CH']['ch'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1].astype(np.float32)

        # AEF embeddings
        if self.aef :
            if self.get_og_idx : 
                og_tile, og_idx = self.lite_to_og_mapping[idx]
                year = 2020
            else: 
                og_tile, og_idx = tile_name, idx
                year = int(file_name.split('-')[1])
            aef_file = self.aef_mapping[og_tile][year]
            aef_data = self.aef_handles[aef_file][og_tile][og_idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)
            aef_data = np.where(aef_data == NODATAVALS, 0, (((aef_data / 127.5) ** 2) * np.sign(aef_data) + 1) / 2) # convert from int8 to float in range [0,1]
            # TODO here, should be NODATAVALS['AEF']
            aef_data = normalize_data(aef_data, self.norm_values['AEF'], 'mean_std', 0, False)
            data.extend([aef_data])

        # TESSERA embeddings
        if self.tessera :
            tessera_emb = self.tessera_handle['embeddings'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)
            tessera_scl = self.tessera_handle['scales'][idx, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1].astype(np.float32)
            tessera_data = tessera_emb * tessera_scl[..., None]
            tessera_data = np.nan_to_num(tessera_data, nan = 0.0)
            tessera_data = normalize_data(tessera_data, self.norm_values['TESSERA'], 'mean_std', 0, False)
            data.extend([tessera_data])

        # Concatenate the data together
        data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)

        # Get the target data
        if self.predict in ['agbd', 'rh98'] :
            target = f['GEDI'][self.predict][idx]
            if self.norm_target : 
                assert self.predict == 'agbd', "Target normalization is currently only implemented for biomass."
                target = normalize_data(target, self.norm_values['GEDI'][self.predict], self.norm_strat)
            target = torch.from_numpy(np.array(target, dtype = np.float32)).to(torch.float)
            if self.log_transform : target = torch.log1p(target)
        else: raise ValueError("Invalid target specified. Please choose one of 'agbd' or 'rh98'.")

        # Return the data
        to_return = (data, biome)
        if self.film and biome_emb is not None : to_return += (biome_emb,)
        to_return = to_return + (target,)
        if self.sim_dist and not self.inference : to_return += (ch,)
        if self.inference: to_return += (lat, lon, ch)
        elif self.crop: to_return += (gt_x, gt_y) # don't return the gt_x gt_y for inference
        if self.return_region: to_return += (region,)
        return to_return


############################################################################################################################
# Execute

if __name__ == '__main__' :

    # Paths come from config.sh at the repo root (see config.py), so this self-test
    # runs anywhere. config.py lives at the repo root, which is not on the path when
    # this module is run from model/.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_paths
    _paths = get_paths(local = True)

    config_dict = {
        "model_path": join(_paths['ckpt'], 'nico_film'),
        "model_name": join(_paths['ckpt'], 'nico_film', 'local'),
        "num_gpus": 1,
        "num_cpus": 8,
        "dataset_path": 'local',
        "augment": False,
        "norm": False,
        "chunk_size": 1,
        "lite": False,
        "lite_eval_big": False,
        "lite_chunk_size": 1,
        "aef": True,
        "drop_overlaps": False,
        "hold_out_region": None,
        "keep_region": False,
        "predict": 'agbd',
        "arch": 'nico_film',
        "model_idx": 0,
        "loss_fn": 'MSE',
        "film": True,
        "latlon": False,
        "ch": False,
        "bands": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        "in_features": 64,
        "s1": False,
        "alos": False,
        "lc": False,
        "dem": False,
        "gedi_dates": False,
        "s2_dates": False,
        "s2_day": False,
        "s2_doy": False,
        "topo": False,
        "aspect": False,
        "slope": False,
        "ft_cat2vec": False,
        "ft_onehot": False,
        "ft_sincos": False,
        "train_mask": False,
        "val_mask": False,
        "test_mask": False,
        "log_transform": False,
        "oversampling": False,
        "emb_cat2vec": False,
        "emb_onehot": True,
        "emb_dist": False,
        "emb_sincos": False,
        "residuals": False,
        "res_norm": False,
        "res_film": False,
        "res_in": False,
        "res_in_central": False,
        "res_in_patch": False,
        "biome_dim": 64,
        "emb_dim": 3,
        "linear_emb": False,
        "region": False,
        "biome": False,
        "debug_film": False,
        "bn": 'yes',
        "rh98_film": False,
        "ensemble": True,
        "n_members": 3,
        "num_outputs": 1,
        "norm_strat": 'pct',
        "prob_norm": False,
        "teacher": '',
        "teacher_arch": 'nico_film',
        "teacher_inpaint": False,
        "ndvi": False,
        "crop": False,
        "slide": True,
        "quality": False,
        "mixed": 'false',
        "cutoff": 0.5,
        "mixed_version": 1,
        "balanced": False,
        "_lambda": 'N/A',
        "filter": 'N/A',
        "pl_only": False,
        "sim_dist": False,
        "similarity": 'JS',
        "similarity_weight": 10.0,
        "SCC_ws": 5,
        "SCC_softmax": False,
        "agb_residuals": False,
        "agb_residuals_file": 'nico_film_17997535-1_17997535-2_17997535-3_train_agb_residuals_stats.pkl',
        "agb_res_all": False,
        "agb_res_one": 'mean',
        "agb_residuals_film": False,
        "debug_latlon": True,
        "new_stats": True,
        "n_epochs": 14,
        "limit": False,
        "batch_size": 128,
        "years": [2019, 2020],
        "freeze_shared": False,
        "random_spec": False,
        "scramble": False,
        "num_spec_layers": 1,
        "sigreg_lambda": 0.0,
        "channel_dims": [32, 32, 64, 128, 128, 128],
        "downsample": False,
        "max_pool": False,
        "leaky_relu": False,
        "num_sepconv_blocks": 8,
        "num_sepconv_filters": 256,
        "long_skip": True,
        "only_entry": True,
        "returns": 'dense',
        "padding_mode": 'zeros',
        "lr": 0.001,
        "step_size": 30,
        "gamma": 0.1,
        "l2": 1e-05,
        "patience": 1000,
        "min_delta": 0.0,
        "reweighting": 'no',
        "tile_name": None,
        "clip": True,
        "output_path": None,
        "n_models": None,
        "patch_size": [25, 25]
    }

    args = argparse.Namespace(**config_dict)

    local_dataset_paths = {'h5': _paths['h5'],
                            'norm': _paths['norm'],
                            'map': _paths['map'],
                            'embeddings': join(_paths['embeddings'], 'AGBD' if not args.lite else 'AGBD-Lite'),
                            'aef_h5': _paths['aef_h5'],
                            'aef_norm': _paths['aef_norm']}

    from tqdm import tqdm

    for mode in ['test'] : #, 'val', 'test'] :
        print('Processing {} data...'.format(mode))
        
        ds = GEDIDataset(local_dataset_paths, chunk_size = 1, mode = mode, args = args, debug = False, years = [2019], film = args.film)

        print('dataset created! length : ', len(ds))
        exit()

        # Create a DataLoader instance
        data_loader = DataLoader(dataset = ds,
                                batch_size = 128,
                                shuffle = True,
                                num_workers = 8,
                                pin_memory = False)

        # Iterate through the DataLoader

        print('starting to iterate...')
        t0 = time.time()
        for batch_samples in tqdm(data_loader):
            continue

            """
            images, _ = batch_samples
            
            # Check for NaN values
            if torch.isnan(images).any() : 
                print('Data is NaN')
            
            # CHeck for inf values
            if torch.isinf(images).any() : 
                print('Data is inf')
            
            # Check that data is in [0,1] range
            if torch.min(images) < 0 or torch.max(images) > 1 : 
                print('Data is not in [0,1] range')
            """
        t1 = time.time()
        print('done!')
        print('took : ', t1 - t0)

