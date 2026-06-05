"""

This script defines the dataset class for the GEDI dataset. It is augmented by the `HelperSampler` class, which
samples from the dataset per biome. This is useful when we want to train the model on a batch of samples from the
same biome.

"""

############################################################################################################################
# IMPORTS

import pandas as pd
import time
import h5py
from biomes import REF_BIOMES
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from os.path import join
import pickle
import math
from os.path import join, exists
from datetime import datetime, timedelta
import argparse
import random
np.seterr(divide = 'ignore') 

# Define the nodata values for each data source
NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

############################################################################################################################
# Helper functions

def initialize_index(fnames, mode, path_mapping, path_h5, biomes) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.

    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - path_mapping (str): the path to the file mapping each mode to its tiles
    - path_h5 (str): the path to the h5 files
    - biomes (list): the list of biomes

    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)

    # Iterate over the biomes
    idx = {}
    for biome in biomes :
        idx[biome] = {}
        
        # Iterate over all files
        for fname in fnames :
            idx[biome][fname] = {}

            with h5py.File(join(path_h5, fname), 'r') as f:
                
                # Get the tiles in this file which belong to the mode
                all_tiles = list(f.keys())
                tiles = np.intersect1d(all_tiles, tile_mapping[mode])
                
                # Iterate over the tiles
                for tile in tiles :
                    
                    # Get the patches that belong to the biome
                    labels = f[tile]['LC'][:, 12, 12, 0]
                    patches = np.where(labels == int(biome))[0]
                    idx[biome][fname][tile] = patches

    total_length = {biome: sum(len(indices) for fdata in idx[biome].values() for indices in fdata.values()) for biome in idx}

    return idx, total_length


def find_index_for_chunk(index, ns, total_length):
    """
    This function finds the files, tiles and sample indices corresponding to given sample numbers.

    Args:
    - index (dict): the index of the dataset. The structure is [file][tile] = [indices]
    - ns (list): the list of sample numbers
    - total_length (int): the total number of samples

    Returns:
    - tup_original_order (list): the list of (file, tile, i) tuples corresponding to the sample numbers
    """

    # Order the ns, but remember the original order
    indexed_list = list(enumerate(ns))
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    tup_sorted = []

    # Iterate over the index to find the file, tile, and row index
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, indices in file_data.items():
            num_rows = len(indices)

            found = []
            for (i, n) in sorted_list:
                if cumulative_sum + num_rows > n:
                    # Calculate the row index within the tile
                    chunk_within_tile = n - cumulative_sum
                    tup = (file_name, tile_name, indices[chunk_within_tile])
                    tup_sorted.append((i, tup))
                    found.append((i, n))
            # Update the list of indices to be found
            for (i, n) in found: sorted_list.remove((i, n))
            
            cumulative_sum += num_rows

    # Reorder the results to match the input order
    tup_original_order = [x for i, x in sorted(tup_sorted, key=lambda x: x[0])]
    return tup_original_order


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


def normalize_data(data, norm_values, norm_strat, nodata_value = None) :
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
        data = np.clip(data, 0, 1)

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


class GEDIDataset_biomes(Dataset):

    def __init__(self, paths, years, mode, args, version = 4, debug = False):

        # Get the parameters
        self.h5_path, self.norm_path, self.mapping = paths['h5'], paths['norm'], paths['map']
        self.mode = mode
        self.years = years
        
        # Get the file names
        self.fnames = []
        for year in self.years : 
            if debug : self.fnames += [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(2)]
            else: self.fnames += [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(20)]
        
        # Initialize the index
        with open("biomes_index_2019-2020_v4.pkl", 'rb') as f: self.index = pickle.load(f)[mode]

        # Only keep the biomes that are in REF_BIOMES
        self.index = {biome: {fname: self.index[biome][fname] for fname in self.fnames} for biome in self.index if biome in REF_BIOMES}
        self.length = {biome: sum(len(indices) for fdata in self.index[biome].values() for indices in fdata.values()) for biome in self.index if biome in REF_BIOMES}
        
        # Only keep the biomes with non-zero length
        self.index = {biome: self.index[biome] for biome in self.index if self.length[biome] > 0}
        self.length = {biome: self.length[biome] for biome in self.length if self.length[biome] > 0}

        # Define the data to use
        self.latlon = args.latlon
        self.bands = args.bands
        self.ch = args.ch
        self.s1 = args.s1
        self.alos = args.alos
        self.lc = args.lc
        self.cat2vec = args.cat2vec
        self.dem = args.dem
        self.gedi_dates = args.gedi_dates
        self.s2_dates = args.s2_dates
        self.patch_size = args.patch_size

        # Define the learning procedure
        self.norm_strat = args.norm_strat
        self.norm_target = args.norm

        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'"

        # Load the normalization values
        if not exists(join(self.norm_path, f'statistics_subset_2019-2020-v{version}.pkl')):
            raise FileNotFoundError(f'The file `statistics_subset_2019-2020-v{version}.pkl` does not exist.')
        with open(join(self.norm_path, f'statistics_subset_2019-2020-v{version}.pkl'), mode = 'rb') as f:
            self.norm_values = pickle.load(f)

        # Open the file handles
        self.handles = {fname: h5py.File(join(self.h5_path, fname), 'r') for fname in self.fnames}

        # Define the window size
        assert self.patch_size[0] == self.patch_size[1], "The patch size must be square"
        self.center = 12 # because the patch size is 25x25 in the .h5 files
        self.window_size = self.patch_size[0] // 2
        
        # Get the cat2vec LC embeddings
        if self.lc and self.cat2vec :
            embeddings = pd.read_csv(join(self.embed_path, "embeddings_train.csv"))
            embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
            self.embeddings = embeddings


    def __len__(self):
        return sum(list(self.length.values()))
    
    def __getitem__(self, tuple):
 
        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx = tuple
        idx = int(idx)
        
        # Get the file handle
        f = self.handles[file_name]

        # Set the order and indices for the Sentinel-2 bands
        if not hasattr(self, 's2_indices') : self.s2_order = list(f[tile_name]['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.bands]

        # Set the order for the Sentinel-1 bands
        if self.s1 and not hasattr(self, 's1_order') : self.s1_order = f[tile_name]['S1_bands'].attrs['order']

        # Set the order for the ALOS bands
        if self.alos and not hasattr(self, 'alos_order') : self.alos_order = f[tile_name]['ALOS_bands'].attrs['order']


        data = []

        # Sentinel-2 bands
        if self.bands != [] :
            
            # Get the bands
            s2_bands = f[tile_name]['S2_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            
            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f[tile_name]['Sentinel_metadata'].keys() : 
                s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx]
            else: s2_boa_offset = 0

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands

            # Normalize the bands
            s2_bands = normalize_bands(s2_bands, self.norm_values['S2_bands'], self.s2_order, self.norm_strat, NODATAVALS['S2_bands'])
            s2_bands = s2_bands[:, :, self.s2_indices]
            
            s2_num_days = f[tile_name]['Sentinel_metadata']['S2_date'][idx]
            s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, self.patch_size)
            s2_num_days = np.full((self.patch_size[0], self.patch_size[1]), s2_num_days).astype(np.float32)
            s2_num_days = normalize_data(s2_num_days, self.norm_values['Sentinel_metadata']['S2_date'], 'min_max')

            data.extend([s2_bands])
            
            # TODO figure out what to do with the Sentinel-2 dates
            if self.s2_dates : data.extend([s2_num_days[..., np.newaxis], s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
            

        # Sentinel-1 bands
        if self.s1:
            s1_bands = f[tile_name]['S1_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            s1_bands = normalize_bands(s1_bands, self.norm_values['S1_bands'], self.s1_order, self.norm_strat)
            
            s1_num_days = f[tile_name]['Sentinel_metadata']['S1_date'][idx, :]
            s1_doy_cos, s1_doy_sin = get_doy(s1_num_days, self.patch_size)
            s1_num_days = np.full((self.patch_size[0], self.patch_size[1]), s1_num_days).astype(np.float32)
            s1_num_days = normalize_data(s1_num_days, self.norm_values['Sentinel_metadata']['S1_date'], 'min_max')
            
            data.extend([s1_bands, s1_num_days[..., np.newaxis], s1_doy_cos[..., np.newaxis], s1_doy_sin[..., np.newaxis]])
        
        # Latitude and longitude data
        lat_offset, lat_decimal = f[tile_name]['GEDI']['lat_offset'][idx], f[tile_name]['GEDI']['lat_decimal'][idx]
        lon_offset, lon_decimal = f[tile_name]['GEDI']['lon_offset'][idx], f[tile_name]['GEDI']['lon_decimal'][idx]
        lat, lon = lat_offset + lat_decimal, lon_offset + lon_decimal
        # TODO should actually be :
        # lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
        # lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
        lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat, lon, self.patch_size)
        if self.latlon : data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
        else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])
        
        # GEDI dates
        # TODO define what to do with the GEDI dates
        if self.gedi_dates :
            gedi_num_days = f[tile_name]['GEDI']['date'][idx]
            gedi_doy_cos, gedi_doy_sin = get_doy(gedi_num_days, self.patch_size)
            gedi_num_days = np.full((self.patch_size[0], self.patch_size[1]), gedi_num_days).astype(np.float32)
            gedi_num_days = normalize_data(gedi_num_days, self.norm_values['GEDI']['date'], 'min_max')
            data.extend([gedi_num_days[..., np.newaxis], gedi_doy_cos[..., np.newaxis], gedi_doy_sin[..., np.newaxis]])

        # ALOS bands
        if self.alos:

            # Get the bands
            alos_bands = f[tile_name]['ALOS_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)

            # Get the gamma naught values
            alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)

            # Normalize the bands
            alos_bands = normalize_bands(alos_bands, self.norm_values['ALOS_bands'], self.alos_order, self.norm_strat, -9999.0)

            data.extend([alos_bands])
        
        # CH data
        if self.ch:
            ch = f[tile_name]['CH']['ch'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            ch = normalize_data(ch, self.norm_values['CH']['ch'], self.norm_strat, NODATAVALS['CH'])
            
            ch_std = f[tile_name]['CH']['std'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            ch_std = normalize_data(ch_std, self.norm_values['CH']['std'], self.norm_strat, NODATAVALS['CH'])

            data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
        
        # LC data
        lc = f[tile_name]['LC'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :]
        biome = lc[self.center, self.center, 0]
        if self.lc:
            if self.cat2vec:
                lc, lc_prob = embed_lc(lc, self.embeddings)
                data.extend([lc, lc_prob[..., np.newaxis]])
            else:
                lc_cos, lc_sin, lc_prob = encode_lc(lc)
                data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
        
        # DEM data
        if self.dem:
            dem = f[tile_name]['DEM'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            dem = normalize_data(dem, self.norm_values['DEM'], self.norm_strat, NODATAVALS['DEM'])
            data.extend([dem[..., np.newaxis]])
        
        # Concatenate the data together
        data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)

        # Get the GEDI target data
        agbd = f[tile_name]['GEDI']['agbd'][idx]
        if self.norm_target :
            agbd = normalize_data(agbd, self.norm_values['GEDI']['agbd'], self.norm_strat)
        agbd = torch.from_numpy(np.array(agbd, dtype = np.float32)).to(torch.float)

        return data, biome, agbd


############################################################################################################################

class PerBiomeDistributedSampler(DistributedSampler):
    """
    Custom DistributedSampler to take care of multi-GPU training with the HelperSampler.
    """

    def __init__(self, dataset, num_replicas = 1, rank = 0, shuffle = True, seed = 0, drop_last = True, batch_size = 32):
        """
        Args:
        - dataset (Dataset): the dataset to sample from
        - num_replicas (int): the number of replicas
        - rank (int): the rank of the current process
        - seed (int): the seed for the random number generator
        - batch_size (int): the batch size

        Warning (TODO) : In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
        """

        # Checking the distributed aspect
        print('(Inside) Rank: ', rank, 'Num replicas: ', num_replicas)
        num_replicas = num_replicas
        rank = rank
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        
        # Setting other attributes
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.initialized = False
        
        # Computing num_samples
        self.num_samples = {}
        for biome, biome_samples in self.dataset.length.items() :
            if biome_samples % self.num_replicas != 0 :
                self.num_samples[biome] = math.ceil((biome_samples - self.num_replicas) / self.num_replicas)
            else:
                self.num_samples[biome] = math.ceil(biome_samples / self.num_replicas)
        
        # Computing total_size
        self.total_size = {biome: self.num_samples[biome] * self.num_replicas for biome in self.num_samples}

        print(f'Initializing Sampler for rank {self.rank} out of {self.num_replicas} replicas.')

    def __iter__(self):

        if not self.initialized:
            print(f'First __iter__ for Sampler for rank {self.rank} out of {self.num_replicas} replicas.')
            self.initialized = True

        # Set the seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = {}
        for biome, biome_samples in self.dataset.length.items() :

            # Shuffle the indices
            indices[biome] = torch.randperm(biome_samples, generator = g).tolist()

            # Remove tail of data to make it evenly divisible
            indices[biome] = indices[biome][ : self.total_size[biome]]
            assert len(indices[biome]) == self.total_size[biome]

            # Subsample for the current rank
            indices[biome] = indices[biome][self.rank : self.total_size[biome] : self.num_replicas]
            assert len(indices[biome]) == self.num_samples[biome]
    
        batch_sampler = HelperSampler(self.dataset, batch_size = self.batch_size, indices = indices, gen = g)
        return iter(batch_sampler)

    def __len__(self):
        """
        Returns the number of batches that each replica will see.
        """
        return sum(biome_samples // self.batch_size for biome_samples in self.num_samples.values())

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
        - epoch (int): Epoch number.
        """
        self.epoch = epoch


class PerBiomeSampler(Sampler):
    """
    Custom Sampler to take care of single-GPU training with the HelperSampler.
    """

    def __init__(self, dataset, shuffle = True, seed = 0, drop_last = True, batch_size = 32):
        """
        Args:
        - dataset (Dataset): the dataset to sample from
        - rank (int): the rank of the current process
        - seed (int): the seed for the random number generator
        - batch_size (int): the batch size
        """

        # Setting other attributes
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Computing num_samples
        self.num_samples = {biome: biome_samples for biome, biome_samples in self.dataset.length.items()}
        
        # Computing total_size
        self.total_size = {biome: self.num_samples[biome] for biome in self.num_samples}

    def __iter__(self):

        # Set the seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = {}
        for biome, biome_samples in self.dataset.length.items() :

            # Shuffle the indices
            indices[biome] = torch.randperm(biome_samples, generator = g).tolist()
            indices[biome] = indices[biome][ : self.total_size[biome]]
            assert len(indices[biome]) == self.total_size[biome]
            assert len(indices[biome]) == self.num_samples[biome]
    
        batch_sampler = HelperSampler(self.dataset, batch_size = self.batch_size, indices = indices, gen = g)
        return iter(batch_sampler)

    def __len__(self):
        """
        Returns the number of batches that each replica will see.
        """
        return sum(biome_samples // self.batch_size for biome_samples in self.num_samples.values())

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
        - epoch (int): Epoch number.
        """
        self.epoch = epoch


def biome_generator(dataset, biome, idxs, batch_size) :
    """
    This function generates the batches for a given biome. Used by HelperSampler.

    Args:
    - dataset (Dataset): the dataset to sample from
    - biome (str): the biome to sample from
    - idxs (list): the indices for this biome
    - batch_size (int): the batch size

    Returns:
    - batch_groups (list): the list of (file, tile, i) tuples for this biome
    """

    # Initialize the index and total length
    index = dataset.index[biome]
    total_length = dataset.length[biome]

    # Yield the (file, tile, i) tuples for this biome, one batch at a time
    for i in range(0, len(idxs) - batch_size + 1, batch_size) :
        ns = idxs[i : i + batch_size]
        batch_groups = find_index_for_chunk(index, ns, total_length)
        yield batch_groups


class HelperSampler(Sampler):
    """
    Define a sampler which samples from the dataset per biome.
    """

    def __init__(self, dataset, batch_size, indices, gen):
        """
        Initialize the sampler.

        Args:
        - dataset (Dataset): the dataset to sample from
        - batch_size (int): the batch size
        - indices (dict): the indices for each biome
        - gen (torch.Generator): the random number generator
        """

        self.lengths = {biome: len(indices[biome]) for biome in indices}
        self.biomes = list(indices.keys())
        self.batch_size = batch_size
        self.gen = gen

        # Construct the generators over eah biome
        self.generators = {biome: biome_generator(dataset, biome, idxs, batch_size) for biome, idxs in indices.items()}
    

    def __iter__(self):
        """
        Notes: called at the beginning of each epoch.
        """

        valid_biomes = self.biomes.copy()
        for _ in range(self.__len__()):

            # Randomly choose a biome until we find a batch
            while True:

                # Choose one biome randomly from the valid biomes
                biome = random.choice(valid_biomes)

                # And get a batch from this biome
                try:
                    yield next(self.generators[biome])
                # Or mark the biome as invalid if it is empty
                except StopIteration:
                    valid_biomes.remove(biome)
                # Move on to the next batch if successful
                break

    def __len__(self):
        return sum(length // self.batch_size for length in self.lengths.values())


############################################################################################################################
# Execute

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.latlon = True
    args.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    args.ch = True
    args.s1 = False
    args.alos = True
    args.lc = True
    args.dem = True
    args.gedi_dates = False
    args.patch_size = [25,25]
    args.norm_strat = 'pct'
    args.norm = False
    args.s2_dates = False
    args.gedi_dates = False
    args.batch_size = 32

    
    for mode in ['val', 'test', 'train'] :
        print('Processing {} data...'.format(mode))
        
        ds = GEDIDataset_biomes({'h5':'/scratch3/gsialelli/patches', 'norm': '/scratch3/gsialelli/patches', 'map': '/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split'}, mode = mode, args = args, debug = True, years = [2019])
        print(type(ds))

        sampler = PerBiomeDistributedSampler(ds, batch_size = args.batch_size)
        data_loader = DataLoader(ds, batch_sampler = sampler, num_workers = 8, pin_memory = True)

        # Iterate through the DataLoader
        print('starting to iterate...')
        t0 = time.time()
        
        for batch_samples in data_loader:
            continue
            """
            images, biomes, agbds = batch_samples
            biomes = np.unique(biomes)
            assert len(biomes) == 1, "There are multiple biomes in the batch"

            print('Biome: ', biomes[0])
            
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