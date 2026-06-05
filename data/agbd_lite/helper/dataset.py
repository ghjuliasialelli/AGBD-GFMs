"""

This script defines the dataset class for the AGBD-Lite dataset.

"""

############################################################################################################################
# Imports

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join
import pickle
from os.path import join
import pandas as pd
from helper_functions import *

############################################################################################################################
# Helper functions

def initialize_index_lite(fname, chunk_size, path_h5) :
    """
    This function returns the total number of chunks in the AGBD-Lite dataset.

    Args:
    - fname (str): the name of the file
    - chunk_size (int): the size of the chunks
    - path_h5 (str): the path to the h5 files

    Returns:
    - gedi_length (int): the total number of GEDI samples in the dataset
    - total_length (int): the total number of chunks in the dataset
    """
    with h5py.File(join(path_h5, fname), 'r') as f:
        gedi_length = len(f['GEDI']['agbd'])
        total_length = (gedi_length // chunk_size) + (1 if (gedi_length % chunk_size != 0) else 0)

    return gedi_length, total_length

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
        if nodata_value is not None : data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct' :
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None : data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else : data = (data - p1) / (p99 - p1)
        if clip : data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None : data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else : data = (data - min_val) / (max_val - min_val)
    
    else: raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data


class GEDIDataset(Dataset):

    def __init__(self, mode, args):

        # Get the parameters
        self.mode = mode
        self.h5_path, self.norm_path, self.embed_path = args.paths.h5, args.paths.norm, args.paths.embeddings
        self.eval_big = args.eval_big
        self.target = args.target

        # Get the file names
        self.lite_chunk_size = args.lite_chunk_size
        if self.eval_big and self.mode == 'test' : self.fname = 'AGBD-test.h5'
        else: self.fname = f'AGBD-Lite-{self.mode}.h5'
        self.gedi_length, self.length = initialize_index_lite(self.fname, self.lite_chunk_size, self.h5_path)

        # Define the data to use
        self.latlon = args.latlon
        self.bands = args.bands
        self.alos = args.alos
        self.lc = args.lc
        self.dem = args.dem
        self.topo = args.topo
        self.aspect = args.aspect
        self.slope = args.slope
        self.s2_dates = args.s2_dates
        self.s2_day = args.s2_day
        self.s2_doy = args.s2_doy
        self.patch_size = args.patch_size

        # Define the learning procedure
        self.norm_strat = args.norm_strat

        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'"

        # Load the normalization values
        with open(self.norm_path, mode = 'rb') as f: self.norm_values = pickle.load(f)

        # Open the file handles
        self.f_handle = h5py.File(join(self.h5_path, self.fname), 'r')

        # Define the window size
        assert self.patch_size[0] == self.patch_size[1], "The patch size must be square"
        self.center_x, self.center_y = 12, 12 # because the patch size is 25x25 in the .h5 files
        self.window_size = self.patch_size[0] // 2
        
        # Get the cat2vec LC embeddings
        embeddings = pd.read_csv(self.embed_path)
        embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
        self.embeddings = embeddings

    def __len__(self):
        return int(self.length)
    
    def __getitem__(self, n):
            
        # Find the file, tile, and row index corresponding to this chunk
        idx_start = n * self.lite_chunk_size
        idx_end = min(idx_start + self.lite_chunk_size, self.gedi_length)   
        f = self.f_handle

        data = []


        # Sentinel-2 bands ------------------------------------------------------------------------
        if self.bands != [] :

            # Set the order and indices for the Sentinel-2 bands
            if not hasattr(self, 's2_order') : self.s2_order = list(f['S2_bands'].attrs['order'])
            if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.bands]
            
            # Get the bands
            s2_bands = f['S2_bands'][idx_start : idx_end, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, self.s2_indices].astype(np.float32)
            
            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f['Sentinel_metadata'].keys() : 
                s2_boa_offset = f['Sentinel_metadata']['S2_boa_offset'][idx_start : idx_end].astype(np.float32)
            else: s2_boa_offset = np.full((s2_bands.shape[0],), 0, dtype=np.float32)
            s2_boa_offset = s2_boa_offset[:, np.newaxis, np.newaxis, np.newaxis]

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands

            # Normalize the bands
            s2_norm_values = {key: np.array([self.norm_values['S2_bands'][band][key] for band in self.bands]) for key in self.norm_values['S2_bands']['B02'].keys()}
            s2_bands = normalize_data(s2_bands, s2_norm_values, self.norm_strat, NODATAVALS['S2_bands'])

            data.extend([s2_bands])
            
            # Get the encoded Sentinel-2 date
            if self.s2_dates : 
                s2_num_days = f['Sentinel_metadata']['S2_date'][idx_start : idx_end]
                s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, self.patch_size)
                s2_num_days = np.full((len(s2_num_days), *self.patch_size, 1), s2_num_days[:, np.newaxis, np.newaxis, np.newaxis])
                s2_num_days = normalize_data(s2_num_days, self.norm_values['Sentinel_metadata']['S2_date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
                if self.s2_day: data.extend([s2_num_days])
                if self.s2_doy: data.extend([s2_doy_cos, s2_doy_sin])
                            

        # Latitude and longitude data -------------------------------------------------------------
        if self.latlon :
            lat_offset, lat_decimal = f['GEDI']['lat_offset'][idx_start : idx_end], f['GEDI']['lat_decimal'][idx_start : idx_end]
            lon_offset, lon_decimal = f['GEDI']['lon_offset'][idx_start : idx_end], f['GEDI']['lon_decimal'][idx_start : idx_end]
            lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
            lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
            lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat, lon, self.patch_size)
            data.extend([lat_cos, lat_sin, lon_cos, lon_sin])
        

        # ALOS bands ------------------------------------------------------------------------------
        if self.alos:

            # Set the order for the ALOS bands
            if not hasattr(self, 'alos_order') : self.alos_order = f['ALOS_bands'].attrs['order']

            # Get the bands
            alos_bands = f['ALOS_bands'][idx_start : idx_end, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :].astype(np.float32)

            # Get the gamma naught values
            alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)

            # Normalize the bands
            alos_norm_values = {key: np.array([self.norm_values['ALOS_bands'][band][key] for band in self.alos_order]) for key in self.norm_values['ALOS_bands']['HH'].keys()}
            alos_bands = normalize_data(alos_bands, alos_norm_values, self.norm_strat, -9999.0)

            data.extend([alos_bands])
        

        # LC data ---------------------------------------------------------------------------------
        lc_data = f['LC'][idx_start : idx_end, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1, :]
        lc_map, lc_prob = lc_data[:, :, :, 0], lc_data[:, :, :, 1]
        biome = lc_data[:, self.patch_size[0] // 2, self.patch_size[1] // 2, 0] # get the biome of the central pixel

        # For the LC input feature
        if self.lc :

            # Scale the class probabilities to [0,1]
            lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)
            lc_prob = lc_prob[..., np.newaxis]

            # Get the cat2vec embedding of the biome
            lc = np.vectorize(lambda x: self.embeddings.get(x, self.embeddings.get(0)), signature = '()->(n)')(lc_map).astype(np.float32)
            data.extend([lc, lc_prob])

        # DEM data --------------------------------------------------------------------------------
        if self.dem:
            dem = f['DEM'][idx_start : idx_end, self.center_x - self.window_size : self.center_x + self.window_size + 1, self.center_y - self.window_size : self.center_y + self.window_size + 1]
            
            if self.topo :
                # Get the slope and aspect
                slope, aspect_cos, aspect_sin = get_topology(dem)
                if self.slope : data.extend([slope[..., np.newaxis]])
                if self.aspect: data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])

            dem = normalize_data(dem, self.norm_values['DEM'], self.norm_strat, NODATAVALS['DEM'])
            data.extend([dem[..., np.newaxis]])        

        # Concatenate the data together -----------------------------------------------------------
        data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 1)).to(torch.float)

        # Get the target data ---------------------------------------------------------------------
        if self.target in ['agbd', 'rh98'] : target = torch.from_numpy(np.array(f['GEDI'][self.target][idx_start : idx_end], dtype = np.float32)).to(torch.float)
        elif self.target == 'biome' : target, _ = torch.mode(torch.from_numpy(np.array(lc_map)).long().flatten(start_dim=1), dim = 1)
        
        # Get the GEDI region (0=Water, 1=Europe, 2=North Asia, 3=Australasia, 4=Africa, 5=South Asia, 6=South America, 7=North America)
        region = f['GEDI']['region_cla'][idx_start : idx_end]

        # Return the data -------------------------------------------------------------------------
        return data, target, biome, region
