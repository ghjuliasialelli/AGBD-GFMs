"""

This script computes the statistics on the train AGBD-Lite dataset.
It is an adapted version of the BiomassDatasetCreation/patches/get_stats.py script.

env: dwn

"""

###################################################################################################
# Imports

import numpy as np
from collections import defaultdict
import h5py
from os.path import join
import pickle
from copy import deepcopy
from scipy.ndimage import distance_transform_edt

NODATAVALS = {'S2_bands' : 0, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

###################################################################################################
# Helper functions

def get_stats(data, stats) :
    """
    This function computes the statistics of the data and updates the stats dictionary.

    Args:
    - data (np.array): The data to compute the statistics on.
    - stats (dict): The dictionary containing the statistics.

    Returns:
    - stats (dict): The updated dictionary containing the statistics.
    """

    # Check that there are no infinite or NaN values
    if np.any(np.isinf(data)) or np.any(np.isnan(data)):
        print('Infinite or NaN values found in the data. Skipping...')
        exit()

    # Cast everything to float32
    data = data.astype(np.float32)

    if data.size == 0: 
        return stats

    # Calculate the statistics
    mean = np.mean(data, dtype = np.float32)
    std = np.std(data, dtype = np.float32)
    num_samples = data.size
    min_val = min(np.min(data), stats['min'])
    max_val = max(np.max(data), stats['max'])
    vals, counts = np.unique(data, return_counts = True)

    # Populate the statistics
    stats['min'] = min_val
    stats['max'] = max_val

    for stat in ['mean', 'std', 'num_samples']:
        stats[stat].append(locals()[stat])

    for val, count in zip(vals, counts):
        stats['hist'][val] += count

    return stats


def init_stats() :
    """
    This function initializes the dictionary containing the statistics.

    Returns:
    - stats (dict): The dictionary containing the statistics.
    """
    
    base_stats = {'mean': [], 'std': [], 'min': np.inf, 'max': -np.inf, 'hist' : defaultdict(int), 'num_samples': []}
    return {'ALOS_bands': {b: deepcopy(base_stats) for b in ['HH', 'HV']},
        'S2_bands': {b: deepcopy(base_stats) for b in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']},
        'DEM': deepcopy(base_stats),
        'topo': {b: deepcopy(base_stats) for b in ['slope']},
        'Sentinel_metadata': {b: deepcopy(base_stats) for b in ['S2_date']},
        'GEDI': {b: deepcopy(base_stats) for b in ['agbd', 'rh98']},
        'LC': {b: deepcopy(base_stats) for b in ['lc_prob']}
    }


def composite_mean_std(means, stds, num_samples) :
    """
    This function computes the composite mean and standard deviation of the data.
    The formula can be found at https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation

    Args:
    - means (list): The list of means of the data.
    - stds (list): The list of standard deviations of the data.
    - num_samples (list): The list of number of samples of the data.

    Returns:
    - composite_mean (float): The composite mean of the data.
    - composite_std (float): The composite standard deviation of the data.
    """

    composite_mean = (1 / sum(num_samples)) * sum([mean * num_sample for mean, num_sample in zip(means, num_samples)])

    A = (1 / (sum(num_samples) - len(num_samples)))
    B = sum([(num_sample - 1) * (std ** 2) + num_sample * (mean ** 2) for mean, std, num_sample in zip(means, stds, num_samples)])
    C = sum(num_samples) * (composite_mean ** 2)

    composite_var = A * (B - C)
    composite_std = np.sqrt(composite_var)

    return composite_mean, composite_std


def get_percentiles(hist) :
    """
    This function computes the 1st and 99th percentiles of the data.

    Args:
    - hist (dict): The dictionary containing the histogram of the data.

    Returns:
    - p1 (float): The 1st percentile of the data.
    - p99 (float): The 99th percentile of the data.
    """

    # Order the histogram
    hist = dict(sorted(hist.items()))

    values, counts = np.array(list(hist.keys())), np.array(list(hist.values()))
    frequencies = np.cumsum(counts) / np.sum(counts)

    # Get the 1st percentile
    p1 = values[len(frequencies[frequencies < 0.01])]

    # Get the 99th percentile
    p99 = values[len(frequencies[frequencies < 0.99])]

    return p1, p99


def aggregate_stats(stats) :
    """
    This function aggregates the statistics over all patches.

    Args:
    - stats (dict): The dictionary containing the statistics.

    Returns:
    - final_stats (dict): The dictionary containing the aggregated statistics.
    """

    final_stats = {}

    for key in stats.keys():
    
        print(key)

        final_stats[key] = {}

        if key == 'DEM' :

            if stats[key]['num_samples'] == [] : 
                print('skipping')
                continue

            final_mean, final_std = composite_mean_std(stats[key]['mean'], stats[key]['std'], stats[key]['num_samples'])
            final_min, final_max = stats[key]['min'], stats[key]['max']
            p1, p99 = get_percentiles(stats[key]['hist'])

            final_stats[key] = {'mean': final_mean, 'std': final_std, 'min': final_min, 'max': final_max, 'p1': p1, 'p99': p99}

        else:

            for band in stats[key].keys():

                if stats[key][band]['num_samples'] == [] : 
                    print('skipping')
                    continue
                    
                final_mean, final_std = composite_mean_std(stats[key][band]['mean'], stats[key][band]['std'], stats[key][band]['num_samples'])
                final_min, final_max = stats[key][band]['min'], stats[key][band]['max']
                p1, p99 = get_percentiles(stats[key][band]['hist'])

                final_stats[key][band] = {'mean': final_mean, 'std': final_std, 'min': final_min, 'max': final_max, 'p1': p1, 'p99': p99}
            
    return final_stats



###################################################################################################
# Code execution

if __name__ == '__main__' :

    num_patches_sim = 10_000
    
    # Initialize the statistics
    stats = init_stats()

    path_h5 = join('/scratch3', 'gsialelli', 'patches')
    fname = join(path_h5, 'AGBD-Lite', 'AGBD-Lite-train.h5')
    with h5py.File(fname, 'r') as f :

        total_len = f['GEDI']['agbd'].shape[0]

        # Iterate over all the datasets in the file
        for key in f.keys():
            print(f'Processing {key}...')

            match key:

                case 'ALOS_bands' :
                    dataset = f[key] # (num_patches, 25, 25, 2)
                    band_order = dataset.attrs['order']

                    # Iterate over the bands
                    for band_idx, band in enumerate(band_order): 
                        for i in range(0, total_len, num_patches_sim):
                            data = dataset[i : i + num_patches_sim, :, :, band_idx] # (num_patches, 25, 25)
                            data = data[data != NODATAVALS[key]]
                            data = 10 * np.log10(data.astype(np.float32) ** 2) - 83.0
                            stats[key][band] = get_stats(data, stats[key][band])
                
                case 'S2_bands' :
                    dataset = f[key] # (num_patches, 25, 25, num_bands)
                    band_order = dataset.attrs['order']

                    # Iterate over the bands
                    for band_idx, band in enumerate(band_order): 
                        for i in range(0, total_len, num_patches_sim):
                            
                            data = dataset[i : i + num_patches_sim, :, :, band_idx] # (num_patches, 25, 25)

                            # Get the BOA flag
                            actual_num_patches = min(data.shape[0], num_patches_sim)
                            if 'S2_boa_offset' in f['Sentinel_metadata'].keys() : boa_offsets = f['Sentinel_metadata']['S2_boa_offset'][i : i + actual_num_patches]
                            else: boa_offsets = np.zeros(actual_num_patches)

                            # Get the surface reflectance values
                            sr_data = (data - boa_offsets[:, np.newaxis, np.newaxis] * 1000) / 10000
                            sr_data[data == NODATAVALS[key]] = NODATAVALS[key]
                            sr_data[sr_data < 0] = 0

                            # Get the statistics
                            data = sr_data[sr_data != NODATAVALS[key]]
                            stats[key][band] = get_stats(data, stats[key][band])
                
                case 'DEM':
                    for i in range(0, total_len, num_patches_sim):
                        dataset = f[key] # (num_patches, 25, 25)
                        data = dataset[i : i + num_patches_sim, :, :]
                        data = data[data != NODATAVALS[key]]
                        stats[key] = get_stats(data, stats[key])
                
                    # Calculate the statistics for the slope
                    for i in range(0, total_len, num_patches_sim):
                        dem = f['DEM'][i : i + num_patches_sim, :, :]
                        mask = (dem == NODATAVALS['DEM'])
                        if np.any(mask) : # Where the DEM is not available, we take the nearest one available
                            _, indices = distance_transform_edt(mask, return_indices = True) # Calculate the distance to the nearest non-invalid cell
                            dem = dem[tuple(indices)]
                        px, py = np.gradient(dem, 10, axis=(1, 2))
                        slope = np.sqrt(px ** 2 + py ** 2)
                        stats['topo']['slope'] = get_stats(slope, stats['topo']['slope'])
                
                # Calculate the statistics for the land cover probabilities
                case 'LC':
                    for i in range(0, total_len, num_patches_sim):
                        lc_data = f['LC'][i : i + num_patches_sim, :, :, :]
                        lc_map, lc_prob = lc_data[:, :, :, 0], lc_data[:, :, :, 1]
                        lc_prob = lc_prob[lc_map != NODATAVALS['LC']] / 100
                        stats[key]['lc_prob'] = get_stats(lc_prob, stats[key]['lc_prob'])

                case 'Sentinel_metadata':
                    for attr in f[key].keys():
                        if attr in ['S2_date']:
                            for i in range(0, total_len, num_patches_sim):
                                dataset = f[key][attr] # (num_patches, 1)
                                data = dataset[i : i + num_patches_sim]
                                stats[key][attr] = get_stats(data, stats[key][attr])
                        else: continue
                
                case 'GEDI':
                    for attr in f[key].keys():
                        if attr in ['agbd', 'rh98']:
                            for i in range(0, total_len, num_patches_sim):
                                dataset = f[key][attr] # (num_patches, 1)
                                data = dataset[i : i + num_patches_sim]
                                stats[key][attr] = get_stats(data, stats[key][attr])
                        else: continue

    final_stats = aggregate_stats(stats)

    # Cast everything to float32
    for key in final_stats.keys():
        if key == 'DEM':
            for stat in final_stats[key].keys():
                final_stats[key][stat] = np.float32(final_stats[key][stat])
        else:
            for band in final_stats[key].keys():
                for stat in final_stats[key][band].keys():
                    final_stats[key][band][stat] = np.float32(final_stats[key][band][stat])

    # Save the statistics
    with open(join(path_h5, 'AGBD-Lite', f'AGBD-Lite-statistics.pkl'), 'wb') as f:
        pickle.dump(final_stats, f)