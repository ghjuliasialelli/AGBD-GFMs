"""

This script finds a subset of indices of the AGBD Dataset, by selecting a subset of footprints per Sentinel-2 tile.
The criterion for selection is to minimize the Wasserstein distance between the AGBD and biome distributions of the full dataset and those of the subsampled dataset.

Command-line arguments:
    --regions : list of regions to subsample
    --fraction : fraction of footprints to sample per tile (default: 0.05)
    --num_trials : number of random subsampling trials to perform (default: 10)

env: dwn

"""

###################################################################################################
# Imports

import argparse
import h5py
from os.path import join, isfile
from os import makedirs
import numpy as np
import pickle
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

###################################################################################################
# Helper functions

def _parser() :
    """
    This function parses command-line arguments for the subsampling script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', type = str, nargs = '+', required = True, help = 'List of regions to subsample')
    parser.add_argument('--fraction', type = float, default = 0.05, help = 'Fraction of footprints to sample per tile')
    parser.add_argument('--num_trials', type = int, default = 5, help = 'Number of random subsampling trials to perform')
    args = parser.parse_args()
    return args.regions, args.fraction, args.num_trials


def compare_dist(dist1, dist2, region, i, key = None, debug = True) :
    """
    This function computes the Wasserstein distance between two distributions.

    Args:
    - dist1 (dict): The first distribution as a dictionary.
    - dist2 (dict): The second distribution as a dictionary.

    Returns:
    - distance (float): The Wasserstein distance between the two distributions.
    """

    assert set(dist1.keys()) == set(dist2.keys()), "The two distributions must have the same keys."
    keys = dist1.keys()

    values1 = np.array([dist1[k] for k in keys]) / np.sum([dist1[k] for k in keys])
    values2 = np.array([dist2[k] for k in keys]) / np.sum([dist2[k] for k in keys])

    distance = wasserstein_distance(values1, values2)

    # Plot, if debug:
    if debug:
        plt.bar(np.arange(len(keys)) - 0.2, values1, width = 0.4, label = 'Reference distribution')
        plt.bar(np.arange(len(keys)) + 0.2, values2, width = 0.4, label = 'Subsampled distribution')
        if key == 'AGB': plt.yscale("log") 
        plt.legend()
        plt.title(f'Region: {region}, Iteration: {i}, Wasserstein Distance: {distance:.4f}')
        makedirs(join('vis', region), exist_ok = True)
        plt.savefig(f'vis/{region}/{key}_{i}.png', bbox_inches = 'tight')
        plt.close()
    
    return distance


def ith_subsample(i, seeds, bins, biomes, h5_fnames, region_tiles, region_agb_dist, region_biome_dist, sample_fraction = 0.05, return_indices = False) :
    """
    This function performs the i-th subsampling of the data for a given region and computes the Wasserstein distances
    between the subsampled distributions and the reference distributions.

    Args:
    - i (int): The index of the subsampling iteration.
    - seeds (list): A list of random seeds for each iteration.
    - bins (list): A list of biomass bins.
    - biomes (list): A list of biome categories.
    - region_tiles (list): A list of tiles in the region.
    - region_agb_dist (dict): The reference AGB distribution for the region.
    - region_biome_dist (dict): The reference biome distribution for the region.
    - sample_fraction (float): The fraction of data to sample from each footprint.
    - return_indices (bool): Whether to return the sampled indices instead of distances.

    Returns:
    - (agb_distance, biome_distance) (tuple): A tuple containing the Wasserstein distances for AGB and biome distributions.
    - all_sampled_indices (dict, optional): The indices of the sampled data points if return_indices is True.
    """

    # Set the i-th seed
    iteration_rng = np.random.default_rng(seeds[i])

    # Initialize placeholders
    lbs, ubs = bins[:-1], bins[1:]
    agb_cum_dist = {f'{lb}-{ub}': 0 for lb, ub in zip(lbs, ubs)}
    biome_cum_dist = {biome: 0 for biome in biomes}

    # Iterate over the footprints in the region
    if return_indices: all_sampled_indices = {}
    for fname in h5_fnames :
        with h5py.File(fname, 'r') as f:
            f_tiles = np.intersect1d(list(f.keys()), region_tiles)
            if len(f_tiles) == 0 : continue
            for tile in f_tiles :
                # Subsample the data
                num_samples = f[tile]['GEDI']['agbd'].shape[0]
                num_subsamples = int(num_samples * sample_fraction)
                if num_subsamples == 0 : continue
                sampled_indices = iteration_rng.choice(num_samples, size = num_subsamples, replace = False)
                sampled_indices.sort()
                # Load AGB and biome data
                agb_data = f[tile]['GEDI']['agbd'][sampled_indices]
                biome_data = f[tile]['LC'][sampled_indices, 12, 12, 0]
                # Update cumulative distributions
                for lb, ub in zip(lbs, ubs): agb_cum_dist[f'{lb}-{ub}'] += np.sum((agb_data >= lb) & (agb_data < ub))
                for biome in biomes: biome_cum_dist[biome] += np.sum(biome_data == biome)
                if return_indices: all_sampled_indices[tile] = sampled_indices
    
    if not return_indices:
        # Compute Wasserstein distances
        agb_distance = compare_dist(region_agb_dist, agb_cum_dist, region, i, key = 'AGB', debug = True)
        biome_distance = compare_dist(region_biome_dist, biome_cum_dist, region, i, key = 'Biome', debug = True)
        return (agb_distance, biome_distance)
    else: return all_sampled_indices


def subsample_region(results, h5_fnames, region, bins, biomes, N = 10, sample_fraction = 0.05) :
    """
    This function performs subsampling of the data for a given region and computes the Wasserstein distances
    between the subsampled distributions and the reference distributions.

    Args:
    - results (dict): The reference distributions for all regions.
    - region (str): The name of the region.
    - bins (list): A list of biomass bins.
    - biomes (list): A list of biome categories.
    - N (int): The number of subsampling iterations to perform.
    - sample_fraction (float): The fraction of data to sample from each footprint.

    Returns:
    - scores (list): A list of tuples containing the Wasserstein distances for AGB and biome distributions for each subsampling iteration.
    """

    # Get the reference distributions
    region_agb_dist, region_biome_dist = results[region]['agb'], results[region]['biome']
    og_num_footprints = sum(region_agb_dist.values())
    print(f"Region: {region}, OG number of footprints: {og_num_footprints}")
    region_tiles = tiles_per_region[region]

    # Try N different subsamples
    seeds = rng.choice(N, size = N, replace = False)
    scores = []
    for i in range(len(seeds)) :
        score = ith_subsample(i, seeds, bins, biomes, h5_fnames, region_tiles, region_agb_dist, region_biome_dist, sample_fraction)
        scores.append(score)
    
    return scores, seeds


###################################################################################################
# Code execution

if __name__ == '__main__' :

    # Set random seed
    rng = np.random.default_rng(seed=1)

    # Paths
    path_to_h5 = '/scratch3/gsialelli/patches'
    h5_fnames = [join(path_to_h5, f'data_subset-2020-v4_{i}-20.h5') for i in range(20)] 

    # Define the regions of interest, biomass bins of interest, and biomes of interest
    bins = np.arange(0, 501, 10)
    biomes = [20, 30, 40, 90, 111, 112, 114, 115, 116, 121, 122, 124, 125, 126]

    # Load the reference distributions
    if not isfile('regional_distributions.pkl') : raise FileNotFoundError('Please run exploration.ipynb first to compute the reference regional distributions.') 
    with open('regional_distributions.pkl', 'rb') as f: results = pickle.load(f)
    
    # Load the tiles per region
    if not isfile('tiles_per_region.pkl') : raise FileNotFoundError('Please run exploration.ipynb first to compute the tiles per region.') 
    with open('tiles_per_region.pkl', 'rb') as f: tiles_per_region = pickle.load(f)

    # Parse command-line arguments
    regions, fraction, num_trials = _parser()

    # For each region, perform subsampling and find the best iteration
    for region in regions :
        scores, seeds = subsample_region(results, h5_fnames, region, bins, biomes, N = num_trials, sample_fraction = fraction)
        # Find the iteration with the lowest Wasserstein distances for AGB, and if tie, for biome
        best_iteration = min(range(len(scores)), key=lambda i: (scores[i][0], scores[i][1]))
        print(f'    Best iteration: {best_iteration}')
        # Get the subsampled indices for the best iteration
        sampled_indices = ith_subsample(best_iteration, seeds, bins, biomes, h5_fnames, tiles_per_region[region], None, None, fraction, return_indices = True)
        print(f'    Number of subsampled footprints: {sum([len(v) for v in sampled_indices.values()])}')
        with open(f'indices/subsampled_indices_{region}.pkl', 'wb') as f: 
            pickle.dump(sampled_indices, f)


"""

To run for a few regions in parallel:
nohup python find_indices.py --regions California Cuba ShaanxiProvince > logs/1.txt 2>&1 &
nohup python find_indices.py --regions Paraguay UnitedRepublicofTanzania NewZealand > logs/2.txt 2>&1 &
nohup python find_indices.py --regions Ghana Austria FrenchGuiana > logs/3.txt 2>&1 &
nohup python find_indices.py --regions Greece Nepal > logs/4.txt 2>&1 &

To merge them all : 

import pickle
regions = ['California', 'Cuba', 'ShaanxiProvince', 'Paraguay', 'UnitedRepublicofTanzania', 'NewZealand', 'Ghana', 'Austria', 'FrenchGuiana', 'Greece', 'Nepal']
all_indices = {}
for region in regions :
    with open(f'indices/subsampled_indices_{region}.pkl', 'rb') as f: 
        region_indices = pickle.load(f)
    all_indices.update(region_indices)

with open(f'/scratch3/gsialelli/patches/subsampled_indices.pkl', 'wb') as f: pickle.dump(all_indices, f)

"""