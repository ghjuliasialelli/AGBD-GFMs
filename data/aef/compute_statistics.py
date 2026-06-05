"""

This script computes the statistics on the AEF train data of the AGBD dataset.

Run:
    sbatch --time 120:00:00 --mem-per-cpu 8G --cpus-per-task 4 --job-name stats --output $SCRATCH/logs/compute_AEF_stats_%j.out --error $SCRATCH/logs/compute_AEF_stats_%j.out --wrap "python compute_statistics.py --year 2020 --regions global"

env: dwn

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import h5py
from os.path import join
import pickle

NODATAVALS = -128

continent_to_region = {'NorthAmerica': ['California', 'Cuba'], 'SouthAmerica': ['Paraguay', 'FrenchGuiana'],
    'Africa': ['UnitedRepublicofTanzania', 'Ghana'], 'Europe': ['Austria', 'Greece'],
    'SouthAsia': ['Nepal', 'ShaanxiProvince'], 'Australasia': ['NewZealand']}

ALL_REGIONS = [r for rs in continent_to_region.values() for r in rs]

###################################################################################################
# Helper functions

def init_stats(num_dims):
    stats = {
        'min' : np.full((num_dims,), np.inf), # (num_dims,)
        'max' : np.full((num_dims,), -np.inf), # (num_dims,)
        'mean' : [], # (num_batches, num_dims)
        'std' : [], # (num_batches, num_dims)
        'num_samples' : [] # (num_batches, num_dims)
    }
    return stats

def update_stats(stats, data) :

    means = np.nanmean(data, axis=(0, 1, 2))
    stds = np.nanstd(data, axis=(0, 1, 2))
    mins = np.nanmin(data, axis=(0, 1, 2))
    maxs = np.nanmax(data, axis=(0, 1, 2))
    counts = np.sum(~np.isnan(data), axis=(0, 1, 2))

    stats['min'] = np.fmin(stats['min'], mins)
    stats['max'] = np.fmax(stats['max'], maxs)
    stats['num_samples'].append(counts)
    stats['mean'].append(means)
    stats['std'].append(stds)

    return stats


def aggregate_stats(stats) :

    # Mean
    means = np.array(stats['mean']) # (num_batches, num_dims)
    num_samples = np.array(stats['num_samples']) # (num_batches, num_dims)
    final_mean = (1 / np.sum(num_samples, axis = 0)) * np.sum(means * num_samples, axis = 0) # (num_dims,)

    # STD
    stds = np.array(stats['std']) # (num_batches, num_dims)
    A = (1 / (sum(num_samples) - len(num_samples))) # (num_dims,)
    B = np.sum(((num_samples - 1) * (stds ** 2) + num_samples * (means ** 2)), axis = 0) # (num_dims,)
    C = np.sum(num_samples, axis = 0) * (final_mean ** 2) # (num_dims,)
    composite_var = A * (B - C) # (num_dims,)
    final_std = np.sqrt(composite_var) # (num_dims,)

    return {'mean' : final_mean, 'std' : final_std, 'min' : stats['min'], 'max' : stats['max']}

###################################################################################################
# Code execution

if __name__ == '__main__' :

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, nargs='+', default=None,
                        help='Year(s) to compute statistics for (e.g. --year 2020). Default: [2019, 2020] (or [2020] when --lite).')
    parser.add_argument('--regions', type=str, nargs='+', default=['global'],
                        choices=['global'] + list(continent_to_region.keys()) + ALL_REGIONS,
                        help='Regions to consider: "global" (all), a continent (e.g. Africa), or specific region(s) (e.g. Ghana).')
    parser.add_argument('--lite', action='store_true',
                        help='Compute statistics on the Lite version of the dataset (uses mapping_lite_to_og.pkl). Requires --regions global and --year 2020.')
    args = parser.parse_args()
    regions_arg = args.regions
    lite = args.lite

    # Resolve / validate years
    if lite :
        if args.year is None :
            years = [2020]
        else :
            years = args.year
            if years != [2020] :
                raise ValueError(f'--lite requires --year 2020 (Lite only contains 2020 data); got {years}.')
        if regions_arg != ['global'] :
            raise ValueError(f'--lite requires --regions global; got {regions_arg}.')
    else :
        years = args.year if args.year is not None else [2019, 2020]

    path_h5 = '/cluster/work/igp_psr/gsialelli/Data/patches/AEF'
    num_patches_sim = 10_000
    num_dims = 64

    # Resolve the regions
    if 'global' in regions_arg :
        regions = ALL_REGIONS
    else :
        regions = []
        for r in regions_arg :
            if r in continent_to_region : regions += continent_to_region[r]
            else : regions.append(r)
        regions = list(dict.fromkeys(regions))

    # List the files
    fnames = []
    for region in regions :
        if region == 'UnitedRepublicofTanzania' :
            fnames += [f'UnitedRepublicofTanzania_{year}_{i}-2.h5' for year in years for i in range(1, 3)]
        else :
            fnames += [f'{region}_{year}.h5' for year in years]

    # List the train tiles. In Lite mode we also get the per-tile patch count cap.
    if lite :
        with open('mapping_lite_to_og.pkl', 'rb') as f :
            lite_train = pickle.load(f)['train'] # list of (tile_id, num_patches_lite)
        train_tiles = {tile : n for tile, n in lite_train}
    else :
        with open('split_to_tiles.pkl', 'rb') as f:
            train_tiles = pickle.load(f)['train']

    stats = init_stats(num_dims)
    for fname in fnames :

        with h5py.File(join(path_h5, fname), 'r') as f :

            # Iterate over all the datasets in the file
            for tile in f.keys() :

                if tile not in train_tiles : continue

                print(f'Processing tile: {tile}...')

                dataset = f[tile] # (num_patches, 25, 25, num_bands)

                # In Lite mode, cap the tile to its Lite patch count
                total_len = dataset.shape[0]
                if lite :
                    total_len = min(total_len, train_tiles[tile])

                # Iterate over the bands
                for i in range(0, total_len, num_patches_sim):

                    end = min(i + num_patches_sim, total_len)
                    data = dataset[i : end, :, :, :].astype(np.float32) # (num_patches_sim, 25, 25, num_bands)

                    # Dequantize, cf. https://source.coop/tge-labs/aef
                    # de_quantized_values = ((values / 127.5) ** 2) * np.sign(values)
                    # we then add 1 and divide by 2 to have everything in [0, 1]
                    data = np.where(data == NODATAVALS, np.nan, (((data / 127.5) ** 2) * np.sign(data) + 1) / 2)

                    # Get the statistics, for each dimension individually and excluding nodata values
                    stats = update_stats(stats, data)


    # Aggregate the statistics
    final_stats = aggregate_stats(stats)

    # Save the statistics
    joined_years = '-'.join(str(y) for y in years)
    joined_regions = '-'.join(regions_arg)
    lite_suffix = '_lite' if lite else ''
    with open(join(path_h5, f'AEF_statistics_{joined_years}_{joined_regions}{lite_suffix}.pkl'), 'wb') as f:
        pickle.dump(final_stats, f)
