"""

This script computes the statistics on the TESSERA train data of the AGBD dataset.

Run:
    sbatch --time 120:00:00 --mem-per-cpu 8G --cpus-per-task 4 --job-name stats --output $SCRATCH/logs/compute_TESSERA_stats_%j.out --error $SCRATCH/logs/compute_TESSERA_stats_%j.out --wrap "python compute_statistics.py --year 2020 --regions global --lite"

env: tessera

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import h5py
from os.path import join
import pickle

continent_to_region = {'NorthAmerica': ['California', 'Cuba'], 'SouthAmerica': ['Paraguay', 'FrenchGuiana'],
    'Africa': ['UnitedRepublicofTanzania', 'Ghana'], 'Europe': ['Austria', 'Greece'],
    'SouthAsia': ['Nepal', 'ShaanxiProvince'], 'Australasia': ['NewZealand']}

ALL_REGIONS = [r for rs in continent_to_region.values() for r in rs]

# TESSERA patches directory, from config.sh at the repo root (see config.py). Picks the
# cluster or local layout automatically; override with AGBD_PROFILE or AGBD_*_TESSERA_H5.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import get_paths as _get_paths, is_local as _is_local
PATH_H5 = _get_paths(local = _is_local())['tessera_h5']
NUM_DIMS = 128
PATCH = 25

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


def dequantize(emb, scl) :
    """Dequantise TESSERA embeddings. `emb` int8 (..., 128), `scl` float32 (...,).

    Missing pixels/samples are flagged by NaN in `scl` (emb is zero there);
    multiplying by NaN propagates, so nanmean/nanstd ignore them correctly.
    """
    return emb.astype(np.float32) * scl[..., None]


def process_file(path, num_patches_sim, stats, sample_filter=None) :
    """Stream /embeddings and /scales from a TESSERA h5 in batches and update stats.

    `sample_filter` is an optional boolean mask over the N samples; if given,
    only samples where the mask is True are included.
    """
    with h5py.File(path, 'r') as f :
        emb_ds = f['embeddings'] # (N, 25, 25, 128) int8
        scl_ds = f['scales']     # (N, 25, 25) float32

        N = emb_ds.shape[0]
        if sample_filter is not None and sample_filter.shape[0] != N :
            raise ValueError(f'sample_filter has length {sample_filter.shape[0]} but file has {N} samples')

        for i in range(0, N, num_patches_sim) :

            end = min(i + num_patches_sim, N)
            emb = emb_ds[i : end] # (b, 25, 25, 128)
            scl = scl_ds[i : end] # (b, 25, 25)

            if sample_filter is not None :
                keep = sample_filter[i : end]
                if not keep.any() : continue
                emb = emb[keep]
                scl = scl[keep]

            data = dequantize(emb, scl) # (b, 25, 25, 128) float32, NaN where scl is NaN

            stats = update_stats(stats, data)

    return stats


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
                        help='Compute statistics on the Lite version of the dataset. Requires --regions global and --year 2020.')
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

    num_patches_sim = 2_000

    # Resolve the regions
    if 'global' in regions_arg :
        regions = ALL_REGIONS
    else :
        regions = []
        for r in regions_arg :
            if r in continent_to_region : regions += continent_to_region[r]
            else : regions.append(r)
        regions = list(dict.fromkeys(regions))

    stats = init_stats(NUM_DIMS)

    if lite :
        # Lite is a single file per split, already filtered to train samples.
        fname = 'TESSERA-Lite-train.h5'
        print(f'Processing {fname}...')
        stats = process_file(join(PATH_H5, fname), num_patches_sim, stats)

    else :
        # Non-Lite layout mirrors AEF: one file per (region, year), with
        # train-tile filtering via split_to_tiles.pkl.
        raise NotImplementedError(
            'Non-Lite TESSERA statistics are not yet supported: only the Lite '
            'dataset is available on disk. When the full dataset is produced, '
            'wire it up here following the AEF convention (per-(region, year) '
            'h5 files filtered by split_to_tiles.pkl).'
        )

    # Aggregate the statistics
    final_stats = aggregate_stats(stats)

    # Save the statistics
    joined_years = '-'.join(str(y) for y in years)
    joined_regions = '-'.join(regions_arg)
    lite_suffix = '_lite' if lite else ''
    with open(join(PATH_H5, f'TESSERA_statistics_{joined_years}_{joined_regions}{lite_suffix}.pkl'), 'wb') as f :
        pickle.dump(final_stats, f)
