"""

This script computes the statistics on the OlmoEarth train data of the AGBD-Lite dataset.

Run:
    sbatch --time 24:00:00 --mem-per-cpu 8G --cpus-per-task 4 --job-name stats --output $SCRATCH/logs/compute_OLMO_stats_%j.out --error $SCRATCH/logs/compute_OLMO_stats_%j.out --wrap "python compute_statistics.py --year 2020 --lite"

Only run on a file that passed `audit.py h5`.

env: tessera

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import h5py
from os.path import join
import pickle

# OLMO patches directory, from config.sh at the repo root (see config.py). Picks the
# cluster or local layout automatically; override with AGBD_PROFILE or AGBD_*_OLMO_H5.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import get_paths as _get_paths, is_local as _is_local
PATH_H5 = _get_paths(local = _is_local())['olmo_h5']
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from audit import dequantize
NUM_DIMS = 128

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


def process_file(path, num_patches_sim, stats, year) :
    """Stream /embeddings from an OLMO h5 in batches and update stats (nodata -> NaN, ignored)."""
    with h5py.File(path, 'r') as f :
        if int(f.attrs['year']) != year : raise ValueError(f'{path} is year {f.attrs["year"]}, expected {year}')
        if not f['done'][:].all() : raise ValueError(f'{path} has unfinished samples: run download.py --continue and audit.py h5 first')
        emb_ds = f['embeddings'] # (N, 25, 25, 128) int8
        N = emb_ds.shape[0]
        for i in range(0, N, num_patches_sim) :
            data = dequantize(emb_ds[i : i + num_patches_sim]) # (b, 25, 25, 128) float32, NaN at nodata
            stats = update_stats(stats, data)
    return stats


###################################################################################################
# Code execution

if __name__ == '__main__' :

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2020, help='Year of the embeddings (Lite only contains 2020 data).')
    parser.add_argument('--lite', action='store_true', help='Compute statistics on the Lite version of the dataset (the only one available).')
    args = parser.parse_args()

    if not args.lite :
        raise NotImplementedError('Only OLMO-Lite exists; pass --lite.')
    if args.year != 2020 :
        raise ValueError(f'--lite requires --year 2020 (Lite only contains 2020 data); got {args.year}.')

    stats = init_stats(NUM_DIMS)
    fname = 'OLMO-Lite-train.h5'
    print(f'Processing {fname}...')
    stats = process_file(join(PATH_H5, fname), 2_000, stats, args.year)

    final_stats = aggregate_stats(stats)
    with open(join(PATH_H5, f'OLMO_statistics_{args.year}_global_lite.pkl'), 'wb') as f :
        pickle.dump(final_stats, f)
