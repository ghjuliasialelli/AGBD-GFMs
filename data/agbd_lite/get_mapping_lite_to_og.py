"""

This script creates a mapping to retrieve, from the index of an element in the AGBD-Lite dataset,
the corresponding element in the original AGBD dataset.

"""

###################################################################################################
# Imports

import h5py
import pickle
from os.path import join
from itertools import product
import numpy as np
from tqdm import tqdm

###################################################################################################
# Helper functions


def get_tile_to_region_mapping(fnames, path_h5) :
    """
    This function creates a mapping from Sentinel-2 tile name to the region it belongs to.

    Args:
    - fnames (list): list of AEF file names
    - path_h5 (str): the path to the AEF .h5 files

    Returns:
    - mapping (dict): dictionary mapping the tile names to the years and the years to the AEF files
    """

    mapping = {}
    for fname in fnames :
        year = int(fname.rstrip('.h5').split('_')[1])
        if year != 2020 : continue
        region = fname.rstrip('.h5').split('_')[0]
        with h5py.File(join(path_h5, fname), 'r') as f:
            for tile in f.keys() :
                if tile not in mapping : mapping[tile] = {}
                mapping[tile] = region
    return mapping

###################################################################################################
# Execution

if __name__ == "__main__":

    # Paths come from config.sh at the repo root (see config.py). REPO is used for the
    # metadata that ships with this repository rather than living in the data dirs.
    import os, sys
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, REPO)
    from config import get_paths
    _paths = get_paths(local = True)

    with open(join(_paths['h5'], 'subsampled_indices.pkl'), 'rb') as f: indices = pickle.load(f)

    mapping = {}
    for mode in ['train', 'val', 'test'] :
        print(f'Processing mode: {mode}...')

        with open(join(REPO, 'data', 'aef', 'split_to_tiles.pkl'), 'rb') as f: tiles_in_mode = pickle.load(f)[mode]

        # Generate indices in a shuffled manner, to be able to write with a chunk size ~ batch size
        tuple_indices = []
        for tile, idxs in tqdm(indices.items()):
            if tile not in tiles_in_mode : continue
            tuple_indices.extend(product([tile], idxs))
        if mode != 'test' :
            np.random.seed(42)
            np.random.shuffle(tuple_indices)
        mapping[mode] = tuple_indices

    # Save the mapping
    with open('mapping_lite_to_og.pkl', 'wb') as f: pickle.dump(mapping, f)


