"""

This script actually performs the subsampling of the AGBD Dataset into an AGBD-Lite version, by generating new .h5 files
for specified indices, obtained from the find_indices.py script. We also subsample the original features, retaining only
the ones specified in the features.json file.

Command-line arguments:
    --mode: set for which to generate the subsampled dataset (train/val/test)
    --keep_all: if set, do not subsample. Can only be set for the test mode.

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
import json
from itertools import product, islice
from tqdm import tqdm

# As taken from BiomassDatasetCreation/patches/create_patches.py
S2_attrs = {'bands' : {'B01': np.uint16, 'B02': np.uint16, 'B03': np.uint16, 'B04': np.uint16, 'B05': np.uint16, 'B06': np.uint16, 
                        'B07': np.uint16, 'B08': np.uint16, 'B8A': np.uint16, 'B09': np.uint16, 'B11': np.uint16, 'B12': np.uint16, 
                        'SCL': np.uint8}}
ALOS_attrs = {'HH': np.uint16, 'HV': np.uint16}

###################################################################################################
# Helper functions

def _parser() :
    """
    This function parses command-line arguments for the subsampling script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, required = True, help = 'Set for which to generate the subsampled dataset (train/val/test)')
    parser.add_argument('--keep_all', action='store_true', help = 'If set, do not subsample.')
    args = parser.parse_args()
    assert args.mode in ['train', 'val', 'test'], "Error: --mode must be one of 'train', 'val', or 'test'."
    if args.keep_all: assert args.mode == 'test', "Error: --keep_all can only be set for the test mode."
    return args.mode, args.keep_all


def get_tile_to_file(fnames) :
    """
    This function maps each S2 tile to the .h5 file(s) it is contained in.
    
    Args:
    - fnames (list): List of .h5 file paths.

    Returns:
    - tile_to_file (dict): Dictionary mapping tile names to .h5 file paths.
    """
    
    if isfile('tile_to_file.pkl') :
        with open('tile_to_file.pkl', 'rb') as f :
            tile_to_file = pickle.load(f)

    else:
        tile_to_file = {}
        for h5_fname in fnames :
            with h5py.File(h5_fname, 'r') as f :
                    tiles = list(f.keys())
                    for tile in tiles :
                            if tile not in tile_to_file :
                                    tile_to_file[tile] = [h5_fname]
                            else: tile_to_file[tile].append(h5_fname)
        
        with open('tile_to_file.pkl', 'wb') as f :
            pickle.dump(tile_to_file, f)
    
    return tile_to_file


def gen_region_from_tile() :
    """
    This function creates a mapping from S2 tiles to regions, represented as numbers,
    ordered alphabetically.

    Args:
    - None

    Returns:
    - tile_to_region (dict): Dictionary mapping tile names to region numbers.
    """

    if isfile('tile_to_region.pkl') :
        with open('tile_to_region.pkl', 'rb') as f :
            tile_to_region = pickle.load(f)

    else:
        # Get the region -> tiles mapping
        with open('tiles_per_region.pkl', 'rb') as f: tiles_per_region = pickle.load(f)
        regions = list(tiles_per_region.keys())
        # Map the regions to numbers, by sorting them alphabetically
        region_to_num = {region: i for i, region in enumerate(sorted(regions))}

        # Invert it to get tiles -> region mapping, with the region represented as a number
        tile_to_region = {}
        for region, tiles in tiles_per_region.items() :
            for tile in tiles :
                tile_to_region[tile] = region_to_num[region]
        with open('tile_to_region.pkl', 'wb') as f :
            pickle.dump(tile_to_region, f)
    
    return tile_to_region


def batched(iterable, n):
    """
    This function yields batches of size n from the given iterable.

    Args:
    - iterable: An iterable to be batched.
    - n (int): The size of each batch.

    Yields:
    - list: A batch of size n from the iterable.
    """
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def initialize_output_file(features, out_f, chunk_size) :
    """
    This function initializes the output .h5 file with the specified features.

    Args:
    - features (list): List of feature dictionaries specifying keys, dtype, and shape.
    - out_f (h5py.File): The output .h5 file handle, already opened in write mode.
    - chunk_size (int): The chunk size to use for datasets.

    Returns:
    - None
    """

    # Initialize the groups and datasets for the file
    for _dict in features :
        feature = _dict["keys"]
        _dtype = np.dtype(_dict["dtype"])
        _shape = tuple(_dict["shape"])
        if len(feature) == 2 : # we make a group + dataset
            group_name, dataset_name = feature
            if group_name not in out_f : out_f.create_group(group_name)
            out_f[group_name].create_dataset(dataset_name, 
                                            shape = (0,) + _shape, 
                                            maxshape = (None,) + _shape, 
                                            chunks = (chunk_size,) + _shape,
                                            dtype = _dtype, compression = 'gzip')
        else: # we make a dataset
            dataset_name = feature[0]
            out_f.create_dataset(dataset_name, 
                                shape = (0,) + _shape, 
                                maxshape = (None,) + _shape, 
                                chunks = (chunk_size,) + _shape,
                                dtype = _dtype, compression = 'gzip')
    
    # Add an extra attribute for the region
    out_f.create_dataset('region', shape = (0,), maxshape = (None,), chunks = (chunk_size,), dtype = np.uint8, compression = 'gzip')
    
    # Set the order attributes for S2 and ALOS bands
    out_f['S2_bands'].attrs['order'] = list(S2_attrs['bands'].keys())[:-1]  # exclude 'SCL' band
    out_f['ALOS_bands'].attrs['order'] = list(ALOS_attrs.keys())


###################################################################################################
# Code execution

if __name__ == '__main__' :


    ############################################################################################################################
    # Initialize variables

    np.random.seed(42) # Set the seed for reproducibility
    path_patches = '/scratch3/gsialelli/patches'  # Path to the patches directory
    mode, keep_all = _parser() # Parse command-line arguments
    chunk_size = 32 # Chunk size for writing datasets
    WRITE_THRESHOLD = 5_000 # Number of samples to write at once
    if keep_all:
        years = [2019, 2020]
        print("Warning: --keep_all is set, so no subsampling will be performed. The output file will contain all the samples from the test set.")
    else: years = [2020]

    # Load the indices to keep for the specified mode
    with open(join(path_patches, 'subsampled_indices.pkl'), 'rb') as f: indices = pickle.load(f)

    # Get the features to keep in the subsampled dataset
    with open("features.json", "r") as f: features = json.load(f)

    # Find the S2 tiles that are part of the specified mode
    with open(join(path_patches, 'biomes_splits_to_name.pkl'), 'rb') as f: tiles_in_mode = pickle.load(f)[mode]
    
    # Get the list of .h5 files to read from
    h5_fnames = [join(path_patches, f'data_subset-{year}-v4_{i}-20.h5') for i in range(20) for year in years]
    output_dir = join(path_patches, 'AGBD-Lite')

    # Mapping from tile to .h5 file
    tile_to_file = get_tile_to_file(h5_fnames)

    # Mapping from tile to region number
    tile_to_region = gen_region_from_tile()

    # Open the .h5 files to read from
    in_handles = {fname: h5py.File(fname, 'r') for fname in h5_fnames}


    ############################################################################################################################
    # Subsample the test data
    if mode == 'test' :

        makedirs(output_dir, exist_ok = True)
        if keep_all: output_fname = f'AGBD-{mode}-v2.h5' # TODO remove v2
        else: output_fname = f'AGBD-Lite-{mode}.h5'
        with h5py.File(join(output_dir, output_fname), 'w') as out_f :

            # Initialize the groups and datasets for the file
            initialize_output_file(features, out_f, chunk_size)

            # Initialize buffers for each feature
            buffers = {tuple(f["keys"]): [] for f in features}
            buffers[('region',)] = []
            rows_in_buffer = 0

            # Iterate over all the indices and write them to the output file
            if keep_all : indices = {tile: None for tile in tiles_in_mode}
            for tile, idxs in tqdm(indices.items()) :
                if (tile not in tiles_in_mode) or (tile not in tile_to_file): continue
                tile_files = tile_to_file[tile]

                if not keep_all : # only keep the files that are from 2020
                    tile_files = [f for f in tile_files if '-2020-' in f]
                    if len(tile_files) == 0 : continue
                
                for tile_file in tile_files :

                    in_f = in_handles[tile_file]

                    if keep_all: idxs = range(in_f[tile]['GEDI']['agbd'].shape[0])
                    for _dict in features :
                        keys = tuple(_dict["keys"])
                        if len(keys) == 2 : # we have a group + dataset
                            group_name, dataset_name = keys
                            data_to_add = in_f[tile][group_name][dataset_name][idxs, ...]
                        else: # we have just a dataset
                            dataset_name = keys[0]
                            data_to_add = in_f[tile][dataset_name][idxs, ...]
                        buffers[keys].append(data_to_add)

                    # Also add the region information
                    tile_region = np.full(len(idxs), tile_to_region[tile], dtype=np.uint8)
                    buffers[('region',)].append(tile_region)

                    # Check if we need to write the buffered data
                    rows_in_buffer += len(idxs)
                    if rows_in_buffer >= WRITE_THRESHOLD:
                        # Write buffered data to the output file
                        for keys, data_list in buffers.items():
                            data_to_add = np.concatenate(data_list, axis=0)
                            out_dataset = out_f[keys[0]][keys[1]] if len(keys) == 2 else out_f[keys[0]]
                            current_size = out_dataset.shape[0]
                            new_size = current_size + data_to_add.shape[0]
                            out_dataset.resize((new_size,) + out_dataset.shape[1:])
                            out_dataset[current_size:new_size, ...] = data_to_add
                            buffers[keys] = []
                        rows_in_buffer = 0
            
            # Write any remaining buffered data to the output file
            if rows_in_buffer > 0:
                for keys, data_list in buffers.items():
                    data_to_add = np.concatenate(data_list, axis=0)
                    out_dataset = out_f[keys[0]][keys[1]] if len(keys) == 2 else out_f[keys[0]]
                    current_size = out_dataset.shape[0]
                    new_size = current_size + data_to_add.shape[0]
                    out_dataset.resize((new_size,) + out_dataset.shape[1:])
                    out_dataset[current_size:new_size, ...] = data_to_add
                    buffers[keys] = []

    
    ############################################################################################################################
    # Subsample the train or val data
    else:

        # Generate indices in a shuffled manner, to be able to write with a chunk size ~ batch size
        tuple_indices = []
        for tile, idxs in indices.items() :
            if tile not in tiles_in_mode : continue
            tile_file = tile_to_file[tile]
            tuple_indices.extend(product([tile_file], [tile], idxs))
        np.random.shuffle(tuple_indices)

        # Now write the subsampled .h5 file
        makedirs(output_dir, exist_ok = True)
        with h5py.File(join(output_dir, f'AGBD-Lite-{mode}.h5'), 'w') as out_f :

            # Initialize the groups and datasets for the file
            initialize_output_file(features, out_f, chunk_size)

            # Iterate over all the indices and write them to the output file
            for batch in tqdm(batched(tuple_indices, chunk_size), total = len(tuple_indices) // chunk_size + 1) :
                for _dict in features :
                    feature = _dict["keys"]
                    if len(feature) == 2 : # we have a group + dataset
                        group_name, dataset_name = feature
                        data_to_add = []
                        for tile_file, tile, idx in batch :
                            data_to_add.append(in_handles[tile_file][tile][group_name][dataset_name][idx : idx + 1])
                        out_dataset = out_f[group_name][dataset_name]
                    else: # we have just a dataset
                        dataset_name = feature[0]
                        data_to_add = []
                        for tile_file, tile, idx in batch :
                            data_to_add.append(in_handles[tile_file][tile][dataset_name][idx : idx + 1])
                        out_dataset = out_f[dataset_name]
                
                    # Append the data to the output dataset
                    data_to_add = np.concatenate(data_to_add, axis = 0)
                    current_size = out_dataset.shape[0]
                    new_size = current_size + data_to_add.shape[0]
                    out_dataset.resize((new_size,) + out_dataset.shape[1:])
                    out_dataset[current_size : new_size, ...] = data_to_add
                
                # Also add the region information
                region_data_to_add = []
                for tile_file, tile, idx in batch :
                    region_data_to_add.append(tile_to_region[tile])
                region_data_to_add = np.array(region_data_to_add, dtype = np.uint8)
                out_dataset = out_f['region']
                current_size = out_dataset.shape[0]
                new_size = current_size + region_data_to_add.shape[0]
                out_dataset.resize((new_size,) + out_dataset.shape[1:])
                out_dataset[current_size : new_size, ...] = region_data_to_add

    ############################################################################################################################
    # Cleanup

    # Close the input .h5 files
    for handle in in_handles.values() : handle.close()