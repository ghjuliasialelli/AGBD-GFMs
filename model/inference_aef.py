"""

This script performs inference for a model trained on AEF embeddings.

The script can be executed by running: bash inference/inference_aef.sh.

"""

#######################################################################################################################
# Imports

import time
from os.path import join
import os, pickle, argparse, json, hashlib, subprocess

# config.py lives at the repo root, but this package is run with model/ as the working
# directory, so the root has to be put on the path explicitly before importing it.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WANDB_ENTITY, get_paths
import torch
import numpy as np
import rasterio as rs
from torch import set_float32_matmul_precision
from models import Net
from wrapper import Model
from torch import set_float32_matmul_precision
import wandb
from dataset import normalize_data, encode_lat_lon
from pyproj import Transformer
import warnings
from parser import str2bool
from datetime import timedelta
import pandas as pd
from inference_residuals import init_args_dataset
from inference_ds import InferenceDataset_v3
from torch.utils.data import DataLoader
from pathlib import Path

# Silencing specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#######################################################################################################################
# Helper functions 

def inf_parser():
    """ 
    Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True, help = 'Path to the dataset')
    parser.add_argument('--year', type = int, required = True, help = 'Year to do inference on.')
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--entity', type = str, default = WANDB_ENTITY, help = 'wandb entity for the model.')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--method", required = True, type = str, help = 'Method used for the composites.')
    parser.add_argument("--batch_size", type = int, default = 2, help = 'Batch size for the dataloader.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--pred_crop", nargs = 4, type = int, default = [0, 0, 0, 0], help = 'Pixels to crop off the predictions (off_ht, off_wl, off_hb, off_wr).')
    parser.add_argument("--masking", type = str2bool, default = 'false', help = 'Whether to mask the input.')
    parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data type to save the predictions.')
    parser.add_argument("--mode", type = str2bool, default = 'false', help = 'Whether to use mode for biome embedding.')
    parser.add_argument("--std", type = str2bool, default = 'true', help = 'Whether to compute and save the STDs in case of ensembling.')
    parser.add_argument("--factor", type = float, default = 5, help = 'Factor for the Gaussian weights.')
    parser.add_argument("--skip_existing", action = 'store_true', help = 'Skip plots whose prediction already exists, i.e. resume an interrupted run. Off by default, so a plain run always recomputes everything; only pass this when you know the existing predictions were written by the same model and the same inputs.')
    parser.add_argument("--plots", type = str, nargs = '+', default = None, help = 'Only run on these AGBRef plot ids (default: all of them).')
    parser.add_argument("--members", type = int, nargs = '+', default = None, help = 'FiLM ensemble only: indices of the members to run, e.g. 0 for member 0 alone (default: all n_members).')
    args = parser.parse_args()

    return args, args.year, args.dataset_path, args.models, args.arch, args.saving_dir, args.tile_name, args.method, args.patch_size, args.pred_crop, args.masking, args.entity, args.dtype, args.mode, args.std, args.batch_size, args.factor


def encode_pixel_coords(transform, crs, height, width):
    """
    Encodes the latitude and longitude of every pixel centre of a raster, the same way the training
    Dataset does (dataset.encode_lat_lon), so that a model trained with `latlon` sees the same features.

    Args:
    - transform: affine.Affine, the transform of the raster
    - crs: the CRS of the raster
    - height, width: int, the shape of the raster

    Returns:
    - coords: np.ndarray of shape (height, width, 4), (lat_cos, lat_sin, lon_cos, lon_sin) in [0,1]
    """

    # Pixel centres, from the raster's own transform: this holds for any orientation (south-up included)
    rows, cols = np.mgrid[0:height, 0:width]
    xs, ys = transform * (cols + 0.5, rows + 0.5)

    # always_xy: otherwise EPSG:4326's authority axis order makes transform() return (lat, lon)
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy = True).transform(xs, ys)
    assert np.all(np.isfinite(lat)) and np.all(np.abs(lat) <= 90) and np.all(np.abs(lon) <= 180), 'Invalid pixel coordinates.'

    return np.stack(encode_lat_lon(lat, lon), axis = -1).astype(np.float32)


def load_input(paths, file_name, norm_values, masking = False, latlon = False):
    """
    Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
    and normalize the input.

    Args:
    - paths: dict, paths to the dataset
    - file_name: str, the name of the file containing the AEF embeddings
    - norm_values: dict, normalization values for the bands
    - masking: bool, whether to apply masking to the input data
    - latlon: bool, whether to prepend the encoded pixel coordinates (for models trained with `latlon`)

    Returns:
    - data: list, the input data for the model
    - mask: np.ndarray, the mask for the input data
    - meta: dict, metadata for the input data
    """
    
    start_time = time.time()
    print('Loading input...')

    # Load the AEF embeddings ------------------------------------------------------------------------------------
    with rs.open(join(paths['aef'], f'{file_name}.tiff')) as src:
        # Read the FULL mosaic. A previous hardcoded Window(0, 0, 1024, 1024) silently dropped
        # everything beyond the top-left 1024x1024 px; a 0.1 deg plot cell is ~1113 px tall at 10 m,
        # so every plot lost its bottom strip (and low-latitude plots a right strip too).
        data = np.transpose(src.read(), axes = (1,2,0)).astype(np.float32)
        meta = src.meta
        meta.update({'height': data.shape[0], 'width': data.shape[1]})

    # Get the mask -----------------------------------------------------------------------------------------------
    if masking: mask = (data[:,:,0] == -128)
    else: mask = None

    # Normalize the AEF embeddings --------------------------------------------------------------------------------
    # data = np.where(data == -128, 0, (((data / 127.5) ** 2) * np.sign(data) + 1) / 2) # TODO put back once model with dataset.py corrected
    data = (((data / 127.5) ** 2) * np.sign(data) + 1) / 2
    data = normalize_data(data, norm_values, 'mean_std', 0, False).astype(np.float32)

    # Prepend the encoded coordinates: in the training Dataset, the lat/lon channels come before the AEF ones
    if latlon:
        coords = encode_pixel_coords(meta['transform'], meta['crs'], data.shape[0], data.shape[1])
        data = np.concatenate([coords, data], axis = -1)

    print('done!')
    end_time = time.time()
    print(f'Loading input took {end_time - start_time} seconds.')

    return data, mask, meta


def predict_patch(model, patch, device, biome_emb, film = False):
    """
    Predict patch for AGBD.

    Args:
    - model: (torch.nn.Module) the model to use for prediction
    - patch: (np.ndarray) the patch to predict
    - device: (torch.device) the device on which to perform inference
    - biome_emb: (torch.Tensor) the biome embedding to use for prediction, if applicable
    - film: (bool) whether to use film mode for prediction
    Returns:
    - preds: (np.ndarray) the predicted AGBD patch
    """

    # Transform the input patch for prediction
    if len(patch.shape) == 3: # (height, width, features)
        patch = torch.unsqueeze(torch.permute(patch, [2,0,1]), 0).to(device) # to (1, features, height, width)
        if film : 
            biome_emb = torch.tensor(np.expand_dims(biome_emb, axis = 0)).to(device)
            preds = model.model((patch, biome_emb)).cpu().detach().numpy()[0, 0, :, :]
        else: preds = model.model(patch).cpu().detach().numpy()[0, 0, :, :]
    elif len(patch.shape) == 4: # (batch, height, width, features)
        patch = torch.permute(patch, [0, 3, 1, 2]).to(device) # to (batch, features, height, width)
        if film : 
            biome_emb = biome_emb.to(device)
            preds = model.model((patch, biome_emb)).cpu().detach().numpy()[:, 0, :, :]
        else: preds = model.model(patch).cpu().detach().numpy()[:, 0, :, :]
    else: raise ValueError('The patch should have either 3 or 4 dimensions.')

    return preds


def efficient_predict_tile_v3(dataloader, models, device, pred_height, pred_width, film = False, film_ensemble = False, n_members = None, members = None):
    """
    This function predicts the AGBD for a Sentinel-2 tile, using a list of models, and a dataloader.
    This approach takes the Gaussian weighted average of overlapping patches, while padding the borders
    of the tile with symmetric padding to avoid edge effects.
    
    Args:
    - dataloader: (torch.utils.data.DataLoader) the dataloader to use for prediction
    - models: (list) the models to use for prediction
    - device: (torch.device) the device on which to perform inference
    - pred_height: (int) the height of the predicted AGBD
    - pred_width: (int) the width of the predicted AGBD
    - members: (list) FiLM ensemble only, the member index of each entry of `models` (default: 0..len(models)-1)
    
    Returns:
    - predictions: (np.ndarray) the predicted AGBD for the Sentinel-2 tile
    """
    
    print('Starting prediction...')
    if film_ensemble and members is None: members = list(range(len(models)))
    
    # Placeholder for the predictions
    summed_predictions = np.full(shape = (len(models), pred_height, pred_width), fill_value = np.nan)
    sum_weights = np.full(shape = (len(models), pred_height, pred_width), fill_value = 0.0)
    
    # Iterate over the batches
    for batch in dataloader :
        
        # Unpack the batch
        patch, biome_emb, pred_indices, patch_weights, crop_indices = batch
        x_indices, y_indices = pred_indices # indices to find the position of the patch in summed_predictions and sum_weights
        v1s, v2s, h1s, h2s = crop_indices # indices to crop the prediction to remove the padded data
        
        # Iterate over the models
        for model_dim, model in enumerate(models) :

            if film_ensemble:
                batch_size = patch.shape[0]
                member_biome_emb = torch.zeros(batch_size, n_members)
                # One-hot on the member's own index, so a member run alone sees the same embedding as in the full ensemble
                member_biome_emb[:, members[model_dim]] = 1.0
                preds = predict_patch(model, patch, device, member_biome_emb, film = True)
            else:
                preds = predict_patch(model, patch, device, biome_emb, film)
            cropped_preds = preds[:, v1s : v2s, h1s : h2s] # crop the predictions to remove the padded data
            
            # Iterate over the predictions
            for i in range(len(preds)) :
                
                # Indices to find the position of the patch in summed_predictions and sum_weights
                indices = (x_indices[i].numpy(), y_indices[i].numpy())

                # Get the weighted prediction for the patch
                patch_weight = patch_weights[i].numpy()
                weighted_pred = cropped_preds * patch_weight

                # Update summed_predictions, taking care of NaN values
                pred_patch = summed_predictions[(model_dim,) + indices]
                summed_predictions[(model_dim,) + indices] = np.where(np.isnan(pred_patch), weighted_pred, pred_patch + weighted_pred)

                # Update sum_weights
                sum_weights[(model_dim,) + indices] += patch_weight
    
    # Reduce the predictions by the weights
    if np.any(sum_weights == 0): print("Warning: There are weights equal to 0. This may lead to NaN values in the predictions.")
    predictions = np.where(sum_weights > 0, summed_predictions / sum_weights, np.nan)
    print('done!')
    
    return predictions

def get_mapping(api, arch) :
    """
    This function constructs two dictionaries, one mapping the wandb name to the run's wandb identifier, and the other
    mapping the wandb name to the checkpoint path. This is done iteratively for all runs in the specified architecture.

    Args:
    - api (wandb.Api): the wandb API
    - arch (str): the architecture of the model
    """

    runs = api.runs(f"{WANDB_ENTITY}/{arch}")
    run_mapping, run_ckpt = {}, {}
    
    for run in runs:
        try:
            run_mapping[run.name] = run.path[-1]
            run_ckpt[run.name] = run.config['model_path']
        except: continue
    
    return run_mapping, run_ckpt

def load_embeddings(cfg, dataset_path, lite):
    if (cfg.get('lc', False) and cfg.get('ft_cat2vec', False)) or (cfg.get('film', False) and cfg.get('emb_cat2vec', False)) :
        embeddings = pd.read_csv(join(dataset_path['embeddings'], f"embeddings_train{'_lite' if lite else ''}.csv"))
        embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
    else: embeddings = None

#######################################################################################################################
# Inference class definition

class Inference:
    """ 
    An `Inference` object loads a PyTorch model and performs AGBD inference at the Sentinel-2 tile level.
    """

    def __init__(self, arch, model_name, paths, tile_name, args, device):
        """
        Initialization method.

        Args:
        - arch (str) : the architecture of the model
        - model_name (str) : the name of the model
        - paths (dict) : the paths to the dataset
        - tile_name (str) : the name of the Sentinel-2 tile
        - args (argparse.Namespace) : the command-line arguments
        - device (torch.device) : the device on which to perform

        Returns:
        - None
        """

        self.arch = arch
        self.model_name = model_name
        self.paths = paths
        self.tile_name = tile_name
        self.args = args     
        self.device = device
        self.load_model()
    
    def load_model(self):
        """ 
        Loads the model, setting self.model.
        """

        # Initialize the model
        model = Net(model_name = self.arch, in_features = self.args.in_features, num_outputs = self.args.num_outputs, 
                    downsample = None,
                    patch_size = self.args.patch_size, pretrained_path = None,
                    local = (self.args.dataset_path == 'local'), device = self.device, biome_dim = self.args.biome_dim, emb_dim = self.args.emb_dim,
                    num_sepconv_blocks = self.args.num_sepconv_blocks, 
                    num_sepconv_filters = self.args.num_sepconv_filters, long_skip = self.args.long_skip, only_entry = self.args.only_entry, 
                    linear_emb = self.args.linear_emb, padding_mode = self.args.padding_mode, returns = self.args.returns,
                    # getattr: runs trained before the `upernet` arch existed have no upernet_* key
                    # in their wandb config; ignored by every other architecture anyway.
                    upernet_channels = getattr(self.args, 'upernet_channels', 512),
                    upernet_pyramid = getattr(self.args, 'upernet_pyramid', 'flat'),
                    upernet_levels = getattr(self.args, 'upernet_levels', 4))

        model = Model(model, lr = self.args.lr, step_size = self.args.step_size, gamma = self.args.gamma, 
                        patch_size = self.args.patch_size, downsample = self.args.downsample, 
                        loss_fn = self.args.loss_fn, film = self.args.film, l2 = self.args.l2, crop = self.args.crop)
    
        state_dict = torch.load(join(self.paths['ckpt'], self.arch, f'{self.model_name}_best.ckpt'), map_location = torch.device(self.device), weights_only = True)['state_dict']
        state_dict = {k:v for k,v in state_dict.items() if 'teacher' not in k}
        model.load_state_dict(state_dict)         
        model.to(self.device)
        model.eval()
        model.model.eval()
        self.model = model.model

#######################################################################################################################
# Code execution

def run_inference():
    
    # Get the command line arguments and set the global variables
    args, year, dataset_path, models, arch, saving_dir, tile_name, method, patch_size, pred_crop, masking, entity, dtype, mode, std, batch_size, factor = inf_parser()

    # Settings
    set_float32_matmul_precision('high')
    cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK')) if os.environ.get('SLURM_CPUS_PER_TASK') is not None else 8
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths. These come from config.sh at the repo root -- edit that file (or set
    # the matching environment variable) to run this elsewhere.
    dataset_path = get_paths(local = (dataset_path == 'local'))

    # Unlike train.py/eval.py, inference reads biomes_splits_to_name.pkl from the splits
    # directory rather than from alongside the patches.
    dataset_path['map'] = dataset_path['splits']

    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    api = wandb.Api()
    wandb_mapping, ckpt_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[models[0]]
    cfg = api.run(f'{WANDB_ENTITY}/{arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args)

    # Determine FiLM mode
    film = getattr(args, 'film', False)
    ensemble = getattr(args, 'ensemble', False)
    if ensemble and film:
        FILM_ENSEMBLE = True
        n_members = getattr(args, 'n_members', 1)
        assert len(models) == 1, "For FiLM ensemble, please specify only one model name."
        members = list(range(n_members)) if args.members is None else args.members
        assert all(0 <= m < n_members for m in members), f'Members {members} out of range for n_members = {n_members}.'
        models = [models[0] for _ in members]
    elif film and not ensemble:
        raise NotImplementedError("FiLM with biome embeddings is not yet supported for AEF inference. Only FiLM ensemble mode is currently implemented.")
    else:
        FILM_ENSEMBLE = False
        n_members = members = None
        assert args.members is None, '--members only applies to FiLM ensembles.'

    dataset_path['embeddings'] += '/AGBD-Lite' if args.lite else '/AGBD'

    # Load the models
    if FILM_ENSEMBLE:
        inference_obj = Inference(arch = arch, model_name = models[0], paths = dataset_path, tile_name = tile_name, args = args, device = device)
        inf_models = [inference_obj.model] * len(models)
    else:
        inference_objects = [Inference(arch = arch, model_name = model_name, paths = dataset_path, tile_name = tile_name, args = args, device = device) for model_name in models]
        inf_models = [inference_object.model for inference_object in inference_objects]
    
    # Load the input
    embeddings = load_embeddings(cfg, dataset_path, args.lite)
    with open(os.path.join(dataset_path['aef_norm'], 'AEF-statistics.pkl'), mode = 'rb') as f: norm_values = pickle.load(f)

    # Right now, the implementation is such that the script runs inference on all of the files for AGBRef
    # TODO: implement flexibility in this, and also the ability to pass a UTM tile, and lookup the AEF tiles/files in it

    # Getting the names of all the files for which to run inference
    tif_files = sorted(Path(dataset_path['aef']).glob("*/*.tiff"))
    tif_files = [f for f in tif_files if f.stem == f.parent.name]
    file_names = [str(f.relative_to(dataset_path['aef'])).replace('.tiff', '') for f in tif_files]
    if args.plots is not None:
        missing = set(args.plots) - {f.split('/')[0] for f in file_names}
        assert not missing, f'No AEF mosaic for plot(s): {sorted(missing)}'
        file_names = [f for f in file_names if f.split('/')[0] in args.plots]

    output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))

    # Record how these predictions were made, next to them
    os.makedirs(output_path, exist_ok = True)
    ckpt_files = [join(dataset_path['ckpt'], arch, f'{m}_best.ckpt') for m in sorted(set(models))]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output = True, text = True).stdout.strip()
    run_config = {'argv': sys.argv, 'cli_args': vars(inf_parser()[0]), 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'host': os.uname().nodename, 'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': git('status', '--porcelain', '--', 'model'),
                  'code_sha256': {f: hashlib.sha256(open(join(repo, 'model', f), 'rb').read()).hexdigest() for f in ['inference_aef.py', 'inference_ds.py', 'dataset.py', 'models.py', 'nico_net_film.py']},
                  'aef_dir': dataset_path['aef'], 'n_inputs': len(file_names), 'film_ensemble': FILM_ENSEMBLE,
                  'n_members': n_members, 'members': members,
                  'ckpts': {c: {'mtime': os.path.getmtime(c), 'sha256': hashlib.sha256(open(c, 'rb').read()).hexdigest()} for c in ckpt_files},
                  'norm_values': join(dataset_path['aef_norm'], 'AEF-statistics.pkl'),
                  'wandb_run': f'{WANDB_ENTITY}/{arch}/{wandb_name}', 'wandb_config': cfg}
    # One file per invocation: a run split into batches (--plots) must not overwrite the earlier batches' record
    with open(join(output_path, f"run_config_{time.strftime('%Y%m%d-%H%M%S')}.json"), 'w') as f: json.dump(run_config, f, indent = 1, default = str)

    for file_name in file_names :

        plot_id = file_name.split('/')[0]

        if args.skip_existing and os.path.exists(os.path.join(output_path, f'{plot_id}.tif')) :
            print(f'Skipping {plot_id} (prediction already exists).')
            continue

        # file_name = '127/127'  # 'xqxrk5qbbjfp4kp3u-0000008192-0000008192'
        img, pred_mask, meta = load_input(dataset_path, file_name, norm_values, masking = masking, latlon = cfg.get('latlon', False))
        print(img.shape)
        assert img.shape[-1] == args.in_features, f'Input has {img.shape[-1]} channels, but the model expects {args.in_features}.'

        # Get the ensemble predictions
        ds_cfg = {**cfg, 'film': False} if FILM_ENSEMBLE else cfg  # FiLM ensemble doesn't need biome data from the dataset
        dataset = InferenceDataset_v3(img, patch_size, pred_crop, ds_cfg, embeddings = embeddings, mode = mode, factor = factor)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = cpus_per_task)
        predictions = efficient_predict_tile_v3(dataloader, inf_models, device, dataset.pred_height, dataset.pred_width, film = cfg.get('film', False), film_ensemble = FILM_ENSEMBLE, n_members = n_members, members = members)

        # Get the average predictions
        avg_preds_variables = np.nanmean(predictions, axis = 0)
        if len(models) > 1 : avg_preds_std = np.nanstd(predictions, axis = 0)

        # Take care of the data type
        if dtype == 'uint16' :
            np_dtype, nodata = np.uint16, 65535
            avg_preds_variables[avg_preds_variables > 65535] = 65535
            if len(models) > 1 : avg_preds_std[avg_preds_std > 65535] = 65535
        elif dtype == 'float32' :
            np_dtype, nodata = np.float32, -9999.0
        else: raise Exception('Invalid dtype.')

        # Cast the data to the appropriate range/data type
        avg_preds_variables[avg_preds_variables < 0] = 0
        avg_preds_variables[np.isinf(avg_preds_variables)] = nodata
        avg_preds_variables[np.isnan(avg_preds_variables)] = nodata
        avg_preds_variables = avg_preds_variables.astype(np_dtype)
        print(avg_preds_variables.shape)
        if len(models) > 1 :
            avg_preds_std[np.isinf(avg_preds_std)] = nodata
            avg_preds_std[np.isnan(avg_preds_std)] = nodata
            avg_preds_std = avg_preds_std.astype(np_dtype)

        # Mask the predictions if needed
        if masking :
            if len(models) > 1 : avg_preds_std[pred_mask] = nodata
            avg_preds_variables[pred_mask] = nodata

        # Save the AGB predictions to GeoTIFF, with dtype uint16
        meta.update(driver = 'GTiff', dtype = np_dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
        if not os.path.exists(output_path): os.makedirs(output_path)
        with rs.open(os.path.join(output_path, f'{plot_id}.tif'), 'w', **meta) as f:
            f.write(avg_preds_variables, 1)
            f.set_band_description(1, 'AGB')
            if len(models) > 1 and std:
                f.write(avg_preds_std, 2)
                f.set_band_description(2, 'STD')

if __name__ == '__main__':
    t0 = time.time()
    run_inference()
    ttotal = time.time() - t0
    print(f'Inference done! in: {str(timedelta(seconds=ttotal))}.')