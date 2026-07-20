"""

This script performs inference for a model trained on the AGBD features (Sentinel-2 + ALOS + DEM +
LC), i.e. the baseline that AEF is compared against. It loads the input data, splits the Sentinel-2
tile into patches, predicts the AGBD for each patch, and mosaics the predictions back into a single
AGBD map for the tile, saved as a GeoTIFF.

Counterpart of `inference_aef.py`, which runs the AEF-embeddings models. The two produce the panels
of the side-by-side map figure, so they must stay consistent: same `Net`/`Model` construction, same
FiLM-ensemble handling, same output layout (`<saving_dir>/<arch>/<'_'.join(models)>/`).

Unlike `inference_aef.py`, this takes a `--product_name` (a specific S2 product) as well as a
`--tile_name`, since a tile has many acquisitions.

"""

#######################################################################################################################
# Imports

import time
from os.path import join
import os, pickle, argparse

# config.py lives at the repo root, but this package is run with model/ as the working
# directory, so the root has to be put on the path explicitly before importing it.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WANDB_ENTITY, get_paths
import torch
import numpy as np
import rasterio as rs
from skimage.transform import rescale
from torch import set_float32_matmul_precision
from models import Net
from wrapper import Model
from torch import set_float32_matmul_precision
import wandb
from inference_helper import *
from dataset import normalize_bands, normalize_data, encode_lc, one_hot_encode, embed_lc, get_doy, get_topology
import warnings
from parser import str2bool
from datetime import timedelta, datetime
import pandas as pd
from inference_residuals import init_args_dataset
from inference_ds import InferenceDataset, InferenceDataset_v2, InferenceDataset_v3
from torch.utils.data import DataLoader
import copy
from itertools import cycle

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
    parser.add_argument("--save_one_only", type = int, default = 0, help = 'Only save first model pred (but ensemble STD).')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--entity', type = str, default = WANDB_ENTITY, help = 'wandb entity for the model.')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--product_name", required = True, type = str, help = 'Product on which to run the prediction.')
    parser.add_argument("--dw", type = str2bool, default = 'false', help = 'Downsample the preds to 50m resolution.')
    parser.add_argument("--batch_size", type = int, default = 2, help = 'Batch size for the dataloader.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--overlap_size", nargs = 2, type = int, default = [100,100], help = 'Size (height,width) of the patches.')
    parser.add_argument("--pred_crop", nargs = 4, type = int, default = [0, 0, 0, 0], help = 'Pixels to crop off the predictions (off_ht, off_wl, off_hb, off_wr).')
    parser.add_argument("--overlap_mode", type = str, default = "last", help = 'Handling of overlapping predictions.')
    parser.add_argument("--masking", type = str2bool, default = 'false', help = 'Whether to mask the input.')
    parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data type to save the predictions.')
    parser.add_argument("--mode", type = str2bool, default = 'false', help = 'Whether to use mode for biome embedding.')
    parser.add_argument("--std", type = str2bool, default = 'true', help = 'Whether to compute and save the STDs in case of ensembling.')
    parser.add_argument("--factor", type = int, default = 5, help = 'Factor for Gaussian patch weighting in v3.')
    # The ALOS year is separable from --year because the 2020 PALSAR mosaic is visibly striped for
    # some tiles (32TNS: horizontal-streak anisotropy 1.71, vs 1.05 for 2018 and 0.92 for 2019, on
    # identical ground), and that striping propagates into the prediction (anisotropy 1.37). The AEF
    # panel is clean because it never ingests ALOS. --year cannot simply be moved: it also gates the
    # least-cloudy-product assert and the canopy-height layer, so flipping it would silently change
    # the S2 acquisition and CH too.
    # NOTE: the model was trained on year-matched S2+ALOS, so setting this creates a deliberate
    # train/inference mismatch. Use it for diagnosis, and disclose it if a figure depends on it.
    parser.add_argument("--alos_year", type = int, default = None,
                        help = 'Year for the ALOS mosaic only (default: --year). Diagnostic override.')
    args = parser.parse_args()
    if args.alos_year is None : args.alos_year = args.year

    return args, args.year, args.dataset_path, args.models, args.arch, args.saving_dir, args.tile_name, args.product_name, args.dw, args.patch_size, args.overlap_size, args.pred_crop, args.overlap_mode, args.masking, args.entity, args.dtype, args.mode, args.std, args.save_one_only, args.batch_size, args.factor


def load_input(year, paths, tile_name, product_name, norm_values, cfg, alos_order = ['HH', 'HV'], embeddings = None, debug = False, alos_year = None):
    """ 
    Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
    and normalize the input.

    Args:
    - paths (dict) : dictionary with keys `norm`, `tiles`, and `ckpt` and with values
        the paths to the corresponding file/folder
    - tile_name (str) : the name of the Sentinel-2 tile to load
    - norm_values (dict) : dictionary with the normalization values
    - cfg (dict) : dictionary with the configuration of the model
    - alos_order (list) : the order of the ALOS bands
    """
    
    start_time = time.time()
    print('Loading input...')

    # Initialize the data
    data = []

    # Sentinel 2 bands -------------------------------------------------------------------------------------------

    # 1. Get the product
    if product_name != "" : # for the few edge cases where we want to do inference on a specific product
        s2_prod = product_name
    else:
        with open(join(paths['tiles'], 'mapping_2019-2020-v2.pkl'), 'rb') as f: least_cloudy_products = pickle.load(f)
        s2_prod = least_cloudy_products[year][tile_name]
        assert year == int(s2_prod.split('_')[2][:4]), 'The year in the product name does not match the year specified.'
    
    # 2. Process the product
    transform, upsampling_shape, s2_bands, crs, bounds, boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta = process_S2_tile(s2_prod, paths['tiles'])
    patch_size = (upsampling_shape[0], upsampling_shape[1])
    scl_band = s2_bands.pop('SCL')

    if debug: 
        data = np.zeros((patch_size[0], patch_size[1], 31)).astype(np.float32)
        mask = np.full(fill_value = False, shape = (patch_size[0], patch_size[1]))

        if cfg.get('region', False) :
            with open(join(paths['region'], 's2_tile_to_region-v3.pkl'), 'rb') as f: region_mapping = pickle.load(f)
            region = region_mapping[tile_name]
            if isinstance(region, list) : region = region[0] # TODO, later, consider implementing multiple regions compatible code
            region = np.squeeze(one_hot_encode(np.full((1,1), region), 'region_cla').astype(np.float32))

        lc_raw = load_LC_data(paths['lc'], tile_name)
        lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
        lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
        biome = lc[..., 0]

        if cfg.get('film', False) : 
            if cfg.get('region', False) : data = (data, biome, region)
            else: data = (data, biome)
        print('done!')

        return data, mask, meta

    if cfg['bands'] != [] : 

        # 3. Get the SR values for the optical bands
        for band, band_value in s2_bands.items() :
            s2_bands[band] = (band_value - boa_offset * 1000) / 10000
    
        # 4. Normalize the data
        s2_order = cfg['bands']
        s2_bands = np.moveaxis(np.array([s2_bands[band] for band in s2_order]), 0, -1)
        s2_bands = normalize_bands(s2_bands, norm_values['S2_bands'], s2_order, cfg['norm_strat'], NODATAVALS['S2'])
        
        data.extend([s2_bands])
    
        # Sentinel-2 dates
        if cfg.get('s2_dates', False) :
            s2_date = datetime.strptime(s2_prod.split('_')[2][:8], '%Y%m%d')
            s2_num_days = (s2_date - datetime.strptime('2019-04-17', '%Y-%m-%d')).days
            s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, patch_size)
            s2_num_days = np.full((patch_size[0], patch_size[1]), s2_num_days).astype(np.float32)
            s2_num_days = normalize_data(s2_num_days, norm_values['Sentinel_metadata']['S2_date'], 'min_max' if cfg['norm_strat'] == 'pct' else cfg['norm_strat'])
            if cfg.get('s2_day', False) : 
                data.extend([s2_num_days[..., np.newaxis]])
            if cfg.get('s2_doy', False) :
                data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
    
    # Get the geographical coordinates ----------------------------------------------------------------------------
    if cfg['latlon']: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
    else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])

    # Get the GEDI dates -----------------------------------------------------------------------------------------
    if cfg.get('gedi_dates', False) :
        data.extend([s2_num_days[..., np.newaxis], s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])

    # Get the ALOS data ------------------------------------------------------------------------------------------
    if cfg.get('alos', False) :
        # 1. Get the data
        alos_raw = load_ALOS_data(tile_name, paths['alos'], min(alos_year if alos_year is not None else year, 2024))
        assert alos_raw, (f"No ALOS mosaic for {tile_name} in "
                          f"{alos_year if alos_year is not None else year}. load_ALOS_data returns "
                          f"an empty dict when the file is absent, and get_tile would then fail "
                          f"obscurely or the band would go all-nodata.")
        alos_tile = get_tile(alos_raw, transform, upsampling_shape, 'ALOS', ALOS_attrs)
        alos_bands = np.moveaxis(np.array([alos_tile['HH'], alos_tile['HV']]), 0, -1)
        # 2. Get the gamma naught values
        alos_bands = np.where(alos_bands == NODATAVALS['ALOS'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)
        # 3. Normalize the data
        alos_bands = normalize_bands(alos_bands, norm_values['ALOS_bands'], alos_order, cfg['norm_strat'], -9999.0)
        data.extend([alos_bands])

    # Get the CH data --------------------------------------------------------------------------------------------
    if cfg.get('ch', False) :
        # 1. Get the data
        ch_bands = load_CH_data(paths['ch'], tile_name, year)
        ch, ch_std = ch_bands['ch'], ch_bands['std']
        # 2. Normalize the data
        ch = normalize_data(ch, norm_values['CH']['ch'], cfg['norm_strat'], NODATAVALS['CH'])
        ch_std = normalize_data(ch_std, norm_values['CH']['std'], cfg['norm_strat'], NODATAVALS['CH'])
        data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
    
    # Get the LC data --------------------------------------------------------------------------------------------
    # 1. Get the data
    lc_raw = load_LC_data(paths['lc'], tile_name)
    lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
    lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
    biome = lc[..., 0]
    
    # 2. Transform the data
    if cfg.get('lc', False) :
        if cfg.get('ft_onehot', False) :
            _, lc_prob = embed_lc(lc, embeddings)
            lc = one_hot_encode(lc[:, :, 0], 'lc')
            data.extend([lc, lc_prob[..., np.newaxis]])
        elif cfg.get('ft_cat2vec', False) :
            lc, lc_prob = embed_lc(lc, embeddings)
            data.extend([lc, lc_prob[..., np.newaxis]])
        elif cfg.get('ft_sincos', False) :
            lc_cos, lc_sin, lc_prob = encode_lc(lc)
            data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
        else: raise ValueError('Invalid encoding for land cover data.')

    # Get the DEM data -------------------------------------------------------------------------------------------
    if cfg.get('dem', False) :
        
        # 1. Get the data
        dem_raw = load_DEM_data(paths['dem'], tile_name)
        dem_tile = get_tile(dem_raw, transform, upsampling_shape, 'DEM', DEM_attrs)
        dem = dem_tile['dem']

        # 2. Get the slope and aspect
        if cfg.get('topo', False) :
            slope, aspect_cos, aspect_sin = get_topology(dem)
            if cfg.get('slope', False) : data.extend([slope[..., np.newaxis]])
            if cfg.get('aspect', False) : data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])

        # 3. Normalize the data
        dem = normalize_data(dem, norm_values['DEM'], cfg['norm_strat'], NODATAVALS['DEM'])
        data.extend([dem[..., np.newaxis]])

    # Get the GEDI region class ----------------------------------------------------------------------------------
    if cfg.get('region', False) :
        with open(join(paths['region'], 's2_tile_to_region-v3.pkl'), 'rb') as f: region_mapping = pickle.load(f)
        region = region_mapping[tile_name]
        if isinstance(region, list) : region = region[0] # TODO, later, consider implementing multiple regions compatible code
        region = np.squeeze(one_hot_encode(np.full((1,1), region), 'region_cla').astype(np.float32))

    # Check that the model requires no residuals -----------------------------------------------------------------
    assert cfg.get('residuals', False) == False, 'Model uses residuals. Please refer to inference_residuals.py'

    # Concatenate the data ---------------------------------------------------------------------------------------
    data = torch.from_numpy(np.concatenate(data, axis = -1)).to(torch.float)
    
    # Append the biome embedding if FiLM is enabled --------------------------------------------------------------
    if cfg.get('film', False) : 
        if cfg.get('region', False) : data = (data, biome, region)
        else: data = (data, biome)

    # Get the mask -----------------------------------------------------------------------------------------------
    # i.e. where it is Water (6) and Snow or ice (11) or No Data (0) or Saturated or defective pixel (1)
    mask = (scl_band == 6) | (scl_band == 11) | (scl_band == 0) | (scl_band == 1)

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


def efficient_predict_tile(dataloader, models, device, pred_height, pred_width, pred_patch_height, pred_patch_width, overlap_mode = 'last'):
    """
    This function predicts the AGBD for a Sentinel-2 tile, using a list of models, and a dataloader.

    Args:
    - dataloader: (torch.utils.data.DataLoader) the dataloader to use for prediction
    - models: (list) the models to use for prediction
    - device: (torch.device) the device on which to perform inference
    - pred_height: (int) the height of the predicted AGBD
    - pred_width: (int) the width of the predicted AGBD

    Returns:
    - predictions: (np.ndarray) the predicted AGBD for the Sentinel-2 tile
    """

    print('Starting prediction...')

    # Placeholder for the predictions
    if overlap_mode == 'last' : predictions = np.full(shape = (len(models), pred_height, pred_width), fill_value = np.nan)
    elif overlap_mode == 'mean' :
        num_dimensions = 3
        dim_cyle = cycle(range(num_dimensions))
        predictions = np.full(shape = (len(models), num_dimensions, pred_height, pred_width), fill_value = np.nan)
    else: raise ValueError("Invalid overlap mode. Use 'last' or 'mean'.")

    # Iterate over the patches
    for batch in dataloader : 

        patch, biome_emb, pred_indices = batch

        # working implementation, v0
        x1s, x2s, y1s, y2s, off_hs, off_ws = pred_indices

        # Iterate over the models
        if overlap_mode == 'last' :
            for dim, model in enumerate(models) :
                preds = predict_patch(model, patch, device, biome_emb)
                for i, x1, x2, y1, y2, off_h, off_w in zip(range(len(preds)), x1s, x2s, y1s, y2s, off_hs, off_ws) :
                    predictions[dim, x1 : x2, y1 : y2] = preds[i][off_h : , off_w :]
    
        elif overlap_mode == 'mean' :
            for model_dim, model in enumerate(models) :
                preds = predict_patch(model, patch, device, biome_emb)
                for i, x1, x2, y1, y2, off_h, off_w in zip(range(len(preds)), x1s, x2s, y1s, y2s, off_hs, off_ws) :
                    dim = next(dim_cyle)
                    predictions[model_dim, dim, x1 : x2, y1 : y2] = preds[i][off_h : , off_w :]
    
        """ not working implementation? 
        x1s, x2s, y1s, y2s, off_hts, off_wls, off_hbs, off_wrs = pred_indices

        # Iterate over the models
        if overlap_mode == 'last' :
            for dim, model in enumerate(models) :
                preds = predict_patch(model, patch, device, biome_emb)
                for i, x1, x2, y1, y2, off_ht, off_wl, off_hb, off_wr in zip(range(len(preds)), x1s, x2s, y1s, y2s, off_hts, off_wls, off_hbs, off_wrs) :
                    predictions[dim, x1 : x2, y1 : y2] = preds[i][off_ht : pred_patch_height - off_hb , off_wl : pred_patch_width - off_wr]
    
        elif overlap_mode == 'mean' :
            for model_dim, model in enumerate(models) :
                preds = predict_patch(model, patch, device, biome_emb)
                for i, x1, x2, y1, y2, off_ht, off_wl, off_hb, off_wr in zip(range(len(preds)), x1s, x2s, y1s, y2s, off_hts, off_wls, off_hbs, off_wrs) :
                    dim = next(dim_cyle)
                    predictions[model_dim, dim, x1 : x2, y1 : y2] = preds[i][off_ht : pred_patch_height - off_hb , off_wl : pred_patch_width - off_wr]
        """
                    
    print('done!')
        
    return predictions


def efficient_predict_tile_v2(dataloader, models, device, pred_height, pred_width, pred_crop):
    """
    This function predicts the AGBD for a Sentinel-2 tile, using a list of models, and a dataloader.
    The difference with v1 is that v2 generates e.g. 250x250 overlapping patches on which inference
    is run, but the predictions are then cropped to non-overlapping patches e.g. 200x200.

    Args:
    - dataloader: (torch.utils.data.DataLoader) the dataloader to use for prediction
    - models: (list) the models to use for prediction
    - device: (torch.device) the device on which to perform inference
    - pred_height: (int) the height of the predicted AGBD
    - pred_width: (int) the width of the predicted AGBD
    - pred_crop: (int) the number of pixels to crop off the predictions

    Returns:
    - predictions: (np.ndarray) the predicted AGBD for the Sentinel-2 tile
    """
    print('Starting prediction...')
    pred_crop = int(pred_crop[0])  # Ensure pred_crop is an integer
    # Placeholder for the predictions
    predictions = np.full(shape = (len(models), pred_height, pred_width), fill_value = np.nan)
    # Iterate over the patches
    for batch in dataloader : 
        # Unpack the batch
        patch, biome_emb, pred_indices = batch
        x1s, x2s, y1s, y2s = pred_indices
        # Iterate over the models
        for dim, model in enumerate(models) :
            preds = predict_patch(model, patch, device, biome_emb)
            cropped_preds = preds[:, pred_crop : -pred_crop, pred_crop : -pred_crop]
            for i, (x1, x2, y1, y2) in enumerate(zip(x1s, x2s, y1s, y2s)) :
                predictions[dim, x1 : x2, y1 : y2] = cropped_preds[i]
    print('done!')
    return predictions


def efficient_predict_tile_v3(dataloader, models, device, pred_height, pred_width, film = False, film_ensemble = False, n_members = None):
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
    
    Returns:
    - predictions: (np.ndarray) the predicted AGBD for the Sentinel-2 tile
    """
    
    print('Starting prediction...')
    
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
                member_biome_emb[:, model_dim] = 1.0
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
        # Mirrors inference_aef.py exactly. This repo's models.py is a trimmed version of the one
        # this script came from: channel_dims, max_pool, leaky_relu, freeze_shared, scramble,
        # random_spec, num_spec_layers, debug_film and bn are no longer accepted by Net/Model.
        model = Net(model_name = self.arch, in_features = self.args.in_features, num_outputs = self.args.num_outputs,
                    downsample = None,
                    patch_size = self.args.patch_size, pretrained_path = None,
                    local = (self.args.dataset_path == 'local'), device = self.device, biome_dim = self.args.biome_dim, emb_dim = self.args.emb_dim,
                    num_sepconv_blocks = self.args.num_sepconv_blocks,
                    num_sepconv_filters = self.args.num_sepconv_filters, long_skip = self.args.long_skip, only_entry = self.args.only_entry,
                    linear_emb = self.args.linear_emb, padding_mode = self.args.padding_mode, returns = self.args.returns)

        model = Model(model, lr = self.args.lr, step_size = self.args.step_size, gamma = self.args.gamma,
                        patch_size = self.args.patch_size, downsample = self.args.downsample,
                        loss_fn = self.args.loss_fn, film = self.args.film, l2 = self.args.l2, crop = self.args.crop)
    
        state_dict = torch.load(join(self.paths['ckpt'], self.arch, f'{self.model_name}_best.ckpt'), map_location = torch.device(self.device), weights_only = True)['state_dict']
        state_dict = {k:v for k,v in state_dict.items() if 'teacher' not in k}
        model.load_state_dict(state_dict) 
        # add the following line if the nico model is not compiled : state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
        
        model.to(self.device)
        model.eval()
        model.model.eval()
        self.model = model.model

#######################################################################################################################
# Code execution

def run_inference():
    
    # Get the command line arguments and set the global variables
    args, year, dataset_path, models, arch, saving_dir, tile_name, product_name, dw, patch_size, overlap_size, pred_crop, overlap_mode, masking, entity, dtype, mode, std, save_one_only, batch_size, factor = inf_parser()

    # Settings
    set_float32_matmul_precision('high')
    if (dataset_path == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths. Read from config.sh (via config.py) rather than hardcoded here, so the shell
    # and Python halves of the pipeline cannot disagree about where the data lives.
    local = (dataset_path == 'local')
    dataset_path = get_paths(local = local)

    # The raw per-tile rasters are optional in get_paths(): installs that only train/eval on the h5
    # patches do not have them, and every other entry point would KeyError. This script is the one
    # that needs them, so check up front and name the missing config.sh variable rather than dying
    # with a bare KeyError deep inside the tile loader.
    prefix = 'AGBD_LOCAL_' if local else 'AGBD_CLUSTER_'
    missing = [k for k in ['tiles', 'alos', 'dem', 'lc', 'region'] if k not in dataset_path]
    if missing :
        raise KeyError(f"inference_agbd.py needs the raw per-tile rasters, but {', '.join(missing)} "
                       f"are not configured. Set {', '.join(prefix + k.upper() for k in missing)} in config.sh.")

    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    api = wandb.Api()
    wandb_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[models[0]]
    cfg = api.run(f'{entity}/{arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args) # add the missing arguments to the config

    # Canopy height is a supported input in load_input() but is not wired into config.sh, because
    # the models this script exists for (the AGBD-features baseline) all have ch=False. Fail here
    # with the fix rather than with a bare KeyError on paths['ch'] once the tile loader is running.
    if cfg.get('ch', False) and 'ch' not in dataset_path :
        raise KeyError(f"model {models[0]} has ch=True, but the canopy-height rasters are not "
                       f"configured. Add {'AGBD_LOCAL_CH' if local else 'AGBD_CLUSTER_CH'} to "
                       f"config.sh and the 'CH' key to config.py's optional-paths list.")

    # Determine FiLM mode. Mirrors inference_aef.py: a FiLM ensemble is ONE checkpoint evaluated
    # n_members times with a different one-hot member embedding, not n separate checkpoints -- so a
    # single --models name is passed and self-triples here. The old inference.py this was copied
    # from predates FiLM ensembling and would have silently run member 0 only, which would make the
    # AGBD-features panel of the map figure incomparable to the AEF one.
    film = getattr(args, 'film', False)
    ensemble = getattr(args, 'ensemble', False)
    if ensemble and film:
        FILM_ENSEMBLE = True
        n_members = getattr(args, 'n_members', 1)
        assert len(models) == 1, "For FiLM ensemble, please specify only one model name."
        models = [models[0] for _ in range(n_members)]
    elif film and not ensemble:
        raise NotImplementedError("FiLM with biome embeddings is not yet supported here. Only FiLM ensemble mode is currently implemented.")
    else:
        FILM_ENSEMBLE = False
        n_members = None

    # Load the models
    if FILM_ENSEMBLE:
        inference_obj = Inference(arch = arch, model_name = models[0], paths = dataset_path, tile_name = tile_name, args = args, device = device)
        inf_models = [inference_obj.model] * len(models)
    else:
        inference_objects = [Inference(arch = arch, model_name = model_name, paths = dataset_path, tile_name = tile_name, args = args, device = device) for model_name in models]
        inf_models = [inference_object.model for inference_object in inference_objects]

    # Load the cat2vec embeddings if needed. The cat2vec directory gained AGBD/ and AGBD-Lite/
    # subdirectories since this script's ancestor was written, which assumed a flat layout; select
    # on the model's own `lite` flag, as inference_aef.py does.
    if cfg.get('ft_cat2vec', False) or cfg.get('emb_cat2vec', False) :
        dataset_path['embeddings'] += '/AGBD-Lite' if cfg.get('lite', False) else '/AGBD'
        embeddings = pd.read_csv(join(dataset_path['embeddings'], 'embeddings_train.csv'))
        embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
    else: embeddings = None

    # Load the input
    new_stats, prob_norm = cfg.get('new_stats', False), cfg.get('prob_norm', False)
    with open(os.path.join(dataset_path['norm'], f"statistics_subset_2019-2020-v4{'-1' if (new_stats or prob_norm) else ''}.pkl"), mode = 'rb') as f: norm_values = pickle.load(f)
    # Under a FiLM ensemble the biome embedding is replaced by a one-hot member index, so the biome
    # data is not wanted anywhere: load_input() would otherwise return a (data, biome) tuple that
    # InferenceDataset_v3 -- told film=False -- would not unpack, and img.shape would fail on it.
    # (inference_aef.py needs this only at the dataset; its load_input never appends biome.)
    ds_cfg = {**cfg, 'film': False} if FILM_ENSEMBLE else cfg
    img, mask, meta = load_input(year, dataset_path, tile_name, product_name, norm_values, ds_cfg, embeddings = embeddings, alos_year = args.alos_year)
    
    # Take care of downsampling, if needed
    if cfg['downsample'] or dw :
        assert dw == cfg['downsample'], 'The downsample argument should be the same as the one in the config.'
        size = 3
        pred_mask = rescale(mask, size / 15)
    else:
        size = 15
        pred_mask = mask

    # Get the ensemble predictions
    #dataset = InferenceDataset(img, size, patch_size, overlap_size, pred_crop, cfg, embeddings = embeddings, mode = mode)
    #dataset = InferenceDataset_v2(img, patch_size, pred_crop, cfg, embeddings = embeddings, mode = mode)
    dataset = InferenceDataset_v3(img, patch_size, pred_crop, ds_cfg, embeddings = embeddings, mode = mode, factor = factor)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = cpus_per_task)
    predictions = efficient_predict_tile_v3(dataloader, inf_models, device, dataset.pred_height, dataset.pred_width, film = cfg.get('film', False), film_ensemble = FILM_ENSEMBLE, n_members = n_members)

    # Get the average predictions
    if save_one_only and (len(models) > 1) : 
        if dtype == 'uint16' : _dtype, _nodata = np.uint16, 65535
        elif dtype == 'float32' : _dtype, _nodata = np.float32, -9999.0
        if overlap_mode == 'last' : first_preds_variables = predictions[save_one_only - 1, :, :]
        elif overlap_mode == 'mean' : first_preds_variables = np.nanmean(predictions[save_one_only - 1, :, :, :], axis = 0)
        first_preds_variables[first_preds_variables < 0] = 0
        first_preds_variables[np.isinf(first_preds_variables)] = _nodata
        first_preds_variables[np.isnan(first_preds_variables)] = _nodata
        first_preds_variables = first_preds_variables.astype(_dtype)

    if overlap_mode == 'mean' : predictions = np.nanmean(predictions, axis = 1)
    avg_preds_variables = np.nanmean(predictions, axis = 0)
    if len(models) > 1 : avg_preds_std = np.nanstd(predictions, axis = 0)

    # Take care of the data type
    if dtype == 'uint16' :
        dtype, nodata = np.uint16, 65535
        avg_preds_variables[avg_preds_variables > 65535] = 65535
        if len(models) > 1 : avg_preds_std[avg_preds_std > 65535] = 65535
    elif dtype == 'float32' : 
        dtype, nodata = np.float32, -9999.0
    else: raise Exception('Invalid dtype.')

    # Cast the data to the appropriate range/data type
    avg_preds_variables[avg_preds_variables < 0] = 0
    avg_preds_variables[np.isinf(avg_preds_variables)] = nodata
    avg_preds_variables[np.isnan(avg_preds_variables)] = nodata
    avg_preds_variables = avg_preds_variables.astype(dtype)
    if len(models) > 1 :
        avg_preds_std[np.isinf(avg_preds_std)] = nodata
        avg_preds_std[np.isnan(avg_preds_std)] = nodata
        avg_preds_std = avg_preds_std.astype(dtype)

    # If running inference on a specific product, use the product name as tile name
    if product_name != "" : 
        tile_name = product_name
        year = 'NA'

    # Save the AGB predictions to GeoTIFF, with dtype uint16
    if save_one_only and (len(models) > 1) :
        meta_v2 = copy.deepcopy(meta)
        meta_v2.update(driver = 'GTiff', dtype = dtype, count = 1, compress = 'lzw', nodata = nodata)
        output_path = join(dataset_path['saving_dir'], arch, models[save_one_only - 1])
        if not os.path.exists(output_path): os.makedirs(output_path)
        with rs.open(os.path.join(output_path, f'{tile_name}_{year}.tif'), 'w', **meta_v2) as f:
            f.write(first_preds_variables, 1)
            f.set_band_description(1, 'AGB')

    meta.update(driver = 'GTiff', dtype = dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
    output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))
    if not os.path.exists(output_path): os.makedirs(output_path)
    with rs.open(os.path.join(output_path, f'{tile_name}_{year}.tif'), 'w', **meta) as f:
        f.write(avg_preds_variables, 1)
        f.set_band_description(1, 'AGB')
        if len(models) > 1 and std:
            f.write(avg_preds_std, 2)
            f.set_band_description(2, 'STD')

    # Save the mask if needed
    if masking :
        meta.update(driver = 'GTiff', dtype = 'uint8', count = 1, compress = 'lzw', nodata = 255)
        output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))
        if not os.path.exists(output_path): os.makedirs(output_path)
        with rs.open(os.path.join(output_path, f'MASK_{tile_name}_{year}.tif'), 'w', **meta) as f:
            f.write(pred_mask, 1)
            f.set_band_description(1, 'MASK')
    

if __name__ == '__main__':
    t0 = time.time()
    run_inference()
    ttotal = time.time() - t0
    print(f'Inference done! in: {str(timedelta(seconds=ttotal))}.')