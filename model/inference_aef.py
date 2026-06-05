"""

This script performs inference for a model trained on AEF embeddings.

The script can be executed by running: bash inference/inference_aef.sh.

"""

#######################################################################################################################
# Imports

import time
from os.path import join
import os, pickle, argparse
import torch
import numpy as np
import rasterio as rs
from torch import set_float32_matmul_precision
from models import Net
from wrapper import Model
from torch import set_float32_matmul_precision
import wandb
from dataset import normalize_data
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
    parser.add_argument('--entity', type = str, default = 'gs-tp-biomass', help = 'wandb entity for the model.')
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
    args = parser.parse_args()

    return args, args.year, args.dataset_path, args.models, args.arch, args.saving_dir, args.tile_name, args.method, args.patch_size, args.pred_crop, args.masking, args.entity, args.dtype, args.mode, args.std, args.batch_size, args.factor


def load_input(paths, file_name, norm_values, masking = False):    
    """ 
    Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
    and normalize the input.

    Args:
    - paths: dict, paths to the dataset
    - file_name: str, the name of the file containing the AEF embeddings
    - norm_values: dict, normalization values for the bands
    - masking: bool, whether to apply masking to the input data

    Returns:
    - data: list, the input data for the model
    - mask: np.ndarray, the mask for the input data
    - meta: dict, metadata for the input data
    """
    
    start_time = time.time()
    print('Loading input...')

    # Load the AEF embeddings ------------------------------------------------------------------------------------
    with rs.open(join(paths['aef'], f'{file_name}.tiff')) as src:
        data = np.transpose(src.read(window = rs.windows.Window(0, 0, 1024, 1024)), axes = (1,2,0)).astype(np.float32)
        meta = src.meta
        meta.update({'height': data.shape[0], 'width': data.shape[1]})

    # Get the mask -----------------------------------------------------------------------------------------------
    if masking: mask = (data[:,:,0] == -128)
    else: mask = None

    # Normalize the AEF embeddings --------------------------------------------------------------------------------
    # data = np.where(data == -128, 0, (((data / 127.5) ** 2) * np.sign(data) + 1) / 2) # TODO put back once model with dataset.py corrected
    data = (((data / 127.5) ** 2) * np.sign(data) + 1) / 2
    data = normalize_data(data, norm_values, 'mean_std', 0, False).astype(np.float32)

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

def get_mapping(api, arch) :
    """
    This function constructs two dictionaries, one mapping the wandb name to the run's wandb identifier, and the other
    mapping the wandb name to the checkpoint path. This is done iteratively for all runs in the specified architecture.

    Args:
    - api (wandb.Api): the wandb API
    - arch (str): the architecture of the model
    """

    runs = api.runs(f"gs-tp-biomass/{arch}")
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
                    linear_emb = self.args.linear_emb, padding_mode = self.args.padding_mode, returns = self.args.returns)

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
    cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK') if os.environ.get('SLURM_CPUS_PER_TASK') is not None else 8
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths
    if dataset_path == 'local' : 
        dataset_path = {'h5':'/scratch3/gsialelli/patches', 
                        'norm': '/scratch3/gsialelli/patches', 
                        'map': '/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split',
                        'ckpt': '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/weights',
                        'embeddings': '/scratch3/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec',
                        'aef': '/scratch3/gsialelli/AEF',
                        'aef_h5': '/scratch3/gsialelli/patches/AEF',
                        'aef_norm': '/scratch3/gsialelli/patches/AEF'}
    else:
        dataset_path = {'h5':'/cluster/work/igp_psr/gsialelli/Data/patches', 
                        'norm': '/cluster/work/igp_psr/gsialelli/Data/patches', 
                        'map': '/cluster/work/igp_psr/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split',
                        'ckpt': '/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes',
                        'embeddings': '/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec',
                        'aef': '/cluster/work/igp_psr/gsialelli/Data/AEF',
                        'aef_h5': '/cluster/work/igp_psr/gsialelli/Data/patches/AEF',
                        'aef_norm': '/cluster/work/igp_psr/gsialelli/Data/patches/AEF'}
    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    api = wandb.Api()
    wandb_mapping, ckpt_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[models[0]]
    cfg = api.run(f'gs-tp-biomass/{arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args)

    # Determine FiLM mode
    film = getattr(args, 'film', False)
    ensemble = getattr(args, 'ensemble', False)
    if ensemble and film:
        FILM_ENSEMBLE = True
        n_members = getattr(args, 'n_members', 1)
        assert len(models) == 1, "For FiLM ensemble, please specify only one model name."
        models = [models[0] for _ in range(n_members)]
    elif film and not ensemble:
        raise NotImplementedError("FiLM with biome embeddings is not yet supported for AEF inference. Only FiLM ensemble mode is currently implemented.")
    else:
        FILM_ENSEMBLE = False
        n_members = None

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

    for file_name in file_names :

        plot_id = file_name.split('/')[0]

        # file_name = '127/127'  # 'xqxrk5qbbjfp4kp3u-0000008192-0000008192'
        img, pred_mask, meta = load_input(dataset_path, file_name, norm_values, masking = masking)
        print(img.shape)

        # Get the ensemble predictions
        ds_cfg = {**cfg, 'film': False} if FILM_ENSEMBLE else cfg  # FiLM ensemble doesn't need biome data from the dataset
        dataset = InferenceDataset_v3(img, patch_size, pred_crop, ds_cfg, embeddings = embeddings, mode = mode, factor = factor)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = cpus_per_task)
        predictions = efficient_predict_tile_v3(dataloader, inf_models, device, dataset.pred_height, dataset.pred_width, film = cfg.get('film', False), film_ensemble = FILM_ENSEMBLE, n_members = n_members)

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
            avg_preds_std[pred_mask] = nodata
            avg_preds_variables[pred_mask] = nodata

        # Save the AGB predictions to GeoTIFF, with dtype uint16
        meta.update(driver = 'GTiff', dtype = np_dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
        output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))
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