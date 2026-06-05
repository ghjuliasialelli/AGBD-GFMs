"""

DESCRIPTION - This script performs inference on a Sentinel-2 tile, using a trained model. To the difference of the
`inference.py` file, this script is designed to make predictions for locations where we have access to the GEDI L4A
footprints, to compute the residuals and provide it as input to the model.

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
from inference_helper import *
import warnings
from parser import str2bool
from datetime import timedelta
from rasterio.transform import rowcol
from dataset import GEDIDataset
from torch.utils.data import DataLoader
from copy import deepcopy

# Silencing specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
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
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the Sentinel-2 tile.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--dw", type = str2bool, default = 'false', help = 'Downsample the preds to 50m resolution.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--overlap_size", nargs = 2, type = int, default = [100,100], help = 'Size (height,width) of the patches.')
    parser.add_argument("--masking", type = str2bool, default = 'false', help = 'Whether to mask the input.')
    args = parser.parse_args()

    return args, args.dataset_path, args.models, args.arch, args.saving_dir, args.tile_name, args.year, args.dw, args.patch_size, args.overlap_size, args.masking

#######################################################################################################################
# Inference class definition

class Inference:

    """ 
    An `Inference` object loads a PyTorch model and performs AGBD inference at the Sentinel-2 tile level.
    """

    def __init__(self, arch, model_name, paths, args, device):

        """
        Initialization method.

        Args:
        - arch (str) : the architecture of the model
        - model_name (str) : the name of the model
        - paths (dict) : the paths to the dataset
        - args (argparse.Namespace) : the command-line arguments
        - device (torch.device) : the device on which to perform

        Returns:
        - None
        """

        self.arch = arch
        self.model_name = model_name
        self.paths = paths
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
        
        # Initialize the Wrapper model
        model = Model(model, lr = self.args.lr, step_size = self.args.step_size, gamma = self.args.gamma, 
                            patch_size = self.args.patch_size, downsample = self.args.downsample, 
                            loss_fn = self.args.loss_fn, film = self.args.film, l2 = self.args.l2, crop = self.args.crop)
    
        state_dict = torch.load(join(self.paths['ckpt'], self.arch, f'{self.model_name}_best.ckpt'), map_location = torch.device(self.device), weights_only = True)['state_dict']
        model.load_state_dict(state_dict) 
        # add the following line if the nico model is not compiled : state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
        
        model.to(self.device)
        model.eval()
        model.model.eval()
        self.model = model.model


def get_teacher(args, device, ckpt_path) :
    """
    This function loads the teacher model, and returns it along with the arguments used to train it.

    Args:
    - args (argparse.Namespace) : the command-line arguments

    Returns:
    - teacher_model (nn.Module) : the teacher model
    - ts_args (argparse.Namespace) : the arguments used to train the teacher model
    """

    # Initialize the arguments
    parser = argparse.ArgumentParser()
    ts_args, _ = parser.parse_known_args()
    default_args = {'augment': False, 'norm': False, 'chunk_size': 1, 'model_idx': None, 'film': False, 'latlon': True, 'ch': False, 'bands': None, 's1': False, 'alos': False, \
                    'lc': False, 'dem': False, 'gedi_dates': False, 's2_dates': False, 's2_day': False, 's2_doy': False, 'topo': False, 'aspect': False, 'slope': False, \
                    'ft_cat2vec': False, 'ft_onehot': False, 'ft_sincos': False, 'emb_cat2vec': False, 'emb_onehot': False, 'emb_dist': False, 'emb_sincos': False, 'residuals': False, \
                    'res_norm': False, 'res_film': False, 'res_in': False, 'res_in_central': False, 'res_in_patch': False, 'biome_dim': 128, 'emb_dim': None, 'linear_emb': False, \
                    'region': False, 'biome': False, 'debug_film': False, 'bn': 'yes', 'rh98_film': False, 'prob_norm': False, 'debug_latlon': False, 'new_stats': False, \
                    'n_epochs': 100, 'limit': False, 'batch_size': 256, 'years': None, 'pretrained_path': None, 'freeze_shared': True, 'random_spec': False, 'scramble': False, \
                    'num_spec_layers': 1, 'channel_dims': None, 'downsample': False, 'max_pool': False, 'leaky_relu': False, 'num_sepconv_blocks': 8, 'num_sepconv_filters': 728, \
                    'long_skip': False, 'only_entry': False, 'lr': 0.0001, 'step_size': 30, 'gamma': 0.1, 'l2': 0.0, 'patience': 3, 'min_delta': 0.0, 'reweighting': None, \
                    'tile_name': None, 'clip': True, 'output_path': None, 'n_models': None, 'patch_size': [25, 25], 'crop' : False, 'padding_mode': 'zeros', 'teacher': '', \
                    'teacher_inpaint': False, 'mixed': False, 'ndvi': False, 'returns': 'dense'}
    for key, value in default_args.items() : setattr(ts_args, key, value)
    
    # Manually set the teacher's patch size to 25x25 for inference
    ts_args.patch_size = args.patch_size
    
    # Get the teacher model config from wandb
    api = wandb.Api()
    wandb_mapping = get_mapping(api, args.teacher_arch)
    wandb_name = wandb_mapping[args.teacher]
    cfg = api.run(f'gs-tp-biomass/{args.teacher_arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(ts_args, key, value)

    # Get the path to the saved model
    teacher_path = ckpt_path
    print(f'Loading teacher model from {teacher_path}.')

    # Load the teacher model
    inference_object = Inference(arch = args.teacher_arch, model_name = args.teacher, paths = {'ckpt': teacher_path}, args = ts_args, device = device)
    teacher_model = inference_object.model

    # Freeze the teacher model
    for param in teacher_model.parameters(): param.requires_grad = False

    return teacher_model, ts_args

def get_s2_indices_from_coords(transform, reprojector, s2_shape, lat, lon, patch_size) :
    
    # Get the row and column corresponding to the footprint center
    lon, lat = reprojector.transform(lon, lat)
    pt_x, pt_y = lon, lat 
    x, y = rowcol(transform, pt_x, pt_y)
    x, y = int(x), int(y)
    
    # Get the size of the window to extract
    x_offset, y_offset = (patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2

    # Check that the patch fits in the tile, otherwise skip (since we need the whole patch)
    if (x - x_offset < 0) or (x + x_offset + 1 > s2_shape[0]) \
        or (y - y_offset < 0) or (y + y_offset + 1 > s2_shape[1]) :
            return None

    return x, y, x_offset, y_offset


def get_s2_info(tile_name, paths) :

    # Get one Sentinel-2 product for that tile
    with open(join(paths['tiles'], 'mapping.pkl'), 'rb') as f: least_cloudy_products = pickle.load(f)
    s2_prod = least_cloudy_products[tile_name]
    date = s2_prod.split('_')[2]

    # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
    path_to_img_data = glob.glob(join(paths['tiles'], s2_prod + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]

    with rs.open(join(path_to_img_data, 'R10m', f'T{tile_name}_{date}_B02_10m.tif')) as src :
        s2_shape = src.shape
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        meta = src.meta
    
    return transform, s2_shape, crs, bounds, meta


def get_mode(tile_name, paths) :

    # Load the mapping from mode to tile name
    with open(join(paths['map'], 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)
    for mode in ['train', 'val', 'test'] :
        if tile_name in tile_mapping[mode] :
            return mode
    
    raise ValueError(f'Tile {tile_name} not found in the mapping.')


def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return CRS.from_string(crs)


def init_args_dataset(args) :

    default_args = {'augment': False, 'norm': False, 'chunk_size': 1, 'model_idx': None, 'film': False, 'latlon': True, 'ch': False, 'bands': None, 's1': False, 'alos': False, \
                    'lc': False, 'dem': False, 'gedi_dates': False, 's2_dates': False, 's2_day': False, 's2_doy': False, 'topo': False, 'aspect': False, 'slope': False, 'ft_cat2vec': False, \
                    'ft_onehot': False, 'ft_sincos': False, 'emb_cat2vec': False, 'emb_onehot': False, 'emb_dist': False, 'emb_sincos': False, 'residuals': False, 'res_norm': False, \
                    'res_film': False, 'res_in': False, 'res_in_central': False, 'res_in_patch': False, 'biome_dim': 128, 'emb_dim': None, 'linear_emb': False, 'region': False, 'biome': False, \
                    'debug_film': False, 'bn': 'yes', 'rh98_film': False, 'prob_norm': False, 'debug_latlon': False, 'new_stats': False, 'n_epochs': 100, 'limit': False, 'batch_size': 256, \
                    'years': None, 'pretrained_path': None, 'freeze_shared': True, 'random_spec': False, 'scramble': False, 'num_spec_layers': 1, 'channel_dims': None, 'downsample': False, \
                    'max_pool': False, 'leaky_relu': False, 'num_sepconv_blocks': 8, 'num_sepconv_filters': 728, 'long_skip': False, 'only_entry': False, 'lr': 0.0001, 'step_size': 30, \
                    'gamma': 0.1, 'l2': 0.0, 'patience': 3, 'min_delta': 0.0, 'reweighting': None, 'tile_name': None, 'clip': True, 'output_path': None, 'n_models': None, 'patch_size': [25, 25], \
                    'crop': False, 'padding_mode': 'zeros', 'mixed': False, 'ndvi': False, 'teacher': '', 'teacher_inpaint': False, 'returns': 'dense', 'agb_residuals': False, \
                    'agb_residuals_film': False, 'agb_residuals_file': 'N/A', 'agb_res_all': False, 'agb_res_one': 'N/A', 'log_transform' : False, 'sim_dist': False, 'hold_out_region': None,
                    'lite_eval_big': False, 'lite_chunk_size': 1, 'aef': False, 'aef_bands': None, 'keep_region': False, 'drop_overlaps': False, 'tessera': False, 'temp_ablation': False,
                    'trained_years': [], 'years_stats': None, 'subsample_2020': False}
    for key, value in default_args.items() :
        if key not in args : setattr(args, key, value)
    
    return args


def get_dense_labels(model, inputs, film, returns = 'dense') :

    if returns == 'pixel' :

        if len(inputs) == 2 : 
            t_images, t_biome_embs = inputs
        kernel, stride = 15, 1
        subpatches = t_images.unfold(2, kernel, stride).unfold(3, kernel, stride) # shape (B, C, 11, 11, 15, 15)
        subpatches = subpatches.reshape(subpatches.shape[0], subpatches.shape[1], -1, kernel, kernel) # shape (B, C, 121, 15, 15)
        subpatches = subpatches.permute(0, 2, 1, 3, 4) # shape (B, 121, C, 15, 15)
        subpatches = subpatches.flatten(0, 1) # shape (B * 121, C, 15, 15)
        if len(inputs) == 2 :
            t_biome_embs = t_biome_embs.unsqueeze(1).expand(-1, 121, -1).flatten(0, 1)
            predictions = model((subpatches, t_biome_embs)) # shape (B * 121, 1, 1, 1)
        else: predictions = model(subpatches) # shape (B * 121, 1, 1, 1)
        predictions = predictions.view(t_images.shape[0], 11, 11, 1, 1) # shape (B, 11, 11, 1, 1)
        predictions = predictions.permute(0, 3, 1, 4, 2).squeeze(3).squeeze(3) # shape (B, 1, 11, 11)

    else:
        predictions = model(inputs)
    
    return predictions


#######################################################################################################################
# Code execution

def run_inference():
    
    # Get the command line arguments and set the global variables
    args, dataset_path, models, arch, saving_dir, tile_name, year, dw, patch_size, overlap_size, masking = inf_parser()

    # Settings
    set_float32_matmul_precision('high')
    if (dataset_path == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths
    if dataset_path == 'local' : 
        dataset_path = {'h5':'/scratch3/gsialelli/patches', 
                        'norm': '/scratch3/gsialelli/patches', 
                        'map': '/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split',
                        'embeddings': '/scratch3/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec',
                        'ckpt': '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/weights',
                        'tiles': '/scratch3/gsialelli/S2_L2A'}
    else:
        dataset_path = {k:args.dataset_path for k in ['h5', 'norm', 'map', 'embeddings']}
    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    api = wandb.Api()
    wandb_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[models[0]]
    cfg = api.run(f'gs-tp-biomass/{arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args)

    # Deal with the patch size
    # if args.crop : patch_size = cfg['patch_size'], TODO consider whether we should put it back
    assert patch_size[0] == patch_size[1]
    if args.returns == 'dense': args.patch_size = patch_size # TODO maybe remove, in this case
    if args.film : assert patch_size[0] <= 25, 'Patch size must be small for FiLM models.'

    # Load the input
    mode = get_mode(tile_name, dataset_path)
    ds_args = deepcopy(args)
    if args.returns == 'pixel' : 
        ds_args.patch_size = [25, 25]
        bs = 8
        patch_size = [11, 11]
    else: bs = 256
    ds = GEDIDataset(dataset_path, chunk_size = 1, mode = mode, args = ds_args, debug = False, years = [year], film = args.film, inference = True, tile_name = tile_name)
    data_loader = DataLoader(dataset = ds, batch_size = bs, shuffle = False, num_workers = 8)

    # Load the models
    inference_objects = [Inference(arch = arch, model_name = model_name, paths = dataset_path, args = args, device = device) for model_name in models]
    inf_models = [inference_object.model for inference_object in inference_objects]
    model = inf_models[0] # TODO in the long run, make it run for the ensemble

    # Get the predictions
    transform, s2_shape, crs, bounds, meta = get_s2_info(tile_name, dataset_path)
    s2_crs = get_CRS_from_S2_tilename(tile_name)
    reprojector = Transformer.from_crs("EPSG:4326", s2_crs, always_xy = True)
    predictions = np.full(shape = (s2_shape[0], s2_shape[1], 1), fill_value = np.nan, dtype = np.float32)
    canopy_height = np.full(shape = (s2_shape[0], s2_shape[1]), fill_value = np.nan, dtype = np.float32)
    for batch in data_loader:
        if args.film: 
            images, biomes, biome_embs, labels, lats, lons, chs = batch
            images, biome_embs = images.to(device), biome_embs.to(device)
            preds = get_dense_labels(model, (images, biome_embs), args.film, args.returns)
        else: 
            images, biomes, labels, lats, lons, chs = batch
            images = images.to(device)
            preds = get_dense_labels(model, images, args.film, args.returns)
        preds = preds.cpu().detach().numpy()[:,0,:,:]

        assert len(lats) == len(lons) == len(preds), 'Lengths do not match.'

        # Populate the predictions
        for lat, lon, pred, ch in zip(lats, lons, preds, chs) :
            
            # Get the indices of the Sentinel-2 tile corresponding to the GEDI footprint
            s2_indices = get_s2_indices_from_coords(transform, reprojector, s2_shape, lat, lon, patch_size)
            if s2_indices is None: continue
            else: x, y, x_offset, y_offset = s2_indices
            
            # Find a dimension that is not already populated
            need_new_dim = True
            for dim in range(predictions.shape[-1]) :

                # If the destination patch already has non NaN values, look at the next dimension
                if np.any(~np.isnan(predictions[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1, dim])) :
                    continue
                
                # Otherwise, we don't need a new dimension and we can just populate the patch
                else: 
                    need_new_dim = False
                    break
            
            # Or create a new dimension if needed
            if need_new_dim :

                print('New dim.')

                dim = predictions.shape[-1]
                new_array = np.full(shape = (s2_shape[0], s2_shape[1], dim + 1), fill_value = np.nan, dtype = np.float32)
                new_array[..., : dim] = predictions
                predictions = new_array

                predictions[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1, -1] = pred
            
            else:
                predictions[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1, dim] = pred
            
            # Write the CH
            canopy_height[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1] = ch
            
    # Take care of overlapping patches, take the mean
    predictions = np.nanmean(predictions, axis = -1)
    
    # Cast negative AGB values to 0, and all values to uint16
    predictions[predictions < 0] = 0
    canopy_height[canopy_height < 0] = 0
    predictions[predictions > 65535] = 65535
    canopy_height[canopy_height > 65535] = 65535
    predictions[np.isinf(predictions)] = 65535
    predictions[np.isnan(predictions)] = 65535
    canopy_height[np.isnan(canopy_height)] = 65535
    canopy_height[canopy_height == 255] = 65535
    predictions = predictions.astype(np.uint16)
    canopy_height = canopy_height.astype(np.uint16)

    # Get the metadata from the original Sentinel 2 tile
    print(f'Saving predictions to {os.path.join(dataset_path["saving_dir"], arch, f"residuals_{tile_name}.tif")}')
    
    # Save the AGB predictions to GeoTIFF, with dtype uint16
    meta.update(driver = 'GTiff', dtype = np.uint16, count = 2, compress = 'lzw', nodata = 65535)
    output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))
    if not os.path.exists(output_path): os.makedirs(output_path)
    with rs.open(os.path.join(output_path, f"residuals_{'_'.join(models)}_{tile_name}.tif"), 'w', **meta) as f:
        f.write(predictions, 1)
        f.set_band_description(1, 'AGB')
        f.write(canopy_height, 2)
        f.set_band_description(2, 'CH')

if __name__ == '__main__':
    t0 = time.time()
    run_inference()
    ttotal = time.time() - t0
    print(f'Inference done! in: {str(timedelta(seconds=ttotal))}.')