"""

This script evaluates the pecified models on the specified set and saves various analyses. The arguments are described
in eval_parser(). In order to run the script, you can use bash eval/eval.sh. 

"""

#######################################################################################################################
# Imports

import os
import sys
from models import Net
from wrapper import Model
from dataset import GEDIDataset
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
# wandb is imported lazily, inside the AGBD_WANDB_LOOKUP branch below, so that evaluating
# the released checkpoints offline does not require the package to be installed at all.
from os.path import join, isdir, exists

# config.py lives at the repo root, but this package is run with model/ as the working
# directory, so the root has to be put on the path explicitly before importing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_paths, WANDB_ENTITY, WANDB_LOOKUP
from wandb_cache import get_mapping_offline
from os import mkdir
import argparse
import numpy as np
import torch
from parser import str2bool, StrOrNone
from inference_residuals import init_args_dataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import h5py

#######################################################################################################################
# Helper functions

def eval_parser() :
    """
    Parser for the evaluation script. The arguments are the following:
    - dataset_path (str): the path to the dataset
    - arch (str): the architecture of the model
    - models (list): the names of the models
    - years (list): the years for which to load the dataset
    - plot_folder (str): the folder where to save the plots
    - mode (str): the mode of the dataset (e.g. 'test')
    - offset (bool): whether to slightly offset the lat/lon values (for debugging purposes)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True, help = 'Path to the dataset')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--years', type = int, nargs = '+', help = 'Year of the dataset')
    parser.add_argument('--plot_folder', type = str, required = True)
    parser.add_argument('--mode', required = True, type = str, help = 'Mode of the dataset')
    parser.add_argument('--offset', required = True, type = str2bool, help = 'Whether to slightly offset the lat/lon.')
    parser.add_argument('--min_offset', type = int, default = 0, help = 'Minimum offset (in km)')
    parser.add_argument('--max_offset', type = int, default = 0, help = 'Maximum offset (in km)')
    parser.add_argument('--return_region', type = str2bool, default = False, help = 'Whether to return the region in the dataset')
    parser.add_argument('--skip_preds', type = str2bool, default = False, help = 'Whether to skip the prediction step')
    parser.add_argument('--lite', type = str2bool, default = False, help = 'Whether to use the AGBD-Lite dataset')
    parser.add_argument('--bs', type = int, default = 32, help = 'Batch size for evaluation')
    parser.add_argument('--drop_overlap', type = str2bool, default = False, help = 'Drop AEF test samples that overlap with train set.')
    parser.add_argument('--region', type = str, default = None, choices = ['NorthAmerica', 'SouthAmerica', 'Africa', 'Europe', 'SouthAsia', 'Australasia'], help = 'Only load data from this region.')
    parser.add_argument('--keep_region', type = str2bool, default = True, help = 'Whether to keep the region in the dataset (only relevant if --region is specified). If False, the specified region will be dropped from the dataset. If True, only the specified region will be kept in the dataset.')
    parser.add_argument('--stats_hold_out_region', type = StrOrNone, required = True, help = 'Region whose stats file to load. Pass "None" for global stats. REQUIRED.')
    parser.add_argument('--stats_keep_region', type = str2bool, required = True, help = 'Whether the stats file is for the kept region (true) or its complement (false). REQUIRED.')
    parser.add_argument('--years_stats', type = StrOrNone, required = True, help = 'Year-string used to pick the AGBD/AEF stats file (e.g. "2019", "2019-2020"). REQUIRED.')
    parser.add_argument('--force', type = str2bool, default = False, help = 'If true, skip reading existing results_file and re-run evaluation.')
    args = parser.parse_args()

    return args, args.dataset_path, args.arch, args.models, args.years, args.plot_folder, args.mode, args.offset, args.min_offset, args.max_offset, args.return_region, args.skip_preds, args.lite, args.bs, args.drop_overlap, args.region, args.keep_region, args.force

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

#######################################################################################################################
# Code execution

if __name__ == '__main__' :

    # Parse the arguments
    args, location, arch, models, years, res_folder, mode, offset, min_offset, max_offset, return_region, skip_preds, lite, bs, drop_overlap, region, keep_region, force = eval_parser()

    if drop_overlap :
        assert mode == 'test', 'Dropping overlaps is only relevant for the test set.'
        assert lite == False, 'Dropping overlaps is only relevant for the full AGBD dataset.'

    # Settings
    set_float32_matmul_precision('high')
    if (location == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset paths come from config.sh at the repo root -- edit that file (or set the
    # matching environment variable) to run this elsewhere.
    local = (location == 'local')
    dataset_path = get_paths(local = local)
    debug = False

    # We get the config for one of the models, either from wandb or -- when
    # AGBD_WANDB_LOOKUP=false in config.sh -- from the offline cache committed to the repo.
    # The cache exists because the checkpoints store a state_dict but no hyper_parameters,
    # so without a config the released weights cannot be rebuilt into a model at all.
    if WANDB_LOOKUP :
        import wandb
        api = wandb.Api()
        wandb_mapping, ckpt_mapping = get_mapping(api, arch)
        wandb_name = wandb_mapping[models[0]]
        cfg = api.run(f'{WANDB_ENTITY}/{arch}/{wandb_name}').config
    else :
        wandb_mapping, ckpt_mapping, cfg_mapping = get_mapping_offline(arch, models)
        cfg = cfg_mapping[models[0]]
    # Don't let training cfg clobber stats decoupling args set on the eval CLI.
    # years_stats / temp_ablation / trained_years are skipped because in older
    # training runs they may be unset, which would silently break stats loading.
    _eval_only = ('stats_hold_out_region', 'stats_keep_region', 'years_stats', 'temp_ablation', 'trained_years')
    for key, value in cfg.items():
        if key in _eval_only: continue
        setattr(args, key, value)
    args = init_args_dataset(args)
    dataset_path['embeddings'] += '/AGBD-Lite' if args.lite else '/AGBD'
    if args.lite and lite: args.lite_eval_big = False
    if args.lite and not lite: args.lite_eval_big = True
    if not offset: min_offset = max_offset = 0
    if region :
        args.hold_out_region = region
        args.keep_region = keep_region
    if args.ensemble and args.film :
        FILM_ENSEMBLE = True
        MEMBER_ID = 0
        assert len(models) == 1, "For FiLM ensemble, we use the same model with different random seeds. Please specify only one model name."
        models = [models[0] for _ in range(args.n_members)]
    else: FILM_ENSEMBLE = False 
    args.drop_overlaps = drop_overlap

    # Load the dataset
    test_dataset = GEDIDataset(paths = dataset_path, years = years, chunk_size = 1, mode = mode, args = args, debug = debug, film = args.film, offset = offset, return_region = return_region, min_offset = min_offset, max_offset = max_offset)
    test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False, num_workers = cpus_per_task, pin_memory = True, prefetch_factor = 4, persistent_workers = True)

    # Evaluate the models
    models_rmses, models_preds = [], []
    if return_region : all_biomes, all_regions = [], []

    # Define the output file name
    output_file_name = f"{arch}_{'_'.join(models)}_{'-'.join([str(year) for year in years])}{'_' + mode if mode != 'test' else ''}{'_skippreds' if skip_preds else ''}{'_lite' if lite else ''}{'_nooverlap' if drop_overlap else ''}{'_' + region if region else ''}.h5"
    if exists(join(res_folder, output_file_name)) and not force :

        with h5py.File(join(res_folder, output_file_name), 'r') as f:
            # Check if the file contains the model_rmses dataset
            if 'model_rmses' in f:
                # If it does, we assume the evaluation has already been done and we can just print the results
                model_rmses = f['model_rmses'][:]
                mean_rmse = np.mean(model_rmses)
                std_rmse = np.std(model_rmses)
                print(f'Ensemble test RMSE: {mean_rmse:.2f}±{std_rmse:.2f}')
                exit(0)

    if not isdir(res_folder): mkdir(res_folder)
    results_file = join(res_folder, output_file_name)
    # Temporary file to store ensemble predictions before averaging
    preds_file = join(res_folder, f"{output_file_name.removesuffix('.h5')}_preds.dat")
    num_samples = len(test_dataset)
    preds_storage = np.memmap(preds_file, dtype = 'float32', mode = 'w+', shape = (num_samples, len(models)))

    with h5py.File(results_file, 'w') as res_f:

        # Create datasets to store predictions and labels
        res_f.create_dataset('predictions', shape = (num_samples,), chunks = (bs,), dtype = 'float32')
        res_f.create_dataset('labels', shape = (num_samples,), chunks = (bs,), dtype = 'float32')
        res_f.create_dataset('biomes', shape = (num_samples,), chunks = (bs,), dtype = 'uint8')
        res_f.create_dataset('regions', shape = (num_samples,), chunks = (bs,), dtype = 'uint8')

        for m, model_name in enumerate(models):

            print(f'Evaluating model {model_name}...')

            model_preds, model_labels = [], []

            # Initialize the model
            model = Net(model_name = args.arch, in_features = args.in_features, num_outputs = args.num_outputs, 
                downsample = None,
                patch_size = args.patch_size,
                local = (args.dataset_path == 'local'), device = device, biome_dim = args.biome_dim, emb_dim = args.emb_dim,
                num_sepconv_blocks = args.num_sepconv_blocks, 
                num_sepconv_filters = args.num_sepconv_filters, long_skip = args.long_skip, only_entry = args.only_entry,
                linear_emb = args.linear_emb, padding_mode = args.padding_mode, returns = args.returns)
            
            model = Model(model, lr = args.lr, step_size = args.step_size, gamma = args.gamma, 
                    patch_size = args.patch_size, downsample = args.downsample, 
                    loss_fn = args.loss_fn, film = args.film, l2 = args.l2, crop = args.crop)
        
            # Get the ckpt path from wandb, or from config.sh when the lookup is disabled.
            # The two layouts differ because the training launchers do: AGBD_CLUSTER_CKPT
            # is the parent of weights/, while AGBD_LOCAL_CKPT already is the weights dir.
            if local : ckpt_path = join(dataset_path['ckpt'], arch)
            elif not WANDB_LOOKUP : ckpt_path = join(dataset_path['ckpt'], 'weights', arch)
            else: ckpt_path = ckpt_mapping[model_name]
            
            # Load the weights
            state_dict = torch.load(join(ckpt_path, f'{model_name}_best.ckpt'), map_location = torch.device(device), weights_only = True)['state_dict']
            # Only keep the student model weights
            state_dict = {k:v for k,v in state_dict.items() if 'teacher' not in k}
            # add the following line if the model is not compiled : state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
            model.load_state_dict(state_dict) 
            
            # Move the model to the GPU and set it to evaluation mode
            model.to(device)
            model.eval()
            model.model.eval()
            model = model.model

            with torch.no_grad():

                print('Total number of steps: ', len(test_loader))
                for i, batch in enumerate(test_loader) :

                    # if i % 200 == 0 : print(f'Step {i}')

                    # Parse the batch, based on cropping/film layers
                    if return_region : batch, regions = batch[:-1], batch[-1]
                    if args.crop : batch, (gt_x, gt_y) = batch[:-1], batch[-1]
                    else: gt_x = gt_y = cfg['patch_size'][0] // 2
                    if args.film :
                        images, biomes, biome_embs, labels = batch
                        if FILM_ENSEMBLE:
                            biome_embs = torch.full_like(biome_embs, 0.0)
                            biome_embs[:, MEMBER_ID] = 1.0
                        images, biome_embs = images.to(device), biome_embs.to(device)
                        predictions = model((images, biome_embs))
                    else:
                        images, biomes, labels = batch
                        images = images.to(device)
                        predictions = model(images)
                    predictions = predictions.cpu().detach().numpy()[:, 0, gt_x, gt_y]
                    
                    # Save the results in the memmap
                    batch_size = len(labels)
                    start_idx = i * bs
                    end_idx = start_idx + batch_size
                    preds_storage[start_idx:end_idx, m] = predictions

                    if m == 0 :
                        res_f['labels'][start_idx:end_idx] = labels.numpy()
                        res_f['biomes'][start_idx:end_idx] = biomes.numpy().astype('uint8')
                        res_f['regions'][start_idx:end_idx] = regions.numpy().astype('uint8')
                
            # Now calculate the test RMSE for this model
            test_preds = preds_storage[:, m]
            test_labels = res_f['labels'][:]
            test_rmse = np.sqrt(np.mean(np.power(test_preds - test_labels, 2)))
            del test_preds, test_labels
            print(f'> Model #{m+1} {mode} RMSE: {test_rmse:.2f}')
            models_rmses.append(test_rmse)

            if FILM_ENSEMBLE: MEMBER_ID += 1
        
        # Calculate the ensemble RMSE
        mean_rmse = np.mean(models_rmses)
        std_rmse = np.std(models_rmses)
        print(f'Ensemble test RMSE: {mean_rmse:.2f}±{std_rmse:.2f}')
        res_f.create_dataset('model_rmses', data = np.array(models_rmses), dtype = 'float32')

        # Calculate the average preds across models
        res_f['predictions'][:] = np.mean(preds_storage, axis = 1)
        
        # Delete preds_storage
        preds_storage.flush()
        if hasattr(preds_storage, '_mmap'): preds_storage._mmap.close()
        del preds_storage
        if os.path.exists(preds_file): os.remove(preds_file)