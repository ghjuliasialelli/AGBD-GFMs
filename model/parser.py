"""

Set-up of argument parsing for models training. Defines the `setup_parser()` function, which returns an `ArgumentParser()`
object containing the command-line arguments.

"""

import argparse

def str2bool(v):
    """ 
    Helper function to parse a string into a boolean.
    
    Args:
     - v (str) : input string to be parsed
    
    Returns:
     - bool : parsed boolean value
    """
    if v in ['true', 'True', 'TRUE']: return True
    elif v in ['false', 'False', 'FALSE']: return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


def strOrFloat(v):
    """
    Helper function to parse a string into a float if possible, otherwise return the string.
    
    Args:
     - v (str) : input string to be parsed
    
    Returns:
     - float or str : parsed float value or original string
    """
    try: return float(v)
    except: return v

def StrOrNone(v):
    """
    Helper function to parse a string if possible, otherwise return None.
    
    Args:
     - v (str) : input string to be parsed
    
    Returns:
     - str or None : parsed string value or None
    """
    if v in ['None', 'none', 'NONE']: return None
    else: return v

def setup_parser():
    """ 
        Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required = True, type = str, help = 'Path to the folder where models should be saved.')
    parser.add_argument('--model_name', required = True, type = str, help = 'Model name, used as the .pth filename to save it.')

    parser.add_argument('--num_gpus', required = True, type = int, help = 'Number of GPUs to use for training, in total.')
    parser.add_argument('--num_cpus', required = True, type = int, help = 'Number of CPUs to use for training, per process.')

    # Dataset #################################################################################################################################################
    parser.add_argument("--dataset_path", type = str, required = True, help = 'Path to the dataset.')
    parser.add_argument("--augment", type = str2bool, default = 'false', help = 'Whether to perform data augmentation.')
    parser.add_argument("--norm", type = str2bool, default = 'false', help = 'Whether to normalize the agbd.')
    parser.add_argument("--chunk_size", type = int, default = 1, help = 'Internal chunk size of the hdf5.')

    parser.add_argument("--lite", type = str2bool, default = 'false', help = 'Whether to use the Lite version of the dataset.')
    parser.add_argument("--lite_eval_big", type = str2bool, default = 'false', help = 'Whether to use the normal test set even when training on the Lite dataset.')
    parser.add_argument("--lite_chunk_size", type = int, default = 32, help = 'Chunk size to use when training on the Lite dataset.')

    parser.add_argument("--aef", type = str2bool, default = 'false', help = 'Whether to include the AEF patches in the input.')
    parser.add_argument("--tessera", type = str2bool, default = 'false', help = 'Whether to include the TESSERA embeddings in the input. Currently only supported with --lite.')
    parser.add_argument("--drop_overlaps", type = str2bool, default = 'false', help = 'Whether to drop the AGBD test patches that are in the AEF train set.')

    parser.add_argument("--hold_out_region", type = StrOrNone, default = None, help = 'Whether to hold out a region for testing.')
    parser.add_argument("--keep_region", type = str2bool, default = 'false', help = 'Whether to keep only the hold-out region instead of discarding it.')
    # Stats-file selection is decoupled from sample filtering. Both are REQUIRED:
    # callers (train + eval scripts) must pass them explicitly. No fallback.
    parser.add_argument("--stats_hold_out_region", type = StrOrNone, required = True, help = 'Region whose stats file to load. Pass "None" for global stats.')
    parser.add_argument("--stats_keep_region", type = str2bool, required = True, help = 'Whether the stats file is for the kept region (true) or its complement (false).')

    parser.add_argument("--predict", type = str, default = 'agbd', help = 'Whether to predict agbd, rh98 or biome.')

    # Model ###################################################################################################################################################
    
    parser.add_argument("--arch",   required = True, type = str, help = 'Network architecture.')
    parser.add_argument("--model_idx", type = int, default = 0, help = 'Model ID, within the ensemble.')
    parser.add_argument("--loss_fn", required = True, type = str, help = 'Which loss function to use for the training. Can be: `RMSE`, `GNLL`, `LNLL`. Not considered if `mt_weighting` is `uncertainty`.')
    parser.add_argument("--film", type = str2bool, default = 'false', help = 'Whether to do FiLM.')

    # inputs
    parser.add_argument("--latlon", type = str2bool, default = 'true', help = 'Whether to include `lon_1` and `lon_2` in the input features.')
    parser.add_argument("--ch", type = str2bool, default = 'false', help = 'Whether to include the `ch` and `ch_std` patches in the input.')
    parser.add_argument("--bands", type = str, nargs = '*', help = 'Sentinel-2 bands (e.g., `B12`) to consider as input for the model.' )
    parser.add_argument("--in_features", required = True, type = int, help = 'Number of features provided as input to the model.')
    parser.add_argument("--s1", type = str2bool, default = 'false', help = 'Whether to include the S1 patches in the input.')
    parser.add_argument("--alos", type = str2bool, default = 'false', help = 'Whether to include the ALOS patches in the input.')
    parser.add_argument("--lc", type = str2bool, default = 'false', help = 'Whether to include the LC patches in the input.')
    parser.add_argument("--dem", type = str2bool, default = 'false', help = 'Whether to include the DEM patches in the input.')
    parser.add_argument("--gedi_dates", type = str2bool, default = 'false', help = 'Whether to include the GEDI dates in the input.')
    parser.add_argument("--s2_dates", type = str2bool, default = 'false', help = 'Whether to include the S2 date and DOY in the input.')
    parser.add_argument("--s2_day", type = str2bool, default = 'false', help = 'Whether to include the S2 date in the input.')
    parser.add_argument("--s2_doy", type = str2bool, default = 'false', help = 'Whether to include the S2 DOY in the input.')
    parser.add_argument("--topo", type = str2bool, default = 'false', help = 'Whether to include more topological information (slope and aspect).')
    parser.add_argument("--aspect", type = str2bool, default = 'false', help = 'Whether to include the aspect.')
    parser.add_argument("--slope", type = str2bool, default = 'false', help = 'Whether to include the slope.')
    parser.add_argument("--ft_cat2vec", type = str2bool, default = 'false', help = 'Whether to use the cat2vec embeddings for the LC data.')
    parser.add_argument("--ft_onehot", type = str2bool, default = 'false', help = 'Whether to use the one-hot encoding for the LC data.')
    parser.add_argument("--ft_sincos", type = str2bool, default = 'false', help = "Whether to use sine/cosine embeddings for the LC data.")

    # Whether to randomly drop the S2 data
    parser.add_argument("--train_mask", type = str2bool, default = 'false', help = 'Whether to randomly drop the S2 data.')
    parser.add_argument("--val_mask", type = str2bool, default = 'false', help = 'Whether to randomly drop the S2 data.')
    parser.add_argument("--test_mask", type = str2bool, default = 'false', help = 'Whether to randomly drop the S2 data.')

    # whether to log transform the AGB
    parser.add_argument("--log_transform", type = str2bool, default = 'false', help = "Whether to log transform the AGB values.")

    # whether to over-sample from the minority AGB bins
    parser.add_argument("--oversampling", type = str2bool, default = 'false', help = "Whether to log transform the AGB values.")

    # embeddings for the FiLM layers
    parser.add_argument("--emb_cat2vec", type = str2bool, default = 'false', help = 'Whether to use the cat2vec embeddings for the biome (for FiLM).')
    parser.add_argument("--emb_onehot", type = str2bool, default = 'false', help = 'Whether to use the one-hot encoding for the biome (for FiLM).')
    parser.add_argument("--emb_dist", type = str2bool, default = 'false', help = "Whether to encode the patch's biomes distribution (for FiLM).")
    parser.add_argument("--emb_sincos", type = str2bool, default = 'false', help = "Whether to use sine/cosine embeddings for the biome (for FiLM).")

    # residuals
    parser.add_argument("--residuals", type = str2bool, default = 'false', help = 'Whether to include CHM - GEDI residuals.')
    parser.add_argument("--res_norm", type = str2bool, default = 'false', help = 'Normalize the residuals (instead of computing them from normalized CH and RH98).')
    parser.add_argument("--res_film", type = str2bool, default = 'false', help = 'Give the central pixel offset to the FiLM layers for conditioning.')
    parser.add_argument("--res_in", type = str2bool, default = 'false', help = 'Give the patch residuals to the model as input.')
    parser.add_argument("--res_in_central", type = str2bool, default = 'false', help = "Compute the patch's residuals as CH_central - RH98_central.")
    parser.add_argument("--res_in_patch", type = str2bool, default = 'false', help = "Compute the patch's residuals as CH_patch - RH98_central.")

    # FiLM arguments
    parser.add_argument("--biome_dim", type = int, default = 128, help = 'Biome embedding dimension for FiLM.')
    parser.add_argument("--emb_dim", type = int, help = 'Dimensionality of the biome embeddings (2 for sine/cosine, 5 for cat2vec).')
    parser.add_argument("--linear_emb", type = str2bool, default = 'false', help = 'Whether to use a Linear layer instead of an MLP for the embeddings.')
    parser.add_argument("--region", type = str2bool, default = 'false', help = 'Whether to put the GEDI region_cla in the FiLM layers.')
    parser.add_argument("--biome", type = str2bool, default = 'false', help = 'Whether to put the biome in the FiLM layers.')
    parser.add_argument("--rh98_film", type = str2bool, default = 'false', help = 'Whether to give the RH98 to the FiLM layers.')
    parser.add_argument("--ensemble", type = str2bool, default = 'false', help = 'Whether to do ensemble FiLM.')
    parser.add_argument("--n_members", type = int, default = 1, help = 'Number of members in the ensemble FiLM.')

    # outputs 
    parser.add_argument("--num_outputs", required = True, type = int, help = 'Number of features outputed by the model.')
    parser.add_argument("--norm_strat", type = str, required = True, help = 'Normalization strategy, one of `pct`, `min_max` and `mean_std`.')
    parser.add_argument("--prob_norm", type = str2bool, default = 'false', help = 'Whether to normalize lc_prob and the slope.')

    # teacher student architecture
    parser.add_argument("--teacher", type = str, default = '', help = 'Teacher model to load.')
    parser.add_argument("--teacher_arch", type = str, default = 'nico_film', help = 'Teacher model architecture.')
    parser.add_argument("--teacher_inpaint", type = str2bool, default = 'false', help = 'Whether to inpaint the central pixel GT in the dense labels.')
    parser.add_argument("--ndvi", type = str2bool, default = 'false', help = 'NDVI sanity check setting.')
    parser.add_argument("--crop", type = str2bool, default = 'false', help = 'Crop the 25x25 patches to 15x15 patches.')
    parser.add_argument("--slide", type = str2bool, default = 'true', help = 'Crop the 25x25 patches to 15x15 patches.')

    # Dense/sparse supervision
    # Quality approach
    parser.add_argument("--quality", type = str2bool, default = 'false', help = 'Use QualityLoss.')
    parser.add_argument("--mixed", type = str, default = 'false', help = 'Whether to use mixed loss for the Student training.')
    parser.add_argument("--cutoff", default = 0.5, type = float, help = 'Cut-off for mixed learning.')
    parser.add_argument("--mixed_version", default = 1, type = int, help = 'Mixed loss version.')
    # Balanced approach
    parser.add_argument("--balanced", type = str2bool, default = 'false', help = 'Use BalancedLoss.')
    parser.add_argument("--_lambda", type = strOrFloat, default = 'N/A', help = 'How to balance sparse and dense supervision.')
    parser.add_argument("--filter", type = str, default = 'N/A', help = 'How to filter the pseudo-labels.')
    # Pseudo labels only
    parser.add_argument("--pl_only", type = str2bool, default = 'false', help = 'Only train on the dense labels.')
    # Conditioning the output on the CH distribution
    parser.add_argument("--sim_dist", type = str2bool, default = 'false', help = 'Output and CH must have a similar distribution.')
    parser.add_argument("--similarity", type = str, default = 'N/A', help = 'Similarity measure to use.')
    parser.add_argument("--similarity_weight", type = float, default = 1.0, help = 'Weight of the similarity loss.')
    parser.add_argument("--SCC_ws", type = int, default = 8, help = 'Window size for the SCC similarity measure.')
    parser.add_argument("--SCC_softmax", type = str2bool, default = 'false', help = 'Apply softmax to the images before SCC.')

    # AGB residuals
    parser.add_argument("--agb_residuals", type = str2bool, default = 'false', help = 'Use AGB residuals.')
    parser.add_argument("--agb_residuals_file", type = str, default = 'N/A', help = 'File from which to read the AGB residuals statistics.')
    parser.add_argument("--agb_res_all", type = str2bool, default = 'false', help = 'Use all available statistics.')
    parser.add_argument("--agb_res_one", type = str, default = 'N/A', help = 'Which statistic to use.')
    parser.add_argument("--agb_residuals_film", type = str2bool, default = 'false', help = 'Use AGB residuals for FiLM layer.')


    # Training ################################################################################################################################################

    # debug flags
    parser.add_argument("--debug_latlon", type = str2bool, default = 'false', help = 'Whether to calculate the lat/lon properly in the Dataset().')
    parser.add_argument("--new_stats", type = str2bool, default = 'false', help = 'Whether to use the newly computed statistics.')

    # model
    parser.add_argument("--n_epochs", default = 100, type = int, help = 'Number of epochs.')
    parser.add_argument("--limit", type = str2bool, default = 'false', help = 'Whether to limit the number of batches to process at each epoch.')
    parser.add_argument("--batch_size", default = 256, type = int, help= 'Batch size.')
    parser.add_argument("--years", type = int, nargs = '+', help = 'Year of the dataset.')
    parser.add_argument("--trained_years", type = int, nargs = '+', help = 'Years used for training.')
    parser.add_argument("--temp_ablation", type = str2bool, default = 'false', help = 'Whether to do the temporal ablation experiments.')
    parser.add_argument("--subsample_2020", type = str2bool, default = 'false', help = 'Whether to subsample the 2020 data to the same number of samples as 2019.')
    parser.add_argument("--years_stats", type = StrOrNone, default = 'None', help = 'Years to consider for the normalization statistics, in the format "2019-2020". If None, it will be inferred from the years or trained_years arguments.')
    parser.add_argument("--sigreg_lambda", type = float, default = 0.0, help = 'Weight for the SigREG loss.')

    # FCN model arguments
    parser.add_argument("--downsample", type = str2bool, default = 'false', help = 'Whether to downsample the patches from 10m resolution to 50m resolution.')
    
    # UNet model arguments

    # Nico model arguments
    parser.add_argument("--num_sepconv_blocks", type = int, default = 8, help = 'Number of sepconv blocks.')
    parser.add_argument("--num_sepconv_filters", type = int, default = 728, help = 'Number of sepconv filters.')
    parser.add_argument("--long_skip", type = str2bool, default = 'false', help = 'Whether to long-skip.')
    parser.add_argument("--only_entry", type = str2bool, default = 'false', help = 'Whether to long-skip.')
    parser.add_argument("--returns", type = str, default = 'dense', help = 'Whether to return dense or pixel-wise predictions.')

    # common to all
    parser.add_argument("--padding_mode", type = str, default = 'zeros', help = 'Padding mode to apply (defaults to zeros).')

    # optimizer & scheduler
    parser.add_argument("--lr", default = 1e-4, type = float, help = 'Learning rate.')
    parser.add_argument("--step_size", default = 30, type = int, help = 'Period of learning rate decay.')
    parser.add_argument("--gamma", default = 0.1, type = float, help = 'Multiplicative factor of learning rate decay.')
    parser.add_argument("--l2", default = 0.0, type = float, help = 'Weight decay (L2 penalty).')

    # early stopping
    parser.add_argument("--patience", default = 3, type = int, help = 'Number of checks with no improvements after which training will be stopped.')
    parser.add_argument("--min_delta", default = 0.0, type = float, help = 'Minimum change in the monitored quantity to qualify as improvement.')

    # re-balancing
    parser.add_argument("--reweighting", type = str, help = 'Method to be used for samples weights reweighting.')
    
    # Predict #################################################################################################################################################
    
    parser.add_argument("--tile_name", type = str, help = 'Path to the tile on which to run the prediction.')
    parser.add_argument("--clip", type = str2bool, default = 'true', help = 'Whether to clip AGBD values to the [0, 500] range.')
    parser.add_argument('--output_path', type = str, help = 'Path to the folder where predictions should be saved.')

    # ensemble
    parser.add_argument("--n_models", type = int, help = 'Number of models to train as an ensemble.')
    parser.add_argument('--patch_size', help = 'Size of the patches to extract, in pixels.', nargs = 2, type = int, default = [25, 25])

    return parser


def check_args(args) :

    valid = True

    # Check that only one of the ft_... flags is set to True
    if args.lc :
        if sum([args.ft_onehot, args.ft_cat2vec, args.ft_sincos]) == 1: 
            valid = True
        else:
            valid = False
            print("More than one is True or none is True.")
    
    if args.film :
        # Check that only one of the emb_... flags is set to True
        if sum([args.emb_onehot, args.emb_cat2vec, args.emb_dist, args.emb_sincos]) == 1:
            valid = True
        else:
            valid = False
            print("More than one is True or none is True.")

        # Check that either (region or biome) or ensemble is True, but not both
        if (args.region or args.biome) and args.ensemble: 
            valid = False
            print("If region or biome is True, then ensemble must be False.")
        if args.ensemble:
            if args.n_members < 2:
                valid = False
                print("If ensemble is True, then n_members must be at least 2.")
    else:
        if args.region or args.biome or args.ensemble:
            valid = False
            print("If film is False, then region, biome and ensemble must be False.")
    
    # Check that if loss is GNLL, then _gaussian needs to be in arch
    if args.loss_fn == 'GNLL' and ('gaussian' not in args.arch) : 
        valid = False
        print("If loss function is GNLL, then architecture must be gaussian.")
    
    # Check that if aspect or slope or dem are true, then topo must be true
    if (args.aspect or args.slope or args.dem) and not args.topo:
        valid = False
        print("If aspect or slope or dem are true, then topo must be true.")

    # Check that if s2_day or s2_doy are true, then s2_dates must be true
    if (args.s2_day or args.s2_doy) and not args.s2_dates:
        valid = False
        print("If s2_day or s2_doy are true, then s2_dates must be true.")

    # Check residuals
    if args.residuals and not (args.res_film or args.res_in):
        valid = False
        print("If residuals is true, then either res_film or res_in needs to be true.")
    if args.res_in and not (args.res_in_central or args.res_in_patch):
        valid = False
        print("If res_in is true, then either res_in_central or res_in_patch needs to be true.")

    # Check patch sizes
    if args.patch_size[0] != args.patch_size[1]:
        valid = False
        print("Patch size must be square.")
    
    # Check that only one of the quality and balanced and pl_only flags is set to True
    #if sum([args.quality, args.balanced, args.pl_only]) != 1: 
    #    valid = False
    #    print("More than one is True or none is True.")

    return valid