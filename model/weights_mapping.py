"""

Helper functions for training, see train.py
This script allows to load the best pre-trained model's weights for the given configuration.

It builds on previously defined scripts, see Models > Baseline > Torben_code. It expects that the
following scripts have been run: 1_wb_api.py and 2_wb_summarizer.py

"""

###################################################################################################
# IMPORTS 

import pandas as pd
from os.path import isfile

###################################################################################################
# Helper functions

def get_name(config):
    """
    This function returns the name of the model based on the configuration.

    Args:
    - config (Namespace): the configuration object

    Returns:
    - name (str): the name of the model
    """
    
    name = ""
    
    if config.lc : name += 'lc' + "_"
    if config.alos : name += 'alos' + "_"
    if config.ch : name += 'ch' + "_"
    if config.dem : name += 'dem' + "_"

    if len(config.bands) == 4: name+= "RGBNIR"
    elif len(config.bands) == 0: pass
    elif len(config.bands) == 12: name += "ALLBANDS"
    
    name += "_" + str(list(config.patch_size)[0])
    return name


def load_results(arch) :
    """
    This function loads the summary of the wandb runs for the pre-trained models of the given
    architecture.

    Args:
    - arch (str): the architecture of the model

    Returns:
    - results (pd.DataFrame): the summary of the results of the pre-trained models
    """

    if 'unet' in arch: arch = 'unet'
    elif 'fcn' in arch: arch = 'fcn'
    elif 'nico' in arch: arch = 'nico'
    else: raise NotImplementedError(f'unknown model name {arch}')

    if not isfile(f'weights/pretrained/{arch}_results.csv') :
        raise FileNotFoundError(f'pretrained model {arch} not found. Run 2_wb_summarizer.py.')
    else: results = pd.read_csv(f'weights/pretrained/{arch}_results.csv')
    
    return results


def get_best_model(config) :
    """
    This function returns the name of the best pre-trained model for the given configuration.

    Args:
    - config (Namespace): the configuration object

    Returns:
    - model (str): the name of the best pre-trained model
    """

    results = load_results(config.arch)
    name = get_name(config)
    model = results[results['Method'] == name].names.values[0].split("'")[1]

    return model


