"""

Configuration loader for the AGBD-GFMs codebase.

Reads `config.sh` -- the same file the shell launchers source -- so that the shell and
Python halves of the pipeline can never disagree about where the data lives. Rather than
re-implementing shell variable expansion (which would drift from config.sh over time),
this actually sources the file with bash and reads back the resulting environment. That
means `${VAR:-default}` overrides behave identically in both languages: setting an
environment variable always wins.

Usage:
    from config import get_paths, WANDB_ENTITY

    paths = get_paths(local = True)   # or local = False for the cluster layout
    paths['h5'], paths['ckpt'], ...

To run on your own machine, edit config.sh, or override a single key without touching it:
    AGBD_LOCAL_H5=/my/patches python train.py ...

"""

###################################################################################################
# Imports

import os
import subprocess
from os.path import join, dirname, abspath, isfile

###################################################################################################
# Loading

CONFIG_SH = join(dirname(abspath(__file__)), 'config.sh')


def _load(config_sh = CONFIG_SH) :
    """
    Source config.sh with bash and capture every AGBD_* variable it defines.

    Sourcing (rather than parsing) is deliberate: config.sh is the single source of truth,
    and bash is the only thing that gets its own expansion rules right.

    Args:
    - config_sh (str): path to config.sh.

    Returns:
    - dict: {variable name: value} for every AGBD_* variable.
    """
    if not isfile(config_sh) :
        raise FileNotFoundError(f'{config_sh} not found. It ships with the repo; do not delete it.')

    # `set -a` exports everything config.sh defines, so `env` sees it. The current
    # environment is passed through, so external overrides still take precedence.
    result = subprocess.run(['bash', '-c', f'set -a; source "{config_sh}"; env'],
                            capture_output = True, text = True, env = os.environ.copy())
    if result.returncode != 0 :
        raise RuntimeError(f'Could not source {config_sh}:\n{result.stderr}')

    cfg = {}
    for line in result.stdout.splitlines() :
        key, sep, value = line.partition('=')
        if sep and key.startswith('AGBD_') : cfg[key] = value
    return cfg


CFG = _load()

# Weights & Biases. The API key is NOT here: wandb reads it from ~/.netrc or $WANDB_API_KEY.
WANDB_ENTITY = CFG.get('AGBD_WANDB_ENTITY', 'gs-tp-biomass')
WANDB_LOOKUP = CFG.get('AGBD_WANDB_LOOKUP', 'true').lower() == 'true'

###################################################################################################
# Paths

def get_paths(local) :
    """
    Return the dataset paths for one of the two profiles.

    The keys match the dicts previously hardcoded in train.py / eval.py / inference_aef.py,
    so they can be swapped in directly.

    Args:
    - local (bool): True for the LOCAL profile, False for the CLUSTER profile.

    Returns:
    - dict: keys 'h5', 'norm', 'map', 'ckpt', 'embeddings', 'aef', 'aef_h5', 'aef_norm',
            'tessera_h5', 'tessera_norm', 'splits', plus 'tiles', 'alos', 'dem', 'lc' and
            'region' when they are defined (raw per-tile rasters; only inference_agbd.py
            needs them, so they are optional and simply absent otherwise).
    """
    prefix = 'AGBD_LOCAL_' if local else 'AGBD_CLUSTER_'
    keys = ['H5', 'NORM', 'MAP', 'CKPT', 'EMBEDDINGS', 'AEF', 'AEF_H5', 'AEF_NORM',
            'TESSERA_H5', 'TESSERA_NORM', 'SPLITS']
    paths = {}
    for k in keys :
        value = CFG.get(prefix + k)
        if value is None : raise KeyError(f'{prefix + k} is not defined in config.sh')
        paths[k.lower()] = value

    # Raw per-tile rasters, needed only by the AGBD-features tile inference (inference_agbd.py).
    # Optional on purpose: they are absent from installations that only train/eval on the h5
    # patches, and every other entry point would otherwise KeyError on them. inference_agbd.py
    # checks for the ones it needs and reports which config.sh variable is missing.
    for k in ['TILES', 'ALOS', 'DEM', 'LC', 'REGION'] :
        value = CFG.get(prefix + k)
        if value is not None : paths[k.lower()] = value

    return paths


def is_local() :
    """
    Whether we are on a local machine or the cluster, using the same rule as config.sh
    (a working directory under /cluster means the cluster). Override with AGBD_ENV.

    Returns:
    - bool: True if local.
    """
    return CFG.get('AGBD_ENV', 'local') != 'cluster'
