"""

Offline stand-in for the Weights & Biases lookup done by eval.py.

eval.py rebuilds each model from the config recorded by the training run, and that config
lives only in wandb: the Lightning checkpoints carry a `state_dict` but no
`hyper_parameters`, so a checkpoint on its own is not enough to reconstruct the network.
Anyone without access to the wandb entity therefore cannot evaluate the released
checkpoints, which is the whole point of releasing them.

This module reads the same information out of a JSON cache committed to the repo, so that
`AGBD_WANDB_LOOKUP=false` gives byte-identical model configs with no network and no
credentials. The cache is produced by export_wandb_cache.py, which must be run by someone
who *does* have wandb access (i.e. us, before release).

Layout, one file per architecture, mirroring what get_mapping() returns:

    model/eval/wandb_cache/<arch>.json
    {
      "entity": "gs-tp-biomass",
      "arch": "nico_film",
      "runs": {
        "<run name>": {"wandb_id": "...", "model_path": "...", "config": {...}}
      }
    }

"""

###################################################################################################
# Imports

import json
from os.path import join, dirname, abspath, isfile

###################################################################################################
# Loading

CACHE_DIR = join(dirname(abspath(__file__)), 'eval', 'wandb_cache')


def cache_path(arch) :
    """
    Path of the cache file for one architecture.

    Args:
    - arch (str): the architecture (= the wandb project), e.g. 'nico_film'.

    Returns:
    - str: the path, which may not exist.
    """
    return join(CACHE_DIR, f'{arch}.json')


def load_cache(arch) :
    """
    Read the cached wandb runs for one architecture.

    Args:
    - arch (str): the architecture (= the wandb project).

    Returns:
    - dict: {run name: {'wandb_id', 'model_path', 'config'}}.
    """
    path = cache_path(arch)
    if not isfile(path) :
        raise FileNotFoundError(
            f'No offline wandb cache for architecture "{arch}" ({path}).\n'
            f'Either set AGBD_WANDB_LOOKUP=true in config.sh to query wandb directly, or\n'
            f'regenerate the cache with:  python model/export_wandb_cache.py --arch {arch}'
        )

    with open(path, 'r') as f : payload = json.load(f)
    return payload['runs']


def get_mapping_offline(arch, models) :
    """
    Offline replacement for eval.py's get_mapping(), restricted to the runs actually asked
    for. get_mapping() silently skips runs it cannot read; here a requested run that is
    absent is an error, because it would otherwise surface much later as a KeyError.

    Args:
    - arch (str): the architecture (= the wandb project).
    - models (list): the run names being evaluated.

    Returns:
    - tuple: (run_mapping, run_ckpt, run_cfg), matching get_mapping() plus the configs.
    """
    runs = load_cache(arch)

    missing = [m for m in models if m not in runs]
    if missing :
        raise KeyError(
            f'Runs {missing} are not in the offline wandb cache for "{arch}" '
            f'({cache_path(arch)}), which holds {len(runs)} run(s).\n'
            f'Regenerate it with:  python model/export_wandb_cache.py --arch {arch}'
        )

    run_mapping = {name : entry['wandb_id'] for name, entry in runs.items()}
    run_ckpt = {name : entry['model_path'] for name, entry in runs.items() if entry.get('model_path')}
    run_cfg = {name : entry['config'] for name, entry in runs.items()}

    return run_mapping, run_ckpt, run_cfg
