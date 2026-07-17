"""

Export the wandb run configs that eval.py needs into a JSON cache committed to the repo.

WHY THIS EXISTS
    The released Lightning checkpoints contain a `state_dict` and nothing else -- no
    `hyper_parameters` -- so the architecture cannot be rebuilt from a checkpoint alone.
    eval.py gets that config from wandb. Anyone outside the wandb entity therefore cannot
    evaluate the released checkpoints. This script snapshots the configs so they can be
    shipped alongside the weights, and eval.py reads them back when
    AGBD_WANDB_LOOKUP=false (see wandb_cache.py).

    It must be run by someone with access to the wandb entity (i.e. us, before release).
    Once the cache is committed, nobody needs wandb again.

Run:
    python export_wandb_cache.py                    # every architecture eval.py uses
    python export_wandb_cache.py --arch nico_film   # just one

env: agbd

"""

###################################################################################################
# Imports

import argparse
import json
import os
import sys
from os.path import join, dirname, abspath, isfile

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from config import WANDB_ENTITY

from wandb_cache import CACHE_DIR, cache_path

# The wandb projects eval.py is pointed at by the committed eval configs
# (model/eval/configs/*.txt and model/eval/runs/*.sh all set arch to one of these).
ARCHS = ['nico_film', 'mlp', 'lp']

###################################################################################################
# Helper functions

def export_arch(api, arch, entity) :
    """
    Snapshot every readable run of one wandb project into the cache file.

    The (name -> id, name -> model_path) extraction mirrors eval.py's get_mapping()
    exactly, including its skip-on-error behaviour, so the offline path cannot disagree
    with the online one. The full run config is stored too, which get_mapping() does not
    return but eval.py separately fetches for models[0].

    Args:
    - api (wandb.Api): the wandb API.
    - arch (str): the architecture (= the wandb project).
    - entity (str): the wandb entity.

    Returns:
    - int: the number of runs written.
    """
    runs = api.runs(f'{entity}/{arch}')

    exported, skipped = {}, 0
    for run in runs :
        try :
            wandb_id = run.path[-1]
            config = dict(run.config)
        except Exception :
            # Same contract as get_mapping(): an unreadable run is skipped, not fatal.
            skipped += 1
            continue

        exported[run.name] = {
            'wandb_id' : wandb_id,
            # get_mapping() requires model_path and drops the run if it is absent; here it
            # is optional, because the offline profile loads checkpoints from AGBD_*_CKPT
            # and never looks at it. Keeping the run means its config stays usable.
            'model_path' : config.get('model_path'),
            'config' : config,
        }

    payload = {'entity' : entity, 'arch' : arch, 'runs' : exported}

    os.makedirs(CACHE_DIR, exist_ok = True)
    path = cache_path(arch)
    with open(path, 'w') as f :
        json.dump(payload, f, indent = 2, sort_keys = True, default = str)

    print(f'  {arch}: {len(exported)} run(s) written to {path}' + (f' ({skipped} unreadable, skipped)' if skipped else ''))
    return len(exported)

###################################################################################################
# Code execution

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type = str, nargs = '+', default = ARCHS, choices = ARCHS,
                        help = 'Architectures (= wandb projects) to export. Default: all of them.')
    parser.add_argument('--entity', type = str, default = WANDB_ENTITY,
                        help = 'wandb entity to read from. Default: AGBD_WANDB_ENTITY from config.sh.')
    args = parser.parse_args()

    import wandb
    api = wandb.Api()

    print(f'Exporting wandb configs from entity "{args.entity}"...')
    total = sum(export_arch(api, arch, args.entity) for arch in args.arch)
    print(f'Done: {total} run(s) cached in {CACHE_DIR}.')
    print('Commit this directory -- it is what lets others evaluate the released checkpoints.')
