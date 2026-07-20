"""

Check that each GFM encoder's pretrained weights ACTUALLY load.

WHY THIS EXISTS
    Every PANGAEA encoder loads its checkpoint with `load_state_dict(..., strict=False)`.
    That means a checkpoint which matches NOTHING loads "successfully" -- the mismatch is
    downgraded to a log warning, and training proceeds from random initialisation. Two of
    our encoders were silently doing exactly that:

      - prithvi : upstream's loader compares the model's parameter names against the
                  checkpoint's, but the IBM checkpoint prefixes every key with "encoder."
                  So 148/148 parameters "missing" -> nothing loaded. (prithvi2_encoder.py
                  strips that prefix; prithvi_encoder.py does not. Same bug in 2.0.)
      - terramind_optical_tiny : the file on disk was a DINO training checkpoint
                  (student/teacher/dino_loss), not TerraMind at all -> 149/149 missing.

    Reading logs is not enough: "Loaded encoder weights successfully" is never emitted, so
    absence of a warning proves nothing, and each encoder reports differently (or not at
    all). This script measures the thing that matters directly: it snapshots every
    parameter, calls the real load path, and reports what fraction of parameters actually
    CHANGED. A tensor that did not change did not receive pretrained values.

HOW TO READ THE OUTPUT
    loaded% ~100  -> good.
    loaded% ~0    -> the encoder is training from random init; the run is invalid.
    A few unchanged parameters are normal (fixed sin-cos position embeddings are
    initialised deterministically and can legitimately match), so anything below ~90% is
    worth investigating rather than only exact zeros.

Run (from inside the pangaea-bench fork, with its env):
    conda activate pangaea-bench
    python /path/to/AGBD-GFMs/benchmark_pangaea/check_encoder_weights.py
    python .../check_encoder_weights.py --encoder prithvi terramind_optical_tiny

env: pangaea-bench

"""

###################################################################################################
# Imports

import argparse
import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

# The encoders the AGBD-GFM paper actually uses. `thor` is excluded: it is not in
# `working_ones` in any of the benchmark_pangaea/*/gen.py generators.
ENCODERS = [
    'croma_optical',
    'dofa',
    'gfmswin',
    'prithvi',
    'prithvi2_100m',
    'remoteclip',
    'satlasnet_si',
    'scalemae',
    'spectralgpt',
    'ssl4eo_moco',
    'terramind_optical_tiny',
]

###################################################################################################
# Helper functions

def build_encoder(name, configs_root, dataset) :
    """
    Instantiate one encoder exactly the way pangaea/run.py does.

    Several encoder configs interpolate values from the dataset (`${dataset.bands}`,
    `${dataset.multi_temporal}`), so loading the encoder yaml on its own raises
    InterpolationKeyError. Composing the full `train` config -- same as the launchers'
    `--config-name=train dataset=agbdlite encoder=<name>` -- resolves them, and has the
    added benefit that this checks the same config the real runs use.

    Args:
    - name (str): the encoder name (= the stem of configs/encoder/<name>.yaml).
    - configs_root (Path): the fork's configs/ directory.
    - dataset (str): the dataset config to compose against.

    Returns:
    - the instantiated encoder.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir = str(configs_root), version_base = None) :
        cfg = compose(config_name = 'train', overrides = [
            f'encoder={name}', f'dataset={dataset}',
            'decoder=reg_upernet', 'preprocessing=reg_resize',
            'criterion=mse', 'task=regression',
        ])
    return instantiate(cfg.encoder)


def check_encoder(name, configs_root, dataset) :
    """
    Instantiate one encoder, run its real weight-loading path, and measure how many of its
    parameters actually received pretrained values.

    The comparison is on parameter VALUES, not on log messages or key names: we clone every
    parameter before loading and compare afterwards. This is what makes the check immune to
    `strict=False` -- a checkpoint that matches nothing simply leaves every tensor untouched.

    Args:
    - name (str): the encoder name (= the stem of configs/encoder/<name>.yaml).
    - configs_root (Path): the fork's configs/ directory.
    - dataset (str): the dataset config to compose against.

    Returns:
    - dict: {'name', 'status', 'loaded', 'total', 'pct', 'note'}.
    """
    if not (configs_root / 'encoder' / f'{name}.yaml').is_file() :
        return {'name': name, 'status': 'ERROR', 'loaded': 0, 'total': 0, 'pct': 0.0,
                'note': f'no config at encoder/{name}.yaml'}

    try :
        encoder = build_encoder(name, configs_root, dataset)
    except Exception as e :
        return {'name': name, 'status': 'ERROR', 'loaded': 0, 'total': 0, 'pct': 0.0,
                'note': f'instantiate failed: {type(e).__name__}: {e}'}

    weights = getattr(encoder, 'encoder_weights', None)
    # satlasnet fetches its own weights through the satlaspretrain_models package, so it has
    # no local checkpoint and nothing here to verify.
    if weights in (None, 'null') :
        return {'name': name, 'status': 'N/A', 'loaded': 0, 'total': 0, 'pct': 0.0,
                'note': 'no encoder_weights (loads via its own library)'}
    if not Path(weights).is_file() :
        return {'name': name, 'status': 'NOFILE', 'loaded': 0, 'total': 0, 'pct': 0.0,
                'note': f'{weights} not found -- fetch it first'}

    before = {n: p.detach().clone() for n, p in encoder.named_parameters()}

    logger = logging.getLogger(f'check.{name}')
    try :
        encoder.load_encoder_weights(logger)
    except Exception as e :
        return {'name': name, 'status': 'ERROR', 'loaded': 0, 'total': 0, 'pct': 0.0,
                'note': f'load_encoder_weights raised: {type(e).__name__}: {e}'}

    changed = sum(1 for n, p in encoder.named_parameters()
                  if not torch.equal(p.detach(), before[n]))
    total = len(before)
    extra = ''

    # Not every encoder loads inside load_encoder_weights(). TerraMind's is literally
    # `pass`: its weights are loaded by build_terrammind_vit() during instantiate(), so by
    # the time we snapshot, they are already in and nothing "changes". Counting changes
    # alone would report a correctly-loaded TerraMind as 0% -- a false alarm. So when
    # nothing changed, fall back to comparing the parameters against the checkpoint's own
    # values: if they are already identical, the weights were loaded earlier, not missing.
    if changed == 0 :
        matched = count_matching_checkpoint(encoder, weights)
        if matched > 0 :
            changed, extra = matched, 'loaded at instantiate(), not in load_encoder_weights()'

    pct = 100.0 * changed / total if total else 0.0

    if pct == 0.0 : status, note = 'BROKEN', 'NOTHING loaded -- random init'
    elif pct < 90.0 : status, note = 'SUSPECT', 'many parameters untouched'
    else : status, note = 'ok', extra

    return {'name': name, 'status': status, 'loaded': changed, 'total': total, 'pct': pct, 'note': note}


def count_matching_checkpoint(encoder, weights) :
    """
    Count parameters whose value is already identical to the checkpoint's.

    Used only as a fallback for encoders that load during construction rather than in
    load_encoder_weights(). Both the raw key and the "encoder."-stripped key are tried,
    because checkpoints differ on that prefix (and mishandling it is exactly the Prithvi
    bug this script exists to catch).

    Args:
    - encoder: the instantiated encoder.
    - weights (str): path to its checkpoint.

    Returns:
    - int: number of parameters value-identical to the checkpoint.
    """
    try :
        ck = torch.load(weights, map_location = 'cpu', weights_only = False)
    except Exception :
        return 0
    if not isinstance(ck, dict) : return 0
    for key in ('model', 'state_dict') :
        if key in ck and isinstance(ck[key], dict) : ck = ck[key]; break

    stripped = {k.replace('encoder.', '', 1): v for k, v in ck.items() if k.startswith('encoder.')}

    matched = 0
    for n, p in encoder.named_parameters() :
        for src in (ck, stripped) :
            v = src.get(n)
            if v is not None and hasattr(v, 'shape') and v.shape == p.shape \
               and torch.equal(p.detach(), v) :
                matched += 1
                break
    return matched

###################################################################################################
# Code execution

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type = str, nargs = '+', default = ENCODERS,
                        help = 'Encoders to check. Default: every encoder the paper uses.')
    parser.add_argument('--config-dir', type = str, default = 'configs',
                        help = 'Path to the fork\'s configs/ (default: relative to cwd, i.e. run '
                               'this from the pangaea-bench checkout).')
    parser.add_argument('--dataset', type = str, default = 'agbdlite',
                        help = 'Dataset config to compose against (some encoder configs '
                               'interpolate ${dataset.bands} etc.).')
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    if not (config_dir / 'encoder').is_dir() :
        sys.exit(f'ERROR: {config_dir}/encoder does not exist. Run this from inside the '
                 f'pangaea-bench fork, or pass --config-dir.')

    # The encoders log their own warnings; silence them so the table stays readable. The
    # measurement does not depend on them -- that is the entire point of this script.
    logging.disable(logging.WARNING)

    print(f'Checking {len(args.encoder)} encoder(s) against {config_dir}\n')
    print(f'{"ENCODER":<26} {"STATUS":<9} {"LOADED":>14}  NOTE')
    print('-' * 78)

    results = []
    for name in args.encoder :
        r = check_encoder(name, config_dir, args.dataset)
        results.append(r)
        loaded = f'{r["loaded"]}/{r["total"]} ({r["pct"]:.0f}%)' if r['total'] else '-'
        print(f'{r["name"]:<26} {r["status"]:<9} {loaded:>14}  {r["note"]}')

    broken = [r for r in results if r['status'] in ('BROKEN', 'SUSPECT')]
    errored = [r for r in results if r['status'] in ('ERROR', 'NOFILE')]

    print()
    if broken :
        print(f'{len(broken)} encoder(s) did NOT load their weights properly:')
        for r in broken : print(f'  - {r["name"]}: {r["loaded"]}/{r["total"]} parameters ({r["pct"]:.0f}%)')
        print('Any run using these trained from random initialisation and must be redone.')
    if errored :
        print(f'{len(errored)} encoder(s) could not be checked: ' + ', '.join(r['name'] for r in errored))
    if not broken and not errored :
        print('All encoders loaded their pretrained weights.')

    sys.exit(1 if broken else 0)
