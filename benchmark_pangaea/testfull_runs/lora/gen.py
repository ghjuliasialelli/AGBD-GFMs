# Generates the FULL-on-FULL test launchers for ONE fine-tuning regime: evaluate a model
# TRAINED on full AGBD (../../trainfull_runs/) on the full AGBD-test set. The regime is
# this script's own parent directory name (frozen / lora), which also selects the
# run-name list it reads, ../../configs/full_<regime>.txt. Copy this file into a sibling
# regime directory and it retargets itself.
#
#   train_runs/     AGBD-Lite -> (tested on) AGBD-Lite   test_runs/
#                   AGBD-Lite -> AGBD-test               evalbig_runs/
#   trainfull_runs/ AGBD      -> AGBD-test               testfull_runs/   <- this
#
# No `_evalbig` copy is needed (unlike evalbig_runs/): a dataset=agbd run already tests
# on AGBD-test. But the saved config pins dataset.root_path_cluster to the TRAINING job's
# $TMPDIR, and run.py reloads that config from ckpt_dir, so a CLI override would be lost.
# The generated launcher therefore refuses to start until that path has been fixed.

import os

path_script = os.path.dirname(os.path.abspath(__file__))
REGIME = os.path.basename(path_script)
configs_file = os.path.join(path_script, '..', '..', 'configs', f'full_{REGIME}.txt')

# Where the full AGBD data lives for the test job (no $TMPDIR staging here).
ROOT_PATH_CLUSTER = '/cluster/scratch/gsialelli'

base_text = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

"""

guard_text = """# The run was trained with dataset.root_path_cluster=${{TMPDIR}}; the test job reads from
# {root}. Refuse to start on a stale path rather than fail mid-load.
config_yaml={run}/configs/config.yaml
if ! grep -qE "root_path_cluster: +{root} *$" "$config_yaml"; then
    echo "root_path_cluster in $config_yaml is not {root}:" >&2
    grep -n "root_path_cluster" "$config_yaml" >&2
    echo "fix it with: perl -i -pe 's|(root_path_cluster: ).*|\\${{1}}{root}|' $config_yaml" >&2
    exit 1
fi

"""


def read_configs(path):
    """Run-directory names, one per line; '#' comments and blank lines skipped."""
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith('#')]


def encoder_of(config):
    """Recover the encoder name from a run-directory name.

    Frozen: <date>_<time>_<hash>_<encoder>_<decoder>_<dataset>
    LoRA:   <date>_<time>_<hash>_<encoder>_lora_<decoder>_<dataset>
    See ../../test_runs/frozen/gen.py.
    """
    encoder = "_".join(config.split('_')[3:-3])
    if encoder.endswith('_lora'):
        encoder = encoder[:-len('_lora')]
    return encoder


configs = read_configs(configs_file)
if not configs:
    raise SystemExit(
        f"No run names in {os.path.normpath(configs_file)} -- nothing to generate.\n"
        f"Train with ../../trainfull_runs/{REGIME}/ and add the run-directory name there first."
    )

# A Lite-trained run tested on AGBD-test is evalbig_runs/, not this. Refuse to mix them up.
not_full = [c for c in configs if c.split('_')[-1] != 'agbd']
if not_full:
    raise SystemExit(
        f"{os.path.normpath(configs_file)} lists runs not trained on full AGBD "
        f"(dataset token != 'agbd'):\n  " + "\n  ".join(not_full) +
        "\nThose belong in configs/<regime>.txt and evalbig_runs/."
    )

encoders = []
for config in configs:

    command = f"""HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py --config-name=test ckpt_dir={config}
"""

    encoder = encoder_of(config)
    encoders.append(encoder)

    print()
    print("Encoder: ", encoder)
    print(command)

    with open(os.path.join(path_script, f"{encoder}.sh"), "w") as f:
        f.write(base_text)
        f.write(guard_text.format(run=config, root=ROOT_PATH_CLUSTER))
        f.write(command)

# The sbatch hints: one per generated launcher. These are run from the pangaea-bench fork.
print(f"# full-on-full, {REGIME}: {len(encoders)} launchers")
for encoder in encoders:
    print(f"sbatch {path_script}/{encoder}.sh")
