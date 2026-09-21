# Generates the per-encoder test launchers for ONE fine-tuning regime: the regime is
# this script's own parent directory name (frozen / lora), which also selects the
# run-name list it reads, ../../configs/<regime>.txt. Copy this file into a sibling
# regime directory and it retargets itself; nothing here is hardcoded to `frozen`.

import os

path_script = os.path.dirname(os.path.abspath(__file__))
REGIME = os.path.basename(path_script)
configs_file = os.path.join(path_script, '..', '..', 'configs', f'{REGIME}.txt')

base_text = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:1

"""

# The trained-run directory names in configs/frozen.txt predate
# `configs/encoder/dofa_optical.yaml`, which was split out of `dofa.yaml` once the AGBD
# dataset configs gained a sar modality (`dofa.yaml` takes ${dataset.bands} and so is
# multimodal there). Those runs were optical, and their saved config.yaml pins the
# optical-only band list, so re-running them from ckpt_dir stays optical. Only the
# generated script name is renamed here, to say so.
ENCODER_ALIASES = {'dofa': 'dofa_optical'}


def read_configs(path):
    """Run-directory names, one per line; '#' comments and blank lines skipped."""
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith('#')]


def encoder_of(config):
    """Recover the encoder name from a run-directory name.

    Frozen: <date>_<time>_<hash>_<encoder>_<decoder>_<dataset>
    LoRA:   <date>_<time>_<hash>_<encoder>_lora_<decoder>_<dataset>
    `decoder` is two tokens (reg_upernet) and `dataset` one, so dropping the first
    three and last three fields leaves the encoder -- plus the `_lora` token that
    run.py appends for the LoRA regime, which is stripped here.
    """
    encoder = "_".join(config.split('_')[3:-3])
    if encoder.endswith('_lora'):
        encoder = encoder[:-len('_lora')]
    return ENCODER_ALIASES.get(encoder, encoder)


configs = read_configs(configs_file)
if not configs:
    raise SystemExit(
        f"No run names in {os.path.normpath(configs_file)} -- nothing to generate.\n"
        f"Collect the {REGIME} training-run directory names into that file first."
    )

encoders = []
for config in configs:

    command = f"""torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 pangaea/run.py --config-name=test ckpt_dir={config}"""

    encoder = encoder_of(config)
    encoders.append(encoder)

    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    # Write the command to a file, next to this generator
    with open(os.path.join(path_script, f"{encoder}.sh"), "w") as f:
        f.write(base_text)
        f.write(command)

# The sbatch hints: one per generated launcher. These are run from the pangaea-bench fork.
print(f"# {REGIME}: {len(encoders)} launchers")
for encoder in encoders:
    print(f"sbatch {path_script}/{encoder}.sh")
