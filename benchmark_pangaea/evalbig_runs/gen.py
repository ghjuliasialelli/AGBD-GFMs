
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

# The trained-run directory names in configs.txt predate `configs/encoder/dofa_optical.yaml`,
# which was split out of `dofa.yaml` once the AGBD dataset configs gained a sar modality
# (`dofa.yaml` takes ${dataset.bands} and so is multimodal there). Those runs were optical,
# and their saved config.yaml pins the optical-only band list, so re-running them from
# ckpt_dir stays optical. Only the generated script name is renamed here, to say so.
ENCODER_ALIASES = {'dofa': 'dofa_optical'}

with open('../configs.txt', 'r') as f: configs = [line.strip() for line in f if line.strip()]

for config in configs :

    command = f"""HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 pangaea/run.py --config-name=test ckpt_dir={config}_evalbig"""
    
    parts = config.split('_')
    encoder = "_".join(parts[3 : -3])
    encoder = ENCODER_ALIASES.get(encoder, encoder)

    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    # Write the command to a file
    with open(f"{encoder}.sh", "w") as f:
        f.write(base_text)
        f.write(command)


working_ones = ['croma_optical', 'dofa_optical', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'ssl4eo_moco']
# Directory holding the generated .sh files (this script's own). Used only for the
# printed `sbatch` hint; was previously a hardcoded path into the predecessor
# AGBD-GFM repo. These launchers are run from the pangaea-bench fork.
import os
path_script = os.path.dirname(os.path.abspath(__file__))
for encoder in working_ones:
    print(f"sbatch {path_script}/{encoder}.sh")