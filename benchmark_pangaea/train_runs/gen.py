
base_text = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:2

"""


for encoder in ['croma_optical', 'dofa', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'spectralgpt', 'ssl4eo_moco', 'terramind_optical_tiny', 'prithvi2_100m', 'thor'] :

    command = f"""torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 pangaea/run.py  --config-name=train  dataset=agbdlite  encoder={encoder}  decoder=reg_upernet  preprocessing=reg_resize  criterion=mse  task=regression batch_size=32 num_workers=6 test_num_workers=6 test_batch_size=32 use_wandb=True task.trainer.eval_interval=1 task.trainer.log_interval=100 task.trainer.eval_interval=1"""
    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    # Write the command to a file
    with open(f"{encoder}.sh", "w") as f:
        f.write(base_text)
        f.write(command)


working_ones = ['croma_optical', 'dofa', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'ssl4eo_moco', 'spectralgpt', 'terramind_optical_tiny', 'prithvi2_100m']
# Directory holding the generated .sh files (this script's own). Used only for the
# printed `sbatch` hint; was previously a hardcoded path into the predecessor
# AGBD-GFM repo. These launchers are run from the pangaea-bench fork.
import os
path_script = os.path.dirname(os.path.abspath(__file__))
for encoder in working_ones:
    print(f"sbatch {path_script}/{encoder}.sh")