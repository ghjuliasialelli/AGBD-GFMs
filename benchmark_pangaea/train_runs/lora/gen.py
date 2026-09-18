# Same runs as ../frozen, but with LoRA fine-tuning of the encoder:
# finetune=true lora=default -> the encoder's attention projections get low-rank
# adapters (configs/lora/default.yaml), everything else in the encoder stays frozen.
# Run directories are named <encoder>_lora_<decoder>_<dataset>.


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


# The optical-only benchmark: the 11 encoders of the paper, plus thor.
OPTICAL_ENCODERS = ['croma_optical', 'dofa_optical', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'spectralgpt', 'ssl4eo_moco', 'terramind_optical_tiny', 'prithvi2_100m', 'thor']

# The encoders that also take the AGBD SAR modality (ALOS-PALSAR-2 HH/HV, listed as VV/VH
# in the dataset configs). croma_joint and terramind_tiny read the SAME weight files as
# their optical counterparts -- one multimodal checkpoint, of which each class loads a
# different subset. dofa_joint / dofa_optical name the two DOFA variants explicitly:
# upstream dofa.yaml takes ${dataset.bands}, which is ambiguous now that the AGBD dataset
# configs carry SAR, and would give both variants the same run-directory name.
SAR_ENCODERS = ['croma_joint', 'terramind_tiny', 'dofa_joint']

import os
path_script = os.path.dirname(os.path.abspath(__file__))

for encoder in OPTICAL_ENCODERS + SAR_ENCODERS :

    command = f"""torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 pangaea/run.py  --config-name=train  dataset=agbdlite  encoder={encoder}  decoder=reg_upernet  preprocessing=reg_resize  criterion=mse  task=regression finetune=true lora=default batch_size=32 num_workers=6 test_num_workers=6 test_batch_size=32 use_wandb=True task.trainer.eval_interval=1 task.trainer.log_interval=100 task.trainer.eval_interval=1"""
    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    # Write the command to a file
    with open(os.path.join(path_script, f"{encoder}.sh"), "w") as f:
        f.write(base_text)
        f.write(command)


working_ones = ['croma_optical', 'dofa_optical', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'ssl4eo_moco', 'spectralgpt', 'terramind_optical_tiny', 'prithvi2_100m'] + SAR_ENCODERS
# The printed `sbatch` hints. These launchers are run from the pangaea-bench fork.
for encoder in working_ones:
    print(f"sbatch {path_script}/{encoder}.sh")