# Multi-temporal runs, LoRA regime.
#
# Same as ../lora, with two differences:
#   dataset=agbdlite-mt   three Sentinel-2 observations per footprint instead of one
#   encoders              only the three that actually consume a temporal axis
#
# LoRA rather than frozen because the multi-temporal inputs shift the distribution the
# encoders were pretrained on -- Prithvi's temporal position embedding and SatlasNet's
# multi-image aggregation both see a T they were not trained at -- so the encoder needs
# to be able to move. Run directories are named <encoder>_lora_<decoder>_<dataset>.

import os
path_script = os.path.dirname(os.path.abspath(__file__))

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

# The encoders that take a temporal axis natively:
#   prithvi / prithvi2_100m  num_frames: ${dataset.multi_temporal} -> the 3D patch embed
#                            tokenises (B, C, T, H, W) directly.
#   satlasnet_mi             the Multi-Image checkpoint; AggregationBackbone splits the
#                            flat channel axis into T images and max-pools across them.
#                            It skips group indices beyond the frames supplied, so T=3
#                            works against a model whose groups are defined for 8.
# satlasnet_si is deliberately absent: it is the Single-Image checkpoint, and its encoder
# squeezes the temporal axis away, so running it here would silently train on one date.
ENCODERS = ['prithvi', 'prithvi2_100m', 'satlasnet_mi']

# `batch_size` is PER GPU. These are NOT measured -- unlike the numbers in ../lora/gen.py,
# which were. Three timesteps put roughly 3x the activations through the encoder, and
# under LoRA the encoder is in the autograd graph, so the single-temporal batch of 32 is
# not expected to fit on a 24 GB 4090. 12 is a starting point chosen by dividing through
# by T with a margin; measure before trusting it, and raise it if there is headroom.
BATCH_SIZE = {'prithvi': 12, 'prithvi2_100m': 12, 'satlasnet_mi': 8}
DEFAULT_BATCH_SIZE = 12

for encoder in ENCODERS :

    batch_size = BATCH_SIZE.get(encoder, DEFAULT_BATCH_SIZE)

    command = f"""torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 pangaea/run.py  --config-name=train  dataset=agbdlite-mt  encoder={encoder}  decoder=reg_upernet  preprocessing=reg_resize  criterion=mse  task=regression finetune=true lora=default batch_size={batch_size} num_workers=6 test_num_workers=6 test_batch_size={batch_size} use_wandb=True task.trainer.eval_interval=1 task.trainer.log_interval=100 task.trainer.eval_interval=1"""
    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    with open(os.path.join(path_script, f"{encoder}.sh"), "w") as f:
        f.write(base_text)
        f.write(command)

# The printed `sbatch` hints. These launchers are run from the pangaea-bench fork.
for encoder in ENCODERS:
    print(f"sbatch {path_script}/{encoder}.sh")
