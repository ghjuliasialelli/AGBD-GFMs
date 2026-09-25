# Generates the FULL-on-FULL training launchers for ONE fine-tuning regime: train on the
# full AGBD dataset (dataset=agbd), as opposed to ../../train_runs/, which trains on
# AGBD-Lite. The regime is this script's own parent directory name (frozen / lora); copy
# this file into a sibling regime directory and it retargets itself.
#
#   train_runs/     AGBD-Lite -> (tested on) AGBD-Lite   test_runs/
#                   AGBD-Lite -> AGBD-test               evalbig_runs/
#   trainfull_runs/ AGBD      -> AGBD-test               testfull_runs/   <- this
#
# Only a handful of encoders are scaled up to full AGBD, so each regime lists its own.
# Collect the resulting run-directory names into ../../configs/full_<regime>.txt, which
# ../../testfull_runs/<regime>/gen.py reads.
#
# Unlike AGBD-Lite (auto-downloaded from Zenodo), full AGBD is staged to $TMPDIR by the
# launcher itself, and run.py is pointed there via dataset.root_path_cluster.
#
# Wall time: the frozen SSL4EO-MoCo run (4x4090, bs=32/GPU) reached only epoch 7 of 80
# after 2d17h, so no full-AGBD run completes n_epochs inside 120h -- the paper's number
# is its checkpoint__best.pth. LoRA runs are slower still (encoder in the autograd graph).

import os

path_script = os.path.dirname(os.path.abspath(__file__))
REGIME = os.path.basename(path_script)

base_text = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

# Move all of the necessary data to $TMPDIR
# 'AGBD-Lite-val.h5' (validation runs on the Lite val split: agbd.yaml has eval_lite: True)
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/AGBD-Lite/AGBD-Lite-val.h5 ${TMPDIR} --transfers 16 --checkers 32
# all the .h5 files for training
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR} --include "*v4_*-20.h5" --transfers 16 --checkers 32
# biomes_splits_to_name.pkl
cp /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}
# 'tiles_per_region.pkl'
cp /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/helper/tiles_per_region.pkl ${TMPDIR}
# 'AEF_overlaps.pkl'
cp /cluster/work/igp_psr/gsialelli/AGBD-GFM/aef-dwn/AEF_overlaps.pkl ${TMPDIR}

"""

# Per-regime run.py overrides and the encoders scaled up to full AGBD.
REGIMES = {
    'frozen': {'overrides': '', 'encoders': ['ssl4eo_moco']},
    'lora':   {'overrides': 'finetune=true lora=default ', 'encoders': ['croma_joint']},
}

# `batch_size` is PER GPU. croma_joint under LoRA OOMs above bs=8 on a 24 GB 4090 -- see
# the measurements in ../../train_runs/lora/gen.py. Kept identical to its AGBD-Lite run
# (20260922_021131_c999d4_croma_joint_lora_reg_upernet_agbdlite) so the two are comparable.
BATCH_SIZE = {'lora': {'croma_joint': 8}}
DEFAULT_BATCH_SIZE = 32

if REGIME not in REGIMES:
    raise SystemExit(f"Unknown regime directory '{REGIME}': expected one of {sorted(REGIMES)}")

overrides = REGIMES[REGIME]['overrides']
for encoder in REGIMES[REGIME]['encoders']:

    batch_size = BATCH_SIZE.get(REGIME, {}).get(encoder, DEFAULT_BATCH_SIZE)

    command = f"""HYDRA_FULL_ERROR=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py  --config-name=train  dataset=agbd  encoder={encoder}  decoder=reg_upernet  preprocessing=reg_resize  criterion=mse  task=regression {overrides}batch_size={batch_size} num_workers=6 test_num_workers=6 test_batch_size=32 use_wandb=True task.trainer.eval_interval=0.25 dataset.root_path_cluster=${{TMPDIR}} task.trainer.log_interval=100
"""
    print()
    print("Encoder: ", encoder)
    print(command)

    with open(os.path.join(path_script, f"{encoder}.sh"), "w") as f:
        f.write(base_text)
        f.write(command)

# The sbatch hints. These launchers are run from the pangaea-bench fork.
print(f"# full-on-full, {REGIME}: {len(REGIMES[REGIME]['encoders'])} launchers")
for encoder in REGIMES[REGIME]['encoders']:
    print(f"sbatch {path_script}/{encoder}.sh")
