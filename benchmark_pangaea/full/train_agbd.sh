#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

# Move all of the necessary data to $TMPDIR
# 'AGBD-Lite-val.h5'
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/AGBD-Lite/AGBD-Lite-val.h5 ${TMPDIR} --transfers 16 --checkers 32
# all the .h5 files for training
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR} --include "*v4_*-20.h5" --transfers 16 --checkers 32
# biomes_splits_to_name.pkl
cp /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}
# 'tiles_per_region.pkl'
cp /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/helper/tiles_per_region.pkl ${TMPDIR}
# 'AEF_overlaps.pkl' 
cp /cluster/work/igp_psr/gsialelli/AGBD-GFM/aef-dwn/AEF_overlaps.pkl ${TMPDIR}

HYDRA_FULL_ERROR=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py \
                            --config-name=train  dataset=agbd  encoder=ssl4eo_moco  decoder=reg_upernet  preprocessing=reg_resize  \
                            criterion=mse  task=regression batch_size=32 num_workers=6 test_num_workers=6 test_batch_size=32 \
                            use_wandb=True task.trainer.eval_interval=0.25 dataset.root_path_cluster=${TMPDIR} task.trainer.log_interval=100