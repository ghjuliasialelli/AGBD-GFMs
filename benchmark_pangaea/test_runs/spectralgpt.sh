#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:1

HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 pangaea/run.py --config-name=test ckpt_dir=20260323_103538_5cf898_spectralgpt_reg_upernet_agbdlite