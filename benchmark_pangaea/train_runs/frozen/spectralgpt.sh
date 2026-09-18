#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:2

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 pangaea/run.py  --config-name=train  dataset=agbdlite  encoder=spectralgpt  decoder=reg_upernet  preprocessing=reg_resize  criterion=mse  task=regression batch_size=32 num_workers=6 test_num_workers=6 test_batch_size=32 use_wandb=True task.trainer.eval_interval=1 task.trainer.log_interval=100 task.trainer.eval_interval=1