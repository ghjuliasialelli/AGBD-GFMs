#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

# Note: since 20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd was trained with root_path_cluster: /scratch/tmp.60489831.gsialelli/
# we need to change it to /cluster/scratch/gsialelli manually in the config.yaml before we can launch this

HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                            --nnodes=1 --nproc_per_node=4 pangaea/run.py \
                                            --config-name=test \
                                            ckpt_dir=20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd