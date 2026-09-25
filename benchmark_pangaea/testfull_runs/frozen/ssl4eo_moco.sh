#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

# The run was trained with dataset.root_path_cluster=${TMPDIR}; the test job reads from
# /cluster/scratch/gsialelli. Refuse to start on a stale path rather than fail mid-load.
config_yaml=20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd/configs/config.yaml
if ! grep -qE "root_path_cluster: +/cluster/scratch/gsialelli *$" "$config_yaml"; then
    echo "root_path_cluster in $config_yaml is not /cluster/scratch/gsialelli:" >&2
    grep -n "root_path_cluster" "$config_yaml" >&2
    echo "fix it with: perl -i -pe 's|(root_path_cluster: ).*|\${1}/cluster/scratch/gsialelli|' $config_yaml" >&2
    exit 1
fi

HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py --config-name=test ckpt_dir=20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd
