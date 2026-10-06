#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:4

# The data path below is passed via test_overrides, which only pangaea-bench's run.py from the
# test_overrides commit onwards honours; an older checkout would silently ignore it and fail at data load.
if ! grep -q "test_overrides" pangaea/run.py; then
    echo "pangaea/run.py does not support test_overrides: update the pangaea-bench checkout (agbd-release)." >&2
    exit 1
fi

AGBD_DUMP_H5=/cluster/scratch/gsialelli/agbd_dumps/20260930_080105_f8cb5c_terramind_tiny_lora_reg_upernet_agbd_agbd_test.h5 HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py --config-name=test ckpt_dir=20260930_080105_f8cb5c_terramind_tiny_lora_reg_upernet_agbd \
    ++test_overrides.dataset.root_path_cluster=/cluster/scratch/gsialelli
