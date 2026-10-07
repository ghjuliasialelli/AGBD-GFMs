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

# Move all of the necessary data to $TMPDIR (the same files and sources as trainfull_runs/,
# minus AGBD-Lite-val.h5, which only validation reads)
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR} --include "*v4_*-20.h5" --transfers 16 --checkers 32
# biomes_splits_to_name.pkl
cp /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}
# 'tiles_per_region.pkl'
cp /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/helper/tiles_per_region.pkl ${TMPDIR}
# 'AEF_overlaps.pkl'
cp /cluster/work/igp_psr/gsialelli/AGBD-GFM/aef-dwn/AEF_overlaps.pkl ${TMPDIR}

# Refuse to start on an incomplete copy: the exact files pangaea's AGBD opens (agbd.py fnames)
for f in biomes_splits_to_name.pkl tiles_per_region.pkl AEF_overlaps.pkl          data_subset-{2019,2020}-v4_{0..19}-20.h5; do
    if [ ! -f "${TMPDIR}/$f" ]; then echo "staging: ${TMPDIR}/$f is missing" >&2; exit 1; fi
done

HYDRA_FULL_ERROR=1 TQDM_DISABLE=1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 pangaea/run.py --config-name=test ckpt_dir=20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd \
    ++test_overrides.dataset.root_path_cluster=${TMPDIR}
