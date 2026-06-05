#!/bin/bash
#SBATCH --job-name=move_aef_data
#SBATCH --time=24:00:00        # 400GB might take a few hours depending on load
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # Match rclone transfers
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/gsialelli/logs/move_data_%j.out
#SBATCH --error=/cluster/scratch/gsialelli/logs/move_data_%j.out

# Paths
SOURCE="/cluster/work/igp_psr/gsialelli/Data/AEF/"
DEST="/cluster/scratch/gsialelli/Data/AEF/"
LIST_SOURCE="/cluster/work/igp_psr/gsialelli/AGBD-GFM/aef-dwn/region_AEF_files/UnitedRepublicofTanzania_AEF_files.txt"

# 1. Create destination
mkdir -p "$DEST"

# 2. Create a temporary file list
TEMP_LIST="move_list_${SLURM_JOB_ID}.txt"
grep "/2020/" "$LIST_SOURCE" | sed 's|.*/||' > "$TEMP_LIST"

echo "Starting move of $(wc -l < $TEMP_LIST) files..."

# 3. Execute rclone
# Using 'copy' first is safer than 'move' for massive transfers; 
# you can manually delete the source once you verify the scratch content.
rclone copy "$SOURCE" "$DEST" \
    --files-from "$TEMP_LIST" \
    --transfers 4 \
    --multi-thread-streams 4 \
    --progress

# 4. Cleanup list
rm "$TEMP_LIST"

echo "Done."