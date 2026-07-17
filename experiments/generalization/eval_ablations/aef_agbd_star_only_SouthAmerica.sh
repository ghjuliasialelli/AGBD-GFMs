#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/eval-%j.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/eval-%j.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=eval
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g

# --- AGBD-GFMs config ----------------------------------------------------------
# Locate config.sh at the repo root (walk up from the working directory) and
# source it: provides $AGBD_ENV and every AGBD_* path used below. Launchers are
# meant to be run from inside the repo (the model/ directory for train & eval).
_agbd_dir="$(pwd)"
while [ "$_agbd_dir" != "/" ] && [ ! -f "$_agbd_dir/config.sh" ]; do _agbd_dir="$(dirname "$_agbd_dir")"; done
if [ ! -f "$_agbd_dir/config.sh" ]; then echo "config.sh not found; run from inside the AGBD-GFMs repo" >&2; exit 1; fi
source "$_agbd_dir/config.sh"

##################################################################################################################
# TO EDIT ########################################################################################################
years=(2019 2020)
arch="nico_film"
mode="test"
bs=256
offset="false"
min_offset=0
max_offset=0
return_region="true"
skip_preds="false"
lite="false"
drop_overlap="true"
models=(64508409-3)

# if lite is true and return_region is false, raise an error
if [ "$lite" == "true" ] && [ "$return_region" == "false" ]; then
    echo "Error: If lite is true, return_region must be true."
    exit 1
fi

echo "Years: ${years[@]}"
echo "Mode: $mode"
echo "Architecture: $arch"
echo "Models: ${models[@]}"

##################################################################################################################

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$AGBD_ENV" == "cluster" ]; then
    echo "Running on a cluster"

    module load stack/2024-06 gcc/12.2.0
    module load stack/2024-06 python_cuda/3.11.6
    source ${AGBD_CLUSTER_VENV}

    dataset_path=$TMPDIR
    plot_folder="${AGBD_CLUSTER_PLOTS}/"

elif [ "$AGBD_ENV" != "cluster" ]; then
    echo "Running on a local machine"
    dataset_path='local'
    plot_folder="${AGBD_LOCAL_PLOTS}/"

else
    echo "Environment unknown"
fi


# Launch evaluation ##############################################################################################

python eval.py  --dataset_path "$dataset_path" --arch "$arch" --models "${models[@]}" --years "${years[@]}" \
                --plot_folder "$plot_folder" --mode "$mode" --offset "$offset" --min_offset "$min_offset" \
                --max_offset "$max_offset" --return_region "$return_region" --skip_preds "$skip_preds" \
                --lite "$lite" --bs "$bs" --drop_overlap "$drop_overlap" --force "true" --region "SouthAmerica" --keep_region "true" --stats_hold_out_region "SouthAmerica" --stats_keep_region "true" --years_stats "2019-2020"
