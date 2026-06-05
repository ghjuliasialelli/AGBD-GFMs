#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/logs/inference-%A_%a.out
#SBATCH --error=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/logs/inference-%A_%a.out
#SBATCH --mem-per-cpu=9G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:15g
#SBATCH --array=1-9

RELAUNCH_FAILS="false"
LIST_FAILS=""

##################################################################################################################
# Parameters

# Main settings
arch="nico"
models=('56644785-1' '56644785-2' '56644785-3')
models=('59620113-1')
arch='nico_film'
year=2020
masking="false" # whether to mask the predictions
entity='gs-tp-biomass'
dtype='float32'
patch_size=(512 512)
batch_size=1
pred_crop=(64 64 64 64) # overlap size
mode="true"
std="true"
method="median"
factor=6

# Tiles to run the inference on
LIST_TILES_FILE="gsialelli/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_${year}.txt"
echo "Will read products from file ${LIST_TILES_FILE}"

##################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"

first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [[ "$first_part" == "cluster" ]]; then
    echo "Running on a cluster"
    LIST_PRODS_FILE="/cluster/work/igp_psr/${LIST_TILES_FILE}"
    source activate nico
elif [[ "$first_part" == "scratch3" ]]; then
    echo "Running on a local machine"
    LIST_PRODS_FILE="/scratch3/${LIST_TILES_FILE}"
    SLURM_ARRAY_TASK_MIN=1
    SLURM_ARRAY_TASK_MAX=1
    SLURM_ARRAY_TASK_ID=1
else
    echo "Environment unknown"
fi

################################################################################################################################
# Parse the tile names from LIST_PRODS_FILE

readarray -t tile_names < ${LIST_PRODS_FILE}

# If running locally, only consider the first tile
if [[ "$first_part" == "scratch3" ]]; then
    # get only the first element
    tile_names=("${tile_names[0]}")
fi

num_tiles=${#tile_names[@]}

# Check if SLURM_ARRAY_TASK_MIN is 1
if [ "$SLURM_ARRAY_TASK_MIN" -ne 1 ]; then
    echo "Assertion failed: SLURM_ARRAY_TASK_MIN is not equal to 1" >&2
    exit 1
fi

# Check if SLURM_ARRAY_TASK_MAX is equal to the length of the array
if [ "$SLURM_ARRAY_TASK_MAX" -ne "$num_tiles" ]; then
    echo "Assertion failed: SLURM_ARRAY_TASK_MAX is not equal to the length of the array" >&2
    exit 1
fi

# Select the i-th element in the array, where i is the current job number - 1 (SLURM_ARRAY_TASK_ID is 1-indexed)
tile=${tile_names[$SLURM_ARRAY_TASK_ID-1]}

# Check if the tile should be skipped
to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
if [[ " $to_skip " =~ " $tile " ]]; then
    echo "Tile ${tile} is in the skip list. Skipping this tile." 
    exit 0
fi

# If RELAUNCH_FAILS is enabled, continue only if the tile is in LIST_FAILS
if [[ "$RELAUNCH_FAILS" == "true" ]]; then
    if [[ ! " ${LIST_FAILS[@]} " =~ " ${tile} " ]]; then
        echo "Tile ${tile} is not in the list of failed tiles. Skipping this tile."
        exit 0
    fi
fi

echo "Launching inference_aef.py for tile: " ${tile}
echo "and for year: " ${year}

################################################################################################################################
# Launch the inference

if [[ "$first_part" == "cluster" ]]; then

    python3 inference_aef.py --models ${models[@]} --arch ${arch} --dataset_path ${TMPDIR} \
            --saving_dir /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/predictions \
            --tile_name $tile --masking $masking --patch_size ${patch_size[@]} \
            --entity $entity --dtype $dtype --year $year --mode $mode --std $std \
            --pred_crop ${pred_crop[@]} --method $method --batch_size $batch_size --factor $factor

else

    python3 inference_aef.py --models ${models[@]} --arch ${arch} --dataset_path local \
            --saving_dir /scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions \
            --tile_name $tile --masking $masking --patch_size ${patch_size[@]} \
            --entity $entity --dtype $dtype --year $year --mode $mode --std $std \
            --pred_crop ${pred_crop[@]} --method $method --batch_size $batch_size --factor $factor

fi