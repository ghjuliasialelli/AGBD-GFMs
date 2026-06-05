#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/scratch/gsialelli/logs/aef-dwn-%A_%a.out
#SBATCH --error=/cluster/scratch/gsialelli/logs/aef-dwn-%A_%a.out
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1

##################################################################################################################
# Main settings

year="None"
output_dir="AEF"
path_txt="AGBD-GFM/aef-dwn/region_AEF_files"
num_workers=$((5 * SLURM_CPUS_PER_TASK))

################################################################################################################################
# Get the i-th region

#regions=("California" "Cuba" "Paraguay" "UnitedRepublicofTanzania" "Ghana" "Austria" "Greece" "Nepal" "ShaanxiProvince" "NewZealand" "FrenchGuiana")
regions=("missing")
num_regions=${#regions[@]}
region=${regions[$SLURM_ARRAY_TASK_ID-1]}

##################################################################################################################
# Establish paths

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)
if [[ "$first_part" == "cluster" ]]; then
    # Check if SLURM_ARRAY_TASK_MIN is 1
    if [ "$SLURM_ARRAY_TASK_MIN" -ne 1 ]; then
        echo "Assertion failed: SLURM_ARRAY_TASK_MIN is not equal to 1" >&2
        exit 1
    fi
    # Check if SLURM_ARRAY_TASK_MAX is equal to the length of the array
    if [ "$SLURM_ARRAY_TASK_MAX" -ne "$num_regions" ]; then
        echo "Assertion failed: SLURM_ARRAY_TASK_MAX is not equal to the length of the array" >&2
        exit 1
    fi
    conda activate awsenv
    # Paths
    base_path="/cluster/work/igp_psr/gsialelli"
    data_base_path="$SCRATCH/Data"
elif [[ "$first_part" == "scratch3" ]]; then
    base_path="/scratch3/gsialelli"
    data_base_path="/scratch3/gsialelli"
fi

output_dir="$data_base_path/$output_dir"
path_txt="$base_path/$path_txt"

##################################################################################################################
# Launch the download

python download.py  --region $region \
                    --output_dir $output_dir \
                    --num_workers $num_workers \
                    --year $year \
                    --path_txt $path_txt