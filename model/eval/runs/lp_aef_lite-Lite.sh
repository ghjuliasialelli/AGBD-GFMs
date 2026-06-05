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

##################################################################################################################
# TO EDIT ########################################################################################################
years=(2019 2020)
arch="lp"
mode="test"
bs=256
offset="false"
min_offset=0
max_offset=0
return_region="true"
skip_preds="false"
lite="true" # evaluate on AGBD-Lite test
drop_overlap="false"
stats_hold_out_region="None" # region whose stats file to load; "None" for global stats
stats_keep_region="false"    # whether the stats file is for the kept region
years_stats="2019-2020"      # years the models were trained on (for picking the stats file)

# if lite is true and return_region is false, raise an error
if [ "$lite" == "true" ] && [ "$return_region" == "false" ]; then
    echo "Error: If lite is true, return_region must be true."
    exit 1
fi

echo "Years: ${years[@]}"
echo "Mode: $mode"
echo "Architecture: $arch"

config="aef_lite" # best, lite, aef, aef_lite
readarray -t models < "eval/configs/evaluation_${arch}_${config}.txt" # Read the file into an array
echo "Models: ${models[@]} (arch: $arch, config: $config)"

##################################################################################################################

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"

    module load stack/2024-06 gcc/12.2.0
    module load stack/2024-06 python_cuda/3.11.6
    source /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/agbd/bin/activate

    dataset_path=$TMPDIR
    plot_folder="/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/eval_plots/"

elif [ "$first_part" == "scratch3" ]; then
    echo "Running on a local machine"
    dataset_path='local'
    plot_folder='/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/eval_plots/'

else
    echo "Environment unknown"
fi


# Launch evaluation ##############################################################################################

python eval.py  --dataset_path "$dataset_path" --arch "$arch" --models "${models[@]}" --years "${years[@]}" \
                --plot_folder "$plot_folder" --mode "$mode" --offset "$offset" --min_offset "$min_offset" \
                --max_offset "$max_offset" --return_region "$return_region" --skip_preds "$skip_preds" \
                --lite "$lite" --bs "$bs" --drop_overlap "$drop_overlap" \
                --stats_hold_out_region "$stats_hold_out_region" --stats_keep_region "$stats_keep_region" \
                --years_stats "$years_stats"
