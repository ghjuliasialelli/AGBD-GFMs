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
arch="nico_film"
mode="test"
bs=512
return_region="true"
skip_preds="false"
lite="true" # whether to use the lite dataset for evaluation

# if lite is true and return_region is false, raise an error
if [ "$lite" == "true" ] && [ "$return_region" == "false" ]; then
    echo "Error: If lite is true, return_region must be true."
    exit 1
fi

echo "Years: ${years[@]}"
echo "Mode: $mode"
echo "Architecture: $arch"

config="lite" # best or lite
readarray -t models < "txt/evaluation_${arch}_${config}.txt" # Read the file into an array
echo "Models: ${models[@]}"

##################################################################################################################

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"

    module load stack/2024-06 python/3.11.6
    module load stack/2024-06 gcc/12.2.0
    module load stack/2024-06 python_cuda/3.11.6
    module load stack/2024-06 py-pip/23.1.2-7aykir4
    source /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/agbd/bin/activate
    
    dataset_path=$TMPDIR
    res_folder="/cluster/work/igp_psr/gsialelli/AGBD-GFM/agbd-lite/eval/results/"
    
elif [ "$first_part" == "scratch3" ]; then
    echo "Running on a local machine"
    dataset_path='local'
    res_folder='/scratch3/gsialelli/AGBD-GFM/agbd-lite/eval/results/'
    
else
    echo "Environment unknown"
fi


# Launch evaluation ##############################################################################################

python eval.py  --dataset_path "$dataset_path" --arch "$arch" --models "${models[@]}" --years "${years[@]}" \
                --res_folder "$res_folder" --mode "$mode" --return_region "$return_region" --skip_preds "$skip_preds" \
                --lite "$lite" --bs "$bs"