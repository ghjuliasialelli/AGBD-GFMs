#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/scratch/gsialelli/logs/aef-patch-%A_%a.out
#SBATCH --error=/cluster/scratch/gsialelli/logs/aef-patch-%A_%a.out
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-4

##################################################################################################################
# Main settings

year=2019
patch_size=25
num_channels=64
batch_size=1024
missing=('37MCT' '17QQF' '17QQC' '58GFN' '30PVS' '59GNQ')

################################################################################################################################
# Get the i-th region

#regions=("California" "Cuba" "Paraguay" "UnitedRepublicofTanzania_1/2" "UnitedRepublicofTanzania_2/2" "Ghana" "Austria" "Greece" "Nepal" "ShaanxiProvince" "NewZealand" "FrenchGuiana")
regions=('UnitedRepublicofTanzania' 'Cuba' 'NewZealand' 'Ghana')
num_regions=${#regions[@]}
_region=${regions[$SLURM_ARRAY_TASK_ID-1]}

# check whether we're downloading missing data
missing_suffix=""
if [ ${#missing[@]} -gt 0 ]; then
    missing_suffix="_missing"
fi

# check if there is _<number> in the region name
if [[ $_region == *"_"* ]]; then
    region=$(echo $_region | cut -d'_' -f1)
    region_number=$(echo $_region | cut -d'_' -f2 | cut -d'/' -f1)
    num_splits=$(echo $_region | cut -d'_' -f2 | cut -d'/' -f2)
    echo "Processing region: $region ($region_number out of $num_splits)"
    output_fname="${region}_${year}_${region_number}-${num_splits}${missing_suffix}.h5"
else
    region=$_region
    region_number=0
    num_splits=1
    echo "Processing region: $region"
    output_fname="${region}_${year}${missing_suffix}.h5"
fi

##################################################################################################################
# Establish paths

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)
if [[ "$first_part" == "cluster" ]]; then
    # Paths
    base_path="/cluster/work/igp_psr/gsialelli"
    data_base_path="/cluster/work/igp_psr/gsialelli/Data"
    write_path="${TMPDIR}"
    final_destination="${data_base_path}/patches/AEF"
    path_tiff="${SCRATCH}/Data/AEF"

elif [[ "$first_part" == "scratch3" ]]; then
    base_path="/scratch3/gsialelli"
    data_base_path="/scratch3/gsialelli"
    write_path="${data_base_path}/patches/AEF"
    path_tiff="${data_base_path}/AEF"
fi


##################################################################################################################
# Launch the download

python create_patches.py    --region $region \
                            --region_number $region_number \
                            --num_splits $num_splits \
                            --year $year \
                            --base_path $base_path \
                            --data_path $data_base_path \
                            --patch_size $patch_size \
                            --num_channels $num_channels \
                            --batch_size $batch_size \
                            --write_path $write_path \
                            --path_tiff $path_tiff \
                            --missing ${missing[@]}


# If on the cluster, move the output .h5 file from TMPDIR to the final destination
if [[ "$first_part" == "cluster" ]]; then
    echo "Moving output .h5 files to final destination"
    
    rsync -av "${write_path}/${output_fname}" "$final_destination/"
fi