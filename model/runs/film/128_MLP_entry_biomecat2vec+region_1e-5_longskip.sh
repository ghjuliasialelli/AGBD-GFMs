#!/bin/bash
#SBATCH --nodes=1                       # This needs to match Trainer(num_nodes=...)
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/logs/training-%A_%a.txt
#SBATCH --error=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/logs/training-%A_%a.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=300G
#SBATCH --array=1-3
#SBATCH --job-name=models
#SBATCH --gpus=rtx_4090:1

module load stack/2024-06 python/3.11.6
module load stack/2024-06 gcc/12.2.0
module load stack/2024-06 python_cuda/3.11.6
module load stack/2024-06 py-pip/23.1.2-7aykir4
source /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/agbd/bin/activate

################################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]
then
    JOB_ID=$SLURM_ARRAY_JOB_ID
    MODEL_IDX=$SLURM_ARRAY_TASK_ID
    NCPUS=$SLURM_CPUS_PER_TASK
    NNODES=$SLURM_NNODES
    NGPUS=$SLURM_GPUS

    if [[ $NGPUS == *":"* ]]; then
        NGPUS=${SLURM_GPUS##*:}
    fi
    
else
    JOB_ID=0
    MODEL_IDX=0
    NGPUS=1
    NCPUS=8
    NNODES=1
fi

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"
    
    # Move the .h5 files
    rsync --include '*v4_*-20.h5' --exclude '*' -aq /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR}

    # Move the file with the statistics
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4.pkl ${TMPDIR}
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4-1.pkl ${TMPDIR}

    # Move the file with the splits
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}

    # Move the file with the embeddings
    rsync -aq /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec/embeddings_train.csv ${TMPDIR}

    # Move the .pkl file with the AGB residuals statistics
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/nico_film_17997535-1_17997535-2_17997535-3_train_agb_residuals_stats.pkl ${TMPDIR}

elif [ "$first_part" == "scratch3" ]; then
    echo "Running on a local machine"
else
    echo "Environment unknown"
fi


##################################################################################################################
# To edit ########################################################################################################

# Loss function
loss_fn='MSE' # GNLL or MSE

# Architecture
arch="nico_film"
if [[ $arch == *"film"* ]]; then
    film="true"
else
    film="false"
fi

# Check that if loss is GNLL, then _gaussian needs to be in arch
if [ "$loss_fn" == "GNLL" ] && [[ "$arch" != *"gaussian"* ]]; then
    echo "If loss function is GNLL, then architecture must be gaussian." 
    exit 1
fi

# Nico_net architecture
num_sepconv_blocks=8
num_sepconv_filters=256
long_skip="true"
returns="dense" # dense or pixel

only_entry="true" # whether to only put FiLM layers in the entry block
l2=0.00001

# Features to include ################################################################################################

# patch size
patch_size=(25 25) # (has to be 2k+1, 2k+1) and 2k+1 should be a multiple of 5
crop="false"

# padding strategy
padding_mode='zeros' # zeros, reflect, replicate, valid

# normalization values
new_stats="true"
norm_strat='pct'

# whether to log transform the AGB
log_transform="false"

# whether to over-sample from the minority AGB bins
oversampling="false"

# canopy height
ch="false"

# canopy height - gedi residuals
residuals="false"
res_norm="false"
res_film="false"
res_in="false"
res_in_central="false"
res_in_patch="false"

# rh98 for film layers
rh98_film="false"

# agb residuals 
agb_residuals="false" # (input)
agb_residuals_film="false" # (FiLM)
agb_residuals_file="nico_film_17997535-1_17997535-2_17997535-3_train_agb_residuals_stats.pkl"
agb_res_all="false"
agb_res_one="mean"

# conditioning the output on the CH
sim_dist="false"
similarity="JS" # can be one of 'SCC', 'JS'
similarity_weight=10.0  #         -10,   10
SCC_ws=5
SCC_softmax="false"

# s2 bands
bands=(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12) #(B02 B03 B04 B08) #(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12)
s2_dates="true"
s2_day="true"
s2_doy="true"

# whether to randomly drop the s2 data
train_mask="false"
val_mask="false"
test_mask="false"

# latitude and longitude
latlon="true"
debug_latlon="true"

# s1 bands
s1="false"

# alos bands
alos="true"

# land cover & how to encode it for the input feature
lc="true"
ft_cat2vec="true"
ft_onehot="false"
ft_sincos="false"

# dem
dem="true"
topo="true"
aspect="true"
slope="true"

# If 1x1 patch, cannot have topo
if [ "${patch_size[0]}" -eq 1 ] && [ "${patch_size[1]}" -eq 1 ]; then
    topo="false"
    aspect="false"
    slope="false"
fi

# gedi dates
gedi_dates="false"

# input for the FiLM layers & how to encode it
region="true"
biome="true"
#res_film is set above, don't forget to check it
emb_onehot="false" # if false, will default to cat2vec embeddings
emb_dist="false"
emb_cat2vec="true"
emb_sincos="false"
biome_dim=128
linear_emb="false"

# Year to train on
years=(2019 2020)

echo "Year: ${years[@]}"
echo "Architecture: $arch"

# Define the dimension of the "biome" embeddings for the FiLM layers
if [ "$biome" == "true" ]; then
    if [ "$emb_onehot" == "true" ] || [ "$emb_dist" == "true" ] ; then
        emb_dim=14 # 14dim embeddings
    elif [ "$emb_cat2vec" == "true" ]; then
        emb_dim=5 # 5dim embeddings
    elif [ "$emb_sincos" == "true" ]; then
        emb_dim=2 # sine and cosine
    else 
        echo "No embedding type selected."
        exit 1
    fi
else 
    emb_dim=0
fi

# Define the dimension of the "region" embeddings for the FiLM layers
if [ "$region" == "true" ]; then
    emb_dim=$((emb_dim+8)) # always onehot region
fi

# Define the dimension of the "residuals" embeddings for the FiLM layers
if [ "$res_film" == "true" ]; then
    emb_dim=$((emb_dim+1)) # 1dim residuals
fi

# If we give the RH98 to the FiLM layers, we need to add 1 dimension
if [ "$rh98_film" == "true" ]; then
    emb_dim=$((emb_dim+1)) # 1dim residuals
fi

# If we give the AGB residuals to the FiLM layers, we need to add 1 dimensions
if [ "$agb_residuals_film" == "true" ]
then
    if [ "$agb_res_all" == "true" ]
    then
        emb_dim=$((emb_dim+5)) # + 5 because ['min', 'max', 'mean', 'median', 'std']
    else
        emb_dim=$((emb_dim+1)) # + 1 because only one feature
    fi
fi

# Check that if aspect or slope are true, then topo must be true
if [ "$aspect" == "true" ] || [ "$slope" == "true" ] && [ "$topo" == "false" ]; then
    echo "If aspect or slope are true, then topo must be true."
    exit 1
fi

# Check that if s2_day or s2_doy are true, then s2_dates must be true
if [ "$s2_day" == "true" ] || [ "$s2_doy" == "true" ] && [ "$s2_dates" == "false" ]; then
    echo "If s2_day or s2_doy are true, then s2_dates must be true."
    exit 1
fi

# if residuals is true, then either res_film or res_in needs to be true. and if res_in is true, res_in_central or res_in_patch needs to be true
if [ "$residuals" == "true" ] && [ "$res_film" == "false" ] && [ "$res_in" == "false" ]; then
    echo "If --residuals is true, then either --res_film or --res_in needs to be true."
    exit 1
fi
if [ "$res_in" == "true" ] && [ "$res_in_central" == "false" ] && [ "$res_in_patch" == "false" ]; then
    echo "If --res_in is true, then either --res_in_central or --res_in_patch needs to be true."
    exit 1
fi
# and if either res_in or res_film is true, then residuals needs to be true, otherwise exit 1
if [ "$res_in" == "true" ] || [ "$res_film" == "true" ]; then
    if [ "$residuals" == "false" ]; then
        echo "If --res_in or --res_film is true, then --residuals needs to be true."
        exit 1
    fi
fi

# Model parameters ###############################################################################################

channel_dims=(32 32 64 128 128 128)
leaky_relu="false"
max_pool="false"
freeze="false"

# Training arguments
n_epochs=14
batch_size=64
limit="false"
reweighting='no'
lr=0.001
step_size=30
gamma=0.1
patience=1000
min_delta=0.0
chunk_size=1
num_outputs=1

# Sanity check
scramble="false"
debug_film="false"

# Compute the number of input features #############################################################################

# s2 bands and lat/lon bands
num_bands=${#bands[@]}
in_features=$((num_bands+2)) # + 2 because always using `lat_1` and `lat_2`
if [ "$latlon" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because `lon_1` and `lon_2`
fi

# canopy height
if [ "$ch" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because `ch` and `ch_std`
fi

# alos bands
if [ "$alos" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because hh and hv
fi

# land cover
if [ "$lc" == "true" ]
then
    if [ "$ft_cat2vec" == "true" ]
    then
        in_features=$((in_features+6)) # + 6 because 5dim embeddings and lc prob
    elif [ "$ft_onehot" == "true" ]
    then
        in_features=$((in_features+15)) # + 14 because 14dim embeddings and lc prob
    elif [ "$ft_sincos" == "true" ]
    then
        in_features=$((in_features+3)) # + 3 because lc sin lc cos and lc prob
    fi
fi

# digital elevation model
if [ "$dem" == "true" ]
then
    if [ "$topo" == "true" ]
    then
        if [ "$aspect" == "true" ]
        then
            in_features=$((in_features+2)) # + 2 because aspect_cos, aspect_si
        fi
        if [ "$slope" == "true" ]
        then
            in_features=$((in_features+1)) # + 1 because slop
        fi
        in_features=$((in_features+1)) # + 1 because elevation
    else
        in_features=$((in_features+1)) # + 1 because elevation
    fi
fi

# gedi dates
if [ "$gedi_dates" == "true" ]
then
    in_features=$((in_features+3)) # + 3 because num_days, cos, sin
fi

# s2 dates
if [ "$s2_dates" == "true" ]
then
    if [ "$s2_day" == "true" ]
    then
        in_features=$((in_features+1)) # + 1 because num_days
    fi
    if [ "$s2_doy" == "true" ]
    then
        in_features=$((in_features+2)) # + 1 because cos, sin
    fi
fi

# residuals
if [ "$res_in" == "true" ]
then
    in_features=$((in_features+1)) # + 1 because residuals
fi

# agb residuals
if [ "$agb_residuals" == "true" ]
then
    if [ "$agb_res_all" == "true" ]
    then
        in_features=$((in_features+5)) # + 5 because ['min', 'max', 'mean', 'median', 'std']
    else
        in_features=$((in_features+1)) # + 1 because only one feature
    fi
fi

# Output path and model name #####################################################################################

if [ "$first_part" == "cluster" ]
then
    model_path=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/weights/${arch}
    dataset_path=$TMPDIR
    model_name=${model_path}/${JOB_ID}-${MODEL_IDX}
else
    model_path=/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/weights/${arch}
    dataset_path='local'
    model_name=${model_path}/local
fi

# Launch training ################################################################################################
echo "NNODES: $NNODES"
echo "NGPUS: $NGPUS"

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=$NNODES --nproc_per_node=$NGPUS \
        train.py    --model_path $model_path \
                    --model_name $model_name \
                    --dataset_path $dataset_path \
                    --augment "false" \
                    --norm "false" \
                    --arch $arch \
                    --model_idx $MODEL_IDX \
                    --loss_fn $loss_fn \
                    --latlon $latlon \
                    --debug_latlon $debug_latlon \
                    --ch $ch \
                    --bands $(IFS=" " ; echo "${bands[*]}") \
                    --in_features $in_features \
                    --s1 $s1 \
                    --alos $alos \
                    --lc $lc \
                    --dem $dem \
                    --topo $topo \
                    --aspect $aspect \
                    --slope $slope \
                    --gedi_dates $gedi_dates \
                    --s2_dates $s2_dates \
                    --s2_day $s2_day \
                    --s2_doy $s2_doy \
                    --num_outputs $num_outputs \
                    --downsample "false" \
                    --n_epochs $n_epochs \
                    --batch_size $batch_size \
                    --lr $lr \
                    --step_size $step_size \
                    --gamma $gamma \
                    --patience $patience \
                    --min_delta $min_delta \
                    --reweighting $reweighting \
                    --norm_strat $norm_strat \
                    --limit $limit \
                    --patch_size ${patch_size[@]} \
                    --chunk_size $chunk_size \
                    --years ${years[@]} \
                    --freeze $freeze \
                    --num_gpus $NGPUS \
                    --num_cpus $NCPUS \
                    --film $film \
                    --biome_dim $biome_dim \
                    --emb_dim $emb_dim \
                    --region $region \
                    --num_sepconv_blocks $num_sepconv_blocks \
                    --num_sepconv_filters $num_sepconv_filters \
                    --long_skip $long_skip \
                    --new_stats $new_stats \
                    --only_entry $only_entry \
                    --l2 $l2 \
                    --residuals $residuals \
                    --res_film $res_film \
                    --res_in $res_in \
                    --res_in_central $res_in_central \
                    --res_in_patch $res_in_patch \
                    --emb_onehot $emb_onehot \
                    --emb_dist $emb_dist \
                    --emb_cat2vec $emb_cat2vec \
                    --emb_sincos $emb_sincos \
                    --ft_cat2vec $ft_cat2vec \
                    --ft_onehot $ft_onehot \
                    --ft_sincos $ft_sincos \
                    --res_norm $res_norm \
                    --linear_emb $linear_emb \
                    --rh98_film $rh98_film \
                    --crop $crop \
                    --padding_mode $padding_mode \
                    --returns $returns \
                    --agb_residuals $agb_residuals \
                    --agb_residuals_file $agb_residuals_file \
                    --agb_res_all $agb_res_all \
                    --agb_res_one $agb_res_one \
                    --agb_residuals_film $agb_residuals_film \
                    --sim_dist $sim_dist \
                    --similarity $similarity \
                    --similarity_weight $similarity_weight \
                    --log_transform $log_transform \
                    --SCC_ws $SCC_ws \
                    --SCC_softmax $SCC_softmax \
                    --oversampling $oversampling \
                    --train_mask $train_mask \
                    --val_mask $val_mask \
                    --test_mask $test_mask