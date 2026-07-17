#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/training-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/training-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=models
#SBATCH --array=1-1
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=300G

# --- AGBD-GFMs config ----------------------------------------------------------
# Locate config.sh at the repo root (walk up from the working directory) and
# source it: provides $AGBD_ENV and every AGBD_* path used below. Launchers are
# meant to be run from inside the repo (the model/ directory for train & eval).
_agbd_dir="$(pwd)"
while [ "$_agbd_dir" != "/" ] && [ ! -f "$_agbd_dir/config.sh" ]; do _agbd_dir="$(dirname "$_agbd_dir")"; done
if [ ! -f "$_agbd_dir/config.sh" ]; then echo "config.sh not found; run from inside the AGBD-GFMs repo" >&2; exit 1; fi
source "$_agbd_dir/config.sh"

################################################################################################################################

# Whether to use the normal AGBD dataset, or the AGBD-Lite dataset
lite="false"
lite_eval_big="false"
lite_chunk_size=1

# Whether to use AEF embeddings
aef="true"
tessera="false"     # TESSERA embeddings (only with --lite)
drop_overlaps="true"

# Whether to move everything to $TMPDIR
tmpdir="true"

################################################################################################################################
# Establish paths

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$AGBD_ENV" == "cluster" ]; then
    module load stack/2024-06 gcc/12.2.0
    module load stack/2024-06 python_cuda/3.11.6
    source ${AGBD_CLUSTER_VENV}

    JOB_ID=$SLURM_ARRAY_JOB_ID
    MODEL_IDX=${SLURM_ARRAY_TASK_ID:-0}
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

if [ "$AGBD_ENV" == "cluster" ]; then
    echo "Running on a cluster"
    if [ "$tmpdir" == "true" ]; then
        echo "Using TMPDIR for dataset"
        if [ "$lite" == "true" ]; then
            rclone copy ${AGBD_CLUSTER_H5}/AGBD-Lite/ ${TMPDIR} --exclude "AGBD-test.h5" --include "*.h5" --include "AGBD-Lite-statistics.pkl" --transfers 16 --checkers 32
            cp ${AGBD_CLUSTER_EMBEDDINGS}/AGBD-Lite/embeddings_train_lite.csv ${TMPDIR}
            cp ${AGBD_CLUSTER_AUX}/agbd-lite/mapping_lite_to_og.pkl ${TMPDIR}
            if [ "$lite_eval_big" == "true" ]; then
                rclone copy ${AGBD_CLUSTER_H5}/AGBD-Lite/AGBD-test.h5 ${TMPDIR} --transfers 16 --checkers 32
            fi
        fi
        if [ "$lite" == "false" ]; then
            rclone copy ${AGBD_CLUSTER_H5}/ ${TMPDIR} --include "*v4_*-20.h5" --include "*statistics*.pkl" --transfers 16 --checkers 32
        fi
        if [ "$aef" == "true" ]; then
            rclone copy ${AGBD_CLUSTER_AEF_H5}/ ${TMPDIR} --include "*.h5" --include "*statistics*.pkl" --transfers 16 --checkers 32
        fi

        cp ${AGBD_CLUSTER_AGB}/biomes_splits_to_name.pkl ${TMPDIR}
        cp ${AGBD_CLUSTER_EMBEDDINGS}/AGBD/embeddings_train.csv ${TMPDIR}
        cp ${AGBD_CLUSTER_HELPER}/tiles_per_region.pkl ${TMPDIR}
        cp ${AGBD_CLUSTER_AUX}/aef-dwn/AEF_overlaps.pkl ${TMPDIR}
    else
        echo "Using SCRATCH for dataset"
    fi
elif [ "$AGBD_ENV" != "cluster" ]; then
    echo "Running on a local machine"
else
    echo "Environment unknown"
fi

##################################################################################################################
# Training config

loss_fn='MSE'

predict="agbd"
if [ "$predict" == "agbd" ] || [ "$predict" == "rh98" ]; then
    num_outputs=1
elif [ "$predict" == "biome" ]; then
    num_outputs=14
else
    echo "Invalid target."
    exit 1
fi

arch="nico_film"
if [[ $arch == *"film"* ]]; then film="true"; else film="false"; fi
if [ "$loss_fn" == "GNLL" ] && [[ "$arch" != *"gaussian"* ]]; then
    echo "If loss is GNLL, arch must be gaussian."; exit 1
fi

num_sepconv_blocks=8
num_sepconv_filters=256
long_skip="true"
returns="dense"
only_entry="true"
l2=0.00001

# patch
patch_size=(25 25)
crop="false"
padding_mode='zeros'

# normalization
new_stats="true"
norm_strat='pct'

log_transform="false"
oversampling="false"

# canopy height
ch="false"

# canopy height residuals
residuals="false"
res_norm="false"
res_film="false"
res_in="false"
res_in_central="false"
res_in_patch="false"
rh98_film="false"

# agb residuals
agb_residuals="false"
agb_residuals_film="false"
agb_residuals_file="nico_film_17997535-1_17997535-2_17997535-3_train_agb_residuals_stats.pkl"
agb_res_all="false"
agb_res_one="mean"

# CH similarity
sim_dist="false"
similarity="JS"
similarity_weight=10.0
SCC_ws=5
SCC_softmax="false"

# === SOURCE-AXIS: input modalities =====================================
bands=()
s2_dates="false"
s2_day="false"
s2_doy="false"

train_mask="false"
val_mask="false"
test_mask="false"

latlon="true"
debug_latlon="true"

s1="false"
alos="false"

lc="true"
ft_cat2vec="true"
ft_onehot="false"
ft_sincos="false"

dem="false"
topo="true"
aspect="true"
slope="true"

if [ "${patch_size[0]}" -eq 1 ] && [ "${patch_size[1]}" -eq 1 ]; then
    topo="false"; aspect="false"; slope="false"
fi

gedi_dates="false"

# FiLM
region="false"
biome="false"
emb_onehot="true"
emb_dist="false"
emb_cat2vec="false"
emb_sincos="false"
biome_dim=64
linear_emb="false"
ensemble="true"
n_members=3

# === SCOPE-AXIS: years & geo-ablation ==================================
years=(2019)
subsample_2020="false"
years_stats="None"

echo "Year: ${years[@]}"
echo "Architecture: $arch"

if [ "$predict" == "biome" ] && [ "$film" == "true" ]; then biome="false"; fi

geo_ablation="false"
keep_region="false"
if [ "$geo_ablation" == "true" ]; then
    regions=("SouthAsia" "Africa" "SouthAmerica")
    if [ "$AGBD_ENV" == "cluster" ]; then
        region_id=$SLURM_ARRAY_TASK_ID
    else
        region_id=1
    fi
    hold_out_region=${regions[$((region_id-1))]}
    if [ "$keep_region" == "true" ]; then
        echo "Training only on region: $hold_out_region"
    else
        echo "Holding out region: $hold_out_region"
    fi
    stats_hold_out_region=$hold_out_region
    stats_keep_region=$keep_region
else
    hold_out_region="None"
    stats_hold_out_region="None"
    stats_keep_region="false"
fi

# === arg checks ========================================================
if [ "$biome" == "true" ]; then
    if [ "$emb_onehot" == "true" ] || [ "$emb_dist" == "true" ]; then emb_dim=14
    elif [ "$emb_cat2vec" == "true" ]; then emb_dim=5
    elif [ "$emb_sincos" == "true" ]; then emb_dim=2
    else echo "No embedding type selected."; exit 1; fi
else emb_dim=0; fi

if [ "$region" == "true" ]; then emb_dim=$((emb_dim+8)); fi
if [ "$res_film" == "true" ]; then emb_dim=$((emb_dim+1)); fi
if [ "$rh98_film" == "true" ]; then emb_dim=$((emb_dim+1)); fi
if [ "$agb_residuals_film" == "true" ]; then
    if [ "$agb_res_all" == "true" ]; then emb_dim=$((emb_dim+5))
    else emb_dim=$((emb_dim+1)); fi
fi
if [ "$ensemble" == "true" ]; then emb_dim=$n_members; fi

if [ "$aspect" == "true" ] || [ "$slope" == "true" ] || [ "$dem" == "true" ] && [ "$topo" == "false" ]; then
    echo "If aspect/slope/dem are true, topo must be true."; exit 1
fi
if [ "$s2_day" == "true" ] || [ "$s2_doy" == "true" ] && [ "$s2_dates" == "false" ]; then
    echo "If s2_day or s2_doy is true, s2_dates must be true."; exit 1
fi
if [ "$residuals" == "true" ] && [ "$res_film" == "false" ] && [ "$res_in" == "false" ]; then
    echo "If residuals is true, then either res_film or res_in must be true."; exit 1
fi
if [ "$res_in" == "true" ] && [ "$res_in_central" == "false" ] && [ "$res_in_patch" == "false" ]; then
    echo "If res_in is true, then res_in_central or res_in_patch must be true."; exit 1
fi
if [ "$res_in" == "true" ] || [ "$res_film" == "true" ]; then
    if [ "$residuals" == "false" ]; then
        echo "If res_in or res_film is true, residuals must be true."; exit 1
    fi
fi

# === model & training ==================================================
channel_dims=(32 32 64 128 128 128)
leaky_relu="false"
max_pool="false"

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
sigreg_lambda=0.0

scramble="false"
debug_film="false"

# === input feature count ===============================================
num_bands=${#bands[@]}
in_features=$((num_bands))
if [ "$latlon" == "true" ]; then in_features=$((in_features+4)); fi
if [ "$ch" == "true" ]; then in_features=$((in_features+2)); fi
if [ "$alos" == "true" ]; then in_features=$((in_features+2)); fi
if [ "$lc" == "true" ]; then
    if [ "$ft_cat2vec" == "true" ]; then in_features=$((in_features+6))
    elif [ "$ft_onehot" == "true" ]; then in_features=$((in_features+15))
    elif [ "$ft_sincos" == "true" ]; then in_features=$((in_features+3))
    fi
fi
if [ "$topo" == "true" ]; then
    if [ "$aspect" == "true" ]; then in_features=$((in_features+2)); fi
    if [ "$slope" == "true" ]; then in_features=$((in_features+1)); fi
    if [ "$dem" == "true" ]; then in_features=$((in_features+1)); fi
fi
if [ "$gedi_dates" == "true" ]; then in_features=$((in_features+3)); fi
if [ "$s2_dates" == "true" ]; then
    if [ "$s2_day" == "true" ]; then in_features=$((in_features+1)); fi
    if [ "$s2_doy" == "true" ]; then in_features=$((in_features+2)); fi
fi
if [ "$res_in" == "true" ]; then in_features=$((in_features+1)); fi
if [ "$agb_residuals" == "true" ]; then
    if [ "$agb_res_all" == "true" ]; then in_features=$((in_features+5))
    else in_features=$((in_features+1)); fi
fi
if [ "$aef" == "true" ]; then in_features=$((in_features+64)); fi

# === paths =============================================================
if [ "$AGBD_ENV" == "cluster" ]; then
    model_path=${AGBD_CLUSTER_CKPT}/weights/${arch}
    if [ "$tmpdir" == "true" ]; then dataset_path=$TMPDIR
    else dataset_path=$SCRATCH; fi
    model_name=${model_path}/${JOB_ID}-${MODEL_IDX}
else
    model_path=${AGBD_LOCAL_CKPT}/${arch}
    dataset_path='local'
    model_name=${model_path}/local
fi

# === launch ============================================================
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
                    --num_gpus $NGPUS \
                    --num_cpus $NCPUS \
                    --film $film \
                    --biome_dim $biome_dim \
                    --emb_dim $emb_dim \
                    --region $region \
                    --biome $biome \
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
                    --test_mask $test_mask \
                    --sigreg_lambda $sigreg_lambda \
                    --lite $lite \
                    --lite_eval_big $lite_eval_big \
                    --lite_chunk_size $lite_chunk_size \
                    --aef $aef \
                    --tessera $tessera \
                    --hold_out_region $hold_out_region \
                    --keep_region $keep_region \
                    --stats_hold_out_region $stats_hold_out_region \
                    --stats_keep_region $stats_keep_region \
                    --predict $predict \
                    --ensemble $ensemble \
                    --n_members $n_members \
                    --drop_overlaps $drop_overlaps \
                    --subsample_2020 $subsample_2020 \
                    --years_stats $years_stats
