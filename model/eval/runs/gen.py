#!/usr/bin/env python3
"""
Generate evaluation shell scripts in this directory.

For each (arch, config) spec, two scripts are emitted:
  - {name}.sh       → lite="false", drop_overlap="true"
  - {name}-Lite.sh  → lite="true",  drop_overlap="false"  (evaluate on AGBD-Lite test)
and, for each REGION_SPECS entry, {name}-regions.sh: array 1-3, full AGBD test restricted to one region per task
(SouthAsia / Africa / SouthAmerica), for the leave-one-region-out Lite runs and their baselines.
"""

from pathlib import Path

# (filename_stem, arch, config)
SPECS = [
    ("lp_aef",                   "lp",        "aef"),
    ("lp_aef_lite",              "lp",        "aef_lite"),
    ("lp_tessera_lite",          "lp",        "tessera_lite"),
    ("mlp_aef",                  "mlp",       "aef"),
    ("mlp_aef_lite",             "mlp",       "aef_lite"),
    ("mlp_tessera_lite",         "mlp",       "tessera_lite"),
    ("nico_tessera_lite",        "nico_film", "ens_tessera_lite"),
    ("lp_olmo_lite",             "lp",        "olmo_lite"),
    ("mlp_olmo_lite",            "mlp",       "olmo_lite"),
    ("nico_olmo_lite",           "nico_film", "ens_olmo_lite"),
    ("nico_olmo64_lite",         "nico_film", "ens_olmo64_lite"),
    ("nico_aef",                 "nico_film", "ens_aef"),
    ("nico_aef_latlon",          "nico_film", "ens_aef_latlon"),
    ("nico_aef_lite",            "nico_film", "ens_aef_lite"),
    ("nico_aef_agbd*",           "nico_film", "ens_aef_agbd*"),
    ("nico_aef_agbd*_lite",      "nico_film", "ens_aef_agbd*_lite"),
    ("nico_aef_agbd*_nolatlon",  "nico_film", "ens_aef_agbd*_nolatlon"),
    ("nico_agbd",                "nico_film", "ens_agbd"),
    ("nico_agbd_lite",           "nico_film", "ens_agbd_lite"),
    ("nico_agbd_nolatlon",       "nico_film", "ens_agbd_nolatlon"),
    ("nico_agbd_s2_nolatlon",    "nico_film", "ens_agbd_s2_nolatlon"),
    ("nico_aef_olmo_lite",       "nico_film", "ens_aef_olmo_lite"),
    ("nico_aef_lite_newstats",      "nico_film", "ens_aef_lite_newstats"),       # --new_AEF_stats (exact Lite AEF stats)
    ("nico_aef_olmo_lite_newstats", "nico_film", "ens_aef_olmo_lite_newstats"),  # --new_AEF_stats (exact Lite AEF stats)
    ("nico_aef_olmo64_lite",       "nico_film", "ens_aef_olmo64_lite"),
    ("nico_aef_tessera_lite",      "nico_film", "ens_aef_tessera_lite"),
    ("nico_olmo_tessera_lite",     "nico_film", "ens_olmo_tessera_lite"),
    ("nico_aef_olmo_tessera_lite", "nico_film", "ens_aef_olmo_tessera_lite"),
]

TEMPLATE = '''#!/bin/bash

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
arch="{arch}"
mode="test"
bs=256
offset="false"
min_offset=0
max_offset=0
return_region="true"
skip_preds="false"
lite="{lite}" # evaluate on AGBD-Lite test
drop_overlap="{drop_overlap}"
stats_hold_out_region="None" # region whose stats file to load; "None" for global stats
stats_keep_region="false"    # whether the stats file is for the kept region
years_stats="2019-2020"      # years the models were trained on (for picking the stats file)

# if lite is true and return_region is false, raise an error
if [ "$lite" == "true" ] && [ "$return_region" == "false" ]; then
    echo "Error: If lite is true, return_region must be true."
    exit 1
fi

echo "Years: ${{years[@]}}"
echo "Mode: $mode"
echo "Architecture: $arch"

config="{config}" # best, lite, aef, aef_lite
readarray -t models < "eval/configs/evaluation_${{arch}}_${{config}}.txt" # Read the file into an array
echo "Models: ${{models[@]}} (arch: $arch, config: $config)"

##################################################################################################################

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$AGBD_ENV" == "cluster" ]; then
    echo "Running on a cluster"

    module load stack/2024-06 gcc/12.2.0
    module load stack/2024-06 python_cuda/3.11.6
    source ${{AGBD_CLUSTER_VENV}}

    dataset_path=$TMPDIR
    plot_folder="${{AGBD_CLUSTER_PLOTS}}/"

elif [ "$AGBD_ENV" != "cluster" ]; then
    echo "Running on a local machine"
    dataset_path='local'
    plot_folder="${{AGBD_LOCAL_PLOTS}}/"

else
    echo "Environment unknown"
fi


# Launch evaluation ##############################################################################################

python eval.py  --dataset_path "$dataset_path" --arch "$arch" --models "${{models[@]}}" --years "${{years[@]}}" \\
                --plot_folder "$plot_folder" --mode "$mode" --offset "$offset" --min_offset "$min_offset" \\
                --max_offset "$max_offset" --return_region "$return_region" --skip_preds "$skip_preds" \\
                --lite "$lite" --bs "$bs" --drop_overlap "$drop_overlap" \\
                --stats_hold_out_region "$stats_hold_out_region" --stats_keep_region "$stats_keep_region" \\
                --years_stats "$years_stats"
'''


# Region-restricted evaluation (full AGBD test, one region per array task) for the leave-one-region-out AGBD-Lite
# runs and their in-distribution baselines. (filename_stem, arch, config, mode)
#   mode "holdout": the config lists one model per region, line t = the model that held out regions[t-1]
#                   (so a relaunched task with a new job ID is just a different line).
#   mode "fixed":   in-distribution baseline; a 1-line config is used for all three regions (the same config file
#                   as the global {name}.sh / {name}-Lite.sh launchers), a 3-line one is read per task like holdout.
REGION_SPECS = [
    ("nico_aef_lite_holdout",         "nico_film", "aef_lite_holdout",           "holdout"),
    ("nico_aef_olmo_lite_holdout",    "nico_film", "aef_olmo_lite_holdout",      "holdout"),
    ("nico_aef_lite_newstats",        "nico_film", "ens_aef_lite_newstats",      "fixed"),
    ("nico_aef_olmo_lite_newstats",   "nico_film", "ens_aef_olmo_lite_newstats", "fixed"),
    ("nico_aef_lite",                 "nico_film", "ens_aef_lite",               "fixed"),
    ("nico_aef_olmo_lite",            "nico_film", "ens_aef_olmo_lite",          "fixed"),
    ("nico_tessera_lite_holdout",     "nico_film", "tessera_lite_holdout",       "holdout"),  # added 2026-11-25
    ("nico_tessera_lite",             "nico_film", "ens_tessera_lite",           "fixed"),    # baseline 64466721-1
]

# @ARCH@ / @CONFIG@ / @MODE@ are substituted with str.replace (the bash below is full of braces, so no str.format).
REGION_TEMPLATE = r'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/eval-%A-%a.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/eval-%A-%a.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=eval
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --array=1-3

# GENERATED by eval/runs/gen.py (REGION_SPECS) -- edit the generator, not this file. Run from model/:
#     sbatch eval/runs/<this file>              # all three regions
#     sbatch --array=2 eval/runs/<this file>    # one region (task t = regions[t-1])
#
# Region-restricted evaluation on the full AGBD test set for the leave-one-region-out AGBD-Lite runs
# (model/runs/nico/*_lite_holdout.sh) and their in-distribution baselines.
# Models come from eval/configs/evaluation_<arch>_<config>.txt:
#   holdout: line t = the model that held out regions[t-1]; normalised with the stats computed WITHOUT that region.
#   fixed:   in-distribution baseline; 1 line = same model for every region, else line t; global Lite stats
#            (original or _exact comes from the model's own config via --new_AEF_stats, nothing to pass).
#
# --region is passed EXPLICITLY on purpose: eval.py copies hold_out_region from the training config, so a held-out
# model evaluated without it would drop exactly the region we want to test. With --region R --keep_region true, both
# the held-out and the in-distribution models are scored on the same full-AGBD test footprints of R (tile-based
# region, as in training; NOT the GEDI region_cla stored in the results' `regions` dataset).
# --stats_hold_out_region is eval-only (not copied from the training config), so it is set here to what each model was
# trained with.
# Output: eval results folder, <arch>_<id>_<id>_<id>_2019-2020_nooverlap_<Region>.h5

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
arch="@ARCH@"
config="@CONFIG@"
model_mode="@MODE@" # holdout | fixed
mode="test"
bs=256
offset="false"
min_offset=0
max_offset=0
return_region="true"
skip_preds="false"
lite="false" # full AGBD test set
drop_overlap="true"
stats_keep_region="false" # stats_hold_out_region is set below: must match what the model was TRAINED with
years_stats="2019-2020"
regions=("SouthAsia" "Africa" "SouthAmerica") # same order as geo_regions in the *_lite_holdout.sh launchers

# --- which model, which region -----------------------------------------------------------------------------------
config_file="eval/configs/evaluation_${arch}_${config}.txt"
if [ ! -f "$config_file" ]; then echo "Config file not found: $config_file (run from model/)" >&2; exit 1; fi
readarray -t lines < "$config_file"
config_models=()
for l in "${lines[@]}"; do l="${l//[[:space:]]/}"; [ -n "$l" ] && config_models+=("$l"); done
task=${SLURM_ARRAY_TASK_ID:-1}
if [ "$task" -lt 1 ] || [ "$task" -gt ${#regions[@]} ]; then echo "Array task $task out of range 1-${#regions[@]}" >&2; exit 1; fi
region=${regions[$(( task - 1 ))]}
if [ "$model_mode" == "holdout" ]; then
    if [ ${#config_models[@]} -ne ${#regions[@]} ]; then
        echo "$config_file must list ${#regions[@]} models (one per region: ${regions[*]}), found ${#config_models[@]}" >&2; exit 1
    fi
    model=${config_models[$(( task - 1 ))]}
    stats_hold_out_region="$region" # trained with the stats of everything but $region (geo_region_stats="true")
elif [ "$model_mode" == "fixed" ]; then
    if [ ${#config_models[@]} -eq 1 ]; then model=${config_models[0]}
    elif [ ${#config_models[@]} -eq ${#regions[@]} ]; then model=${config_models[$(( task - 1 ))]}
    else echo "$config_file must list 1 or ${#regions[@]} models, found ${#config_models[@]}" >&2; exit 1; fi
    stats_hold_out_region="None" # in-distribution model: global Lite stats
else
    echo "Unknown model_mode '$model_mode' (holdout | fixed)" >&2; exit 1
fi
if ! [[ "$model" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "Model '$model' (task $task, $region) is not a <JOBID>-<task> ID: fill in $config_file first" >&2; exit 1
fi
models=("$model")
echo "Model: $model  Region: $region  Stats: $stats_hold_out_region  (config: $config, mode: $model_mode, task: $task)"

##################################################################################################################

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
                --lite "$lite" --bs "$bs" --drop_overlap "$drop_overlap" \
                --region "$region" --keep_region "true" \
                --stats_hold_out_region "$stats_hold_out_region" --stats_keep_region "$stats_keep_region" \
                --years_stats "$years_stats"
'''


def main():
    out_dir = Path(__file__).resolve().parent
    generated = []

    for stem, arch, config in SPECS:
        variants = [("", "false", "true")]
        if "_lite" in stem:
            variants.append(("-Lite", "true", "false"))
        for suffix, lite, drop_overlap in variants:
            path = out_dir / f"{stem}{suffix}.sh"
            path.write_text(TEMPLATE.format(
                arch=arch, config=config, lite=lite, drop_overlap=drop_overlap,
            ))
            generated.append(str(path))

    for stem, arch, config, model_mode in REGION_SPECS:
        path = out_dir / f"{stem}-regions.sh"
        path.write_text(REGION_TEMPLATE.replace("@ARCH@", arch).replace("@CONFIG@", config).replace("@MODE@", model_mode))
        generated.append(str(path))

    print(f"Generated {len(generated)} scripts in {out_dir}/")
    for s in sorted(generated):
        print(f"  {s}")


if __name__ == "__main__":
    main()
