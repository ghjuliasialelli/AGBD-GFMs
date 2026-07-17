#!/usr/bin/env python3
"""
Generate evaluation shell scripts for ablation studies.

Generates scripts for:
1. Geographical generalization: for each config (aef, agbd, aef_agbd*) and region
   (SouthAsia, Africa, SouthAmerica), evaluate models trained "without" the region,
   "only" on the region, and the "general" model (trained on all regions).
2. Temporal generalization: for each config, evaluate models trained on 2019-2020
   and 2019-only, both tested on 2020 data.

Geographical scripts use --region to only load the relevant region's test data.
Temporal scripts evaluate on the full test set.
"""

from pathlib import Path

# Regions: index 1=SouthAsia, 2=Africa, 3=SouthAmerica (matching SLURM array task IDs)
REGIONS = ["SouthAsia", "Africa", "SouthAmerica"]

# ============================================================================================
# Job IDs for trained models
# For "without" and "only": dict of {task_id: "jobid-taskid"} since they are array jobs
# For "2019": single model string (array=1-1)
# For "gen": fill in with the general model (trained on all regions, 2019-2020) job IDs
# ============================================================================================
JOBS = {
    "aef": {
        "without": {1: "65148996-1", 2: "65148996-2", 3: "65148996-3"}, # {1: "64424736-1", 2: "64424736-2", 3: "64424736-3"},
        "only":    {1: "64424763-1", 2: "64424763-2", 3: "64424763-3"},
        "2019":    "64424767-1",
        "2020":    "64508286-1",
        "2019-2020": "64424796-1",
        "gen":     "59620113-1",
    },
    "agbd": {
        "without": {1: "60736796-1", 2: "60736796-2", 3: "60866344-3"},
        "only":    {1: "60736794-1", 2: "60736794-2", 3: "60736794-3"},
        "2019":    "60736792-1",
        "2020":    "64283650-1",
        "2019-2020": "64283705-1",
        "gen":     "60856555-1",
    },
    "aef_agbd_star": {
        "without": {1: "64424805-1", 2: "64424805-2", 3: "64424805-3"},
        "only":    {1: "64424821-1", 2: "64424821-2", 3: "64508409-3"},
        "2019":    "64508484-1",
        "2020":    "64424837-1",
        "2019-2020": "64424853-1",
        "gen":     "60824638-1",
    },
}

# Display name mapping (for filenames)
CONFIG_DISPLAY = {
    "aef": "aef",
    "agbd": "agbd",
    "aef_agbd_star": "aef_agbd_star",
}

EVAL_TEMPLATE = '''#!/bin/bash

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
years=({years})
arch="nico_film"
mode="test"
bs=256
offset="false"
min_offset=0
max_offset=0
return_region="true"
skip_preds="false"
lite="false"
drop_overlap="{drop_overlap}"
models=({models})

# if lite is true and return_region is false, raise an error
if [ "$lite" == "true" ] && [ "$return_region" == "false" ]; then
    echo "Error: If lite is true, return_region must be true."
    exit 1
fi

echo "Years: ${{years[@]}}"
echo "Mode: $mode"
echo "Architecture: $arch"
echo "Models: ${{models[@]}}"

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
                --lite "$lite" --bs "$bs" --drop_overlap "$drop_overlap" --force "true" {region_flag} {stats_flags}
'''


def main():
    # Emit next to this generator (the tracked eval_ablations/ dir in the repo).
    # Previously this pointed at a personal path outside the repo, so regenerating
    # silently left the committed scripts untouched.
    script_dir = Path(__file__).resolve().parent
    script_dir.mkdir(parents=True, exist_ok=True)

    generated_scripts = []

    for config_key, ablations in JOBS.items():
        display = CONFIG_DISPLAY[config_key]
        drop_overlap = "true"  # consistent across all configs for fair comparison

        # ===== GEOGRAPHICAL GENERALIZATION =====

        # "without" and "only" models: one model per region.
        # Stats must come from the model's training distribution, not the eval region:
        #   - "without X": trained on everything except X -> stats hold out X, keep_region=false
        #   - "only X":    trained only on X             -> stats hold out X, keep_region=true
        for ablation_type in ["without", "only"]:
            models_dict = ablations[ablation_type]
            stats_keep = "true" if ablation_type == "only" else "false"
            for region_idx, region in enumerate(REGIONS, 1):
                model_name = models_dict[region_idx]
                name = f"{display}_{ablation_type}_{region}"

                script_path = script_dir / f"{name}.sh"
                script_path.write_text(EVAL_TEMPLATE.format(
                    years="2019 2020",
                    drop_overlap=drop_overlap,
                    models=model_name,
                    region_flag=f'--region "{region}" --keep_region "true"',
                    stats_flags=f'--stats_hold_out_region "{region}" --stats_keep_region "{stats_keep}" --years_stats "2019-2020"',
                ))
                generated_scripts.append(str(script_path))

        # "gen" model: trained on all regions -> global stats (stats_hold_out_region=None)
        gen_model = ablations["gen"]
        for region in REGIONS:
            name = f"{display}_gen_{region}"

            script_path = script_dir / f"{name}.sh"
            script_path.write_text(EVAL_TEMPLATE.format(
                years="2019 2020",
                drop_overlap=drop_overlap,
                models=gen_model,
                region_flag=f'--region "{region}" --keep_region "true"',
                stats_flags='--stats_hold_out_region "None" --stats_keep_region "false" --years_stats "2019-2020"',
            ))
            generated_scripts.append(str(script_path))

        # ===== TEMPORAL GENERALIZATION =====
        # years_stats must reflect the years the model was trained on, not the eval years.

        # "2019" model evaluated on 2020 only
        model_2019 = ablations["2019"]
        name_2019 = f"{display}_2019_eval2020"
        script_path = script_dir / f"{name_2019}.sh"
        script_path.write_text(EVAL_TEMPLATE.format(
            years="2020",
            drop_overlap=drop_overlap,
            models=model_2019,
            region_flag="",
            stats_flags='--stats_hold_out_region "None" --stats_keep_region "false" --years_stats "2019"',
        ))
        generated_scripts.append(str(script_path))

        # "2020" model evaluated on 2020 only
        model_2020 = ablations["2020"]
        name_2020 = f"{display}_2020_eval2020"
        script_path = script_dir / f"{name_2020}.sh"
        script_path.write_text(EVAL_TEMPLATE.format(
            years="2020",
            drop_overlap=drop_overlap,
            models=model_2020,
            region_flag="",
            stats_flags='--stats_hold_out_region "None" --stats_keep_region "false" --years_stats "2020"',
        ))
        generated_scripts.append(str(script_path))

        # "2019-2020" model evaluated on 2020 only
        model_2019_2020 = ablations["2019-2020"]
        name_2019_2020 = f"{display}_2019-2020_eval2020"
        script_path = script_dir / f"{name_2019_2020}.sh"
        script_path.write_text(EVAL_TEMPLATE.format(
            years="2020",
            drop_overlap=drop_overlap,
            models=model_2019_2020,
            region_flag="",
            stats_flags='--stats_hold_out_region "None" --stats_keep_region "false" --years_stats "2019-2020"',
        ))
        generated_scripts.append(str(script_path))


    # Print summary
    print(f"Generated {len(generated_scripts)} eval scripts in {script_dir}/")
    for s in sorted(generated_scripts):
        print(f"  {s}")


if __name__ == "__main__":
    main()
