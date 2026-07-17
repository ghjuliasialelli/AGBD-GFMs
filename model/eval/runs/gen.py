#!/usr/bin/env python3
"""
Generate evaluation shell scripts in this directory.

For each (arch, config) spec, two scripts are emitted:
  - {name}.sh       → lite="false", drop_overlap="true"
  - {name}-Lite.sh  → lite="true",  drop_overlap="false"  (evaluate on AGBD-Lite test)
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
    ("nico_aef",                 "nico_film", "ens_aef"),
    ("nico_aef_lite",            "nico_film", "ens_aef_lite"),
    ("nico_aef_agbd*",           "nico_film", "ens_aef_agbd*"),
    ("nico_aef_agbd*_lite",      "nico_film", "ens_aef_agbd*_lite"),
    ("nico_aef_agbd*_nolatlon",  "nico_film", "ens_aef_agbd*_nolatlon"),
    ("nico_agbd",                "nico_film", "ens_agbd"),
    ("nico_agbd_lite",           "nico_film", "ens_agbd_lite"),
    ("nico_agbd_nolatlon",       "nico_film", "ens_agbd_nolatlon"),
    ("nico_agbd_s2_nolatlon",    "nico_film", "ens_agbd_s2_nolatlon"),
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

    print(f"Generated {len(generated)} scripts in {out_dir}/")
    for s in sorted(generated):
        print(f"  {s}")


if __name__ == "__main__":
    main()
