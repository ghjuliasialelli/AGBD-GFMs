#!/bin/bash
#
# Prepare the `_evalbig` run directories for one fine-tuning regime.
#
#   bash gen_evalbig.sh frozen
#   bash gen_evalbig.sh lora
#
# For each trained run named in ../configs/<regime>.txt this copies the run directory
# to <run>_evalbig and flips its saved config to evaluate on the full AGBD-test set
# instead of the AGBD-Lite test split. Afterwards, `python <regime>/gen.py` writes the
# launchers that point at those copies.
#
# Existing `_evalbig` directories are left alone, so the script is resumable.

set -u

regime="${1:-}"
if [ -z "$regime" ]; then
    echo "usage: bash gen_evalbig.sh <regime>    # regime: frozen | lora" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
configs_file="$script_dir/../configs/${regime}.txt"

if [ ! -f "$configs_file" ]; then
    echo "Error: no such regime list: $configs_file" >&2
    exit 2
fi

# Where the trained run directories live
current_directory=$(pwd)
first_part=$(echo "$current_directory" | cut -d'/' -f2)
if [ "$first_part" == "cluster" ]
then
    path_to_folders="/cluster/work/igp_psr/gsialelli/pangaea-bench"
else
    path_to_folders="/scratch3/gsialelli/pangaea-bench"
fi
base_path=${path_to_folders%/}

# Run directory names, one per line; '#' comments and blank lines skipped.
folders=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%$'\r'}"
    case "$line" in
        ''|'#'*) continue ;;
    esac
    folders+=("$line")
done < "$configs_file"

if [ "${#folders[@]}" -eq 0 ]; then
    echo "Error: $configs_file lists no runs -- collect the $regime run names there first." >&2
    exit 1
fi

echo "Regime: $regime  (${#folders[@]} runs, from ${configs_file})"

# Iterate over the folders, copy them, and modify the config.yaml file
for dir_name in "${folders[@]}"; do
    src_path="$base_path/$dir_name"
    new_dir_path="${src_path}_evalbig"

    if [ -d "$new_dir_path" ]; then
        echo "Skipping: $new_dir_path already exists."
        continue
    fi

    if [ -d "$src_path" ]; then
        cp -r "$src_path" "$new_dir_path"
        echo "Created: $new_dir_path"
        config_file="$new_dir_path/configs/config.yaml"
        if [ -f "$config_file" ]; then
            perl -i -pe 's/eval_big: false/eval_big: true/g' "$config_file"
            perl -i -pe 's/(wandb_run_id:\s+)(\S+)/$1$2_evalbig/g' "$config_file"
            echo "  Successfully updated: $config_file"
        else
            echo "  Warning: $config_file not found."
        fi
    else
        echo "Error: Directory $src_path not found. Skipping..."
    fi
done

echo "Process complete!"
echo "Next: python ${script_dir}/${regime}/gen.py"
