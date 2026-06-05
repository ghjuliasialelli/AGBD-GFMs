#!/bin/bash


# Path to the configs
current_directory=$(pwd)
first_part=$(echo "$current_directory" | cut -d'/' -f2)
if [ "$first_part" == "cluster" ]
then
    path_to_folders="/cluster/work/igp_psr/gsialelli/pangaea-bench"
else
    path_to_folders="/scratch3/gsialelli/pangaea-bench"
fi
base_path=${path_to_folders%/}


# Configurations for which to do it, read from configs.txt
readarray -t folders < configs.txt

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