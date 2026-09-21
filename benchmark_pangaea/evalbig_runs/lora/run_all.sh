#!/bin/bash
#
# Run this regime's evalbig launchers locally, one after another, each logging to
# logs/<encoder>.txt. (The launchers carry #SBATCH headers for the cluster; run with
# `bash` they are just shell scripts, which is how these were run interactively.)
#
#   bash run_all.sh                             # every launcher in this directory
#   bash run_all.sh spectralgpt prithvi2_100m   # only the named encoders
#
# Serial on purpose: concurrent runs compete for GPU and host memory.
#
# Run from the pangaea-bench fork checkout -- the launchers invoke `pangaea/run.py`
# by relative path.

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$script_dir/logs"

if [ "$#" -gt 0 ]; then
    encoders=("$@")
else
    encoders=()
    for f in "$script_dir"/*.sh; do
        name=$(basename "$f" .sh)
        [ "$name" = "run_all" ] && continue
        encoders+=("$name")
    done
fi

if [ "${#encoders[@]}" -eq 0 ]; then
    echo "No launchers in $script_dir -- run \`python gen.py\` first." >&2
    exit 1
fi

for encoder in "${encoders[@]}"; do
    launcher="$script_dir/${encoder}.sh"
    log="$script_dir/logs/${encoder}.txt"
    if [ ! -f "$launcher" ]; then
        echo "Missing launcher: $launcher -- skipping." >&2
        continue
    fi
    echo "=== ${encoder} -> ${log}"
    bash "$launcher" > "$log" 2>&1
    echo "    exit=$?"
done
