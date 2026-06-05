#!/bin/bash
# Launch all ablation evaluation jobs locally, one after the other.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Scripts to skip
# - all scripts with eval2020, because will run on the cluster

SKIP=(
    # eval2020 scripts (will run on the cluster)
    aef_2019-2020_eval2020.sh
    aef_2019_eval2020.sh
    aef_2020_eval2020.sh
    aef_agbd_star_2019-2020_eval2020.sh
    aef_agbd_star_2019_eval2020.sh
    aef_agbd_star_2020_eval2020.sh
    agbd_2019-2020_eval2020.sh
    agbd_2019_eval2020.sh
    agbd_2020_eval2020.sh
)

for script in "$SCRIPT_DIR"/*.sh; do
    name="$(basename "$script")"
    [ "$name" = "launch_all.sh" ] && continue
    for s in "${SKIP[@]}"; do [ "$name" = "$s" ] && continue 2; done
    echo "=========================================="
    echo "Running: $(basename "$script")"
    echo "=========================================="
    bash "$script" || true
    echo ""
    echo ""
    echo ""
done
