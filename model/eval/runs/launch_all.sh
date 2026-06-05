#!/bin/bash
# Launch all evaluation jobs locally, one after the other.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$SCRIPT_DIR"/*.sh; do
    name="$(basename "$script")"
    [ "$name" = "launch_all.sh" ] && continue
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    bash "$script" || true
    echo ""
    echo ""
    echo ""
done
