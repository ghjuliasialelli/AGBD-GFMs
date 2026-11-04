#!/bin/bash
# Check whether OlmoEarth has finished backfilling --year for every AGBD-Lite sample.
# Appends a dated summary to $LOG and exits 0 only when all three splits are 100% covered,
# so it can gate the OLMO-Lite build. Metadata only (provenance parquet, ~7 MB when it changed).
#
#   bash check_backfill.sh            # or from crontab, e.g. every 2 days at 07:13:
#   13 7 */2 * * bash /scratch3/gsialelli/AGBD-GFMs/data/olmoearth/check_backfill.sh
#
# env: tessera

YEAR="${YEAR:-2020}"
LITE="${LITE:-/scratch3/gsialelli/patches/AGBD-Lite}"
CACHE="${CACHE:-/scratch3/gsialelli/patches/OLMO/provenance}"
LOG="${LOG:-/scratch3/gsialelli/patches/OLMO/backfill_${YEAR}.log}"
CONDA="${CONDA:-/scratch2/gsialelli/miniconda3/bin/conda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$LOG")"
out=$(cd "$HERE" && "$CONDA" run -n tessera python -u audit.py coverage --year "$YEAR" --cache-dir "$CACHE" \
      --input "$LITE/AGBD-Lite-train.h5" "$LITE/AGBD-Lite-val.h5" "$LITE/AGBD-Lite-test.h5" 2>&1)
status=$?
{
    echo "=== $(date -Is)  exit=$status"
    echo "$out" | grep -E "coverage|PASS|FAIL|Error|Traceback"
} >> "$LOG"
echo "$out" | grep -E "coverage|PASS|FAIL|Error"
exit $status
