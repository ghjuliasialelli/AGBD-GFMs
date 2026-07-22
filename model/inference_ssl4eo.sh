#!/bin/bash
#
# Run the SSL4EO-MoCo (fine-tuned on AGBD full) AGB maps for the three map-figure tiles.
#
# Two stages in two conda envs, because `process_S2_tile` needs skimage (absent from
# pangaea-bench) and the model needs torch 2.4 + hydra + timm (absent from agbd):
#   A. cache_s2_window.py   (agbd)          -- extract the S2 window the AEF panel defines
#   B. inference_ssl4eo.py  (pangaea-bench) -- run the model over it
#
# Everything is SERIALISED. Concurrent loads have OOMed this box repeatedly, and stage A alone
# holds a whole 10980^2 Sentinel-2 tile. Do not parallelise the tiles.
#
# Resumable: both stages skip when their output exists, so a kill costs one tile, not the run.
# Pass --overwrite (below) to force a recompute.
#
# Usage:
#     bash model/inference_ssl4eo.sh            # all three tiles
#     TILES=49SBT bash model/inference_ssl4eo.sh   # just one
#
set -u -o pipefail

CONDA_ROOT=/scratch2/gsialelli/miniconda3
AGBD_PY="$CONDA_ROOT/envs/agbd/bin/python"
PB_ENV="$CONDA_ROOT/envs/pangaea-bench"
PB_PY="$PB_ENV/bin/python"

REPO=/scratch3/gsialelli/AGBD-GFMs
PANGAEA=/scratch3/gsialelli/pangaea-bench
OUT_ROOT=/scratch3/gsialelli/ssl4eo_maps
LOG_DIR="$OUT_ROOT/logs"

# Centre-pixel-only, 30 m. keep_px MUST stay 1: the off-centre outputs of this model are a
# relu-clamped aliasing pattern keyed to the patch frame, not biomass. See inference_ssl4eo.py.
KEEP_PX=1
STRIDE=3

# Sentinel-2 product per tile. All three are pinned to the SAME acquisition the AGBD-features panel
# used, so the rows of the figure differ by the model and nothing else -- see the filenames under
# predictions_maps/nico_film/59620098-1_*.
# 32TPT was re-pinned 2026-07-22 from 20200908 (R065) to 20200915 (R022): an AGBD-features 32TPT
# panel appeared (predictions_maps/.../S2A_..._20200915T101031_..._T32TPT_..._NA.tif, Jul 21 22:30),
# and make_map_figure.py derives both the AGBD row AND the S2 RGB row from THAT product, so the
# earlier 20200908 SSL4EO panel was a 7-day/different-orbit scene mismatch on the Europe column.
# The 20200908 cache/pred are archived under $OUT_ROOT/archive_32TPT_20200908/.
declare -A PRODUCT=(
  [49SBT]=S2A_MSIL2A_20200826T032541_N0500_R018_T49SBT_20230418T063839
  [59GPM]=S2B_MSIL2A_20200223T222539_N0500_R029_T59GPM_20230515T102828
  [32TPT]=S2A_MSIL2A_20200915T101031_N0500_R022_T32TPT_20230416T031135
)

TILES="${TILES:-49SBT 59GPM 32TPT}"

mkdir -p "$OUT_ROOT/cache" "$OUT_ROOT/preds" "$LOG_DIR"

for tile in $TILES ; do

    product="${PRODUCT[$tile]:-}"
    if [ -z "$product" ] ; then
        echo "!! No product pinned for $tile -- add one to PRODUCT above. Skipping."
        continue
    fi

    cache="$OUT_ROOT/cache/$tile.npz"
    pred="$OUT_ROOT/preds/$tile.tif"

    echo "=============================================================="
    echo "$tile  ($product)"
    echo "=============================================================="

    # Stage A -- extract the window (agbd env) ---------------------------------------------------
    if [ -f "$cache" ] ; then
        echo "[A] $cache exists, skipping."
    else
        echo "[A] caching S2 window -> $cache"
        ( cd "$REPO" && "$AGBD_PY" -u model/cache_s2_window.py \
            --tile_name "$tile" --product_name "$product" --out "$cache" ) \
            2>&1 | tee "$LOG_DIR/${tile}_cache.log"
        if [ ! -f "$cache" ] ; then
            echo "!! Stage A produced no cache for $tile; see $LOG_DIR/${tile}_cache.log. Skipping."
            continue
        fi
    fi

    # Stage B -- run the model (pangaea-bench env) ------------------------------------------------
    if [ -f "$pred" ] ; then
        echo "[B] $pred exists, skipping."
        continue
    fi
    echo "[B] running inference -> $pred  (keep_px=$KEEP_PX, stride=$STRIDE, ~1.7 h)"
    # PROJ_LIB is set per-command, never exported globally: it is per-environment, and pointing
    # one env at another's share/proj has broken a third env in this pipeline before.
    ( cd "$PANGAEA" && PROJ_LIB="$PB_ENV/share/proj" PANGAEA_ROOT="$PANGAEA" \
        "$PB_PY" -u "$REPO/model/inference_ssl4eo.py" \
            --cache "$cache" --out "$pred" \
            --keep_px "$KEEP_PX" --stride "$STRIDE" ) \
        2>&1 | tee "$LOG_DIR/${tile}_infer.log"

done

echo
echo "Done. Predictions in $OUT_ROOT/preds, per-run stats in *_stats.json, logs in $LOG_DIR."
