# =============================================================================
# AGBD-GFMs configuration
# =============================================================================
#
# Single source of truth for machine-specific paths and settings. Both the shell
# launchers and the Python entry points read this file:
#
#   bash    :  source config.sh          # values land in the environment
#   python  :  from config import ...     # config.py sources this same file
#
# ---- Profiles ---------------------------------------------------------------
# There are three profiles. Which one applies is detected automatically, but you
# can force it with AGBD_PROFILE=<name>:
#
#   local   The portable default. ALL data lives inside this repository, under
#           ./data (see the AGBD_LOCAL_* block). This is what an external user
#           gets; a download script populates ./data with the large files.
#   pfpc28  The authors' workstation (data spread across /scratch3/gsialelli).
#   euler   The authors' SLURM cluster (data under /cluster/work/igp_psr, staged
#           into $TMPDIR by the launchers).
#
# Detection: hostname pf-pc28 -> pfpc28; a working directory under /cluster ->
# euler; under /scratch3 -> pfpc28; anything else -> local.
#
# `local` and `pfpc28` run the same code path (AGBD_ENV=local): the Python entry
# points read the data directly from the AGBD_LOCAL_* paths. `euler` runs the
# cluster path (AGBD_ENV=cluster): the launchers stage data into $TMPDIR first.
#
# ---- Overrides --------------------------------------------------------------
# Every value is `${VAR:-default}`, so exporting a variable of the same name
# overrides it without editing this file -- the environment always wins:
#
#   AGBD_LOCAL_H5=/my/patches bash model/runs/nico/agbd.sh
#
# ---- Secrets ----------------------------------------------------------------
# SECRETS DO NOT BELONG HERE -- this file is committed to git.
# The wandb API key is read by wandb itself from ~/.netrc or $WANDB_API_KEY.
# Never write a key into this file.
# =============================================================================


# ----- Weights & Biases ------------------------------------------------------
# Entity and project only. NOT the API key (see above).
AGBD_WANDB_ENTITY="${WANDB_ENTITY:-gs-tp-biomass}"

# model/eval.py queries wandb for two things: each run's checkpoint directory, and the
# training config it needs to rebuild the model (the checkpoints store a state_dict but
# no hyper_parameters, so the config cannot be recovered from the weights alone).
# Set to "false" to skip the lookup entirely: checkpoints are then loaded from
# AGBD_*_CKPT, and the configs come from the offline cache in model/eval/wandb_cache/.
# That is the setting for anyone without access to the entity above -- i.e. anyone
# evaluating the released checkpoints. Regenerate the cache with model/export_wandb_cache.py.
AGBD_WANDB_LOOKUP="${AGBD_WANDB_LOOKUP:-true}"


# ----- Repository root -------------------------------------------------------
# Absolute path to this repo (the directory holding config.sh). The `local`
# profile hangs every data path off it, so the repo is fully self-contained.
AGBD_REPO="${AGBD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"


# ----- Profile selection -----------------------------------------------------
if [ -z "${AGBD_PROFILE:-}" ]; then
    case "$(hostname 2>/dev/null)" in
        pf-pc28|pf-pc28.*) AGBD_PROFILE="pfpc28" ;;
        *)
            case "$(pwd)" in
                /cluster/*)  AGBD_PROFILE="euler"  ;;
                /scratch3/*) AGBD_PROFILE="pfpc28" ;;
                *)           AGBD_PROFILE="local"  ;;
            esac ;;
    esac
fi


# ----- CLUSTER (euler) profile ----------------------------------------------
# Always defined; only used when AGBD_ENV=cluster. The launchers stage these
# into $TMPDIR, so the paths here are the *source* locations on /cluster/work.
AGBD_CLUSTER_WORK="${AGBD_CLUSTER_WORK:-/cluster/work/igp_psr/gsialelli}"
AGBD_CLUSTER_SCRATCH="${AGBD_CLUSTER_SCRATCH:-/cluster/scratch/gsialelli}"

# Python environment activated by the cluster launchers.
AGBD_CLUSTER_VENV="${AGBD_CLUSTER_VENV:-${AGBD_CLUSTER_WORK}/EcosystemAnalysis/Models/Biomes/agbd/bin/activate}"

AGBD_CLUSTER_H5="${AGBD_CLUSTER_H5:-${AGBD_CLUSTER_WORK}/Data/patches}"
AGBD_CLUSTER_NORM="${AGBD_CLUSTER_NORM:-${AGBD_CLUSTER_WORK}/Data/patches}"
AGBD_CLUSTER_MAP="${AGBD_CLUSTER_MAP:-${AGBD_CLUSTER_WORK}/Data/patches}"

# NOTE: no trailing /weights here -- the cluster layout genuinely differs from
# the local one on this key. The launchers append /weights/${arch} themselves.
AGBD_CLUSTER_CKPT="${AGBD_CLUSTER_CKPT:-${AGBD_CLUSTER_WORK}/EcosystemAnalysis/Models/Biomes}"

AGBD_CLUSTER_EMBEDDINGS="${AGBD_CLUSTER_EMBEDDINGS:-${AGBD_CLUSTER_WORK}/EcosystemAnalysis/Models/Baseline/cat2vec}"

AGBD_CLUSTER_AEF="${AGBD_CLUSTER_AEF:-${AGBD_CLUSTER_WORK}/Data/AEF}"
AGBD_CLUSTER_AEF_H5="${AGBD_CLUSTER_AEF_H5:-${AGBD_CLUSTER_WORK}/Data/patches/AEF}"
AGBD_CLUSTER_AEF_NORM="${AGBD_CLUSTER_AEF_NORM:-${AGBD_CLUSTER_WORK}/Data/patches/AEF}"

AGBD_CLUSTER_TESSERA_H5="${AGBD_CLUSTER_TESSERA_H5:-${AGBD_CLUSTER_WORK}/Data/patches/TESSERA}"
AGBD_CLUSTER_TESSERA_NORM="${AGBD_CLUSTER_TESSERA_NORM:-${AGBD_CLUSTER_WORK}/Data/patches/TESSERA}"

AGBD_CLUSTER_SPLITS="${AGBD_CLUSTER_SPLITS:-${AGBD_CLUSTER_WORK}/BiomassDatasetCreation/Data/download_Sentinel/biomes_split}"

# Raw per-tile rasters -- only inference_agbd.py (AGBD-features tile inference) reads these.
# DEM lives alongside ALOS. REGION holds s2_tile_to_region-v3.pkl.
AGBD_CLUSTER_TILES="${AGBD_CLUSTER_TILES:-${AGBD_CLUSTER_WORK}/Data/S2_L2A}"
AGBD_CLUSTER_ALOS="${AGBD_CLUSTER_ALOS:-${AGBD_CLUSTER_WORK}/Data/ALOS}"
AGBD_CLUSTER_DEM="${AGBD_CLUSTER_DEM:-${AGBD_CLUSTER_WORK}/Data/ALOS}"
AGBD_CLUSTER_LC="${AGBD_CLUSTER_LC:-${AGBD_CLUSTER_WORK}/Data/LC}"
AGBD_CLUSTER_REGION="${AGBD_CLUSTER_REGION:-${AGBD_CLUSTER_WORK}/Data}"

# Auxiliary files the launchers stage into $TMPDIR.
AGBD_CLUSTER_AGB="${AGBD_CLUSTER_AGB:-${AGBD_CLUSTER_WORK}/Data/AGB}"
AGBD_CLUSTER_HELPER="${AGBD_CLUSTER_HELPER:-${AGBD_CLUSTER_WORK}/EcosystemAnalysis/Models/Biomes/helper}"

# Predecessor repo (AGBD-GFM, no "s") that still holds a couple of derived files
# on the cluster: agbd-lite/mapping_lite_to_og.pkl and aef-dwn/AEF_overlaps.pkl.
# Copies of both are tracked in this repo (data/agbd_lite/, data/aef/).
AGBD_CLUSTER_AUX="${AGBD_CLUSTER_AUX:-${AGBD_CLUSTER_WORK}/AGBD-GFM}"

# Where eval.py writes its plots on the cluster.
AGBD_CLUSTER_PLOTS="${AGBD_CLUSTER_PLOTS:-${AGBD_CLUSTER_WORK}/EcosystemAnalysis/Models/Biomes/eval_plots}"


# ----- LOCAL profile (local / pfpc28) ---------------------------------------
# Defaults below describe the portable `local` layout (everything under ./data).
# The pfpc28 case overrides them with the authors' /scratch3 layout.
case "$AGBD_PROFILE" in

    pfpc28)
        AGBD_ENV="local"
        AGBD_LOCAL_ROOT="${AGBD_LOCAL_ROOT:-/scratch3/gsialelli}"
        AGBD_LOCAL_H5="${AGBD_LOCAL_H5:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_NORM="${AGBD_LOCAL_NORM:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_MAP="${AGBD_LOCAL_MAP:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_CKPT="${AGBD_LOCAL_CKPT:-${AGBD_LOCAL_ROOT}/EcosystemAnalysis/Models/Biomes/weights}"
        AGBD_LOCAL_EMBEDDINGS="${AGBD_LOCAL_EMBEDDINGS:-${AGBD_LOCAL_ROOT}/EcosystemAnalysis/Models/Baseline/cat2vec}"
        AGBD_LOCAL_AEF="${AGBD_LOCAL_AEF:-${AGBD_LOCAL_ROOT}/AEF}"
        AGBD_LOCAL_AEF_H5="${AGBD_LOCAL_AEF_H5:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_AEF_NORM="${AGBD_LOCAL_AEF_NORM:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_TESSERA_H5="${AGBD_LOCAL_TESSERA_H5:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        AGBD_LOCAL_TESSERA_NORM="${AGBD_LOCAL_TESSERA_NORM:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        AGBD_LOCAL_SPLITS="${AGBD_LOCAL_SPLITS:-${AGBD_LOCAL_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/biomes_split}"
        AGBD_LOCAL_PLOTS="${AGBD_LOCAL_PLOTS:-${AGBD_LOCAL_ROOT}/EcosystemAnalysis/Models/Biomes/eval_plots}"
        # Raw per-tile rasters -- only inference_agbd.py (AGBD-features tile inference) reads these.
        # DEM lives alongside ALOS. REGION holds s2_tile_to_region-v3.pkl.
        AGBD_LOCAL_TILES="${AGBD_LOCAL_TILES:-${AGBD_LOCAL_ROOT}/S2_L2A}"
        AGBD_LOCAL_ALOS="${AGBD_LOCAL_ALOS:-${AGBD_LOCAL_ROOT}/ALOS}"
        AGBD_LOCAL_DEM="${AGBD_LOCAL_DEM:-${AGBD_LOCAL_ROOT}/ALOS}"
        AGBD_LOCAL_LC="${AGBD_LOCAL_LC:-${AGBD_LOCAL_ROOT}/LC}"
        AGBD_LOCAL_REGION="${AGBD_LOCAL_REGION:-${AGBD_LOCAL_ROOT}/BiomassDatasetCreation/Data/download_Sentinel}"
        ;;

    euler)
        # AGBD_ENV=cluster: the LOCAL_* paths below are unused (the launchers
        # read from $TMPDIR), but are defined so get_paths() never KeyErrors.
        AGBD_ENV="cluster"
        AGBD_LOCAL_ROOT="${AGBD_LOCAL_ROOT:-${AGBD_REPO}/data}"
        AGBD_LOCAL_H5="${AGBD_LOCAL_H5:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_NORM="${AGBD_LOCAL_NORM:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_MAP="${AGBD_LOCAL_MAP:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_CKPT="${AGBD_LOCAL_CKPT:-${AGBD_LOCAL_ROOT}/weights}"
        AGBD_LOCAL_EMBEDDINGS="${AGBD_LOCAL_EMBEDDINGS:-${AGBD_LOCAL_ROOT}/cat2vec}"
        AGBD_LOCAL_AEF="${AGBD_LOCAL_AEF:-${AGBD_LOCAL_ROOT}/AEF}"
        AGBD_LOCAL_AEF_H5="${AGBD_LOCAL_AEF_H5:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_AEF_NORM="${AGBD_LOCAL_AEF_NORM:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_TESSERA_H5="${AGBD_LOCAL_TESSERA_H5:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        AGBD_LOCAL_TESSERA_NORM="${AGBD_LOCAL_TESSERA_NORM:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        AGBD_LOCAL_SPLITS="${AGBD_LOCAL_SPLITS:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_PLOTS="${AGBD_LOCAL_PLOTS:-${AGBD_LOCAL_ROOT}/eval_plots}"
        ;;

    local|*)
        # Portable, self-contained: everything under ./data in this repo.
        AGBD_ENV="local"
        AGBD_LOCAL_ROOT="${AGBD_LOCAL_ROOT:-${AGBD_REPO}/data}"
        AGBD_LOCAL_H5="${AGBD_LOCAL_H5:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_NORM="${AGBD_LOCAL_NORM:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_MAP="${AGBD_LOCAL_MAP:-${AGBD_LOCAL_ROOT}/patches}"
        # Trained checkpoints: <ckpt>/<arch>/<run_id>_best.ckpt
        AGBD_LOCAL_CKPT="${AGBD_LOCAL_CKPT:-${AGBD_LOCAL_ROOT}/weights}"
        # cat2vec embeddings; "/AGBD" or "/AGBD-Lite" is appended at use time.
        AGBD_LOCAL_EMBEDDINGS="${AGBD_LOCAL_EMBEDDINGS:-${AGBD_LOCAL_ROOT}/cat2vec}"
        AGBD_LOCAL_AEF="${AGBD_LOCAL_AEF:-${AGBD_LOCAL_ROOT}/AEF}"
        AGBD_LOCAL_AEF_H5="${AGBD_LOCAL_AEF_H5:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_AEF_NORM="${AGBD_LOCAL_AEF_NORM:-${AGBD_LOCAL_ROOT}/patches/AEF}"
        AGBD_LOCAL_TESSERA_H5="${AGBD_LOCAL_TESSERA_H5:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        AGBD_LOCAL_TESSERA_NORM="${AGBD_LOCAL_TESSERA_NORM:-${AGBD_LOCAL_ROOT}/patches/TESSERA}"
        # data/patches is THE dataset directory: it mirrors what the cluster launchers
        # stage into $TMPDIR (patches + statistics + biomes_splits_to_name.pkl +
        # tiles_per_region.pkl + AEF_overlaps.pkl). inference_aef.py reads the splits
        # mapping from here too, so SPLITS points at the same place.
        AGBD_LOCAL_SPLITS="${AGBD_LOCAL_SPLITS:-${AGBD_LOCAL_ROOT}/patches}"
        AGBD_LOCAL_PLOTS="${AGBD_LOCAL_PLOTS:-${AGBD_LOCAL_ROOT}/eval_plots}"
        # Raw per-tile rasters -- only inference_agbd.py (AGBD-features tile inference) reads these,
        # and only if you run it; they are optional, so leave them unset if you have no tile data.
        # DEM lives alongside ALOS. REGION holds s2_tile_to_region-v3.pkl.
        AGBD_LOCAL_TILES="${AGBD_LOCAL_TILES:-${AGBD_LOCAL_ROOT}/S2_L2A}"
        AGBD_LOCAL_ALOS="${AGBD_LOCAL_ALOS:-${AGBD_LOCAL_ROOT}/ALOS}"
        AGBD_LOCAL_DEM="${AGBD_LOCAL_DEM:-${AGBD_LOCAL_ROOT}/ALOS}"
        AGBD_LOCAL_LC="${AGBD_LOCAL_LC:-${AGBD_LOCAL_ROOT}/LC}"
        AGBD_LOCAL_REGION="${AGBD_LOCAL_REGION:-${AGBD_LOCAL_ROOT}/download_Sentinel}"
        ;;
esac


# ----- SLURM logs ------------------------------------------------------------
# The #SBATCH --output/--error lines in the launchers are literal (SLURM parses
# those comments before bash runs, so they cannot read a variable). They point
# at ${AGBD_CLUSTER_SCRATCH}/logs on euler; edit them there if you relocate.
AGBD_LOG_DIR="${AGBD_LOG_DIR:-${AGBD_CLUSTER_SCRATCH}/logs}"


# ----- Analysis / comparison -------------------------------------------------
# ESA CCI biomass rasters (CCI_<tile>_19.tif).
AGBD_CCI_DIR="${AGBD_CCI_DIR:-${AGBD_LOCAL_ROOT}/CCI}"

# Per-plot AGB prediction rasters, written by model/inference_aef.py.
AGBD_PREDICTIONS="${AGBD_PREDICTIONS:-${AGBD_LOCAL_ROOT}/predictions}"

# Sentinel-2 tile index shapefile (Name column, EPSG:4326).
AGBD_S2_INDEX="${AGBD_S2_INDEX:-${AGBD_LOCAL_ROOT}/sentinel_2_index_shapefile.shp}"


# ----- Export ----------------------------------------------------------------
export AGBD_PROFILE AGBD_ENV AGBD_REPO
