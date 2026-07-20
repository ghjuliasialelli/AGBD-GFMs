#!/bin/bash
#
# OPTIONAL: fetch the exact GFM encoder weight files our benchmark runs used.
#
# YOU PROBABLY DO NOT NEED THIS.
#   The normal PANGAEA pipeline downloads most of these for you on first use, straight from
#   the original authors' hosts (HuggingFace, Zenodo, GitHub releases, Google Drive) -- just
#   run the launchers in this directory and let it fetch. Use this script only if you want
#   bit-identical inputs to ours, or if an upstream host has moved/disappeared.
#
#   The one encoder the normal pipeline cannot fetch is gfmswin: its `download_url` is
#   `null` (on upstream PANGAEA too), so gfm.pth must otherwise be downloaded by hand from
#   the OneDrive link in github.com/mmendiet/GFM. satlasnet_si needs no file at all -- it
#   pulls its own weights via the `satlaspretrain_models` package.
#
# WHAT THIS IS
#   A Zenodo archive of the ORIGINAL authors' weights, unmodified, recording exactly which
#   bytes produced the numbers in our paper. It is a provenance record, not a distribution
#   channel: the authoritative source for each model is its own project. Every model's
#   licence and citation are listed in the Zenodo record and in ATTRIBUTION.md next to this
#   script -- please cite the original publications for any encoder you use.
#
# WHERE THESE GO
#   PANGAEA resolves `encoder_weights: ./pretrained_models/<file>` RELATIVE TO ITS OWN
#   WORKING DIRECTORY, and the launchers here are run from inside the pangaea-bench fork.
#   So the weights belong in <fork checkout>/pretrained_models/, NOT in the AGBD-GFMs repo.
#   Point this script at the fork with --dest, or run it from there.
#
# Run:
#   bash fetch_encoder_weights.sh                       # into ./pretrained_models
#   bash fetch_encoder_weights.sh --dest /path/to/pangaea-bench/pretrained_models
#   AGBD_ZENODO_RECORD=1234567 bash fetch_encoder_weights.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------------------
# The Zenodo record holding the weights. Override with $AGBD_ZENODO_RECORD.
# TODO(release): replace with the real record id once the deposit is published.
ZENODO_RECORD="${AGBD_ZENODO_RECORD:-REPLACE_WITH_RECORD_ID}"

DEST="./pretrained_models"
while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "$ZENODO_RECORD" = "REPLACE_WITH_RECORD_ID" ]; then
    echo "ERROR: no Zenodo record id set." >&2
    echo "Edit ZENODO_RECORD at the top of this script, or run with AGBD_ZENODO_RECORD=<id>." >&2
    exit 1
fi

# One entry per encoder actually used by the paper: "<file>|<md5>|<encoder>".
# The md5s are of the exact files our runs loaded (computed 2026-07-17), so a truncated or
# corrupted download is caught here rather than surfacing later as an opaque shape or
# state_dict error. If you re-upload a file to Zenodo, update its md5 here.
FILES=(
    "CROMA_large.pt|6375e92f3c5d715bb4c4fc7087353f56|croma_optical"
    "DOFA_ViT_base_e100.pth|4c76995c8fbd95a456581dbe9b0e2397|dofa"
    "Prithvi_100M.pt|882987da172c18e11fe95828aa0090ca|prithvi"
    "Prithvi_EO_V2_100M_TL.pt|da41788f5e73d3374762c48acaf8b1cb|prithvi2_100m"
    "RemoteCLIP-ViT-B-32.pt|f99b4a164b93d5c1e9459960af5a9a11|remoteclip"
    "scalemae-vitlarge-800.pth|67d6803a25bfce51af9b5d7170e2d330|scalemae"
    "SpectralGPT+.pth|aeaa08388f0937e3f07ecbd7757f20dc|spectralgpt"
    "B13_vits16_moco_0099.pth|b8d8c32f5142229dc6f89c46c1fd53ac|ssl4eo_moco"
    "TerraMind_v1_tiny.pt|d345af451025c537689a2b8169731cae|terramind_optical_tiny"
    "gfm.pth|095e4840e320216b41d7d9549ddc397d|gfmswin"
)

mkdir -p "$DEST"
echo "Fetching ${#FILES[@]} encoder weights from Zenodo record ${ZENODO_RECORD}"
echo "  destination: $(cd "$DEST" && pwd)"
echo "  total download: ~8.9 GB (skipped if already present and intact)"
echo

for entry in "${FILES[@]}"; do
    IFS='|' read -r fname md5 encoder <<< "$entry"
    target="${DEST}/${fname}"

    # Already there and intact? Leave it alone -- these are large files.
    if [ -f "$target" ]; then
        if echo "${md5}  ${target}" | md5sum --check --status 2>/dev/null; then
            echo "  [have]  ${fname}  (${encoder})"
            continue
        fi
        echo "  [bad ]  ${fname} failed its checksum -- refetching"
        rm -f "$target"
    fi

    url="https://zenodo.org/records/${ZENODO_RECORD}/files/${fname}?download=1"
    echo "  [get ]  ${fname}  (${encoder})"
    # --location follows Zenodo's redirect to its storage backend; --fail turns an HTTP
    # error into a non-zero exit instead of writing an HTML error page to the .pt file.
    curl --fail --location --progress-bar --output "${target}.part" "$url"

    # Verify BEFORE moving into place, so a bad download can never masquerade as a good
    # file on a rerun (the .part is left behind for inspection).
    echo "${md5}  ${target}.part" | md5sum --check --status \
        || { echo "ERROR: ${fname} downloaded but its checksum is wrong (left at ${target}.part)." >&2; exit 1; }
    mv "${target}.part" "$target"
done

echo
echo "Done. ${#FILES[@]} files in $(cd "$DEST" && pwd)."
echo "satlasnet_si needs no file here (it downloads via the satlaspretrain_models package)."
echo "Please cite the original papers for any encoder you use -- see ATTRIBUTION.md."
