"""

Reproject ESA CCI BIOMASS v6.0 (2020) onto the AEF prediction grid of each map-figure tile, so that
a column of the figure compares three sources over pixel-identical ground.

CCI ships as 10 deg x 10 deg blocks at 100 m in EPSG:4326; the AEF predictions are ~10 m in the
tile's UTM zone. The AEF prediction defines the target grid exactly (CRS, transform, width, height)
-- it is the smallest of the three sources, and matching it is what makes the comparison
like-for-like rather than three different areas stacked.

Resampling is NEAREST, deliberately, and ALL tiles are built this way.

Nearest does not interpolate: every output pixel copies an original 100 m CCI cell, so the crop
contains CCI's real values and nothing invented. CCI is being put on a 10 m grid here, and an
interpolating method (bilinear/cubic/average) would smooth it into looking like it resolves detail
the product does not have. Nearest renders its true resolution honestly, as 10x10 blocks.

Reprojecting onto the AEF grid at all is still worth it: it keeps the three rows of a figure column
pixel-co-registered over identical ground. The only cost is a sub-cell positional snap of at most
half a CCI cell (~50 m), which is negligible for a 100 m product and changes no value.

(The crops that originally existed for 32TNS/45RXL/59GPM were made with BILINEAR -- established by
regenerating them and diffing: bilinear reproduces them to ~0.001 t/ha mean abs difference, nearest
differs on 21-85% of pixels. They have since been regenerated with nearest so every column is built
identically. Never mix methods across columns.)

NOTE: these crops are for DISPLAY only. For metrics, sample the SOURCE block instead -- a resampled
crop and the source disagree by up to ~77 t/ha at edges.

Zeros are kept as 0 Mg/ha. In ESA CCI a 0 is a real measurement of zero biomass -- it is NOT a
nodata sentinel and NOT a forest mask, so it must not be masked away. Only pixels falling outside
the source block become nodata (-9999), matching the AEF/AGBD prediction convention.

Usage:
    python crop_cci_maps.py --tiles 49SBT
    python crop_cci_maps.py                # all tiles listed in CCI_BLOCKS

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import rasterio as rs
from rasterio.warp import reproject, Resampling
from os.path import join, exists
from os import makedirs

###################################################################################################
# Configuration

PRED_AEF = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film/59620113-1_59620113-1_59620113-1"
CCI_DIR = "/scratch3/gsialelli/CCI"
OUT_DIR = "/scratch3/gsialelli/CCI/maps"

CCI_TEMPLATE = "{block}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv6.0.tif"

# The 10 deg CCI block each tile falls in. Checked against the AEF window bounds at run time rather
# than trusted, since a tile near a block edge would otherwise be silently half-filled.
CCI_BLOCKS = {
    "59GPM" : "S40E170",
    "32TNS" : "N50E000",
    "45RXL" : "N30E080",
    "49SBT" : "N40E100",
    # 32TPT replaces 32TNS for the Europe column: 32TNS's 40 km AEF window contains ZERO GEDI
    # footprints in any year (the window is Graubuenden/Valtellina, not Austria), so it can never
    # carry an honest per-tile metric. 32TPT has 165,788 footprints in the same-sized window
    # (measured, not assumed) and its AEF tiles exist in tile_to_aefiles.pkl.
    "32TPT" : "N50E010",
}

NODATA = -9999.0

###################################################################################################
# Helpers

def block_bounds(block) :
    """
    Parse a CCI block name into its lon/lat bounds. The name gives the block's NORTH-WEST corner,
    and each block spans 10 deg.

    Args:
    - block (str): e.g. 'N40E100' or 'S40E170'.

    Returns:
    - tuple: (left, bottom, right, top) in EPSG:4326.
    """
    lat = int(block[1:3]) * (1 if block[0] == "N" else -1)
    lon = int(block[4:7]) * (1 if block[3] == "E" else -1)
    return (lon, lat - 10, lon + 10, lat)


def crop_tile(tile, block) :
    """
    Reproject the CCI block onto the tile's AEF prediction grid and write it out.

    Args:
    - tile (str): MGRS tile name, e.g. '49SBT'.
    - block (str): CCI block name, e.g. 'N40E100'.

    Returns:
    - str: a status message.
    """
    aef_path = join(PRED_AEF, f"{tile}.tif")
    cci_path = join(CCI_DIR, CCI_TEMPLATE.format(block = block))
    out_path = join(OUT_DIR, f"{tile}_CCI.tif")

    if not exists(aef_path) : return f"Skipped {tile}: no AEF prediction at {aef_path}"
    if not exists(cci_path) : return f"Skipped {tile}: no CCI block at {cci_path}"

    with rs.open(aef_path) as ref :
        dst_crs, dst_transform = ref.crs, ref.transform
        dst_h, dst_w = ref.height, ref.width
        # The AEF window in lon/lat, to confirm the chosen block actually contains it.
        l, b, r, t = rs.warp.transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)

    bl, bb, br, bt = block_bounds(block)
    if not (bl <= l and r <= br and bb <= b and t <= bt) :
        return (f"Error {tile}: window ({l:.3f},{b:.3f},{r:.3f},{t:.3f}) is not contained in block "
                f"{block} ({bl},{bb},{br},{bt}); it would be silently half nodata. Pick the right "
                f"block, or mosaic two blocks first.")

    with rs.open(cci_path) as src :
        dst = np.full((dst_h, dst_w), NODATA, dtype = np.float32)
        reproject(
            source = rs.band(src, 1),
            destination = dst,
            src_transform = src.transform, src_crs = src.crs,
            dst_transform = dst_transform, dst_crs = dst_crs,
            src_nodata = None, dst_nodata = NODATA,
            resampling = Resampling.nearest,
        )

    meta = {
        "driver" : "GTiff", "height" : dst_h, "width" : dst_w, "count" : 1,
        "dtype" : "float32", "crs" : dst_crs, "transform" : dst_transform,
        "nodata" : NODATA, "compress" : "deflate",
    }
    makedirs(OUT_DIR, exist_ok = True)
    with rs.open(out_path, "w", **meta) as out :
        out.write(dst, 1)

    valid = dst[dst != NODATA]
    if valid.size == 0 : return f"Error {tile}: wrote an entirely-nodata crop."
    return (f"Wrote {out_path}  ({dst_h}x{dst_w}, valid {valid.size / dst.size:.3f}, "
            f"median {np.median(valid):.1f} t/ha, max {valid.max():.1f})")


###################################################################################################
# Main

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type = str, nargs = "*", default = sorted(CCI_BLOCKS),
                        help = "Tiles to crop; default all in CCI_BLOCKS.")
    args = parser.parse_args()

    for tile in args.tiles :
        if tile not in CCI_BLOCKS :
            print(f"  Skipped {tile}: no CCI block configured.")
            continue
        print(f"  {crop_tile(tile, CCI_BLOCKS[tile])}")
