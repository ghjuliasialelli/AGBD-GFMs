"""
Build the AGBRef plot polygons from the source .Rdata grid.

Each AGBRef record in ``AGBref_fin_10km_*.Rdata`` is the aggregate of the field plots
falling inside a **0.1 degree grid cell** (POINT_X / POINT_Y are the cell centres). The
plot footprint is therefore that 0.1 deg cell. We pad it by a fixed 1 km margin so the
downloaded AEF embeddings and ESA-CCI crops always fully contain the cell; the statistics
in ``comparison.py`` clip back to the exact 0.1 deg cell, so the margin only guarantees
coverage and never enters the numbers.

The margin is applied in **true metres per axis** (dlat, dlon computed separately), so it
stays a constant 1 km at every latitude.

    IMPORTANT -- why we do NOT use ``to_crs(3857).buffer(...)``:
    EPSG:3857 (Web Mercator) is conformal, not equidistant. Its "metres" are true only at
    the equator; at latitude phi one projected unit is cos(phi) ground metres. Buffering a
    fixed distance there produced squares of side 10.5*cos(lat) km -- ~10.5 km at the
    equator but only ~3 km at 73N -- while the ground truth still covers the whole 0.1 deg
    cell. That was the original bug (see comparison/agbref -- exploration.ipynb, cell 4).

Outputs (same geometry, two formats, both consumed downstream):
    <out_dir>/AGBRef.geojson  -> comparison.py + BiomassDatasetCreation/CCI/process_tiles.py
    <out_dir>/AGBref.gpkg     -> data/aef/download_aoi.py (AEF downloads)

Usage:
    python make_agbref_polygons.py                       # writes into ./data
    python make_agbref_polygons.py --out_dir /some/dir   # writes elsewhere too
"""

import argparse
from pathlib import Path

import numpy as np
import pyreadr
import geopandas as gpd
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
DEFAULT_RDATA = HERE / "data" / "AGBref_fin_10km_2025-04-01.Rdata"
DEFAULT_OUT_DIR = HERE / "data"

# Grid cell + download margin
HALF_DEG = 0.05          # 0.1 deg cell -> 0.05 deg half-width
BUFFER_M = 1000.0        # download margin, true metres (clipped away for statistics)
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0

# Match the notebook's record selection: recent enough, and >= 5 field plots per cell.
MIN_YEAR = 2017
MIN_N = 5
KEEP_COLS = ["POINT_X", "POINT_Y", "n", "AGB_T_HA", "varTot", "AVG_YEAR"]


def load_records(rdata_path):
    """Read the source grid and apply the AGBRef record selection."""
    results = pyreadr.read_r(str(rdata_path))["mpAGB"]
    data = results[(results["AVG_YEAR"] >= MIN_YEAR) & (results["n"] >= MIN_N)][KEEP_COLS]
    return data.reset_index(drop=True)


def cell_with_buffer(x, y):
    """0.1 deg cell centred on (x, y), padded by BUFFER_M true metres on every side."""
    dlat = HALF_DEG + BUFFER_M / M_PER_DEG_LAT
    dlon = HALF_DEG + BUFFER_M / (M_PER_DEG_LON * np.cos(np.radians(y)))
    return box(x - dlon, y - dlat, x + dlon, y + dlat)


def build_polygons(data):
    gdf = gpd.GeoDataFrame(
        data,
        geometry=[cell_with_buffer(x, y) for x, y in zip(data["POINT_X"], data["POINT_Y"])],
        crs="EPSG:4326",
    )
    gdf["AVG_YEAR"] = gdf["AVG_YEAR"].astype(int)
    return gdf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdata", type=Path, default=DEFAULT_RDATA, help="source .Rdata grid")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="where to write the two files")
    args = ap.parse_args()

    data = load_records(args.rdata)
    gdf = build_polygons(data)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = args.out_dir / "AGBRef.geojson"
    gpkg_path = args.out_dir / "AGBref.gpkg"
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"wrote {geojson_path}")
    print(f"wrote {gpkg_path}")
    print(f"{len(gdf)} plots | lat range {gdf['POINT_Y'].min():.1f}..{gdf['POINT_Y'].max():.1f}")


if __name__ == "__main__":
    main()
