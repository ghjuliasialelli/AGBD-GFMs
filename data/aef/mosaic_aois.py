"""
Mosaic multi-tile AOI folders into a single GeoTIFF per AOI.

For AOI folders containing more than one .tiff file, merges them into a single
file. Single-tile folders have their file renamed to {aoi_id}.tiff. Handles
south-up (positive y-res) tiles by flipping them to north-up before merging.

env: awsenv
"""

import rasterio as rs
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from os.path import join, basename, exists
from os import listdir, rename
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse


def _open_northup(path):
    """Open a rasterio dataset, flipping to north-up if needed.
    Returns a MemoryFile-backed dataset (caller must close)."""
    src = rs.open(path)
    if src.transform.e > 0:
        # South-up: flip data and transform
        data = src.read()[:, ::-1, :]
        t = src.transform
        new_transform = Affine(t.a, t.b, t.c, t.d, -t.e, t.f + t.e * src.height)
        meta = src.meta.copy()
        meta["transform"] = new_transform
        src.close()
        memfile = MemoryFile()
        dst = memfile.open(**meta)
        dst.write(data)
        return memfile, dst
    return None, src


def process_aoi(aoi_dir):
    """Mosaic or rename .tiff files in an AOI directory."""
    tiffs = sorted(glob(join(aoi_dir, "*.tiff")))
    aoi_id = basename(aoi_dir)
    target = join(aoi_dir, f"{aoi_id}.tiff")

    if exists(target):
        return f"Skipped (already done): {aoi_id}"

    if len(tiffs) == 0:
        return f"Skipped (no tiles): {aoi_id}"

    if len(tiffs) == 1:
        rename(tiffs[0], target)
        return f"Renamed: {aoi_id}"

    try:
        memfiles = []
        datasets = []
        for f in tiffs:
            mf, ds = _open_northup(f)
            if mf is not None:
                memfiles.append(mf)
            datasets.append(ds)

        mosaic, out_transform = merge(datasets, nodata=-128)
        meta = datasets[0].meta.copy()
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()

        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "deflate",
        })

        with rs.open(target, "w", **meta) as dst:
            dst.write(mosaic)

        return f"Mosaicked ({len(tiffs)} tiles): {aoi_id}"

    except Exception as e:
        return f"Error {aoi_id}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aef_dir", type=str, default="/scratch3/gsialelli/AEF")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    aoi_dirs = sorted(
        join(args.aef_dir, d)
        for d in listdir(args.aef_dir)
    )

    print(f"Processing {len(aoi_dirs)} AOIs.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_aoi, aoi_dirs):
            print(f"  {result}")

    print("Done.")
