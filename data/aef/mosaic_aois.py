"""
Mosaic multi-tile AOI folders into a single GeoTIFF per AOI.

For AOI folders containing more than one .tiff file, merges them into a single
file. Single-tile folders have their file renamed to {aoi_id}.tiff. Handles
south-up (positive y-res) tiles by flipping them to north-up before merging,
and AOIs whose tiles straddle a UTM zone boundary by reprojecting them to a
common CRS first.

env: awsenv
"""

import rasterio as rs
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.crs import CRS
from os.path import join, basename, exists
from os import listdir, rename, remove
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse

# An AGBRef cell is 0.1 deg plus a 1 km buffer, i.e. at most ~1.4 km / 10 m ~ 1400 px per side.
# Anything dramatically larger means the tiles were placed in the wrong coordinate space and the
# merge produced a mostly-empty bounding box spanning the gap. See _target_crs below.
MAX_EXPECTED_PX = 4000
AEF_NODATA = -128


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
        with memfile.open(**meta) as dst:
            dst.write(data)
        # Reopen read-only: WarpedVRT rejects write-mode sources (deprecated in rasterio, and slated
        # to become an error), and the cross-zone merge path wraps this dataset in one.
        return memfile, memfile.open()
    return None, src


def _target_crs(datasets):
    """Pick the CRS to mosaic an AOI in: the UTM zone containing the centre of the tiles' union.

    rasterio's merge() does NOT reproject. It adopts the first dataset's CRS and unions the input
    bounds numerically, so tiles from different UTM zones get placed as though their eastings were
    all in the same zone -- ~550 km apart for adjacent zones. The merged raster then spans the gap:
    an AGBRef cell that should be ~1150 px wide came out 55948 px, 96% nodata. Cells straddling a
    zone boundary (108 deg W for 12/13, 138 deg E for 53/54, ...) are legitimate, so the fix is to
    warp everything into one zone rather than to drop them.

    Either adjacent zone is defensible for an ~11 km cell; the union centre is used so the choice
    is deterministic and does not depend on the order glob happens to return the crops in.
    """
    lefts, bottoms, rights, tops = [], [], [], []
    for ds in datasets:
        l, b, r, t = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        lefts.append(l); bottoms.append(b); rights.append(r); tops.append(t)
    lon = (min(lefts) + max(rights)) / 2
    lat = (min(bottoms) + max(tops)) / 2
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def process_aoi(aoi_dir, force=False):
    """Mosaic or rename .tiff files in an AOI directory."""
    aoi_id = basename(aoi_dir)
    target = join(aoi_dir, f"{aoi_id}.tiff")

    # Never feed the previous mosaic back into itself: exclude <aoi>.tiff from the inputs.
    # (Matters when force=True, i.e. re-mosaicking after a re-download of the per-tile crops.)
    tiffs = sorted(t for t in glob(join(aoi_dir, "*.tiff")) if basename(t) != f"{aoi_id}.tiff")

    if exists(target) and not force:
        return f"Skipped (already done): {aoi_id}"

    if len(tiffs) == 0:
        return f"Skipped (no tiles): {aoi_id}"

    if len(tiffs) == 1:
        # A single tile still has to be normalised to north-up. The AEF files are south-up on S3
        # (y_res = +10), while the training patches are cut with gdal.Warp (data/aef/create_patches.py),
        # which always emits north-up -- and the models are trained with augment=False, so they never
        # saw a flipped sample. Renaming a south-up file straight to the target (the old behaviour)
        # therefore fed the model vertically mirrored embeddings. Only rename when already north-up.
        with rs.open(tiffs[0]) as src:
            if src.transform.e < 0:
                rename(tiffs[0], target)
                return f"Renamed (already north-up): {aoi_id}"

        memfile, ds = _open_northup(tiffs[0])
        data, meta = ds.read(), ds.meta.copy()
        ds.close()
        if memfile is not None:
            memfile.close()
        with rs.open(target, "w", **meta) as dst:
            dst.write(data)
        remove(tiffs[0])
        return f"Flipped to north-up: {aoi_id}"

    try:
        memfiles = []
        datasets = []
        for f in tiffs:
            mf, ds = _open_northup(f)
            if mf is not None:
                memfiles.append(mf)
            datasets.append(ds)

        # Tiles straddling a UTM zone boundary have to be warped into one CRS before merging;
        # merge() itself would silently misplace them by a whole zone width. Nearest resampling
        # keeps the int8 embeddings categorical-safe (no interpolation between band values).
        dst_crs = _target_crs(datasets)
        vrts = [WarpedVRT(ds, crs=dst_crs, resampling=Resampling.nearest,
                          src_nodata=AEF_NODATA, nodata=AEF_NODATA)
                for ds in datasets] if len({ds.crs for ds in datasets}) > 1 else []

        mosaic, out_transform = merge(vrts or datasets, nodata=AEF_NODATA)
        meta = datasets[0].meta.copy()
        for v in vrts:
            v.close()
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()

        meta.update({
            "crs": dst_crs,
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "deflate",
        })

        # Cheap total invariant, deliberately not a spot check: a merge that misplaces tiles blows
        # the raster up by ~50x, which is obvious in the metadata alone. Failing loudly here beats
        # writing a 96%-nodata mosaic that only surfaces later as an OOM -- or worse, as a
        # plausible-looking prediction if the box happens to have enough RAM to finish.
        if mosaic.shape[1] > MAX_EXPECTED_PX or mosaic.shape[2] > MAX_EXPECTED_PX:
            return (f"Error {aoi_id}: implausible mosaic {mosaic.shape[1]}x{mosaic.shape[2]} px "
                    f"(> {MAX_EXPECTED_PX}); tiles likely misplaced across CRSs.")

        with rs.open(target, "w", **meta) as dst:
            dst.write(mosaic)

        return f"Mosaicked ({len(tiffs)} tiles, {dst_crs}): {aoi_id}"

    except Exception as e:
        return f"Error {aoi_id}: {e}"


if __name__ == "__main__":
    from functools import partial
    from os.path import isdir
    parser = argparse.ArgumentParser()
    parser.add_argument("--aef_dir", type=str, default="/scratch3/gsialelli/AEF")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Re-mosaic even if <aoi>.tiff exists (e.g. after re-downloading the crops).")
    args = parser.parse_args()

    aoi_dirs = sorted(
        join(args.aef_dir, d)
        for d in listdir(args.aef_dir)
        if isdir(join(args.aef_dir, d))
    )

    print(f"Processing {len(aoi_dirs)} AOIs (force={args.force}).")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(partial(process_aoi, force=args.force), aoi_dirs):
            print(f"  {result}")

    print("Done.")
