"""
Audit every AEF AOI mosaic against invariants that must hold for all of them.

Written after a cross-UTM-zone merge bug silently corrupted 35/780 mosaics (rasterio's merge()
does not reproject, so tiles in a neighbouring zone were placed ~550 km away and the raster grew
to span the gap -- 55948 px wide, 96% nodata). It went unnoticed because the mosaics had only been
spot-checked visually, on a sample that happened to contain no cross-zone AOI. The bug was plainly
visible in the metadata the whole time.

Hence this: metadata-only, so it runs over all 780 in seconds, and total rather than sampled. Cheap
enough that there is no excuse for eyeballing a handful instead.

Usage:
    python audit_mosaics.py [--aef_dir /scratch3/gsialelli/AEF]

Exit code is 0 if every AOI passes, 1 otherwise, so it can gate a pipeline.
"""

import argparse
import rasterio as rs
from os import listdir
from os.path import join, isdir, basename, exists

# An AGBRef cell is 0.1 deg + 1 km buffer -> at most ~1.4 km per side at 10 m resolution.
MAX_EXPECTED_PX = 4000
MIN_EXPECTED_PX = 100
AEF_NODATA = -128
AEF_BANDS = 64
# Above this, the mosaic is essentially empty: a real cell is near-fully covered. The cross-zone
# bug produced 96-100% nodata, whereas healthy mosaics sit near 0.
MAX_NODATA_FRAC = 0.60


def audit_aoi(aoi_dir):
    """Check one AOI mosaic. Returns a list of failure strings (empty == healthy)."""
    aoi_id = basename(aoi_dir)
    target = join(aoi_dir, f"{aoi_id}.tiff")
    if not exists(target):
        return [f"{aoi_id}: no mosaic"]

    fails = []
    try:
        with rs.open(target) as s:
            if s.width > MAX_EXPECTED_PX or s.height > MAX_EXPECTED_PX:
                fails.append(f"{aoi_id}: implausible size {s.height}x{s.width} px "
                             f"(> {MAX_EXPECTED_PX}) -- tiles misplaced across CRSs?")
            if s.width < MIN_EXPECTED_PX or s.height < MIN_EXPECTED_PX:
                fails.append(f"{aoi_id}: suspiciously small {s.height}x{s.width} px")
            if s.count != AEF_BANDS:
                fails.append(f"{aoi_id}: {s.count} bands, expected {AEF_BANDS}")
            # The models are trained on gdal.Warp'd (north-up) patches with augment=False, so a
            # south-up mosaic feeds them vertically mirrored input.
            if s.transform.e > 0:
                fails.append(f"{aoi_id}: south-up (e={s.transform.e})")
            if s.crs is None:
                fails.append(f"{aoi_id}: no CRS")
            # Only read a band if the geometry is sane -- reading a 55948 px mosaic costs GBs.
            if not fails:
                frac = (s.read(1) == AEF_NODATA).mean()
                if frac > MAX_NODATA_FRAC:
                    fails.append(f"{aoi_id}: {frac:.1%} nodata (> {MAX_NODATA_FRAC:.0%})")
    except Exception as e:
        fails.append(f"{aoi_id}: unreadable ({e})")

    return fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aef_dir", type=str, default="/scratch3/gsialelli/AEF")
    args = parser.parse_args()

    aoi_dirs = sorted(join(args.aef_dir, d) for d in listdir(args.aef_dir)
                      if isdir(join(args.aef_dir, d)))

    all_fails = []
    for d in aoi_dirs:
        all_fails.extend(audit_aoi(d))

    print(f"Audited {len(aoi_dirs)} AOI mosaics.")
    if all_fails:
        print(f"FAILED: {len(all_fails)} problem(s)\n")
        for f in all_fails: print(f"  {f}")
        raise SystemExit(1)
    print("All healthy: size plausible, 64 bands, north-up, CRS set, mostly-valid pixels.")
