"""

Build TESSERA-{split}-{year}.h5 for the full AGBD (one file per split and year), to evaluate Lite-trained
TESSERA models on the full test set. Same patches as TESSERA-Lite (download.py's extract_patch), tiles from
Source Cooperative's npy/v1 (the release dl2.geotessera.org served; it matched TESSERA-Lite-test.h5 exactly):

    /embeddings   int8    (N, 25, 25, 128)   quantised
    /scales       float32 (N, 25, 25)        dequant factors, NaN where TESSERA has no pixel
    /done         bool    (N,)               rows whose data has reached the disk
    /GEDI/{lat,lon}_{decimal,offset}
    /tiles/{name,file,start,count}           row of (tile, i) = tiles/start[tile] + i

Rows follow olmoearth/audit.py's full_layout() (files 0..19, tiles sorted, rows in order), the same as the
OLMO-{split}-{year}.h5 files. A tile that fails to download aborts the run (--continue resumes it); only a
tile absent from the manifest leaves NaN pixels.

Run, with the tessera conda environment:
    python download_full.py --input <dir with data_subset-{year}-v4_{i}-20.h5> --split test --year 2019
                            --output <dir>/TESSERA-test-2019.h5 --cache-dir <dir>
Then gate on: python audit_full.py --tessera <dir>/TESSERA-test-2019.h5 --agbd <input> --year 2019

"""

###################################################################################################
# Imports
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "olmoearth"))  # append: "download" must stay this directory's
from audit import GEDI_KEYS, full_gedi, full_layout  # noqa: E402
import download as D  # noqa: E402  (the TESSERA-Lite builder, in this directory)


###################################################################################################
# Helpers

def parse_arguments():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required = True, help = "directory with the data_subset-{year}-v4_{i}-20.h5 files")
    ap.add_argument("--output", required = True, help = "TESSERA-{split}-{year}.h5")
    ap.add_argument("--cache-dir", required = True, help = "manifest and (temporary) tiles; tiles are deleted once used")
    ap.add_argument("--year", type = int, required = True, choices = (2019, 2020))
    ap.add_argument("--split", default = "test", choices = ("train", "val", "test"))
    ap.add_argument("--mapping", default = None, help = "tile -> split pickle (default <input>/biomes_splits_to_name.pkl)")
    ap.add_argument("--workers", type = int, default = 8, help = "parallel tile downloads")
    ap.add_argument("--lookahead", type = int, default = None, help = "upcoming tiles to prefetch (default 2*workers)")
    ap.add_argument("--retries", type = int, default = 6, help = "attempts per file before aborting")
    ap.add_argument("--min-coverage", type = float, default = 1.0, help = "minimum fraction of samples whose tile is in the manifest")
    ap.add_argument("--min-free-gb", type = float, default = 50, help = "abort when the output disk has less than this free")
    ap.add_argument("--commit-every", type = int, default = 25, help = "tiles between flushes to disk")
    ap.add_argument("--max-tiles", type = int, default = None, help = "debug: process only this many tiles")
    ap.add_argument("--continue", dest = "continue_run", action = "store_true", help = "resume: only rows not marked done")
    ap.add_argument("--accepted-checksums", default = str(Path(__file__).resolve().parent / "accepted_checksums.txt"),
                    help = "'<sha256>  <year/grid_X_Y/file.npy>' lines: exact files accepted although they differ from the published SHA256")
    return ap.parse_args()


###################################################################################################
# Code execution

def main() -> None:
    args = parse_arguments()
    input_dir, output = Path(args.input), Path(args.output)
    if output.name != f"TESSERA-{args.split}-{args.year}.h5": raise SystemExit(f"output must be named TESSERA-{args.split}-{args.year}.h5, got {output.name}")
    if output.exists() != args.continue_run:
        raise SystemExit(f"{output} {'exists: use --continue, or delete it to start fresh' if output.exists() else 'does not exist: nothing to --continue'}")
    cache_dir = Path(args.cache_dir); tiles_dir = cache_dir / f"tiles_{args.split}_{args.year}"
    tiles_dir.mkdir(parents = True, exist_ok = True)
    mapping = Path(args.mapping) if args.mapping else input_dir / "biomes_splits_to_name.pkl"

    # The samples, in the flat order of the OLMO full-test files
    layout = full_layout(input_dir, args.year, args.split, mapping)
    gedi = full_gedi(input_dir, layout)
    lons, lats = D._lonlat_from_gedi(gedi["lat_decimal"], gedi["lat_offset"], gedi["lon_decimal"], gedi["lon_offset"])
    n = len(lats); assert n > 0
    manifest, manifest_sha = D.load_manifest(cache_dir, args.year)
    print(f"full AGBD {args.split} {args.year}: {len(layout)} S2 tiles, {n:,} samples; {len(manifest):,} TESSERA tiles in the manifest", flush = True)

    primary = [D.tile_from_world(float(a), float(b)) for a, b in zip(lons, lats)]
    covered = np.fromiter((D.tile_key(t) in manifest for t in primary), bool, n)
    print(f"samples whose tile is in the manifest: {covered.mean():.4%}", flush = True)
    if covered.mean() < args.min_coverage: raise SystemExit(f"coverage below --min-coverage {args.min_coverage}: not building")

    mode = "r+" if args.continue_run else "w"
    with h5py.File(output, mode, libver = "latest") as dst:

        if args.continue_run:
            if int(dst.attrs["year"]) != args.year or dst.attrs["split"] != args.split: raise SystemExit("output is for another year or split")
            if dst.attrs["manifest_sha256"] != manifest_sha: raise SystemExit("the manifest changed since this file was started")
            if dst["done"].shape[0] != n or any(not np.array_equal(dst[f"GEDI/{k}"][:], gedi[k]) for k in GEDI_KEYS): raise SystemExit("output does not match these samples")
        else:
            dst.attrs.update({"year": args.year, "split": args.split, "source": f"{D.BASE_URL} (year/grid_X_Y)", "tessera_version": "v1", "manifest_sha256": manifest_sha,
                              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "input": str(input_dir), "mapping": str(mapping),
                              "layout": "N,H,W,C (C = 128 TESSERA dims); row of (tile, i) = tiles/start[tile] + i", "dequantize": "embeddings * scales[..., None]; NaN scale = no pixel"})
            dst.create_dataset("embeddings", shape = (n, D.PATCH, D.PATCH, D.BANDS), dtype = "i1", chunks = (1, D.PATCH, D.PATCH, D.BANDS), compression = "lzf")
            dst.create_dataset("scales", shape = (n, D.PATCH, D.PATCH), dtype = "f4", chunks = (1, D.PATCH, D.PATCH), compression = "lzf", fillvalue = np.nan)
            dst.create_dataset("done", shape = (n,), dtype = bool, fillvalue = False)
            g = dst.create_group("GEDI")
            for k, v in gedi.items(): g.create_dataset(k, data = v)
            t = dst.create_group("tiles")
            t.create_dataset("name", data = np.array([x["tile"] for x in layout], dtype = "S"))
            t.create_dataset("file", data = np.array([x["file"] for x in layout], dtype = "S"))
            t.create_dataset("start", data = np.array([x["start"] for x in layout], dtype = np.int64))
            t.create_dataset("count", data = np.array([x["count"] for x in layout], dtype = np.int64))
            D._flush_and_fsync(dst)

        done = dst["done"][:]
        todo = np.flatnonzero(~done)
        print(f"already done {done.sum():,}, remaining {len(todo):,}", flush = True)
        groups, tile_refs = {}, {}
        for k in todo:
            k = int(k); groups.setdefault(primary[k], []).append(k)
            for t in D.required_tiles_for_sample(float(lons[k]), float(lats[k]), primary[k]): tile_refs[t] = tile_refs.get(t, 0) + 1
        order = sorted(groups) # neighbouring tiles end up close in the order, so they leave the cache sooner
        if args.max_tiles: order = order[: args.max_tiles]
        print(f"{len(order):,} primary tiles to process, {len(tile_refs):,} tiles including neighbours", flush = True)

        # which files were let through is part of the provenance of this output (before SWMR: it forbids new attributes)
        lines = [l.split() for l in Path(args.accepted_checksums).read_text().splitlines() if l.strip() and not l.startswith("#")]
        accepted = {path: sha for sha, path in lines if path.startswith(f"{args.year}/")}
        dst.attrs["checksum_mismatch_accepted"] = "; ".join(f"{path} {sha}" for path, sha in sorted(accepted.items()))

        D._enable_swmr(dst)
        emb_ds, scl_ds, done_ds = dst["embeddings"], dst["scales"], dst["done"]
        fetcher = D.TileFetcher(manifest, tiles_dir, tile_refs, args.workers, verbose = False, retries = args.retries, accepted = accepted)
        lookahead = args.lookahead if args.lookahead is not None else 2 * args.workers
        for t in order[: lookahead]: fetcher.submit(t)

        def commit():
            # data first, flag after: a crash in between redoes rows, it never marks unwritten rows done
            D._flush_and_fsync(dst)
            done_ds[:] = done
            D._flush_and_fsync(dst)

        t0, n_written, pending = time.time(), 0, []
        try:
            for ti, tile in enumerate(order):
                if ti + lookahead < len(order): fetcher.submit(order[ti + lookahead])
                for k in groups[tile]:
                    result = D.extract_patch(float(lons[k]), float(lats[k]), tile, fetcher)
                    if result is None: raise RuntimeError(f"sample {k}: primary tile {tile} is not available")
                    emb_ds[k], scl_ds[k] = result
                    for t in D.required_tiles_for_sample(float(lons[k]), float(lats[k]), tile): fetcher.release(t)
                    pending.append(k); n_written += 1
                if (ti + 1) % args.commit_every == 0 or ti + 1 == len(order):
                    done[pending] = True; pending = []
                    commit()
                    elapsed = time.time() - t0
                    free = shutil.disk_usage(output.parent).free / 1e9
                    cache = sum(p.stat().st_size for p in tiles_dir.glob("*.npy")) / 1e9
                    print(f"  tiles {ti + 1}/{len(order)} rows {done.sum():,}/{n:,} ({n_written / elapsed:.0f} rows/s, {fetcher.n_bytes / 1e6 / elapsed:.0f} MB/s download, "
                          f"{elapsed / 3600:.2f} h) cache {cache:.1f} GB free {free:.0f} GB", flush = True)
                    if free < args.min_free_gb: raise RuntimeError(f"only {free:.0f} GB free on the output disk (< --min-free-gb {args.min_free_gb})")
        finally:
            fetcher.close()

    if done.all():
        shutil.rmtree(tiles_dir, ignore_errors = True)
        print(f"ALL DONE: {n:,} rows in {output}", flush = True)
    else:
        print(f"partial: {done.sum():,}/{n:,} rows done in {output}", flush = True)


if __name__ == "__main__":
    main()
