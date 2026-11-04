"""

Build OLMO-Lite-{split}.h5 from AGBD-Lite-{split}.h5.

For each sample, extracts the 25x25 OlmoEarth v1.3 embedding patch (year=2020, all 128 dims)
whose centre pixel is the 10 m pixel containing the GEDI shot (sign-aware lat/lon decoding),
and writes to the output HDF5 file with this structure:
    /embeddings              int8    (N, 25, 25, 128)   raw quantised values, -128 = nodata
    /done                    bool    (N,)               sample has been extracted (resume marker)
    /zone, /row, /col        int16/int32 (N,)           centre pixel in utm<zone>/embeddings
    /GEDI/{lat,lon}_{decimal,offset}                    copied from source

Dequantise at use time (signed square root, NOT TESSERA's emb * scale):
    q = emb.astype(float32); v = sign(q) * (|q| / 127.5) ** 2; v[q == -128] = NaN
Mask nodata on the raw -128 BEFORE dequantising or normalising.

Source: https://source.coop/olmoearth/olmoearth-v1-3-embeddings (Zarr v3, one group per UTM
zone, north-up, 10 m, every zone in its NORTHERN UTM CRS EPSG:326NN -- southern samples have
negative northings; never derive the CRS from an S2 tile name, which gives 327NN).
Licence: OlmoEarth Artifact License -- its use restrictions (section 2) must be carried into any
redistribution of the output files.

Pitfalls handled here:
- 2017-2022 are only partially computed and being backfilled. The script refuses to start
  unless the provenance tables say >= --min-coverage of the samples fall in written windows
  (default 1.0), and records the provenance ETag + build version in the output attrs. --continue
  refuses to resume across a change of build version.
- A Zarr inner chunk (256x256 px x 64 bands, ~3 MB compressed) is the smallest decodable unit,
  so reads are done per whole chunk, ref-counted and cached: each chunk is downloaded once.
- Nodata pixels are written as -128, never filled. Fetch failures abort (resume with --continue);
  they are never written as nodata.

Usage:
    python download.py  --input <path to AGBD-Lite-{split}.h5>
                        --output <path to OLMO-Lite-{split}.h5>
                        --cache-dir <dir for the provenance parquet>
                        --debug N --seed S (optional, N random samples)

    Resume an interrupted run (processes samples whose /done flag is False):
    python download.py  --input ... --output ... --cache-dir ... --continue

    then validate with audit.py h5 before computing statistics.

    with the tessera conda environment (zarr 3, fsspec + aiohttp, h5py, pyproj, pandas/pyarrow)

"""

###################################################################################################
# Imports
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import zarr
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit import (STORE_URL, lonlat_from_gedi, provenance_coverage, split_from_name,  # noqa: E402
                   zone_for_lon)

PATCH = 25
HALF = PATCH // 2
BANDS = 128
CHUNK = 256       # inner chunk edge in pixels; a fetch unit is one (CHUNK x CHUNK x 128) block
NODATA = -128
YEAR = 2020


###################################################################################################
# Store access

def open_store() -> zarr.Group:
    # Without this (zarr >= 3.3), nearby chunk ranges are coalesced and the gaps downloaded too.
    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    store = zarr.storage.FsspecStore.from_url(STORE_URL, read_only=True)
    return zarr.open_group(store, mode="r")


class Zone:
    """One utm<NN> group: its array, time index for the year, and lon/lat -> pixel transform."""

    def __init__(self, root: zarr.Group, zone: int, year: int):
        self.zone = zone
        group = root[f"utm{zone:02d}"]
        self.arr = group["embeddings"]
        crs = group.attrs["proj:code"]
        assert crs == f"EPSG:326{zone:02d}", f"utm{zone:02d} has CRS {crs}, expected EPSG:326{zone:02d}"
        a, b, c, d, e, f = group.attrs["spatial:transform"]
        assert b == 0 and d == 0 and a == 10 and e == -10, f"utm{zone:02d}: unexpected transform {group.attrs['spatial:transform']}"
        assert group.attrs.get("spatial:registration", "pixel") == "pixel"
        self.c, self.f = c, f
        years = group["time"][:]
        hit = np.flatnonzero(years == year)
        if len(hit) != 1: raise SystemExit(f"utm{zone:02d}: year {year} not in time axis {years.tolist()}")
        self.t = int(hit[0])
        self.H, self.W = self.arr.shape[2:]
        self.to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def pixel(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row/col of the pixel CONTAINING each point (floor, not round: round picks the nearest
        pixel corner and shifts the patch by one pixel half of the time)."""
        x, y = self.to_utm.transform(lon, lat)
        col = np.floor((np.asarray(x) - self.c) / 10.0).astype(np.int64)
        row = np.floor((self.f - np.asarray(y)) / 10.0).astype(np.int64)
        return row, col

    def read_chunk(self, cr: int, cc: int) -> np.ndarray:
        """(128, CHUNK, CHUNK) int8 block, chunk-aligned so no partial-chunk waste."""
        r0, c0 = cr * CHUNK, cc * CHUNK
        return np.asarray(self.arr[self.t, :, r0:min(r0 + CHUNK, self.H), c0:min(c0 + CHUNK, self.W)])


class ChunkCache:
    """Threaded chunk fetcher with per-chunk ref counts (freed when the last sample needing it is
    done) plus a hard cap on resident chunks as a memory backstop (8 MiB each)."""

    def __init__(self, zones: dict[int, Zone], refs: dict[tuple, int], workers: int, max_chunks: int, retries: int):
        self.zones = zones
        self.refs = dict(refs)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: dict[tuple, Future] = {}
        self.loaded: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.max_chunks = max_chunks
        self.retries = retries
        self.n_fetched = 0

    def _fetch(self, key: tuple) -> np.ndarray:
        zone, cr, cc = key
        for attempt in range(self.retries + 1):
            try:
                return self.zones[zone].read_chunk(cr, cc)
            except Exception as exc:
                if attempt == self.retries: raise RuntimeError(f"chunk {key} failed after {self.retries + 1} attempts") from exc
                time.sleep(2 ** attempt)

    def submit(self, key: tuple) -> None:
        if key in self.loaded or key in self.futures: return
        self.futures[key] = self.pool.submit(self._fetch, key)

    def get(self, key: tuple) -> np.ndarray:
        if key in self.loaded:
            self.loaded.move_to_end(key)
            return self.loaded[key]
        if key not in self.futures: self.submit(key)
        data = self.futures.pop(key).result()
        self.n_fetched += 1
        self.loaded[key] = data
        while len(self.loaded) > self.max_chunks: self.loaded.popitem(last=False)
        return data

    def release(self, key: tuple) -> None:
        self.refs[key] -= 1
        if self.refs[key] <= 0:
            self.refs.pop(key)
            self.loaded.pop(key, None)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def chunks_for(row: int, col: int, zone: int) -> list[tuple]:
    return [(zone, cr, cc)
            for cr in range((row - HALF) // CHUNK, (row + HALF) // CHUNK + 1)
            for cc in range((col - HALF) // CHUNK, (col + HALF) // CHUNK + 1)]


def assemble_patch(row: int, col: int, zone: int, cache: ChunkCache) -> np.ndarray:
    """(25, 25, 128) int8 patch centred on (row, col), stitched from up to 4 chunks."""
    patch = np.full((BANDS, PATCH, PATCH), NODATA, dtype=np.int8)
    r0, c0 = row - HALF, col - HALF
    for key in chunks_for(row, col, zone):
        _, cr, cc = key
        block = cache.get(key)
        br0, bc0 = cr * CHUNK, cc * CHUNK
        rs, re = max(r0, br0), min(r0 + PATCH, br0 + block.shape[1])
        cs, ce = max(c0, bc0), min(c0 + PATCH, bc0 + block.shape[2])
        if rs >= re or cs >= ce: continue
        patch[:, rs - r0:re - r0, cs - c0:ce - c0] = block[:, rs - br0:re - br0, cs - bc0:ce - bc0]
    return np.ascontiguousarray(patch.transpose(1, 2, 0))


###################################################################################################
# Extraction

def run_extraction(todo: np.ndarray, zone: np.ndarray, row: np.ndarray, col: np.ndarray,
                   zones: dict[int, Zone], dst: h5py.File, workers: int, lookahead: int,
                   max_chunks: int, retries: int) -> int:
    """Extract `todo` (output indices), writing /embeddings and /done. Returns # written."""

    # Group samples by the chunk holding their centre, processed in spatial order.
    groups: dict[tuple, list[int]] = defaultdict(list)
    refs: dict[tuple, int] = defaultdict(int)
    for k in todo:
        groups[(int(zone[k]), int(row[k]) // CHUNK, int(col[k]) // CHUNK)].append(int(k))
        for key in chunks_for(int(row[k]), int(col[k]), int(zone[k])): refs[key] += 1
    order = sorted(groups)
    print(f"{len(todo):,} samples in {len(order):,} centre chunks, {len(refs):,} unique chunks to fetch "
          f"(~{len(refs) * 3 / 1024:.0f} GB on the wire)", flush=True)

    emb_ds, done_ds = dst["embeddings"], dst["done"]
    cache = ChunkCache(zones, refs, workers, max_chunks, retries)
    group_chunks = {g: sorted({c for k in groups[g] for c in chunks_for(int(row[k]), int(col[k]), int(zone[k]))}) for g in order}
    for g in order[:lookahead]:
        for c in group_chunks[g]: cache.submit(c)

    n_written, t0 = 0, time.time()
    try:
        for gi, g in enumerate(order):
            if gi + lookahead < len(order):
                for c in group_chunks[order[gi + lookahead]]: cache.submit(c)
            for k in groups[g]:
                emb_ds[k] = assemble_patch(int(row[k]), int(col[k]), int(zone[k]), cache)
                done_ds[k] = True
                n_written += 1
                for c in chunks_for(int(row[k]), int(col[k]), int(zone[k])): cache.release(c)
            if (gi + 1) % 200 == 0 or gi + 1 == len(order):
                dst.flush()
                el = time.time() - t0
                print(f"  groups {gi + 1:,}/{len(order):,}  samples {n_written:,}/{len(todo):,}  "
                      f"chunks fetched {cache.n_fetched:,}/{len(refs):,}  resident {len(cache.loaded)}  "
                      f"{el:.0f}s ({n_written / max(el, 1e-9):.1f} samples/s)", flush=True)
    finally:
        cache.close()
    return n_written


def store_fingerprint(root: zarr.Group) -> dict:
    a = root.attrs
    return {"store_url": STORE_URL,
            "build_version": a["geoemb:build_version"],
            "quantization": json.dumps(a["geoemb:quantization"]),
            "model": a["geoemb:model"]}


###################################################################################################
# Code execution

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="AGBD-Lite-{split}.h5")
    ap.add_argument("--output", required=True, help="OLMO-Lite-{split}.h5")
    ap.add_argument("--cache-dir", required=True, help="Directory for the provenance parquet files.")
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="Refuse to start unless at least this fraction of the samples fall in windows the "
                         "provenance tables mark as written for --year (default 1.0: wait for the backfill).")
    ap.add_argument("--workers", type=int, default=16, help="Parallel chunk download threads.")
    ap.add_argument("--lookahead", type=int, default=None, help="Centre-chunk groups to prefetch (default 4*workers).")
    ap.add_argument("--max-chunks", type=int, default=384,
                    help="Cap on resident decoded chunks, 8 MiB each (default 384 = 3 GiB).")
    ap.add_argument("--retries", type=int, default=4, help="Retries per chunk fetch before aborting.")
    ap.add_argument("--debug", type=int, default=None, help="Process only N samples (random with --seed, else strided).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--continue", dest="continue_run", action="store_true",
                    help="Resume: open the existing output, verify it matches the input and the store build, "
                         "and process the samples whose /done flag is False.")
    args = ap.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file(): raise SystemExit(f"input file not found: {input_path}")
    in_split, out_split = split_from_name(input_path.name), split_from_name(output_path.name)
    if in_split is None or out_split is None or in_split != out_split:
        raise SystemExit(f"split mismatch or not inferable: input {input_path.name} vs output {output_path.name}")
    if output_path.exists() and not args.continue_run:
        raise SystemExit(f"output already exists: {output_path}\n  use --continue to resume, or move it away to start fresh.")
    if args.continue_run and not output_path.exists(): raise SystemExit(f"--continue but no output at {output_path}")
    lookahead = args.lookahead if args.lookahead is not None else 4 * args.workers
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- samples --------------------
    with h5py.File(input_path, "r") as src:
        n_total = src["GEDI/lat_decimal"].shape[0]
        if args.debug:
            if args.seed is not None: keep = np.sort(np.random.default_rng(args.seed).choice(n_total, size=args.debug, replace=False))
            else: keep = np.arange(0, n_total, max(1, n_total // args.debug))[: args.debug]
        else: keep = np.arange(n_total)
        gedi = {k: src[f"GEDI/{k}"][:][keep] for k in ("lat_decimal", "lat_offset", "lon_decimal", "lon_offset")}
    lon, lat = lonlat_from_gedi(gedi["lat_decimal"], gedi["lat_offset"], gedi["lon_decimal"], gedi["lon_offset"])
    n = len(lon)
    assert n > 0, "no samples selected"
    print(f"{n:,} samples: {input_path} -> {output_path} (year {args.year})", flush=True)

    # -------------------- coverage gate --------------------
    cov = provenance_coverage(lon, lat, args.year, cache_dir)
    print(f"provenance {args.year}: {cov['frac']:.2%} of samples in written windows (etag {cov['etag']})", flush=True)
    for z, (m, k) in sorted(cov["per_zone"].items()):
        if m < k: print(f"    utm{z:02d}: {m:,}/{k:,} covered", flush=True)
    if cov["frac"] < args.min_coverage:
        raise SystemExit(f"coverage {cov['frac']:.2%} < --min-coverage {args.min_coverage:.2%}: "
                         f"{args.year} is not fully computed yet. Not building.")

    # -------------------- grid positions --------------------
    root = open_store()
    fp = store_fingerprint(root)
    zone = zone_for_lon(lon)
    zones = {int(z): Zone(root, int(z), args.year) for z in np.unique(zone)}
    row, col = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for z, Z in zones.items():
        m = zone == z
        row[m], col[m] = Z.pixel(lon[m], lat[m])
        assert (row[m] >= HALF).all() and (row[m] < Z.H - HALF).all() and (col[m] >= HALF).all() and (col[m] < Z.W - HALF).all(), \
            f"utm{z:02d}: a patch falls off the array"

    # -------------------- output --------------------
    if args.continue_run:
        dst = h5py.File(output_path, "r+")
        if dst["embeddings"].shape[0] != n: raise SystemExit(f"output has {dst['embeddings'].shape[0]} samples, input selection has {n}")
        for k, v in gedi.items():
            if not np.array_equal(dst[f"GEDI/{k}"][:], v): raise SystemExit(f"GEDI/{k} differs from the input -- wrong file?")
        if int(dst.attrs["year"]) != args.year: raise SystemExit(f"output is year {dst.attrs['year']}, asked for {args.year}")
        if dst.attrs["build_version"] != fp["build_version"]:
            raise SystemExit(f"store build changed since this file was started:\n  file : {dst.attrs['build_version']}\n"
                             f"  store: {fp['build_version']}\nRefusing to mix builds; start a fresh file.")
        if not (np.array_equal(dst["row"][:], row) and np.array_equal(dst["col"][:], col) and np.array_equal(dst["zone"][:], zone)):
            raise SystemExit("recomputed centre pixels differ from the stored ones -- refusing to continue")
        todo = np.flatnonzero(~dst["done"][:])
        print(f"already done: {n - len(todo):,}/{n:,}; remaining {len(todo):,}", flush=True)
    else:
        dst = h5py.File(output_path, "w")
        dst.create_dataset("embeddings", shape=(n, PATCH, PATCH, BANDS), dtype="i1",
                           chunks=(1, PATCH, PATCH, BANDS), compression="lzf", fillvalue=NODATA)
        dst.create_dataset("done", shape=(n,), dtype=bool, fillvalue=False)
        dst.create_dataset("zone", data=zone.astype(np.int16))
        dst.create_dataset("row", data=row.astype(np.int32))
        dst.create_dataset("col", data=col.astype(np.int32))
        g = dst.create_group("GEDI")
        for k, v in gedi.items(): g.create_dataset(k, data=v)
        if args.debug: dst.create_dataset("source_index", data=keep)
        dst.attrs.update({"year": args.year, "nodata": NODATA, "layout": "N,H,W,C (C = 128 OlmoEarth dims)",
                          "dequantize": "sign(q) * (|q| / 127.5) ** 2, NaN where q == -128",
                          "centre_pixel": "pixel containing the GEDI shot (floor), patch rows/cols centre +- 12",
                          "provenance_windows_etag": cov["etag"], "provenance_coverage": cov["frac"],
                          "input": str(input_path.resolve()), "input_size": input_path.stat().st_size,
                          "input_mtime": input_path.stat().st_mtime,
                          "created_utc": datetime.now(timezone.utc).isoformat(),
                          "licence": "OlmoEarth Artifact License (https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/LICENSE.txt)",
                          **fp})
        todo = np.arange(n)

    try:
        n_written = run_extraction(todo, zone, row, col, zones, dst, args.workers, lookahead, args.max_chunks, args.retries)
        dst.flush()
    finally:
        dst.close()
    print(f"\nwrote {n_written:,}/{len(todo):,} samples -> {output_path}\nnext: python audit.py h5 --olmo {output_path} --agbd {input_path}")


if __name__ == "__main__":
    main()
