"""

Build TESSERA-Lite-{split}.h5 from AGBD-Lite-{split}.h5.

For each sample, extracts the 25x25 Tessera embedding patch (year=2020)
centered on the GEDI shot (lat_decimal + lat_offset, lon_decimal + lon_offset)
and writes to the output HDF5 file with this structure:
    /embeddings              int8    (N, 25, 25, 128)   quantised
    /scales                  float32 (N, 25, 25)        dequant factors
    /GEDI/{lat,lon}_{decimal,offset}                    copied from source

Dequantise at use time:  patch = emb[i].astype(float32) * scl[i][..., None]
Out-of-coverage samples carry NaN scales, index-aligned with the source.
Source: https://github.com/ucam-eo/geotessera

Usage:
    python download.py  --input <path to AGBD-Lite-{split}.h5>
                        --output <path to TESSERA-Lite-{split}.h5>
                        --debug N (optional, for visualisation)

    Retry only the samples that previously errored (reads
    {output}.errors.txt, writes into the existing output file):
    python download.py  --input ... --output ... --cache-dir ... --retry-errors

    with the tessera conda environment

"""

###################################################################################################
# Imports
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine, from_origin

from geotessera import GeoTessera
from geotessera.registry import tile_from_world
from geotessera.store import _zone_for_lon

PATCH = 25
HALF = PATCH // 2
BANDS = 128
YEAR = 2020
PX = 10.0  # metres per pixel

# AGBD stores 12 S2 bands starting with B01, so RGB (B04, B03, B02) is at
# indices 3, 2, 1. Adjust if your file differs.
S2_RGB_IDX = (3, 2, 1)


def tile_epsg(tlon: float, tlat: float) -> int:
    """EPSG code for the UTM zone containing a 0.1° tile centre."""
    zone = _zone_for_lon(tlon)
    return (32600 if tlat >= 0 else 32700) + zone


def compute_tile_affine(tlon: float, tlat: float, to_utm: Transformer) -> Affine:
    """Project the 0.1° cell bounds to UTM — matches Tessera tile generation."""
    west, east = tlon - 0.05, tlon + 0.05
    south, north = tlat - 0.05, tlat + 0.05
    xs, ys = to_utm.transform(
        [west, east, west, east],
        [north, north, south, south],
    )
    return from_origin(min(xs), max(ys), PX, PX)


def _flush_and_fsync(dst: h5py.File) -> None:
    """Flush HDF5 caches, then fsync the underlying fd so data hits disk.

    `flush()` alone only pushes to the kernel page cache; a node crash
    between flush and the kernel's own writeback can still corrupt the
    file's metadata.
    """
    dst.flush()
    try:
        fd = dst.id.get_vfd_handle()
        os.fsync(fd)
    except (OSError, AttributeError):
        pass


def _enable_swmr(dst: h5py.File) -> bool:
    """Enable SWMR writer mode if the file supports it.

    SWMR keeps the file in a crash-consistent state: a killed writer
    leaves a valid file truncated at the last flush, rather than a
    corrupt one with dangling metadata. Requires the file to have been
    created with libver='latest'; returns False on older files.
    """
    try:
        dst.swmr_mode = True
        return True
    except (ValueError, RuntimeError) as exc:
        print(f"  (swmr not enabled: {exc}; file created with older libver)")
        return False


class TileFetcher:
    """Parallel tile loader with per-tile ref counts and disk eviction.

    `tile_refs[t]` starts at the total number of samples that will ever read
    from tile `t` (as primary or neighbour). Each call to `release(t)` after
    consuming one sample decrements the count; on reaching zero the tile's
    two `.npy` files are removed from the cache directory.
    """

    def __init__(
        self,
        gt: GeoTessera,
        tile_refs: dict[tuple[float, float], int],
        workers: int,
        verbose: bool,
    ):
        self.gt = gt
        self.tile_refs = dict(tile_refs)
        self.verbose = verbose
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: dict[tuple[float, float], Future] = {}
        self.loaded: dict[tuple[float, float], dict | None] = {}

    def submit(self, tile: tuple[float, float]) -> None:
        if tile in self.loaded or tile in self.futures:
            return
        if self.tile_refs.get(tile, 0) <= 0:
            return
        self.futures[tile] = self.pool.submit(self._fetch, tile)

    def _fetch(self, tile: tuple[float, float]) -> dict | None:
        tlon, tlat = tile
        try:
            emb_file = self.gt.registry.fetch(
                year=YEAR, lon=tlon, lat=tlat,
                is_scales=False, progressbar=False,
            )
            scl_file = self.gt.registry.fetch(
                year=YEAR, lon=tlon, lat=tlat,
                is_scales=True, progressbar=False,
            )
            # mmap: patch extraction only touches a 25x25 region of a
            # ~1000x1000 tile, so the OS only pages in what we read and
            # can evict under pressure. Drastically cuts resident RAM.
            tile_emb = np.load(emb_file, mmap_mode="r")
            tile_scl = np.load(scl_file, mmap_mode="r")
            if tile_scl.ndim == 3:
                tile_scl = tile_scl[..., 0]
            crs = f"EPSG:{tile_epsg(tlon, tlat)}"
            to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            affine = compute_tile_affine(tlon, tlat, to_utm)
            return {
                "emb": tile_emb,
                "scl": tile_scl,
                "crs": crs,
                "affine": affine,
                "inv_affine": ~affine,
                "to_utm": to_utm,
                "to_wgs": to_wgs,
                "emb_path": Path(emb_file),
                "scl_path": Path(scl_file),
            }
        except Exception as exc:
            if self.verbose:
                print(f"  tile ({tlon:+.2f},{tlat:+.2f}) MISSING: {exc}")
            return None

    def get(self, tile: tuple[float, float]) -> dict | None:
        """Block until tile is loaded (or known-missing) and return its data."""
        if tile in self.loaded:
            return self.loaded[tile]
        # Neighbour tiles discovered pixel-by-pixel may not have been
        # pre-registered in tile_refs (corner approximation misses some),
        # in which case submit() is a no-op. Fetch synchronously here.
        if tile not in self.futures:
            self.futures[tile] = self.pool.submit(self._fetch, tile)
        data = self.futures.pop(tile).result()
        self.loaded[tile] = data
        return data

    def release(self, tile: tuple[float, float]) -> None:
        if tile not in self.tile_refs:
            return
        self.tile_refs[tile] -= 1
        if self.tile_refs[tile] > 0:
            return
        self.tile_refs.pop(tile, None)
        data = self.loaded.pop(tile, None)
        if data is not None:
            for p in (data["emb_path"], data["scl_path"]):
                try:
                    p.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def extract_patch(
    lon: float,
    lat: float,
    primary_tile: tuple[float, float],
    fetcher: TileFetcher,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (emb 25x25x128 int8, scl 25x25 float32) or None if primary missing.

    Fast path when the patch lies entirely within the primary tile;
    otherwise composes pixels from neighbour tiles.
    """
    primary = fetcher.get(primary_tile)
    if primary is None:
        return None

    e, n = primary["to_utm"].transform(lon, lat)
    col_f, row_f = primary["inv_affine"] * (e, n)
    col, row = int(round(col_f)), int(round(row_f))
    r0, c0 = row - HALF, col - HALF
    H, W = primary["emb"].shape[:2]

    if 0 <= r0 and 0 <= c0 and r0 + PATCH <= H and c0 + PATCH <= W:
        return (
            primary["emb"][r0:r0 + PATCH, c0:c0 + PATCH].copy(),
            primary["scl"][r0:r0 + PATCH, c0:c0 + PATCH].copy(),
        )

    patch_emb = np.zeros((PATCH, PATCH, BANDS), dtype=np.int8)
    patch_scl = np.full((PATCH, PATCH), np.nan, dtype=np.float32)

    primary_aff = primary["affine"]
    primary_to_wgs = primary["to_wgs"]

    for ii in range(PATCH):
        for jj in range(PATCH):
            p_row = r0 + ii
            p_col = c0 + jj
            if 0 <= p_row < H and 0 <= p_col < W:
                patch_emb[ii, jj] = primary["emb"][p_row, p_col]
                patch_scl[ii, jj] = primary["scl"][p_row, p_col]
                continue
            e_px, n_px = primary_aff * (p_col + 0.5, p_row + 0.5)
            lon_px, lat_px = primary_to_wgs.transform(e_px, n_px)
            nb_tile = tile_from_world(float(lon_px), float(lat_px))
            if nb_tile == primary_tile:
                continue
            nb = fetcher.get(nb_tile)
            if nb is None:
                continue
            e2, n2 = nb["to_utm"].transform(float(lon_px), float(lat_px))
            c2, r2 = nb["inv_affine"] * (e2, n2)
            r2i, c2i = int(round(r2)), int(round(c2))
            nH, nW = nb["emb"].shape[:2]
            if 0 <= r2i < nH and 0 <= c2i < nW:
                patch_emb[ii, jj] = nb["emb"][r2i, c2i]
                patch_scl[ii, jj] = nb["scl"][r2i, c2i]

    return patch_emb, patch_scl


def required_tiles_for_sample(
    lon: float, lat: float, primary_tile: tuple[float, float]
) -> set[tuple[float, float]]:
    """Tiles potentially touched by a 25x25 patch centred at (lon, lat).

    Approximates a 125 m skirt in both directions using WGS84 degrees.
    125 m ≈ 1.125e-3° lat; for lon we divide by cos(lat) to stay safe.
    """
    dlat = HALF * PX / 111_320.0
    dlon = HALF * PX / max(111_320.0 * np.cos(np.deg2rad(lat)), 1.0)
    out = {primary_tile}
    for dy in (-dlat, dlat):
        for dx in (-dlon, dlon):
            out.add(tile_from_world(lon + dx, lat + dy))
    return out


def plot_debug(
    s2_bands: np.ndarray,
    emb_i8: np.ndarray,
    scl_f32: np.ndarray,
    sample_indices: np.ndarray,
    output_png: str,
) -> None:
    import matplotlib.pyplot as plt

    n = len(s2_bands)

    finite_per_patch = np.isfinite(scl_f32).reshape(n, -1).sum(axis=1)
    n_any = int((finite_per_patch > 0).sum())
    n_full = int((finite_per_patch == PATCH * PATCH).sum())
    print(f"debug coverage: {n_full}/{n} fully valid, {n_any}/{n} partially valid")

    rgb = s2_bands[..., list(S2_RGB_IDX)].astype(np.float32)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1)

    with np.errstate(invalid="ignore"):
        emb_f32 = emb_i8.astype(np.float32) * scl_f32[..., None]
    flat = emb_f32.reshape(-1, BANDS)
    valid = np.isfinite(flat).all(axis=1)
    if valid.sum() < 3:
        tessera = np.zeros((n, PATCH, PATCH, 3), dtype=np.float32)
    else:
        mean = flat[valid].mean(axis=0)
        centred = flat[valid] - mean
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        comps = vt[:3]
        proj = (flat - mean) @ comps.T
        tessera = proj.reshape(n, PATCH, PATCH, 3)
        for c in range(3):
            v = tessera[..., c]
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                continue
            lo_c, hi_c = np.percentile(finite, [2, 98])
            tessera[..., c] = np.clip(
                (v - lo_c) / max(hi_c - lo_c, 1e-9), 0, 1
            )
        tessera = np.nan_to_num(tessera)

    fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n), squeeze=False)
    for k in range(n):
        axes[k, 0].imshow(rgb[k])
        axes[k, 0].set_title(f"sample {sample_indices[k]} — S2 RGB")
        axes[k, 0].axis("off")
        axes[k, 1].imshow(tessera[k])
        title = f"sample {sample_indices[k]} — Tessera PCA-RGB"
        if not np.isfinite(scl_f32[k]).any():
            title += " (NaN)"
        axes[k, 1].set_title(title)
        axes[k, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    print(f"wrote {output_png}")


def _write_nan(scl_ds: h5py.Dataset, emb_ds: h5py.Dataset, k: int) -> None:
    """Mark sample `k` as missing: NaN scales, zero embedding."""
    scl_ds[k] = np.full((PATCH, PATCH), np.nan, dtype=np.float32)
    emb_ds[k] = np.zeros((PATCH, PATCH, BANDS), dtype=np.int8)


def run_extraction(
    samples: list[tuple[int, float, float]],
    emb_ds: h5py.Dataset,
    scl_ds: h5py.Dataset,
    cache_dir: Path,
    workers: int,
    lookahead: int,
    verbose: bool,
    max_consecutive_errors: int,
) -> tuple[int, int, list[int]]:
    """Extract patches for `samples` = [(h5_index, lon, lat), ...].

    Writes into `emb_ds`/`scl_ds` at each sample's h5_index. On per-sample
    exception, fills that index with NaN/zero and records the index.
    Aborts if `max_consecutive_errors` exceptions occur back-to-back.

    Returns (n_written, n_missing, error_indices).
    """
    tile_refs: dict[tuple[float, float], int] = defaultdict(int)
    tile_to_local: dict[tuple[float, float], list[tuple[int, float, float]]] = defaultdict(list)
    sample_tiles: dict[int, set[tuple[float, float]]] = {}

    for k, lon, lat in samples:
        pri = tile_from_world(lon, lat)
        tile_to_local[pri].append((k, lon, lat))
        req = required_tiles_for_sample(lon, lat, pri)
        sample_tiles[k] = req
        for t in req:
            tile_refs[t] += 1

    processing_order = list(tile_to_local.keys())
    print(f"{len(tile_to_local):,} primary tiles, "
          f"{len(tile_refs):,} unique tiles incl. neighbours")

    gt = GeoTessera(embeddings_dir=str(cache_dir))
    fetcher = TileFetcher(gt, tile_refs, workers=workers, verbose=verbose)

    for t in processing_order[:lookahead]:
        fetcher.submit(t)

    n_written = 0
    n_missing = 0
    error_indices: list[int] = []
    consecutive_errors = 0
    t0 = time.time()

    try:
        for ti, primary_tile in enumerate(processing_order):
            far_idx = ti + lookahead
            if far_idx < len(processing_order):
                fetcher.submit(processing_order[far_idx])

            for k, lon, lat in tile_to_local[primary_tile]:
                try:
                    result = extract_patch(lon, lat, primary_tile, fetcher)
                except Exception as exc:
                    print(f"  sample {k} ({lon:.4f},{lat:.4f}) ERROR: {exc!r}")
                    _write_nan(scl_ds, emb_ds, k)
                    error_indices.append(k)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"aborting: {consecutive_errors} consecutive errors "
                            f"(>= --max-consecutive-errors={max_consecutive_errors})"
                        ) from exc
                else:
                    consecutive_errors = 0
                    if result is None:
                        _write_nan(scl_ds, emb_ds, k)
                        n_missing += 1
                    else:
                        emb_ds[k], scl_ds[k] = result
                        n_written += 1
                for t in sample_tiles[k]:
                    fetcher.release(t)

            if (ti + 1) % 50 == 0 or ti + 1 == len(processing_order):
                _flush_and_fsync(emb_ds.file)
                elapsed = time.time() - t0
                print(f"  tiles {ti + 1}/{len(processing_order)} "
                      f"written={n_written} miss={n_missing} "
                      f"err={len(error_indices)} "
                      f"live={len(fetcher.loaded)} ({elapsed:.0f}s)")
    finally:
        fetcher.close()

    return n_written, n_missing, error_indices


def _lonlat_from_gedi(
    lat_dec: np.ndarray, lat_off: np.ndarray,
    lon_dec: np.ndarray, lon_off: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat_dec64 = lat_dec.astype(np.float64)
    lon_dec64 = lon_dec.astype(np.float64)
    lats = np.sign(lat_dec64) * (np.abs(lat_dec64) + lat_off.astype(np.float64))
    lons = np.sign(lon_dec64) * (np.abs(lon_dec64) + lon_off.astype(np.float64))
    return lons, lats


def _write_errors_file(path: Path, indices: list[int]) -> None:
    if indices:
        path.write_text("\n".join(str(i) for i in indices) + "\n")
        print(f"wrote {len(indices)} error indices → {path}")
    elif path.exists():
        path.unlink()
        print(f"removed {path} (no errors)")


def _read_errors_file(path: Path) -> list[int]:
    idx = sorted({int(x) for x in path.read_text().split() if x.strip()})
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="AGBD-Lite-{split}.h5")
    ap.add_argument("--output", required=True, help="TESSERA-Lite-{split}.h5")
    ap.add_argument("--cache-dir", required=True,
                    help="Persistent directory for cached tile .npy files. "
                         "Tiles are downloaded here and deleted automatically "
                         "once no remaining sample needs them.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel tile download threads (default: 8).")
    ap.add_argument("--lookahead", type=int, default=None,
                    help="How many upcoming primary tiles to prefetch "
                         "(default: 2*workers).")
    ap.add_argument("--debug", type=int, default=None,
                    help="If set, process only N samples and write a PNG "
                         "comparing S2 RGB to Tessera PCA-RGB.")
    ap.add_argument("--seed", type=int, default=None,
                    help="With --debug: pick N random samples using this seed. "
                         "Without: use evenly-spaced stride.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-consecutive-errors", type=int, default=50,
                    help="Abort if this many per-sample exceptions occur "
                         "back-to-back (default: 50). Single failures are "
                         "logged, marked NaN, and processing continues.")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-process only the indices listed in "
                         "{output}.errors.txt, writing into the existing "
                         "output file (opened in r+ mode).")
    ap.add_argument("--continue", dest="continue_run", action="store_true",
                    help="Resume an interrupted run: open the existing "
                         "output in r+ mode, verify GEDI coords match the "
                         "input, and process only indices whose scales are "
                         "still entirely NaN (untouched or previously "
                         "missing/errored).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # -------------------- sanity checks --------------------
    if args.continue_run and args.retry_errors:
        raise SystemExit(
            "--continue and --retry-errors are mutually exclusive; pick one."
        )

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input file not found: {args.input}")

    output_path = Path(args.output)
    output_exists = output_path.exists()
    resume_flag = args.continue_run or args.retry_errors

    # Split must match between input and output filenames
    # (AGBD-Lite-{split}.h5 ↔ TESSERA-Lite-{split}.h5). Catches the common
    # mistake of pointing --output at the wrong split's file.
    def _split_from_name(name: str) -> str | None:
        for split in ("train", "val", "test"):
            if f"-{split}." in name or name.endswith(f"-{split}.h5"):
                return split
        return None

    in_split = _split_from_name(input_path.name)
    out_split = _split_from_name(output_path.name)
    if in_split is None:
        raise SystemExit(
            f"could not infer split (train/val/test) from input filename: "
            f"{input_path.name}"
        )
    if out_split is None:
        raise SystemExit(
            f"could not infer split (train/val/test) from output filename: "
            f"{output_path.name}"
        )
    if in_split != out_split:
        raise SystemExit(
            f"split mismatch: input is '{in_split}' ({input_path.name}) but "
            f"output is '{out_split}' ({output_path.name}). Refusing to mix "
            f"splits."
        )

    if output_exists and not resume_flag:
        raise SystemExit(
            f"output file already exists: {args.output}\n"
            f"  refusing to overwrite. Use --continue to resume an "
            f"interrupted run, --retry-errors to re-process failed samples, "
            f"or delete the file to start fresh."
        )
    if resume_flag and not output_exists:
        flag = "--continue" if args.continue_run else "--retry-errors"
        raise SystemExit(
            f"output file not found: {args.output}\n"
            f"  {flag} requires an existing output file. Run without it "
            f"to create a fresh one."
        )

    if args.workers < 1:
        raise SystemExit(f"--workers must be >= 1 (got {args.workers})")
    if args.lookahead is not None and args.lookahead < 0:
        raise SystemExit(f"--lookahead must be >= 0 (got {args.lookahead})")

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    lookahead = args.lookahead if args.lookahead is not None else 2 * args.workers
    errors_path = Path(str(args.output) + ".errors.txt")

    # -------------------- retry mode --------------------
    if args.retry_errors:
        if not errors_path.exists():
            raise SystemExit(
                f"--retry-errors requires {errors_path}, but it does not "
                f"exist. There are no failed samples to retry."
            )
        err_indices = _read_errors_file(errors_path)
        if not err_indices:
            print(f"{errors_path} is empty; removing")
            errors_path.unlink()
            return
        print(f"retrying {len(err_indices)} samples from {errors_path}")

        with h5py.File(args.input, "r") as src:
            lat_dec = src["GEDI/lat_decimal"][:][err_indices]
            lat_off = src["GEDI/lat_offset"][:][err_indices]
            lon_dec = src["GEDI/lon_decimal"][:][err_indices]
            lon_off = src["GEDI/lon_offset"][:][err_indices]
        lons, lats = _lonlat_from_gedi(lat_dec, lat_off, lon_dec, lon_off)
        samples = [
            (err_indices[i], float(lons[i]), float(lats[i]))
            for i in range(len(err_indices))
        ]

        with h5py.File(args.output, "r+", libver="latest") as dst:
            _enable_swmr(dst)
            emb_ds = dst["embeddings"]
            scl_ds = dst["scales"]
            try:
                n_written, n_missing, new_errors = run_extraction(
                    samples, emb_ds, scl_ds,
                    cache_dir=cache_dir, workers=args.workers,
                    lookahead=lookahead, verbose=args.verbose,
                    max_consecutive_errors=args.max_consecutive_errors,
                )
            except RuntimeError as exc:
                print(f"\n{exc}")
                # Keep original errors file untouched on abort.
                return
            _flush_and_fsync(dst)

        print(f"\nretry: wrote {n_written}/{len(err_indices)} "
              f"miss={n_missing} still-error={len(new_errors)}")
        _write_errors_file(errors_path, new_errors)
        return

    # -------------------- continue mode --------------------
    if args.continue_run:
        if not Path(args.output).exists():
            print(f"no existing output at {args.output}; nothing to continue")
            return

        with h5py.File(args.input, "r") as src:
            n_total = src["GEDI/lat_decimal"].shape[0]
            if args.debug:
                if args.seed is not None:
                    rng = np.random.default_rng(args.seed)
                    keep = np.sort(rng.choice(n_total, size=args.debug, replace=False))
                else:
                    stride = max(1, n_total // args.debug)
                    keep = np.arange(0, n_total, stride)[: args.debug]
            else:
                keep = np.arange(n_total)
            src_lat_dec = src["GEDI/lat_decimal"][:][keep]
            src_lat_off = src["GEDI/lat_offset"][:][keep]
            src_lon_dec = src["GEDI/lon_decimal"][:][keep]
            src_lon_off = src["GEDI/lon_offset"][:][keep]

        lons, lats = _lonlat_from_gedi(
            src_lat_dec, src_lat_off, src_lon_dec, src_lon_off
        )
        n = len(lats)

        with h5py.File(args.output, "r+", libver="latest") as dst:
            emb_ds = dst["embeddings"]
            scl_ds = dst["scales"]
            if scl_ds.shape[0] != n:
                raise SystemExit(
                    f"shape mismatch: output has {scl_ds.shape[0]} samples, "
                    f"input subset has {n}. Refusing to continue."
                )
            for name, src_arr in (
                ("lat_decimal", src_lat_dec), ("lat_offset", src_lat_off),
                ("lon_decimal", src_lon_dec), ("lon_offset", src_lon_off),
            ):
                dst_arr = dst[f"GEDI/{name}"][:]
                if not np.array_equal(dst_arr, src_arr):
                    raise SystemExit(
                        f"GEDI/{name} differs between input and existing "
                        f"output — refusing to continue (wrong file?)."
                    )

            # Stream the scales dataset to find untouched (all-NaN) rows.
            print(f"scanning {n:,} rows for already-processed samples...")
            done_mask = np.zeros(n, dtype=bool)
            chunk = 4096
            for start in range(0, n, chunk):
                stop = min(n, start + chunk)
                block = scl_ds[start:stop]
                done_mask[start:stop] = np.isfinite(block).reshape(stop - start, -1).any(axis=1)
            todo = np.flatnonzero(~done_mask)
            print(f"already done: {done_mask.sum():,} / {n:,}; "
                  f"remaining: {len(todo):,}")
            if len(todo) == 0:
                print("nothing to do.")
                return

            # Enable SWMR only after all reads (swmr disallows certain
            # operations; scanning above uses the standard writer mode).
            _enable_swmr(dst)

            samples = [(int(k), float(lons[k]), float(lats[k])) for k in todo]
            try:
                n_written, n_missing, error_indices = run_extraction(
                    samples, emb_ds, scl_ds,
                    cache_dir=cache_dir, workers=args.workers,
                    lookahead=lookahead, verbose=args.verbose,
                    max_consecutive_errors=args.max_consecutive_errors,
                )
            except RuntimeError as exc:
                print(f"\n{exc}")
                return
            _flush_and_fsync(dst)

        print(f"\ncontinue: wrote {n_written}/{len(todo)} "
              f"miss={n_missing} errors={len(error_indices)}")
        _write_errors_file(errors_path, error_indices)
        return

    # -------------------- initial run --------------------
    with h5py.File(args.input, "r") as src:
        n_total = src["GEDI/lat_decimal"].shape[0]
        if args.debug:
            if args.seed is not None:
                rng = np.random.default_rng(args.seed)
                keep = np.sort(rng.choice(n_total, size=args.debug, replace=False))
                print(f"debug mode: {len(keep)} random samples "
                      f"(seed={args.seed}) out of {n_total:,}")
            else:
                stride = max(1, n_total // args.debug)
                keep = np.arange(0, n_total, stride)[: args.debug]
                print(f"debug mode: {len(keep)} evenly-spaced samples "
                      f"out of {n_total:,}")
        else:
            keep = np.arange(n_total)

        lat_dec = src["GEDI/lat_decimal"][:][keep]
        lat_off = src["GEDI/lat_offset"][:][keep]
        lon_dec = src["GEDI/lon_decimal"][:][keep]
        lon_off = src["GEDI/lon_offset"][:][keep]
        s2_bands = src["S2_bands"][sorted(keep.tolist())] if args.debug else None

    lons, lats = _lonlat_from_gedi(lat_dec, lat_off, lon_dec, lon_off)
    n = len(lats)
    print(f"{n:,} samples: {args.input} → {args.output}")

    # h5_index == positional index in the output datasets (0..n-1).
    samples = [(k, float(lons[k]), float(lats[k])) for k in range(n)]

    # libver='latest' is required to enable SWMR below; it also keeps the
    # file in a crash-consistent state across flushes.
    with h5py.File(args.output, "w", libver="latest") as dst:
        emb_ds = dst.create_dataset(
            "embeddings", shape=(n, PATCH, PATCH, BANDS), dtype="i1",
            chunks=(1, PATCH, PATCH, BANDS), compression="lzf",
        )
        scl_ds = dst.create_dataset(
            "scales", shape=(n, PATCH, PATCH), dtype="f4",
            chunks=(1, PATCH, PATCH), compression="lzf",
            fillvalue=np.nan,
        )
        gedi = dst.create_group("GEDI")
        gedi.create_dataset("lat_decimal", data=lat_dec)
        gedi.create_dataset("lat_offset",  data=lat_off)
        gedi.create_dataset("lon_decimal", data=lon_dec)
        gedi.create_dataset("lon_offset",  data=lon_off)

        # Must be enabled after all datasets/groups are created: SWMR
        # forbids creating new objects once active.
        _enable_swmr(dst)

        aborted = False
        try:
            n_written, n_missing, error_indices = run_extraction(
                samples, emb_ds, scl_ds,
                cache_dir=cache_dir, workers=args.workers,
                lookahead=lookahead, verbose=args.verbose,
                max_consecutive_errors=args.max_consecutive_errors,
            )
        except RuntimeError as exc:
            print(f"\n{exc}")
            aborted = True
            n_written = n_missing = 0
            error_indices = []
        _flush_and_fsync(dst)

        if not aborted:
            print(f"\nwrote {n_written}/{n} patches "
                  f"(tile-missing={n_missing}, errors={len(error_indices)})")
            _write_errors_file(errors_path, error_indices)

        if args.debug:
            emb_out = emb_ds[:]
            scl_out = scl_ds[:]

    if aborted:
        return

    print(f"wrote {args.output}")

    if args.debug:
        png = args.output.rsplit(".", 1)[0] + "_debug.png"
        plot_debug(s2_bands, emb_out, scl_out, keep, png)


if __name__ == "__main__":
    main()
