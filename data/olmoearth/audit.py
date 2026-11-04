"""

Audits for OLMO-Lite. Every subcommand exits non-zero on failure, so they can gate a pipeline.

    coverage : metadata only, no embeddings read. For each AGBD-Lite file, the fraction of samples
               whose centre falls in a window that the OlmoEarth provenance tables mark as written
               for --year, overall and per UTM zone. Use it to check whether the 2020 backfill is
               done (the store covers only some countries for 2017-2022):
                   python audit.py coverage --input AGBD-Lite-{train,val,test}.h5 --cache-dir DIR

    h5       : post-build checks of OLMO-Lite-{split}.h5 against its AGBD-Lite source:
               - every sample done, N and GEDI coordinates identical to the source;
               - nodata (-128) pixel fraction and centre-pixel nodata count under thresholds;
               - no sample that is entirely 0 (never-written rows look like that, not like -128);
               - ALIGNMENT: a linear map from the embeddings to S2 NDVI (from the AGBD-Lite S2
                 patches, i.e. independent data) is fitted on half of a random subset and scored
                 on the other half for the patch as stored, flipped up-down / left-right, and
                 shifted by 1-2 px. The stored orientation must score best: this catches a
                 south-up read, a transposed patch and an off-by-one centre pixel.
                   python audit.py h5 --olmo OLMO-Lite-test.h5 --agbd AGBD-Lite-test.h5

env: tessera

"""

###################################################################################################
# Imports
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

STORE_URL = "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
PROVENANCE_URL = "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/provenance"
SHARD = 2048
# The source.coop proxy returns 403 to Python's default "Python-urllib" User-Agent (and to HEAD).
HEADERS = {"User-Agent": "agbd-gfms/olmo-lite"}
NODATA = -128

# AGBD stores 12 S2 bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12.
S2_B04, S2_B08 = 3, 7


###################################################################################################
# Shared helpers (imported by download.py)

def lonlat_from_gedi(lat_dec, lat_off, lon_dec, lon_off) -> tuple[np.ndarray, np.ndarray]:
    """lat_decimal is SIGNED, lat_offset unsigned: value = sign(dec) * (|dec| + off)."""
    lat_dec, lon_dec = np.asarray(lat_dec, np.float64), np.asarray(lon_dec, np.float64)
    lats = np.sign(lat_dec) * (np.abs(lat_dec) + np.asarray(lat_off, np.float64))
    lons = np.sign(lon_dec) * (np.abs(lon_dec) + np.asarray(lon_off, np.float64))
    return lons, lats


def zone_for_lon(lon) -> np.ndarray:
    return (np.floor((np.asarray(lon) + 180) / 6).astype(np.int64) % 60) + 1


def split_from_name(name: str) -> str | None:
    for split in ("train", "val", "test"):
        if f"-{split}." in name or name.endswith(f"-{split}.h5"): return split
    return None


def _zone_transforms() -> dict[int, list]:
    with urllib.request.urlopen(urllib.request.Request(f"{STORE_URL}/zarr.json", headers=HEADERS), timeout=60) as r: meta = json.load(r)
    md = meta["consolidated_metadata"]["metadata"]
    return {int(k[3:]): v["attributes"]["spatial:transform"] for k, v in md.items() if k.startswith("utm") and "/" not in k}


def fetch_provenance(year: int, cache_dir: Path) -> tuple[Path, str]:
    """Download provenance/windows/{year}.parquet unless the cached copy has the same ETag."""
    url = f"{PROVENANCE_URL}/windows/{year}.parquet"
    with urllib.request.urlopen(urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"}), timeout=60) as r:
        etag = r.headers["ETag"].strip('"')
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, tag_file = cache_dir / f"windows_{year}.parquet", cache_dir / f"windows_{year}.etag"
    if not (path.exists() and tag_file.exists() and tag_file.read_text() == etag):
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=600) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.replace(path)
        tag_file.write_text(etag)
    return path, etag


def provenance_coverage(lon: np.ndarray, lat: np.ndarray, year: int, cache_dir: Path) -> dict:
    """Fraction of points whose 2048-px window is `written` for `year`, overall and per zone."""
    import pandas as pd
    from pyproj import Transformer
    path, etag = fetch_provenance(year, Path(cache_dir))
    w = pd.read_parquet(path, columns=["zone", "col", "row", "written"])
    written = set(map(tuple, w.loc[w.written, ["zone", "col", "row"]].to_numpy().tolist()))
    tfs = _zone_transforms()
    zone = zone_for_lon(lon)
    hit = np.zeros(len(lon), bool)
    per_zone = {}
    for z in np.unique(zone):
        m = np.flatnonzero(zone == z)
        a, _, c, _, e, f = tfs[int(z)]
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:326{int(z):02d}", always_xy=True).transform(lon[m], lat[m])
        wc = (np.floor((np.asarray(x) - c) / a).astype(np.int64)) // SHARD
        wr = (np.floor((np.asarray(y) - f) / e).astype(np.int64)) // SHARD
        hit[m] = [(int(z), int(cc), int(rr)) in written for cc, rr in zip(wc, wr)]
        per_zone[int(z)] = (int(hit[m].sum()), len(m))
    return {"frac": float(hit.mean()), "hit": hit, "per_zone": per_zone, "etag": etag}


def dequantize(q: np.ndarray) -> np.ndarray:
    """int8 -> float32 signed square root; nodata (-128) -> NaN. Mask on the raw value first."""
    v = q.astype(np.float32)
    out = np.sign(v) * (np.abs(v) / 127.5) ** 2
    out[q == NODATA] = np.nan
    return out


###################################################################################################
# coverage

def cmd_coverage(args) -> int:
    import h5py
    ok = True
    for p in args.input:
        with h5py.File(p, "r") as f:
            g = f["GEDI"]
            lon, lat = lonlat_from_gedi(g["lat_decimal"][:], g["lat_offset"][:], g["lon_decimal"][:], g["lon_offset"][:])
        assert len(lon) > 0, f"{p}: no samples"
        cov = provenance_coverage(lon, lat, args.year, Path(args.cache_dir))
        print(f"{Path(p).name}: n={len(lon):,}  {args.year} coverage {cov['frac']:.2%}  (windows etag {cov['etag']})")
        for z, (m, k) in sorted(cov["per_zone"].items()):
            print(f"    utm{z:02d}: {m:>7,}/{k:<7,} {m / k:7.1%}")
        ok &= cov["frac"] >= args.min_coverage
    print("PASS" if ok else f"FAIL: coverage below {args.min_coverage:.2%}")
    return 0 if ok else 1


###################################################################################################
# h5

def _r2(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    pred = X @ coef
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def alignment_check(emb: np.ndarray, s2: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """R^2 of an embedding -> NDVI linear map, fitted on the stored orientation of half the samples
    and scored on the other half under several geometric transforms of the embedding patch."""
    red, nir = s2[..., S2_B04].astype(np.float32), s2[..., S2_B08].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    feats = dequantize(emb)  # (n, 25, 25, 128)
    n = len(emb)
    perm = rng.permutation(n)
    fit, ev = perm[: n // 2], perm[n // 2:]
    m = 4  # crop margin so shifted versions stay inside the patch

    def pairs(idx, tf):
        F = np.stack([tf(feats[i]) for i in idx])[:, m:-m, m:-m].reshape(-1, feats.shape[-1])
        y = ndvi[idx][:, m:-m, m:-m].reshape(-1)
        ok = np.isfinite(F).all(1) & np.isfinite(y)
        return np.c_[F[ok], np.ones(ok.sum())], y[ok]

    transforms = {
        "stored": lambda p: p,
        "flip_ud": lambda p: p[::-1],
        "flip_lr": lambda p: p[:, ::-1],
        "transpose": lambda p: p.transpose(1, 0, 2),
        "shift_r+1": lambda p: np.roll(p, 1, 0), "shift_r-1": lambda p: np.roll(p, -1, 0),
        "shift_c+1": lambda p: np.roll(p, 1, 1), "shift_c-1": lambda p: np.roll(p, -1, 1),
        "shift_r+2": lambda p: np.roll(p, 2, 0), "shift_c+2": lambda p: np.roll(p, 2, 1),
    }
    X, y = pairs(fit, transforms["stored"])
    assert len(y) > 1000, f"only {len(y)} valid pixels to fit the alignment check"
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {name: _r2(*pairs(ev, tf), coef) for name, tf in transforms.items()}


def cmd_h5(args) -> int:
    import h5py
    fails = []
    with h5py.File(args.olmo, "r") as fo, h5py.File(args.agbd, "r") as fa:
        n = fo["embeddings"].shape[0]
        src_idx = fo["source_index"][:] if "source_index" in fo else np.arange(fa["GEDI/lat_decimal"].shape[0])
        print(f"{args.olmo}: n={n:,} year={fo.attrs.get('year')} build={fo.attrs.get('build_version')}")
        if len(src_idx) != n: fails.append(f"N mismatch: olmo {n} vs source selection {len(src_idx)}")
        for k in ("lat_decimal", "lat_offset", "lon_decimal", "lon_offset"):
            if not np.array_equal(fo[f"GEDI/{k}"][:], fa[f"GEDI/{k}"][:][src_idx]): fails.append(f"GEDI/{k} differs from source")
        if int(fo.attrs["year"]) != args.year: fails.append(f"year attr {fo.attrs['year']} != {args.year}")
        done = fo["done"][:]
        if not done.all(): fails.append(f"{(~done).sum():,} samples not done (resume with download.py --continue)")

        # Stream the whole population for nodata / all-zero rows.
        nod_px, nod_centre, all_zero, bs = 0, 0, 0, 2048
        per_sample_nod = np.zeros(n, np.int32)
        for s in range(0, n, bs):
            e = fo["embeddings"][s:s + bs]
            nod = (e == NODATA).all(-1)
            if ((e == NODATA).any(-1) != nod).any(): fails.append(f"rows {s}-{s + bs}: pixels with partial nodata bands")
            per_sample_nod[s:s + bs] = nod.reshape(len(e), -1).sum(1)
            nod_px += int(nod.sum())
            nod_centre += int(nod[:, 12, 12].sum())
            all_zero += int((e.reshape(len(e), -1) == 0).all(1).sum())
        frac = nod_px / (n * 625)
        print(f"nodata pixels {frac:.4%}  samples with any nodata {(per_sample_nod > 0).sum():,}  "
              f"centre nodata {nod_centre:,}  all-zero samples {all_zero:,}")
        if frac > args.max_nodata_frac: fails.append(f"nodata pixel fraction {frac:.4%} > {args.max_nodata_frac:.4%}")
        if nod_centre > args.max_centre_nodata: fails.append(f"{nod_centre} centre pixels are nodata (> {args.max_centre_nodata})")
        if all_zero: fails.append(f"{all_zero} samples are entirely 0 (never written?)")

        # Alignment against S2 NDVI on a random subset of samples without nodata.
        rng = np.random.default_rng(args.seed)
        cand = np.flatnonzero((per_sample_nod == 0) & done)
        pick = np.sort(rng.choice(cand, size=min(args.n_align, len(cand)), replace=False))
        assert len(pick) >= 50, f"only {len(pick)} clean samples for the alignment check"
        emb = np.stack([fo["embeddings"][i] for i in pick])
        s2 = np.stack([fa["S2_bands"][int(src_idx[i])] for i in pick])
    scores = alignment_check(emb, s2, rng)
    best = max(scores, key=scores.get)
    print(f"alignment R^2 vs S2 NDVI (n={len(pick)} samples):")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]): print(f"    {k:<10} {v:.4f}")
    margin = scores["stored"] - max(v for k, v in scores.items() if k != "stored")
    if best != "stored" or margin < args.min_align_margin:
        fails.append(f"alignment: best transform is '{best}', stored-vs-next margin {margin:.4f} < {args.min_align_margin}")

    for f_ in fails: print("FAIL:", f_)
    print("PASS" if not fails else f"{len(fails)} check(s) failed")
    return 0 if not fails else 1


###################################################################################################
# Code execution

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--input", nargs="+", required=True, help="AGBD-Lite-{split}.h5 file(s)")
    c.add_argument("--cache-dir", required=True)
    c.add_argument("--year", type=int, default=2020)
    c.add_argument("--min-coverage", type=float, default=1.0)
    h = sub.add_parser("h5")
    h.add_argument("--olmo", required=True, help="OLMO-Lite-{split}.h5")
    h.add_argument("--agbd", required=True, help="the AGBD-Lite-{split}.h5 it was built from")
    h.add_argument("--year", type=int, default=2020)
    h.add_argument("--max-nodata-frac", type=float, default=1e-3, help="max fraction of nodata pixels over all patches")
    h.add_argument("--max-centre-nodata", type=int, default=0, help="max number of samples whose centre pixel is nodata")
    h.add_argument("--n-align", type=int, default=600, help="samples used by the alignment check")
    h.add_argument("--min-align-margin", type=float, default=0.005, help="required R^2 lead of the stored orientation")
    h.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(cmd_coverage(args) if args.cmd == "coverage" else cmd_h5(args))
