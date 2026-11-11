"""

Audit of a TESSERA-{split}-{year}.h5 built by download_full.py. Exits non-zero on failure, so it can gate a pipeline:

    - every row done; /tiles and the GEDI coordinates identical to a fresh layout of the AGBD files;
    - over the whole population: NaN pixel fraction, NaN centre pixels, rows never written (all-NaN scales) and
      all-zero embeddings under thresholds;
    - ALIGNMENT: embeddings -> S2 NDVI, the stored orientation must score best (olmoearth/audit.py's check);
    - with --lite-ref (2020 only): rows at the coordinates of a TESSERA-Lite file must be bit-identical to it.

Run, with the tessera conda environment:
    python audit_full.py --tessera TESSERA-test-2020.h5 --agbd <dir with data_subset files> --year 2020 --lite-ref TESSERA-Lite-test.h5

"""

###################################################################################################
# Imports
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "olmoearth"))  # append: "download" must stay this directory's
from audit import GEDI_KEYS, alignment_check, full_gedi, full_layout, layout_from_h5  # noqa: E402


def parse_arguments():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tessera", required = True, help = "TESSERA-{split}-{year}.h5")
    ap.add_argument("--agbd", required = True, help = "directory with the data_subset files it was built from")
    ap.add_argument("--year", type = int, required = True)
    ap.add_argument("--lite-ref", default = None, help = "TESSERA-Lite-{split}.h5 to compare with (2020 only)")
    ap.add_argument("--max-nan-frac", type = float, default = 1e-3, help = "max fraction of NaN pixels over all patches")
    ap.add_argument("--max-centre-nan", type = int, default = 0, help = "max number of samples whose centre pixel is NaN")
    ap.add_argument("--max-no-data-frac", type = float, default = 1e-3, help = "max fraction of rows whose scales are all 0, TESSERA's no-data encoding (0.05%% of TESSERA-Lite-test)")
    ap.add_argument("--n-align", type = int, default = 600, help = "samples used by the alignment check")
    ap.add_argument("--min-align-margin", type = float, default = 0.005, help = "required R^2 lead of the stored orientation")
    ap.add_argument("--allow-partial", action = "store_true", help = "debug: audit only the rows marked done")
    ap.add_argument("--seed", type = int, default = 0)
    return ap.parse_args()


###################################################################################################
# Code execution

def main() -> int:
    args = parse_arguments()
    fails, agbd = [], Path(args.agbd)
    with h5py.File(args.tessera, "r") as ft:
        n = ft["embeddings"].shape[0]
        print(f"{args.tessera}: n={n:,} year={ft.attrs['year']} split={ft.attrs['split']} manifest={ft.attrs['manifest_sha256'][:12]}", flush = True)
        if int(ft.attrs["year"]) != args.year: fails.append(f"year attr {ft.attrs['year']} != {args.year}")
        layout = layout_from_h5(ft)
        redo = full_layout(agbd, int(ft.attrs["year"]), ft.attrs["split"], Path(ft.attrs["mapping"]))
        if layout != redo: fails.append("/tiles table differs from the layout recomputed from the AGBD files")
        if sum(t["count"] for t in redo) != n: fails.append("N differs from the AGBD files")
        src_gedi = full_gedi(agbd, redo)
        for k in GEDI_KEYS:
            if not np.array_equal(ft[f"GEDI/{k}"][:], src_gedi[k]): fails.append(f"GEDI/{k} differs from the AGBD files")
        done = ft["done"][:]
        if not done.all() and not args.allow_partial: fails.append(f"{(~done).sum():,} rows not done")
        n_done = int(done.sum()); assert n_done > 0, "no row is done"

        # The whole population (of done rows)
        nan_px, nan_centre, never, zero, no_data, bs = 0, 0, 0, 0, 0, 4096
        per_sample_nan = np.zeros(n, np.int32)
        for s in range(0, n, bs):
            keep = done[s : s + bs]
            if not keep.any(): continue
            scl, emb = ft["scales"][s : s + bs][keep], ft["embeddings"][s : s + bs][keep]
            nan = ~np.isfinite(scl)
            per_sample_nan[s : s + bs][keep] = nan.reshape(len(scl), -1).sum(1)
            nan_px += int(nan.sum()); nan_centre += int(nan[:, 12, 12].sum()); never += int(nan.reshape(len(scl), -1).all(1).sum())
            # all-zero embeddings with all-zero scales are TESSERA's no-data (also in the April TESSERA-Lite); with any other scales, something went wrong
            all_zero, zero_scales = (emb.reshape(len(emb), -1) == 0).all(1), (scl.reshape(len(scl), -1) == 0).all(1)
            zero += int((all_zero & ~zero_scales).sum()); no_data += int((all_zero & zero_scales).sum())
        frac = nan_px / (n_done * 625)
        print(f"rows done {n_done:,}/{n:,}  NaN pixels {frac:.4%}  samples with any NaN {(per_sample_nan > 0).sum():,}  centre NaN {nan_centre:,}  all-NaN rows {never:,}  "
              f"no-data rows (embeddings and scales all 0) {no_data:,} ({no_data / n_done:.4%})  other all-zero rows {zero:,}", flush = True)
        if no_data / n_done > args.max_no_data_frac: fails.append(f"no-data rows {no_data / n_done:.4%} > {args.max_no_data_frac:.4%}")
        if frac > args.max_nan_frac: fails.append(f"NaN pixel fraction {frac:.4%} > {args.max_nan_frac:.4%}")
        if nan_centre > args.max_centre_nan: fails.append(f"{nan_centre} centre pixels are NaN (> {args.max_centre_nan})")
        if never: fails.append(f"{never} done rows have no pixel at all")
        if zero: fails.append(f"{zero} done rows have an all-zero embedding but non-zero scales")

        # Alignment against S2 NDVI
        rng = np.random.default_rng(args.seed)
        cand = np.flatnonzero(done & (per_sample_nan == 0))
        pick = np.sort(rng.choice(cand, size = min(args.n_align, len(cand)), replace = False))
        assert len(pick) >= 50, f"only {len(pick)} clean samples for the alignment check"
        starts = np.array([t["start"] for t in redo])
        def s2_patch(k):
            t = redo[int(np.searchsorted(starts, k, side = "right")) - 1]
            with h5py.File(agbd / t["file"], "r") as f: return f[t["tile"]]["S2_bands"][k - t["start"]]
        feats = np.stack([ft["embeddings"][i].astype(np.float32) * ft["scales"][i][..., None] for i in pick])
        scores = alignment_check(feats, np.stack([s2_patch(int(i)) for i in pick]), rng)
        best = max(scores, key = scores.get)
        print(f"alignment R^2 vs S2 NDVI (n={len(pick)} samples):")
        for k, v in sorted(scores.items(), key = lambda kv: -kv[1]): print(f"    {k:<10} {v:.4f}")
        margin = scores["stored"] - max(v for k, v in scores.items() if k != "stored")
        if best != "stored" or margin < args.min_align_margin: fails.append(f"alignment: best transform is '{best}', stored-vs-next margin {margin:.4f} < {args.min_align_margin}")

        # Bit-identical to TESSERA-Lite at the same coordinates
        if args.lite_ref:
            with h5py.File(args.lite_ref, "r") as fl:
                keys = {key: i for i, key in enumerate(zip(*(fl[f"GEDI/{k}"][:].tolist() for k in GEDI_KEYS)))}
                rows = [(k, keys[key]) for k, key in enumerate(zip(*(src_gedi[g].tolist() for g in GEDI_KEYS))) if done[k] and key in keys]
                assert len(rows) > 0, "no row of this file is in the TESSERA-Lite reference"
                differ = sum(not (np.array_equal(ft["embeddings"][a], fl["embeddings"][b]) and np.array_equal(ft["scales"][a], fl["scales"][b], equal_nan = True)) for a, b in rows)
                print(f"TESSERA-Lite cross-check: {len(rows):,} rows at Lite coordinates ({len(keys):,} Lite coordinates), {differ:,} differ", flush = True)
                if differ: fails.append(f"{differ} rows differ from {args.lite_ref}")

    for f in fails: print("FAIL:", f)
    print("PASS" if not fails else f"{len(fails)} check(s) failed")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
