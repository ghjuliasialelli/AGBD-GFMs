"""
ALOS-PALSAR-2 DN histograms of the FULL AGBD train split of one year, for the SAR clip bounds of the cross-year run
(train on 2019 only): configs/dataset/agbd.yaml's bounds come from AGBD-Lite train, which is 2020-only data, so the
2019 model gets bounds from 2019 training data, with the same recipe (compute_lite_holdout_stats.stats_from_hist:
valid DN > 1, p0.1 / p99.9 of the dB values rounded outwards). Train tiles = biomes_splits_to_name.pkl['train'],
as pangaea/datasets/agbd.py. Writes AGBD-train-<year>-sar-histograms.npz (sar[band, dn], int64) into --out_dir.

Run: .venv/bin/python benchmark_pangaea/holdout/full_year_sar_hist.py --year 2019
"""
import argparse, os, pickle, time
from os.path import join
import h5py
import numpy as np

ALOS_BANDS = ['HH', 'HV']
K = 65536

if __name__ == '__main__' :
    p = argparse.ArgumentParser()
    p.add_argument('--year', type = int, required = True)
    p.add_argument('--h5_dir', default = '/scratch3/gsialelli/patches')
    p.add_argument('--out_dir', default = '/scratch3/gsialelli/patches')
    args = p.parse_args()

    with open(join(args.h5_dir, 'biomes_splits_to_name.pkl'), 'rb') as f : train_tiles = pickle.load(f)['train']
    sar = np.zeros((len(ALOS_BANDS), K), dtype = np.int64)
    n_rows, t0 = 0, time.time()
    for i in range(20) :
        fname = f'data_subset-{args.year}-v4_{i}-20.h5'
        with h5py.File(join(args.h5_dir, fname), 'r') as f :
            for tile in np.intersect1d(list(f.keys()), train_tiles) :
                ds = f[tile]['ALOS_bands']
                idx = [list(ds.attrs['order']).index(b) for b in ALOS_BANDS]
                for s in range(0, ds.shape[0], 20000) :
                    dn = ds[s : s + 20000][..., idx].astype(np.int64)
                    for bi in range(len(ALOS_BANDS)) :
                        d = dn[..., bi]
                        sar[bi] += np.bincount(d[d > 1], minlength = K) # DN 0 / 1 = nodata, as agbd.py alos_dn_to_db
                n_rows += ds.shape[0]
        print(f'{fname}: {n_rows:,} train rows so far, {time.time() - t0:.0f}s', flush = True)
    out = join(args.out_dir, f'AGBD-train-{args.year}-sar-histograms.npz')
    np.savez(out + '.tmp.npz', sar = sar, alos_bands = np.array(ALOS_BANDS), n_rows = n_rows, year = args.year)
    os.replace(out + '.tmp.npz', out)
    print(f'Wrote {out} ({n_rows:,} rows)')
