"""

Leave-one-region-out AGBD-Lite inputs for the PANGAEA runs (TerraMind+SAR LoRA, ../train_runs/lora_holdout/).

Writes three things:

1. AGBD-Lite-continents.npz (into --out_dir, default /scratch3/gsialelli/patches/AGBD-Lite)
   One int8 continent code per AGBD-Lite row, for train / val / test, plus `names` and `agbd_sha1_<split>` (sha1 of
   the split's GEDI/agbd column: the dataset refuses a file with another row order, e.g. AGBD-Lite-MT3-*). It is what
   pangaea/datasets/agbdlite.py reads to drop (train, val) or keep (test) a region. The labels are built
   with model/dataset.py's lite_rows_for_region (mapping_lite_to_og.pkl + tiles_per_region.pkl), i.e. the
   SAME definition the AEF / OLMO hold-out runs used, and the continents are asserted to PARTITION every split.

2. configs/dataset/agbdlite-holdout-<R>.yaml (into --pangaea_dir)
   `agbdlite` + hold_out_region / test_region = R + the S2 / SAR normalisation statistics recomputed on the
   Lite train rows OUTSIDE R (as the AEF / OLMO hold-out runs use region-excluded stats). The recipe is that of
   configs/dataset/agbdlite.yaml:
   - optical: mean / std / min / max of the surface reflectance of valid pixels, exactly as
     data/agbd_lite/compute_statistics.py (sr = (DN - 1000*BOA_offset) / 10000, DN == 0 and sr <= 0 excluded).
     NormalizeMinMax only uses min / max: these are the true extremes, so one outlier pixel can move them.
   - sar: ALOS-PALSAR-2 gamma0 in dB (20 log10 DN - 83) of valid pixels (DN > 1). Clip bounds = p0.1 / p99.9,
     rounded OUTWARDS to whole dB (floor / ceil); mean / std on the clipped dB. The global file was written by
     hand (no script in either repo); this rule reproduces its four bounds and its mean / std, which is asserted.
   Before writing anything, the stats over ALL train rows must reproduce configs/dataset/agbdlite.yaml
   (exit 1 otherwise): that is the check that the recipe here is the one the baseline was trained with.

3. AGBD-Lite-train-histograms.npz (into --out_dir): the per-continent DN histograms the stats come from, so any
   other combination can be derived without re-reading the h5 (--from_cache).

All statistics are computed EXACTLY from integer histograms (S2: DN - 1000*offset; SAR: DN), per continent,
in one pass; "everything but R" is the sum of the other continents' histograms.

Run (local, CPU, ~15-30 min for the pass; the rest is instant):
    .venv/bin/python benchmark_pangaea/holdout/compute_lite_holdout_stats.py
    .venv/bin/python benchmark_pangaea/holdout/compute_lite_holdout_stats.py --from_cache   # re-derive only

env: .venv (imports model/dataset.py and the repo-root config.py)

"""

###################################################################################################
# Imports

import argparse
import hashlib
import os
import sys
from os.path import join, dirname, abspath, isfile

import h5py
import numpy as np
import yaml

REPO = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, join(REPO, 'model'))
from config import get_paths, is_local # repo-root config.py
from dataset import continent_to_region, lite_rows_for_region

CONTINENTS = list(continent_to_region) # code i <-> CONTINENTS[i], stored in the npz as `names`
HOLDOUT = ['SouthAsia', 'Africa', 'SouthAmerica'] # the regions of the AEF / OLMO hold-out runs
SPLITS = ['train', 'val', 'test']
S2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] # agbdlite.py order
ALOS_BANDS = ['HH', 'HV'] # agbdlite.py order (yaml names them VV / VH)
K = 65536 # histogram length: S2 values DN - 1000*offset > 0 and SAR DN are both in [0, 65535]
BATCH = 8192
P_LO, P_HI = 0.001, 0.999

###################################################################################################
# Row -> continent

def continent_labels(paths, split, n_rows) :
    """One int8 code per row of AGBD-Lite-<split>.h5, from model/dataset.py; asserts the continents partition it."""
    label = np.full(n_rows, -1, dtype = np.int8)
    for ci, c in enumerate(CONTINENTS) :
        rows = lite_rows_for_region(split, paths['aef_norm'], paths['map'], c, True, n_rows)
        assert np.all(label[rows] == -1), f'{split}: rows assigned to two continents ({c})'
        label[rows] = ci
    assert np.all(label >= 0), f'{split}: {np.sum(label < 0)} rows belong to no continent'
    return label

###################################################################################################
# Histogram pass

def histogram_pass(path_train, label) :
    """Per-continent integer histograms of the train split: s2[c, band, k] and sar[c, band, dn]."""
    n_c = len(CONTINENTS)
    s2 = np.zeros((n_c, len(S2_BANDS), K), dtype = np.int64)
    sar = np.zeros((n_c, len(ALOS_BANDS), K), dtype = np.int64)
    with h5py.File(path_train, 'r') as f :
        n = f['GEDI']['agbd'].shape[0]
        assert n == len(label)
        s2_order, alos_order = list(f['S2_bands'].attrs['order']), list(f['ALOS_bands'].attrs['order'])
        s2_idx, alos_idx = [s2_order.index(b) for b in S2_BANDS], [alos_order.index(b) for b in ALOS_BANDS]
        has_offset = 'S2_boa_offset' in f['Sentinel_metadata']
        for i in range(0, n, BATCH) :
            j = min(i + BATCH, n)
            lab = label[i:j].astype(np.int64)[:, None, None] # (b, 1, 1)
            dn_s2 = f['S2_bands'][i:j][..., s2_idx].astype(np.int64) # (b, 25, 25, 12)
            off = f['Sentinel_metadata']['S2_boa_offset'][i:j].astype(np.int64) if has_offset else np.zeros(j - i, dtype = np.int64)
            assert np.isin(off, [0, 1]).all(), f'unexpected S2_boa_offset values {np.unique(off)}'
            k = dn_s2 - 1000 * off[:, None, None, None]
            for bi in range(len(S2_BANDS)) :
                kb, db = k[..., bi], dn_s2[..., bi]
                valid = (db != 0) & (kb > 0) # == compute_statistics.py: nodata DN 0 and sr <= 0 are dropped
                idx = (np.broadcast_to(lab, kb.shape)[valid] * K + kb[valid])
                s2[:, bi, :] += np.bincount(idx, minlength = n_c * K).reshape(n_c, K)
            dn_sar = f['ALOS_bands'][i:j][..., alos_idx].astype(np.int64)
            for bi in range(len(ALOS_BANDS)) :
                db = dn_sar[..., bi]
                valid = db > 1 # DN 0 and 1 (= -83 dB) are nodata, as in agbd.py alos_dn_to_db
                idx = np.broadcast_to(lab, db.shape)[valid] * K + db[valid]
                sar[:, bi, :] += np.bincount(idx, minlength = n_c * K).reshape(n_c, K)
            if (i // BATCH) % 10 == 0 : print(f'  {j:>8,d} / {n:,d} rows', flush = True)
    return s2, sar

###################################################################################################
# Stats from histograms

def weighted_mean_std(values, counts) :
    w = counts.astype(np.float64)
    mean = np.sum(w * values) / w.sum()
    std = np.sqrt(np.sum(w * (values - mean) ** 2) / w.sum())
    return mean, std


def percentile(values, counts, q) :
    """Same convention as data/agbd_lite/compute_statistics.py get_percentiles: values[#(cdf < q)]."""
    cdf = np.cumsum(counts) / counts.sum()
    return values[np.sum(cdf < q)]


def stats_from_hist(s2, sar) :
    """s2: (12, K), sar: (2, K) histograms of one set of rows -> the four yaml stats per modality."""
    k = np.arange(K)
    out = {'data_mean': {'optical': [], 'sar': []}, 'data_std': {'optical': [], 'sar': []},
           'data_min': {'optical': [], 'sar': []}, 'data_max': {'optical': [], 'sar': []}, 'raw_sar_percentiles': []}
    for h in s2 :
        nz = np.flatnonzero(h)
        sr = k[nz].astype(np.float64) / 10000
        mean, std = weighted_mean_std(sr, h[nz])
        out['data_mean']['optical'].append(mean); out['data_std']['optical'].append(std)
        out['data_min']['optical'].append(k[nz[0]] / 10000); out['data_max']['optical'].append(k[nz[-1]] / 10000)
    for h in sar :
        nz = np.flatnonzero(h)
        db = 20 * np.log10(k[nz].astype(np.float64)) - 83.0
        p_lo, p_hi = percentile(db, h[nz], P_LO), percentile(db, h[nz], P_HI)
        lo, hi = float(np.floor(p_lo)), float(np.ceil(p_hi))
        mean, std = weighted_mean_std(np.clip(db, lo, hi), h[nz])
        out['data_mean']['sar'].append(mean); out['data_std']['sar'].append(std)
        out['data_min']['sar'].append(lo); out['data_max']['sar'].append(hi)
        out['raw_sar_percentiles'].append((p_lo, p_hi))
    return out


def check_against(new, ref, label) :
    """Compare to a yaml's stats. min/max exact (to float32), mean/std to 1e-4 relative. Returns ok."""
    ok = True
    for stat in ['data_mean', 'data_std', 'data_min', 'data_max'] :
        for mod in ['optical', 'sar'] :
            a, b = np.asarray(new[stat][mod], dtype = np.float64), np.asarray(ref[stat][mod], dtype = np.float64)
            if stat in ['data_min', 'data_max'] : good = np.allclose(np.float32(a), np.float32(b), rtol = 0, atol = 1e-6)
            else : good = np.allclose(a, b, rtol = 1e-4, atol = 0)
            if not good :
                ok = False
                print(f'  MISMATCH {label} {stat}.{mod}:\n    new {np.round(a, 7).tolist()}\n    ref {np.round(b, 7).tolist()}')
    return ok


def fmt(values) :
    return '[' + ', '.join(f'{float(np.float32(v)):.8g}' for v in values) + ']'


def holdout_yaml(region, st, n_rows, md5_mapping) :
    kept = [c for c in CONTINENTS if c != region]
    raw =', '.join(f'{b}: p0.1={p[0]:.2f} p99.9={p[1]:.2f}' for b, p in zip(['VV(HH)', 'VH(HV)'], st['raw_sar_percentiles']))
    return f"""# GENERATED by AGBD-GFMs/benchmark_pangaea/holdout/compute_lite_holdout_stats.py -- do not edit by hand.
# AGBD-Lite with {region} held out: its rows are dropped from train and val, and the test split keeps ONLY its rows.
# Normalisation statistics recomputed on the {n_rows:,} Lite train rows outside {region} ({', '.join(kept)}),
# with the recipe of agbdlite.yaml (reproduced to the digit on all train rows before this file was written).
# Raw SAR percentiles (dB, before rounding outwards): {raw}
# Row -> continent: AGBD-Lite-continents.npz (mapping_lite_to_og.pkl md5 {md5_mapping}).
defaults:
  - agbdlite
  - _self_

hold_out_region: {region}
test_region: {region}

data_mean:
  optical: {fmt(st['data_mean']['optical'])}
  sar: {fmt(st['data_mean']['sar'])}

data_std:
  optical: {fmt(st['data_std']['optical'])}
  sar: {fmt(st['data_std']['sar'])}

data_min:
  optical: {fmt(st['data_min']['optical'])}
  sar: {fmt(st['data_min']['sar'])}

data_max:
  optical: {fmt(st['data_max']['optical'])}
  sar: {fmt(st['data_max']['sar'])}
"""

###################################################################################################
# Code execution

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--lite_dir', default = '/scratch3/gsialelli/patches/AGBD-Lite', help = 'AGBD-Lite-{train,val,test}.h5')
    parser.add_argument('--out_dir', default = '/scratch3/gsialelli/patches/AGBD-Lite', help = 'Where the two .npz files go.')
    parser.add_argument('--pangaea_dir', default = '/scratch3/gsialelli/pangaea-bench', help = 'Where configs/dataset/ lives.')
    parser.add_argument('--holdout', nargs = '+', default = HOLDOUT, choices = CONTINENTS)
    parser.add_argument('--from_cache', action = 'store_true', help = 'Reuse AGBD-Lite-train-histograms.npz instead of reading the h5.')
    args = parser.parse_args()

    paths = get_paths(local = is_local())
    path_mapping = join(paths['aef_norm'], 'mapping_lite_to_og.pkl')
    with open(path_mapping, 'rb') as f : md5_mapping = hashlib.md5(f.read()).hexdigest()
    path_labels = join(args.out_dir, 'AGBD-Lite-continents.npz')
    path_hist = join(args.out_dir, 'AGBD-Lite-train-histograms.npz')

    # 1. Row -> continent, every split -------------------------------------------------------------------------
    print(f'Row -> continent (mapping {path_mapping}, md5 {md5_mapping}; tiles_per_region in {paths["map"]})')
    labels, fingerprints = {}, {}
    for split in SPLITS :
        with h5py.File(join(args.lite_dir, f'AGBD-Lite-{split}.h5'), 'r') as f :
            agbd = f['GEDI']['agbd'][:]
        n_rows = len(agbd)
        # Same hash as pangaea/datasets/agbdlite.py agbd_fingerprint: ties the labels to this exact row order.
        fingerprints[f'agbd_sha1_{split}'] = hashlib.sha1(np.ascontiguousarray(agbd).tobytes()).hexdigest()
        labels[split] = continent_labels(paths, split, n_rows)
        print(f'  {split:5s} ' + '  '.join(f'{c} {np.sum(labels[split] == ci):,}' for ci, c in enumerate(CONTINENTS)))

    # 2. Histograms of the train split -------------------------------------------------------------------------
    if args.from_cache :
        with np.load(path_hist) as z :
            assert [str(c) for c in z['names']] == CONTINENTS and np.array_equal(z['train_labels'], labels['train']), \
                f'{path_hist} was built with other continent labels; rerun without --from_cache'
            s2, sar = z['s2'], z['sar']
        print(f'Histograms from {path_hist}')
    else :
        print('Histogram pass over AGBD-Lite-train.h5')
        s2, sar = histogram_pass(join(args.lite_dir, 'AGBD-Lite-train.h5'), labels['train'])

    # 3. Gate: all train rows must reproduce agbdlite.yaml -------------------------------------------------------
    with open(join(args.pangaea_dir, 'configs', 'dataset', 'agbdlite.yaml')) as f : ref = yaml.safe_load(f)
    glob = stats_from_hist(s2.sum(0), sar.sum(0))
    print('Global SAR percentiles (dB): ' + ', '.join(f'{b} p0.1={p[0]:.3f} p99.9={p[1]:.3f}' for b, p in zip(ALOS_BANDS, glob['raw_sar_percentiles'])))
    if not check_against(glob, ref, 'global vs agbdlite.yaml') :
        sys.exit('The recipe does not reproduce configs/dataset/agbdlite.yaml on all train rows; nothing written.')
    print('Global stats reproduce configs/dataset/agbdlite.yaml.')

    # 4. Write -------------------------------------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok = True)
    tmp = path_labels + '.tmp.npz'
    np.savez(tmp, names = np.array(CONTINENTS), mapping_md5 = md5_mapping, **labels, **fingerprints)
    os.replace(tmp, path_labels)
    print(f'Wrote {path_labels}')
    if not args.from_cache :
        tmp = path_hist + '.tmp.npz'
        np.savez_compressed(tmp, names = np.array(CONTINENTS), train_labels = labels['train'], s2 = s2, sar = sar,
                            s2_bands = np.array(S2_BANDS), alos_bands = np.array(ALOS_BANDS))
        os.replace(tmp, path_hist)
        print(f'Wrote {path_hist}')

    for region in args.holdout :
        ci = CONTINENTS.index(region)
        keep = [i for i in range(len(CONTINENTS)) if i != ci]
        st = stats_from_hist(s2[keep].sum(0), sar[keep].sum(0))
        n_rows = int(np.sum(labels['train'] != ci))
        out = join(args.pangaea_dir, 'configs', 'dataset', f'agbdlite-holdout-{region}.yaml')
        with open(out, 'w') as f : f.write(holdout_yaml(region, st, n_rows, md5_mapping))
        print(f'\n{region} held out ({n_rows:,} train rows) -> {out}')
        for stat in ['data_mean', 'data_std', 'data_min', 'data_max'] :
            for mod in ['optical', 'sar'] :
                a, b = np.asarray(st[stat][mod]), np.asarray(ref[stat][mod], dtype = np.float64)
                rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
                print(f'  {stat:9s} {mod:7s} max rel change vs global {rel.max():7.2%}  (band {int(rel.argmax())})')
