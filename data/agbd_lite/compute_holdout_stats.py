"""

Region-excluded AEF / OlmoEarth normalisation statistics on the AGBD-Lite train set, for the leave-one-region-out
Lite runs (model/runs/nico/*_lite_holdout.sh).

One pass over the Lite train rows: each row is assigned to its (tile-based) continent with the SAME helper the model
uses to drop the held-out region (model/dataset.py: lite_rows_for_region), and per-batch mean/std/count/min/max are
accumulated per continent. The stats for "everything but R" are then the pooled aggregate over the other continents
(same pooled-variance formula as data/aef/compute_statistics.py and data/olmoearth/compute_statistics.py).

Exactly the Lite train rows are used, through mapping_lite_to_og.pkl for AEF (as dataset.py reads AEF for Lite).
NOTE: the existing AEF_statistics_2020_global_lite.pkl was NOT computed on these rows: data/aef/compute_statistics.py
--lite reads mapping_lite_to_og['train'] as (tile, n_patches) while it is (tile, patch_idx) per Lite row, so it took
the first `last idx seen` patches of each of the 281 Lite tiles (~7.4M patches). --check_global reports how far the
exact-rows global stats are from it; OLMO (row-aligned with Lite) must reproduce its global file to float precision.

Output (into --out_dir, default: the AEF / OLMO stats folders from config.sh), named like the full-AGBD complements:
    AEF_statistics_2020_<kept continents>_lite.pkl     e.g. AEF_statistics_2020_Europe-Australasia-Africa-NorthAmerica-SouthAmerica_lite.pkl
    OLMO_statistics_2020_<kept continents>_lite.pkl
which is what model/dataset.py loads for a Lite model with --stats_hold_out_region R (--stats_keep_region false).

Also --write_global: <PREFIX>_statistics_2020_global_lite_exact.pkl, the global stats on exactly the Lite train rows
(used by Lite models trained with --new_AEF_stats). Per-continent partials are saved too, for other combinations.

Run (local; AEF ~1h, OLMO ~1.5h, CPU-bound):
    .venv/bin/python data/agbd_lite/compute_holdout_stats.py --exclude SouthAsia Africa SouthAmerica --check_global --write_global

env: .venv (imports model/dataset.py)

"""

###################################################################################################
# Imports

import argparse
import os
import pickle
import sys
from collections import defaultdict
from os.path import join, exists, dirname, abspath

import h5py
import numpy as np

REPO = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, join(REPO, 'model'))
sys.path.insert(0, join(REPO, 'data', 'olmoearth'))
from config import get_paths, is_local
from dataset import continent_to_region, lite_rows_for_region
from audit import dequantize as olmo_dequantize # int8 -> float32, NaN at nodata (exactly what the OLMO stats used)

CONTINENTS = ["Europe", "SouthAsia", "Australasia", "Africa", "NorthAmerica", "SouthAmerica"] # order of the file names
AEF_NODATA = -128
AEF_DIMS, OLMO_DIMS = 64, 128
BATCH = 2_000

###################################################################################################
# Stats helpers (same estimator as data/aef/compute_statistics.py and data/olmoearth/compute_statistics.py)

def init_stats(num_dims) :
    return {'min': np.full((num_dims,), np.inf), 'max': np.full((num_dims,), -np.inf), 'mean': [], 'std': [], 'num_samples': []}


def update_stats(stats, data) :
    stats['min'] = np.fmin(stats['min'], np.nanmin(data, axis = (0, 1, 2)))
    stats['max'] = np.fmax(stats['max'], np.nanmax(data, axis = (0, 1, 2)))
    stats['num_samples'].append(np.sum(~np.isnan(data), axis = (0, 1, 2)))
    stats['mean'].append(np.nanmean(data, axis = (0, 1, 2)))
    stats['std'].append(np.nanstd(data, axis = (0, 1, 2)))
    return stats


def merge_stats(parts) :
    """Concatenate the per-batch lists of several partial stats (pooling is then done by aggregate_stats)."""
    out = init_stats(len(parts[0]['min']))
    for p in parts :
        out['min'], out['max'] = np.fmin(out['min'], p['min']), np.fmax(out['max'], p['max'])
        for k in ('mean', 'std', 'num_samples') : out[k] += p[k]
    return out


def aggregate_stats(stats) :
    means, stds, n = np.array(stats['mean']), np.array(stats['std']), np.array(stats['num_samples'])
    final_mean = np.sum(means * n, axis = 0) / np.sum(n, axis = 0)
    A = 1 / (np.sum(n, axis = 0) - len(n))
    B = np.sum((n - 1) * stds ** 2 + n * means ** 2, axis = 0)
    C = np.sum(n, axis = 0) * final_mean ** 2
    return {'mean': final_mean, 'std': np.sqrt(A * (B - C)), 'min': stats['min'], 'max': stats['max']}


def kept_str(excluded) :
    return '-'.join(c for c in CONTINENTS if c != excluded)

###################################################################################################
# Row -> continent

def continent_of_rows(paths, lite_to_og) :
    """Label every Lite train row with its continent; assert the continents PARTITION the rows."""
    n = len(lite_to_og)
    label = np.full(n, -1, dtype = np.int8)
    for ci, c in enumerate(CONTINENTS) :
        rows = lite_rows_for_region('train', paths['aef_norm'], paths['map'], c, True, n)
        assert np.all(label[rows] == -1), f'rows assigned to two continents ({c})'
        label[rows] = ci
    assert np.all(label >= 0), f'{np.sum(label < 0)} Lite train rows belong to no continent'
    for ci, c in enumerate(CONTINENTS) : print(f'  {c:13s} {np.sum(label == ci):>8,d} rows')
    return label

###################################################################################################
# Feature passes

def aef_pass(paths, lite_to_og, label) :
    """Per-continent AEF stats over exactly the Lite train rows, read as dataset.py does (2020 file of the tile)."""
    aef_dir = paths['aef_h5']
    fnames = [f for f in os.listdir(aef_dir) if f.endswith('.h5') and '_2020' in f]
    tile_to_file = {}
    for fname in fnames :
        with h5py.File(join(aef_dir, fname), 'r') as f :
            for tile in f.keys() : tile_to_file.setdefault(tile, fname)
    by_tile = defaultdict(list) # tile -> list of (og_idx, continent)
    for row, (tile, idx) in enumerate(lite_to_og) : by_tile[str(tile)].append((int(idx), int(label[row])))
    missing = [t for t in by_tile if t not in tile_to_file]
    assert not missing, f'{len(missing)} Lite tiles have no AEF 2020 file: {missing[:5]}'

    stats = [init_stats(AEF_DIMS) for _ in CONTINENTS]
    handles = {}
    done = 0
    for tile, items in sorted(by_tile.items()) :
        fname = tile_to_file[tile]
        if fname not in handles : handles[fname] = h5py.File(join(aef_dir, fname), 'r')
        ds = handles[fname][tile]
        conts = {c for _, c in items}
        assert len(conts) == 1, f'tile {tile} spans continents {conts}'
        ci = conts.pop()
        idx = np.array([i for i, _ in items])
        uniq, counts = np.unique(idx, return_counts = True)
        assert uniq.max() < ds.shape[0], f'{tile}: Lite idx {uniq.max()} >= {ds.shape[0]} AEF patches'
        for s in range(0, len(uniq), BATCH) :
            u, k = uniq[s : s + BATCH], counts[s : s + BATCH]
            data = ds[u.tolist()].astype(np.float32) # (b, 25, 25, 64), sorted unique indices for h5py
            data = np.repeat(data, k, axis = 0) # keep duplicate Lite rows (if any) with their multiplicity
            data = np.where(data == AEF_NODATA, np.nan, (((data / 127.5) ** 2) * np.sign(data) + 1) / 2)
            stats[ci] = update_stats(stats[ci], data)
        done += len(idx)
        print(f'\r  AEF {done:>8,d} / {len(lite_to_og):,d}', end = '', flush = True)
    print()
    assert done == len(lite_to_og)
    for h in handles.values() : h.close()
    return stats


def olmo_pass(paths, label) :
    """Per-continent OLMO stats; OLMO-Lite-train.h5 is row-aligned with the Lite train set."""
    stats = [init_stats(OLMO_DIMS) for _ in CONTINENTS]
    with h5py.File(join(paths['olmo_h5'], 'OLMO-Lite-train.h5'), 'r') as f :
        assert int(f.attrs['year']) == 2020 and f['done'][:].all()
        emb = f['embeddings']
        assert emb.shape[0] == len(label), f'OLMO-Lite-train has {emb.shape[0]} rows, Lite train has {len(label)}'
        for s in range(0, emb.shape[0], BATCH) :
            data = olmo_dequantize(emb[s : s + BATCH]) # (b, 25, 25, 128)
            lab = label[s : s + BATCH]
            for ci in np.unique(lab) : stats[ci] = update_stats(stats[ci], data[lab == ci])
            print(f'\r  OLMO {min(s + BATCH, emb.shape[0]):>8,d} / {emb.shape[0]:,d}', end = '', flush = True)
    print()
    return stats

###################################################################################################
# Main

def report_diff(name, new, ref) :
    for k in ('mean', 'std', 'min', 'max') :
        a, b = np.asarray(new[k], dtype = np.float64), np.asarray(ref[k], dtype = np.float64)
        print(f'    {name} {k:4s}: max |diff| = {np.max(np.abs(a - b)):.3e}   max rel = {np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)):.3e}')


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', nargs = '*', default = ['SouthAsia', 'Africa', 'SouthAmerica'], choices = CONTINENTS)
    parser.add_argument('--features', nargs = '+', default = ['aef', 'olmo'], choices = ['aef', 'olmo'])
    parser.add_argument('--out_dir', default = None, help = 'Write all files here (default: AEF / OLMO stats folders).')
    parser.add_argument('--check_global', action = 'store_true', help = 'Compare the pooled global stats to the existing *_2020_global_lite.pkl.')
    parser.add_argument('--write_global', action = 'store_true', help = 'Also write the pooled stats over ALL Lite train rows as <PREFIX>_statistics_2020_global_lite_exact.pkl (model flag --new_AEF_stats).')
    parser.add_argument('--overwrite', action = 'store_true')
    args = parser.parse_args()

    paths = get_paths(local = is_local())
    with open(join(paths['aef_norm'], 'mapping_lite_to_og.pkl'), 'rb') as f : lite_to_og = pickle.load(f)['train']
    print(f'Lite train rows: {len(lite_to_og):,d}')
    label = continent_of_rows(paths, lite_to_og)

    for feat in args.features :
        print(f'\n== {feat.upper()} ==')
        per_cont = aef_pass(paths, lite_to_og, label) if feat == 'aef' else olmo_pass(paths, label)
        prefix, stats_dir = ('AEF', paths['aef_norm']) if feat == 'aef' else ('OLMO', paths['olmo_norm'])
        out_dir = args.out_dir or stats_dir
        with open(join(out_dir, f'{prefix}_lite_train_per_continent_partials.pkl'), 'wb') as f : # any other combination later, without re-reading
            pickle.dump({'continents': CONTINENTS, 'partials': per_cont}, f)

        if args.check_global :
            pooled = aggregate_stats(merge_stats(per_cont))
            ref_path = join(stats_dir, f'{prefix}_statistics_2020_global_lite.pkl')
            with open(ref_path, 'rb') as f : ref = pickle.load(f)
            print(f'  pooled (all continents) vs {ref_path}:')
            report_diff(prefix, pooled, ref)

        if args.write_global :
            final = aggregate_stats(merge_stats(per_cont))
            out = join(out_dir, f'{prefix}_statistics_2020_global_lite_exact.pkl')
            if exists(out) and not args.overwrite : raise FileExistsError(f'{out} exists (pass --overwrite)')
            with open(out + '.tmp', 'wb') as f : pickle.dump(final, f)
            os.replace(out + '.tmp', out)
            print(f'  wrote {out}  ({len(label):,d} rows)')

        for excl in args.exclude :
            kept = [per_cont[i] for i, c in enumerate(CONTINENTS) if c != excl]
            final = aggregate_stats(merge_stats(kept))
            assert all(np.all(np.isfinite(final[k])) for k in final), f'non-finite stats excluding {excl}'
            out = join(out_dir, f'{prefix}_statistics_2020_{kept_str(excl)}_lite.pkl')
            if exists(out) and not args.overwrite : raise FileExistsError(f'{out} exists (pass --overwrite)')
            tmp = out + '.tmp'
            with open(tmp, 'wb') as f : pickle.dump(final, f)
            os.replace(tmp, out)
            n_rows = int(np.sum(label != CONTINENTS.index(excl)))
            print(f'  wrote {out}  ({n_rows:,d} rows)')
