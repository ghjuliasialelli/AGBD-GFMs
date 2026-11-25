"""
Coordinates for the FULL AGBD test set (overlaps NOT dropped, 3,303,722 rows), plus a per-row flag
`excluded` = row is listed in AEF_overlaps.pkl (i.e. dropped from the paper's test set).
Same replay as experiments/spatial_autocorrelation/test_coordinates.py; the order must match the
results files' labels EXACTLY or the script exits non-zero. Also tries the year-outer file order for
any file that fails the default (i outer, year inner) order.

Run: python -u experiments/aef_leakage/full_coordinates.py RESULTS.h5 [RESULTS.h5 ...]
"""
import pickle, sys
from os.path import join, dirname, abspath
import h5py, numpy as np
sys.path.insert(0, join(dirname(abspath(__file__)), '..', '..'))
sys.path.insert(0, join(dirname(abspath(__file__)), '..', 'spatial_autocorrelation'))
from config import get_paths
from test_coordinates import load_tile, decode, FIELDS

OUT = 'experiments/aef_leakage/data/test_coordinates_full.npz'
p = get_paths(local = True); h5_path, map_path = p['h5'], p['map']
cache = join(h5_path, 'gedi_scalar_cache')
with open(join(map_path, 'biomes_splits_to_name.pkl'), 'rb') as f: test_tiles = pickle.load(f)['test']
with open(join(map_path, 'AEF_overlaps.pkl'), 'rb') as f: overlap = pickle.load(f)

def replay(fnames):
    cols = {k: [] for k in FIELDS}; exc, tid, tiles, key = [], [], [], []
    for fname in fnames:
        with h5py.File(join(h5_path, fname), 'r') as f: all_tiles = list(f.keys())
        for tile in np.intersect1d(all_tiles, test_tiles):
            arrs, _ = load_tile(cache, h5_path, fname, tile)
            n = len(arrs['agbd'])
            e = np.zeros(n, bool); e[np.asarray(overlap.get(fname, {}).get(tile, []), dtype = int)] = True
            for k in FIELDS: cols[k].append(arrs[k])
            exc.append(e)
            if tile not in tiles: tiles.append(str(tile))
            tid.append(np.full(n, tiles.index(str(tile)), np.int32))
            # order-independent row key: (canonical file index, tile index in the sorted test tiles, row)
            assert n < 10**6
            key.append(np.int64(CANON.index(fname)) * 10**9 + np.int64(TT.index(tile)) * 10**6 + np.arange(n))
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    return dict(lat = decode(cols['lat_decimal'], cols['lat_offset']), lon = decode(cols['lon_decimal'], cols['lon_offset']),
                agbd = cols['agbd'], region_cla = cols['region_cla'], excluded = np.concatenate(exc),
                tile_id = np.concatenate(tid), tiles = np.array(tiles), key = np.concatenate(key))

CANON = sorted(f'data_subset-{y}-v4_{i}-20.h5' for i in range(20) for y in (2019, 2020))
TT = sorted(test_tiles)
orders = {'i_outer': [f'data_subset-{y}-v4_{i}-20.h5' for i in range(20) for y in (2019, 2020)],
          'year_outer': [f'data_subset-{y}-v4_{i}-20.h5' for y in (2019, 2020) for i in range(20)]}
R = {k: replay(v) for k, v in orders.items()}
for k, r in R.items():
    print(f"{k}: n={len(r['agbd']):,}, excluded={r['excluded'].sum():,}, kept={(~r['excluded']).sum():,}")
match = {}
for res in sys.argv[1:]:
    with h5py.File(res) as h: L = h['labels'][:]
    hits = [k for k, r in R.items() if len(L) == len(r['agbd']) and np.array_equal(L, r['agbd'])]
    print(('OK   ' if hits else 'FAIL ') + res.split('/')[-1], hits)
    match[res] = hits
assert all(match.values()), 'some results file matches no replayed order'
used = sorted({h[0] for h in match.values()})
for k in used:
    out = OUT.replace('.npz', f'_{k}.npz'); np.savez(out, **R[k]); print('wrote', out)
