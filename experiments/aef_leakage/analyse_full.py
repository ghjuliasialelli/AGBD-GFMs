"""
Were we right to exclude test samples inside AEF pre-training patches? Paired AEF-vs-AGBD comparison
on the FULL test set (3,303,722 rows, incl. the 495,745 excluded ones), binned by distance to the
nearest AEF pre-training location, split into excluded / kept.

Runs (3-seed ensembles, the only pairs with full-test predictions; NOT the paper's single runs):
  full-trained: AGBD features 17997535 vs AEF 56026335 (no-overlap RMSE 52.70 / 50.70)
  Lite-trained: AGBD features 55140725 vs AEF 56549587
17997535 is stored in year-outer file order; it is mapped to the common order by row key.
Run: python -u experiments/aef_leakage/analyse_full.py
"""
import numpy as np, h5py, pandas as pd
from scipy.spatial import cKDTree

D = 'experiments/aef_leakage/data/'
R = '/scratch3/gsialelli/AGBD-GFM/agbd-lite/eval/results/nico_film_{0}-1_{0}-2_{0}-3_2019-2020.h5'
AEF_CSV = '/scratch3/gsialelli/AGBD-GFM/aef-dwn/distribution/training_coordinates.csv'
PAIRS = {'full-trained': ('17997535', '56026335'), 'Lite-trained': ('55140725', '56549587')}
ORDER = {'17997535': 'year_outer'}
EDGES = [0, 0.32, 0.64, 0.905, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0]
B, RE = 1000, 6371008.8
rng = np.random.default_rng(0)

c = dict(np.load(D + 'test_coordinates_full_i_outer.npz'))
cy = np.load(D + 'test_coordinates_full_year_outer.npz')
y = c['agbd'].astype('float64'); exc = c['excluded']; tile = c['tile_id']
# permutation year_outer -> i_outer by row key
perm_src = np.argsort(cy['key']); perm_dst = np.argsort(c['key'])
assert np.array_equal(cy['key'][perm_src], c['key'][perm_dst])
to_i = np.empty(len(y), np.int64); to_i[perm_dst] = perm_src     # i_outer row j <- year_outer row to_i[j]

def preds(rid):
    with h5py.File(R.format(rid)) as h: p, L = h['predictions'][:].astype('float64'), h['labels'][:]
    if ORDER.get(rid) == 'year_outer': p, L = p[to_i], L[to_i]
    assert np.array_equal(L, c['agbd']), rid
    return p

# distances
import pandas as pd
aef = pd.read_csv(AEF_CSV, usecols = ['longitude', 'latitude']).drop_duplicates()
def unit(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
ch, _ = cKDTree(unit(aef.latitude.values, aef.longitude.values)).query(unit(c['lat'], c['lon']), workers = 8)
d = 2 * RE * np.arcsin(np.clip(ch / 2, 0, 1)) / 1000
np.save(D + 'dist_to_aef_full_i_outer.npy', d)
print(f'excluded: n={exc.sum():,}, dist km max {d[exc].max():.3f} (square half-diagonal 0.905); '
      f'kept with d<0.64: {np.sum(~exc & (d < 0.64)):,}')

rows = []
for pname, (ra, rb) in PAIRS.items():
    pb, pa = preds(ra), preds(rb)            # ra = AGBD, rb = AEF
    se_b, se_a = (pb - y) ** 2, (pa - y) ** 2
    for m, lab in [(~exc, 'kept (paper)'), (exc, 'excluded'), (np.ones_like(exc), 'all')]:
        print(f'{pname:12s} {lab:13s} n={m.sum():9,}  RMSE AEF {np.sqrt(se_a[m].mean()):6.2f}  AGBD {np.sqrt(se_b[m].mean()):6.2f}  '
              f'rel adv {100*(1-np.sqrt(se_a[m].mean()/se_b[m].mean())):5.2f}%  mean AGBD {y[m].mean():6.1f}')
    b = np.digitize(d, EDGES) - 1
    for status, sm in [('excluded', exc), ('kept', ~exc)]:
        for k in range(len(EDGES) - 1):
            s = sm & (b == k); n = int(s.sum())
            if n < 500: continue
            ut, inv = np.unique(tile[s], return_inverse = True)
            A = np.bincount(inv, se_a[s]); Bb = np.bincount(inv, se_b[s]); N = np.bincount(inv)
            idx = rng.integers(0, len(ut), (B, len(ut)))
            r = np.sqrt(A[idx].sum(1) / Bb[idx].sum(1))
            rows.append(dict(pair = pname, status = status, bin_km = f'{EDGES[k]}-{EDGES[k+1]}', n = n, n_tiles = len(ut),
                             mean_agbd = y[s].mean(), rmse_AEF = np.sqrt(A.sum() / N.sum()), rmse_AGBD = np.sqrt(Bb.sum() / N.sum()),
                             rel_adv = 100 * (1 - np.sqrt(A.sum() / Bb.sum())),
                             rel_lo = 100 * (1 - np.percentile(r, 97.5)), rel_hi = 100 * (1 - np.percentile(r, 2.5))))
    print()
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string(index = False))
df.to_csv(D + 'binned_full.csv', index = False)
