"""
Leakage test requested by the reviewer: does accuracy depend on the test sample's distance to the
nearest AEF pre-training location?

Paired comparison on identical rows (verified label match): AEF (nico_film 59620113-1) vs AGBD
features (nico_film 59620098-1). Distance and geography/biomass composition are confounded, so the
leakage signal is the AEF ADVANTAGE vs distance (RMSE_AGBD - RMSE_AEF, and RMSE_AEF/RMSE_AGBD), not
the raw AEF error. CIs: tile-cluster bootstrap (samples within an S2 tile are not independent).

Run: python -u experiments/aef_leakage/analyse.py
"""
import numpy as np, h5py, pandas as pd

RES = 'model/eval/results/nico_film_{0}_{0}_{0}_2019-2020_nooverlap.h5'
MODELS = {'AEF': '59620113-1', 'AGBD': '59620098-1'}
EDGES_KM = [0, 0.64, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0]
B = 1000
rng = np.random.default_rng(0)

c = np.load('experiments/spatial_autocorrelation/data/test_coordinates_nooverlap.npz')
d = np.load('experiments/aef_leakage/data/dist_to_aef_nooverlap.npz')['dist_m'] / 1000
y = c['agbd'].astype('float64'); tile = c['tile_id']; reg = c['region_cla']
err = {}
for m, rid in MODELS.items():
    with h5py.File(RES.format(rid)) as h:
        assert np.array_equal(h['labels'][:], c['agbd']), f'{m}: label order mismatch'
        err[m] = h['predictions'][:].reshape(-1).astype('float64') - y
assert len(d) == len(y)
se_a, se_b = err['AEF'] ** 2, err['AGBD'] ** 2
print(f"overall n={len(y):,}  RMSE AEF {np.sqrt(se_a.mean()):.2f}  AGBD {np.sqrt(se_b.mean()):.2f}\n")

def binstats(mask, label):
    """Per-bin RMSEs with tile-cluster bootstrap CI on the gap and ratio."""
    rows = []
    b = np.digitize(d, EDGES_KM) - 1
    for k in range(len(EDGES_KM) - 1):
        s = mask & (b == k)
        n = int(s.sum())
        if n < 2000: continue
        t = tile[s]; ut, inv = np.unique(t, return_inverse = True)
        A = np.bincount(inv, se_a[s]); Bb = np.bincount(inv, se_b[s]); N = np.bincount(inv)
        ra, rb = np.sqrt(A.sum() / N.sum()), np.sqrt(Bb.sum() / N.sum())
        idx = rng.integers(0, len(ut), (B, len(ut)))
        bn = N[idx].sum(1); ba = np.sqrt(A[idx].sum(1) / bn); bb = np.sqrt(Bb[idx].sum(1) / bn)
        g, r = bb - ba, ba / bb
        rows.append(dict(group = label, bin_km = f'{EDGES_KM[k]}-{EDGES_KM[k+1]}', n = n, n_tiles = len(ut),
                         mean_agbd = y[s].mean(), rmse_AEF = ra, rmse_AGBD = rb, gap = rb - ra,
                         gap_lo = np.percentile(g, 2.5), gap_hi = np.percentile(g, 97.5),
                         ratio = ra / rb, ratio_lo = np.percentile(r, 2.5), ratio_hi = np.percentile(r, 97.5)))
    return rows

rows = binstats(np.ones(len(y), bool), 'all')
names = {int(r): None for r in np.unique(reg)}
for r in np.unique(reg): rows += binstats(reg == r, f'region_cla={r}')
# biomass strata (composition control)
for lo, hi in [(0, 50), (50, 150), (150, 300), (300, 1e9)]:
    rows += binstats((y >= lo) & (y < hi), f'agbd {lo}-{hi if hi < 1e9 else "inf"}')
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(df.round(3).to_string(index = False))
df.to_csv('experiments/aef_leakage/data/binned_gap.csv', index = False)

# within-region slope of the per-sample squared-error advantage vs log10 distance,
# with region x biomass-bin fixed effects (demeaning), tile-cluster bootstrap
dd = se_b - se_a  # >0 where AEF better
x = np.log10(np.maximum(d, 0.05))
grp = reg.astype(np.int64) * 10 + np.digitize(y, [50, 150, 300])
def demean(v):
    u, inv = np.unique(grp, return_inverse = True)
    return v - (np.bincount(inv, v) / np.bincount(inv))[inv]
xd, yd = demean(x), demean(dd)
ut, inv = np.unique(tile, return_inverse = True)
Sxy = np.bincount(inv, xd * yd); Sxx = np.bincount(inv, xd * xd)
slope = Sxy.sum() / Sxx.sum()
idx = rng.integers(0, len(ut), (B, len(ut)))
bs = Sxy[idx].sum(1) / Sxx[idx].sum(1)
print(f"\nFE slope of (SE_AGBD - SE_AEF) on log10(dist km): {slope:.1f} Mg^2/ha^2 per decade "
      f"[95% tile-bootstrap {np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}]; "
      f"mean advantage {dd.mean():.1f}. Leakage predicts a NEGATIVE slope.")
