"""
Distance from every full-AGBD-test sample (no-overlap set, eval order) to the nearest AEF
pre-training sample location.

AEF locations: aef-dwn/distribution/training_coordinates.csv (8.4M rows, one per training example;
the same location recurs across time windows, so coordinates are de-duplicated). The paper's
exclusion (data/aef/check_test_overlap.py) dropped test samples inside a 1.28 km x 1.28 km square
(UTM, +-640 m) around each AEF location, so every remaining sample should be >= 640 m away, except
where the AEF point lies outside the test sample's S2 tile (the script only looked within the tile).

Nearest neighbour on the unit sphere (cKDTree on 3D unit vectors), chord -> great-circle metres.

Output npz: dist_m (float64, per test row, eval order), n_aef_unique, and the 1st/2nd nearest index
not stored (not needed).

Run: python -u experiments/aef_leakage/distances.py
"""
import sys, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

AEF_CSV = '/scratch3/gsialelli/AGBD-GFM/aef-dwn/distribution/training_coordinates.csv'
COORDS = 'experiments/spatial_autocorrelation/data/test_coordinates_nooverlap.npz'
OUT = 'experiments/aef_leakage/data/dist_to_aef_nooverlap.npz'
R = 6371008.8

def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

t = time.time()
aef = pd.read_csv(AEF_CSV, usecols = ['longitude', 'latitude'], dtype = 'float64')
n_rows = len(aef)
assert aef.notna().all().all(), 'NaN coordinates in AEF csv'
assert aef.latitude.between(-90, 90).all() and aef.longitude.between(-180, 180).all()
aef = aef.drop_duplicates()
print(f'AEF rows {n_rows:,}, unique locations {len(aef):,} ({time.time()-t:.0f}s)', flush = True)

c = np.load(COORDS)
lat, lon = c['lat'].astype('float64'), c['lon'].astype('float64')
assert len(lat) == 2807977 and np.isfinite(lat).all() and np.isfinite(lon).all()

tree = cKDTree(unit(aef.latitude.values, aef.longitude.values))
chord, _ = tree.query(unit(lat, lon), k = 1, workers = 8)
dist = 2 * R * np.arcsin(np.clip(chord / 2, 0, 1))
print(f'query done ({time.time()-t:.0f}s)', flush = True)
assert np.isfinite(dist).all()

q = np.percentile(dist, [0, 1, 5, 25, 50, 75, 95, 99, 100]) / 1000
print('distance percentiles (km) 0/1/5/25/50/75/95/99/100:', np.round(q, 2))
print(f'n < 640 m: {np.sum(dist < 640):,}   n < 905 m: {np.sum(dist < 905):,}')
np.savez(OUT, dist_m = dist, n_aef_unique = len(aef), n_aef_rows = n_rows)
print('wrote', OUT)
