"""
Standalone recompute of the AGBref comparison metrics for the three manuscript configs
(All / Subset-1500km / Japan-inclusive), from the CACHED per-plot npz (no raster extraction).

Faithfully mirrors comparison.py::apply_filters (notrain + region + max500; MIN_COVERAGE is None so
the coverage filter is OFF) and print_stats. Validates the All/666 config against the known-good CSV
metrics_notrain_skip_jap_mean_max500.csv before trusting the derived Subset/Japan numbers.

Read-only w.r.t. comparison.py and its outputs; writes nothing except stdout.
"""
import numpy as np, geopandas as gpd, pickle, pandas as pd
from shapely.validation import make_valid

CACHE = "results/comparison_results.npz"
SPLITS_PKL = "/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split/biomes_splits_to_name.pkl"
S2_TILES_SHP = "/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
REGIONS_PATH = "/scratch3/gsialelli/BiomassDatasetCreation/Data/countrySelection/AOIs.geojson"
DEFAULT_AGBD_REGIONS = ["California","Cuba","FrenchGuiana","Paraguay","Austria",
                        "UnitedRepublicofTanzania","Ghana","Nepal","ShaanxiProvince","Greece"]
MAX_VALUE = 500

res = dict(np.load(CACHE, allow_pickle=True))
res.pop("_geom_version", None)
n = len(res["point_x"])
centers = gpd.GeoSeries(gpd.points_from_xy(res["point_x"], res["point_y"]), crs="EPSG:4326")
regions = gpd.read_file(REGIONS_PATH)

# training-tile union (notrain)
with open(SPLITS_PKL, "rb") as f:
    train_names = [str(t) for t in pickle.load(f)["train"]]
s2 = gpd.read_file(S2_TILES_SHP, engine="pyogrio").drop_duplicates(subset=["Name"])
train_union = s2[s2["Name"].isin(train_names)].unary_union
in_train = centers.intersects(train_union).values

def region_union_buffered(names, buffer_m):
    sel = regions[regions["name"].isin(names)]
    if buffer_m and buffer_m > 0:
        sel_m = sel.to_crs("+proj=cea +datum=WGS84")
        sel_m["geometry"] = sel_m.geometry.buffer(buffer_m)
        sel = sel_m.to_crs("EPSG:4326")
        sel["geometry"] = sel.geometry.apply(make_valid)
    return make_valid(sel.unary_union)

def build_keep(drop_train=True, agbd_regions=None, agbd_buffer_m=1_500_000, skip_regions=None):
    keep = np.ones(n, dtype=bool)
    if drop_train:
        keep &= ~in_train
    if agbd_regions is not None:
        ru = region_union_buffered(agbd_regions, agbd_buffer_m)
        keep &= centers.intersects(ru).values
    if skip_regions is not None:
        su = regions[regions["name"].isin(skip_regions)].unary_union
        keep &= ~centers.intersects(su).values
    over_max = res["agbref"] > MAX_VALUE
    keep &= ~over_max
    return keep

def stats(name, pred, truth):
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    p, t = pred[valid], truth[valid]
    bias = np.mean(p - t); rmse = np.sqrt(np.mean((p - t) ** 2)); mae = np.mean(np.abs(p - t))
    r = np.corrcoef(p, t)[0, 1]
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
    print(f"  {name}: n={len(p)}  R2={r2:.4f}  r={r:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ME={bias:.4f}")
    return dict(n=len(p), r2=r2, r=r, rmse=rmse, mae=mae, me=bias)

def report(title, keep):
    print(f"\n=== {title}  (kept {keep.sum()}/{n}) ===")
    s = {}
    for src in ("nico_mean", "cci_mean"):
        s[src] = stats(src, res[src][keep], res["agbref"][keep])
    return s

# 1) ALL (validate vs metrics_notrain_skip_jap_mean_max500.csv: nico 44.79/28.31/-1.18/0.752/0.563)
keep_all = build_keep(drop_train=True, agbd_regions=None, skip_regions=["Japan"])
report("ALL (notrain, skip Japan, max500)", keep_all)

# 2) SUBSET (AGBD regions + 1500 km buffer)
keep_sub = build_keep(drop_train=True, agbd_regions=DEFAULT_AGBD_REGIONS,
                      agbd_buffer_m=1_500_000, skip_regions=["Japan"])
report("SUBSET (notrain, AGBD+1500km, skip Japan, max500)", keep_sub)

# 3) JAPAN-INCLUSIVE: All incl Japan (753), Japan only (87), All-but-Japan (666, CCI check)
keep_incl = build_keep(drop_train=True, agbd_regions=None, skip_regions=None)
jp_union = regions[regions["name"].isin(["Japan"])].unary_union
in_jp = centers.intersects(jp_union).values
report("ALL incl. Japan (notrain, max500)", keep_incl)
report("JAPAN only", keep_incl & in_jp)
report("ALL but Japan (== config 1)", keep_incl & ~in_jp)
