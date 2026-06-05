"""
Compare AGB predictions from nico_film and CCI against AGBRef ground truth.

AGBRef: single AGB value per polygon (t/ha)
nico_film: 10m resolution rasters (EPSG varies by tile)
CCI: ~100m resolution rasters (EPSG:4326)

For each of the 780 AGBRef polygons, we:
1. Build a proper square buffer in the plot's local UTM zone
2. Mask the nico_film and CCI rasters by that geometry
3. Aggregate pixel values to a single value (mean, median)
4. Compare against the AGBRef ground truth

Optional filters:
- Exclude plots that overlap with training-set Sentinel-2 tiles
- Restrict to plots within specified geographic regions
"""

PLOT = True # Set to True to enable plotting (scatter + binned)
MAX_VALUE = 500
METHODS = ["mean"] # ["mean", "median", "p90"]

# AGBRef plots are 10km x 10km squares; half-side = 5000m
_BUFFER_M = 5000

# --- Training-set exclusion (set to False to disable) ---
# When True, automatically loads training tile names from SPLITS_PKL and excludes
# AGBRef plots whose center falls inside a training-set Sentinel-2 tile.
# Alternatively, pass an explicit list of tile names, e.g. ["32TQM", "33UUP"].
DROP_TRAIN_TILES = True
SPLITS_PKL = "/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split/biomes_splits_to_name.pkl"
# Path to the .shp file containing Sentinel-2 tile geometries (must have a "Name" column)
S2_TILES_SHP = "/scratch3/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"

# --- Region filtering (set both to None to disable) ---
REGIONS_PATH = "/scratch3/gsialelli/BiomassDatasetCreation/Data/countrySelection/AOIs.geojson"
# AGBD training regions (buffered). Set to None to disable, True to use all
# default AGBD regions (minus NewZealand), or an explicit list.
AGBD_REGIONS = False
AGBD_REGIONS_BUFFER_M = 1_500_000  # buffer around AGBD regions (meters), 750_000
_DEFAULT_AGBD_REGIONS = ["California", "Cuba", "FrenchGuiana", "Paraguay", "Austria", "UnitedRepublicofTanzania", "Ghana", "Nepal", "ShaanxiProvince", "Greece"]
# Extra regions to include (exact boundaries, no buffer). Set to None to disable.
EXTRA_REGIONS = None # None  # e.g. ["Gabon", "Cameroon"]

# Regions to exclude (even if included by the above filters). Set to None to disable.
SKIP_REGIONS = ["Japan"] # ["Japan"] # List of region names to exclude (even if included by the above filters)

# --- Year filtering (set to None to disable) ---
# List of years to include, e.g. [2019, 2020]. Plots with AVG_YEAR not in this list are dropped.
FILTER_YEARS = None

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping, box
from pyproj import CRS
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import tempfile
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

# === Paths ===
BASE_DIR = Path(__file__).parent
AGBREF_PATH = BASE_DIR / "data" / "AGBRef.geojson"
NICO_DIR = Path("/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions/nico_film/59620113-1_59620113-1_59620113-1")
CCI_DIR = Path("/scratch3/gsialelli/CCI")
CACHE_PATH = BASE_DIR / "results" / "comparison_results.npz"

# === Output naming (auto-built from active flags) ===
def _build_suffix():
    parts = []
    if DROP_TRAIN_TILES:
        parts.append("notrain")
    if AGBD_REGIONS:
        parts.append(f"agbd{AGBD_REGIONS_BUFFER_M // 1000}km")
    if EXTRA_REGIONS is not None:
        parts.append("extra_" + "_".join(r[:3].lower() for r in EXTRA_REGIONS))
    if SKIP_REGIONS is not None:
        parts.append("skip_" + "_".join(r[:3].lower() for r in SKIP_REGIONS))
    if FILTER_YEARS is not None:
        parts.append("yr_" + "_".join(str(y) for y in FILTER_YEARS))
    parts.append("_".join(METHODS))
    parts.append(f"max{MAX_VALUE}")
    return "_".join(parts) if parts else "all"

OUTPUT_SUFFIX = _build_suffix()
PLOT_DIR = BASE_DIR / "plots"


# =========================================================================
# Helper: UTM zone from lat/lon
# =========================================================================
def utm_epsg_from_lonlat(lon, lat):
    """Return the EPSG code of the UTM zone for a given lon/lat (WGS84)."""
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number  # North
    else:
        return 32700 + zone_number  # South


# =========================================================================
# Helper: build a proper square buffer in the local UTM zone
# =========================================================================
def make_utm_square(lon, lat, half_size_m=_BUFFER_M):
    """
    Create a square polygon of side 2*half_size_m centered on (lon, lat),
    constructed in the local UTM zone and returned in EPSG:4326.

    Also returns the UTM EPSG code used.
    """
    utm_epsg = utm_epsg_from_lonlat(lon, lat)
    pt = gpd.GeoSeries(
        [gpd.points_from_xy([lon], [lat])[0]], crs="EPSG:4326"
    ).to_crs(epsg=utm_epsg).iloc[0]
    square = box(
        pt.x - half_size_m, pt.y - half_size_m,
        pt.x + half_size_m, pt.y + half_size_m,
    )
    # Convert back to EPSG:4326
    square_4326 = gpd.GeoSeries([square], crs=f"EPSG:{utm_epsg}").to_crs("EPSG:4326").iloc[0]
    return square_4326, utm_epsg


# =========================================================================
# Helper: reproject a raster to a target CRS (in memory)
# =========================================================================
def _reproject_raster_to_crs(src, target_crs):
    """Reproject an open rasterio dataset to target_crs. Returns (data, transform, crs, nodata)."""
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )
    dst_data = np.empty((src.count, height, width), dtype=src.dtypes[0])
    reproject(
        source=src.read(),
        destination=dst_data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return dst_data, transform, target_crs, width, height


# =========================================================================
# Core: extract raster stats in the plot's local UTM CRS
# =========================================================================
def extract_raster_stats(raster_path, geom_4326, nodata, utm_epsg):
    """
    Mask a raster by a polygon geometry and return mean/median/p90 of valid pixels.
    Both the geometry and raster are reprojected to the plot's local UTM CRS
    so that pixel areas are correct and the buffer is a true square.

    Parameters
    ----------
    raster_path : Path to the raster file
    geom_4326   : Shapely geometry in EPSG:4326
    nodata      : nodata value for the raster
    utm_epsg    : EPSG code of the local UTM zone for this plot

    Returns
    -------
    mean, median, p90 of valid pixels, or (nan, nan, nan) if no valid pixels
    """
    target_crs = CRS.from_epsg(utm_epsg)

    with rasterio.open(raster_path) as src:
        # Reproject geometry to the local UTM CRS
        geom_utm = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(epsg=utm_epsg).iloc[0]

        if CRS(src.crs) == target_crs:
            # Raster already in the right CRS — mask directly
            try:
                out_image, _ = rio_mask(src, [mapping(geom_utm)], crop=True, all_touched=True)
            except ValueError:
                return np.nan, np.nan, np.nan
        else:
            # Reproject raster to local UTM, then mask
            dst_data, dst_transform, dst_crs, dst_w, dst_h = _reproject_raster_to_crs(src, target_crs)
            # Write to a temporary in-memory raster for masking
            profile = src.profile.copy()
            profile.update(crs=dst_crs, transform=dst_transform, width=dst_w, height=dst_h)
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.close()
            with rasterio.open(tmp.name, "w", **profile) as dst:
                dst.write(dst_data)
            try:
                with rasterio.open(tmp.name) as dst:
                    out_image, _ = rio_mask(dst, [mapping(geom_utm)], crop=True, all_touched=True)
            except ValueError:
                Path(tmp.name).unlink(missing_ok=True)
                return np.nan, np.nan, np.nan
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    data = out_image[0].astype(np.float32)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    return float(np.mean(valid)), float(np.median(valid)), float(np.percentile(valid, 90))


# =========================================================================
# Main extraction loop
# =========================================================================
def compute_results():
    """
    Run the main extraction loop for ALL plots and return the results dict.
    No filtering is applied here — filters are applied post-hoc so that cached
    results can be reused across different filter settings.
    """
    gdf = gpd.read_file(AGBREF_PATH)
    n = len(gdf)
    print(f"Loaded {n} AGBRef polygons")

    # Rebuild square buffers in local UTM zones
    print("Rebuilding plot geometries in local UTM zones...")
    new_geoms = []
    utm_codes = []
    for _, row in gdf.iterrows():
        sq, epsg = make_utm_square(row["POINT_X"], row["POINT_Y"], _BUFFER_M)
        new_geoms.append(sq)
        utm_codes.append(epsg)
    gdf["geometry"] = new_geoms
    gdf["utm_epsg"] = utm_codes

    # --- Extract raster stats for every plot ---
    print(f"Evaluating {n} plots")

    results = {
        # Metadata (for post-hoc filtering)
        "point_x": gdf["POINT_X"].values.astype(np.float64),
        "point_y": gdf["POINT_Y"].values.astype(np.float64),
        "avg_year": gdf["AVG_YEAR"].values.astype(np.float64),
        # Values
        "agbref": np.full(n, np.nan),
        "nico_mean": np.full(n, np.nan),
        "nico_median": np.full(n, np.nan),
        "nico_p90": np.full(n, np.nan),
        "cci_mean": np.full(n, np.nan),
        "cci_median": np.full(n, np.nan),
        "cci_p90": np.full(n, np.nan),
    }

    for i in range(n):
        row = gdf.iloc[i]
        geom = row.geometry
        utm_epsg = row["utm_epsg"]
        results["agbref"][i] = row["AGB_T_HA"]

        # --- nico_film (file index matches row order in the GeoJSON) ---
        nico_path = NICO_DIR / f"{i}.tif"
        if nico_path.exists():
            mean, median, p90 = extract_raster_stats(nico_path, geom, nodata=-9999.0, utm_epsg=utm_epsg)
            results["nico_mean"][i] = mean
            results["nico_median"][i] = median
            results["nico_p90"][i] = p90
        else:
            print(f"  WARNING: nico_film {i}.tif not found")

        # --- CCI ---
        year = max(row["AVG_YEAR"], 2018)
        year_2digit = int(year) - 2000
        cci_path = CCI_DIR / f"CCI_AGBRef_{i}_{year_2digit}.tif"
        if cci_path.exists():
            mean, median, p90 = extract_raster_stats(cci_path, geom, nodata=65535, utm_epsg=utm_epsg)
            results["cci_mean"][i] = mean
            results["cci_median"][i] = median
            results["cci_p90"][i] = p90
        else:
            print(f"  WARNING: CCI {cci_path.name} not found")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n}")

    print(f"Done processing {n} polygons.")
    return results


def apply_filters(results):
    """
    Apply post-hoc filters (training tiles, regions, years, max value) to
    a full results dict. Returns a filtered copy.
    """
    n = len(results["agbref"])
    keep = np.ones(n, dtype=bool)

    # --- Training-tile exclusion ---
    if DROP_TRAIN_TILES:
        if isinstance(DROP_TRAIN_TILES, list):
            train_tile_names = DROP_TRAIN_TILES
        else:
            with open(SPLITS_PKL, "rb") as f:
                train_tile_names = [str(t) for t in pickle.load(f)["train"]]
        s2_tiles = gpd.read_file(S2_TILES_SHP, engine="pyogrio").drop_duplicates(subset=["Name"])
        train_tiles = s2_tiles[s2_tiles["Name"].isin(train_tile_names)]
        if not train_tiles.empty:
            train_union = train_tiles.unary_union
            centers = gpd.GeoSeries(
                gpd.points_from_xy(results["point_x"], results["point_y"]), crs="EPSG:4326"
            )
            in_train = centers.intersects(train_union)
            n_dropped = in_train.sum()
            keep &= ~in_train.values
            print(f"  Dropped {n_dropped} plots overlapping training-set S2 tiles")

    # --- Region filtering (AGBD regions with buffer + extra regions) ---
    if AGBD_REGIONS or EXTRA_REGIONS is not None or SKIP_REGIONS is not None:
        from shapely.validation import make_valid
        import pandas as pd
        regions = gpd.read_file(REGIONS_PATH)
        parts = []

        # AGBD training regions with buffer
        if AGBD_REGIONS:
            agbd_names = AGBD_REGIONS if isinstance(AGBD_REGIONS, list) else _DEFAULT_AGBD_REGIONS
            sel = regions[regions["name"].isin(agbd_names)]
            if not sel.empty and AGBD_REGIONS_BUFFER_M > 0:
                sel_m = sel.to_crs("+proj=cea +datum=WGS84")
                sel_m["geometry"] = sel_m.geometry.buffer(AGBD_REGIONS_BUFFER_M)
                sel = sel_m.to_crs("EPSG:4326")
                sel["geometry"] = sel.geometry.apply(make_valid)
            parts.append(sel)
            print(f"  AGBD regions: {list(agbd_names)} + {AGBD_REGIONS_BUFFER_M/1000:.0f}km buffer")

        # Extra explicit regions (exact boundaries)
        if EXTRA_REGIONS is not None:
            extra = regions[regions["name"].isin(EXTRA_REGIONS)]
            if not extra.empty:
                parts.append(extra)
                print(f"  Extra regions: {EXTRA_REGIONS}")
            else:
                print(f"  WARNING: none of {EXTRA_REGIONS} found in regions file")

        if SKIP_REGIONS is not None:
            skip = regions[regions["name"].isin(SKIP_REGIONS)]
            if not skip.empty:
                skip_union = skip.unary_union
                centers = gpd.GeoSeries(
                    gpd.points_from_xy(results["point_x"], results["point_y"]), crs="EPSG:4326"
                )
                in_skip = centers.intersects(skip_union)
                n_skipped = in_skip.sum()
                keep &= ~in_skip.values
                print(f"  Excluded {n_skipped} plots overlapping SKIP_REGIONS: {SKIP_REGIONS}")
            else:
                print(f"  WARNING: none of {SKIP_REGIONS} found in regions file")

        if parts:
            combined = pd.concat(parts, ignore_index=True)
            region_union = make_valid(combined.unary_union)
            centers = gpd.GeoSeries(
                gpd.points_from_xy(results["point_x"], results["point_y"]), crs="EPSG:4326"
            )
            in_region = centers.intersects(region_union)
            n_dropped = (~in_region).sum()
            keep &= in_region.values
            print(f"  Dropped {n_dropped} plots outside selected regions")

    # --- Year filtering ---
    if FILTER_YEARS is not None:
        in_years = np.isin(results["avg_year"].astype(int), FILTER_YEARS)
        n_dropped = (~in_years).sum()
        keep &= in_years
        print(f"  Dropped {n_dropped} plots with AVG_YEAR not in {FILTER_YEARS}")

    # --- Max AGB value ---
    over_max = results["agbref"] > MAX_VALUE
    n_over = (over_max & ~np.isnan(results["agbref"])).sum()
    keep &= ~over_max
    print(f"  Dropped {n_over} plots with AGBRef > {MAX_VALUE} t/ha")

    n_kept = keep.sum()
    print(f"  => {n_kept}/{n} plots retained")

    return {k: v[keep] for k, v in results.items()}


# === Load cached results or compute ===
if CACHE_PATH.exists():
    print(f"Loading cached results from {CACHE_PATH}")
    results = dict(np.load(CACHE_PATH))
else:
    results = compute_results()
    np.savez(CACHE_PATH, **results)
    print(f"Results cached to {CACHE_PATH}")

# === Apply filters ===
print("\nApplying filters...")
results = apply_filters(results)

# === Summary stats ===
def print_stats(name, pred, truth):
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    p, t = pred[valid], truth[valid]
    if len(p) == 0:
        print(f"  {name}: no valid comparisons")
        return
    bias = np.mean(p - t)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    corr = np.corrcoef(p, t)[0, 1]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {name}: n={len(p)}, bias={bias:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, r={corr:.3f}, R²={r2:.3f}")

print("\n=== Comparison Statistics (t/ha) ===")
if "mean" in METHODS:
    print_stats("nico_mean  ", results["nico_mean"], results["agbref"])
    print_stats("cci_mean   ", results["cci_mean"], results["agbref"])
if "median" in METHODS:
    print_stats("nico_median", results["nico_median"], results["agbref"])
    print_stats("cci_median ", results["cci_median"], results["agbref"])
if "p90" in METHODS:
    print_stats("nico_p90   ", results["nico_p90"], results["agbref"])
    print_stats("cci_p90    ", results["cci_p90"], results["agbref"])



if PLOT:

    COLORS = {
        "nico": "#0084FF",  # blue
        "cci":  "#C02BF2",  # orange
    }

    BIN_EDGES = np.arange(0, 550, 25)
    SIZE_THRESHOLDS = [10, 20, 50, 100, 200]
    SIZE_VALUES     = [10, 25, 50, 90, 140, 200]

    def get_marker_size(count):
        for thresh, size in zip(SIZE_THRESHOLDS, SIZE_VALUES):
            if count < thresh:
                return size
        return SIZE_VALUES[-1]

    lim = 400

    # Columns = models, rows = plot type (scatter top, binned bottom)
    method_configs = [m for m in ["mean", "median", "p90"] if m in METHODS]
    col_configs = []
    for m in method_configs:
        col_configs.append(("nico_" + m, f"Nico Film ({m})", "nico"))
        col_configs.append(("cci_"  + m, f"ESA CCI ({m})",       "cci"))
    n_cols = len(col_configs)

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # --- Row 0: scatter ---
    for col_idx, (key, title, color_key) in enumerate(col_configs):
        ax = axes[0, col_idx]
        pred  = results[key]
        truth = results["agbref"]
        valid = ~np.isnan(pred) & ~np.isnan(truth)
        p, t  = pred[valid], truth[valid]

        color = COLORS[color_key]
        ax.scatter(t, p, alpha=0.5, s=10, color=color)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        if 'Nico Film' in title: _title = 'Ours'
        if 'CCI' in title: _title = 'ESA CCI'
        ax.set_title(_title, fontsize=14)
        ax.set_aspect("equal")

        bias   = np.mean(p - t)
        rmse   = np.sqrt(np.mean((p - t) ** 2))
        corr   = np.corrcoef(p, t)[0, 1]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2     = 1 - ss_res / ss_tot
        mae   = np.mean(np.abs(p - t))
        ax.text(0.05, 0.95, f"r={corr:.3f}\nR²={r2:.3f}\nRMSE={rmse:.1f}\nME={bias:.1f}\nMAE={mae:.1f}",
                transform=ax.transAxes, va="top", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Row 1: binned ---
    for col_idx, (key, title, color_key) in enumerate(col_configs):
        ax = axes[1, col_idx]
        pred  = results[key]
        truth = results["agbref"]
        valid = ~np.isnan(pred) & ~np.isnan(truth)
        p, t  = pred[valid], truth[valid]

        color = 'black' # COLORS[color_key]
        bin_centers, bin_means, bin_stds, bin_sizes = [], [], [], []
        for j in range(len(BIN_EDGES) - 1):
            lo, hi = BIN_EDGES[j], BIN_EDGES[j + 1]
            mask = (t >= lo) & (t < hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_means.append(np.mean(p[mask]))
            bin_stds.append(np.std(p[mask]))
            bin_sizes.append(mask.sum())

        bin_centers  = np.array(bin_centers)
        bin_means    = np.array(bin_means)
        bin_stds     = np.array(bin_stds)
        marker_sizes = [get_marker_size(s) for s in bin_sizes]

        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="none",
                    ecolor=color, elinewidth=1, capsize=2, zorder=1, alpha=0.6)
        ax.scatter(bin_centers, bin_means, s=marker_sizes, c=color, zorder=2)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

    # Size legend on bottom-left axis
    legend_labels = ["< 10", "10–20", "20–50", "50–100", "100–200", "> 200"]
    for size, label in zip(SIZE_VALUES, legend_labels):
        axes[1, 0].scatter([], [], s=size, c="gray", label=label)
    axes[1, 0].legend(title="#/bin", loc="upper left", fontsize=11,
                      title_fontsize=12, framealpha=0.8, labelspacing=1.0)

    # Axis labels — only on edges
    for ax in axes[1, :]:
        ax.set_xlabel("AGBRef (t/ha)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted AGB (t/ha)", fontsize=12)
    
    # Increase the fontsize of the x and y ticks
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Row labels on the left
    """
    for row_idx, row_label in enumerate(["Scatter", "Binned"]):
        axes[row_idx, 0].annotate(
            row_label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.18, 0.5), textcoords="axes fraction",
            fontsize=12, fontweight="bold", ha="right", va="center", rotation=90,
        )
    """

    plt.tight_layout()
    combined_path = PLOT_DIR / f"comparison_combined_{OUTPUT_SUFFIX}.png"
    plt.savefig(combined_path, dpi=1200, bbox_inches="tight")
    print(f"\nCombined plot saved to {combined_path}")
    plt.close()
