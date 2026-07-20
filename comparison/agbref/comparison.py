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

NOTE: the absolute paths below (SPLITS_PKL, S2_TILES_SHP, REGIONS_PATH, NICO_DIR,
CCI_DIR) point to the authors' cluster layout and MUST be edited for your environment.
Paths built from BASE_DIR (AGBRef, cached results) are repo-relative and need no change.
"""

PLOT = True # Set to True to enable plotting (scatter + binned)
MAX_VALUE = 500
METHODS = ["mean"] # ["mean", "median", "p90"]

# Each AGBRef record aggregates the field plots inside a 0.1 deg grid cell, so statistics are
# taken over that exact cell (half-width 0.05 deg in lon/lat). The AGBRef *geometry* files carry an
# extra 1 km download margin (see comparison/agbref/make_agbref_polygons.py); that margin is only
# there so the AEF/CCI crops fully contain the cell and is deliberately clipped away here.
HALF_DEG = 0.05
_BUFFER_M = 5000  # legacy 10km-square half-side; kept only for make_utm_square (no longer used)

# --- Coverage accounting ---
# Statistics clip to the 0.1 deg cell, but a source raster can still fall short of it (e.g. an
# interrupted download or a bad crop), in which case its mean would be an average over an unknown
# sub-area. Coverage is therefore measured explicitly, against the cell rather than against the
# raster's own extent (a short raster would otherwise report itself as fully covered).
#
# Each plot's 0.1 deg cell is rasterised onto a canonical 10m UTM grid and each source is
# reprojected onto it, which gives:
#   <src>_cov      fraction of the cell where that source has a valid pixel
#   common_cov     fraction where BOTH sources are valid
#   <src>_mean_common  mean over the common-valid pixels only, i.e. the two sources compared over
#                      exactly the same ground area
# With the 1 km download margin these should read ~100% for every plot; anything well below flags a
# raster that needs re-downloading or re-cropping.
COVERAGE = True     # Set to False to skip the common-grid pass (roughly halves the runtime)
GRID_RES_M = 10     # Resolution of the canonical comparison grid, in metres

# Minimum fraction of the plot that must be covered for a plot to be kept. Plots below this are
# dropped, because their mean is taken over a sub-area and is not an estimate of the plot mean.
# Set to None to disable the filter and only report coverage.
MIN_COVERAGE = None

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
# Created here rather than assumed: the directory is a gitignored output, so a fresh clone has no
# plots/ and every savefig at the end of the run dies -- after the full comparison has been computed.
PLOT_DIR.mkdir(parents = True, exist_ok = True)


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


def make_cell(lon, lat, half_deg=HALF_DEG):
    """
    The exact 0.1 deg grid cell centred on (lon, lat), as an EPSG:4326 polygon, plus the local
    UTM EPSG code.

    This is the footprint the AGBRef ground truth (AGB_T_HA) aggregates over, so every statistic
    is taken over this cell. The AGBRef geometry files carry an extra 1 km download margin so the
    AEF/CCI crops fully contain the cell; that margin is deliberately excluded here. Note the cell
    is built directly in lon/lat (not via a projected buffer) so it never picks up the Web-Mercator
    cos(lat) distortion that produced the original shrunk squares.
    """
    return box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg), utm_epsg_from_lonlat(lon, lat)


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
# Helper: canonical comparison grid for a plot
# =========================================================================
def make_utm_grid(lon, lat, res_m=GRID_RES_M):
    """
    Build the canonical grid covering a plot's 0.1 deg cell, in the plot's local UTM zone, at
    res_m resolution.

    The grid is derived from the cell geometry, NOT from any raster, which is the whole point:
    coverage measured against a raster's own extent is 100% by construction, even when the
    raster falls short. The cell is nearly axis-aligned in its own UTM zone, so its UTM bounding
    box is essentially the cell itself.

    Returns
    -------
    transform : Affine transform of the grid (north-up)
    width, height : grid size in pixels
    utm_epsg : EPSG code of the local UTM zone
    """
    cell_4326, utm_epsg = make_cell(lon, lat)
    minx, miny, maxx, maxy = (
        gpd.GeoSeries([cell_4326], crs="EPSG:4326").to_crs(epsg=utm_epsg).iloc[0].bounds
    )
    width = int(round((maxx - minx) / res_m))
    height = int(round((maxy - miny) / res_m))
    # from_origin takes the top-left corner, and yields a north-up transform
    transform = rasterio.transform.from_origin(minx, maxy, res_m, res_m)
    return transform, width, height, utm_epsg


def _reproject_onto_grid(raster_path, nodata, transform, width, height, utm_epsg):
    """
    Reproject a raster's first band onto a fixed destination grid, returning a float32 array
    where invalid (nodata, or outside the source extent) pixels are NaN.

    Because the destination grid is fixed by the plot rather than by the source, pixels that the
    source simply does not reach stay NaN - which is exactly the truncation we want to measure.
    """
    dst = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=transform,
            dst_crs=CRS.from_epsg(utm_epsg),
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst


def extract_common_stats(nico_path, cci_path, lon, lat):
    """
    Put both sources on the same canonical 10m grid over the plot, and measure how much of the
    plot each one actually covers, plus their means over the pixels where both are valid.

    CCI is ~100m in EPSG:4326 and nico_film is 10m in a UTM zone, so CCI is upsampled onto the
    10m grid with nearest-neighbour: this replicates each CCI value across its ~10x10 block
    rather than inventing intermediate values, so the CCI mean is unchanged where coverage is
    complete, and is restricted to the right ground area where it is not.

    Returns
    -------
    dict with nico_cov, cci_cov, common_cov (fractions of the plot) and
    nico_mean_common, cci_mean_common (means over the common-valid pixels, or NaN).
    """
    transform, width, height, utm_epsg = make_utm_grid(lon, lat)
    n_px = width * height
    out = {"nico_cov": np.nan, "cci_cov": np.nan, "common_cov": np.nan,
           "nico_mean_common": np.nan, "cci_mean_common": np.nan}

    try:
        nico = _reproject_onto_grid(nico_path, -9999.0, transform, width, height, utm_epsg)
        # CCI has NO nodata: ESA declares none, and 0 means AGB = 0 Mg/ha (the maps are not
        # forest-masked). This used to pass 65535, which matched 0% of pixels -- inert, but it
        # implied the zeros were missing data. Every CCI pixel is valid, so `common` below is
        # governed entirely by where nico_film/AEF is valid, which is the intended restriction.
        cci = _reproject_onto_grid(cci_path, None, transform, width, height, utm_epsg)
    except Exception as e:
        print(f"  WARNING: could not build the common grid: {e}")
        return out

    nico_valid, cci_valid = ~np.isnan(nico), ~np.isnan(cci)
    common = nico_valid & cci_valid

    out["nico_cov"] = float(nico_valid.sum()) / n_px
    out["cci_cov"] = float(cci_valid.sum()) / n_px
    out["common_cov"] = float(common.sum()) / n_px

    if common.any():
        out["nico_mean_common"] = float(nico[common].mean())
        out["cci_mean_common"] = float(cci[common].mean())

    return out


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

    # Clip every plot to its exact 0.1 deg cell (the footprint AGB_T_HA aggregates over),
    # discarding the 1 km download margin carried by the geometry file. Built directly in lon/lat,
    # so no projected-buffer cos(lat) distortion.
    print("Clipping plots to their 0.1 deg grid cells...")
    new_geoms = []
    utm_codes = []
    for _, row in gdf.iterrows():
        cell, epsg = make_cell(row["POINT_X"], row["POINT_Y"])
        new_geoms.append(cell)
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
        # Coverage of the plot, and the means restricted to the commonly-covered area
        "nico_cov": np.full(n, np.nan),
        "cci_cov": np.full(n, np.nan),
        "common_cov": np.full(n, np.nan),
        "nico_mean_common": np.full(n, np.nan),
        "cci_mean_common": np.full(n, np.nan),
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
            # nodata=None: see extract_common_stats -- CCI declares no nodata and 0 is a real
            # 0 Mg/ha value, so every pixel inside the plot polygon counts.
            mean, median, p90 = extract_raster_stats(cci_path, geom, nodata=None, utm_epsg=utm_epsg)
            results["cci_mean"][i] = mean
            results["cci_median"][i] = median
            results["cci_p90"][i] = p90
        else:
            print(f"  WARNING: CCI {cci_path.name} not found")

        # --- Coverage, and the two sources compared over the same ground area ---
        if COVERAGE and nico_path.exists() and cci_path.exists():
            for k, v in extract_common_stats(nico_path, cci_path, row["POINT_X"], row["POINT_Y"]).items():
                results[k][i] = v

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

    # --- Coverage ---
    if COVERAGE and "common_cov" in results and MIN_COVERAGE is not None:
        cov = results["common_cov"]
        # A NaN coverage means the grid could not be built at all, which is also a failure
        short = ~(cov >= MIN_COVERAGE)
        n_short = (short & keep).sum()
        keep &= ~short
        print(f"  Dropped {n_short} plots covering < {MIN_COVERAGE:.0%} of the plot area")

    n_kept = keep.sum()
    print(f"  => {n_kept}/{n} plots retained")

    return {k: v[keep] for k, v in results.items()}


# === Load cached results or compute ===
# The cache predates the coverage keys, so a cache without them must be rebuilt rather than
# silently reused: the whole point of this pass is to notice short rasters.
_REQUIRED_KEYS = {"nico_cov", "cci_cov", "common_cov", "nico_mean_common", "cci_mean_common"}

# Bumped whenever the geometry the stats are taken over changes, so an old cache is never silently
# reused. "cell-0.1deg-v1" = stats clipped to the exact 0.1 deg grid cell (was a 10km UTM square,
# built off cos(lat)-shrunk polygons). Stored as a separate npz key, NOT inside results (every
# results value must stay a length-n array for the post-hoc filter at the end of compute_results).
_GEOM_VERSION = "cell-0.1deg-v1"

results = None
if CACHE_PATH.exists():
    cached = dict(np.load(CACHE_PATH))
    cached_ver = str(cached.pop("_geom_version", "legacy-10km-square"))
    missing_keys = _REQUIRED_KEYS - set(cached)
    if cached_ver != _GEOM_VERSION:
        print(f"Cache at {CACHE_PATH} was built for geometry '{cached_ver}', need '{_GEOM_VERSION}'; recomputing")
    elif COVERAGE and missing_keys:
        print(f"Cache at {CACHE_PATH} predates the coverage keys ({', '.join(sorted(missing_keys))}); recomputing")
    else:
        print(f"Loading cached results from {CACHE_PATH}")
        results = cached

if results is None:
    results = compute_results()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, _geom_version=np.array(_GEOM_VERSION), **results)
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

# === Coverage report ===
# This is the check that the original truncation escaped: without it, a mean over a fraction of
# the plot is indistinguishable from a mean over the whole plot.
def print_coverage(name, cov):
    finite = cov[~np.isnan(cov)]
    n_nan = int(np.isnan(cov).sum())
    if len(finite) == 0:
        print(f"  {name}: no coverage measured ({n_nan} NaN)")
        return
    print(f"  {name}: median={np.median(finite):6.1%}  mean={finite.mean():6.1%}  "
          f"min={finite.min():6.1%}   full(>=99%)={int((finite >= 0.99).sum()):3d}/{len(finite)}  "
          f"<50%={int((finite < 0.5).sum()):3d}  NaN={n_nan}")

if COVERAGE and "common_cov" in results:
    print("\n=== Plot coverage (fraction of the 10km x 10km plot) ===")
    print_coverage("nico_film ", results["nico_cov"])
    print_coverage("CCI       ", results["cci_cov"])
    print_coverage("common    ", results["common_cov"])

    n_short = int((results["common_cov"] < 0.99).sum())
    if n_short:
        print(f"\n  WARNING: {n_short} plot(s) are not fully covered, so their mean is an average "
              f"over a sub-area\n           rather than an estimate of the plot mean. If this is "
              f"not expected, the AEF\n           download is truncated again - see AEF_COVERAGE_TODO.md.")

print("\n=== Comparison Statistics (t/ha) ===")
if "mean" in METHODS:
    print_stats("nico_mean  ", results["nico_mean"], results["agbref"])
    print_stats("cci_mean   ", results["cci_mean"], results["agbref"])
    if COVERAGE and "nico_mean_common" in results:
        # Same plots, but both sources averaged over exactly the same ground area
        print_stats("nico_common", results["nico_mean_common"], results["agbref"])
        print_stats("cci_common ", results["cci_mean_common"], results["agbref"])
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
