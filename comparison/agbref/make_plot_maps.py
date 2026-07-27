"""
Per-plot map comparison: for a handful of AGBRef plots, show a Sentinel-2 true-colour crop for
context, the AEF ("Ours") prediction crop and the ESA CCI crop, then -- because the AGBRef ground
truth is a single aggregated value, not a map -- a panel comparing the *distributions* of the two
sources' pixel values against that ground truth drawn as a point with its reported uncertainty.

The Sentinel-2 panel is the pre-composited 8-bit TCI (true-colour image, B04/B03/B02), used as-is:
it is already brightness-balanced by ESA, so no per-band stretch is applied (an independent per-band
percentile stretch rescales the band ratios and shifts hue -- the wrong thing for a context image).
TCI files live in data/s2_tci/ named T<tile>_<datetime>_TCI_10m.<ext>; a plot is matched to its
tile via the s2_tiles field of example_plots_s2.geojson, and when a tile has several acquisitions
the least-cloudy one over the plot cell is chosen automatically (see pick_s2).

Why a distribution and not a solid GT tile: the ground truth is one number per plot (AGB_T_HA, the
mean over `n` field plots, with total variance `varTot`), so rendering it as an image wastes a panel
and invites a false pixel-to-pixel reading. Instead the GT is a *reference the maps are read
against*: a vertical line (+/- sqrt(varTot) band) on the distribution panel, and a tick on the
shared map colourbar. The distribution panel then shows honestly what a scalar-vs-map comparison
actually is -- where each source's pixels sit relative to the target -- and makes CCI's high-biomass
saturation tail visible.

Both sources are reprojected onto the plot's canonical 10 m UTM cell grid -- the SAME grid
comparison.py uses for its coverage/common-mean pass -- so the two map panels are pixel-aligned and
each is exactly the ground the reported means summarise. The four grid helpers below are copied
verbatim from comparison.py; if that file's make_utm_grid / reproject-onto-grid / make_cell /
utm_epsg_from_lonlat ever change, update them here too (they are pure geometry, no shared state).

viridis is deliberate and correct here: unlike the learned-feature figure (where viridis would
falsely imply "biomass"), these panels ARE biomass in t/ha.

Usage:
    python make_plot_maps.py [--plots 292 304 293 26 290 11] [--out <path w/o ext>]
"""

import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# === Paths (match comparison.py) ===
BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "results" / "comparison_results.npz"
AGBREF_PATH = BASE_DIR / "data" / "AGBRef.geojson"
PLOTS_GEOJSON = BASE_DIR / "data" / "example_plots_s2.geojson"
S2_DIR = BASE_DIR / "data" / "s2_tci"
NICO_DIR = Path("/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions/nico_film/59620113-1_59620113-1_59620113-1")
CCI_DIR = Path("/scratch3/gsialelli/CCI")

GRID_RES_M = 10
HALF_DEG = 0.05
CMAP = "viridis"
BG = 0.85  # grey for nodata

# Scale bar on the Sentinel-2 panel, same style/constants as make_map_figure.py so the two
# qualitative figures read as one system. The cell is 0.1 deg on a side (~11 km at these
# latitudes), so 0.25 of the panel snaps to the 2 km entry.
SCALEBAR_FRAC = 0.25
SCALEBAR_NICE_KM = (1, 2, 5, 10, 20)
SCALEBAR_COLOR = "white"
SCALEBAR_OUTLINE = "black"

# Source colours: identical to comparison.py's scatter figure so the two figures read as one system.
C_OURS = "#0084FF"   # AEF / nico_film
C_CCI = "#C02BF2"    # ESA CCI

# Default example plots: spread across the AGBRef range, in the published retained set (no Japan,
# |lat| <= 55 so the 0.1 deg cell is not latitude-shrunk). Chosen to be representative, not
# favourable: CCI is closer at low biomass, AEF far closer at high biomass (26, 11 are tropical
# forest where CCI overshoots past 360 t/ha).
# Plot 292 (tile 12STG, GT 0 t/ha over arid ground) was dropped: an aggregated 0 t/ha reference is
# the least reliable anchor of the set. Its TCI is still in data/s2_tci/ but no longer plotted.
DEFAULT_PLOTS = [304, 293, 26, 290, 11]

# Human-readable location for each plot, so the row label carries a geography a reader recognises
# rather than an opaque plot index. Derived from each plot's POINT_X/POINT_Y (WGS84) in
# AGBRef.geojson / the results cache, resolved to country (and state/region where it disambiguates):
#   304  (-108.55, 37.95)  SW Colorado, USA       (semi-arid, GT 4 t/ha)
#   293  (-108.55, 37.75)  SW Colorado, USA       (semi-arid, GT 25 t/ha)
#    26  ( 11.55,   4.15)  Cameroon               (tropical forest, GT 88 t/ha)
#   290  (-80.45,  37.45)  Virginia, USA          (Appalachian forest, GT 164 t/ha)
#    11  ( 12.85,   3.15)  Cameroon               (dense tropical forest, GT 281 t/ha)
# Keyed by plot index; falls back to "plot {i}" for any plot not listed.
PLOT_REGION = {
    304: "SW Colorado, USA",
    293: "SW Colorado, USA",
    26:  "Cameroon",
    290: "Virginia, USA",
    11:  "Cameroon",
}


# ============================ helpers copied from comparison.py ============================
def utm_epsg_from_lonlat(lon, lat):
    """Return the EPSG code of the UTM zone for a given lon/lat (WGS84)."""
    zone_number = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone_number


def make_cell(lon, lat, half_deg=HALF_DEG):
    """The exact 0.1 deg grid cell centred on (lon, lat), as EPSG:4326 bounds, plus the UTM EPSG."""
    from shapely.geometry import box
    return box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg), utm_epsg_from_lonlat(lon, lat)


def make_utm_grid(lon, lat, res_m=GRID_RES_M):
    """Canonical north-up grid covering the plot's 0.1 deg cell, in its local UTM zone."""
    import geopandas as gpd
    cell_4326, utm_epsg = make_cell(lon, lat)
    minx, miny, maxx, maxy = (
        gpd.GeoSeries([cell_4326], crs="EPSG:4326").to_crs(epsg=utm_epsg).iloc[0].bounds
    )
    width = int(round((maxx - minx) / res_m))
    height = int(round((maxy - miny) / res_m))
    transform = rasterio.transform.from_origin(minx, maxy, res_m, res_m)
    return transform, width, height, utm_epsg


def reproject_onto_grid(raster_path, nodata, transform, width, height, utm_epsg):
    """Reproject a raster's band 1 onto the fixed grid; invalid pixels become NaN."""
    dst = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=nodata,
            dst_transform=transform, dst_crs=CRS.from_epsg(utm_epsg), dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst
# ==========================================================================================


def cci_path_for(i, avg_year):
    """CCI filename for plot i, matching comparison.py's year rule (clamped to >= 2018)."""
    yy = int(max(avg_year, 2018)) - 2000
    return CCI_DIR / f"CCI_AGBRef_{i}_{yy}.tif"


def _tile_of(fname):
    """Extract the MGRS tile ('32NQK') from a TCI filename token 'T32NQK_...'."""
    tok = Path(fname).name.split("_")[0]
    return tok[1:] if tok.startswith("T") else tok


def _date_of(fname):
    """Human-readable acquisition date 'YYYY-MM-DD' from the second filename token."""
    try:
        d = Path(fname).name.split("_")[1][:8]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    except Exception:
        return "?"


def load_s2(path, lon, lat):
    """
    Crop a TCI (3-band uint8, native tile UTM) onto the plot's 10 m cell grid.

    All six downloaded tiles are in the same UTM zone as their plot, so this is effectively a crop
    and 10 m re-grid (nearest, to keep the balanced TCI colours untouched). Returns (H, W, 3) uint8
    and a (H, W) bool validity mask (TCI encodes off-swath / fill as 0,0,0).
    """
    transform, width, height, utm_epsg = make_utm_grid(lon, lat)
    dst = np.zeros((3, height, width), dtype=np.uint8)
    with rasterio.open(path) as src:
        for b in range(3):
            reproject(
                source=rasterio.band(src, b + 1), destination=dst[b],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=CRS.from_epsg(utm_epsg),
                resampling=Resampling.nearest,
            )
    rgb = dst.transpose(1, 2, 0)
    valid = rgb.sum(axis=2) > 0
    return rgb, valid


def _cloud_score(rgb, valid):
    """Fraction of the cell lost to cloud or fill: bright-in-all-channels (~cloud) plus nodata."""
    if valid.sum() == 0:
        return 1.0
    bright = (rgb.min(axis=2) > 180) & valid   # clouds are bright and near-neutral in TCI
    return (bright.sum() + (~valid).sum()) / valid.size


def pick_s2(tiles, lon, lat):
    """
    Choose the TCI file for a plot: any file in S2_DIR whose tile is one of `tiles` (a plot can
    overlap several MGRS tiles, e.g. across a UTM-zone boundary, and only one may be downloaded),
    and among those the one with the lowest cloud+fill fraction over the plot cell. Returns
    (path, date, score) or None if none of the candidate tiles were downloaded.
    """
    want = set(tiles)
    cands = sorted(p for p in S2_DIR.glob("*") if _tile_of(p) in want)
    if not cands:
        return None
    scored = []
    for p in cands:
        rgb, valid = load_s2(p, lon, lat)
        scored.append((_cloud_score(rgb, valid), p))
    scored.sort(key=lambda s: s[0])
    best_score, best = scored[0]
    if len(cands) > 1:
        print(f"    {_tile_of(best)}: {len(cands)} candidate scene(s) -> chose {best.name} "
              f"(cloud+fill {best_score*100:.1f}%); others "
              + ", ".join(f"{_tile_of(p)}/{p.name.split('_')[1][:8]}={s*100:.1f}%" for s, p in scored[1:]))
    return best, _date_of(best), best_score


def load_plot(i, lon, lat, avg_year):
    """Return (nico, cci) arrays on the plot's common 10 m cell grid, NaN outside valid data."""
    transform, width, height, utm_epsg = make_utm_grid(lon, lat)
    nico = reproject_onto_grid(NICO_DIR / f"{i}.tif", -9999.0, transform, width, height, utm_epsg)
    # CCI has no nodata and 0 is a real 0 t/ha value (see comparison.py), so pass nodata=None.
    cci = reproject_onto_grid(cci_path_for(i, avg_year), None, transform, width, height, utm_epsg)
    return nico, cci


def draw_map(ax, arr, vmax, title):
    """Show a biomass array (NaN=nodata) with a shared 0..vmax viridis scale."""
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(color=str(BG))
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def value_label(ax, text):
    """Annotate a panel bottom-left, in the same white-on-dark style as the per-panel RMSE labels
    of make_map_figure.py -- so the number sits on the map it describes rather than in the title.

    Every annotated number in this figure is a biomass in t/ha averaged over the plot cell (the
    map panels' cell mean, the S2 panel's AGBRef reference), so the word "mean" is left to the
    caption rather than repeated on all three panels of every row.
    """
    ax.text(0.04, 0.05, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))


def add_scalebar(ax, shape, res_m=GRID_RES_M):
    """
    Draw a scale bar top-right, sized from the panel's own ground extent.

    Top-right rather than bottom-right (as in make_map_figure.py) because the bottom-left of this
    panel already carries the AGBRef badge, and the cell is portrait-shaped at mid latitudes -- the
    two collide along the bottom edge.

    The panel is an imshow of the plot's canonical UTM cell grid with no `extent`, so axes
    coordinates are pixel indices and the grid resolution (`res_m`) is exact by construction --
    it is the grid this figure builds, not an assumed resolution of some source raster.
    """
    h, w = shape[:2]
    if w <= 0 or h <= 0:
        return
    width_km = w * res_m / 1000
    length_km = min(SCALEBAR_NICE_KM, key=lambda k: abs(k - SCALEBAR_FRAC * width_km))
    bar_px = length_km * 1000 / res_m

    margin = 0.05 * w
    x1 = w - margin
    x0 = x1 - bar_px
    y = 0.09 * h
    stroke = [pe.withStroke(linewidth=3, foreground=SCALEBAR_OUTLINE)]
    ax.plot([x0, x1], [y, y], color=SCALEBAR_COLOR, lw=2, solid_capstyle="butt",
            path_effects=stroke, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.025 * h, f"{length_km} km", color=SCALEBAR_COLOR,
            fontsize=9, fontweight="bold", ha="center", va="bottom", path_effects=stroke)


def draw_distribution(ax, nico, cci, gt, gt_std, xmax, first_row, last_row):
    """
    Distribution panel: normalised histograms of the two sources' valid pixels, against the GT
    point (+/- std band). x-axis is biomass in t/ha, matched to the map colourbar's 0..xmax so the
    GT line sits at the same fraction as the colourbar tick.
    """
    nv = nico[np.isfinite(nico)]
    cv = cci[np.isfinite(cci)]
    bins = np.linspace(0, xmax, 41)
    for data, color, label in ((nv, C_OURS, "Ours (AEF)"), (cv, C_CCI, "ESA CCI")):
        if data.size:
            ax.hist(data, bins=bins, density=True, color=color, alpha=0.40, label=label)
            ax.hist(data, bins=bins, density=True, histtype="step", color=color, lw=1.3)

    # GT as the reference the maps are read against: solid black line + shaded reported uncertainty.
    if gt_std and np.isfinite(gt_std) and gt_std > 0:
        ax.axvspan(max(0, gt - gt_std), gt + gt_std, color="0.4", alpha=0.18, zorder=0)
    ax.axvline(gt, color="k", lw=1.8, zorder=5, label="AGBRef GT")
    # Each source's mean, dashed and colour-matched, so over/under-shoot vs GT is legible.
    if nv.size:
        ax.axvline(float(nv.mean()), color=C_OURS, ls="--", lw=1.4, zorder=4)
    if cv.size:
        ax.axvline(float(cv.mean()), color=C_CCI, ls="--", lw=1.4, zorder=4)

    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)
    if last_row:
        ax.set_xlabel("AGB (t/ha)", fontsize=9)
    if first_row:
        ax.set_title("Pixel distribution vs GT", fontsize=10)
        ax.legend(fontsize=9.5, loc="upper right", framealpha=0.85, handlelength=1.6)


def draw_s2(ax, rgb, valid, title):
    """Show a TCI crop as-is (no stretch); nodata pixels drawn as neutral grey, plus a scale bar.

    The tile name now lives in the row label, and the acquisition date is not printed at all (the
    other qualitative figures do not print it either).
    """
    disp = rgb.copy()
    disp[~valid] = int(BG * 255)
    ax.imshow(disp, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    add_scalebar(ax, disp.shape)
    if title:
        ax.set_title(title, fontsize=10)


def main(plots, out_path, dpi):
    import geopandas as gpd
    cache = dict(np.load(CACHE_PATH, allow_pickle=True))
    px, py = cache["point_x"], cache["point_y"]
    yr, agb = cache["avg_year"], cache["agbref"]

    # varTot / n live in the geometry file, not the cache. Row order matches the cache and the
    # <i>.tif filenames (compute_results iterates the geojson in order).
    gdf = gpd.read_file(AGBREF_PATH)
    var_tot = gdf["varTot"].values
    n_field = gdf["n"].values

    # plot_index -> list of overlapping MGRS tiles, from the s2_tiles field written by the geojson
    # step (e.g. "12STG(100%), 11SQB(100%)"). A plot can straddle a UTM-zone boundary and overlap
    # several tiles; pick_s2 uses whichever was actually downloaded.
    sel = gpd.read_file(PLOTS_GEOJSON)
    tiles_of_plot = {int(r.plot_index): [t.split("(")[0].strip() for t in str(r.s2_tiles).split(",")]
                     for r in sel.itertuples()}

    nrows = len(plots)
    fig = plt.figure(figsize=(3.0 * 3 + 0.5 + 3.4, 3.0 * nrows))
    # Sentinel-2 | Ours map | CCI map | shared colourbar | spacer | distribution
    # The empty spacer column keeps the colourbar's tick labels off the distribution's left edge.
    gs = gridspec.GridSpec(nrows, 6, width_ratios=[1, 1, 1, 0.05, 0.22, 1.35],
                           wspace=0.10, hspace=0.28, left=0.08, right=0.97, top=0.94, bottom=0.05)

    for r, i in enumerate(plots):
        nico, cci = load_plot(i, px[i], py[i], yr[i])
        gt = float(agb[i])
        gt_std = float(np.sqrt(var_tot[i])) if np.isfinite(var_tot[i]) else np.nan

        # One scale per row (0 .. p99 of everything shown, incl. the GT band), used for BOTH map
        # panels AND the distribution x-axis. Rows are independent because plot means span 0..380.
        stack = np.concatenate([
            nico[np.isfinite(nico)], cci[np.isfinite(cci)],
            np.array([gt + (gt_std if np.isfinite(gt_std) else 0.0)]),
        ])
        vmax = max(float(np.percentile(stack, 99)) if stack.size else 1.0, 1.0)
        n_mean, c_mean = float(np.nanmean(nico)), float(np.nanmean(cci))

        # --- Sentinel-2 context (leftmost) ---
        axs = fig.add_subplot(gs[r, 0])
        tile = "?"
        picked = pick_s2(tiles_of_plot.get(i, []), px[i], py[i])
        if picked is not None:
            path, date, _ = picked
            tile = _tile_of(path)
            rgb, valid = load_s2(path, px[i], py[i])
            draw_s2(axs, rgb, valid, ("Sentinel-2 (TCI)" if r == 0 else ""))
        else:
            axs.text(0.5, 0.5, "S2 not found", ha="center", va="center",
                     fontsize=9, color="0.4", transform=axs.transAxes)
            axs.set_facecolor("0.95"); axs.set_xticks([]); axs.set_yticks([])
            if r == 0:
                axs.set_title("Sentinel-2 (TCI)", fontsize=10)
        # Row label: recognisable geography, then the identifiers a reader may want to look up --
        # the MGRS tile the context image comes from and the AGBRef plot index.
        region = PLOT_REGION.get(i, "unknown")
        axs.set_ylabel(f"{region}\n({tile}) (plot {i})", fontsize=9, fontweight="bold", labelpad=6)
        # The AGBRef reference sits on the S2 panel in the same badge style as the two map means,
        # so all three numbers of a row are read the same way: a cell-mean biomass in t/ha.
        value_label(axs, f"{gt:.0f}±{gt_std:.0f} t/ha (n={int(n_field[i])})")

        # Each map's cell mean is annotated ON the panel (bottom-left) rather than in its title,
        # matching the per-panel RMSE labels of the prediction-map figure.
        ax0 = fig.add_subplot(gs[r, 1])
        draw_map(ax0, nico, vmax, ("Ours (AEF)" if r == 0 else ""))
        value_label(ax0, f"{n_mean:.0f} t/ha")
        ax1 = fig.add_subplot(gs[r, 2])
        draw_map(ax1, cci, vmax, ("ESA CCI" if r == 0 else ""))
        value_label(ax1, f"{c_mean:.0f} t/ha")

        # Shared colourbar for the two maps, with the GT drawn on it as the reference.
        cax = fig.add_subplot(gs[r, 3])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=CMAP), cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("t/ha", fontsize=7)
        if 0 <= gt <= vmax:
            cb.ax.axhline(gt, color="k", lw=1.6)

        axd = fig.add_subplot(gs[r, 5])
        draw_distribution(axd, nico, cci, gt, gt_std, vmax, first_row=(r == 0), last_row=(r == nrows - 1))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", type=int, nargs="+", default=DEFAULT_PLOTS)
    ap.add_argument("--out", type=str, default=str(BASE_DIR / "plots" / "plot_maps_AEF_vs_CCI_vs_AGBRef"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    main(args.plots, args.out, args.dpi)
