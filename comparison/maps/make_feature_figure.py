"""

Compose the feature-visualisation figure: for each location, a zoomed-in true-colour Sentinel-2 crop
next to the PCA -> RGB rendering of each model's learned activations (AGBD-features vs AEF).

Three columns, nothing else: the satellite image, and the two activation maps side by side. There is
deliberately no biomass panel and no viridis anywhere -- viridis reads as "this is a biomass map",
which these are not. The activation panels are literal RGB: the 256-D penultimate activation of each
model is reduced to 3 dimensions by PCA, and those 3 components are shown as red, green and blue.

Reading the activation panels: colour = position in the 3-D PCA of the 256-D activation space, so
*similar colour = similar learned representation*. Absolute hues carry no meaning and are NOT
comparable between panels or between rows -- each PCA is fit independently on its own zoom window,
and PCA component signs and scales are arbitrary. What *is* readable, and is the point of the
figure, is spatial structure: which areas each representation groups together, and which it
separates.

Inputs come from model/extract_features.py, which saves the RAW activations (not an RGB), so the
zoom window, the PCA and the colour mapping can all be changed here in seconds without re-running
the models (a full Sentinel-2 tile load is minutes and ~15 GB):
    <tile>_agbd_feat.npy / <tile>_aef_feat.npy      (H, W, 256) activations
    <tile>_agbd_valid.npy / <tile>_aef_valid.npy    (H, W) bool, nodata mask
    <tile>_s2rgb.tif                                true-colour backdrop, same extent

Usage:
    python make_feature_figure.py [--zoom_px 128] [--features <dir>] [--out <path w/o ext>]

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import rasterio as rs
from os.path import join, exists, dirname, abspath
from os import makedirs

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe

###################################################################################################
# Configuration

# Same locations (and disclosure caveats) as the quantitative map figure; see make_map_figure.py.
# `offset` is the top-left of the zoom window inside the extracted crop, in px; None = centred.
# Europe was 32TNS until 2026-07-21. It was swapped for 32TPT to match make_map_figure.py, which
# dropped 32TNS because its AEF window contains ZERO GEDI footprints in any year (the window is
# Graubuenden/Valtellina, not Austria despite the label) and so could never carry an honest metric.
# The two figures must show the same three regions or a reader will reasonably assume they do.
#
# Region labels are copied from make_map_figure.py verbatim. 49SBT was "South Asia" here and "Asia"
# there; it is Qinling, Shaanxi, CHINA -- East Asia, not South Asia -- so the vaguer shared label is
# the correct one to keep, and the mismatch between the two figures is now gone.
TILES = [
    {"tile": "59GPM", "region": "Australasia", "offset": None},
    {"tile": "32TPT", "region": "Europe",      "offset": None},
    {"tile": "49SBT", "region": "Asia",        "offset": None},
]

# Column labels. The "(PCA -> RGB)" qualifier was dropped from the three activation columns: it was
# repeated verbatim in every one, and the caption already states that each activation panel is the
# 256-D penultimate map reduced to 3 PCA components shown as RGB. "AGBD features" -> "AGBD" for the
# same brevity (the row is the AGBD-features model's activations either way).
COLS = [
    {"key": "s2",     "label": "Sentinel-2 (true colour)"},
    {"key": "agbd",   "label": "AGBD\nactivations"},
    {"key": "aef",    "label": "AEF\nactivations"},
    {"key": "ssl4eo", "label": "SSL4EO-MoCo\nactivations"},
]

# The SSL4EO-MoCo column is a different beast from the two nico_film ones and the figure must not
# pretend otherwise (see model/extract_features_ssl4eo.py). It is a different architecture (ViT +
# RegUPerNet, not XceptionS2_FiLM), a different tap (the 512-D input to conv_reg, not the 256-D
# pre-predictions map), S2-only input, and -- because that model predicts one centre pixel per
# 25x25 patch -- its features are extracted on a 30 m grid and nearest-upsampled to the shared 10 m
# crop, so its panel is legitimately blocky at 30 m. So this is a CROSS-ARCHITECTURE "what each
# model learns" comparison, not the symmetric same-layer/only-the-input-differs comparison of the
# AGBD/AEF pair. The caption must say which columns are commensurable.

BG = 0.85          # grey for nodata pixels
PCT = (2, 98)      # robust percentile stretch per PCA component

# Scale bar for the zoom window. The zoom is a fixed ~1.3 km square, but printing "1.3 km" as text
# on every row (as before) is both cluttered and less useful than a bar a reader can lay against the
# structure. All panels of a row share the same ground extent (z px at 10 m), so one bar on the
# Sentinel-2 panel states the scale for the whole row.
SB_FRAC = 0.30                       # target bar length as a fraction of the panel width
SB_NICE_M = (100, 200, 250, 500, 1000)  # snap to a round length
SB_COLOR = "white"                   # outlined in black so it reads over S2 and PCA colours alike
SB_OUTLINE = "black"

###################################################################################################
# Helpers

def add_scalebar(ax, width_px, res_m = 10) :
    """
    Draw a scale bar in the bottom-right of a feature panel, sized from the panel's ground width.

    Panels are drawn with imshow and no extent, so axes coordinates are pixel indices; the bar
    length in pixels is (length_m / res_m). White with a black outline so it stays legible over both
    the Sentinel-2 crop and the PCA-RGB activations.

    Args:
    - ax: the panel's axes.
    - width_px (int): panel width in pixels (the zoom size z; panels are square).
    - res_m (int): ground resolution of a pixel, in metres (10 m for the shared crop grid).
    """
    width_m = width_px * res_m
    target_m = SB_FRAC * width_m
    length_m = min(SB_NICE_M, key = lambda k : abs(k - target_m))
    bar_px = length_m / res_m

    margin = 0.05 * width_px
    x1 = width_px - margin
    x0 = x1 - bar_px
    y = width_px - margin

    stroke = [pe.withStroke(linewidth = 3, foreground = SB_OUTLINE)]
    ax.plot([x0, x1], [y, y], color = SB_COLOR, lw = 2, solid_capstyle = "butt",
            path_effects = stroke, clip_on = False)
    label = f"{length_m / 1000:.1f} km" if length_m >= 1000 else f"{length_m} m"
    ax.text((x0 + x1) / 2, y - 0.03 * width_px, label, color = SB_COLOR,
            fontsize = 9, fontweight = "bold", ha = "center", va = "bottom",
            path_effects = stroke)

def pca_rgb(feats, valid) :
    """
    Reduce a (H, W, D) activation map to a (H, W, 3) RGB image via PCA on the valid pixels.

    Standardises the activations, fits PCA(3) on valid pixels only, then maps each component to
    [0, 1] by its own robust percentiles. Fitting on the zoom window (rather than a wider area) is
    deliberate: it spends the full colour range on the structure actually being shown.

    Args:
    - feats (np.ndarray): (H, W, D) activations.
    - valid (np.ndarray): (H, W) bool, True where the pixel is real data.

    Returns:
    - rgb (np.ndarray): (H, W, 3) float32 in [0, 1], neutral grey where invalid.
    - explained (np.ndarray): (3,) explained-variance ratio.
    """
    H, W, D = feats.shape
    flat = feats.reshape(-1, D)
    m = valid.reshape(-1)
    if m.sum() < 100 :
        raise ValueError(f"Too few valid pixels for PCA ({int(m.sum())}).")

    scaler = StandardScaler().fit(flat[m])
    pca = PCA(n_components = 3, random_state = 0).fit(scaler.transform(flat[m]))
    proj = pca.transform(scaler.transform(flat[m]))  # sklearn svd_flip -> deterministic signs

    rgb = np.full((H * W, 3), BG, dtype = np.float32)
    for c in range(3) :
        lo, hi = np.percentile(proj[:, c], PCT)
        rgb[m, c] = np.clip((proj[:, c] - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb.reshape(H, W, 3), pca.explained_variance_ratio_


def zoom_slice(shape, zoom_px, offset) :
    """
    Top-left of the zoom window inside an extracted crop.

    Args:
    - shape (tuple): (H, W) of the extracted crop.
    - zoom_px (int): size of the zoom window.
    - offset (tuple or None): explicit (row, col), or None to centre it.

    Returns:
    - (int, int, int): row, col, and the (possibly reduced) zoom size.
    """
    H, W = shape
    z = min(zoom_px, H, W)
    if offset is None :
        return (H - z) // 2, (W - z) // 2, z
    r, c = offset
    return max(0, min(r, H - z)), max(0, min(c, W - z)), z


def load_s2(path, r, c, z) :
    """Read the true-colour crop and slice the same zoom window; returns (z, z, 3) in [0, 1]."""
    with rs.open(path) as s :
        a = s.read().astype(np.float32)
    a = np.transpose(a, (1, 2, 0)) / 255.0
    return np.clip(a[r:r + z, c:c + z], 0, 1)


###################################################################################################
# Figure

def make_figure(features, out_path, zoom_px, dpi) :
    """
    Compose and save the figure.

    Args:
    - features (str): directory holding the per-tile arrays from extract_features.py.
    - out_path (str): output path WITHOUT extension; .pdf and .png are both written.
    - zoom_px (int): size of the zoom window, in 10 m pixels.
    - dpi (int): resolution for the raster output.
    """
    nrows, ncols = len(TILES), len(COLS)
    fig = plt.figure(figsize = (3.2 * ncols, 3.2 * nrows + 0.6))
    gs = gridspec.GridSpec(nrows, ncols, wspace = 0.04, hspace = 0.08,
                           left = 0.06, right = 0.98, top = 0.90, bottom = 0.03)

    missing = []
    for r_i, spec in enumerate(TILES) :
        tile = spec["tile"]

        # The zoom window is chosen once per row, from the AEF activation grid, and then applied
        # identically to every panel -- all three share the extracted crop's extent, so the same
        # pixel window is the same ground.
        aef_f = join(features, f"{tile}_aef_feat.npy")
        if exists(aef_f) :
            shape = np.load(aef_f, mmap_mode = "r").shape[:2]
            zr, zc, z = zoom_slice(shape, zoom_px, spec["offset"])
        else :
            zr = zc = 0 ; z = zoom_px

        for c_i, col in enumerate(COLS) :
            ax = fig.add_subplot(gs[r_i, c_i])
            ax.set_xticks([]) ; ax.set_yticks([])

            if col["key"] == "s2" :
                path = join(features, f"{tile}_s2rgb.tif")
                ok = exists(path)
                if ok :
                    ax.imshow(load_s2(path, zr, zc, z), interpolation = "nearest")
                    # One scale bar per row, on the Sentinel-2 panel: every column shares this row's
                    # ground extent (z px at 10 m), so it states the scale for the whole row.
                    add_scalebar(ax, z)
            else :
                fp = join(features, f"{tile}_{col['key']}_feat.npy")
                vp = join(features, f"{tile}_{col['key']}_valid.npy")
                ok = exists(fp) and exists(vp)
                if ok :
                    feats = np.load(fp)[zr:zr + z, zc:zc + z]
                    valid = np.load(vp)[zr:zr + z, zc:zc + z]
                    rgb, ev = pca_rgb(feats, valid)
                    ax.imshow(rgb, interpolation = "nearest")
                    print(f"{tile:6s} {col['key']:4s} explained variance {ev.round(3).tolist()}")

            if not ok :
                missing.append(f"{tile}/{col['key']}")
                ax.text(0.5, 0.5, "not found", ha = "center", va = "center",
                        fontsize = 9, color = "0.5", transform = ax.transAxes)
                ax.set_facecolor("0.95")

            if r_i == 0 :
                ax.set_title(col["label"], fontsize = 11, fontweight = "bold", pad = 8)
            if c_i == 0 :
                # Region and tile on ONE line now (was two), and the window size is shown by the
                # scale bar on the S2 panel rather than spelled out as "1.3 km" text here.
                ax.set_ylabel(f'{spec["region"]} ({tile})',
                              fontsize = 11, fontweight = "bold", labelpad = 8)

    makedirs(dirname(abspath(out_path)), exist_ok = True)
    for ext in ("pdf", "png") :
        fig.savefig(f"{out_path}.{ext}", dpi = dpi, bbox_inches = "tight")
        print(f"Saved {out_path}.{ext}")
    plt.close(fig)

    if missing :
        print(f"\nWARNING: {len(missing)} panel(s) missing (drawn as placeholders):")
        for m in missing : print(f"  {m}")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    here = dirname(abspath(__file__))
    parser.add_argument("--features", type = str, default = join(here, "features"))
    parser.add_argument("--out", type = str, default = join(here, "plots", "feature_pca_AEF_vs_AGBD-features"))
    parser.add_argument("--zoom_px", type = int, default = 128, help = "Zoom window size in 10 m px (128 = 1.28 km).")
    parser.add_argument("--dpi", type = int, default = 300)
    args = parser.parse_args()
    make_figure(args.features, args.out, args.zoom_px, args.dpi)
