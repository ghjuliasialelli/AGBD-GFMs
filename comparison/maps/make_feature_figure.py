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

###################################################################################################
# Configuration

# Same locations (and disclosure caveats) as the quantitative map figure; see make_map_figure.py.
# `offset` is the top-left of the zoom window inside the extracted crop, in px; None = centred.
TILES = [
    {"tile": "59GPM", "region": "Australasia", "offset": None},
    {"tile": "32TNS", "region": "Europe",      "offset": None},
    {"tile": "49SBT", "region": "South Asia",  "offset": None},
]

COLS = [
    {"key": "s2",   "label": "Sentinel-2 (true colour)"},
    {"key": "agbd", "label": "AGBD features\nactivations (PCA $\\rightarrow$ RGB)"},
    {"key": "aef",  "label": "AEF\nactivations (PCA $\\rightarrow$ RGB)"},
]

BG = 0.85          # grey for nodata pixels
PCT = (2, 98)      # robust percentile stretch per PCA component

###################################################################################################
# Helpers

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
                if ok : ax.imshow(load_s2(path, zr, zc, z), interpolation = "nearest")
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
                ax.set_ylabel(f'{spec["region"]}\n({tile})  {z * 10 / 1000:.1f} km',
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
