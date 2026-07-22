"""

Quick visual check of an SSL4EO-MoCo prediction against the AEF panel on identical ground.

This is a LOOK-AT-IT script, not a figure for the paper: it exists so a preview run can be judged
before committing hours of inference. Colours and limits follow make_map_figure.py (viridis,
0-400 t/ha, grey nodata) so the eye calibrates the same way it does on the real figure.

The AEF panel is read through `from_bounds` on the SSL4EO raster's own bounds, so the two panels
are the same ground even though they are on different grids (30 m vs 10 m) -- the extent comes from
`src.bounds`, never from a filename.

Usage (agbd env):
    python comparison/maps/preview_ssl4eo.py \
        --pred /scratch3/gsialelli/ssl4eo_maps/diag/32TPT_preview.tif \
        --cache /scratch3/gsialelli/ssl4eo_maps/cache/32TPT.npz

"""

import argparse
import json
from os.path import join, dirname, abspath, exists
from os import makedirs

import numpy as np
import rasterio as rs
from rasterio.windows import from_bounds

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED_AEF = ('/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/'
            'nico_film/59620113-1_59620113-1_59620113-1')

# make_map_figure.py conventions.
VMIN, VMAX = 0, 400
CMAP = 'viridis'
CBAR_LABEL = 'AGB Density [t/ha]'
MASK_COLOR = '0.85'

BAND_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']


def stretch(rgb, pct = (2, 98)) :
    """
    Percentile-stretch an (H, W, 3) float array to [0, 1] for display.

    Args:
    - rgb (np.ndarray): (H, W, 3) reflectance.
    - pct (tuple): low/high percentiles.

    Returns:
    - np.ndarray: (H, W, 3) in [0, 1].
    """
    out = np.zeros_like(rgb, dtype = np.float32)
    for c in range(3) :
        lo, hi = np.percentile(rgb[..., c], pct)
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def main() :

    p = argparse.ArgumentParser()
    p.add_argument('--pred', required = True, help = 'SSL4EO prediction GeoTIFF.')
    p.add_argument('--cache', required = True, help = 'The .npz the prediction was made from.')
    p.add_argument('--out', default = None, help = 'Output PNG (default: alongside --pred).')
    args = p.parse_args()

    out_path = args.out or args.pred.replace('.tif', '.png')

    # SSL4EO prediction --------------------------------------------------------------------------
    with rs.open(args.pred) as src :
        ssl = src.read(1)
        ssl = np.where(ssl == src.nodata, np.nan, ssl)
        bounds, crs, res = src.bounds, src.crs, src.res
        tags = src.tags()
    print(f'SSL4EO: {ssl.shape}, {res[0]:.1f} m, {crs}')
    print(f'  tags: keep_px={tags.get("keep_px")} stride={tags.get("stride")} '
          f'res={tags.get("resolution_m")} m')

    # Sentinel-2 true colour, from the same cache the model read ---------------------------------
    with open(args.cache.replace('.npz', '.json')) as f : side = json.load(f)
    m_left, m_top = side['margins_obtained'][0], side['margins_obtained'][1]
    sr = np.load(args.cache)['sr']
    n = ssl.shape[0] * int(tags.get('stride', 3))
    idx = [BAND_ORDER.index(b) for b in ('B4', 'B3', 'B2')]
    rgb = np.stack([sr[i, m_top : m_top + n, m_left : m_left + n] for i in idx], axis = -1)
    del sr
    rgb = stretch(rgb)

    # AEF panel on the SAME ground ---------------------------------------------------------------
    aef_path = join(PRED_AEF, f'{side["tile"]}.tif')
    aef = None
    if exists(aef_path) :
        with rs.open(aef_path) as src :
            assert src.crs == crs, f'AEF is {src.crs}, prediction is {crs}; bounds are not comparable'
            win = from_bounds(*bounds, transform = src.transform)
            aef = src.read(1, window = win, out_shape = ssl.shape,
                           masked = True).astype(np.float32).filled(np.nan)
        print(f'AEF: same window, resampled {int(win.height)}x{int(win.width)} -> {ssl.shape}')
    else :
        print(f'No AEF panel at {aef_path}; showing two panels only.')

    # Plot ---------------------------------------------------------------------------------------
    panels = [('Sentinel-2 (true colour)', rgb, None),
              (f'SSL4EO-MoCo on AGBD\n{res[0]:.0f} m, centre-pixel only', ssl, CMAP)]
    if aef is not None :
        panels.append(('AEF (nico_film)\n10 m, resampled to match', aef, CMAP))

    fig, axes = plt.subplots(1, len(panels), figsize = (5.2 * len(panels), 5.6))
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(MASK_COLOR)

    for ax, (title, data, cm) in zip(np.atleast_1d(axes), panels) :
        if cm is None :
            ax.imshow(data)
        else :
            im = ax.imshow(data, cmap = cmap, vmin = VMIN, vmax = VMAX, interpolation = 'nearest')
        ax.set_title(title, fontsize = 11)
        ax.set_xticks([]) ; ax.set_yticks([])

    cbar = fig.colorbar(im, ax = list(np.atleast_1d(axes)), fraction = 0.025, pad = 0.02)
    cbar.set_label(CBAR_LABEL)

    km = (bounds[2] - bounds[0]) / 1000
    fig.suptitle(f'{side["tile"]} preview -- {km:.1f} km across -- {side["product"]}', fontsize = 10)

    makedirs(dirname(abspath(out_path)), exist_ok = True)
    fig.savefig(out_path, dpi = 130, bbox_inches = 'tight')
    print(f'Wrote {out_path}')

    # Numbers, so the eye is not the only check --------------------------------------------------
    v = ssl[np.isfinite(ssl)]
    print(f'\nSSL4EO over valid px: mean {v.mean():.1f}, median {np.median(v):.1f}, '
          f'p1 {np.percentile(v, 1):.1f}, p99 {np.percentile(v, 99):.1f}, max {v.max():.1f}')
    print(f'  exact zeros: {100 * (v == 0).mean():.2f}%')
    if aef is not None :
        a = aef[np.isfinite(aef) & np.isfinite(ssl)]
        s = ssl[np.isfinite(aef) & np.isfinite(ssl)]
        print(f'AEF   over same px: mean {a.mean():.1f}, median {np.median(a):.1f}, '
              f'p99 {np.percentile(a, 99):.1f}')
        print(f'  correlation SSL4EO vs AEF: r = {np.corrcoef(s, a)[0, 1]:.3f}  '
              f'(n = {len(a):,})')


if __name__ == '__main__' :
    main()
