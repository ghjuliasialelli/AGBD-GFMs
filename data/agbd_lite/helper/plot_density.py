"""

Density scatter (2D-histogram) of predicted vs. GEDI-reference AGB, one panel per model, on the
full AGBD test set. Reproduces the two-panel `manuscript/imgs/density.png` (AGBD features, AEF) and
generalises it to an arbitrary number of models so a third series - e.g. the SSL4EO-MoCo GFM - can
be shown in the same figure with an identical, shared colour scale.

Each model is `--model PATH TITLE`, repeated. Every results file must contain per-sample
`predictions` and `labels` datasets (raw AGB, Mg/ha). Panels share one LogNorm colour scale and one
colourbar, so densities are comparable across panels. Each panel is annotated with Pearson r,
R^2 (coefficient of determination, 1 - SS_res/SS_tot), RMSE and mean bias, computed on ALL finite
pairs (not only those inside the plotted 0-max window).

Run:  python plot_density.py \
          --model ../eval/results/nico_film_59620098-..._nooverlap.h5 AGBD \
          --model ../eval/results/nico_film_59620113-..._nooverlap.h5 AEF \
          --model ../eval/results/ssl4eo_moco_..._agbd_test.h5 SSL4EO-MoCo \
          --out ../../../manuscript/imgs/density.png

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from os import makedirs
from os.path import dirname, abspath


###################################################################################################
# Helpers

def parse_arguments():
    parser = argparse.ArgumentParser(description='Density scatter of predicted vs reference AGB, N models.')
    parser.add_argument('--model', action='append', nargs='+', metavar=('PATH TITLE', ''), required=True,
                        help='A model as: PATH TITLE (TITLE may contain spaces if quoted). Repeat per panel.')
    parser.add_argument('--max_agb', type=float, default=500.0, help='Axis upper limit (Mg/ha).')
    parser.add_argument('--bins', type=int, default=150, help='Number of 2D-histogram bins per axis.')
    parser.add_argument('--out', type=str, required=True, help='Output image path.')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI.')
    args = parser.parse_args()

    models = []
    for spec in args.model:
        if len(spec) < 2:
            parser.error(f'--model needs PATH and TITLE, got {spec}')
        models.append({'path': spec[0], 'title': ' '.join(spec[1:])})
    return models, args.max_agb, args.bins, args.out, args.dpi


def load(path):
    with h5py.File(path, 'r') as f:
        p = f['predictions'][:].astype(np.float64)
        l = f['labels'][:].astype(np.float64)
    m = np.isfinite(p) & np.isfinite(l)
    return p[m], l[m]


def stats(preds, labels):
    r = np.corrcoef(preds, labels)[0, 1]
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    bias = np.mean(preds - labels)
    return r, r2, rmse, bias


###################################################################################################
# Code execution

if __name__ == "__main__":

    models, max_agb, nbins, out_path, dpi = parse_arguments()
    edges = np.linspace(0, max_agb, nbins + 1)

    # First pass: load, compute stats, histogram, and track the global max count for a shared scale.
    vmax = 1
    for m in models:
        print(f"Loading {m['title']} from {m['path']}")
        preds, labels = load(m['path'])
        m['stats'] = stats(preds, labels)
        H, _, _ = np.histogram2d(labels, preds, bins=[edges, edges])   # x=reference, y=predicted
        m['H'] = H
        vmax = max(vmax, H.max())
        r, r2, rmse, bias = m['stats']
        print(f"  {m['title']}: N={len(preds):,}  r={r:.3f}  R2={r2:.3f}  RMSE={rmse:.1f}  bias={bias:+.1f}")

    norm = LogNorm(vmin=1, vmax=vmax)
    n = len(models)

    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 1.2, 6.2), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        # imshow of the transposed histogram so rows->y (predicted), cols->x (reference)
        mesh = ax.imshow(m['H'].T, origin='lower', extent=[0, max_agb, 0, max_agb],
                         aspect='equal', cmap='Greens', norm=norm, interpolation='nearest')
        ax.plot([0, max_agb], [0, max_agb], 'k--', linewidth=1.5)           # 1:1 line
        ax.grid(True, color='white', linewidth=0.8)
        ax.set_axisbelow(False)
        ax.set_xlim(0, max_agb)
        ax.set_ylim(0, max_agb)
        ax.set_title(m['title'])
        ax.set_xlabel('GEDI reference AGBD [Mg/ha]')

        r, r2, rmse, bias = m['stats']
        txt = f"r={r:.3f}\nR$^2$={r2:.3f}\nRMSE={rmse:.1f}\nbias={bias:.1f}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='0.6'))

    axes[0].set_ylabel('Predicted AGB [Mg/ha]')

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.046 / n * 2, pad=0.02)
    cbar.set_label('Number of samples')

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
