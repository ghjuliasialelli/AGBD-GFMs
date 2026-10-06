"""

Predicted vs. GEDI-reference AGB for N models on the full AGBD test set (2,807,977 samples).

The published figure (manuscript/imgs/density.png, command below) is `--style binned --overlay --lines`:
left, the median prediction per 25 Mg/ha reference bin for every model with a shaded 25th-75th
percentile band; right, the reference-AGBD histogram on the same bins (linear), showing the long tail.
It replaces both the earlier per-model density panels and the binned-residual boxplot (binned.png).

Each model is `--model PATH TITLE`, repeated. Every results file must contain per-sample
`predictions` and `labels` datasets (raw AGB, Mg/ha). r, R^2 (1 - SS_res/SS_tot), RMSE and mean bias
are computed on ALL finite pairs (not only those inside the plotted 0-max window).

Run (from the repo root; this is the published manuscript/imgs/density.png):
      python plotting/plot_density.py \
          --model model/eval/results/nico_film_59620098-1_59620098-1_59620098-1_2019-2020_nooverlap.h5 AGBD features \
          --model model/eval/results/nico_film_59620113-1_59620113-1_59620113-1_2019-2020_nooverlap.h5 AEF \
          --model benchmark_pangaea/results/ssl4eo_moco_20260316_155017_agbd_test.h5 SSL4EO-MoCo \
          --style binned --overlay --lines --box_fontsize 20 \
          --out manuscript/imgs/density.png

Display options (with none of them, the script draws the earlier per-model 2D-histogram panels):
  --norm column     normalise each reference-AGB column to sum to 1, i.e. show P(pred | ref). Removes
                    the dominance of the low-AGB samples so the high-AGB behaviour is readable.
  --quantiles       overlay the conditional median (solid) and 10th/90th percentiles (dotted) of the
                    prediction per reference bin of width --q_width.
  --diff_vs I       panels other than model I show (this model - model I) on a diverging scale, using
                    the column-normalised histograms; panel I is shown as usual.
  --style binned    instead of a 2D histogram, plot the mean prediction per 25 Mg/ha reference bin with
                    +-1 std error bars and marker size by bin count, as in the bottom row of
                    manuscript/imgs/agbref.png (plotting/regen_agbref_fig.py).
  --overlay         with --style binned: all models in a single panel, coloured, slightly offset in x.

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
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
    parser.add_argument('--norm', choices=['count', 'column'], default='count',
                        help='count: raw counts; column: each reference bin sums to 1, i.e. P(pred | ref).')
    parser.add_argument('--quantiles', action='store_true', help='Overlay conditional median and 10/90th percentiles.')
    parser.add_argument('--q_width', type=float, default=20.0, help='Reference-bin width (Mg/ha) for --quantiles.')
    parser.add_argument('--diff_vs', type=int, default=None, help='Index of the model the others are differenced against.')
    parser.add_argument('--style', choices=['density', 'binned'], default='density',
                        help='density: 2D histogram; binned: mean +- std per reference bin (agbref.png style).')
    parser.add_argument('--bin_width', type=float, default=25.0, help='Reference-bin width (Mg/ha) for --style binned.')
    parser.add_argument('--overlay', action='store_true', help='With --style binned: all models in one panel.')
    parser.add_argument('--y', choices=['pred', 'residual', 'diff'], default='pred',
                        help='With --lines. pred: median prediction per bin; residual: median (pred - ref) per bin; '
                             'diff: median of model 2 minus median of model 1 per bin, with a paired bootstrap 95%% CI '
                             '(needs exactly 2 models scored on the same samples).')
    parser.add_argument('--n_boot', type=int, default=200, help='Bootstrap resamples for --y diff.')
    parser.add_argument('--lines', action='store_true',
                        help='With --overlay: median line + shaded interquartile band per model, no error bars.')
    parser.add_argument('--box_fontsize', type=float, default=15, help='Font size of the metrics box.')
    args = parser.parse_args()

    models = []
    for spec in args.model:
        if len(spec) < 2:
            parser.error(f'--model needs PATH and TITLE, got {spec}')
        models.append({'path': spec[0], 'title': ' '.join(spec[1:])})
    return models, args.max_agb, args.bins, args.out, args.dpi, args.norm, args.quantiles, args.q_width, \
           args.diff_vs, args.box_fontsize, args.style, args.bin_width, args.overlay, args.lines, args.y, args.n_boot


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


def column_normalise(H):
    # H is indexed [reference, predicted]; each reference bin sums to 1 (empty bins stay 0)
    tot = H.sum(axis=1, keepdims=True)
    return np.divide(H, tot, out=np.zeros_like(H), where=tot > 0)


def conditional_quantiles(preds, labels, max_agb, width, pcts=(10, 50, 90)):
    edges = np.arange(0, max_agb + width, width)
    centres, q = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (labels >= lo) & (labels < hi)
        if sel.sum() < 50: continue
        centres.append((lo + hi) / 2)
        q.append(np.percentile(preds[sel], pcts))
    return np.array(centres), np.array(q)


# Marker size by samples per bin: the agbref.png classes, scaled from its 666 plots to the 2.8M-sample
# test set (bins hold 8.9k-1.4M samples at 25 Mg/ha).
SIZE_THRESHOLDS = [10e3, 20e3, 50e3, 100e3, 500e3]
SIZE_VALUES = [10, 25, 50, 90, 140, 200]
SIZE_LABELS = ['< 10k', '10k–20k', '20k–50k', '50k–100k', '100k–500k', '> 500k']
OVERLAY_COLORS = ['#0072B2', '#E69F00', '#009E73']      # same as plot_binned_models.py / binned.png


def marker_size(c):
    for th, s in zip(SIZE_THRESHOLDS, SIZE_VALUES):
        if c < th: return s
    return SIZE_VALUES[-1]


def binned_mean_std(preds, labels, max_agb, width):
    edges = np.arange(0, max_agb + width, width)
    bc, bm, bs, bn = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (labels >= lo) & (labels < hi)
        if sel.sum() == 0: continue
        bc.append((lo + hi) / 2); bm.append(preds[sel].mean()); bs.append(preds[sel].std()); bn.append(sel.sum())
    return np.array(bc), np.array(bm), np.array(bs), np.array(bn)


def metrics_box(ax, stats_, fontsize, **pos):
    r, r2, rmse, bias = stats_
    txt = f"r={r:.3f}\nR$^2$={r2:.3f}\nRMSE={rmse:.1f}\nbias={bias:.1f}"
    ax.text(pos.get('x', 0.04), pos.get('y', 0.96), txt, transform=ax.transAxes, va=pos.get('va', 'top'),
            ha=pos.get('ha', 'left'), fontsize=fontsize, linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.9, edgecolor='0.6'))


def plot_median_diff(ax, models, edges, n_boot, box_fs, seed=0):
    # Median of model 2 minus median of model 1 per reference bin, with a paired bootstrap 95% CI: both
    # models are resampled with the same indices, which requires them to be scored on the same samples.
    if len(models) != 2:
        raise SystemExit('--y diff needs exactly 2 models')
    a, b = models
    if not np.array_equal(a['labels'], b['labels']):
        raise SystemExit('--y diff needs both models scored on the same samples in the same order')
    rng = np.random.default_rng(seed)
    centres, d, lo, hi = [], [], [], []
    for l0, l1 in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero((a['labels'] >= l0) & (a['labels'] < l1))
        if sel.size < 50: continue
        pa, pb = a['preds'][sel], b['preds'][sel]
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, sel.size, sel.size)
            boot[i] = np.median(pb[idx]) - np.median(pa[idx])
        centres.append((l0 + l1) / 2); d.append(np.median(pb) - np.median(pa))
        lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
        print(f'  {l0:4.0f}-{l1:4.0f}: diff={d[-1]:+6.2f}  95% CI [{lo[-1]:+6.2f}, {hi[-1]:+6.2f}]  n={sel.size:,}')
    col = OVERLAY_COLORS[1]
    ax.fill_between(centres, lo, hi, color=col, alpha=0.3, linewidth=0, label='95% bootstrap CI')
    ax.plot(centres, d, color=col, linewidth=2.6, marker='o', markersize=4.5, label='difference of medians')
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='no difference')
    ax.set_ylabel(f"{b['title']} $-$ {a['title']}, median prediction [Mg/ha]")
    ax.legend(loc='upper left', fontsize=box_fs * 0.72, framealpha=0.95)


def plot_binned(models, max_agb, width, overlay, box_fs, out_path, dpi, lines=False, y_mode='pred', n_boot=200):
    plt.rcParams.update({'font.size': 15})
    for m in models:
        m['b'] = binned_mean_std(m['preds'], m['labels'], max_agb, width)

    if overlay and lines:
        # Main panel plus, to its right, the reference-AGBD histogram on the same bins (linear counts) to show
        # the long tail: every model is scored on the same labels, so one histogram serves all.
        fig, (ax, hax) = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={'wspace': 0.25})
        hax.set_box_aspect(1)   # same height as the square main panel
        edges = np.arange(0, max_agb + width, width)
        counts, _ = np.histogram(models[0]['labels'], bins=edges)
        hax.bar(edges[:-1], counts / 1e3, width=width, align='edge', color='0.6', edgecolor='white', linewidth=0.8)
        hax.set_xlim(0, max_agb); hax.set_ylim(0, counts.max() / 1e3 * 1.05)
        hax.set_xlabel('GEDI reference AGBD [Mg/ha]'); hax.set_ylabel('Test samples [thousands]')
        hax.set_title('Reference AGBD distribution', fontsize=box_fs * 0.75)
        hax.grid(True, axis='y', color='0.9', linewidth=0.8); hax.set_axisbelow(True)
        for sp in ('top', 'right'): hax.spines[sp].set_visible(False)
        print('  reference samples per bin:', counts.tolist())
        if y_mode == 'diff':
            plot_median_diff(ax, models, edges, n_boot, box_fs)
        else:
            for m, col in zip(models, OVERLAY_COLORS):
                y = m['preds'] - m['labels'] if y_mode == 'residual' else m['preds']
                c, q = conditional_quantiles(y, m['labels'], max_agb, width, pcts=[25, 50, 75])
                ax.fill_between(c, q[:, 0], q[:, 2], color=col, alpha=0.18, linewidth=0)
                ax.plot(c, q[:, 1], color=col, linewidth=2.6, marker='o', markersize=4.5,
                        label=f"{m['title']}  (RMSE={m['stats'][2]:.1f}, R$^2$={m['stats'][1]:.3f})")
            if y_mode == 'residual':
                ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='no bias')
                ax.set_ylabel('Prediction $-$ reference [Mg/ha]')
                loc = 'lower left'
            else:
                ax.plot([0, max_agb], [0, max_agb], 'k--', linewidth=1, alpha=0.5, label='1:1')
                ax.set_ylim(0, max_agb); ax.set_ylabel('Predicted AGB [Mg/ha]')
                loc = 'upper left'
            ax.legend(loc=loc, fontsize=box_fs * 0.72, framealpha=0.95,
                      title='median, shaded: 25th–75th pct.', title_fontsize=box_fs * 0.62)
        ax.set_xlim(0, max_agb); ax.set_box_aspect(1)
        ax.set_xlabel('GEDI reference AGBD [Mg/ha]')
        ax.grid(True, color='0.9', linewidth=0.8); ax.set_axisbelow(True)
        makedirs(dirname(abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f'\nSaved {out_path}')
        return

    if overlay:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 8))
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width * 0.22
        for m, col, dx in zip(models, OVERLAY_COLORS, offsets):
            bc, bm, bs, bn = m['b']
            ax.errorbar(bc + dx, bm, yerr=bs, fmt='none', ecolor=col, elinewidth=1.2, capsize=2, alpha=0.8, zorder=1)
            ax.scatter(bc + dx, bm, s=[marker_size(n) for n in bn], color=col, edgecolor='black', linewidth=0.4,
                       zorder=2)
            ax.plot([], [], 'o', color=col, markersize=9,
                    label=f"{m['title']}  (RMSE={m['stats'][2]:.1f}, r={m['stats'][0]:.3f})")
        ax.plot([0, max_agb], [0, max_agb], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_agb); ax.set_ylim(0, max_agb); ax.set_aspect('equal')
        ax.set_xlabel('GEDI reference AGBD [Mg/ha]'); ax.set_ylabel('Predicted AGB [Mg/ha]')
        leg1 = ax.legend(loc='upper left', fontsize=box_fs * 0.75, framealpha=0.9)
        ax.add_artist(leg1)
    else:
        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, models):
            bc, bm, bs, bn = m['b']
            ax.errorbar(bc, bm, yerr=bs, fmt='none', ecolor='black', elinewidth=1, capsize=2, zorder=1, alpha=0.6)
            ax.scatter(bc, bm, s=[marker_size(c) for c in bn], c='black', zorder=2)
            ax.plot([0, max_agb], [0, max_agb], 'k--', linewidth=1, alpha=0.5)
            ax.set_xlim(0, max_agb); ax.set_ylim(0, max_agb); ax.set_aspect('equal')
            ax.set_title(m['title']); ax.set_xlabel('GEDI reference AGBD [Mg/ha]')
            # all bins sit at or below the 1:1 line, so the top left is free for the metrics
            metrics_box(ax, m['stats'], box_fs)
        axes[0].set_ylabel('Predicted AGB [Mg/ha]')

    plt.tight_layout()

    # The size legend goes under the panels in one row: inside a panel it hides the high-AGB error bars.
    handles = [plt.scatter([], [], s=s, c='gray', label=l) for s, l in zip(SIZE_VALUES, SIZE_LABELS)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    bottom = min(a.get_tightbbox(renderer).transformed(to_fig).y0 for a in fig.axes)
    fig.legend(handles=handles, title='samples per bin', loc='upper center', bbox_to_anchor=(0.5, bottom - 0.01),
               ncol=len(handles), fontsize=box_fs * 0.65, title_fontsize=box_fs * 0.7, framealpha=0.9,
               columnspacing=1.2, handletextpad=0.4)

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')


###################################################################################################
# Code execution

if __name__ == "__main__":

    models, max_agb, nbins, out_path, dpi, norm_mode, quantiles, q_width, diff_vs, box_fs, style, bin_width, \
        overlay, lines, y_mode, n_boot = parse_arguments()
    edges = np.linspace(0, max_agb, nbins + 1)

    # First pass: load, compute stats, histogram, and track the global max count for a shared scale.
    vmax = 1
    for m in models:
        print(f"Loading {m['title']} from {m['path']}")
        preds, labels = load(m['path'])
        m['stats'] = stats(preds, labels)
        if style == 'binned':
            m['preds'], m['labels'] = preds, labels
            r, r2, rmse, bias = m['stats']
            print(f"  {m['title']}: N={len(preds):,}  r={r:.3f}  R2={r2:.3f}  RMSE={rmse:.1f}  bias={bias:+.1f}")
            continue
        H, _, _ = np.histogram2d(labels, preds, bins=[edges, edges])   # x=reference, y=predicted
        m['H'] = column_normalise(H) if norm_mode == 'column' else H
        if quantiles: m['q'] = conditional_quantiles(preds, labels, max_agb, q_width)
        vmax = max(vmax, m['H'].max())
        r, r2, rmse, bias = m['stats']
        print(f"  {m['title']}: N={len(preds):,}  r={r:.3f}  R2={r2:.3f}  RMSE={rmse:.1f}  bias={bias:+.1f}")

    if style == 'binned':
        plot_binned(models, max_agb, bin_width, overlay, box_fs, out_path, dpi, lines, y_mode, n_boot)
        raise SystemExit

    if norm_mode == 'column':
        vmin, vmax = 1e-4, max(m['H'].max() for m in models)
        cbar_label = 'Fraction of samples per reference bin'
    else:
        vmin, cbar_label = 1, 'Number of samples'
    norm = LogNorm(vmin=vmin, vmax=vmax)
    n = len(models)

    if diff_vs is not None:
        ref_C = column_normalise(models[diff_vs]['H']) if norm_mode == 'count' else models[diff_vs]['H']
        for i, m in enumerate(models):
            if i == diff_vs: continue
            C = column_normalise(m['H']) if norm_mode == 'count' else m['H']
            m['D'] = C - ref_C
        dmax = max(np.abs(m['D']).max() for m in models if 'D' in m)
        dnorm = SymLogNorm(linthresh=1e-3, vmin=-dmax, vmax=dmax)

    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 1.2, 6.2), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        # imshow of the transposed histogram so rows->y (predicted), cols->x (reference)
        if 'D' in m:
            dmesh = ax.imshow(m['D'].T, origin='lower', extent=[0, max_agb, 0, max_agb],
                              aspect='equal', cmap='RdBu_r', norm=dnorm, interpolation='nearest')
            ax.set_facecolor('white')
        else:
            mesh = ax.imshow(m['H'].T, origin='lower', extent=[0, max_agb, 0, max_agb],
                             aspect='equal', cmap='Greens', norm=norm, interpolation='nearest')
        ax.plot([0, max_agb], [0, max_agb], 'k--', linewidth=1.5)           # 1:1 line
        if quantiles:
            c, q = m['q']
            ax.plot(c, q[:, 1], color='black', linewidth=2.2, label='median')
            ax.plot(c, q[:, 0], color='black', linewidth=1.5, linestyle=':', label='10th/90th pct.')
            ax.plot(c, q[:, 2], color='black', linewidth=1.5, linestyle=':')
        ax.grid(True, color='white', linewidth=0.8)
        ax.set_axisbelow(False)
        ax.set_xlim(0, max_agb)
        ax.set_ylim(0, max_agb)
        ax.set_title(m['title'] if 'D' not in m else f"{m['title']} $-$ {models[diff_vs]['title']}")
        ax.set_xlabel('GEDI reference AGBD [Mg/ha]')

        r, r2, rmse, bias = m['stats']
        txt = f"r={r:.3f}\nR$^2$={r2:.3f}\nRMSE={rmse:.1f}\nbias={bias:.1f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left', fontsize=box_fs, linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.9, edgecolor='0.6'))

    axes[0].set_ylabel('Predicted AGB [Mg/ha]')

    if quantiles:
        axes[-1].legend(loc='lower right', fontsize=box_fs * 0.8, framealpha=0.9)

    if diff_vs is None:
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.046 / n * 2, pad=0.02)
        cbar.set_label(cbar_label)
    else:
        cbar = fig.colorbar(mesh, ax=axes[diff_vs], fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label)
        dcbar = fig.colorbar(dmesh, ax=[a for i, a in enumerate(axes) if i != diff_vs], fraction=0.046 / n * 2, pad=0.02)
        dcbar.set_label('Difference in fraction per reference bin')

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
