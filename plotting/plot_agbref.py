"""

AGBref comparison figure (manuscript Fig. `fig:agbref`, imgs/agbref.png) in the style of plotting/img/binned.png:
`fcn_film` on AEF ("Ours") and ESA CCI against AGBref, on the per-plot CSV of the *All* configuration
(666 plots). Same palette, despined axes, alternating bin bands, linear distribution panel with K/M tick
labels, and best metric values in bold.

Residuals are binned by AGBref value in 50 Mg/ha bins. As in binned.png, the boxes hide outlier points
(whiskers at 1.5 IQR); they stay in the scatters and in every metric. Above 300 Mg/ha there are only 12 plots (0 in
350-400, 1 in 400-450), so those are merged into one `300+` bin; --max_bin moves that cut.

Variants (--variant):
  residuals   binned-residual boxplots (AEF vs CCI side by side) + AGBref distribution panel.
              The direct analogue of binned.png; RMSE and R^2 in the legend.
  points      as `residuals`, with every plot drawn as a jittered dot over its box, since several
              bins hold only 11-22 plots and a box alone hides that.
  scatter     a top row of per-plot scatter panels (one per product, all five table metrics with the
              better value in bold) and the distribution panel, above full-width residual boxes.
              Keeps the current figure's content (scatter + binned).

Run (from the repo root):
      python plotting/plot_agbref.py --variant scatter --out plotting/img/agbref_scatter.png

"""

###################################################################################################
# Imports

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_binned_models import DEFAULT_COLORS, _darken, _median_color

HERE = os.path.dirname(os.path.abspath(__file__))
PERPLOT = os.path.join(HERE, '..', 'comparison', 'agbref', 'results', 'perplot_notrain_skip_jap_mean_max500.csv')

# Same colours as binned.png: AEF #117733, ESA CCI #CC6677 (DEFAULT_COLORS order: AGBD, AEF, GFM, AEF+, CCI)
PRODUCTS = [('nico_mean', 'Ours (fcn_film $\\times$ AEF)', DEFAULT_COLORS[1]),
            ('cci_mean', 'ESA CCI', DEFAULT_COLORS[4])]


###################################################################################################
# Helpers

def parse_arguments():
    parser = argparse.ArgumentParser(description='AGBref comparison figure in the binned.png style.')
    parser.add_argument('--variant', choices=['residuals', 'points', 'scatter'], default='scatter')
    parser.add_argument('--bin_size', type=int, default=50, help='AGBref bin width (Mg/ha).')
    parser.add_argument('--max_bin', type=int, default=300, help='Plots at or above this are merged into one bin.')
    parser.add_argument('--out', type=str, required=True, help='Output image path.')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI.')
    args = parser.parse_args()
    return args.variant, args.bin_size, args.max_bin, args.out, args.dpi


def load():
    rows = list(csv.DictReader(open(PERPLOT)))
    ref = np.array([float(r['agbref']) for r in rows])
    products = []
    for col, label, color in PRODUCTS:
        pred = np.array([float(r[col]) for r in rows])
        valid = np.isfinite(pred) & np.isfinite(ref)
        assert valid.sum() > 0, f'{col}: no valid plots'
        products.append({'label': label, 'color': color, 'pred': pred[valid], 'ref': ref[valid]})
    return ref, products


def metrics(pred, ref):
    res = pred - ref
    return {'r': float(np.corrcoef(pred, ref)[0, 1]),
            'R$^2$': float(1 - np.sum(res ** 2) / np.sum((ref - ref.mean()) ** 2)),
            'RMSE': float(np.sqrt(np.mean(res ** 2))),
            'MAE': float(np.mean(np.abs(res))),
            'ME': float(np.mean(res))}


# How each metric is printed, and which value is best: higher r/R^2, lower RMSE/MAE, |ME| closest to 0
FORMATS = {'r': '.3f', 'R$^2$': '.3f', 'RMSE': '.1f', 'MAE': '.1f', 'ME': '.1f'}
BEST = {'r': max, 'R$^2$': max, 'RMSE': min, 'MAE': min, 'ME': lambda v: min(v, key=abs)}


def fmt_metric(name, value, all_values):
    """Format a metric as printed, in bold (mathtext) if it is the best printed value across products."""
    txt = format(value, FORMATS[name])
    best = BEST[name]([float(format(v, FORMATS[name])) for v in all_values])
    return f'$\\mathbf{{{txt}}}$' if float(txt) == best else txt


def bin_edges(bin_size, max_bin):
    lbs = list(range(0, max_bin, bin_size)) + [max_bin]
    ubs = list(range(bin_size, max_bin + 1, bin_size)) + [np.inf]
    labels = [f'{lb}-{ub}' for lb, ub in zip(lbs[:-1], ubs[:-1])] + [f'{max_bin}+']
    return lbs, ubs, labels


def residual_row(ax, hax, ref, products, bin_size, max_bin, with_points, fs):
    lbs, ubs, labels_x = bin_edges(bin_size, max_bin)
    n_bins, n_prod = len(lbs), len(products)
    for p in products:
        res = p['pred'] - p['ref']
        p['binned'] = [res[(p['ref'] >= lb) & (p['ref'] < ub)] for lb, ub in zip(lbs, ubs)]
    counts = np.array([((ref >= lb) & (ref < ub)).sum() for lb, ub in zip(lbs, ubs)])
    print('  plots per bin:', dict(zip(labels_x, counts.tolist())))

    width = 0.8 / n_prod
    offsets = (np.arange(n_prod) - (n_prod - 1) / 2.0) * width
    for i in range(n_bins):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='0.94', zorder=0)

    rng = np.random.default_rng(0)
    handles = []
    for p, off in zip(products, offsets):
        edge = _darken(p['color'])
        pos = [i + off for i in range(n_bins)]
        bp = ax.boxplot(p['binned'], positions=pos, widths=width * 0.82, patch_artist=True, showfliers=False,
                        zorder=3, boxprops=dict(facecolor=p['color'], edgecolor=edge, linewidth=1.1,
                                                alpha=0.55 if with_points else 1.0),
                        whiskerprops=dict(color=edge, linewidth=1.1), capprops=dict(color=edge, linewidth=1.1),
                        medianprops=dict(color=_median_color(p['color']) if not with_points else 'black', linewidth=1.6),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=p['color'], markeredgecolor=edge,
                                        alpha=0.6))
        handles.append(bp['boxes'][0])
        if with_points:
            for x, vals in zip(pos, p['binned']):
                ax.scatter(x + rng.uniform(-width * 0.3, width * 0.3, len(vals)), vals, s=7, color=edge,
                           alpha=0.55, linewidths=0, zorder=4)

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(labels_x)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_xlabel('AGBref bins ($Mg/ha$)', fontsize=fs)
    ax.set_ylabel('Residuals ($AGB_{pred} - AGB_{AGBref}$)', fontsize=fs)
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, zorder=2)
    ax.yaxis.grid(True, color='0.85', linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ms = [p['metrics'] for p in products]
    ax.legend(handles, [f"{p['label']}  (RMSE={fmt_metric('RMSE', p['metrics']['RMSE'], [m['RMSE'] for m in ms])}, "
                        f"R$^2$={fmt_metric('R$^2$', p['metrics']['R$^2$'], [m['R$^2$'] for m in ms])})"
                        for p in products],
              frameon=True, framealpha=0.95, edgecolor='0.8', loc='lower left')

    hax.bar(range(n_bins), counts, width=0.85, color='0.6', edgecolor='white', linewidth=0.8)
    hax.set_xticks(range(n_bins))
    hax.set_xticklabels(labels_x, rotation=45, ha='right')
    hax.set_xlim(-0.5, n_bins - 0.5)
    hax.set_xlabel('AGBref bins ($Mg/ha$)', fontsize=fs)
    hax.set_ylabel('AGBref plots', fontsize=fs)
    hax.set_title('AGBref distribution', fontsize=fs)
    hax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: '0' if v == 0 else (f'{v / 1e6:g}M' if v >= 1e6 else (f'{v / 1e3:g}K' if v >= 1e3 else f'{v:g}'))))
    hax.yaxis.grid(True, color='0.85', linewidth=0.7)
    hax.set_axisbelow(True)
    for spine in ('top', 'right'):
        hax.spines[spine].set_visible(False)


def scatter_row(axes, products, lim, fs):
    ms = [p['metrics'] for p in products]
    for ax, p in zip(axes, products):
        ax.scatter(p['ref'], p['pred'], s=12, color=p['color'], edgecolor=_darken(p['color']), linewidths=0.3,
                   alpha=0.6, zorder=3)
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.6, zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.set_title(p['label'], fontsize=fs)
        ax.set_xlabel('AGBref ($Mg/ha$)', fontsize=fs)
        ax.grid(True, color='0.9', linewidth=0.7, zorder=0)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        txt = '\n'.join(f"{k}={fmt_metric(k, p['metrics'][k], [m[k] for m in ms])}" for k in FORMATS)
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left', fontsize=fs * 0.85, linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='0.8'))
    axes[0].set_ylabel('Predicted AGB ($Mg/ha$)', fontsize=fs)


###################################################################################################
# Code execution

if __name__ == '__main__':

    variant, bin_size, max_bin, out_path, dpi = parse_arguments()
    ref, products = load()
    print(f'loaded {len(ref)} plots')
    for p in products:
        p['metrics'] = metrics(p['pred'], p['ref'])
        print(f"  {p['label']}: " + '  '.join(f"{k}={v:{FORMATS[k]}}" for k, v in p['metrics'].items()))

    plt.rcParams.update({'font.size': 13})
    fs = 14

    if variant == 'scatter':
        # Top: the two scatters plus the distribution panel, all square; bottom: residual boxes, full width
        fig = plt.figure(figsize=(18, 12.5))
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.95])
        saxes = top.subplots(1, 3, gridspec_kw={'wspace': 0.3})
        scatter_row(saxes[:2], products, lim=450, fs=fs)
        saxes[2].set_box_aspect(1)
        ax = bottom.subplots(1, 1)
        residual_row(ax, saxes[2], ref, products, bin_size, max_bin, with_points=False, fs=fs)
    else:
        fig, (ax, hax) = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={'width_ratios': [2.7, 1], 'wspace': 0.18})
        residual_row(ax, hax, ref, products, bin_size, max_bin, with_points=(variant == 'points'), fs=fs)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
