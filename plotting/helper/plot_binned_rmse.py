"""

Per-bin RMSE for an arbitrary number of models, on the AGBD (full) test set.

Companion to plot_binned_models.py (which shows the residual distribution per bin): this one shows the
RMSE of each model within each reference-AGB bin. Two panels share the x axis:
  - top:    per-bin RMSE (Mg/ha), one grouped bar per model;
  - bottom: per-bin RMSE minus that of a reference model (default: the first --model), so the few-Mg/ha
            differences that are invisible on the top panel's 20-200 Mg/ha scale can be read.
            Below zero = more accurate than the reference in that bin.

Each model is passed as `--model PATH LABEL [COLOR]`, repeated. Every results file must contain per-sample
`predictions` and `labels` datasets (raw AGB, Mg/ha). Residuals are binned by each model's OWN labels.

Run (from the repo root; colours default to DEFAULT_COLORS by --model order):
      R=model/eval/results; f(){ echo $R/nico_film_$1-1_$1-1_$1-1_2019-2020_nooverlap.h5; }
      python plotting/helper/plot_binned_rmse.py \
          --model $(f 64424610) "AEF-Lite" "#332288"            --model $(f 19819522) "OLMO-Lite (128)" "#117733" \
          --model $(f 19870583) "OLMO-Lite (64)" "#88CCEE"      --model $(f 20004081) "AEF+OLMO-Lite (128)" "#44AA99" \
          --model $(f 20004143) "AEF+OLMO-Lite (64)" "#CC6677"  --model $(f 20420538) "AEF+TESSERA-Lite" "#DDCC77" \
          --model $(f 20420834) "OLMO+TESSERA-Lite" "#AA4499" \
          --hatch_best --out plotting/img/binned_rmse_olmo_tessera_vs_aef_lite_fulltest.png

Poster version (Lite-trained models on the full test set; names only, no scores in the legend). Colours
are Tol "bright", chosen so that yellow (AEF) + blue (OlmoEarth) = green (AEF + OlmoEarth); grey = baseline:
      R=model/eval/results; f(){ echo $R/nico_film_$1-1_$1-1_$1-1_2019-2020_nooverlap.h5; }
      python plotting/helper/plot_binned_rmse.py \
          --model $(f 59684071) "AGBD features" "#555555" --model $(f 64424610) "AEF" "#DDAA33" \
          --model $(f 19819522) "OlmoEarth" "#4477AA"     --model $(f 20004081) "AEF + OlmoEarth" "#228833" \
          --hatch_best --no_scores --hatch_label "Best in bin" --out plotting/img/binned_rmse_lite_fulltest_poster.png

`--hatch_best` hatches, per bin, the lowest-RMSE model plus any within `--tie_tol` Mg/ha (default 1) of it.

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import makedirs
from os.path import dirname, abspath

###################################################################################################
# Helpers

# Model colours (Paul Tol 'muted', user-set 2026-10-06): AGBD #332288, AEF #117733, AEF+ #44AA99,
# other GFM (SSL4EO / TerraMind) #88CCEE, ESA CCI #CC6677.
# Default by --model order; a colour given on the command line wins.
DEFAULT_COLORS = ["#332288", "#117733", "#88CCEE", "#44AA99", "#CC6677", "#DDCC77", "#AA4499"]


def _darken(hex_color, factor=0.6):
    """Darker shade of a hex colour, for bar edges. Non-hex names fall back to black."""
    if not (isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7):
        return 'black'
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(r * factor), int(g * factor), int(b * factor))


def _contrast(color):
    """White on dark fills, black on light ones (for hatch lines). Non-hex names fall back to black."""
    if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
        return 'black'
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else 'black'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Per-bin RMSE for N models on the AGBD test set.')
    parser.add_argument('--model', action='append', nargs='+', metavar=('PATH LABEL', 'COLOR'), required=True,
                        help='A model as: PATH LABEL [COLOR]. Repeat --model for each series.')
    parser.add_argument('--ref', type=str, default=None,
                        help='LABEL of the reference model for the difference panel (default: the first --model).')
    parser.add_argument('--bin_size', type=int, default=50, help='Size of the AGB label bins.')
    parser.add_argument('--max_agb', type=int, default=500, help='Upper edge of the last bin (Mg/ha).')
    parser.add_argument('--out', type=str, required=True, help='Output image path.')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI.')
    parser.add_argument('--hatch_best', action='store_true',
                        help='Hatch, in each bin, the bar of the model with the lowest per-bin RMSE (both panels).')
    parser.add_argument('--tie_tol', type=float, default=1.0,
                        help='With --hatch_best: models within this many Mg/ha of the bin\'s lowest RMSE count as '
                             'tied and are hatched too (default 1.0; 0 hatches the single winner).')
    parser.add_argument('--no_scores', action='store_true',
                        help='Legend shows model names only, without the overall scores (e.g. for a poster).')
    parser.add_argument('--hatch_label', type=str, default=None,
                        help='With --hatch_best: legend text for the hatch (default: a full description of the '
                             'metric and tie tolerance; e.g. "Best in bin" for a poster).')
    args = parser.parse_args()

    models = []
    for i, spec in enumerate(args.model):
        if len(spec) < 2:
            parser.error(f'--model needs at least PATH and LABEL, got {spec}')
        color = spec[2] if len(spec) >= 3 else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        models.append({'path': spec[0], 'label': spec[1], 'color': color})
    ref = args.ref if args.ref is not None else models[0]['label']
    if ref not in [m['label'] for m in models]:
        parser.error(f'--ref {ref!r} is not one of the --model labels')
    return models, ref, args.bin_size, args.max_agb, args.out, args.dpi, args.hatch_best, args.tie_tol, args.no_scores, args.hatch_label


def load_residuals(path):
    """Residuals (pred - label) and labels, keeping only finite pairs."""
    with h5py.File(path, 'r') as f:
        preds = f['predictions'][:].astype(np.float64)
        labels = f['labels'][:].astype(np.float64)
    valid = np.isfinite(preds) & np.isfinite(labels)
    if (~valid).any():
        print(f'  {path}: dropping {int((~valid).sum()):,} non-finite pairs')
    return preds[valid] - labels[valid], labels[valid]


###################################################################################################
# Code execution

if __name__ == "__main__":

    models, ref, bin_size, max_agb, out_path, dpi, hatch_best, tie_tol, no_scores, hatch_label = parse_arguments()

    bins = np.arange(0, max_agb + 1, bin_size)
    lbs, ubs = bins[:-1], bins[1:]
    labels_x = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]
    n_bins, n_models = len(lbs), len(models)

    for m in models:
        print(f"Loading {m['label']} from {m['path']}")
        residuals, labels = load_residuals(m['path'])
        m['rmse'] = float(np.sqrt(np.mean(residuals ** 2)))
        m['counts'] = np.array([((labels >= lb) & (labels < ub)).sum() for lb, ub in zip(lbs, ubs)])
        assert (m['counts'] > 0).all(), f"{m['label']}: empty bin(s) {m['counts'].tolist()}"
        m['bin_rmse'] = np.array([np.sqrt(np.mean(residuals[(labels >= lb) & (labels < ub)] ** 2))
                                  for lb, ub in zip(lbs, ubs)])
        print(f"  N={len(labels):,}  RMSE={m['rmse']:.3f}  per-bin: " + ' '.join(f'{v:.2f}' for v in m['bin_rmse']))

    counts = models[0]['counts']
    for m in models[1:]:
        if not np.array_equal(m['counts'], counts):
            print(f"  WARNING: per-bin counts of {m['label']} differ from {models[0]['label']}")

    ref_rmse = next(m for m in models if m['label'] == ref)['bin_rmse']
    print(f'\nPer-bin RMSE minus {ref}:')
    for m in models:
        m['delta'] = m['bin_rmse'] - ref_rmse
        print(f"  {m['label']:>22s}: " + ' '.join(f'{v:+6.2f}' for v in m['delta']))

    # Per-bin winners: every model within tie_tol of the bin's lowest RMSE
    winners = np.zeros((n_models, n_bins), dtype=bool)
    if hatch_best:
        table = np.array([m['bin_rmse'] for m in models])
        winners = table <= table.min(axis=0) + tie_tol
        print(f'\nLowest per-bin RMSE (ties within {tie_tol:g} Mg/ha):')
        for i in range(n_bins):
            order = np.argsort(table[:, i])
            nxt = next((table[j, i] for j in order if not winners[j, i]), np.nan)
            print(f"  {labels_x[i]:>8s}: " + ', '.join(f"{models[j]['label']} ({table[j, i]:.2f})"
                                                       for j in order if winners[j, i]) + f"; next {nxt:.2f}")

    # N bars per bin, spread inside a slot of width 0.8 centred on the bin index
    slot = 0.8
    width = slot / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    plt.rcParams.update({'font.size': 13})
    fig, (ax, dax) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [1.25, 1], 'hspace': 0.08})

    for a in (ax, dax):
        # Faint alternating bands, one per AGB bin, as in plot_binned_models.py
        for i in range(n_bins):
            if i % 2 == 0:
                a.axvspan(i - 0.5, i + 0.5, color='0.94', zorder=0)
        a.yaxis.grid(True, color='0.85', linewidth=0.7, zorder=1)
        a.set_axisbelow(True)
        for spine in ('top', 'right'):
            a.spines[spine].set_visible(False)

    best_rmse = min(round(m['rmse'], 1) for m in models)
    handles = []
    for k, (m, off) in enumerate(zip(models, offsets)):
        pos = np.arange(n_bins) + off
        edge = _darken(m['color'])
        h = ax.bar(pos, m['bin_rmse'], width=width * 0.88, color=m['color'], edgecolor=edge, linewidth=0.8, zorder=3)
        handles.append(h)
        if m['label'] != ref:
            dax.bar(pos, m['delta'], width=width * 0.88, color=m['color'], edgecolor=edge, linewidth=0.8, zorder=3)
        else:
            # The reference is 0 everywhere by construction: mark it with a thin tick rather than nothing
            dax.scatter(pos, np.zeros(n_bins), marker='_', s=60, color=edge, zorder=3)
        # Hatch the winning bars as unfilled overlays, so the hatch colour can contrast with the fill
        w = winners[k]
        if w.any():
            hatch_kw = dict(width=width * 0.88, facecolor='none', edgecolor=_contrast(m['color']), linewidth=0,
                            hatch='///', zorder=3.5)
            ax.bar(pos[w], m['bin_rmse'][w], **hatch_kw)
            if m['label'] != ref:
                dax.bar(pos[w], m['delta'][w], **hatch_kw)
            else:
                # The reference has no bar on the difference panel: a hatched square at 0 marks its win
                sq = dax.scatter(pos[w], np.zeros(w.sum()), marker='s', s=90, facecolor=m['color'],
                                 edgecolor=_contrast(m['color']), linewidth=0.6, zorder=3.6)
                sq.set_hatch('////')

    def _rmse_txt(v):
        txt = f'{v:.1f}'
        return f'$\\mathbf{{{txt}}}$' if float(txt) == best_rmse else txt
    legend_labels = [m['label'] if no_scores else f"{m['label']}  (overall RMSE={_rmse_txt(m['rmse'])})"
                     for m in models]
    if hatch_best:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='white', edgecolor='0.3', hatch='///'))
        legend_labels.append(hatch_label if hatch_label is not None else
                             'Lowest RMSE in bin' + (f' (ties within {tie_tol:g} Mg/ha)' if tie_tol > 0 else ''))
    ax.legend(handles, legend_labels,
              frameon=True, framealpha=0.95, edgecolor='0.8', loc='upper left', ncol=1, fontsize=12)
    ax.set_ylabel('Per-bin RMSE (Mg/ha)', fontsize=14)

    dax.axhline(0, color='black', linestyle='--', alpha=0.6, zorder=2)
    dax.set_ylabel(f'$\\Delta$RMSE vs {ref} (Mg/ha)', fontsize=14)
    dax.set_xticks(range(n_bins))
    dax.set_xticklabels(labels_x)
    dax.set_xlim(-0.5, n_bins - 0.5)
    dax.set_xlabel('AGB bins (Mg/ha)', fontsize=14)
    # note in the bottom corner whose outer bins have the shallowest deltas, so bars don't run through it
    edge = lambda sl: max(abs(m['delta'][sl]).max() for m in models)
    left = edge(slice(0, 2)) < edge(slice(-2, None))
    dax.text(0.005 if left else 0.995, 0.03, f'below 0 = more accurate than {ref}', transform=dax.transAxes,
             ha='left' if left else 'right', va='bottom', fontsize=11, color='0.35')

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
