"""

Binned-residual boxplot for an arbitrary number of models, on the AGBD (full) test set.

This generalises the two-model overall plot in plot.py (which is hard-wired to a model vs a
baseline) to N models, so a third series - e.g. a GFM such as SSL4EO-MoCo evaluated through the
pangaea benchmark - can be shown in the same panel as fcn_film on AGBD features vs AEF embeddings.

A second panel, to the right, shows the reference-AGB distribution on the same bins (linear counts), and
each model's overall RMSE and R^2 (1 - SS_res/SS_tot, over all finite pairs) are reported in the legend.

Each model is passed as `--model PATH LABEL [COLOR]`, repeated. Every results file must contain, at
minimum, per-sample `predictions` and `labels` datasets (raw AGB, Mg/ha). Residuals are binned by
the model's OWN labels, so the models need not share the same rows - though on the full AGBD test
set they do (2,807,977 samples each).

Run (from the repo root; colours default to DEFAULT_COLORS by --model order):
      python plotting/helper/plot_binned_models.py \
          --model model/eval/results/nico_film_59620098-1_59620098-1_59620098-1_2019-2020_nooverlap.h5 "AGBD features" \
          --model model/eval/results/nico_film_59620113-1_59620113-1_59620113-1_2019-2020_nooverlap.h5 "AEF" \
          --model benchmark_pangaea/results/20260930_080105_f8cb5c_terramind_tiny_lora_reg_upernet_agbd_agbd_test.h5 "TerraMind" \
          --out plotting/img/binned.png

Poster version (Lite-trained models on the full test set; names only, no scores in the legend). Colours
are Tol "bright", chosen so that yellow (AEF) + blue (OlmoEarth) = green (AEF + OlmoEarth); grey = baseline:
      R=model/eval/results; f(){ echo $R/nico_film_$1-1_$1-1_$1-1_2019-2020_nooverlap.h5; }
      python plotting/helper/plot_binned_models.py \
          --model $(f 59684071) "AGBD features" "#555555" --model $(f 64424610) "AEF" "#DDAA33" \
          --model $(f 19819522) "OlmoEarth" "#4477AA"     --model $(f 20004081) "AEF + OlmoEarth" "#228833" \
          --hatch_best median --no_scores --hatch_label "Best in bin" --out plotting/img/binned_lite_fulltest_poster.png

Add `--hatch_best rmse` (or mae / median) to hatch, in every bin, the box of the model that wins that bin;
models within `--tie_tol` Mg/ha (default 1) of the winner count as tied and are hatched too.

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import PathPatch, Patch
import h5py
from os import makedirs
from os.path import dirname, abspath

###################################################################################################
# Helpers

# Fallback palette, used when a model is given without an explicit colour. This is the Okabe-Ito
# qualitative palette: designed to stay distinguishable for all common colour-vision deficiencies
# (protan/deutan/tritan) and to hold up in greyscale. https://jfly.uni-koeln.de/color/
# Model colours (Paul Tol 'muted', user-set 2026-10-06): AGBD #332288, AEF #117733, AEF+ #44AA99,
# other GFM (SSL4EO / TerraMind) #88CCEE, ESA CCI #CC6677.
# Default by --model order (AGBD, AEF, other GFM, AEF+, CCI); a colour given on the command line wins.
DEFAULT_COLORS = ["#332288", "#117733", "#88CCEE", "#44AA99", "#CC6677", "#DDCC77"]

# No hatching: the Okabe-Ito hues are themselves CVD-safe, so a plain fill reads cleaner.
DEFAULT_HATCHES = ["", "", "", "", "", ""]


def _median_color(color):
    """White median on dark fills, black on light ones (a black median vanishes on #332288).
    Non-hex colour names fall back to black."""
    if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
        return 'black'
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else 'black'


def _darken(hex_color, factor=0.6):
    """Return a darker shade of a hex colour, for box/whisker edges (adds contrast against the
    lighter fill). Non-hex names (e.g. 'green') are returned unchanged."""
    if not (isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7):
        return 'black'
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(r * factor), int(g * factor), int(b * factor))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Binned-residual boxplot for N models on the AGBD test set.')
    parser.add_argument('--model', action='append', nargs='+', metavar=('PATH LABEL', 'COLOR'), required=True,
                        help='A model as: PATH LABEL [COLOR]. Repeat --model for each series.')
    parser.add_argument('--bin_size', type=int, default=50, help='Size of the AGB label bins.')
    parser.add_argument('--max_agb', type=int, default=500, help='Upper edge of the last bin (Mg/ha).')
    parser.add_argument('--out', type=str, required=True, help='Output image path (e.g. .../imgs/binned.png).')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI.')
    parser.add_argument('--ymin', type=float, default=None,
                        help='Lower y-limit of the residual panel, to make room for a tall legend (default: auto).')
    parser.add_argument('--hatch_best', choices=['none', 'rmse', 'mae', 'median'], default='none',
                        help='Hatch, in each bin, the box of the model that wins that bin: lowest per-bin RMSE, '
                             'lowest MAE, or smallest |median residual|. Default none (no hatching).')
    parser.add_argument('--tie_tol', type=float, default=1.0,
                        help='With --hatch_best: also hatch every model whose bin score is within this many Mg/ha '
                             'of the best, i.e. treat them as tied (default 1.0; 0 hatches the single winner).')
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
        path, label = spec[0], spec[1]
        color = spec[2] if len(spec) >= 3 else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        hatch = DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)]
        models.append({'path': path, 'label': label, 'color': color, 'hatch': hatch})
    return models, args.bin_size, args.max_agb, args.out, args.dpi, args.ymin, args.hatch_best, args.tie_tol, args.no_scores, args.hatch_label


def load_residuals(path):
    """Load per-sample predictions and labels and return (residuals, labels), dropping non-finite
    pairs (a GFM head can emit the odd nan/inf; leaving them in would empty a whole bin's box)."""
    with h5py.File(path, 'r') as f:
        preds = f['predictions'][:].astype(np.float64)
        labels = f['labels'][:].astype(np.float64)
    valid = np.isfinite(preds) & np.isfinite(labels)
    dropped = int((~valid).sum())
    if dropped:
        print(f'  {path}: dropping {dropped:,} non-finite pairs')
    preds, labels = preds[valid], labels[valid]
    return preds - labels, labels


def overall_metrics(residuals, labels):
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((labels - labels.mean()) ** 2))
    return rmse, r2


###################################################################################################
# Code execution

if __name__ == "__main__":

    models, bin_size, max_agb, out_path, dpi, ymin, hatch_best, tie_tol, no_scores, hatch_label = parse_arguments()

    bins = np.arange(0, max_agb + 1, bin_size)
    lbs, ubs = bins[:-1], bins[1:]
    labels_x = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]
    n_bins, n_models = len(lbs), len(models)

    # Load every model's residuals and bin them by its own labels
    for m in models:
        print(f"Loading {m['label']} from {m['path']}")
        residuals, labels = load_residuals(m['path'])
        m['rmse'], m['r2'] = overall_metrics(residuals, labels)
        print(f"  {m['label']}: N={len(labels):,}  RMSE={m['rmse']:.3f} Mg/ha  R2={m['r2']:.3f}")
        m['binned'] = [residuals[(labels >= lb) & (labels < ub)] for lb, ub in zip(lbs, ubs)]
        m['counts'] = np.array([len(b) for b in m['binned']])

    # The distribution panel shows one histogram, so the models must agree on the per-bin counts
    # (true on the full AGBD test set; a DDP-padded dump differs by a few duplicated samples).
    counts = models[0]['counts']
    for m in models[1:]:
        if not np.array_equal(m['counts'], counts):
            print(f"  WARNING: per-bin counts of {m['label']} differ from {models[0]['label']} by up to "
                  f"{np.abs(m['counts'] - counts).max()} samples; the histogram shows {models[0]['label']}'s")
    print('  samples per bin:', counts.tolist())

    # Per-bin winner (index into models) under the chosen metric, or None when hatching is off
    winners = None
    if hatch_best != 'none':
        score = {'rmse': lambda r: np.sqrt(np.mean(r ** 2)), 'mae': lambda r: np.mean(np.abs(r)),
                 'median': lambda r: abs(np.median(r))}[hatch_best]
        scores = np.array([[score(m['binned'][i]) if len(m['binned'][i]) else np.inf for i in range(n_bins)]
                           for m in models])
        # winners[i] = indices of every model within tie_tol of the bin's best score (ties are all hatched)
        winners = [np.flatnonzero(scores[:, i] <= scores[:, i].min() + tie_tol) for i in range(n_bins)]
        for i in range(n_bins):
            order = np.argsort(scores[:, i])
            nxt = next((scores[j, i] for j in order if j not in winners[i]), np.nan)
            print(f"  {labels_x[i]:>8s}: best {hatch_best} (tol {tie_tol:g}) = "
                  + ', '.join(f"{models[j]['label']} ({scores[j, i]:.2f})" for j in order if j in winners[i])
                  + f"; next {nxt:.2f}")

    # N boxes per bin, evenly spread inside a slot of total width 0.8 centred on the bin index
    slot = 0.8
    width = slot / n_models
    # offsets: symmetric about 0, one per model
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    plt.rcParams.update({'font.size': 13, 'hatch.linewidth': 0.6})
    fig, (ax, hax) = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={'width_ratios': [2.7, 1], 'wspace': 0.18})

    # Faint alternating background bands, one per AGB bin, so the eye groups each triple of boxes
    # together and reads the bin structure without needing gridlines between every box.
    for i in range(n_bins):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='0.94', zorder=0)

    handles = []
    for m, off in zip(models, offsets):
        edge = _darken(m['color'])
        positions = [i + off for i in range(n_bins)]
        bp = ax.boxplot(m['binned'], positions=positions, widths=width * 0.82, patch_artist=True,
                        showfliers=False, zorder=3,
                        boxprops=dict(facecolor=m['color'], edgecolor=edge, linewidth=1.1,
                                      hatch=m['hatch']),
                        whiskerprops=dict(color=edge, linewidth=1.1),
                        capprops=dict(color=edge, linewidth=1.1),
                        medianprops=dict(color=_median_color(m['color']), linewidth=1.6))
        handles.append(bp['boxes'][0])
        m['boxes'] = bp['boxes']

    # Hatch each bin's winning box. The hatch is drawn as an unfilled overlay so its colour can contrast
    # with the fill (white on dark, black on light); a hatch on the box itself takes the edge colour,
    # which vanishes on dark fills.
    if winners is not None:
        for i, ws in enumerate(winners):
            for w in ws:
                box = models[w]['boxes'][i]
                ax.add_patch(PathPatch(box.get_path(), facecolor='none', edgecolor=_median_color(models[w]['color']),
                                       linewidth=0, hatch='///', zorder=3.5))

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(labels_x)
    ax.set_xlim(-0.5, n_bins - 0.5)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_xlabel('AGB bins (Mg/ha)', fontsize=14)
    ax.set_ylabel('AGBD Test residuals ($AGB_{pred} - AGB_{GEDI}$)', fontsize=14)
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, zorder=2)
    ax.yaxis.grid(True, color='0.85', linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    # Best overall RMSE (lowest) and R^2 (highest) in bold, via mathtext
    best_rmse = min(round(m['rmse'], 1) for m in models)
    best_r2 = max(round(m['r2'], 3) for m in models)
    def _num(v, fmt, best):
        txt = format(v, fmt)
        return f"$\\mathbf{{{txt}}}$" if float(txt) == best else txt
    legend_labels = [m['label'] if no_scores else
                     f"{m['label']}  (RMSE={_num(m['rmse'], '.1f', best_rmse)}, "
                     f"R$^2$={_num(m['r2'], '.3f', best_r2)})" for m in models]
    if winners is not None:
        handles.append(Patch(facecolor='white', edgecolor='0.3', hatch='///'))
        legend_labels.append(hatch_label if hatch_label is not None else
                             {'rmse': 'Lowest RMSE in bin', 'mae': 'Lowest MAE in bin',
                              'median': 'Median residual closest to 0'}[hatch_best]
                             + (f' (ties within {tie_tol:g} Mg/ha)' if tie_tol > 0 else ''))
    ax.legend(handles, legend_labels,
              frameon=True, framealpha=0.95, edgecolor='0.8', loc='lower left', ncol=1)

    # Reference-AGB distribution on the same bins, linear (a log axis hides how few high-AGB samples there are)
    hax.bar(range(n_bins), counts, width=0.85, color='0.6', edgecolor='white', linewidth=0.8)
    hax.set_xticks(range(n_bins))
    hax.set_xticklabels(labels_x, rotation=45, ha='right')
    hax.set_xlim(-0.5, n_bins - 0.5)
    hax.set_xlabel('AGB bins (Mg/ha)', fontsize=14)
    hax.set_ylabel('Test samples', fontsize=14)
    # Tick labels carry their own unit (500K, 1M, 1.5M) instead of a '[thousands]' axis label
    hax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: '0' if v == 0 else (f'{v / 1e6:g}M' if v >= 1e6 else f'{v / 1e3:g}K')))
    hax.set_title('Reference AGB distribution', fontsize=14)
    hax.yaxis.grid(True, color='0.85', linewidth=0.7)
    hax.set_axisbelow(True)
    for spine in ('top', 'right'):
        hax.spines[spine].set_visible(False)

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
