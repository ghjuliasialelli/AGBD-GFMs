"""

Binned-residual boxplot for an arbitrary number of models, on the AGBD (full) test set.

This generalises the two-model overall plot in plot.py (which is hard-wired to a model vs a
baseline) to N models, so a third series - e.g. a GFM such as SSL4EO-MoCo evaluated through the
pangaea benchmark - can be shown in the same panel as fcn_film on AGBD features vs AEF embeddings.

Each model is passed as `--model PATH LABEL [COLOR]`, repeated. Every results file must contain, at
minimum, per-sample `predictions` and `labels` datasets (raw AGB, Mg/ha). Residuals are binned by
the model's OWN labels, so the models need not share the same rows - though on the full AGBD test
set they do (2,807,977 samples each).

Run:  python plot_binned_models.py \
          --model results/nico_film_59620098-..._nooverlap.h5 "AGBD features" green \
          --model results/nico_film_59620113-..._nooverlap.h5 "AEF"           skyblue \
          --model results/ssl4eo_moco_..._agbd_test.h5         "SSL4EO-MoCo"   "#C02BF2" \
          --out ../../../manuscript/imgs/binned.png

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

# Fallback palette, used when a model is given without an explicit colour. This is the Okabe-Ito
# qualitative palette: designed to stay distinguishable for all common colour-vision deficiencies
# (protan/deutan/tritan) and to hold up in greyscale. https://jfly.uni-koeln.de/color/
DEFAULT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

# No hatching: the Okabe-Ito hues are themselves CVD-safe, so a plain fill reads cleaner.
DEFAULT_HATCHES = ["", "", "", "", "", ""]


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
    args = parser.parse_args()

    models = []
    for i, spec in enumerate(args.model):
        if len(spec) < 2:
            parser.error(f'--model needs at least PATH and LABEL, got {spec}')
        path, label = spec[0], spec[1]
        color = spec[2] if len(spec) >= 3 else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        hatch = DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)]
        models.append({'path': path, 'label': label, 'color': color, 'hatch': hatch})
    return models, args.bin_size, args.max_agb, args.out, args.dpi


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


###################################################################################################
# Code execution

if __name__ == "__main__":

    models, bin_size, max_agb, out_path, dpi = parse_arguments()

    bins = np.arange(0, max_agb + 1, bin_size)
    lbs, ubs = bins[:-1], bins[1:]
    labels_x = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]
    n_bins, n_models = len(lbs), len(models)

    # Load every model's residuals and bin them by its own labels
    for m in models:
        print(f"Loading {m['label']} from {m['path']}")
        residuals, labels = load_residuals(m['path'])
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        print(f"  {m['label']}: N={len(labels):,}  overall RMSE={rmse:.3f} Mg/ha")
        m['binned'] = [residuals[(labels >= lb) & (labels < ub)] for lb, ub in zip(lbs, ubs)]

    # N boxes per bin, evenly spread inside a slot of total width 0.8 centred on the bin index
    slot = 0.8
    width = slot / n_models
    # offsets: symmetric about 0, one per model
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    plt.rcParams.update({'font.size': 13, 'hatch.linewidth': 0.6})
    fig, ax = plt.subplots(figsize=(13, 6.5))

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
                        medianprops=dict(color='black', linewidth=1.6))
        handles.append(bp['boxes'][0])

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(labels_x)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_xlabel('AGB bins ($Mg/ha$)', fontsize=14)
    ax.set_ylabel('AGBD Test residuals ($AGB_{pred} - AGB_{true}$)', fontsize=14)
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, zorder=2)
    ax.yaxis.grid(True, color='0.85', linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(handles, [m['label'] for m in models], frameon=True, framealpha=0.95,
              edgecolor='0.8', loc='lower left', ncol=n_models)
    plt.tight_layout()

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
