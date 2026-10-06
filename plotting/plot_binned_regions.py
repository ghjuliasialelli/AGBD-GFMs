"""

Binned RMSE by AGB bin, for the three cross-region (zero-shot) models, one panel per target region.

Replaces the signed "AEF minus AGBD" table in SOUTH_ASIA_TRANSFER.md: plotting the RMSE levels
themselves removes the sign convention and makes the crossover legible. The lower row shows each
region's share of test samples per bin, which is what turns a tail-only deficit into a 20 Mg/ha
aggregate gap in South Asia and into nothing at all in Africa and South America.

Every results file must contain per-sample `predictions` and `labels` (raw AGB, Mg/ha), as written
by model/eval.py. Samples are binned by their own labels; all three series within a panel share the
same rows, so the bin populations are identical across series.

Run (from the repo root, `run` env):
      python plotting/plot_binned_regions.py \
          --out plotting/img/south_asia_binned.png

"""

###################################################################################################
# Imports

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from os import makedirs
from os.path import dirname, abspath, join, exists

###################################################################################################
# Helpers

# Default evaluation-results directory ($AGBD_LOCAL_PLOTS on the authors' workstation).
DEFAULT_PLOTS = '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/eval_plots'

# Cross-region ("without") job IDs per target region, as reported in tab:geo_africa. Note the AEF
# Africa entry: eval_ablations/gen.py names 65148996-2, which was never evaluated; the published
# 35.08 comes from 64424736-2.
JOBS = {
    'SouthAsia':    {'AGBD features': '60736796-1', 'AEF': '65148996-1', 'AEF$^+$': '64424805-1'},
    'Africa':       {'AGBD features': '60736796-2', 'AEF': '64424736-2', 'AEF$^+$': '64424805-2'},
    'SouthAmerica': {'AGBD features': '60866344-3', 'AEF': '64424736-3', 'AEF$^+$': '64424805-3'},
}

# AGBD features in the repo blue (comparison/agbref/comparison.py); the two AEF variants in two
# shades of green, dark for AEF and light for AEF+, so they read as one family.
COLORS = {'AGBD features': '#0084FF', 'AEF': 'green', 'AEF$^+$': 'yellowgreen'}

# Short names for the per-panel legend box. Padded to a common width and rendered in a monospace
# font so the RMSE values line up vertically.
SHORT = {'AGBD features': 'AGBD', 'AEF': 'AEF', 'AEF$^+$': 'AEF+'}

PRETTY = {'SouthAsia': 'South Asia', 'Africa': 'Africa', 'SouthAmerica': 'South America'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Binned RMSE per AGB bin for the cross-region models.')
    parser.add_argument('--plots_dir', type=str, default=DEFAULT_PLOTS, help='Directory holding the eval .h5 files.')
    parser.add_argument('--bin_size', type=int, default=50, help='Size of the AGB label bins.')
    parser.add_argument('--max_agb', type=int, default=500, help='Upper edge of the last bin (Mg/ha).')
    parser.add_argument('--out', type=str, required=True, help='Output image path.')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI.')
    args = parser.parse_args()
    return args.plots_dir, args.bin_size, args.max_agb, args.out, args.dpi


def load_preds(plots_dir, job, region):
    """Load per-sample predictions and labels for one (job, region) evaluation."""
    path = join(plots_dir, f'nico_film_{job}_{job}_{job}_2019-2020_nooverlap_{region}.h5')
    if not exists(path):
        raise FileNotFoundError(f'missing evaluation file: {path}')
    with h5py.File(path, 'r') as f:
        preds, labels = f['predictions'][:].astype(np.float64), f['labels'][:].astype(np.float64)
    valid = np.isfinite(preds) & np.isfinite(labels)
    return preds[valid], labels[valid]


###################################################################################################
# Code execution

if __name__ == "__main__":

    plots_dir, bin_size, max_agb, out_path, dpi = parse_arguments()

    bins = np.arange(0, max_agb + 1, bin_size)
    lbs, ubs = bins[:-1], bins[1:]
    labels_x = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]
    n_bins = len(lbs)
    # The last bin absorbs everything at or above its lower edge (AGB is clipped at 500 Mg/ha).
    labels_x[-1] = f'{lbs[-1]}+'

    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.4), sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': [2.4, 1]})

    for col, region in enumerate(['SouthAsia', 'Africa', 'SouthAmerica']):
        ax, axb = axes[0, col], axes[1, col]

        counts, handles = None, []
        for label, job in JOBS[region].items():
            preds, labs = load_preds(plots_dir, job, region)
            idx = np.clip(np.digitize(labs, bins) - 1, 0, n_bins - 1)
            rmse = np.array([np.sqrt(np.mean((preds[idx == i] - labs[idx == i]) ** 2))
                             if (idx == i).any() else np.nan for i in range(n_bins)])
            if counts is None:
                counts = np.array([(idx == i).sum() for i in range(n_bins)], dtype=float)
                overall = {}
            overall[label] = float(np.sqrt(np.mean((preds - labs) ** 2)))
            line, = ax.plot(range(n_bins), rmse, marker='o', markersize=6.0, linewidth=2.0,
                            color=COLORS[label], markeredgecolor='0.25', markeredgewidth=0.6,
                            zorder=3)
            handles.append(line)

        share = 100 * counts / counts.sum()
        axb.bar(range(n_bins), share, width=0.7, color='0.55', edgecolor='0.3', linewidth=0.6, zorder=3)

        # One legend per panel, carrying that region's overall RMSEs. Monospace + a fixed-width
        # name field keeps the numbers vertically aligned; the frame comes from the legend itself.
        ax.set_title(PRETTY[region], fontsize=15, pad=8)
        leg_labels = [f'{SHORT[k]:<5}{v:>5.1f}' for k, v in overall.items()]
        ax.legend(handles, leg_labels, title='overall RMSE (t/ha)', loc='upper left',
                  frameon=True, framealpha=0.95, edgecolor='0.8', fancybox=False,
                  prop={'family': 'monospace', 'size': 12}, title_fontsize=11.5,
                  handlelength=1.6, borderpad=0.7, labelspacing=0.45)

        for a in (ax, axb):
            a.set_xlim(-0.5, n_bins - 0.5)
            a.yaxis.grid(True, color='0.85', linewidth=0.7, zorder=2)
            a.set_axisbelow(True)
            for spine in ('top', 'right'):
                a.spines[spine].set_visible(False)

        axb.set_xticks(range(n_bins))
        axb.set_xticklabels(labels_x, rotation=45, ha='right')
        axb.set_xlabel('AGB bins ($t/ha$)', fontsize=14)
        if col == 0:
            ax.set_ylabel('Test RMSE ($t/ha$)', fontsize=14)
            axb.set_ylabel('share of\ntest set (%)', fontsize=13)
        axb.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    plt.tight_layout()

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'\nSaved {out_path}')
