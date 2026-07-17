"""

This script creates a plot of the per-region improvement of one model over another, on the AGBD
test set. It complements plot.py, which bins the residuals by AGB value; here, the metrics are
instead broken down by world region.

It was written to compare NicoNet+FiLM trained on the AGBD features against the same architecture
trained on the AEF embeddings, but any two results files are accepted.

The results .h5 files must contain the following datasets:
- 'predictions': Model predictions on the test set.
- 'labels': True AGBD values for the test set.
- 'biomes': Biome classification for each test sample.
- 'regions': Region classification for each test sample.

Both files must have been produced by the same eval.py configuration (same `mode`, and same
`--drop_overlap`), so that their rows line up; this is checked on the labels before plotting.

The left panel shows the metric per region for both models, the right panel the improvement of
the model over the baseline (positive = the model is better). Regions are sorted by improvement,
so the regions with the biggest gap - the ones worth running inference on - come out on top.

Run:    python plot_regions.py --model <model_file> --baseline <baseline_model_file>
                               --model_label <label> --baseline_label <label>
                               --metric <rmse|mae|me> --saving_dir <saving_directory>

e.g.  python plot_regions.py  --model results/nico_film_59620113-1_59620113-1_59620113-1_2019-2020_nooverlap.h5
                              --baseline results/nico_film_59620098-1_59620098-1_59620098-1_2019-2020_nooverlap.h5
                              --model_label AEF --baseline_label 'AGBD features'

"""

###################################################################################################
# Imports

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import makedirs
import argparse

###################################################################################################
# Helper functions and global variables

acceptable_regions = {1: 'Europe', 2: 'North Asia', 3: 'Australasia', 4: 'Africa', 5: 'South Asia', 6: 'South America', 7: 'North America'}

# Region 2 (North Asia) and region 5 (South Asia) are merged into a single Asia region, as is done
# in plot.py; both hold the Nepal and ShaanxiProvince AOIs.
MERGED_REGIONS = {2: 5}
MERGED_NAMES = {5: 'Asia'}

COLORS = {'model': "#0084FF", 'baseline': "#C02BF2"}

# Improvement panel: the model is better (positive) / the baseline is better (negative)
DELTA_COLORS = {'better': "#0084FF", 'worse': "#C02BF2"}

METRICS = ['rmse', 'mae', 'me']
METRIC_NAMES = {'rmse': 'RMSE', 'mae': 'MAE', 'me': 'ME'}


def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Plot per-region model performance comparisons on the AGBD test set.')
    parser.add_argument('--model', type = str, required = True, help = 'Path to the model results.')
    parser.add_argument('--baseline', type = str, required = True, help = 'Path to the baseline model results.')
    parser.add_argument('--model_label', type = str, default = 'Model', help = 'Legend label for the model.')
    parser.add_argument('--baseline_label', type = str, default = 'Baseline', help = 'Legend label for the baseline model.')
    parser.add_argument('--metric', type = str, default = 'rmse', choices = METRICS, help = 'Metric to compare per region.')
    parser.add_argument('--merge_asia', type = lambda s: s.lower() == 'true', default = True, help = 'Whether to merge North and South Asia, as in plot.py.')
    parser.add_argument('--saving_dir', type = str, default = 'plots', help = 'Directory to save the plots.')
    args = parser.parse_args()
    return args.model, args.baseline, args.model_label, args.baseline_label, args.metric, args.merge_asia, args.saving_dir


def load_results(path):
    """
    This function loads a results .h5 file.

    Args:
    - path (str): path to the results .h5 file.

    Returns:
    - results (dict): the 'preds', 'labels', 'biomes' and 'regions' arrays.
    """
    with h5py.File(path, 'r') as f:
        return {'preds': f['predictions'][:], 'labels': f['labels'][:],
                'biomes': f['biomes'][:], 'regions': f['regions'][:]}


def compute_metric(preds, labels, metric):
    """
    This function computes a metric between predictions and labels.

    Args:
    - preds (np.ndarray): the predictions.
    - labels (np.ndarray): the labels.
    - metric (str): one of 'rmse', 'mae', 'me'.

    Returns:
    - value (float): the metric value, or np.nan if there are no samples.
    """
    if len(preds) == 0: return np.nan
    residuals = preds - labels
    if metric == 'rmse': return float(np.sqrt(np.mean(residuals ** 2)))
    elif metric == 'mae': return float(np.mean(np.abs(residuals)))
    else: return float(np.mean(residuals))


###################################################################################################
# Code execution

if __name__ == "__main__":

    # Parse arguments and setup variables
    model_path, baseline_path, model_label, baseline_label, metric, merge_asia, saving_dir = parse_arguments()
    makedirs(saving_dir, exist_ok = True)

    # Load the results files
    model_results = load_results(model_path)
    baseline_results = load_results(baseline_path)

    # Check that the two files line up, so that they can be compared region by region
    if len(model_results['labels']) != len(baseline_results['labels']):
        raise ValueError(f"The two results files hold a different number of samples "
                         f"({len(model_results['labels'])} vs {len(baseline_results['labels'])}); they were "
                         f"produced by different eval.py configurations and cannot be compared.")
    if not np.allclose(model_results['labels'], baseline_results['labels'], equal_nan = True):
        raise ValueError("The two results files hold different labels; they were produced by different "
                         "eval.py configurations (check `mode` and `--drop_overlap`) and cannot be compared.")

    # The rows line up, so a single region array describes both files
    regions = model_results['regions']
    if merge_asia:
        for src, dst in MERGED_REGIONS.items(): regions = np.where(regions == src, dst, regions)

    # Drop any sample whose prediction or label is not finite
    valid = np.isfinite(model_results['preds']) & np.isfinite(baseline_results['preds']) & np.isfinite(model_results['labels'])
    if not valid.all(): print(f'Dropping {(~valid).sum()} samples with non-finite predictions or labels')

    # Compute the metric per region, for both models
    region_codes = sorted(c for c in np.unique(regions) if c in acceptable_regions)
    names, model_values, baseline_values, counts = [], [], [], []
    for code in region_codes:
        mask = (regions == code) & valid
        if mask.sum() == 0: continue
        names.append(MERGED_NAMES.get(code) if merge_asia and code in MERGED_NAMES else acceptable_regions[code])
        model_values.append(compute_metric(model_results['preds'][mask], model_results['labels'][mask], metric))
        baseline_values.append(compute_metric(baseline_results['preds'][mask], baseline_results['labels'][mask], metric))
        counts.append(int(mask.sum()))
    names = np.array(names)
    model_values, baseline_values, counts = np.array(model_values), np.array(baseline_values), np.array(counts)

    # The improvement of the model over the baseline. For RMSE and MAE, lower is better, so the
    # improvement is baseline - model. For ME, what matters is the distance from zero.
    if metric == 'me': deltas = np.abs(baseline_values) - np.abs(model_values)
    else: deltas = baseline_values - model_values

    # Overall metric, over all regions
    overall_mask = valid
    overall_model = compute_metric(model_results['preds'][overall_mask], model_results['labels'][overall_mask], metric)
    overall_baseline = compute_metric(baseline_results['preds'][overall_mask], baseline_results['labels'][overall_mask], metric)
    print(f'\nOverall AGBD Test {METRIC_NAMES[metric]} ({model_label}): {overall_model:.4f} Mg/ha')
    print(f'Overall AGBD Test {METRIC_NAMES[metric]} ({baseline_label}): {overall_baseline:.4f} Mg/ha')

    # Print the per-region table, sorted by improvement
    order = np.argsort(deltas)[::-1]
    print(f'\nAGBD Test {METRIC_NAMES[metric]} per region ($Mg/ha$), sorted by improvement:')
    print(f"  {'Region':<15}{model_label:>16}{baseline_label:>16}{'Improvement':>14}{'N':>12}")
    for i in order:
        print(f'  {names[i]:<15}{model_values[i]:>16.2f}{baseline_values[i]:>16.2f}{deltas[i]:>14.2f}{counts[i]:>12,}')

    # ---------------------------------------------------------------------------------------------
    # Plot ----------------------------------------------------------------------------------------

    # Regions are sorted by improvement, so that the biggest gaps come out on top
    names, model_values, baseline_values, deltas, counts = names[order], model_values[order], baseline_values[order], deltas[order], counts[order]
    positions = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize = (14, 6), sharey = True)

    # --- Left panel: the metric per region, for both models ---
    ax = axes[0]
    ax.barh(positions - 0.2, model_values, height = 0.35, color = COLORS['model'], label = model_label)
    ax.barh(positions + 0.2, baseline_values, height = 0.35, color = COLORS['baseline'], label = baseline_label)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{n}\n(n={c:,})' for n, c in zip(names, counts)])
    ax.invert_yaxis()
    ax.set_xlabel(f'AGBD Test {METRIC_NAMES[metric]} ($Mg/ha$)')
    ax.legend(loc = 'lower right')
    ax.grid(axis = 'x', alpha = 0.3)
    ax.set_axisbelow(True)

    # --- Right panel: the improvement of the model over the baseline ---
    ax = axes[1]
    colors = [DELTA_COLORS['better'] if d > 0 else DELTA_COLORS['worse'] for d in deltas]
    ax.barh(positions, deltas, height = 0.55, color = colors)
    ax.axvline(0, color = 'black', linestyle = '--', alpha = 0.5)
    ax.set_xlabel(f'{METRIC_NAMES[metric]} improvement of {model_label} over {baseline_label} ($Mg/ha$)')
    ax.grid(axis = 'x', alpha = 0.3)
    ax.set_axisbelow(True)

    # Annotate each bar with its value, on the outer side of the bar
    span = max(np.abs(deltas).max(), 1e-6)
    for pos, delta in zip(positions, deltas):
        offset = 0.02 * span if delta > 0 else -0.02 * span
        ax.text(delta + offset, pos, f'{delta:+.2f}', va = 'center', ha = 'left' if delta > 0 else 'right', fontsize = 10)
    ax.set_xlim(min(deltas.min() * 1.25, -0.1 * span), max(deltas.max() * 1.25, 0.1 * span))

    plt.tight_layout()
    saving_path = join(saving_dir, f'per-region-{metric}_{model_label}-vs-{baseline_label}.png'.replace(' ', '-'))
    plt.savefig(saving_path, dpi = 300, bbox_inches = 'tight')
    print(f'\nPlot saved to {saving_path}')
