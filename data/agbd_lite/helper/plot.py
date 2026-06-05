"""

This script creates plots to compare the performance of the supervised baseline vs. any supplied
model predictions on the AGBD-Lite test set. It generates binned residual boxplots and spider plots
of RMSE per biome, both overall and per region.

The results .h5 files must contain the following datasets:
- 'predictions': Model predictions on the test set.
- 'labels': True AGBD values for the test set.
- 'biomes': Biome classification for each test sample.
- 'regions': Region classification for each test sample.


Run:    python plot.py  --model <model_file> --baseline <baseline_model_file> 
                        --bin_size <bin_size> --saving_dir <saving_directory>

e.g.  python plot.py    --model results/GFM_test_results.h5
                        --baseline results/supervised_test_results.h5

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

REF_BIOMES = {20: 'Shrubs', 30: 'Herbaceous vegetation', 40: 'Cultivated', 90: 'Herbaceous wetland',
              111: 'Closed-ENL', 112: 'Closed-EBL', 114: 'Closed-DBL', 115: 'Closed-mixed', 116: 'Closed-other',
              121: 'Open-ENL', 122: 'Open-EBL', 124: 'Open-DBL', 125: 'Open-mixed', 126: 'Open-other'}
acceptable_biomes = [20, 30, 40, 90, 111, 112, 114, 115, 116, 121, 122, 124, 125, 126]
categories = ['Shrubs', 'HV', 'Crops', 'HW', 'C-ENL', 'C-EBL', 'C-DBL', 'C-M', 'C-O', 'O-ENL', 'O-EBL', 'O-DBL', 'O-M', 'O-O']
acceptable_regions = {1: 'Europe', 2: 'North Asia', 3:'Australasia', 4:'Africa', 5:'South Asia', 6:'South America', 7:'North America'}

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Plot model performance comparisons on AGBD-Lite test set.')
    parser.add_argument('--model', type = str, required = True, help = 'Path to the model results.')
    parser.add_argument('--baseline', type = str, required = True, help = 'Path to the baseline model results.')
    parser.add_argument('--bin_size', type = int, default = 50, help = 'Size of the bins for residual boxplots.')
    parser.add_argument('--saving_dir', type = str, default = 'plots', help = 'Directory to save the plots.')
    args = parser.parse_args()
    return args.model, args.baseline, args.bin_size, args.saving_dir

###################################################################################################
# Code execution

if __name__ == "__main__":

    # Parse arguments and setup variables
    model_results, baseline_results, bin_size, saving_dir = parse_arguments()
    makedirs(saving_dir, exist_ok=True)
    bins = np.arange(0, 501, bin_size)
    lbs, ubs = bins[:-1], bins[1:]
    labels = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]


    # ---------------------------------------------------------------------------------------------
    # Overall performance -------------------------------------------------------------------------

    # Load results files
    
    with h5py.File(model_results, 'r') as f:
        model_results = {'preds': f['predictions'][:], 'labels': f['labels'][:], 
                         'biomes': f['biomes'][:], 'regions': f['regions'][:]}
    with h5py.File(baseline_results, 'r') as f:
        baseline_results = {'preds': f['predictions'][:], 'labels': f['labels'][:],
                          'biomes': f['biomes'][:], 'regions': f['regions'][:]}

    # Plot overall binned residual boxplots, for whole vs lite
    model_binned_res, baseline_binned_res = [], []
    residuals_model = model_results['preds'][:] - model_results['labels'][:]
    residuals_baseline = baseline_results['preds'][:] - baseline_results['labels'][:]
    for lb, ub in zip(lbs, ubs):
        mask_model = (model_results['labels'][:] >= lb) & (model_results['labels'][:] < ub)
        model_binned_res.append(residuals_model[mask_model])
        mask_baseline = (baseline_results['labels'][:] >= lb) & (baseline_results['labels'][:] < ub)
        baseline_binned_res.append(residuals_baseline[mask_baseline])
    fig, ax = plt.subplots(figsize=(12, 6))    
    positions_model = [i - 0.2 for i in range(len(lbs))]
    positions_baseline = [i + 0.2 for i in range(len(lbs))]
    bp_model = ax.boxplot(model_binned_res, positions=positions_model, widths=0.3, patch_artist=True,
                          showfliers=False, boxprops=dict(facecolor="skyblue"))
    bp_baseline = ax.boxplot(baseline_binned_res, positions=positions_baseline, widths=0.3, patch_artist=True,
                           showfliers=False, boxprops=dict(facecolor="green"))    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('AGB bins ($Mg/ha$)')
    ax.set_ylabel('AGBD Test residuals ($AGB_{pred} - AGB_{true}$)')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.legend([bp_model["boxes"][0], bp_baseline["boxes"][0]], ['Trained on AGBD', 'Trained on AGBD-Lite'])
    plt.tight_layout()
    plt.savefig(join(saving_dir, 'binned-residuals_model-vs-lite.png'))

    # RMSE
    rmse_model = np.sqrt(np.mean((model_results['preds'] - model_results['labels'])**2))
    rmse_baseline = np.sqrt(np.mean((baseline_results['preds'] - baseline_results['labels'])**2))
    print(f'Overall AGBD Test RMSE (model): {rmse_model:.4f} Mg/ha')
    print(f'Overall AGBD Test RMSE (baseline): {rmse_baseline:.4f} Mg/ha')

    # Plot spider plot with RMSE per biome, whole vs lite
    biomes = np.array(acceptable_biomes)
    model_rmse_per_biome, baseline_rmse_per_biome = [], []
    for biome in biomes:
        biome_mask_model = model_results['biomes'][:] == biome
        biome_mask_baseline = baseline_results['biomes'][:] == biome
        if np.sum(biome_mask_model) == 0:
            model_rmse_per_biome.append(0.0)
            baseline_rmse_per_biome.append(0.0)
        else:
            rmse_model = np.sqrt(np.mean((model_results['preds'][biome_mask_model] - model_results['labels'][biome_mask_model])**2))
            rmse_baseline = np.sqrt(np.mean((baseline_results['preds'][biome_mask_baseline] - baseline_results['labels'][biome_mask_baseline])**2))
            model_rmse_per_biome.append(rmse_model)
            baseline_rmse_per_biome.append(rmse_baseline)
    valid_indices = [i for i, val in enumerate(model_rmse_per_biome) if val != 0.0]
    biomes = biomes[valid_indices]
    model_rmse_per_biome = [model_rmse_per_biome[i] for i in valid_indices]
    baseline_rmse_per_biome = [baseline_rmse_per_biome[i] for i in valid_indices]
    angles = np.linspace(0, 2 * np.pi, len(biomes), endpoint=False).tolist()
    model_rmse_per_biome += model_rmse_per_biome[:1]
    baseline_rmse_per_biome += baseline_rmse_per_biome[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, model_rmse_per_biome, color='skyblue', linewidth=2, label='Trained on AGBD')
    ax.fill(angles, model_rmse_per_biome, color='skyblue', alpha=0.25)
    ax.plot(angles, baseline_rmse_per_biome, color='green', linewidth=2, label='Trained on AGBD-Lite')
    ax.fill(angles, baseline_rmse_per_biome, color='green', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([REF_BIOMES[b] for b in biomes])
    ax.set_title('AGBD Test RMSE per biome', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(join(saving_dir, 'spider-plot_model-vs-lite.png'))


    # ---------------------------------------------------------------------------------------------
    # Per region performance ----------------------------------------------------------------------
    region_saving_dir = join(saving_dir, 'region')
    makedirs(region_saving_dir, exist_ok=True)
    regions = np.unique(model_results['regions'][:])
    
    for region in regions:

        # Merge North and South Asia
        if region == 2: continue
        if region == 5 : 
            region_mask_model = (model_results['regions'][:] == 5) | (model_results['regions'][:] == 2)
            region_mask_baseline = (baseline_results['regions'][:] == 5) | (baseline_results['regions'][:] == 2)
        else: 
            region_mask_model = model_results['regions'][:] == region
            region_mask_baseline = baseline_results['regions'][:] == region

        # Make the binned residuals plot
        model_binned_res, baseline_binned_res = [], []
        residuals_model = model_results['preds'][region_mask_model] - model_results['labels'][region_mask_model]
        residuals_baseline = baseline_results['preds'][region_mask_baseline] - baseline_results['labels'][region_mask_baseline]
        for lb, ub in zip(lbs, ubs):
            mask_model = (model_results['labels'][region_mask_model] >= lb) & (model_results['labels'][region_mask_model] < ub)
            model_binned_res.append(residuals_model[mask_model])
            mask_baseline = (baseline_results['labels'][region_mask_baseline] >= lb) & (baseline_results['labels'][region_mask_baseline] < ub)
            baseline_binned_res.append(residuals_baseline[mask_baseline])
        fig, ax = plt.subplots(figsize=(12, 6))    
        positions_model = [i - 0.2 for i in range(len(lbs))]
        positions_baseline = [i + 0.2 for i in range(len(lbs))]
        bp_model = ax.boxplot(model_binned_res, positions=positions_model, widths=0.3, patch_artist=True, 
                              showfliers=False, boxprops=dict(facecolor="skyblue"))
        bp_baseline = ax.boxplot(baseline_binned_res, positions=positions_baseline, widths=0.3, patch_artist=True, 
                               showfliers=False, boxprops=dict(facecolor="green"))    
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('AGB bins ($Mg/ha$)')
        ax.set_ylabel(f'AGBD Test residuals in {acceptable_regions[region]} ($AGB_{{pred}} - AGB_{{true}}$)')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.legend([bp_model["boxes"][0], bp_baseline["boxes"][0]], ['Trained on AGBD', 'Trained on AGBD-Lite'])
        plt.tight_layout()
        plt.savefig(join(region_saving_dir, f'{acceptable_regions[region]}_binned-residuals.png'))

        # Make the spider plot with RMSE per biome
        biomes = np.array(acceptable_biomes)
        model_rmse_per_biome, baseline_rmse_per_biome = [], []
        for biome in biomes:
            biome_mask_model = model_results['biomes'][:] == biome
            combined_mask_model = region_mask_model & biome_mask_model
            biome_mask_baseline = baseline_results['biomes'][:] == biome
            combined_mask_baseline = region_mask_baseline & biome_mask_baseline
            if np.sum(combined_mask_model) == 0:
                model_rmse_per_biome.append(0.0)
                baseline_rmse_per_biome.append(0.0)
            else:
                rmse_model = np.sqrt(np.mean((model_results['preds'][combined_mask_model] - model_results['labels'][combined_mask_model])**2))
                rmse_baseline = np.sqrt(np.mean((baseline_results['preds'][combined_mask_baseline] - baseline_results['labels'][combined_mask_baseline])**2))
                model_rmse_per_biome.append(rmse_model)
                baseline_rmse_per_biome.append(rmse_baseline)
        valid_indices = [i for i, val in enumerate(model_rmse_per_biome) if val != 0.0]
        biomes = biomes[valid_indices]
        model_rmse_per_biome = [model_rmse_per_biome[i] for i in valid_indices]
        baseline_rmse_per_biome = [baseline_rmse_per_biome[i] for i in valid_indices]
        angles = np.linspace(0, 2 * np.pi, len(biomes), endpoint=False).tolist()
        model_rmse_per_biome += model_rmse_per_biome[:1]
        baseline_rmse_per_biome += baseline_rmse_per_biome[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, model_rmse_per_biome, color='skyblue', linewidth=2, label='Trained on AGBD')
        ax.fill(angles, model_rmse_per_biome, color='skyblue', alpha=0.25)
        ax.plot(angles, baseline_rmse_per_biome, color='green', linewidth=2, label='Trained on AGBD-Lite')
        ax.fill(angles, baseline_rmse_per_biome, color='green', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([REF_BIOMES[b] for b in biomes])
        ax.set_title(f'AGBD Test RMSE per biome, in {acceptable_regions[region]}', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.savefig(join(region_saving_dir, f'{acceptable_regions[region]}_spider-plot.png'))
