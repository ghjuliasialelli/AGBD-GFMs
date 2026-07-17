"""

This script computes per-band Integrated Gradients (IG) attributions for the AEF model,
to understand which of the 64 AEF input channels most influence the AGBD predictions.

Usage:
    python aef_importance.py \
        --dataset_path local \
        --mode test \
        --years 2019 2020 \
        --n_samples 5000 or all \
        --n_steps 50 \
        --output_dir aef \
        --lite False \
        --override

"""

#######################################################################################################################
# Imports

import argparse
import numpy as np
import torch
import wandb

# config.py lives at the repo root, but this package is run with model/ as the working
# directory, so the root has to be put on the path explicitly before importing it.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WANDB_ENTITY
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader

from models import Net
from wrapper import Model
from dataset import GEDIDataset
from inference_residuals import init_args_dataset
from parser import str2bool

#######################################################################################################################
# Helpers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--years', type=int, nargs='+', default=[2019, 2020])
    parser.add_argument('--n_samples', type=str, default='5000', help='Max samples to process')
    parser.add_argument('--n_steps', type=int, default=50, help='IG integration steps')
    parser.add_argument('--output_dir', type=str, default='aef')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lite', type=str2bool, default=False, help='Whether to use AGBD-Lite embeddings')
    parser.add_argument('--override', action='store_true', help='Whether to override existing output files')
    return parser.parse_args()


def get_mapping(api, arch):
    runs = api.runs(f"{WANDB_ENTITY}/{arch}")
    run_mapping, run_ckpt = {}, {}
    for run in runs:
        try:
            run_mapping[run.name] = run.path[-1]
            run_ckpt[run.name] = run.config['model_path']
        except:
            continue
    return run_mapping, run_ckpt


def load_model(args, model_name, arch, dataset_path, device):
    """Load the NicoNet_FiLM model from checkpoint, mirroring eval.py."""

    net = Net(
        model_name=arch,
        in_features=args.in_features,
        num_outputs=args.num_outputs,
        downsample=None,
        patch_size=args.patch_size,
        local=(args.dataset_path == 'local'),
        device=device,
        biome_dim=args.biome_dim,
        emb_dim=args.emb_dim,
        num_sepconv_blocks=args.num_sepconv_blocks,
        num_sepconv_filters=args.num_sepconv_filters,
        long_skip=args.long_skip,
        only_entry=args.only_entry,
        linear_emb=args.linear_emb,
        padding_mode=args.padding_mode,
        returns=args.returns,
    )

    wrapped = Model(
        net,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        patch_size=args.patch_size,
        downsample=args.downsample,
        loss_fn=args.loss_fn,
        film=args.film,
        l2=args.l2,
        crop=args.crop,
    )

    ckpt_path = join(dataset_path['ckpt'], arch)
    state_dict = torch.load(
        join(ckpt_path, f'{model_name}_best.ckpt'),
        map_location=torch.device(device),
        weights_only=True,
    )['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if 'teacher' not in k}
    wrapped.load_state_dict(state_dict)

    model = wrapped.model
    model.to(device)
    model.eval()
    return model


def integrated_gradients(model, images, biome_embs, center, n_steps):
    """
    Compute Integrated Gradients for each input channel.

    Args:
        model: NicoNet_FiLM instance (already on device, in eval mode)
        images: (B, C, H, W) tensor on device, requires no grad
        biome_embs: (B, emb_dim) tensor on device
        center: int, index of center pixel (= patch_size // 2)
        n_steps: number of integration steps

    Returns:
        ig: (B, C, H, W) tensor of IG attributions
    """
    baseline = torch.zeros_like(images)
    alphas = torch.linspace(0, 1, n_steps, device=images.device)
    accumulated_grads = torch.zeros_like(images)

    for alpha in alphas:
        interp = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
        output = model((interp, biome_embs))[:, 0, center, center]
        output.sum().backward()
        accumulated_grads += interp.grad.detach()

    ig = (images - baseline) * (accumulated_grads / n_steps)
    return ig  # (B, C, H, W)


def plot_importances(band_importance_center, output_dir):
    """Bar chart, lollipop chart, and cumulative importance curve."""

    importance = band_importance_center / band_importance_center.sum() * 100  # relative %
    order = np.argsort(importance)[::-1]

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(order)), importance[order], color='steelblue', edgecolor='none')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(i) for i in order], fontsize=10)
    ax.set_ylabel('Attribution (%)', fontsize=11)
    ax.set_xlabel('Band #', fontsize=11)
    ax.set_title('Center-pixel attribution (AEF bands)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = join(output_dir, 'aef_band_importance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {out_path}')

    # --- Lollipop chart ---
    fig, ax = plt.subplots(figsize=(14, 5))
    xs = range(len(order))
    vals = importance[order]
    ax.vlines(xs, 0, vals, color='steelblue', linewidth=0.8, alpha=0.6)
    ax.plot(xs, vals, 'o', color='steelblue', markersize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(i) for i in order], fontsize=7)
    ax.set_ylabel('Attribution (%)', fontsize=11)
    ax.set_xlabel('Band #', fontsize=11)
    ax.set_title('Center-pixel attribution (AEF bands)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = join(output_dir, 'aef_band_importance_lollipop.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {out_path}')

    # --- Cumulative importance curve --
    """
    It's a cumulative importance curve showing how many AEF spectral bands you need (ranked by Integrated  
    Gradients attribution) to account for a given fraction of the total attribution.                     
                                                                                                            
    Concretely:                                                                                            
    - importance = mean |Integrated Gradients| at the center pixel, per AEF band
    - order = bands sorted by descending importance
    - The curve plots the cumulative sum of these sorted attributions (normalized to 1) against the number
    of top bands included                                                                                  
                                                                                                            
    The annotations mark how many bands reach the 50%, 80%, and 90% thresholds. So for example, "12 bands →
    80%" means the top 12 most-attributed AEF bands account for 80% of the total center-pixel attribution.

    It answers the question: how concentrated is the model's reliance across the AEF spectral bands? A     
    steep curve means a few bands dominate; a flat one means the model uses many bands roughly equally.
    """
    sorted_vals = importance[order]
    cumsum = np.cumsum(sorted_vals) / sorted_vals.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumsum) + 1), cumsum, color='steelblue', linewidth=2)
    for threshold in [0.5, 0.8, 0.9]:
        n = int(np.searchsorted(cumsum, threshold)) + 1
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(n, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.annotate(f'{n} bands → {int(threshold*100)}%', xy=(n, threshold),
                    xytext=(n + 1, threshold - 0.04), fontsize=9, color='gray')
    ax.set_xlabel('Number of top bands', fontsize=11)
    ax.set_ylabel('Cumulative attribution fraction', fontsize=11)
    ax.set_title('Cumulative center-pixel attribution (AEF bands)', fontsize=12)
    ax.set_xlim(1, len(cumsum))
    ax.set_ylim(0, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = join(output_dir, 'aef_band_importance_cumulative.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {out_path}')


#######################################################################################################################
# Main

if __name__ == '__main__':

    cli = parse_args()

    # Check if the output files already exist
    out_full = join(cli.output_dir, 'band_importances_full.npy')
    out_center = join(cli.output_dir, 'band_importances_center.npy')
    if not cli.override and exists(out_full) and exists(out_center):
        print(f'Output files already exist at {out_full} and {out_center}. Use --override to recompute.')

        # Load existing results
        band_importance_center = np.load(out_center)

        # Print top-10 bands
        top10 = np.argsort(band_importance_center)[::-1][:10]
        print('\nTop-10 AEF bands by center-pixel importance:')
        for rank, band_idx in enumerate(top10):
            print(f'  {rank+1}. AEF_{band_idx:02d}  importance={band_importance_center[band_idx]:.6f}')

        plot_importances(band_importance_center, cli.output_dir)
        exit(0)

    set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Paths ---
    if cli.dataset_path == 'local':
        dataset_path = {
            'h5':         '/scratch3/gsialelli/patches',
            'norm':       '/scratch3/gsialelli/patches',
            'map':        '/scratch3/gsialelli/patches',
            'ckpt':       '/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/weights',
            'embeddings': '/scratch3/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec',
            'aef_h5':     '/scratch3/gsialelli/patches/AEF',
            'aef_norm':   '/scratch3/gsialelli/patches/AEF',
        }
    else:
        dataset_path = {
            'h5':         '/cluster/work/igp_psr/gsialelli/Data/patches',
            'norm':       '/cluster/work/igp_psr/gsialelli/Data/patches',
            'map':        '/cluster/work/igp_psr/gsialelli/Data/patches',
            'ckpt':       '/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes',
            'embeddings': '/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec',
            'aef_h5':     '/cluster/work/igp_psr/gsialelli/Data/patches/AEF',
            'aef_norm':   '/cluster/work/igp_psr/gsialelli/Data/patches/AEF',
        }

    # --- Load wandb config ---
    arch = 'nico_film'
    config_file = 'eval/configs/evaluation_nico_film_ens_aef.txt'
    with open(config_file) as f:
        model_name = f.read().strip()  # '59620113-1'
    print(f'Model name: {model_name}')

    args = argparse.Namespace()
    args.dataset_path = cli.dataset_path

    api = wandb.Api()
    wandb_mapping, ckpt_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[model_name]
    cfg = api.run(f'{WANDB_ENTITY}/{arch}/{wandb_name}').config
    for key, value in cfg.items():
        setattr(args, key, value)
    args = init_args_dataset(args)
    if cli.lite: 
        args.lite = True
        args.eval_big = False
        args.drop_overlaps = False

    dataset_path['embeddings'] += '/AGBD-Lite' if args.lite else '/AGBD'

    # FiLM ensemble: same single model, 3 members
    assert args.ensemble and args.film, 'Expected ensemble FiLM model'
    MEMBER_ID = 0  # use member 0 for attribution

    # --- Load model ---
    model = load_model(args, model_name, arch, dataset_path, device)
    print('Model loaded.')

    # --- Load dataset ---
    dataset = GEDIDataset(
        paths=dataset_path,
        years=cli.years,
        chunk_size=1,
        mode=cli.mode,
        args=args,
        film=True,
        offset=False,
        return_region=False,
    )
    loader = DataLoader(dataset, batch_size=cli.bs, shuffle=True, num_workers=4, pin_memory=True)
    print(f'Dataset size: {len(dataset)}')

    center = args.patch_size[0] // 2

    # --- Compute IG attributions ---
    # https://alan-turing-institute.github.io/tea-techniques/techniques/integrated-gradients/ 
    n_bands = args.in_features
    band_importance_full   = np.zeros(n_bands, dtype=np.float64)
    band_importance_center = np.zeros(n_bands, dtype=np.float64)
    samples_processed = 0

    if cli.n_samples == 'all': cli.n_samples = len(dataset)
    else: cli.n_samples = int(cli.n_samples)

    for i, batch in enumerate(loader):
        if samples_processed >= cli.n_samples:
            break

        images, biomes, biome_embs, labels = batch

        # Clamp batch to remaining n_samples quota
        remaining = cli.n_samples - samples_processed
        if images.shape[0] > remaining:
            images, biome_embs = images[:remaining], biome_embs[:remaining]

        images = images.to(device)
        biome_embs = biome_embs.to(device)

        # Fix biome_embs to ensemble member 0
        biome_embs = torch.full_like(biome_embs, 0.0)
        biome_embs[:, MEMBER_ID] = 1.0

        with torch.enable_grad():
            ig = integrated_gradients(model, images, biome_embs, center, cli.n_steps)

        # Aggregate: mean |IG| across spatial dims (H, W) and then over batch
        band_importance_full   += ig.abs().mean(dim=(2, 3)).sum(dim=0).cpu().numpy()
        band_importance_center += ig[:, :, center, center].abs().sum(dim=0).cpu().numpy()

        samples_processed += images.shape[0]
        if i % 10 == 0:
            print(f'  Processed {samples_processed}/{cli.n_samples} samples')

    # Normalize by number of samples
    band_importance_full   /= samples_processed
    band_importance_center /= samples_processed
    print(f'Done. Processed {samples_processed} samples.')

    # --- Save results ---
    makedirs(cli.output_dir, exist_ok=True)
    np.save(join(cli.output_dir, 'band_importances_full.npy'),   band_importance_full)
    np.save(join(cli.output_dir, 'band_importances_center.npy'), band_importance_center)
    print('Saved importances.')

    plot_importances(band_importance_center, cli.output_dir)

    # Print top-10 bands (0-indexed)
    top10 = np.argsort(band_importance_center)[::-1][:10]
    print('\nTop-10 AEF bands by center-pixel importance:')
    for rank, band_idx in enumerate(top10):
        print(f'  {rank+1}. AEF_{band_idx:02d}  importance={band_importance_center[band_idx]:.6f}')
