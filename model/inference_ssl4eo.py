"""

Stage B of the SSL4EO-MoCo map pipeline: run the SSL4EO-MoCo + RegUPerNet model fine-tuned on AGBD
over a cached Sentinel-2 window, and write an AGB prediction GeoTIFF on the Sentinel-2 grid.

Counterpart of `inference_aef.py` / `inference_agbd.py`, which produce the other rows of the map
figure. Run `cache_s2_window.py` (agbd env) first, then this (pangaea-bench env).

WHY 25x25 PATCHES RESIZED TO 224, AND NOT NATIVE 224x224 PATCHES
----------------------------------------------------------------
The AGBD training patches are 25x25 px at 10 m, upsampled to 224x224 by `ResizeToEncoder` before
normalisation. So the encoder was fine-tuned at an effective 1.12 m/px, seeing 250 m of ground per
image and ~17.9 m per ViT patch token. Feeding it a native 224x224 crop would be 10 m/px, 2.24 km
of ground, ~160 m per token -- a ~9x scale shift. It would not error, it would just be wrong. The
encoder is also hard-wired to 224 (`PatchEmbed(224, 16, ...)` plus a sin-cos `pos_embed` built for
a 14x14 grid), so native 224 is not even reachable without changing the encoder.

WHY A 5x5 BLOCK IS KEPT PER PATCH, AND WHY THAT NEEDS CHECKING
--------------------------------------------------------------
The AGBD dataset labels ONLY the centre pixel of each patch (`agbd.py:301-302` fills the target
with ignore_index and writes the GEDI value at [12, 12]), and both the trainer and the evaluator
mask on it. So of the 625 output pixels, exactly one ever received a gradient. Tiling the map with
whole 25x25 blocks would make 624/625 of the map unsupervised output.

It was tempting to keep the central k x k block anyway and pay k^2 times less compute: the trained
part of the model is the UPerNet head (the encoder is frozen, `finetune: false`) and it is fully
convolutional, so an off-centre output ought to be the same learned function on a shifted receptive
field. That argument is WRONG here, and the per-offset diagnostic below is what showed it. Measured
on 49SBT, 1600 patches, mean prediction by offset from the patch centre:

    [[107.33   6.74 118.48   6.28  97.29]
     [  8.03  11.83   6.64  14.9    8.02]
     [105.3    4.39 108.07   4.26  91.15]     <- centre = 108.07
     [  5.9   11.41   5.18  13.8    7.27]
     [ 81.58   4.77  84.05   5.48  67.12]]

Each cell averages 1600 patches of statistically identical ground, so a 20x spread cannot be
terrain. Only the EVEN-offset lattice carries signal at all; the odd offsets come out of the head
negative and are clamped to exactly 0 by RegUPerNet's `torch.relu`. The surviving even offsets
still range 67-118 t/ha, a 76% spread. Checked independently against the written raster: absolute
pixel parity is flat (39.07 / 39.38 / 39.26 / 39.93) while phase within the block grid reproduces
the table above, and an FFT of the map peaks at periods of 2.5 and 5 px. So the structure is keyed
to the patch coordinate frame -- it is the model, not the ground.

Consequence: --keep_px must be 1. Values > 1 are retained only to reproduce that diagnostic.

THE RESOLUTION / COMPUTE DIAL
------------------------------
Centre-only costs one forward pass per output pixel. At ~340 patches/s on the 3090:

    stride 1  ->  10 m,  18.4 M patches,  ~15 h per 40 km window
    stride 3  ->  30 m,   2.0 M patches,  ~1.7 h
    stride 5  ->  50 m,   0.7 M patches,  ~36 min

With --keep_px 1, --stride s gives an honest map at s x 10 m: every pixel is a true centre-pixel
prediction, the grid is simply coarser. Note that make_map_figure.py renders each panel at
RENDER_PX = 1400 from a ~4000 px raster, so a 30 m map (~1410 px) is already at the figure's
rendering resolution and nothing visible is lost.

Usage (pangaea-bench env):
    python -u model/inference_ssl4eo.py --cache /scratch3/gsialelli/ssl4eo_maps/cache/49SBT.npz \
        --out /scratch3/gsialelli/ssl4eo_maps/preds/49SBT.tif --keep_px 1 --stride 3

"""

###################################################################################################
# Imports

import argparse
import json
import logging
import os
import sys
import time
from os import makedirs
from os.path import abspath, basename, dirname, exists, join

import numpy as np
import rasterio as rs
import torch
from rasterio.transform import Affine
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# pangaea is not installed into this repo's path; it lives beside it.
PANGAEA = os.environ.get('PANGAEA_ROOT', '/scratch3/gsialelli/pangaea-bench')
sys.path.insert(0, PANGAEA)

from hydra.utils import instantiate
from omegaconf import OmegaConf

###################################################################################################
# Configuration

DEFAULT_RUN = ('/scratch3/gsialelli/AGBD-GFMs/benchmark_pangaea/full/runs/'
               '20260316_155017_36a16f_ssl4eo_moco_reg_upernet_agbd')

PATCH = 25        # the AGBD patch size the model was trained on; dataset.img_size in the config
HALF = PATCH // 2 # 12 px of context each side of the pixel being predicted

NODATA = -9999.0

###################################################################################################
# Helpers

def parse_args() :
    """
    Returns the parsed command-line arguments.
    """

    p = argparse.ArgumentParser(description = __doc__,
                                formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', type = str, required = True,
                   help = 'The .npz written by cache_s2_window.py.')
    p.add_argument('--out', type = str, required = True,
                   help = 'Output GeoTIFF path.')
    p.add_argument('--run_dir', type = str, default = DEFAULT_RUN,
                   help = 'Benchmark run directory holding configs/config.yaml and '
                          'checkpoint__best.pth.')
    p.add_argument('--ckpt', type = str, default = 'checkpoint__best.pth',
                   help = 'Checkpoint file name inside --run_dir.')
    p.add_argument('--keep_px', type = int, default = 1,
                   help = 'Side of the central block kept from each 25x25 prediction. Must be odd. '
                          'MEASURED TO BE UNUSABLE ABOVE 1 -- see the module docstring. Kept as a '
                          'flag only so the diagnostic that established that can be reproduced.')
    p.add_argument('--stride', type = int, default = None,
                   help = 'Spacing between patch centres, in source pixels. Default = --keep_px, '
                          'which gives a dense 10 m map. With --keep_px 1, a stride of s gives an '
                          'honest map at s x 10 m resolution: every pixel is a true centre-pixel '
                          'prediction, the grid is just coarser. This is the resolution/compute '
                          'dial -- see the module docstring for the numbers.')
    p.add_argument('--batch_size', type = int, default = 32,
                   help = 'Forward-pass batch size. 32 saturates the 3090 at ~2.6 GB; larger '
                          'batches are not faster (measured) and only cost memory.')
    p.add_argument('--num_workers', type = int, default = 6,
                   help = 'Dataloader workers. The preprocessor runs ~308 samples/s per worker '
                          'against a ~340 samples/s GPU, so 6 keeps the GPU fed with headroom.')
    p.add_argument('--limit_px', type = int, default = None,
                   help = 'Predict only the top-left N x N pixels of the output. For the '
                          'diagnostic runs; None = the whole window.')
    p.add_argument('--overwrite', action = 'store_true',
                   help = 'Recompute even if the output already exists.')
    return p.parse_args()


def block_centres(out_n, step, half, lo, hi) :
    """
    Choose the patch centres that tile `out_n` output pixels with blocks of `keep` pixels.

    Centres are clamped into [lo, hi] so the 25x25 patch always lies inside the cached window; the
    last block of an axis whose length is not a multiple of `keep` would otherwise want a centre
    that pushes the patch past the edge. Clamping makes that block overlap its neighbour slightly
    rather than run off the end, and the overlap is harmless because both predictions are equally
    valid -- the later one simply overwrites.

    Args:
    - out_n (int): number of source pixels along the axis to cover.
    - step (int): spacing between consecutive centres, in source pixels.
    - half (int): half-width of the block kept per patch (0 for centre-only).
    - lo (int): smallest legal centre, in window coordinates.
    - hi (int): largest legal centre, in window coordinates.

    Returns:
    - np.ndarray: the centres, in window coordinates.
    """

    n_blocks = int(np.ceil(out_n / step))
    centres = lo + half + np.arange(n_blocks) * step
    return np.clip(centres, lo, hi)


class WindowDataset(Dataset) :
    """
    Yields one preprocessed 25x25 patch per output block.

    The preprocessing chain is the *real* one instantiated from the run's own config
    (ResizeToEncoder -> BandFilter -> NormalizeMinMax -> BandPadding), not a reimplementation.
    That is affordable because it was measured at ~308 samples/s per worker against a ~340
    samples/s GPU, so the reference implementation is not the bottleneck and there is no reason to
    risk a hand-rolled copy drifting from it.
    """

    def __init__(self, sr, centres_y, centres_x, preprocessor) :
        """
        Args:
        - sr (np.ndarray): (12, H, W) surface reflectance for the cached window.
        - centres_y (np.ndarray): patch centre rows, in window coordinates.
        - centres_x (np.ndarray): patch centre cols, in window coordinates.
        - preprocessor: the instantiated pangaea test preprocessor.
        """
        self.sr = sr
        self.cy = centres_y
        self.cx = centres_x
        self.pre = preprocessor
        self.nx = len(centres_x)

    def __len__(self) :
        return len(self.cy) * self.nx

    def __getitem__(self, n) :
        i, j = divmod(n, self.nx)
        y, x = int(self.cy[i]), int(self.cx[j])

        patch = self.sr[:, y - HALF : y + HALF + 1, x - HALF : x + HALF + 1]
        # (C, H, W) -> (C, T, H, W); the dataset is single-temporal so T = 1.
        patch = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(1)

        out = self.pre({'image': {'optical': patch},
                        'target': torch.full((PATCH, PATCH), -1.0),
                        'metadata': {}})
        return out['image']['optical'], n


def build_model(run_dir, ckpt_name, device) :
    """
    Rebuild the encoder + decoder from the run's saved config and load the fine-tuned checkpoint.

    `load_encoder_weights` is deliberately NOT called: the fine-tuned checkpoint already carries
    every `encoder.*` key, and calling it would additionally require the pretrained weights to sit
    at a path relative to the pangaea working directory. The load is `strict = True` so that a
    config that does not match the checkpoint fails loudly here rather than silently producing a
    partly-random model.

    Args:
    - run_dir (str): the benchmark run directory.
    - ckpt_name (str): checkpoint file name inside it.
    - device (torch.device): where to put the model.

    Returns:
    - torch.nn.Module: the decoder (which owns the encoder), in eval mode.
    - omegaconf.DictConfig: the run config.
    """

    cfg = OmegaConf.load(join(run_dir, 'configs', 'config.yaml'))

    assert cfg.encoder.input_size == 224, f'Expected a 224 px encoder, got {cfg.encoder.input_size}'
    assert cfg.dataset.img_size == PATCH, (
        f'This script assumes the model was trained on {PATCH}x{PATCH} patches, but the run config '
        f'says img_size = {cfg.dataset.img_size}. The patch extraction below would feed the model '
        f'a different ground extent than it was trained on.')

    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder, encoder = encoder)

    ckpt = torch.load(join(run_dir, ckpt_name), map_location = 'cpu', weights_only = False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
    decoder.load_state_dict(state, strict = True)

    return decoder.to(device).eval(), cfg


###################################################################################################
# Main

def main() :

    args = parse_args()
    assert args.keep_px % 2 == 1, f'--keep_px must be odd, got {args.keep_px}'
    assert args.keep_px <= PATCH, f'--keep_px cannot exceed the patch size {PATCH}'

    stride = args.stride if args.stride is not None else args.keep_px
    assert stride >= args.keep_px, f'--stride {stride} < --keep_px {args.keep_px} would double-write'
    assert stride == args.keep_px or args.keep_px == 1, (
        f'--stride {stride} with --keep_px {args.keep_px} leaves gaps between blocks. Use either '
        f'stride == keep_px (a dense 10 m map) or keep_px == 1 (a coarse map at stride x 10 m).')
    coarse = args.keep_px == 1 and stride > 1

    if exists(args.out) and not args.overwrite :
        print(f'{args.out} already exists; pass --overwrite to recompute. Nothing to do.')
        return

    logging.basicConfig(level = logging.WARNING)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    # Load the cached window ---------------------------------------------------------------------
    side_path = args.cache.replace('.npz', '.json')
    with open(side_path) as f : side = json.load(f)
    cache = np.load(args.cache)
    sr, valid = cache['sr'], cache['valid']
    n_bands, H, W = sr.shape
    print(f'Window {side["tile"]}: {W} x {H} px, {n_bands} bands, {side["crs"]}, '
          f'product {side["product"]}')

    m_left, m_top, m_right, m_bot = side['margins_obtained']
    assert min(side['margins_obtained']) >= HALF, (
        f'Cached window has margins {side["margins_obtained"]} px but {HALF} are needed for every '
        f'output pixel to get a full {PATCH}x{PATCH} patch. Re-run cache_s2_window.py with '
        f'--margin_px {HALF}.')

    # The predictable region: everything the margins allow a full patch for.
    out_h, out_w = H - m_top - m_bot, W - m_left - m_right
    if args.limit_px is not None :
        out_h, out_w = min(out_h, args.limit_px), min(out_w, args.limit_px)
        print(f'--limit_px {args.limit_px}: predicting only the top-left {out_w} x {out_h} px')
    print(f'Source region: {out_w} x {out_h} px, keep_px = {args.keep_px}, stride = {stride}')

    # Model --------------------------------------------------------------------------------------
    model, cfg = build_model(args.run_dir, args.ckpt, device)
    pre = instantiate(cfg.preprocessing.test, dataset_cfg = cfg.dataset, encoder_cfg = cfg.encoder,
                      _recursive_ = False)
    print(f'Model loaded; preprocessing = {[type(p).__name__ for p in pre.preprocessor]}')

    # Block layout -------------------------------------------------------------------------------
    k, h = args.keep_px, args.keep_px // 2
    cy = block_centres(out_h, stride, h, m_top, H - 1 - HALF)
    cx = block_centres(out_w, stride, h, m_left, W - 1 - HALF)
    n_blocks = len(cy) * len(cx)
    print(f'{n_blocks:,} patches ({len(cy)} x {len(cx)} centres)')

    ds = WindowDataset(sr, cy, cx, pre)
    loader = DataLoader(ds, batch_size = args.batch_size, num_workers = args.num_workers,
                        shuffle = False, pin_memory = True, prefetch_factor = 4)

    # In coarse mode the raster IS the grid of patch centres, one pixel per prediction, at
    # stride x 10 m. In dense mode it is the full-resolution source region.
    if coarse : pred = np.full((len(cy), len(cx)), NODATA, dtype = np.float32)
    else :      pred = np.full((out_h, out_w), NODATA, dtype = np.float32)

    # Per-offset accumulators: for every (dy, dx) in [-h, h]^2, the running sum / sum of squares /
    # count of predictions written at that offset from a patch centre. This is the diagnostic that
    # says whether keeping a k x k block is legitimate; see the module docstring.
    off_sum = np.zeros((k, k), dtype = np.float64)
    off_sqs = np.zeros((k, k), dtype = np.float64)
    off_cnt = np.zeros((k, k), dtype = np.int64)

    # Inference ----------------------------------------------------------------------------------
    nx = len(cx)
    with torch.no_grad() :
        for batch, idx in tqdm(loader, desc = f'{side["tile"]} keep={k}', unit = 'batch') :
            batch = batch.to(device, non_blocking = True)
            with torch.autocast('cuda', dtype = torch.float16, enabled = device.type == 'cuda') :
                logits = model({'optical': batch}, output_shape = (PATCH, PATCH))
            # Back to fp32 before it touches the accumulators: fp16 saturates around 65k and the
            # squares of a few-hundred-t/ha prediction would otherwise be fine, but the mean of
            # ~700k of them would not.
            block = logits[:, 0, HALF - h : HALF + h + 1, HALF - h : HALF + h + 1].float().cpu().numpy()

            if coarse :
                # One prediction per output pixel; the grid of centres IS the raster.
                ii, jj = np.divmod(idx.numpy(), nx)
                pred[ii, jj] = block[:, 0, 0]
                off_sum[0, 0] += block[:, 0, 0].sum()
                off_sqs[0, 0] += (block[:, 0, 0].astype(np.float64) ** 2).sum()
                off_cnt[0, 0] += block.shape[0]
                continue

            for b in range(block.shape[0]) :
                n = int(idx[b])
                i, j = divmod(n, nx)
                y, x = int(cy[i]), int(cx[j])

                # Where this block lands in output coordinates, clipped to the output extent.
                r0, c0 = y - m_top - h, x - m_left - h
                sr0, sc0 = max(0, -r0), max(0, -c0)
                r0, c0 = max(0, r0), max(0, c0)
                r1, c1 = min(out_h, r0 + k - sr0), min(out_w, c0 + k - sc0)
                if r1 <= r0 or c1 <= c0 : continue

                sub = block[b, sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)]
                pred[r0 : r1, c0 : c1] = sub

                off_sum[sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)] += sub
                off_sqs[sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)] += sub.astype(np.float64) ** 2
                off_cnt[sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)] += 1

    # Mask ---------------------------------------------------------------------------------------
    # Models happily predict on nodata, so the mask is applied rather than left to the figure.
    # In coarse mode the mask is sampled at the predicted locations, not averaged over the cell:
    # the prediction belongs to that one pixel, so its validity is that one pixel's validity.
    if coarse : vmask = valid[np.ix_(cy, cx)]
    else :      vmask = valid[m_top : m_top + out_h, m_left : m_left + out_w]
    n_masked = int((~vmask).sum())
    pred[~vmask] = NODATA
    print(f'Masked {n_masked:,} invalid px ({100 * n_masked / vmask.size:.2f}%)')

    unwritten = int((pred == NODATA).sum()) - n_masked
    assert unwritten == 0, (f'{unwritten:,} output pixels were never written -- the block layout '
                            f'does not tile the output. This would leave nodata holes in the map.')

    # Per-offset diagnostic ----------------------------------------------------------------------
    off_mean = off_sum / np.maximum(off_cnt, 1)
    off_std = np.sqrt(np.maximum(off_sqs / np.maximum(off_cnt, 1) - off_mean ** 2, 0))
    centre = off_mean[h, h]
    spread = float(off_mean.max() - off_mean.min())
    print('\nPer-offset mean prediction (t/ha), offset from patch centre:')
    print(np.round(off_mean, 2))
    print(f'centre = {centre:.2f} | across-offset spread = {spread:.2f} t/ha '
          f'({100 * spread / max(abs(centre), 1e-6):.1f}% of centre)')
    if spread > 0.02 * abs(centre) :
        print('WARNING: the per-offset means differ by more than 2% of the centre value. A '
              f'periodic {k} px pattern will be present in the map. Consider a smaller --keep_px.')

    # Write --------------------------------------------------------------------------------------
    win_transform = Affine(*side['transform'])

    if coarse :
        # Each raster pixel is centred on the ground point that was actually predicted. Anchoring
        # the corner at (cx[0], cy[0]) instead would shift the whole map by half a cell -- half of
        # 30 m at stride 3, which is a real georeferencing error, not a rounding detail.
        px_w, px_h = win_transform.a * stride, win_transform.e * stride
        first_x, first_y = win_transform * (int(cx[0]) + 0.5, int(cy[0]) + 0.5)
        out_transform = Affine(px_w, 0.0, first_x - px_w / 2,
                               0.0, px_h, first_y - px_h / 2)
    else :
        out_transform = win_transform * Affine.translation(m_left, m_top)

    makedirs(dirname(abspath(args.out)), exist_ok = True)
    profile = {'driver': 'GTiff', 'height': pred.shape[0], 'width': pred.shape[1], 'count': 1,
               'dtype': 'float32', 'crs': side['crs'], 'transform': out_transform,
               'nodata': NODATA, 'compress': 'lzw'}
    tmp = args.out + '.tmp.tif'
    with rs.open(tmp, 'w', **profile) as dst :
        dst.write(pred, 1)
        dst.update_tags(tile = side['tile'], product = side['product'],
                        model = 'ssl4eo_moco + RegUPerNet, fine-tuned on AGBD full',
                        run = basename(args.run_dir.rstrip('/')), keep_px = str(k),
                        stride = str(stride), resolution_m = str(10 * stride if coarse else 10),
                        patch = f'{PATCH}x{PATCH} resized to 224')
    os.replace(tmp, args.out)

    stats = {
        'tile': side['tile'], 'product': side['product'], 'run': args.run_dir,
        'keep_px': k, 'stride': stride, 'coarse': coarse,
        'resolution_m': 10 * stride if coarse else 10,
        'n_patches': n_blocks, 'out_shape': list(pred.shape),
        'limit_px': args.limit_px,
        'offset_mean': off_mean.tolist(), 'offset_std': off_std.tolist(),
        'offset_count': off_cnt.tolist(),
        'centre_mean': float(centre), 'offset_spread': spread,
        'masked_px': n_masked, 'valid_frac': float(vmask.mean()),
        'pred_mean': float(pred[vmask].mean()) if vmask.any() else None,
        'pred_median': float(np.median(pred[vmask])) if vmask.any() else None,
        'runtime_s': time.time() - t0,
    }
    with open(args.out.replace('.tif', '_stats.json'), 'w') as f :
        json.dump(stats, f, indent = 2)

    valid_pred = pred[vmask]
    print(f'\nWrote {args.out}')
    print(f'AGB over valid px: mean {valid_pred.mean():.1f}, median {np.median(valid_pred):.1f}, '
          f'p99 {np.percentile(valid_pred, 99):.1f} t/ha')
    print(f'Total {time.time() - t0:.0f} s')


if __name__ == '__main__' :
    main()
