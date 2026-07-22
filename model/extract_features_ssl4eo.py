"""

Feature-space visualisation, SSL4EO-MoCo row: PCA -> RGB of what the SSL4EO-MoCo + RegUPerNet model
learns, to sit beside the two nico_film rows produced by extract_features.py.

Why this is a SEPARATE script (and not a --which ssl4eo branch of extract_features.py). The nico_film
extractor feeds one big crop-plus-margin through a fully-convolutional body and slices a dense,
per-pixel feature map out of a SINGLE forward pass -- exact because convolutions are local. SSL4EO-
MoCo cannot be used that way: it is a ViT hard-wired to a 224 px input built from a 25x25 patch, and
it predicts only the CENTRE pixel of that patch (see inference_ssl4eo.py's docstring). So the only
faithful way to get a spatial feature map is the same sliding-window pass the prediction map uses --
one 25x25 patch per output location -- tapping the penultimate feature at each patch centre. This
script therefore reuses inference_ssl4eo.py's model loader, preprocessor and WindowDataset verbatim
and only adds the feature hook, so the features are produced by byte-identical preprocessing to the
30 m prediction map.

Tap point. The analog of nico_film's `body.predictions` is the decoder's final regression conv
`model.conv_reg` (a 1x1 conv). A forward-pre-hook captures its input: the (B, 512, Hf, Wf) feature
map that RegUPerNet turns into a biomass value (512-dim, vs nico_film's 256 -- PCA->RGB does not
care). Per patch we sample the vector at the feature-map location the CENTRE output pixel reads,
i.e. the representation behind the one pixel the model was trained and evaluated on.

Resolution. --stride 3 reproduces the 30 m prediction maps (keep_px 1, coarse): every feature is a
true centre-pixel feature, the grid is simply at stride x 10 m. The resulting coarse grid is
resampled (nearest) onto the SAME 10 m crop grid the nico_film features use -- read from the crop's
<tile>_meta.json (bounds + crs, written by extract_features.py) -- so make_feature_figure.py can
pixel-slice all columns identically. The 3x blockiness of the SSL4EO panel is real and honest: its
effective resolution IS 30 m here.

NOTE this breaks the "same body / same layer / same semantics" symmetry of the nico_film pair: this
is a different architecture, a different tap, and S2-only input. The figure becomes a
cross-architecture "what each model learns" comparison; the caption must say so.

Usage (pangaea-bench env; run one tile at a time):
    python -u extract_features_ssl4eo.py --tile 49SBT \
        --cache /scratch3/gsialelli/ssl4eo_maps/cache/49SBT.npz \
        --out_dir ../comparison/maps/features --stride 3

"""

###################################################################################################
# Imports

import argparse
import json
import os
import sys
from os.path import abspath, dirname, exists, join

import numpy as np
import rasterio as rs
import torch
from rasterio.transform import Affine
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, dirname(abspath(__file__)))
# Reuse the vetted loader / preprocessor / dataset from the prediction-map script, so the features
# are tapped from byte-identical preprocessing to the 30 m map. Importing does not run its main().
import inference_ssl4eo as SSL

###################################################################################################
# Feature sampling

def central_vector(feat) :
    """
    Sample the feature-map vector at the location the centre output pixel reads.

    The decoder produces `conv_reg`'s input at (Hf, Wf), applies the 1x1 conv, then bilinearly
    resizes to the (PATCH, PATCH) output; the kept pixel is the centre [HALF, HALF]. Under
    align_corners=False that output pixel maps to continuous feature coordinate
    (HALF + 0.5) * Hf / PATCH - 0.5, so we bilinearly sample the feature map there -- the exact
    representation behind the prediction that is kept.

    Args:
    - feat (torch.Tensor): (B, D, Hf, Wf) captured conv_reg input.

    Returns:
    - np.ndarray: (B, D) float32 centre-pixel feature vectors.
    """
    B, D, Hf, Wf = feat.shape
    fy = (SSL.HALF + 0.5) * Hf / SSL.PATCH - 0.5
    fx = (SSL.HALF + 0.5) * Wf / SSL.PATCH - 0.5
    y0, x0 = int(np.floor(fy)), int(np.floor(fx))
    y1, x1 = min(y0 + 1, Hf - 1), min(x0 + 1, Wf - 1)
    wy, wx = fy - y0, fx - x0
    v = (feat[:, :, y0, x0] * (1 - wy) * (1 - wx) + feat[:, :, y0, x1] * (1 - wy) * wx
         + feat[:, :, y1, x0] * wy * (1 - wx) + feat[:, :, y1, x1] * wy * wx)
    return v.float().cpu().numpy()


def lattice(lo, hi, limit_hi) :
    """
    Patch centres on the global {HALF + stride*k} lattice (the prediction map's lattice) that fall
    in [lo, hi), clamped so the 25x25 patch stays inside the window (centre in [HALF, limit_hi]).

    Sharing the map's lattice means each feature sits on a ground point the 30 m map actually
    predicted -- the SSL4EO feature grid is a crop of the prediction grid, not a re-phased one.
    """
    start = max(lo, SSL.HALF)
    k0 = int(np.ceil((start - SSL.HALF) / STRIDE))
    ks = np.arange(k0, k0 + int(np.ceil((hi - start) / STRIDE)) + 1)
    c = SSL.HALF + STRIDE * ks
    c = c[(c >= lo) & (c < hi) & (c <= limit_hi)]
    return c.astype(int)


###################################################################################################
# Main

STRIDE = 3

def main() :
    global STRIDE
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--cache", required=True, help="ssl4eo_maps/cache/<tile>.npz written by cache_s2_window.py")
    p.add_argument("--out_dir", default=join(dirname(abspath(__file__)), "..", "comparison", "maps", "features"))
    p.add_argument("--stride", type=int, default=3, help="Patch-centre spacing; 3 = 30 m, matching the prediction map.")
    p.add_argument("--run_dir", default=SSL.DEFAULT_RUN)
    p.add_argument("--ckpt", default="checkpoint__best.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=6)
    args = p.parse_args()
    STRIDE = args.stride

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- The crop the nico_film features already cover (bounds + crs on the S2 tile grid) ---
    meta_path = join(args.out_dir, f"{args.tile}_meta.json")
    with open(meta_path) as f :
        meta = json.load(f)
    left, bottom, right, top = meta["bounds"]
    crop_px = int(meta["crop_px"])
    crop_crs = meta["crs"]

    # --- Cached S2 window (same one the 30 m map was made from) ---
    with open(args.cache.replace(".npz", ".json")) as f :
        side = json.load(f)
    cache = np.load(args.cache)
    sr, valid = cache["sr"], cache["valid"]
    _, H, W = sr.shape
    assert str(side["crs"]) == str(crop_crs), f"cache CRS {side['crs']} != crop CRS {crop_crs}"
    cwt = Affine(*side["transform"])

    # Crop box -> cache pixel window (round to nearest pixel, as run_agbd does). The crop's top-left
    # is (left, top) because the transform is north-up (e < 0).
    inv = ~cwt
    c0f, r0f = inv * (left, top)
    c0, r0 = int(round(c0f)), int(round(r0f))
    r1, c1 = r0 + crop_px, c0 + crop_px
    assert 0 <= r0 and r1 <= H and 0 <= c0 and c1 <= W, \
        f"crop window rows[{r0},{r1}) cols[{c0},{c1}) falls outside cache {H}x{W}"

    cy = lattice(r0, r1, H - 1 - SSL.HALF)
    cx = lattice(c0, c1, W - 1 - SSL.HALF)
    Hc, Wc = len(cy), len(cx)
    print(f"{args.tile}: crop rows[{r0},{r1}) cols[{c0},{c1}) -> {Hc}x{Wc} centres at {10*STRIDE} m "
          f"({Hc*Wc:,} patches)")

    # --- Model + preprocessor (reused verbatim) ---
    model, cfg = SSL.build_model(args.run_dir, args.ckpt, device)
    pre = SSL.instantiate(cfg.preprocessing.test, dataset_cfg=cfg.dataset, encoder_cfg=cfg.encoder,
                          _recursive_=False)

    captured = {}
    def pre_hook(module, inputs) :
        captured["f"] = inputs[0].detach()
    handle = model.conv_reg.register_forward_pre_hook(pre_hook)

    ds = SSL.WindowDataset(sr, cy, cx, pre)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True, prefetch_factor=4)

    feat_grid = None
    try :
        with torch.no_grad() :
            for batch, idx in tqdm(loader, desc=f"{args.tile} ssl4eo feats", unit="batch") :
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda") :
                    model({"optical": batch}, output_shape=(SSL.PATCH, SSL.PATCH))
                cvec = central_vector(captured["f"])  # (B, D)
                if feat_grid is None :
                    feat_grid = np.zeros((Hc, Wc, cvec.shape[1]), dtype=np.float32)
                ii, jj = np.divmod(idx.numpy(), Wc)
                feat_grid[ii, jj] = cvec
    finally :
        handle.remove()

    valid_grid = valid[np.ix_(cy, cx)]

    # --- Resample the coarse (stride x 10 m) grid onto the 10 m crop grid used by the nico_film
    # features, so all figure columns share one pixel grid. Nearest centre per 10 m pixel. ---
    rows = np.clip(np.round((np.arange(r0, r0 + crop_px) - cy[0]) / STRIDE).astype(int), 0, Hc - 1)
    cols = np.clip(np.round((np.arange(c0, c0 + crop_px) - cx[0]) / STRIDE).astype(int), 0, Wc - 1)
    feat10 = feat_grid[np.ix_(rows, cols)]                 # (crop_px, crop_px, D)
    valid10 = valid_grid[np.ix_(rows, cols)].astype(bool)  # (crop_px, crop_px)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(join(args.out_dir, f"{args.tile}_ssl4eo_feat.npy"), feat10)
    np.save(join(args.out_dir, f"{args.tile}_ssl4eo_valid.npy"), valid10)
    crop_transform = cwt * Affine.translation(c0, r0)
    with open(join(args.out_dir, f"{args.tile}_ssl4eo_geo.json"), "w") as f :
        json.dump({"transform": list(crop_transform)[:6], "crs": str(crop_crs),
                   "height": crop_px, "width": crop_px,
                   "native_res_m": 10 * STRIDE, "native_shape": [Hc, Wc],
                   "feat_dim": int(feat10.shape[2]), "tap": "conv_reg",
                   "run": os.path.basename(args.run_dir.rstrip("/"))}, f, indent=2)
    print(f"  saved {args.tile}_ssl4eo_feat.npy {feat10.shape}  valid {valid10.mean():.3f}")


if __name__ == "__main__" :
    main()
