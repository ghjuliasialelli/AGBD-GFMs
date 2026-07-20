"""

Feature-space visualisation experiment: PCA -> RGB of what each model *learns*.

For the two `nico_film` models compared in the paper -- one on the 31 hand-crafted AGBD features
(S2 + ALOS + DEM + LC + lat/lon), one on the 64-dim AEF embeddings -- this script extracts the
model's *penultimate feature map* (the `num_sepconv_filters`-dim tensor that feeds the final
`predictions` 1x1 conv, i.e. the representation the biomass regressor actually reads), reduces it to
3 dimensions with PCA, and writes it as a georeferenced RGB GeoTIFF. `make_feature_figure.py` then
lays those next to the true-colour Sentinel-2 backdrop, per location, so one can see *what the
extracted features pick up, where* -- and compare the two representations on high-biomass ground.

Why this tap point. Both models share the exact same `XceptionS2_FiLM` body; the only difference is
the input. Tapping the pre-regressor feature map is therefore the one *symmetric* place to compare
"the features each model learns": same layer, same dimensionality (256), same semantics (the vector
the linear regressor turns into a biomass value). It is captured with a forward-pre-hook on
`body.predictions`, which sees exactly the post-long-skip activation (see nico_net_film.py:243).

Design decisions (deliberate, and worth not re-deriving):
  - SINGLE FiLM member (member 0). Both released models are FiLM *ensembles* of 3 members
    (ensemble=True, n_members=3): one checkpoint evaluated 3x with a one-hot member embedding.
    Averaging feature maps across members is not meaningful -- each member applies a different FiLM
    modulation, so their activations do not live in a shared basis. Prediction variance across
    members is low, so member 0 is representative. (Contrast the *prediction* pipeline, which does
    average, because predictions are commensurable and features are not.)
  - BOUNDED CROP, not the whole window. A full 40 km window at 256 features is
    4000*4000*256*4 B = 16 GB, which this box has OOMed on before. A few-km crop is legible, memory-
    trivial, and is exactly the "high-biomass case" the experiment is about. Because the models use
    zero padding (padding_mode='zeros' -> output size == input size) and convolutions are local, the
    features on the crop are *exact* as long as we feed a margin wider than the receptive field
    (MARGIN px of real neighbouring data on every side, then slice the centre back out).
  - NODATA IS MASKED BEFORE PCA. Models happily produce features over nodata (ocean, cloud); those
    pixels are excluded from the PCA fit and rendered transparent, not fed in to skew the projection.
    The mask is taken from the raw input (AEF sentinel -128 / S2 SCL), before normalisation.

Crop centring (an *independent* reference, decided before looking, to avoid cherry-picking a crop
that flatters either of our own models):
  reference priority = ESA CCI v6.0 crop -> AEF prediction -> AGBD prediction -> window centre.
  Among candidate CROP x CROP windows on a coarse stride grid, fully inside the valid area, pick the
  one with the highest mean reference biomass. Deterministic (ties -> smallest row, then col).
  Override with --center "<row> <col>" (AEF-window pixel coords) when a location has no reference.

Usage (run one location at a time; serialise -- a full S2 tile load is ~15 GB):
    conda run -n agbd python -u extract_features.py --tile 59GPM \
        --product S2B_MSIL2A_20200223T222539_N0500_R029_T59GPM_20230515T102828 \
        --which both --out_dir ../comparison/maps/features

"""

###################################################################################################
# Imports

import os, sys, json, glob, argparse
from os.path import join, exists, dirname, abspath, basename

sys.path.insert(0, dirname(dirname(abspath(__file__))))  # repo root, for config.py
from config import get_paths

import numpy as np
import rasterio as rs
from rasterio.windows import from_bounds, Window, transform as window_transform
from rasterio.transform import array_bounds
import torch

# The two inference entry points. We reuse their (vetted) model loader and input pipelines verbatim,
# so the features we tap are produced by byte-identical preprocessing to training/eval -- the one
# invariant that matters most here ("inference must mirror training").
import inference_aef as AEF
import inference_agbd as AGBD
from inference_residuals import init_args_dataset
from wandb_cache import load_cache

###################################################################################################
# Configuration

ARCH = "nico_film"
AGBD_MODEL = "59620098-1"   # nico_film on the 31 hand-crafted AGBD features
AEF_MODEL  = "59620113-1"   # nico_film on the 64-dim AEF embeddings
MEMBER = 0                  # single FiLM ensemble member (see module docstring)

# Where the 40 km AEF windows live (NOT paths['aef'], which is the 780 AGBRef plots). Matches the
# AGBD_LOCAL_AEF the map-figure inference was run with.
AEF_TILES_DIR = "/scratch3/gsialelli/AEF_tiles"
# ESA CCI v6.0 crops already reprojected onto each AEF window's grid (crop-centring reference).
CCI_DIR = "/scratch3/gsialelli/CCI/maps"
# Existing per-tile predictions, used only as fallback centring references.
PRED_AEF  = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps_aef/nico_film"
PRED_AGBD = "/scratch3/gsialelli/EcosystemAnalysis/Models/Biomes/predictions_maps/nico_film"

CROP_PX = 320       # ~3.2 km at 10 m
MARGIN = 48         # real-context border fed to the net, wider than the receptive field, then sliced off
STRIDE = 64         # coarse grid of candidate crop origins for the centring search
MIN_VALID = 0.98    # a candidate crop must be at least this fraction valid (non-nodata) reference

###################################################################################################
# Model loading (offline: config from the committed wandb cache, no wandb API / credentials)

def make_args(cfg) :
    """
    Build the argparse.Namespace that `Inference` expects, from a run's cached config -- the same
    attribute set run_inference() assembles, minus the wandb round-trip.

    Args:
    - cfg (dict): the run's training config, from model/eval/wandb_cache/nico_film.json.

    Returns:
    - argparse.Namespace: with every cfg key set as an attribute, plus the dataset defaults.
    """
    args = argparse.Namespace()
    args.dataset_path = "local"
    for k, v in cfg.items() : setattr(args, k, v)
    args = init_args_dataset(args)
    return args


def load_net(which, cfg, paths, device) :
    """
    Load one model and return its inner network (the `Net`), in eval mode.

    Reuses the module's own `Inference` class so the checkpoint-loading path is identical to the
    prediction pipeline. AEF and AGBD define an identical `Inference`; either works, but we use each
    module's own to stay faithful.

    Args:
    - which (str): 'aef' or 'agbd'.
    - cfg (dict): the run's config.
    - paths (dict): dataset paths (must contain 'ckpt').
    - device (torch.device): where to place the model.

    Returns:
    - torch.nn.Module: the `Net`, eval()-ed, with weights loaded.
    """
    args = make_args(cfg)
    model_name = AEF_MODEL if which == "aef" else AGBD_MODEL
    Inference = AEF.Inference if which == "aef" else AGBD.Inference
    inf = Inference(arch=ARCH, model_name=model_name, paths=paths, tile_name="NA", args=args, device=device)
    return inf.model  # the Net; .model is NicoNet_FiLM; .model.body is XceptionS2_FiLM


###################################################################################################
# Feature extraction

def extract_penultimate(net, crop, n_members, device) :
    """
    Run the net on one crop (member 0) and return the penultimate feature map and the biomass
    prediction, both at the crop's spatial resolution.

    A forward-pre-hook on `body.predictions` captures its input, i.e. the post-long-skip
    `num_sepconv_filters`-dim activation -- the representation the regressor reads. The forward
    return value is the biomass prediction, kept for context panels.

    Args:
    - net (torch.nn.Module): the `Net` (nico_film).
    - crop (np.ndarray): (H, W, C) input, already normalised, with real-context margins included.
    - n_members (int): FiLM ensemble size (= emb_dim); the member embedding is one-hot of this length.
    - device (torch.device): compute device.

    Returns:
    - feats (np.ndarray): (H, W, D) penultimate features (float32).
    - pred (np.ndarray): (H, W) biomass prediction (float32).
    """
    body = net.model.body  # XceptionS2_FiLM
    captured = {}

    def pre_hook(module, inputs) :
        # inputs is a 1-tuple holding the (B, D, H, W) activation fed to the predictions conv.
        captured["feats"] = inputs[0].detach()

    handle = body.predictions.register_forward_pre_hook(pre_hook)
    try :
        x = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device).float()  # (1, C, H, W)
        b = torch.zeros(1, n_members, device=device)
        b[0, MEMBER] = 1.0
        with torch.no_grad() :
            pred = net((x, b))
    finally :
        handle.remove()

    feats = captured["feats"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, D)
    pred = pred[0, 0].cpu().numpy().astype(np.float32)  # (H, W)
    return feats, pred


def save_feats(out_dir, tile, which, feats, valid, transform, crs) :
    """
    Save the raw penultimate activations and their validity mask.

    Deliberately raw, not already reduced to RGB: PCA, zoom level and colour mapping are
    *presentation* choices, and keeping them out of here means make_feature_figure.py can re-crop
    and re-fit the PCA in seconds instead of re-running the models (a full S2 tile load is minutes
    and ~15 GB). The georeferencing of the crop is recorded alongside so the figure can cut the
    matching Sentinel-2 window.

    Args:
    - out_dir (str): destination directory.
    - tile (str): MGRS tile.
    - which (str): 'aef' or 'agbd'.
    - feats (np.ndarray): (H, W, D) activations.
    - valid (np.ndarray): (H, W) bool.
    - transform (Affine), crs: georeferencing of the crop.
    """
    np.save(join(out_dir, f"{tile}_{which}_feat.npy"), feats.astype(np.float32))
    np.save(join(out_dir, f"{tile}_{which}_valid.npy"), valid.astype(bool))
    with open(join(out_dir, f"{tile}_{which}_geo.json"), "w") as f :
        json.dump({"transform": list(transform)[:6], "crs": str(crs),
                   "height": int(feats.shape[0]), "width": int(feats.shape[1])}, f, indent=2)


###################################################################################################
# Crop geometry

def read_ref(tile) :
    """
    Read a crop-centring reference on the AEF window grid: CCI, else AEF pred, else AGBD pred.

    Args:
    - tile (str): MGRS tile, e.g. '59GPM'.

    Returns:
    - (np.ndarray or None, str): the reference biomass array (NaN nodata) and a label of its source,
      or (None, 'window-center') if nothing is available.
    """
    candidates = [
        (join(CCI_DIR, f"{tile}_CCI.tif"), "CCI"),
        (join(PRED_AEF, f"*/{tile}.tif"), "AEF-pred"),
        (join(PRED_AGBD, f"*/*_T{tile}_*.tif"), "AGBD-pred"),
    ]
    for pat, label in candidates :
        hits = sorted(glob.glob(pat))
        if hits :
            with rs.open(hits[0]) as s :
                a = s.read(1).astype(np.float32)
                if s.nodata is not None : a[a == s.nodata] = np.nan
            return a, label
    return None, "window-center"


def pick_center(ref, shape) :
    """
    Choose the crop origin (top-left row, col) in AEF-window pixel coords.

    Args:
    - ref (np.ndarray or None): reference biomass on the AEF grid, or None.
    - shape (tuple): (H, W) of the AEF window.

    Returns:
    - (int, int): (row, col) top-left of the CROP x CROP crop.
    """
    H, W = shape
    if ref is None or ref.shape != (H, W) :
        return (H - CROP_PX) // 2, (W - CROP_PX) // 2

    best, best_rc = -np.inf, ((H - CROP_PX) // 2, (W - CROP_PX) // 2)
    for r in range(0, H - CROP_PX + 1, STRIDE) :
        for c in range(0, W - CROP_PX + 1, STRIDE) :
            block = ref[r:r + CROP_PX, c:c + CROP_PX]
            valid = np.isfinite(block)
            if valid.mean() < MIN_VALID :
                continue
            mean_bio = np.nanmean(block)
            if mean_bio > best :
                best, best_rc = mean_bio, (r, c)
    return best_rc


def crop_with_margin(arr, r, c) :
    """
    Slice a CROP x CROP crop at (r, c) with a MARGIN of real data on every side, clamped to the array.

    Args:
    - arr (np.ndarray): (H, W, C) source.
    - r, c (int): crop top-left in `arr` pixel coords.

    Returns:
    - (np.ndarray, int, int): the (with-margin) slice, and the row/col offset of the crop *within* it.
    """
    H, W = arr.shape[:2]
    r0, c0 = max(0, r - MARGIN), max(0, c - MARGIN)
    r1, c1 = min(H, r + CROP_PX + MARGIN), min(W, c + CROP_PX + MARGIN)
    return arr[r0:r1, c0:c1], r - r0, c - c0


###################################################################################################
# I/O

def write_gray(path, arr, transform, crs, nodata=-9999.0) :
    """Write an (H, W) float array as a 1-band float32 GeoTIFF."""
    H, W = arr.shape
    out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    meta = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                crs=crs, transform=transform, compress="lzw", nodata=nodata)
    with rs.open(path, "w", **meta) as dst :
        dst.write(out, 1)


def read_s2_rgb(product, path_s2, bounds, crs) :
    """
    Read a true-colour (B04, B03, B02) crop from a Sentinel-2 SAFE, windowed to `bounds`.

    Args:
    - product (str): S2 L2A product name.
    - path_s2 (str): directory holding <product>.SAFE.
    - bounds (tuple): (left, bottom, right, top) in `crs`.
    - crs: the CRS of `bounds` (must equal the S2 tile's; asserted).

    Returns:
    - (np.ndarray, Affine): (h, w, 3) uint8 percentile-stretched RGB and its transform.
    """
    img_dir = glob.glob(join(path_s2, product + ".SAFE", "GRANULE", "*", "IMG_DATA", "R10m"))[0]
    _, _, date, _, _, tname, _ = product.split("_")
    ext = "jp2" if glob.glob(join(img_dir, f"{tname}_{date}_B02_10m.jp2")) else "tif"

    chans, out_transform = [], None
    for band in ("B04", "B03", "B02") :
        with rs.open(join(img_dir, f"{tname}_{date}_{band}_10m.{ext}")) as src :
            assert src.crs == crs, f"S2 {band} is {src.crs}, expected {crs}"
            win = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=win).astype(np.float32)
            if out_transform is None :
                out_transform = src.window_transform(win)
        chans.append(data)
    rgb = np.stack(chans, axis=-1)
    # Percentile stretch (a plain true-colour view; not used for any quantitative claim).
    lo, hi = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return (rgb * 255).astype(np.uint8), out_transform


###################################################################################################
# Per-model runners

def run_aef(tile, cfg, paths, device, r, c, out_dir, aef_meta) :
    """Extract AEF-model PCA-RGB features for the chosen crop; write GeoTIFFs. Returns bounds."""
    with open(join(paths["aef_norm"], "AEF-statistics.pkl"), "rb") as f :
        import pickle ; norm = pickle.load(f)
    data, mask, _ = AEF.load_input(paths, f"{tile}/{tile}", norm, masking=True)  # (H, W, 64)

    sub, rr, cc = crop_with_margin(data, r, c)
    valid_full = ~mask if mask is not None else np.ones(data.shape[:2], bool)
    vsub, _, _ = crop_with_margin(valid_full[..., None], r, c)
    vsub = vsub[..., 0]

    net = load_net("aef", cfg, paths, device)
    feats, pred = extract_penultimate(net, sub, cfg.get("n_members", 3), device)
    # Slice the CROP x CROP centre back out (margins were only for exact border features).
    feats = feats[rr:rr + CROP_PX, cc:cc + CROP_PX]
    pred = pred[rr:rr + CROP_PX, cc:cc + CROP_PX]
    valid = vsub[rr:rr + CROP_PX, cc:cc + CROP_PX]

    transform = window_transform(Window(c, r, CROP_PX, CROP_PX), aef_meta["transform"])
    save_feats(out_dir, tile, "aef", feats, valid, transform, aef_meta["crs"])
    write_gray(join(out_dir, f"{tile}_aef_pred.tif"), np.where(valid, pred, np.nan), transform, aef_meta["crs"])
    print(f"  [aef]  feats {feats.shape}  valid {valid.mean():.3f}")
    return array_bounds(CROP_PX, CROP_PX, transform), aef_meta["crs"]


def run_agbd(tile, product, cfg, paths, device, bounds, crs, out_dir) :
    """Extract AGBD-features-model PCA-RGB for the same geo `bounds`; write GeoTIFFs."""
    import pickle
    new_stats = cfg.get("new_stats", False) or cfg.get("prob_norm", False)
    with open(join(paths["norm"], f"statistics_subset_2019-2020-v4{'-1' if new_stats else ''}.pkl"), "rb") as f :
        norm = pickle.load(f)
    embeddings = None
    if cfg.get("ft_cat2vec", False) :
        import pandas as pd
        emb_dir = paths["embeddings"] + ("/AGBD-Lite" if cfg.get("lite", False) else "/AGBD")
        e = pd.read_csv(join(emb_dir, "embeddings_train.csv"))
        embeddings = {v: np.array([a, b, cc, d, ee]) for v, a, b, cc, d, ee in
                      zip(e.mapping, e.dim0, e.dim1, e.dim2, e.dim3, e.dim4)}

    # FiLM ensemble: load_input must not append the biome tuple (we drive members ourselves).
    ds_cfg = {**cfg, "film": False}
    data, mask, meta = AGBD.load_input(2020, paths, tile, product, norm, ds_cfg, embeddings=embeddings)
    data = data.numpy() if hasattr(data, "numpy") else np.asarray(data)  # (H, W, 31)

    # Map the AEF-defined geo bounds to this S2 tile's pixel grid.
    win = from_bounds(*bounds, transform=meta["transform"])
    r, c = int(round(win.row_off)), int(round(win.col_off))
    assert meta["crs"] == crs, f"S2 tile is {meta['crs']}, AEF window is {crs}"

    sub, rr, cc = crop_with_margin(data, r, c)
    valid_full = ~mask
    vsub, _, _ = crop_with_margin(valid_full[..., None], r, c)
    vsub = vsub[..., 0]

    # Free the ~15 GB full-tile array (the crop slices are views into it) before the forward pass,
    # so a full S2 tile's memory is not held alongside anything else. Run this model in its OWN
    # process, separate from the AEF one: together they OOM this box (load_input peaks ~4x the tile).
    import gc
    sub, vsub = np.ascontiguousarray(sub), np.ascontiguousarray(vsub)
    del data, valid_full, mask ; gc.collect()

    net = load_net("agbd", cfg, paths, device)
    feats, pred = extract_penultimate(net, sub, cfg.get("n_members", 3), device)
    feats = feats[rr:rr + CROP_PX, cc:cc + CROP_PX]
    pred = pred[rr:rr + CROP_PX, cc:cc + CROP_PX]
    valid = vsub[rr:rr + CROP_PX, cc:cc + CROP_PX]

    transform = window_transform(Window(c, r, CROP_PX, CROP_PX), meta["transform"])
    save_feats(out_dir, tile, "agbd", feats, valid, transform, crs)
    write_gray(join(out_dir, f"{tile}_agbd_pred.tif"), np.where(valid, pred, np.nan), transform, crs)
    # True-colour backdrop, same crop.
    s2rgb, s2t = read_s2_rgb(product, paths["tiles"], bounds, crs)
    meta_rgb = dict(driver="GTiff", height=s2rgb.shape[0], width=s2rgb.shape[1], count=3,
                    dtype="uint8", crs=crs, transform=s2t, compress="lzw")
    with rs.open(join(out_dir, f"{tile}_s2rgb.tif"), "w", **meta_rgb) as dst :
        for i in range(3) : dst.write(s2rgb[:, :, i], i + 1)
    print(f"  [agbd] feats {feats.shape}  valid {valid.mean():.3f}")


###################################################################################################
# Main

def main() :
    global CROP_PX
    p = argparse.ArgumentParser()
    p.add_argument("--tile", required=True)
    p.add_argument("--product", required=True, help="S2 L2A product name (no extension) for the AGBD model + RGB.")
    p.add_argument("--which", choices=["aef", "agbd", "both"], default="both")
    p.add_argument("--out_dir", default=join(dirname(abspath(__file__)), "..", "comparison", "maps", "features"))
    p.add_argument("--center", nargs=2, type=int, default=None, help="Override crop top-left '<row> <col>' in AEF-window px.")
    p.add_argument("--crop_px", type=int, default=CROP_PX,
                   help="Size of the extracted crop, in 10 m px. Extract wide (default 320 = 3.2 km); "
                        "make_feature_figure.py zooms into a sub-window of this without re-running the models.")
    args = p.parse_args()
    CROP_PX = args.crop_px

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    paths = get_paths(local=True)
    paths["aef"] = AEF_TILES_DIR  # point AEF loader at the 40 km windows, not the AGBRef plots

    runs = load_cache(ARCH)
    cfg_aef = runs[AEF_MODEL]["config"]
    cfg_agbd = runs[AGBD_MODEL]["config"]

    # AEF window metadata + crop choice (the AEF window defines the shared extent, as in the map figure).
    with rs.open(join(AEF_TILES_DIR, args.tile, f"{args.tile}.tiff")) as s :
        aef_meta = {"transform": s.transform, "crs": s.crs, "height": s.height, "width": s.width}

    if args.center is not None :
        r, c = args.center
        ref_label = "manual"
    else :
        ref, ref_label = read_ref(args.tile)
        r, c = pick_center(ref, (aef_meta["height"], aef_meta["width"]))
    print(f"Tile {args.tile}: crop top-left (row,col)=({r},{c}) size {CROP_PX}px  ref={ref_label}")

    bounds = crs = None
    if args.which in ("aef", "both") :
        print("Extracting AEF features...")
        bounds, crs = run_aef(args.tile, cfg_aef, paths, device, r, c, args.out_dir, aef_meta)

    if args.which in ("agbd", "both") :
        print("Extracting AGBD features...")
        if bounds is None :
            # AEF not run this invocation: recover the crop bounds from the AEF window transform.
            tr = window_transform(Window(c, r, CROP_PX, CROP_PX), aef_meta["transform"])
            bounds, crs = array_bounds(CROP_PX, CROP_PX, tr), aef_meta["crs"]
        run_agbd(args.tile, args.product, cfg_agbd, paths, device, bounds, crs, args.out_dir)

    # Record what was done, for the figure caption and reproducibility.
    with open(join(args.out_dir, f"{args.tile}_meta.json"), "w") as f :
        json.dump({"tile": args.tile, "product": args.product, "crop_px": CROP_PX,
                   "origin_rowcol": [int(r), int(c)], "ref": ref_label, "member": MEMBER,
                   "bounds": [float(b) for b in bounds] if bounds else None,
                   "crs": str(crs) if crs else None}, f, indent=2)
    print("done!")


if __name__ == "__main__" :
    main()
