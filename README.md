# AGBD-GFMs

Unified codebase for **"Above-ground Biomass Estimation with Geospatial Foundation Models"**
(Sialelli, Scheibenreif, Wegner, Schindler).

## Repository structure

```
AGBD-GFMs/
├── pyproject.toml / uv.lock    # Python environment (uv — the supported route)
├── environment.yml / requirements.txt   # conda / pip equivalents, unpinned
│
├── data/                       # dataset construction + auxiliary files (see "Data")
│   ├── patches/                # THE dataset dir under the `local` profile
│   │   ├── biomes_splits_to_name.pkl    # train/val/test split, as S2 tile names
│   │   ├── *_statistics_*.pkl           # normalisation stats (tracked; .h5 are not)
│   │   └── AEF/  TESSERA/               # + mapping_lite_to_og.pkl under AEF/
│   ├── agbd_lite/              # AGBD-Lite construction, subsampling, regional stats
│   │   ├── indices/            # the exact indices sampled to build AGBD-Lite
│   │   ├── tile_to_region.pkl  tile_to_file.pkl  tiles_per_region.pkl
│   │   ├── eval/               # eval launcher + run-ID configs
│   │   │   ├── weights/        # trained supervised checkpoints (.ckpt)
│   │   │   ├── results/        # cached predictions (.h5) — plots need no GPU
│   │   │   └── embeddings/     # cat2vec land-cover embeddings
│   │   ├── helper/             # plotting + exploration
│   │   └── hf/                 # dataset card published as prs-eth/AGBD-Lite
│   ├── aef/                    # AlphaEarth Foundations (AEF) download & patch creation
│   └── tessera/                # TESSERA embedding download & statistics
│
├── benchmark_pangaea/          # 11-GFM benchmark in the PANGAEA framework
│   ├── train_runs/  test_runs/  evalbig_runs/  full/  throughput/
│
├── model/                      # supervised SOTA + embedding heads (flat package)
│   ├── train.py  eval.py
│   ├── models.py              # arch registry: lp / mlp / nico_film
│   ├── nico_net_film.py       # NicoNet + FiLM (supervised SOTA)
│   ├── mlp.py  lp.py          # MLP / linear-probing heads for AEF & TESSERA embeddings
│   ├── wrapper.py  loss.py  biomes.py  parser.py  weights_mapping.py
│   ├── dataset.py
│   ├── aef_importance.py + aef/*.npy     # AEF band-importance analysis
│   ├── inference_aef.py  inference_ds.py  inference_residuals.py  inference_helper.py
│   ├── inference/            # inference_aef.sh + per-tile lists
│   ├── eval/                  # eval.sh + configs/ + non-ablation run launchers
│   └── runs/                  # training launchers: nico/ lp-mlp/ + gen_launchers.py
│
├── experiments/
│   └── generalization/        # geographical + temporal ablation launchers
│       ├── train_ablations/   eval_ablations/
│
├── comparison/                 # paper figure scripts (each writes into its own plots/)
│   ├── agbref/                # AGBRef + ESA CCI biomass-map comparison
│   └── maps/                  # prediction-map + feature-PCA figures, GEDI per-tile metrics
│       ├── make_map_figure.py  make_feature_figure.py
│       ├── gedi_gpkg_tile_metrics.py  tile_metrics.py  gedi_scatter.py
│       └── features/  plots/  # cached crops + rendered figures
│
└── tools/
    └── check_flags.py         # launcher flag consistency check
```

## Data

None of the datasets are bundled here — they are far too large. This is where each one
comes from:

| dataset | source | used by |
|---|---|---|
| **AGBD** (full) | [`prs-eth/AGBD`](https://huggingface.co/datasets/prs-eth/AGBD) on HuggingFace. Manual download (`auto_download: False`). | supervised runs with `lite="false"`; the full-AGBD SSL4EO-MoCo benchmark run |
| **AGBD-Lite** | [`prs-eth/AGBD-Lite`](https://huggingface.co/datasets/prs-eth/AGBD-Lite) on HuggingFace, mirrored on Zenodo record [`18485030`](https://zenodo.org/records/18485030). Auto-downloads on first use inside the benchmark fork; download by hand for the supervised code here. | the 11-GFM benchmark; all `lite="true"` runs |
| **AEF** (AlphaEarth Foundations) | [`source.coop/tge-labs/aef`](https://source.coop/tge-labs/aef). Fetch with `data/aef/download.py`, then build patches with `data/aef/create_patches.py`. | all AEF experiments |
| **TESSERA** | [`ucam-eo/geotessera`](https://github.com/ucam-eo/geotessera). Built from the AGBD-Lite `.h5` by `data/tessera/download.py`. | all TESSERA experiments |
| **ESA CCI biomass** | ESA CCI Biomass, per-S2-tile rasters (`CCI_<tile>_19.tif`, ~100 m, EPSG:4326), pointed to by `CCI_DIR` in `comparison/agbref/comparison.py`. | `comparison/agbref/` only |
| **AGBRef** | in-repo: `comparison/agbref/data/` (`AGBRef.geojson`, `AGBref.gpkg`, the `.Rdata`). **Not** included: the 5 Sentinel-2 true-colour tiles `make_plot_maps.py` draws on — see [Sentinel-2 tiles for the AGBRef figure](#sentinel-2-tiles-for-the-agbref-figure) below. | `comparison/agbref/` only |

Dataset statistics (`AGBD-Lite-statistics.pkl`, `AGBD_statistics_2019-2020_global.pkl`,
`statistics_subset_2019-2020-v4*.pkl`) ship with the datasets above, and can also be
regenerated with the `compute_statistics.py` scripts under `data/`.

### Auxiliary files (these *are* in the repo)

The runs need several small mapping/split files. All of them are tracked here, so you do
not need to hunt for them:

| file | what it is |
|---|---|
| `data/patches/biomes_splits_to_name.pkl` | train/val/test split as S2 tile names (306/67/106). Read by `dataset.py`, `gen_subsample.py`, `inference_residuals.py`. |
| `data/patches/AEF/mapping_lite_to_og.pkl` | AGBD-Lite index → AGBD index. Regenerable with `data/agbd_lite/get_mapping_lite_to_og.py`. |
| `data/agbd_lite/eval/embeddings/{AGBD,AGBD-Lite}/embeddings_train.csv` | cat2vec land-cover embeddings used by the FiLM layers. |
| `data/aef/AEF_overlaps.pkl` | AGBD test patches that overlap AEF's training set; removed when `drop_overlaps="true"`. |
| `data/agbd_lite/tiles_per_region.pkl`, `tile_to_region.pkl`, `tile_to_file.pkl` | tile ↔ region ↔ `.h5` mappings. |
| `data/agbd_lite/indices/subsampled_indices_*.pkl` | the exact indices sampled to build AGBD-Lite, per region. |
| `data/agbd_lite/eval/weights/*.ckpt` | trained supervised (NicoNet+FiLM) checkpoints. |
| `data/agbd_lite/eval/results/*.h5` | cached predictions, so the plots can be reproduced without a GPU. |

### Sentinel-2 tiles for the AGBRef figure

`comparison/agbref/make_plot_maps.py` renders a Sentinel-2 true-colour column alongside the
biomass maps. The tiles are ~130 MB each, so they are **not** in the repo — put them in
`comparison/agbref/data/s2_tci/` and the script picks them up automatically. These 5 are exactly
the scenes the published figure draws, one per plot in `DEFAULT_PLOTS`. All are Level-2A TCI
(`B04/B03/B02`, pre-composited 8-bit, 10 m, 10980×10980), used as-is with no restretch:

| file | tile CRS | serves | region |
|---|---|---|---|
| `T12SYH_20200623T175919_TCI_10m.jp2` | EPSG:32612 | plot 304 | SW Colorado, USA |
| `T12SYG_20200623T175919_TCI_10m.jp2` | EPSG:32612 | plot 293 | SW Colorado, USA |
| `T32NQK_20180117T093321_TCI_10m.jp2` | EPSG:32632 | plot 26 | Cameroon |
| `T17SNB_20180708T155819_TCI_10m.jp2` | EPSG:32617 | plot 290 | Virginia, USA |
| `T33NTD_20180104T092351_TCI_10m.jp2` | EPSG:32633 | plot 11 | Cameroon |

md5, in the order above:

```
593766328e9b887ec5f66a7b3e356bb7  T12SYH_20200623T175919_TCI_10m.jp2
2c629862fbea26cfe75d54217bcb0eb0  T12SYG_20200623T175919_TCI_10m.jp2
e9aa00cd18af218bc24c2e777978620b  T32NQK_20180117T093321_TCI_10m.jp2
0512c955cc822593be009e46274ad056  T17SNB_20180708T155819_TCI_10m.jp2
3fc0e4df41d745055a810bfffe2603c1  T33NTD_20180104T092351_TCI_10m.jp2
```

Download from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/) (search the tile
ID and acquisition datetime, take `IMG_DATA/R10m/*_TCI_10m.jp2` from the L2A product). **Match on
the acquisition timestamp in the filename, not the product name** — the same acquisition is
republished under different processing baselines (`N0500` vs `N9999`), which changes the product
name but not the TCI bytes you need.

`pick_s2()` scores every scene it finds for a tile by cloud+fill fraction over the plot cell and
takes the lowest, printing its choice. With exactly these 5 files each plot has a single
candidate, so the selection is forced and reproduces the published panels. Adding further scenes
for the same tile is safe — the scoring decides — but the extra files are not needed.

**Note on the launchers:** the `.sh` files stage these inputs into `$TMPDIR` by copying from
the authors' cluster paths (`/cluster/work/igp_psr/gsialelli/...`), which you cannot read.
Point those `cp`/`rclone` lines at your own copies before running.


## Configuration

All machine-specific settings live in one file: [`config.sh`](config.sh). Both halves of the
codebase read it — the shell launchers `source` it, and Python gets the same values through
`config.py`, which sources that very file rather than re-parsing it, so the two can never
disagree.

It ships with the authors' paths, so the published runs stay fully specified. To run
elsewhere, either edit `config.sh`, or override any single key from the environment — the
environment always wins, so you never have to modify the file to try something:

```bash
# point at your own data without touching config.sh
AGBD_LOCAL_ROOT=/my/data bash model/runs/nico/agbd.sh

# or override one key
AGBD_LOCAL_H5=/somewhere/else python model/train.py ...
```

There are three profiles, because the machines lay the data out differently:

| profile | for | data layout |
|---|---|---|
| `local` | **anyone else — the portable default** | everything inside this repo, under `./data` |
| `pfpc28` | the authors' workstation | spread across `/scratch3/gsialelli` |
| `euler` | the authors' SLURM cluster | `/cluster/work/igp_psr`, staged into `$TMPDIR` |

The profile is auto-detected (hostname `pf-pc28` → `pfpc28`; a working directory under
`/cluster` → `euler`; anything else → `local`). Force it with `AGBD_PROFILE=local|pfpc28|euler`.
Setting `AGBD_LOCAL_ROOT` / `AGBD_CLUSTER_WORK` cascades to every path derived from it, so a
fresh machine usually needs only those.

`local` and `pfpc28` share the same code path (`AGBD_ENV=local`): the Python entry points read
the data straight from the `AGBD_LOCAL_*` paths. `euler` uses `AGBD_ENV=cluster`, where the
launchers stage the data into `$TMPDIR` first.

> The `#SBATCH --output/--error` lines in the launchers stay literal: SLURM parses those
> comments before bash runs, so they cannot read a variable. They only matter on `euler`.

> **Your wandb API key does not go in `config.sh`** — that file is committed to git.
> `config.sh` holds only the entity and project. wandb reads the key itself from `~/.netrc`
> or `$WANDB_API_KEY`.


## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/), and `uv.lock` is committed, so
this resolves to the exact versions the paper was produced with:

```bash
uv sync                    # creates .venv/ from uv.lock (python 3.10.9, torch+cu118)

# then set your paths (see Configuration above)
$EDITOR config.sh
```

Run things with `uv run <cmd>` (no manual activation needed), or `source .venv/bin/activate` if
you prefer. The launcher `.sh` files call `python` directly, so activate first when using those.

Two optional extras are separated out because they need system libraries and only a couple of
scripts use them — `pycurl` (libcurl + openssl headers) and `pyreadr` (reads the AGBRef `.Rdata`):

```bash
uv sync --extra extras
```

`environment.yml` and `requirements.txt` are kept as unpinned conda/pip equivalents for anyone who
cannot use uv. They describe the same dependency set but resolve versions freshly — pick **one**
route and do not mix them (running the pip one inside the conda env reinstalls torch over the
conda build).

This environment covers the supervised model, the data-prep scripts, and the figure scripts under
`comparison/`. Two things are deliberately outside it:

- **The PANGAEA benchmark** has its own dependencies and its own environment, installed from the
  fork (see `benchmark_pangaea/README.md`).
- **`geotessera`** requires python ≥ 3.11 and so cannot coexist with this project's 3.10 pin.
  `data/tessera/download.py` needs it — install it in a separate environment.

`model/train.py` takes ~90 flags and is not meant to be called by hand — the launcher `.sh`
files are the real entry point, and each one encodes the exact configuration of a paper
experiment:

```bash
# Supervised SOTA (NicoNet + FiLM) on AGBD
bash model/runs/nico/agbd.sh          # or: sbatch model/runs/nico/agbd.sh

# Embedding heads on AEF / TESSERA
bash model/runs/lp-mlp/mlp_aef.sh     # lp_* / mlp_* × aef / tessera

# Evaluate + plot; pick the experiment via `config=` (see model/eval/configs/*.txt)
bash model/eval/eval.sh

# Evaluate AGBD-Lite and write the cached prediction .h5 (config= best | lite | aef ...)
bash data/agbd_lite/eval/eval.sh

# Inference on AEF embeddings
bash model/inference/inference_aef.sh
```

There are two evaluators, and they do different things: `model/eval/eval.sh` is the general
one (produces plots, supports `drop_overlap`/`offset`, 12 configs under `model/eval/configs/`),
while `data/agbd_lite/eval/eval.sh` is AGBD-Lite specific and writes the cached prediction
`.h5` files into `--res_folder`. Both select their run IDs by reading a `.txt` config, so
`config="lite"` → `evaluation_nico_film_lite.txt` → run IDs `55140725-{1,2,3}`.

Which launcher maps to which experiment:

| directory | experiments |
|---|---|
| `model/runs/nico/` | supervised SOTA per input type: `agbd.sh`, `aef.sh`, `tessera_lite.sh`, `aef_agbd*.sh` (AEF+AGBD features), plus `_lite` / `_nolatlon` variants |
| `model/runs/lp-mlp/` | linear-probe / MLP heads on AEF & TESSERA embeddings |
| `experiments/generalization/train_ablations/` | geographic hold-out + temporal (2019/2020) ablations |
| `experiments/generalization/eval_ablations/` | evaluation counterparts of the above |
| `benchmark_pangaea/train_runs/` | the 11-GFM benchmark on AGBD-Lite (runs in the fork) |
| `benchmark_pangaea/full/` | SSL4EO-MoCo trained on full AGBD |
| `benchmark_pangaea/throughput/` | throughput / efficiency measurements |

### Regenerating the launchers

**The `.sh` launchers are generated — do not hand-edit them.** Each group has a generator that
owns a shared template plus a per-experiment knob table; edit the generator and re-run it.

| generator | emits |
|---|---|
| `model/runs/gen_launchers.py` | 18 training launchers (`nico/`, `lp-mlp/`) |
| `model/eval/runs/gen.py` | 24 evaluation launchers |
| `experiments/generalization/train_ablations/gen.py` | 15 ablation training launchers |
| `experiments/generalization/eval_ablations/gen.py` | 37 ablation evaluation launchers |

Each is idempotent: re-running reproduces the committed scripts exactly. Knob defaults match
`train.py`'s argparse defaults, so a knob an experiment does not set behaves exactly as if the
flag were never passed. Derived values (`in_features`, `emb_dim`, `num_outputs`, `film`) are
computed in bash by the template, not baked in, so changing a knob stays correct.

Paths inside the launchers are SLURM/cluster paths and must be updated for your
environment. The same applies to absolute paths in the data-prep and analysis scripts
(e.g. `data/aef/`, `comparison/agbref/comparison.py`). The wandb entity defaults to
`gs-tp-biomass` but can be overridden with the `WANDB_ENTITY` environment variable.

**wandb is required for evaluation:** `model/eval.py` resolves each run's checkpoint path
by querying the wandb API for the run's config, so re-running `eval.py` against the
original run IDs needs access to the authors' wandb project. Training does not have this
constraint.


## Large artifacts

Model weights and cached predictions are tracked in this repo (see the auxiliary-files
table above) — up to GitHub's 100 MB per-file limit.

One artifact exceeds that limit and is therefore **not** bundled:

- `benchmark_pangaea/full/runs/**/checkpoint__best.pth` (461 MB) — SSL4EO-MoCo trained on
  full AGBD. The run's `configs/config.yaml` *is* tracked, so the run is fully specified
  and can be reproduced from scratch.

<!-- TODO: upload the 461 MB full-AGBD checkpoint to the AGBD-Lite Zenodo record (18485030)
and link it here. -->

The predecessor repository
[`ghjuliasialelli/AGBD-GFM`](https://github.com/ghjuliasialelli/AGBD-GFM) is superseded by
this one; every artifact needed for the paper has been copied across.


## The PANGAEA benchmark lives in a fork

The multi-model benchmark runs inside the [PANGAEA](https://github.com/VMarsocci/pangaea-bench)
framework. `benchmark_pangaea/` here contains only the launchers; the dataset classes and
encoder code live in our fork,
[`ghjuliasialelli/pangaea-bench`](https://github.com/ghjuliasialelli/pangaea-bench) at tag
`agbd-gfm-paper` (commit `b9470d0`). See `benchmark_pangaea/README.md` for details.


## License

Released under the MIT License — see [`LICENSE`](LICENSE). Note that the PANGAEA benchmark
fork is a derivative of `VMarsocci/pangaea-bench` and is licensed separately under GPL-3.0.


## Citation

<!-- TODO: update once the arXiv preprint is posted (ID, URL, year). See CITATION.cff. -->

If you use this code, please cite:

```bibtex
@article{sialelli2026agbdgfm,
  title   = {Above-ground Biomass Estimation with Geospatial Foundation Models},
  author  = {Sialelli, Ghjulia and Scheibenreif, Linus and Wegner, Jan Dirk and Schindler, Konrad},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
