# PANGAEA benchmark

This directory holds only the **launchers** for our geospatial foundation model (GFM)
benchmark: an 11-model comparison on AGBD-Lite, plus a single SSL4EO-MoCo run on the
full AGBD dataset. The benchmark itself runs inside the
[PANGAEA](https://github.com/VMarsocci/pangaea-bench) framework — the AGBD / AGBD-Lite
dataset classes, the encoder code and configs, and `pangaea/run.py` live in our fork,
not here.

## The fork

Everything in this directory is meant to be run from a checkout of our fork. A single
fork reproduces the **entire** paper (both the AGBD-Lite benchmark and the full-AGBD
run):

- **Repo:** https://github.com/ghjuliasialelli/pangaea-bench
- **Branch / tag:** `agbd-release` (tag `agbd-gfm-paper`)
- **Pinned commit:** `b9470d05e9a9c6aea1fd9cb0b57f5e8fd5c7b23b`

```bash
git clone https://github.com/ghjuliasialelli/pangaea-bench.git
cd pangaea-bench
git checkout agbd-gfm-paper       # or: git checkout b9470d0
# then follow the fork's README to set up the environment
```

The fork is a derivative of `VMarsocci/pangaea-bench` and inherits its **GPL-3.0**
license. `agbd-release` is one squashed commit on top of upstream `ca19b83` containing
all of our additions; the messy `agbd` development branch is preserved for history.

### What the fork adds over upstream PANGAEA

- `pangaea/datasets/agbdlite.py`, `agbdlite-seg.py` — AGBD-Lite (regression / seg).
  Auto-downloads from Zenodo (record `18485030`) on first use.
- `pangaea/datasets/agbd.py`, `configs/dataset/agbd.yaml` — the **full** AGBD dataset
  (used only by `full/`; see below).
- `configs/dataset/agbdlite*.yaml`, `configs/preprocessing/reg_resize.yaml`.
- Encoder code/configs not shipped upstream: Prithvi-2 (`prithvi2_encoder.py`, IBM
  Apache-2.0 header preserved), THOR, TerraMind, plus a SpectralGPT shape fix.
- `pangaea/engine/trainer.py`: fractional `eval_interval` (intra-epoch validation).

The 11 benchmarked encoders are: `croma_optical`, `dofa_optical`, `gfmswin`, `prithvi`,
`remoteclip`, `satlasnet_si`, `scalemae`, `spectralgpt`, `ssl4eo_moco`,
`terramind_optical_tiny`, `prithvi2_100m`. All 11 — plus `decoder=reg_upernet`,
`task=regression`, `criterion=mse`, and both dataset configs — resolve at the pinned
commit.

### The SAR runs

The paper's 11 encoders are optical-only. The AGBD datasets also carry a SAR modality
(ALOS-PALSAR-2 gamma naught, HH/HV), which the dataset configs list under the band names
`VV`/`VH` so the Sentinel-1-pretrained encoders match it — co-pol to co-pol, cross-pol to
cross-pol. That pairing is a real domain shift (L-band vs C-band), not a like-for-like
substitution.

Three encoders consume it: **`croma_joint`**, **`terramind_tiny`** and **`dofa_joint`**.
`train_runs/gen.py` and `throughput/gen.py` hold them in a `SAR_ENCODERS` list, separate
from `OPTICAL_ENCODERS`, so the optical benchmark stays reproducible on its own.

- `croma_joint` and `terramind_tiny` load the *same weight files* as `croma_optical` and
  `terramind_optical_tiny`. Both upstream checkpoints are single multimodal models (CROMA
  ships `s1_encoder` / `s2_encoder` / `joint_encoder` in one file; TerraMind ships
  per-modality embeddings), and each encoder class loads a different subset. No extra
  downloads are needed.
- `dofa_optical` / `dofa_joint` are ours. Upstream `dofa.yaml` takes `${dataset.bands}`,
  so on a dataset carrying SAR it silently becomes multimodal, under a run-directory name
  indistinguishable from an optical run. The two explicit configs avoid that. **The
  paper's DOFA runs are `dofa_optical`.**
- These runs need a fork newer than the pinned commit above: the pin has neither the
  SAR-enabled AGBD dataset configs nor `dofa_optical.yaml` / `dofa_joint.yaml`.

## How to use these launchers

Copy the relevant `.sh` files into the fork checkout and submit them there. They are
SLURM scripts written for our cluster — the `#SBATCH` directives, log paths, and GPU
type (`rtx_4090:2`) must be edited for your environment.

Each directory has a `gen.py`: running it regenerates that directory's per-encoder `.sh`
files *and* prints the `sbatch …` commands to submit them (one per encoder). Edit the
generator, not the generated scripts. `evalbig_runs/` additionally ships a `run_all.sh`.

| directory | what it does |
|---|---|
| `train_runs/` | trains each of the 11 encoders on AGBD-Lite (`--config-name=train dataset=agbdlite encoder=<e> decoder=reg_upernet preprocessing=reg_resize task=regression`). One run per encoder. |
| `test_runs/` | evaluates a trained run on the AGBD-Lite test split (`--config-name=test ckpt_dir=<run>`). |
| `evalbig_runs/` | evaluates a trained run on the full **AGBD-test** set instead of the Lite test split (`ckpt_dir=<run>_evalbig`). |
| `gen_evalbig.sh` | prepares the `_evalbig` run dirs: copies each trained run listed in `configs.txt` and flips its config to evaluate on AGBD-test. |
| `throughput/` | inference throughput + the RMSE-vs-throughput Pareto figure. See `throughput/README.md`. |
| `full/` | the **SSL4EO-MoCo run trained on the full AGBD dataset** (`dataset=agbd`), a result reported in the paper. See below. |

`configs.txt` lists the trained-run output directory names (timestamped, one per
encoder). `test_runs/` and `evalbig_runs/` reference these names via `ckpt_dir=…`, so
regenerate them if your run names differ.

### The full-AGBD run (`full/`)

`full/` trains and evaluates SSL4EO-MoCo on the **full** AGBD dataset (`dataset=agbd`),
as opposed to the AGBD-Lite subset used for the 11-model benchmark. It runs from the same
fork checkout as everything else.

Unlike AGBD-Lite (which auto-downloads from Zenodo), the full-AGBD run needs data that is
**not** bundled or auto-fetched: the full-AGBD `.h5` files plus `biomes_splits_to_name.pkl`,
`AEF_overlaps.pkl`, and `tiles_per_region.pkl`. `full/train_agbd.sh` stages these to
`$TMPDIR` via `rclone`/`cp` and passes `dataset.root_path_cluster=${TMPDIR}`; adapt those
paths to wherever you hold the full dataset. It also uses `task.trainer.eval_interval=0.25`
(the fractional-eval feature added to `trainer.py`).

## Reproducing the benchmark end to end

```bash
# inside the pinned pangaea-bench checkout, with these launchers copied in:

# 1. train all 11 encoders. `python gen.py` writes the per-encoder .sh files and
#    prints the sbatch commands; submit them.
cd train_runs && python gen.py        # then submit the printed `sbatch <encoder>.sh` lines
#   -> each run produces a timestamped output dir; collect those names into configs.txt

# 2. evaluate each trained run on the AGBD-Lite test split (reads configs.txt)
cd test_runs && python gen.py         # then submit the printed sbatch lines

# 3. (optional) evaluate on the full AGBD-test set instead of the Lite split
bash gen_evalbig.sh                   # prepares the _evalbig run dirs from configs.txt
bash evalbig_runs/run_all.sh

# 4. (optional) the full-AGBD SSL4EO-MoCo run — needs the full dataset staged (see above)
sbatch full/train_agbd.sh
sbatch full/eval.sh
```

AGBD-Lite downloads itself from Zenodo on first run (`auto_download: True` in
`configs/dataset/agbdlite.yaml`), so no manual data setup is needed for the Lite runs.
The full-AGBD run (`full/`) is the exception — see above.
