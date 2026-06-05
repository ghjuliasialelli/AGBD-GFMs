# AGBD-GFMs

Unified codebase for **"Above-ground Biomass Estimation with Geospatial Foundation Models"**
(Sialelli, Scheibenreif, Wegner, Schindler).

## Repository structure

```
AGBD-GFMs/
├── environment.yml / requirements.txt   # Python environment (Biomes model)
│
├── data/                       # dataset + pre-computed embedding products
│   ├── agbd_lite/              # AGBD-Lite construction, subsampling, regional stats
│   │   └── eval/               # AGBD-Lite eval launchers + configs (.sh/.txt only)
│   ├── aef/                    # AlphaEarth Foundations (AEF) download & patch creation
│   └── tessera/                # TESSERA embedding download & statistics
│
├── benchmark_pangaea/          # 11-GFM benchmark in the PANGAEA framework
│   ├── train_runs/  test_runs/  evalbig_runs/  full/  throughput/
│
├── model/                      # supervised SOTA + embedding heads (flat package)
│   ├── train.py  eval.py
│   ├── models.py              # arch registry: lp / mlp / nico_film  (trimmed — see Notes)
│   ├── nico_net_film.py       # NicoNet + FiLM (supervised SOTA)
│   ├── mlp.py  lp.py          # MLP / linear-probing heads for AEF & TESSERA embeddings
│   ├── wrapper.py  wrapper_ts.py  loss.py  biomes.py  parser.py  weights_mapping.py
│   ├── dataset.py  dataset_biomes_sampler.py  dataset_ts.py
│   ├── aef_importance.py + aef/*.npy     # AEF band-importance analysis
│   ├── inference_aef.py + inference_aef.sh   # inference on AEF embeddings (+ helpers)
│   ├── eval/                  # eval.sh + configs/ + non-ablation run launchers
│   └── runs/                  # training launchers: film/ nico/ lp-mlp/
│
├── experiments/
│   └── generalization/        # geographical + temporal ablation launchers
│       ├── train_ablations/   eval_ablations/
│
└── comparison/
    └── agbref/                # AGBRef + ESA CCI biomass-map comparison
```

## Quick start

```bash
conda env create -f environment.yml
pip install -r requirements.txt

# Train the supervised SOTA model (NicoNet + FiLM)
python model/train.py --arch nico_film --dataset_path /path/to/AGBD ...

# Train an embedding head on AEF / TESSERA
python model/train.py --arch mlp --dataset_path /path/to/AEF ...   # or --arch lp

# Evaluate
bash model/eval/eval.sh

# Inference on AEF embeddings
bash model/inference/inference_aef.sh
```

Run-launcher `.sh` files under `model/runs/`, `model/eval/runs/`, and
`experiments/generalization/` contain the exact hyper-parameters used for each
table/experiment in the paper. Paths inside them are SLURM/cluster paths and must be
updated for your environment.


## Large artifacts not bundled (>10 MB)

To keep the repo lightweight, files larger than 10 MB were excluded. Retrieve them from the
original repositories if needed:

- `data/agbd_lite/eval/weights/*.ckpt` — trained AGBD-Lite checkpoints (3 × ~18 MB)
- `data/agbd_lite/eval/results/*.h5` — cached prediction results (~33 MB)
- `data/agbd_lite/mapping_lite_to_og.pkl`, `data/aef/mapping_lite_to_og.pkl` (~15 MB each)
- Various output-heavy notebooks from the source repos.
