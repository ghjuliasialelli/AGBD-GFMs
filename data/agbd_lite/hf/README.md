---
license: cc-by-nc-4.0
viewer: false
task_categories:
- image-segmentation
size_categories:
- 100K<n<1M
tags:
- biomass
- above-ground-biomass
- remote-sensing
- earth-observation
- GEDI
- Sentinel-2
- ALOS-PALSAR
pretty_name: AGBD-Lite
---

# 🌲 AGBD-Lite: A Representative Subset of the AGBD Biomass Dataset 🌳

**AGBD-Lite** is a compact, distribution-matched subset (≈595k samples, ~5%) of the full
[**AGBD** dataset](https://huggingface.co/datasets/prs-eth/AGBD) (≈16M samples), a machine-learning-ready
benchmark for global Above-Ground Biomass (AGB) estimation from satellite imagery.

It is designed as a **screening tool**: a small, cheap-to-download stand-in for AGBD that lets you
benchmark and iterate on models before committing to a full-scale run on the complete dataset. It was
introduced to make the evaluation of Geospatial Foundation Models (GFMs) tractable.

## 🧪 How it was built

AGBD-Lite subsamples ≈5% of GEDI footprints **per Sentinel-2 tile**, choosing the footprints that
**minimise the Wasserstein distance between the biome distributions of the subsampled and the full
dataset**, for each region. As a result:

- Model **rankings are preserved exactly** relative to the full dataset.
- For our best configuration, training on AGBD-Lite comes **within 3.5%** of full-data RMSE
  (52.72 vs. 50.92 Mg/ha) at ≈5% of the cost.
- Results on AGBD-Lite are therefore a *slightly pessimistic* estimate of full-AGBD performance, while
  **relative comparisons transfer reliably**.

The code that generates AGBD-Lite (footprint selection, feature subsetting, and `.h5` writing) lives in
the [AGBD-GFMs repository](https://github.com/ghjuliasialelli/AGBD-GFMs) under `data/agbd_lite/`.

## 📦 Contents

| File | Split | Samples | Size |
|---|---|---:|---:|
| `AGBD-Lite-train.h5` | train | 371,849 | 5.4 GB |
| `AGBD-Lite-val.h5`   | val   | 103,550 | 1.5 GB |
| `AGBD-Lite-test.h5`  | test  | 119,991 | 1.7 GB |
| `AGBD-Lite-statistics.pkl` | — | — | (per-feature mean/std on train, for normalisation) |
| | **total** | **595,390** | |

Each sample is a **25×25 pixel** patch (250 m × 250 m at 10 m resolution) centred on a GEDI footprint,
with the footprint's AGB label at the centre.

## 🗂️ Schema

Every `.h5` file has the same layout. Datasets share a common first axis `N` (number of samples).

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `S2_bands` | `(N, 25, 25, 12)` | uint16 | Sentinel-2 L2A reflectance, 12 bands. Band order in `.attrs['order']`: `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` (SCL excluded). |
| `ALOS_bands` | `(N, 25, 25, 2)` | uint16 | ALOS PALSAR-2 yearly mosaic, `.attrs['order'] = ['HH','HV']`. |
| `DEM` | `(N, 25, 25)` | int16 | Copernicus DEM elevation (m). |
| `LC` | `(N, 25, 25, 2)` | uint8 | Land cover (2 channels). |
| `GEDI/agbd` | `(N,)` | float32 | **Target**: above-ground biomass density (Mg/ha). |
| `GEDI/rh98` | `(N,)` | float32 | GEDI relative height at 98% (canopy height proxy, m). |
| `GEDI/lat_decimal` | `(N,)` | float32 | Signed fractional part of the footprint latitude (see decoding below). |
| `GEDI/lat_offset` | `(N,)` | uint8 | Unsigned integer part of the footprint latitude. |
| `GEDI/lon_decimal` | `(N,)` | float32 | Signed fractional part of the footprint longitude. |
| `GEDI/lon_offset` | `(N,)` | uint8 | Unsigned integer part of the footprint longitude. |
| `GEDI/pft_class` | `(N,)` | uint8 | GEDI plant functional type class. |
| `GEDI/region_cla` | `(N,)` | uint8 | GEDI region class. |
| `Sentinel_metadata/S2_date` | `(N,)` | int16 | Sentinel-2 acquisition date (days offset). |
| `Sentinel_metadata/S2_boa_offset` | `(N,)` | uint8 | Sentinel-2 BOA reflectance offset (processing-baseline correction). |
| `region` | `(N,)` | uint8 | Region index (regions numbered alphabetically). |

> **Decoding coordinates.** Latitude and longitude are split into a signed decimal part and an unsigned
> integer offset. Reconstruct with:
> `lat = sign(lat_decimal) * (abs(lat_decimal) + lat_offset)` and likewise for longitude.

For the full definition of every feature (units, encodings, ancillary variables), see the
[AGBD dataset card](https://huggingface.co/datasets/prs-eth/AGBD).

## 🚀 Quickstart

AGBD-Lite ships as raw `.h5` files (no `datasets` loader / no viewer). Download and open with `h5py`:

```python
from huggingface_hub import hf_hub_download
import h5py, numpy as np

path = hf_hub_download(
    repo_id="prs-eth/AGBD-Lite",
    filename="AGBD-Lite-train.h5",
    repo_type="dataset",
)

with h5py.File(path, "r") as f:
    agbd = f["GEDI"]["agbd"][:]          # (N,)  target, Mg/ha
    s2   = f["S2_bands"]                 # (N, 25, 25, 12), order in .attrs["order"]
    print(f["S2_bands"].attrs["order"])
    print("N =", agbd.shape[0], "| mean AGB =", np.nanmean(agbd))
```

To grab the whole dataset at once:

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="prs-eth/AGBD-Lite", repo_type="dataset")
```

## 🔗 Related resources

- **Full dataset:** [`prs-eth/AGBD`](https://huggingface.co/datasets/prs-eth/AGBD)
- **Generation code:** [`AGBD-GFMs`](https://github.com/ghjuliasialelli/AGBD-GFMs) → `data/agbd_lite/`
- **Trained supervised weights & test predictions:** available on Zenodo (see the generation repo README).

## 📄 License

Released under **CC BY-NC 4.0**, following the parent AGBD dataset.

## 🤝 Citing

AGBD-Lite is a subset of AGBD. If you use it, please cite the AGBD dataset:

```bibtex
@article{Sialelli2025,
  title = {AGBD: A Global-scale Biomass Dataset},
  volume = {X-G-2025},
  ISSN = {2194-9050},
  url = {http://dx.doi.org/10.5194/isprs-annals-X-G-2025-829-2025},
  DOI = {10.5194/isprs-annals-x-g-2025-829-2025},
  journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  publisher = {Copernicus GmbH},
  author = {Sialelli, Ghjulia and Peters, Torben and Wegner, Jan D. and Schindler, Konrad},
  year = {2025},
  month = jul,
  pages = {829--838}
}
```

<!-- TODO: add the GFM-benchmark paper citation (where AGBD-Lite is introduced) once available. -->
