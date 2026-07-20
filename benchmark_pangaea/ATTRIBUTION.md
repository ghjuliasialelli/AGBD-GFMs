# Pretrained GFM encoder weights — sources, licences, citations

The AGBD-GFM benchmark evaluates 11 geospatial foundation model encoders. **None of these
models are ours.** They are the work of the groups listed below, and we use their published
weights unmodified.

The normal [PANGAEA](https://github.com/VMarsocci/pangaea-bench) pipeline downloads most of
them for you from the original authors' hosts; `fetch_encoder_weights.sh` in this directory
is an *optional* convenience that pulls the exact files our runs used from our Zenodo
archive. Either way, **the authoritative source for each model is its own project, and you
should cite the original publication for any encoder you use.**

| Encoder | File | Weights licence | Original source |
|---|---|---|---|
| `croma_optical` | `CROMA_large.pt` | MIT | [huggingface.co/antofuller/CROMA](https://huggingface.co/antofuller/CROMA) |
| `dofa` | `DOFA_ViT_base_e100.pth` | CC-BY-4.0 | [huggingface.co/XShadow/DOFA](https://huggingface.co/XShadow/DOFA) |
| `prithvi` | `Prithvi_100M.pt` | Apache-2.0 | [huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) |
| `prithvi2_100m` | `Prithvi_EO_V2_100M_TL.pt` | Apache-2.0 | [huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL) |
| `remoteclip` | `RemoteCLIP-ViT-B-32.pt` | **not stated** | [huggingface.co/chendelong/RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) |
| `scalemae` | `scalemae-vitlarge-800.pth` | **CC-BY-NC-4.0** (NonCommercial) | [github.com/bair-climate-initiative/scale-mae](https://github.com/bair-climate-initiative/scale-mae) |
| `spectralgpt` | `SpectralGPT+.pth` | CC-BY-4.0 | [zenodo.org/records/8412455](https://zenodo.org/records/8412455) |
| `ssl4eo_moco` | `B13_vits16_moco_0099.pth` | CC-BY-4.0 | [github.com/zhu-xlab/SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) |
| `terramind_optical_tiny` | `TerraMind_v1_tiny.pt` | Apache-2.0 | [huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) |
| `gfmswin` | `gfm.pth` | **not stated** | [github.com/mmendiet/GFM](https://github.com/mmendiet/GFM) (weights via OneDrive) |
| `satlasnet_si` | *(none — see note)* | ODC-BY | [github.com/allenai/satlaspretrain_models](https://github.com/allenai/satlaspretrain_models) |

Verified 2026-07-17. Licences change; check the source before relying on this table.

## Things that are easy to get wrong

**Code licence ≠ weights licence.** Several of these projects release their *code* under
Apache-2.0 while the *weights* carry different terms. SSL4EO-S12 states this explicitly
("the dataset and pretrained model weights are released under the CC-BY-4.0 license"), and
SatlasNet's checkpoints are ODC-BY while its code is Apache-2.0. Reading the repository's
licence badge will give you the wrong answer for both.

**Scale-MAE is NonCommercial.** `scalemae-vitlarge-800.pth` is CC-BY-NC-4.0, and GitHub's
licence detector reports `NOASSERTION` for that repo, so nothing warns you — you have to
read the LICENSE file. If you use Scale-MAE, the NC restriction applies to you too.

**Two have no stated weights licence: `gfm.pth` (GFM-Swin) and RemoteCLIP.** Their code
repositories are Apache-2.0, but that covers the code; neither publishes terms for the
checkpoints themselves. We archive the exact files we ran for reproducibility, but we make
no claim about your rights to reuse them — that is between you and the original authors.

**GFM-Swin has an extra attribution ask.** Its README requests that users also cite the
original data sources behind GeoPile (references 9, 29, 33, 35, 48 in the GFM paper), since
GeoPile is itself a collection of other datasets.

**satlasnet_si needs no file here.** It downloads its own weights through the
`satlaspretrain_models` package at runtime, so it never appears in `pretrained_models/`.

## Papers to cite

- **CROMA** — Fuller et al., *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders*, NeurIPS 2023.
- **DOFA** — Xiong et al., *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation*.
- **Prithvi** — Jakubik et al., *Foundation Models for Generalist Geospatial Artificial Intelligence* (and Prithvi-EO-2.0 for the v2 model).
- **RemoteCLIP** — Liu et al., *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*.
- **Scale-MAE** — Reed et al., *Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning*, ICCV 2023.
- **SpectralGPT** — Hong et al., *SpectralGPT: Spectral Remote Sensing Foundation Model*, IEEE TPAMI.
- **SSL4EO-S12** — Wang et al., *SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation*.
- **TerraMind** — Jakubik et al., *TerraMind: Large-Scale Generative Multimodality for Earth Observation*.
- **GFM-Swin** — Mendieta et al., *Towards Geospatial Foundation Models via Continual Pretraining*, ICCV 2023.
- **SatlasNet** — Bastani et al., *SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding*, ICCV 2023.
- **PANGAEA** — Marsocci et al., *PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models* — the benchmark harness this builds on.
