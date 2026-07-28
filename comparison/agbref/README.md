# AGBRef + ESA CCI comparison

Figure scripts comparing our predictions against the AGBRef reference plots and the ESA CCI
biomass map. Outputs go to `plots/`, cached intermediates to `results/`.

Two inputs are not in the repo:

- **ESA CCI biomass** — per-S2-tile rasters (`CCI_<tile>_19.tif`, ~100 m, EPSG:4326), pointed
  to by `CCI_DIR` in `comparison.py`.
- **The 5 Sentinel-2 true-colour tiles** `make_plot_maps.py` draws on — see below.

The AGBRef data itself (`AGBRef.geojson`, `AGBref.gpkg`, the `.Rdata`) *is* tracked, under
`data/`.


## Sentinel-2 tiles for the AGBRef figure

`make_plot_maps.py` renders a Sentinel-2 true-colour column alongside the biomass maps. The
tiles are ~130 MB each, so they are **not** in the repo — put them in `data/s2_tci/` and the
script picks them up automatically. These 5 are exactly the scenes the published figure draws,
one per plot in `DEFAULT_PLOTS`. All are Level-2A TCI (`B04/B03/B02`, pre-composited 8-bit,
10 m, 10980×10980), used as-is with no restretch:

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
