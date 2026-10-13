"""

Global density of the AGBref plots on an H3 hexagonal grid (manuscript Fig. `AGBref_density_h3`).

Each AGBref plot centre (POINT_X, POINT_Y) is assigned to one H3 cell at resolution H3_RES, and the
cells are coloured by plot count on a log scale, so a few crowded cells do not wash out the rest.
Hexagons that cross the antimeridian are split in two before drawing.

Ported from cell 16 of EcosystemAnalysis/Models/Biomes/AGBRef/exploration.ipynb (the original
source of this figure), unchanged except that it reads the in-repo copy of the plot table. The
780 plot centres in comparison/agbref/data/AGBRef.geojson are identical to the notebook's.

Needs `h3` and `cartopy`: only the `agbref` env has both (the `run` env has neither).

Run (from the repo root, `agbref` env):
      python plotting/helper/plot_agbref_density_h3.py \
          --out plotting/img/AGBref_density_h3.png

"""

###################################################################################################
# Imports

import argparse
import json
import h3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from os import makedirs
from os.path import dirname, abspath, join

###################################################################################################
# Helpers

HERE = dirname(abspath(__file__))
DEFAULT_PLOTS = join(HERE, '..', '..', 'comparison', 'agbref', 'data', 'AGBRef.geojson')


def parse_arguments():
    parser = argparse.ArgumentParser(description='H3-hex density map of the AGBref plot locations.')
    parser.add_argument('--plots', type=str, default=DEFAULT_PLOTS, help='AGBref plots (GeoJSON with POINT_X/POINT_Y).')
    parser.add_argument('--h3_res', type=int, default=2,
                        help='H3 resolution: 0 ~4.4M km^2, 2 ~86k km^2, 3 ~12k km^2 per cell.')
    parser.add_argument('--out', type=str, required=True, help='Output image path.')
    parser.add_argument('--dpi', type=int, default=1200, help='Figure DPI.')
    args = parser.parse_args()
    return args.plots, args.h3_res, args.out, args.dpi


def load_centres(path):
    """Return the (lat, lon) plot centres. Plain json, so the PROJ database is not needed."""
    with open(path) as f:
        features = json.load(f)['features']
    return [(ft['properties']['POINT_Y'], ft['properties']['POINT_X']) for ft in features]


def hex_polygons(cell):
    """Return the cell's hexagon as shapely polygon(s) in lon/lat, split at the antimeridian."""
    boundary = h3.cell_to_boundary(cell)  # list of (lat, lon)
    lons = [lon for _, lon in boundary]
    lats = [lat for lat, _ in boundary]
    if max(lons) - min(lons) > 180:
        west = [(lon - 360 if lon > 0 else lon, lat) for lat, lon in boundary]
        east = [(lon + 360 if lon < 0 else lon, lat) for lat, lon in boundary]
        return [Polygon(west), Polygon(east)]
    return [Polygon(list(zip(lons, lats)))]

###################################################################################################
# Code execution

if __name__ == "__main__":

    plots_path, h3_res, out_path, dpi = parse_arguments()

    centres = load_centres(plots_path)
    assert len(centres) > 0, f'no plots in {plots_path}'

    # Aggregate the plot centres into H3 cells
    counts = {}
    for lat, lon in centres:
        c = h3.latlng_to_cell(lat, lon, h3_res)
        counts[c] = counts.get(c, 0) + 1
    assert sum(counts.values()) == len(centres)
    print(f'{len(centres)} plots in {len(counts)} H3 cells (res {h3_res}), max {max(counts.values())} per cell')

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.6,
    })

    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11, 6), dpi=300)
    ax = plt.axes(projection=proj)
    ax.set_global()

    # Background
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#f5f1e8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#444444", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor="#888888", zorder=1)
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle=(0, (1, 3)))

    # Log colour scale, so a few crowded cells don't wash out the rest
    norm = LogNorm(vmin=1, vmax=max(counts.values()))
    cmap = plt.get_cmap("plasma")

    for cell, n in counts.items():
        for poly in hex_polygons(cell):
            ax.add_geometries([poly], crs=data_crs, facecolor=cmap(norm(n)), edgecolor="white",
                              linewidth=0.3, alpha=0.95, zorder=2)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.04, shrink=0.55, extend="max")
    cbar.set_label("Number of plots per H3 cell", fontsize=12)
    cbar.ax.tick_params(labelsize=9, width=0.3)
    cbar.outline.set_linewidth(0.25)

    plt.tight_layout()

    makedirs(dirname(abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f'\nSaved {out_path}')
