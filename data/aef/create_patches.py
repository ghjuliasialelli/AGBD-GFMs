"""

This script creates .h5 files containing AEF data patches for specified regions, following the
footprints found in patches/data_subset-*-v4_*-20.h5.

env: dwn

Usage: 

"""

###################################################################################################
# Imports

import h5py
from os.path import join, exists, basename, isfile
import numpy as np
import pickle
from os import makedirs
import rasterio as rs
from rasterio.windows import Window
import geopandas as gpd
from pyproj import CRS, Transformer
from rasterio.vrt import WarpedVRT
from osgeo import gdal
import timeit
import argparse

###################################################################################################
# Helper functions

def parser_args():
    """
    Parse command-line arguments for creating AEF patches.
    """
    parser = argparse.ArgumentParser(description = "Create .h5 files containing AEF data patches for specified regions.")
    parser.add_argument('--region', type = str, required = True, help = 'Region name to create AEF patches for.')
    parser.add_argument('--region_number', type = int, required = True, help = 'Region number.')
    parser.add_argument('--num_splits', type = int, required = True, help = 'Number of splits for the region.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the AEF files to use.')
    parser.add_argument('--base_path', type = str, required = True, help = 'Base path.')
    parser.add_argument('--data_path', type = str, required = True, help = 'Data path.')
    parser.add_argument('--patch_size', type = int, default = 25, help = 'Size of the patches to extract.')
    parser.add_argument('--num_channels', type = int, default = 64, help = 'Number of channels in the AEF data.')
    parser.add_argument('--batch_size', type = int, default = 1000, help = 'Number of patches to write at once to the output .h5 file.')
    parser.add_argument('--write_path', type = str, required = True, help = 'Path to write the output .h5 file.')
    parser.add_argument('--path_tiff', type = str, required = True, help = 'Path where the .tiff AEF files are located.')
    parser.add_argument('--missing', nargs='*', default=[], help='List of missing AEF files to consider.')
    args = parser.parse_args()
    return args.region, args.region_number, args.num_splits, args.year, args.base_path, args.data_path, args.patch_size, args.num_channels, args.batch_size, args.write_path, args.path_tiff, args.missing

def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return CRS.from_string(crs)


def get_coordinates(f, tile, crs) :

    # Extract lon/lat info
    lons_dec = f[tile]['GEDI']['lon_decimal'][:]
    lats_dec = f[tile]['GEDI']['lat_decimal'][:]
    lons_offset = f[tile]['GEDI']['lon_offset'][:]
    lats_offset = f[tile]['GEDI']['lat_offset'][:]
    # Convert to actual lat/lon
    lats = np.sign(lats_dec) * (np.abs(lats_dec) + lats_offset)
    lons = np.sign(lons_dec) * (np.abs(lons_dec) + lons_offset)
    # reproject lat lon to local crs
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    return np.array(xs), np.array(ys)


def mosaic_aef(aef_files, crs, tile_geom) :

    # Mosaic the AEF files, to the extent of the tile
    aef_datasets = []
    for aef_f in aef_files :
        with rs.open(aef_f) as src:
            vrt = WarpedVRT(src, crs = crs, resampling = rs.enums.Resampling.nearest)
            aef_datasets.append(vrt)

    mosaic, out_transform = rs.merge.merge(aef_datasets, bounds=tile_geom.bounds, nodata=-128)
    for src in aef_datasets : src.close()

    # Put mosaic into a rasterio dataset for easy windowing
    mosaic_meta = {
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'count': mosaic.shape[0],
        'dtype': mosaic.dtype,
        'crs': crs,
        'transform': out_transform
    }

    return mosaic, out_transform, mosaic_meta

gdal.UseExceptions()
def create_mosaic_vrt(src_files, output_vrt, crs):
    """
    Builds a mosaic VRT from files with different sizes and CRSs.
    """

    print('    Creating mosaic...')

    # Instead of passing the raw paths, pass a warped VRT for each file
    standardized_srcs = []
    for f in src_files:
        vsi_path = f"/vsimem/fixed_{basename(f)}.vrt"
        gdal.Warp(vsi_path, f, options = gdal.WarpOptions(dstSRS = crs.to_wkt(), resampleAlg='nearest'))
        standardized_srcs.append(vsi_path)

    options = gdal.BuildVRTOptions(
        separate=False,
        resolution='highest',
        VRTNodata=-128,
        srcNodata=-128,
        resampleAlg='nearest'
    )
    
    # This automatically handles the CRS alignment and spatial extent
    vrt_ds = gdal.BuildVRT(output_vrt, standardized_srcs, options=options)
    vrt_ds.FlushCache()


###################################################################################################
# Code execution

if __name__ == "__main__":

    region, region_number, num_splits, year, base_path, data_path, patch_size, num_channels, batch_size, write_path, path_tiff, missing_tiles = parser_args()

    # Existing .h5 files to read the footprints from
    path_h5 = join(data_path, 'patches')
    h5_files = [f'data_subset-{year}-v4_{i}-20.h5' for i in range(20)]

    # Get the tiles for which we should create the patches
    with open(join(base_path, 'AGBD-GFM', 'aef-dwn', 'tiles_per_region.pkl'), 'rb') as f:
        tile_region_map = pickle.load(f)
    if len(missing_tiles) > 0 : s2_tiles = [t for t in tile_region_map[region] if t in missing_tiles]
    else: s2_tiles = tile_region_map[region] 

    # Mapping from Sentinel-2 tiles to AEF files
    with open('tile_to_aefiles.pkl', 'rb') as f: tile_to_aefiles = pickle.load(f)

    # Sentinel-2 tiles geometries
    path_shp = join(base_path, "BiomassDatasetCreation", "Data", "download_Sentinel", "sentinel_2_index_shapefile.shp")
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio').drop_duplicates(subset = ['Name'])

    # Handle the region number
    if (region_number == 0) and (num_splits == 1) : output_fname = f"{region}_{year}{'_missing' if len(missing_tiles) > 0 else ''}.h5"
    else : output_fname = f"{region}_{year}_{region_number}-{num_splits}{'_missing' if len(missing_tiles) > 0 else ''}.h5"

    makedirs(write_path, exist_ok=True)
    file_path = join(write_path, output_fname)
    if isfile(file_path):
        print(f'File {file_path} already exists. Appending.')
        write_mode = 'a'
    else: write_mode = 'w'
    with h5py.File(file_path, write_mode) as aef_file:

        print(aef_file.keys())

        for h5_file in h5_files :
            print(h5_file)

            with h5py.File(join(path_h5, h5_file), 'r') as f:
                file_tiles = list(f.keys())
                selected_tiles = [tile for tile in file_tiles if tile in s2_tiles]
                if len(selected_tiles) == 0 : continue

                # If needed, split the tiles among multiple processes
                if not ((region_number == 0) and (num_splits == 1)) :
                    selected_tiles = np.array_split(selected_tiles, num_splits)[region_number - 1]
                
                for tile in selected_tiles :
                    print('>>', tile)
                    start_time = timeit.default_timer()

                    # Create dataset in output h5 file
                    if tile not in aef_file :
                        aef_file.create_dataset(tile, dtype = np.int8, compression = 'gzip',
                                                shape = (0, patch_size, patch_size, num_channels), 
                                                maxshape = (None, patch_size, patch_size, num_channels), 
                                                chunks = (1, patch_size, patch_size, num_channels))
                    dset = aef_file[tile]                    

                    # Get the tile's geometry, in local CRS
                    crs = get_CRS_from_S2_tilename(tile)
                    local_grid_df = grid_df.to_crs(crs)
                    tile_geom = local_grid_df[local_grid_df['Name'] == tile].geometry.values[0]

                    xs, ys = get_coordinates(f, tile, crs)
                    if len(xs) == 0 :
                        print(f'    Skipping (no footprints)')
                        continue
                    if len(xs) == len(dset) :
                        print(f'    Skipping (patches already processed)')
                        continue

                    # Mosaic the various AEF files together
                    aef_files = tile_to_aefiles[tile][year]
                    aef_files = [join(path_tiff, basename(afile)) for afile in aef_files]
                    aef_files = [af for af in aef_files if exists(af)]
                    if len(aef_files) == 0 :
                        print(f'    Skipping (no AEF files)')
                        continue

                    print('    Mosaicking AEF files...')

                    # 1. Standardize every source file virtually
                    # This fixes the "Positive NS resolution" error before the mosaic starts
                    print('standardizing source files...')
                    standardized_virts = []
                    for i, _file in enumerate(aef_files):
                        vsi_path = f'/vsimem/{tile}-std_{i}.vrt'
                        # Warping to the target CRS here fixes the orientation
                        gdal.Warp(vsi_path, _file, options=gdal.WarpOptions(
                            dstSRS=crs,
                            format='VRT',
                            resampleAlg='nearest'
                        ))
                        standardized_virts.append(vsi_path)


                    # 2. Build the mosaic from the standardized virtual files
                    print('building mosaic VRT...')
                    vsi_mosaic = f'/vsimem/{tile}_mosaic.vrt'
                    gdal.BuildVRT(vsi_mosaic, standardized_virts)
                    
                    # 3. Warp the mosaic to the target CRS and extent
                    print('warping mosaic to final .tif...')
                    vsi_final = f'/vsimem/{tile}_final_clipped.tif'
                    warp_options = gdal.WarpOptions(
                        dstSRS=crs,
                        format='GTiff',
                        outputBounds=tile_geom.bounds,
                        resampleAlg='nearest',
                        srcNodata=-128,
                        dstNodata=-128,
                        multithread=True,
                        warpOptions=['NUM_THREADS=ALL_CPUS']
                    )
                    gdal.Warp(vsi_final, vsi_mosaic, options=warp_options)

                    with rs.open(vsi_final) as vrt_dataset:
                        print(f'    Extracting {len(xs)} patches...')

                        # Find the locations of the footprints in the raster
                        inv_transform = ~vrt_dataset.transform
                        cols, rows = inv_transform * (xs, ys)
                        rows = np.round(rows).astype(int)
                        cols = np.round(cols).astype(int)

                        min_c, max_c = cols.min() - 12, cols.max() + 13
                        min_r, max_r = rows.min() - 12, rows.max() + 13
                        big_window = Window(min_c, min_r, max_c - min_c, max_r - min_r)
                        print(f'    Reading big window: cols {min_c}-{max_c}, rows {min_r}-{max_r}')
                        big_patch = vrt_dataset.read(window = big_window, boundless = True, fill_value = -128).astype('int8', copy=False)

                        print('    Extracting patches from big array...')

                        # Write every batch_size elements to file
                        current_size = dset.shape[0]
                        total_patches = len(xs)
                        dset.resize((current_size + total_patches,) + dset.shape[1:])

                        for i in range(0, total_patches, batch_size) :
                            end_idx = min(i + batch_size, total_patches)
                            actual_batch_size = end_idx - i
                            patches = np.full((actual_batch_size, patch_size, patch_size, num_channels), -128, dtype=np.int8)

                            for j, (r, c) in enumerate(zip(rows[i:end_idx], cols[i:end_idx])) :
                                local_r, local_c = r - min_r, c - min_c
                                patch = big_patch[:, local_r-12 : local_r+13, local_c-12 : local_c+13]
                                # check if the patch is all 0s
                                if np.all(patch == -128) or np.all(patch == 0) :
                                    print(f'    Warning: patch is all nodata values.')
                                patches[j, ...] = patch.transpose(1, 2, 0)
                            
                            write_start = current_size + i
                            write_end = write_start + actual_batch_size
                            dset[write_start : write_end, ...] = patches
                    
                    # Memory cleanup
                    del big_patch, patches
                    gdal.Unlink(vsi_mosaic)
                    gdal.Unlink(vsi_final)
                    for vsi in standardized_virts:
                        gdal.Unlink(vsi)
                    aef_file.flush()
                
                    # print time in minutes
                    print(f'    Done in {(timeit.default_timer() - start_time) / 60:.2f} minutes.')






                    


