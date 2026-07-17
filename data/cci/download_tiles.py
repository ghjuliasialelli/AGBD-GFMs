"""

This scripts launches the download of the required ESA CCI yearly AGB predictions for a given year.
The tiles to download are listed in a .txt file (--tilenames), and the script will download them in the specified output
folder (--output_path).

Execution:
    python download_tiles.py    --tilenames /path/to/tile_names.txt 
                                --output_path /path/to/output_folder 
                                --year 2019
                                --version (default = 6.0)

"""

############################################################################################################################
# IMPORTS 

import pycurl
from io import BytesIO
from os.path import join
import argparse
import multiprocessing as mp
from multiprocessing import get_context
from os.path import exists
import numpy as np

############################################################################################################################
# Helper functions

def setup_parser():
    """ 
    Set up the parser for the command line arguments.

    Description of the command line arguments:
        --tilenames : .txt file containing the names of the tiles to download
        --output_path : path to the folder where the tiles will be downloaded
        --year: year for which to download the tiles
    """
    
    parser = argparse.ArgumentParser(description = 'Download ESA CCI tiles.')
    parser.add_argument('--AOI', type = str, nargs = '*', default = [],
                        help = 'The AOI(s) for which to list the available granules')
    parser.add_argument("--tilenames", type = str,
                    help = "Path to the .txt file listing the tiles to consider.") 
    parser.add_argument("--output_path", type = str, required = True, 
                    help = "Path to the folder where the tiles will be downloaded.")
    parser.add_argument("--year", type = int, required = True,
                    help = "Year for which to download the tiles.")
    parser.add_argument("--version", type = str, default = '6.0',
                    help = "Version of the ESA CCI product to download (default = 6.0).")

    parser.add_argument('--i', help = 'Process split i/N.', type = int, default = 0)
    parser.add_argument('--N', help = 'Total number of splits.', type = int, default = 1)

    args = parser.parse_args()

    if args.AOI != [] : 
        tilenames = f"/scratch3/gsialelli/BiomassDatasetCreation/CCI/CCI_{'_'.join(args.AOI)}.txt"
    else: tilenames = args.tilenames

    return tilenames, args.output_path, args.year, args.i, args.N, args.version


def check_if_downloaded(tile_name, year, output_path, version = 6.0) :
    """
    Check if the tile has already been downloaded.

    Args:
    - tile_name: str, name of the tile to download
    - year: int, year for which to download the tile
    - output_path: str, path to the folder where the tiles will be downloaded
    - version: str, version of the ESA CCI product to download

    Returns:
    - bool, True if the tile has already been downloaded, False otherwise
    """
    
    # In its .zip form
    if exists(join(output_path, f'{tile_name}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-{year}-fv{version}.tif')) and \
        exists(join(output_path, f'{tile_name}_ESACCI-BIOMASS-L4-AGB_SD-MERGED-100m-{year}-fv{version}.tif')) :
        return True


############################################################################################################################
# Main function

def download_CCI_tile(items) :
    """
    Download the ESA CCI tile corresponding to the given tile name and year.

    This function downloads the tile from the CEDA Archive Web Browser. No credentials are needed.

    Args:
    - items: tuple, containing the following elements:
        - tile_name: str, name of the tile to download
        - year: int, year for which to download the tile
        - output_path: str, path to the folder where the tiles will be downloaded
    
    Returns:
    - None
    """

    # Unpack the items, necessary for the multiprocessing
    tile_name, year, output_path, version = items

    # Check if the file has already been downloaded
    dwned = check_if_downloaded(tile_name, year, output_path, version)
    if dwned :
        print('already downloaded', tile_name) 
        return (True, tile_name)

    dwned = False
    
    # Iterate over the AGB and AGB_SD products
    for elem in ['AGB_SD', 'AGB'] :
        
        # Construct the download url
        fname = f'{tile_name}_ESACCI-BIOMASS-L4-{elem}-MERGED-100m-{year}-fv{version}.tif'
        url = f'https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v{version}/geotiff/{year}/{fname}?download=1'

        # Download with curl
        try:
            curl = pycurl.Curl()
            curl.setopt(pycurl.URL, url)
            buffer = BytesIO()
            curl.setopt(pycurl.WRITEDATA, buffer)
            curl.perform()
            response = curl.getinfo(pycurl.RESPONSE_CODE)
        except Exception as e:
            print('coud not download', tile_name, e)

        # Download the file, when available
        if response == 200 :
            print('downloading', tile_name)
            with open(join(output_path, fname), 'wb') as f: 
                f.write(buffer.getvalue())
            dwned = True
        else :
            print('not available', tile_name)

    return (dwned, tile_name)

############################################################################################################################
# Execute

if __name__ == "__main__":

    tilenames, output_path, year, i, N, version = setup_parser()
    tile_info_list = open(tilenames, 'r').read().splitlines()

    print(f'split {i}/{N}')
    tile_info_list = np.array_split(tile_info_list, N)[i]
    print(f'Number of tiles to download: {len(tile_info_list)}')

    for tile_info in tile_info_list :
        if '_' in tile_info : tile_info, year = tile_info.split('_')
        download_CCI_tile((tile_info, year, output_path, version))
