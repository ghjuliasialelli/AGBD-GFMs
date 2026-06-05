"""

This script downloads AEF data files, as listed in <region>_AEF_files.txt files.
Launch it with the scripts/download.sh script.

env: awsenv

Usage: python download.py --region <region_name> --output_dir <output_directory>
         --num_workers <number_of_parallel_downloads> --year <year> --path_txt <path_to_txt_files>

"""

###################################################################################################
# Imports

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import argparse
from os.path import join, exists, basename, getsize
from os import makedirs

###################################################################################################
# Helper functions

def int_or_None(value):
    """
    Convert a string to an integer or return None if the string is 'None'.
    """
    if value == 'None':
        return None
    else:
        return int(value)

def parse_args():
    """
    Parse command-line arguments for downloading AEF data files.
    """
    parser = argparse.ArgumentParser(description = "Download AEF data files for a specified region.")
    parser.add_argument('--region', type = str, required = True, help = 'Region name to download AEF files for.')
    parser.add_argument('--output_dir', type = str, default = '/scratch3/gsialelli/AEF', help = 'Output directory to save downloaded files.')
    parser.add_argument('--num_workers', type = int, default = 40, help = 'Number of parallel download workers.')
    parser.add_argument('--year', type = int_or_None, required = True, help = 'Year of the AEF files to download.')
    parser.add_argument('--path_txt', type = str, required = True, help = 'Path to the directory containing region_AEF_files folder.')
    args = parser.parse_args()
    return args.region, args.output_dir, args.num_workers, args.year, args.path_txt

def get_files_uris(region, year, path_txt) :
    """
    This function reads the list of AEF file URIs for the specified region from a text file.
    
    Args:
    - region (str): The name of the region.
    - year (int): The year to filter the AEF files by.
    - path_txt (str): The path to the directory containing region_AEF_files folder.

    Returns:
    - file_uris (list): A list of file URIs to download.
    """
    with open(join(path_txt, f'{region}_AEF_files.txt'), 'r') as f :
        file_uris = [line.strip() for line in f.readlines()]
    if year is None:  return file_uris
    else: return [uri for uri in file_uris if f'/{year}/' in uri]

def download_file(uri, output_dir):
    """
    This function downloads a file from the given S3 URI to the specified output directory.

    Args:
    - uri (str): The S3 URI of the file to download.
    - output_dir (str): The local directory to save the downloaded file.

    Returns:
    - status (str): A message indicating the download status.
    """
    key = '/'.join(uri.split('/')[4:])
    fname = basename(key)
    local_path = join(output_dir, fname)    
    
    # Check if the file was already (succesfully) downloaded
    if exists(local_path):
        info = s3.head_object(Bucket = 'tge-labs', Key = key)
        if getsize(local_path) == info['ContentLength']:
            return f"Skipped: {fname} (Already exists)"
    
    # Download the file
    try:
        s3.download_file('tge-labs', key, local_path)
        return f"Downloaded: {fname}"
    except Exception as e:
        return f"Error {uri}: {e}"

###################################################################################################
# Code execution

if __name__ == '__main__':

    # Setup
    region, output_dir, num_workers, year, path_txt = parse_args()
    file_uris = get_files_uris(region, year, path_txt)
    makedirs(output_dir, exist_ok=True)
    s3 = boto3.client('s3', endpoint_url='https://data.source.coop', config=Config(signature_version=UNSIGNED))
    
    # Launch the parallel downloads
    with ThreadPoolExecutor(max_workers = num_workers) as executor:
        results = list(executor.map(lambda uri: download_file(uri, output_dir), file_uris))
    
    # Check how many files were downloaded successfully, and how many failed
    success_count = sum(1 for r in results if r.startswith("Downloaded"))
    error_count = sum(1 for r in results if r.startswith("Error"))
    print(f"Download completed: {success_count} files downloaded, {error_count} errors.")

    # Write to a log file the ones that failed
    if error_count > 0:
        makedirs('dwn_errors', exist_ok=True)
        with open(join('dwn_errors', f'{region}_download_errors.log'), 'w') as log_file:
            for r in results:
                if r.startswith("Error"):
                    # Parse only the URI from the error message
                    uri = r.split("Error ")[1].split(' ')[1].rstrip(':')
                    log_file.write(f"{uri}\n")