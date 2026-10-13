"""

Generate every figure that goes into the manuscript, in two flavours:

    plotting/manuscript/high_res/<name>.pdf|.jpg   charts as vector PDF, maps as full-resolution JPEG
    plotting/manuscript/compressed/<name>.png      every figure as a pngquant-compressed PNG (<= 300 dpi)

Each figure is first rendered by its script in plotting/helper/ into plotting/img/ (the scripts'
usual output directory, so img/ stays in sync with what ships), then exported from there:
  - high_res/:   'pdf' figures copy img/<name>.pdf; 'jpeg' figures convert img/<name>.png to JPEG at
                 its native dpi (transparency is flattened onto white).
  - compressed/: img/<name>.png (rendered alongside the PDF, same figure) is palette-quantised with
                 `pngquant --quality=65-80 --speed 1`; if pngquant fails (e.g. exit 99, the quality
                 floor cannot be met) the PNG is copied as is. A PNG above 300 dpi (the H3 map is
                 1200 dpi) is first downsampled to 300 dpi.

Scripts run one at a time, each in the conda env it needs (`agbref` for the H3 map: h3 + cartopy;
`run` for the rest: rasterio + geopandas + h5py). make_map_figure.py reads ~15 rasters at native
10 m resolution and is memory-heavy, so nothing else runs alongside it.

Every rendered file is checked to be newer than the start of its render, so a script that silently
writes elsewhere cannot leave a stale figure to be exported as if it were fresh.

Other files in high_res/ and compressed/ (e.g. pipeline_figure.pdf) are left untouched.

Run (from anywhere, with the `run` env, which has Pillow):
      python plotting/gen_figures.py                         # render + export all figures
      python plotting/gen_figures.py --only binned fig_r2    # just these
      python plotting/gen_figures.py --skip-render           # re-export the existing img/ renders

"""

###################################################################################################
# Imports

import argparse
import shutil
import subprocess
import sys
import time
from os import makedirs
from os.path import dirname, abspath, join, exists, getmtime, getsize

from PIL import Image

###################################################################################################
# Helpers

HERE = dirname(abspath(__file__))
REPO = dirname(HERE)
HELPER = join(HERE, 'helper')
IMG = join(HERE, 'img')
OUT_DIR = join(HERE, 'manuscript')

ENVS = '/scratch2/gsialelli/miniconda3/envs'
PYTHON = {'run': join(ENVS, 'run', 'bin', 'python'), 'agbref': join(ENVS, 'agbref', 'bin', 'python')}

COMPRESSED_DPI = 300
PNGQUANT = ['pngquant', '--quality=65-80', '--speed', '1', '--force']
HIGH_RES_QUALITY = 95

# Inputs of binned.png: the full-AGBD-test results (see plot_binned_models.py)
BINNED_MODELS = [
    '--model', 'model/eval/results/nico_film_59620098-1_59620098-1_59620098-1_2019-2020_nooverlap.h5', 'AGBD features',
    '--model', 'model/eval/results/nico_film_59620113-1_59620113-1_59620113-1_2019-2020_nooverlap.h5', 'AEF',
    '--model', 'benchmark_pangaea/results/20260930_080105_f8cb5c_terramind_tiny_lora_reg_upernet_agbd_agbd_test.h5',
    'TerraMind',
]

# name: output stem in img/ and in manuscript/. env: conda env to run in. format: 'pdf' or 'jpeg'.
# renders: one argument list per script call, run from the repo root. Scripts that take --out are
# called once per extension they must produce; those with a fixed OUT write .pdf and .png together.
FIGURES = [
    {'name': 'AGBref_density_h3', 'env': 'agbref', 'format': 'jpeg',
     'renders': [['plot_agbref_density_h3.py', '--out', 'plotting/img/AGBref_density_h3.png']]},
    {'name': 'fig_benchmark', 'env': 'run', 'format': 'pdf',
     'renders': [['fig_benchmark.py']]},
    {'name': 'fig_cost', 'env': 'run', 'format': 'pdf',
     'renders': [['fig_cost.py']]},
    {'name': 'binned', 'env': 'run', 'format': 'pdf',
     'renders': [['plot_binned_models.py', *BINNED_MODELS, '--out', f'plotting/img/binned.{ext}']
                 for ext in ('pdf', 'png')]},
    {'name': 'south_asia_binned', 'env': 'run', 'format': 'pdf',
     'renders': [['plot_binned_regions.py', '--out', f'plotting/img/south_asia_binned.{ext}']
                 for ext in ('pdf', 'png')]},
    {'name': 'prediction_maps', 'env': 'run', 'format': 'jpeg',
     'renders': [['make_map_figure.py', '--out', 'plotting/img/prediction_maps']]},
    {'name': 'agbref_scatter', 'env': 'run', 'format': 'pdf',
     'renders': [['plot_agbref.py', '--variant', 'scatter', '--out', f'plotting/img/agbref_scatter.{ext}']
                 for ext in ('pdf', 'png')]},
    {'name': 'fig_r2', 'env': 'run', 'format': 'pdf',
     'renders': [['fig_r2.py']]},
]


def parse_arguments():
    names = [f['name'] for f in FIGURES]
    parser = argparse.ArgumentParser(description='Render and export all manuscript figures.')
    parser.add_argument('--only', type=str, nargs='+', choices=names, default=None,
                        help='Only these figures (default: all).')
    parser.add_argument('--skip-render', action='store_true',
                        help='Do not run the scripts; export the renders already in plotting/img/.')
    args = parser.parse_args()
    return args.only, args.skip_render


def render(fig):
    """Run the figure's script(s) and check that each expected file in img/ was (re)written."""
    t0 = time.time()
    for argv in fig['renders']:
        cmd = [PYTHON[fig['env']], '-u', join(HELPER, argv[0]), *argv[1:]]
        print(f'\n$ {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    for path in source_files(fig):
        if not exists(path) or getmtime(path) < t0 - 1:
            sys.exit(f'{fig["name"]}: {path} was not (re)written by its script')


def source_files(fig):
    """The img/ files a figure is exported from (the PNG feeds compressed/ in both cases)."""
    stem = join(IMG, fig['name'])
    return [stem + '.pdf', stem + '.png'] if fig['format'] == 'pdf' else [stem + '.png']


def flatten(im):
    """RGB copy of `im`, with any transparency composited onto white (JPEG has no alpha)."""
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.getchannel('A'))
        return bg
    return im.convert('RGB')


def compress(name, compressed_dir):
    """pngquant img/<name>.png into compressed/, capped at COMPRESSED_DPI; copy it if pngquant fails."""
    src, dest = join(IMG, name + '.png'), join(compressed_dir, name + '.png')
    Image.MAX_IMAGE_PIXELS = None  # the maps are large on purpose
    with Image.open(src) as im:
        dpi = round(im.info.get('dpi', (COMPRESSED_DPI,))[0])
        if dpi > COMPRESSED_DPI:
            scale = COMPRESSED_DPI / dpi
            im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS) \
              .save(dest, dpi=(COMPRESSED_DPI, COMPRESSED_DPI))
            src = dest
    if src != dest:
        shutil.copyfile(src, dest)
    res = subprocess.run([*PNGQUANT, '--output', dest, dest])
    if res.returncode != 0:
        print(f'  pngquant failed (exit {res.returncode}) on {name}: keeping the uncompressed PNG')
    return dest


def export_pdf(name, high_res_dir, compressed_dir):
    shutil.copyfile(join(IMG, name + '.pdf'), join(high_res_dir, name + '.pdf'))
    return [join(high_res_dir, name + '.pdf'), compress(name, compressed_dir)]


def export_jpeg(name, high_res_dir, compressed_dir):
    Image.MAX_IMAGE_PIXELS = None  # the maps are large on purpose
    with Image.open(join(IMG, name + '.png')) as im:
        dpi = round(im.info.get('dpi', (COMPRESSED_DPI,))[0])
        rgb = flatten(im)
    rgb.save(join(high_res_dir, name + '.jpg'), quality=HIGH_RES_QUALITY, subsampling=0, dpi=(dpi, dpi))
    return [join(high_res_dir, name + '.jpg'), compress(name, compressed_dir)]


def describe(path):
    """'<size> MB, <w>x<h> px' (pixels for raster files only)."""
    size = f'{getsize(path) / 1e6:6.2f} MB'
    if path.endswith(('.jpg', '.png')):
        with Image.open(path) as im:
            size += f', {im.width}x{im.height} px'
    return size

###################################################################################################
# Code execution

if __name__ == "__main__":

    only, skip_render = parse_arguments()
    figures = [f for f in FIGURES if only is None or f['name'] in only]

    high_res_dir, compressed_dir = join(OUT_DIR, 'high_res'), join(OUT_DIR, 'compressed')
    makedirs(high_res_dir, exist_ok=True)
    makedirs(compressed_dir, exist_ok=True)

    written = []
    for fig in figures:
        print(f'\n{"#" * 99}\n# {fig["name"]}', flush=True)
        if not skip_render:
            render(fig)
        missing = [p for p in source_files(fig) if not exists(p)]
        if missing:
            sys.exit(f'{fig["name"]}: missing {missing}')
        export = export_pdf if fig['format'] == 'pdf' else export_jpeg
        written += export(fig['name'], high_res_dir, compressed_dir)

    print(f'\n{"#" * 99}\n# Written ({len(written)} files)')
    for path in written:
        print(f'  {path[len(HERE) + 1:]:<50} {describe(path)}')
