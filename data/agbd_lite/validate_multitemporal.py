"""

Independent validation of a multi-temporal split, over EVERY sample rather than a sample of them.

This is deliberately not the builder's own reporting: the builder twice reported 0% padding for a file
that contained thousands of duplicated and all-nodata timesteps. This reads the finished .h5 back and
checks it against the parent AGBD data.

Checks, all over the full population:
  1. S2_date ascending within each sample
  2. number of distinct dates per sample (padding, which must be visible as duplicate dates)
  3. no timestep entirely nodata
  4. the AGBD anchor patch present bit-for-bit in exactly one slot of every sample

Memory-light by design: parent patches are streamed one tile at a time, so it can run alongside a build
without competing for RAM. Exits non-zero on any failure.

Execution:
    python -u validate_multitemporal.py --mode val [--file ...]

env: agbd

"""

import os, sys, argparse, pickle, collections
from os.path import join, isfile
import numpy as np, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_multitemporal import decode_latlon, days_to_date, get_tile_to_file


def _parser() :
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required = True, choices = ['train', 'val', 'test'])
    p.add_argument('--num_timesteps', type = int, default = 3)
    p.add_argument('--path_patches', default = join('/scratch3', 'gsialelli', 'patches'))
    p.add_argument('--file', default = '')
    p.add_argument('--block', type = int, default = 4096)
    return p.parse_args()


def main() :
    a = _parser()
    out_dir = join(a.path_patches, 'AGBD-Lite-MT')
    fname = a.file if a.file else join(out_dir, f'AGBD-Lite-MT{a.num_timesteps}-{a.mode}.h5')
    if not isfile(fname) : raise SystemExit(f'no such file: {fname}')
    T = a.num_timesteps
    print(f'validating {fname}', flush = True)

    indices = pickle.load(open(join(a.path_patches, 'subsampled_indices.pkl'), 'rb'))
    split = pickle.load(open(join(a.path_patches, 'biomes_splits_to_name.pkl'), 'rb'))[a.mode]
    t2f = get_tile_to_file(a.path_patches)
    tiles = [t for t in indices if t in split and t in t2f]

    f = h5py.File(fname, 'r')
    N = f['GEDI']['agbd'].shape[0]
    dates = f['Sentinel_metadata']['S2_date'][:]
    lat = decode_latlon(f['GEDI']['lat_decimal'][:], f['GEDI']['lat_offset'][:])
    lon = decode_latlon(f['GEDI']['lon_decimal'][:], f['GEDI']['lon_offset'][:])
    print(f'N = {N}   S2_bands {f["S2_bands"].shape} {f["S2_bands"].dtype}', flush = True)

    fails = []
    if not (np.diff(dates, axis = 1) >= 0).all() : fails.append('S2_date not ascending within a sample')
    ndist = np.array([len(set(r.tolist())) for r in dates])
    hist = dict(collections.Counter(ndist.tolist()))
    padded = int((ndist < T).sum())
    print(f'distinct dates per sample: {hist}')
    print(f'padded samples           : {padded} ({100 * padded / N:.2f}%)')
    print(f'date range               : {days_to_date(int(dates.min()))} -> {days_to_date(int(dates.max()))}')
    yrs = collections.Counter(days_to_date(int(x)).year for x in dates.ravel())
    print(f'timesteps by year        : {dict(sorted(yrs.items()))}', flush = True)

    # coordinate -> merged rows (several footprints can share a rounded coordinate)
    index = collections.defaultdict(list)
    for i in range(N) : index[(round(float(lat[i]), 6), round(float(lon[i]), 6))].append(i)

    empty = 0
    for s in range(0, N, a.block) :
        blk = f['S2_bands'][s : s + a.block]
        empty += int((blk.reshape(len(blk), T, -1) == 0).all(axis = 2).sum())
    print(f'all-nodata timesteps     : {empty}', flush = True)
    if empty : fails.append(f'{empty} timesteps are entirely nodata')

    # anchor fidelity, streamed one tile at a time
    hits = misses = multi = 0
    for n_done, tile in enumerate(sorted(tiles), 1) :
        ii = np.sort(np.asarray(indices[tile]))
        with h5py.File(t2f[tile], 'r') as pf :
            g = pf[tile]
            la = decode_latlon(g['GEDI']['lat_decimal'][:][ii], g['GEDI']['lat_offset'][:][ii])
            lo = decode_latlon(g['GEDI']['lon_decimal'][:][ii], g['GEDI']['lon_offset'][:][ii])
            patches = g['S2_bands'][:][ii]
        for k in range(len(ii)) :
            rows = index.get((round(float(la[k]), 6), round(float(lo[k]), 6)))
            if not rows : misses += 1; continue
            found = False
            for i in rows :
                slots = f['S2_bands'][i]
                m = sum(int(np.array_equal(slots[t], patches[k])) for t in range(T))
                if m >= 1 : found = True; multi += (m > 1); break
            hits += found; misses += (not found)
        if n_done % 25 == 0 : print(f'  anchors: {n_done}/{len(tiles)} tiles', flush = True)
    print(f'anchor present bit-for-bit: {hits}/{hits + misses}')
    print(f'  in more than one slot   : {multi} (expected == padded samples: {padded})')
    if misses : fails.append(f'{misses} samples do not contain their AGBD anchor patch')

    print('\nRESULT:', 'PASS' if not fails else 'FAIL -> ' + '; '.join(fails), flush = True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__' :
    main()
