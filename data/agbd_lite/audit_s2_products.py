"""Population audit: per (tile, product) SCL stats -> can every tile give 3 usable timesteps?
Reads only the 20m SCL band (~1.3MB) straight out of the zip. No unzipping, tiny memory."""
import os, re, sys, glob, pickle, zipfile, csv, time
import numpy as np, rasterio as rs
from multiprocessing import Pool

S2DIR = '/scratch3/gsialelli/S2_L2A'
LITE  = '/scratch3/gsialelli/AGBD-GFMs/data/agbd_lite/indices'
PAT   = re.compile(r'S2[ABC]_MSIL2A_(\d{8})T\d{6}_N\d{4}_R\d{3}_T(\w{5})_')

def scl_path(prod):
    p = os.path.join(S2DIR, prod)
    if prod.endswith('.zip'):
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if '/R20m/' in n and '_SCL_20m.jp2' in n]
        return f'/vsizip/{p}/{names[0]}' if names else None
    hits = glob.glob(f'{p}/GRANULE/*/IMG_DATA/R20m/*_SCL_20m.*')
    return hits[0] if hits else None

def job(args):
    tile, date, prod = args
    try:
        sp = scl_path(prod)
        if sp is None:
            return dict(tile=tile, date=date, prod=prod, err='no_SCL')
        with rs.open(sp) as src:
            a = src.read(1)
        n = a.size
        c = np.bincount(a.ravel(), minlength=12)
        return dict(tile=tile, date=date, prod=prod, err='',
                    nodata=c[0]/n,                      # 0 = no data (outside swath)
                    cloud=(c[8]+c[9]+c[10])/n,          # med/high prob cloud + cirrus
                    shadow=c[3]/n, snow=c[11]/n,
                    clear=(c[4]+c[5]+c[6]+c[7])/n)      # veg, bare, water, unclassified
    except Exception as e:
        return dict(tile=tile, date=date, prod=prod, err=type(e).__name__+':'+str(e)[:80])

if __name__ == '__main__':
    tiles = set()
    for f in glob.glob(f'{LITE}/*.pkl'):
        tiles |= set(pickle.load(open(f,'rb')).keys())
    work = []
    for p in os.listdir(S2DIR):
        m = PAT.match(p)
        if m and m.group(1)[:4] in ('2019','2020') and m.group(2) in tiles:
            work.append((m.group(2), m.group(1), p))
    work.sort()
    print(f'{len(tiles)} tiles, {len(work)} products to audit', flush=True)
    t0 = time.time()
    fields = ['tile','date','prod','err','nodata','cloud','shadow','snow','clear']
    with open('scl_audit.csv','w',newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        with Pool(6) as pool:
            for i, r in enumerate(pool.imap_unordered(job, work, chunksize=4), 1):
                w.writerow({k: r.get(k,'') for k in fields})
                if i % 100 == 0:
                    fh.flush(); print(f'{i}/{len(work)}  {time.time()-t0:.0f}s', flush=True)
    print(f'DONE {len(work)} products in {time.time()-t0:.0f}s', flush=True)
