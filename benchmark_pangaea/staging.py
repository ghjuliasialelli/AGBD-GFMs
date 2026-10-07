# Staging of the full AGBD dataset to the job's $TMPDIR, shared by the launcher generators of
# trainfull_runs/ (training) and testfull_runs/ (testing), so that a test job reads exactly the
# files its training job read, from the same place. pangaea's AGBD dataset reads everything from
# ONE directory (dataset.root_path_cluster), but on the cluster the files live in four, hence
# the copy. Both launchers then point root_path_cluster at ${TMPDIR}.

# Every file AGBD needs for train / val / test, as (source on the cluster, comment)
_FILES = [
    ("/cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl", "biomes_splits_to_name.pkl"),
    ("/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Biomes/helper/tiles_per_region.pkl", "'tiles_per_region.pkl'"),
    ("/cluster/work/igp_psr/gsialelli/AGBD-GFM/aef-dwn/AEF_overlaps.pkl", "'AEF_overlaps.pkl'"),
]

# Exactly what trainfull_runs/ launchers have always run (kept byte-identical).
STAGE_TRAIN = """# Move all of the necessary data to $TMPDIR
# 'AGBD-Lite-val.h5' (validation runs on the Lite val split: agbd.yaml has eval_lite: True)
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/AGBD-Lite/AGBD-Lite-val.h5 ${TMPDIR} --transfers 16 --checkers 32
# all the .h5 files for training
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR} --include "*v4_*-20.h5" --transfers 16 --checkers 32
""" + "".join(f"# {comment}\ncp {src} ${{TMPDIR}}\n" for src, comment in _FILES)

# The test split needs the same files minus the Lite validation set. The check fails the job
# before torchrun, with the missing file named, instead of deep inside dataset instantiation.
STAGE_TEST = """# Move all of the necessary data to $TMPDIR (the same files and sources as trainfull_runs/,
# minus AGBD-Lite-val.h5, which only validation reads)
rclone copy /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR} --include "*v4_*-20.h5" --transfers 16 --checkers 32
""" + "".join(f"# {comment}\ncp {src} ${{TMPDIR}}\n" for src, comment in _FILES) + """
# Refuse to start on an incomplete copy: the exact files pangaea's AGBD opens (agbd.py fnames)
for f in biomes_splits_to_name.pkl tiles_per_region.pkl AEF_overlaps.pkl \
         data_subset-{2019,2020}-v4_{0..19}-20.h5; do
    if [ ! -f "${TMPDIR}/$f" ]; then echo "staging: ${TMPDIR}/$f is missing" >&2; exit 1; fi
done
"""
