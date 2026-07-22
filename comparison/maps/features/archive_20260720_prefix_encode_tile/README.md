# Archived 2026-07-20 — superseded by the encode_tile lat/lon fix

`model/inference_helper.py::encode_tile` fed the model CONSTANT, WRONG coordinates (pyproj axis
swap + an `np.arange` that returned an empty array and broadcast to a scalar). Fixed 2026-07-20
16:43. Everything here was produced BEFORE that fix and is therefore not comparable to current
output.

- `map_AEF_vs_AGBD-features.*` — also predates the 49SBT (Asia) column entirely.
- `feature_pca_AEF_vs_AGBD-features.*` — built on AGBD activations made with the broken coordinates.

Post-fix the AGBD-features medians moved by 32TNS −22.9, 49SBT **+85.5**, 59GPM −0.8 t/ha, so these
figures show materially different numbers from the current predictions. Kept for provenance only.
Nothing deleted.
