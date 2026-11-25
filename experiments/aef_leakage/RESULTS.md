# AEF pre-training leakage vs distance (reviewer request, 2026-11-25)

Question: is test accuracy better close to AEF pre-training locations (residual leakage beyond the
1.28 km patch exclusion)?

- AEF locations: `/scratch3/gsialelli/AGBD-GFM/aef-dwn/distribution/training_coordinates.csv`
  (8,412,511 rows, 4,812,486 unique points; exclusion in `data/aef/check_test_overlap.py` = +-640 m UTM square).
- Test set: paper no-overlap full test, n = 2,807,977, coordinates from
  `experiments/spatial_autocorrelation/data/test_coordinates_nooverlap.npz` (exact label match verified).
- Models (paired, same rows): AEF nico_film 59620113-1 (RMSE 50.91) vs AGBD features 59620098-1 (53.72).
- Scripts: `distances.py` -> `data/dist_to_aef_nooverlap.npz`; `analyse.py` -> `data/binned_gap.csv`.
  CIs = 1000x tile-cluster bootstrap.

## Findings
1. Every test sample is within 17.8 km of an AEF training point (median 1.83 km, 95th pct 6.5 km).
   The testable range is 0.64-20 km, not "near vs far".
2. 969 test samples are < 640 m from an AEF point (missed by the exclusion: it only searched AEF
   points within the sample's own S2 tile). Negligible (0.03%).
3. Raw AEF RMSE falls with distance (59.6 -> 25.8), but so does AGBD RMSE (62.5 -> 27.9): far samples
   are low biomass (mean AGBD 92 -> 27 Mg/ha). Composition, not leakage.
4. AEF advantage is flat in absolute terms (gap 2.9 Mg/ha at 0.64-1 km vs 2.1-3.0 elsewhere, CIs overlap) and
   in relative terms slightly LARGER far away (RMSE ratio AEF/AGBD 0.954 near vs 0.924 at 8-20 km).
   Same within each GEDI region and within biomass strata (strongest at 150-300 Mg/ha: ratio 0.956 near -> 0.90 far).
5. Region x biomass fixed-effects slope of (SE_AGBD - SE_AEF) on log10 distance: +51 per decade
   [95% CI -23, +149]. Leakage predicts a negative slope; sign is opposite and CI includes 0.

Conclusion: no evidence of residual leakage from AEF pre-training at 0.64-20 km.

## Inside the excluded patches (analyse_full.py, 2026-11-25)
Full test incl. the 495,745 excluded rows (all within 0.905 km = square half-diagonal of an AEF point).
Only 3-seed ensembles have full-test predictions (not the paper's single runs): full-trained AGBD 17997535 vs
AEF 56026335 (no-overlap 52.70/50.70); Lite-trained 55140725 vs 56549587. 17997535 is in year-outer file
order -> remapped by row key (full_coordinates.py; all 4 label arrays verified exact).

| pair | kept (paper) rel. adv. | excluded rel. adv. | excluded, 0-0.32 km |
|---|---|---|---|
| full-trained | 3.80% | 4.00% | 4.23% [3.11, 5.47] |
| Lite-trained | 9.24% | 9.57% | 9.56% [7.97, 11.72] |

Excluded samples are higher biomass (mean 94.8 vs 69.7 Mg/ha), so both models have higher RMSE there
(60.3 / 62.8). AEF's relative advantage inside the patches is the same as outside, within CIs:
no measurable leakage even for directly overlapping samples. The exclusion was conservative, not necessary.
