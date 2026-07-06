# GeoTransolver Pressure/WSS Without CV-Assisted OOF Calibration

This directory keeps the surface-field model and residual-normalized CP
machinery, but uses an ordinary split conformal protocol instead of the
CV-assisted OOF score aggregation used in the main surface-field workflow.

It is used for the paper comparison between ordinary split CP and
CV-assisted OOF CP.

## Key Differences From `Geotransolver_Pressure`

- No OOF score merge.
- Calibration scores come from one held-out calibration split.
- The model and conformal modes remain the same:
  `global_abs`, `point_sigma`, and `case_sigma`.

Use the scripts in `scripts/` for train/test runs and
`cp_compare_global_point_case.py` for interval evaluation.
