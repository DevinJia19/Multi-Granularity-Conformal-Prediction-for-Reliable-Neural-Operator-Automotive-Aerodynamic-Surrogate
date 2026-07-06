# GeoTransolver Pressure/WSS Smoothness Ablation

This directory is the no-smoothness ablation for the residual-scale field.
It matches the main surface-field GeoTransolver configuration except that
the smoothness regularization is disabled:

```yaml
training:
  sigma_smooth_weight: 0.0
```

The directory is used to compare residual-scale coherence, local variation,
coverage, and interval width against the main model with
`sigma_smooth_weight = 0.01`.

Run the training and inference scripts as in `Geotransolver_Pressure`, then use
the conformal calibration scripts to compare `global_abs`, `point_sigma`, and
`case_sigma` intervals.
