# GeoTransolver Pressure/WSS Conformal Prediction

This directory contains the main surface-field workflow for pressure and wall
shear stress (WSS). The model predicts four channels:

```text
pressure, wss_x, wss_y, wss_z
```

It also predicts a positive residual-scale field `sigma`, which is used for
residual-normalized conformal prediction.

## Paper Settings

- Backbone: GeoTransolver.
- Output channels: 4 surface fields.
- Architecture: 20 layers, hidden size 256, 8 heads, 128 slices.
- Optimizer: AdamW, learning rate `3e-4`.
- Scheduler: StepLR with step size 50 and gamma 0.5.
- Residual-scale loss weight: `sigma_loss_weight = 0.05`.
- Smoothness loss weight: `sigma_smooth_weight = 0.01`.
- Smoothness neighborhood: `sigma_smooth_k = 8`.
- Smoothness point cap: `sigma_smooth_max_points = 2048`.
- Nominal coverage: 90% (`alpha = 0.1`).

## Conformal Modes

`cp_compare_global_point_case.py` evaluates three calibration strategies:

- `global_abs`: absolute residual calibration, interval `pred +/- q_abs`.
- `point_sigma`: point-adaptive normalized calibration, interval
  `pred +/- q_sigma * sigma`.
- `case_sigma`: case-wise normalized calibration, interval
  `pred +/- q_case * sigma`.

## Main Files

- `conf/config.yaml`: paper configuration for GeoTransolver with sigma head.
- `geotransolver_sigma.py`: GeoTransolver wrapper with pointwise residual-scale
  prediction.
- `train.py`: training loop for mean and residual-scale branches.
- `inference_on_zarr.py`: writes physical `pred`, `target`, and `sigma` arrays.
- `cp_calibrate_cvplus_normalized.py`: OOF normalized CP calibration.
- `cp_compare_global_point_case.py`: global/point/case CP comparison.
- `cp_write_vtp_global_point_case.py`: optional VTP export for ParaView.
- `case_level_deterministic_metrics.py`: deterministic case-level metrics.

## Typical Workflow

```bash
python compute_normalizations.py
sbatch scripts/train_cvplus_4fold.sh
sbatch scripts/train_final_autocfd.sh
sbatch scripts/calibrate_cvplus_only.sh
python cp_compare_global_point_case.py --help
```

Prepared DrivAerML surface data and trained checkpoints are not included.
