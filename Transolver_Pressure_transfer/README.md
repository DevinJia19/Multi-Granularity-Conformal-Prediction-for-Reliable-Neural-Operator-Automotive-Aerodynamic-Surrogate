# Transolver Pressure/WSS Backbone-Transfer Experiment

This directory replaces the GeoTransolver surface-field backbone with the plain
PhysicsNeMo Transolver while keeping the pressure/WSS conformal evaluation
workflow aligned with the GeoTransolver surface-field experiments.

## Paper Role

The experiment tests whether the residual-scale and conformal calibration
workflow transfers across neural-operator backbones.

## Key Settings

- Backbone: `physicsnemo.models.transolver.transolver.Transolver`.
- Output channels: `pressure`, `wss_x`, `wss_y`, `wss_z`.
- Architecture: 20 layers, hidden size 256, 8 heads, 128 slices.
- Residual-scale wrapper: `transolver_sigma.TransolverWithSigma`.
- Residual-scale loss weight: `sigma_loss_weight = 0.05`.
- Smoothness weight: `sigma_smooth_weight = 0.01`.
- Nominal coverage: 90% (`alpha = 0.1`).

## Main Workflow

```bash
sbatch scripts/run_transolver_single_cp.sh
bash scripts/generate_final_intervals_global_point_case.sh
```

The single-run protocol uses:

- official train pool: 400 cases,
- model train split: 300 cases,
- conformal calibration split: 100 cases,
- official validation split: 34 cases,
- official test split: 50 cases.

The script does not merge OOF scores, does not average models, and does not run
repeated folds.

## Important Files

- `conf/config.yaml`: plain Transolver with residual-scale head.
- `transolver_sigma.py`: residual-scale wrapper for the Transolver backbone.
- `scripts/run_transolver_single_cp.sh`: one-run CP entry point.
- `cp_compare_global_point_case.py`: computes `qhat` and test metrics.
- `cp_generate_intervals_global_point_case.py`: writes final calibrated
  `lower`/`upper` interval files.
- `cp_write_vtp_global_point_case.py`: optional VTP export.
