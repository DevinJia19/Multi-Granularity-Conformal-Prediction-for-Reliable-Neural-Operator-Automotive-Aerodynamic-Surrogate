# Multi-Granularity Conformal Prediction for Reliable Neural-Operator Automotive Aerodynamic Surrogates

This workspace contains the code used for the paper experiments on calibrated
uncertainty quantification for neural-operator aerodynamic surrogate models on
the DrivAerML / AutoCFD vehicle splits.

The code is organized by experiment group rather than as one monolithic
package. Each directory is intended to be run from its own root.

## Experiment Map

| Directory | Paper role |
| --- | --- |
| `Geotransolver_Cd_CP` | Main GeoTransolver scalar drag-coefficient (`Cd`) CQR workflow, CV-assisted OOF calibration, Monte Carlo stability, and calibration-size sensitivity. |
| `Geotransolver_Cd_CP_splitcp5_autocfd` | Ordinary split-CP 5-fold AutoCFD baseline for scalar `Cd`; no OOF merge and no final refit. |
| `Geotransolver_Cd_CP_standard_transolver` | Plain Transolver scalar `Cd` backbone-transfer experiment. |
| `Geotransolver_Pressure` | Main GeoTransolver surface-field workflow for pressure and wall shear stress with residual-scale learning, smoothness regularization, and normalized CP. |
| `Geotransolver_Pressure_splitcp4_autocfd` | Ordinary split-CP 4-fold AutoCFD baseline for pressure/WSS. |
| `Geotransolver_Pressure_without_cvplus` | Surface-field smoothness ablation with `sigma_smooth_weight = 0.01`. |
| `Geotransolver_Pressure_without_smooth` | Surface-field smoothness ablation with `sigma_smooth_weight = 0`. |
| `Transolver_Pressure_transfer` | Plain Transolver pressure/WSS backbone-transfer experiment. |

## Paper Configuration Summary

Scalar `Cd` uses quantile regression at `(0.05, 0.50, 0.95)` followed by
asymmetric conformalized quantile regression. The GeoTransolver `Cd` model uses
4 layers, hidden size 192, 4 attention heads, 32 slices, a 96-dimensional pooled
output, and a `256-128-64` MLP head.

Surface fields use four channels: pressure, `wss_x`, `wss_y`, and `wss_z`.
The main GeoTransolver surface model uses 20 layers, hidden size 256, 8 heads,
128 slices, a residual-scale branch, `sigma_loss_weight = 0.05`, and
`sigma_smooth_weight = 0.01` with `k = 8` local neighbors and at most 2048
points for smoothness regularization.

All conformal experiments target nominal 90% coverage (`alpha = 0.1`).

## External Requirements

This repository does not include the raw DrivAerML/AutoCFD data, large model
checkpoints, or cluster-specific software modules. The scripts expect:

- Prepared DrivAerML STL files for scalar `Cd` experiments.
- Prepared Zarr surface data for pressure/WSS experiments.
- `physicsnemo` with GeoTransolver and Transolver support.
- PyTorch, NumPy, pandas, PyVista, Hydra/OmegaConf, and the dependencies used by
  PhysicsNeMo data pipes.

## Suggested Run Order

For scalar `Cd`, start in `Geotransolver_Cd_CP`:

```bash
python split_dataset.py
sbatch scripts/train_cvplus_all.sh
sbatch scripts/train_full_90.sh
sbatch scripts/test.sh
sbatch scripts/evaluate_cp_replicates.sh
python run_cd_calibration_size_sensitivity.py
```

For surface pressure/WSS, start in `Geotransolver_Pressure`:

```bash
python compute_normalizations.py
sbatch scripts/train_cvplus_4fold.sh
sbatch scripts/train_final_autocfd.sh
sbatch scripts/calibrate_cvplus_only.sh
python cp_compare_global_point_case.py --help
```

Use the split-CP and transfer directories when reproducing the corresponding
ablation or backbone-transfer tables.
