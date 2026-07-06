# GeoTransolver Pressure/WSS Ordinary Split-CP 4-Fold Baseline

This directory contains the ordinary split conformal baseline for pressure and
wall shear stress under the AutoCFD official DrivAerML split.

## Protocol

- Official train pool: 400 cases.
- Official validation split: 34 cases.
- Official test split: 50 cases.
- The 400 training cases are divided into 4 folds.
- For each fold, 300 cases train the model and 100 cases calibrate the conformal
  scores.
- The official test split is evaluated by each fold model.
- OOF scores are not merged and a final 400-case refit model is not evaluated in
  this baseline.

## Main Command

```bash
sbatch scripts/run_splitcp_4fold.sh
```

Optional VTP export after the split-CP run:

```bash
sbatch scripts/write_splitcp_4fold_vtp.sh
```

## Outputs

```text
results/pressure_splitcp_4fold/fold_0/
results/pressure_splitcp_4fold/fold_1/
results/pressure_splitcp_4fold/fold_2/
results/pressure_splitcp_4fold/fold_3/
results/pressure_splitcp_4fold/splitcp_4fold_summary.json
results/pressure_splitcp_4fold/comparison_table_4fold_mean.csv
```
