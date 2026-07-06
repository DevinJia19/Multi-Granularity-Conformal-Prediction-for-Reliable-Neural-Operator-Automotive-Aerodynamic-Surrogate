# GeoTransolver Cd Ordinary Split-CP 5-Fold Baseline

This directory contains the ordinary split conformal baseline for scalar `Cd`
under the AutoCFD official DrivAerML split. It is used to compare ordinary
split CP against the CV-assisted OOF workflow in the main `Geotransolver_Cd_CP`
directory.

## Protocol

- Official train pool: 400 cases.
- Official validation split: 34 cases.
- Official test split: 50 cases.
- The 400 training cases are divided into 5 folds.
- For each fold, 4 folds train the model and 1 fold is used only for conformal
  calibration.
- The same official test set is evaluated once per fold model.
- OOF scores are not merged.
- A final 400-case refit model is not trained in this baseline.

## Main Command

```bash
sbatch scripts/run_splitcp_5fold.sh
```

Useful overrides:

```bash
RUN_TRAIN=0 RUN_TEST=1 RUN_MC=0 sbatch scripts/run_splitcp_5fold.sh
SPLITCP_START_FOLD=2 sbatch scripts/run_splitcp_5fold.sh
CP_EVAL_R=1000 sbatch scripts/run_splitcp_5fold.sh
```

## Outputs

Per-fold outputs are written under:

```text
results/splitcp_5fold/fold_0/
results/splitcp_5fold/fold_1/
...
results/splitcp_5fold/fold_4/
```

Aggregated outputs:

```text
results/splitcp_5fold/splitcp_5fold_summary.json
results/splitcp_5fold/splitcp_5fold_fold_metrics_flat.csv
```
