# GeoTransolver Cd Conformal Prediction

This directory contains the main scalar drag-coefficient (`Cd`) workflow for
the paper. The model predicts the `0.05`, `0.50`, and `0.95` quantiles and then
uses asymmetric conformalized quantile regression (CQR) to obtain calibrated
90% prediction intervals.

## Paper Settings

- Backbone: GeoTransolver with GALE attention.
- Output: scalar `Cd`.
- Quantiles: `0.05`, `0.50`, `0.95`.
- GeoTransolver: 4 layers, hidden size 192, 4 heads, 32 slices.
- Pooled readout: 96-dimensional structured pooled features.
- Quantile head: MLP `256 -> 128 -> 64 -> 3`.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-5`, batch size `8`.
- Nominal coverage: 90% (`alpha = 0.1`).

The executable configuration is in `train.py`. The standalone `config.py` is a
clean English reference copy of the same paper settings.

## Main Files

- `split_dataset.py`: builds the AutoCFD official train/validation/test split
  and internal CV folds.
- `train.py`: trains the GeoTransolver quantile-regression model.
- `test.py`: evaluates raw quantile intervals and calibrated CQR intervals.
- `cqr_common.py`: shared asymmetric CQR utilities.
- `cvplus_oof.py`: writes out-of-fold predictions for one CV fold.
- `merge_cvplus_oof.py`: merges OOF predictions and writes `hat_q.json`.
- `evaluate_cp_replicates.py`: Monte Carlo calibration/evaluation resampling.
- `run_cd_calibration_size_sensitivity.py`: Appendix calibration-size study.

## Typical Workflow

```bash
python split_dataset.py
sbatch scripts/train_cvplus_all.sh
sbatch scripts/train_full_90.sh
CQR_HAT_Q_JSON=results/cvplus/hat_q.json sbatch scripts/test.sh
sbatch scripts/evaluate_cp_replicates.sh
python run_cd_calibration_size_sensitivity.py
```

## Outputs

Common outputs are written under:

- `data_splits/`: official and fold-specific CSV files.
- `checkpoints/`: trained model weights.
- `results/cvplus/`: OOF prediction caches and merged CQR calibration values.
- `results/predictions.csv`: per-case test predictions.
- `results/per_sample_cd_cp_intervals.csv`: calibrated intervals.
- `results/cd_calibration_size_sensitivity/`: Appendix sensitivity outputs.

Raw data and trained checkpoints are not stored in this repository.
