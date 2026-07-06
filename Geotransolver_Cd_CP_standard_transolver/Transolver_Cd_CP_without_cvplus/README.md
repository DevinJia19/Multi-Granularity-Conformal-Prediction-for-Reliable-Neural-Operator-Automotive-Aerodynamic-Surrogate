# Transolver Cd Backbone-Transfer Experiment

This directory replaces the GeoTransolver scalar `Cd` backbone with the plain
PhysicsNeMo Transolver while keeping the quantile-regression head and CQR
workflow aligned with the scalar `Cd` experiment.

## Paper Role

The experiment checks whether the conformal workflow transfers across
neural-operator backbones. It uses the same scalar targets, quantiles, and CQR
metrics as the GeoTransolver run.

## Key Settings

- Backbone: `physicsnemo.models.transolver.transolver.Transolver`.
- Output: `Cd` quantiles at `0.05`, `0.50`, and `0.95`.
- Structured pooled output: 96 features.
- Quantile head: MLP `256 -> 128 -> 64 -> 3`.
- Nominal coverage: 90% (`alpha = 0.1`).

## Workflow

```bash
python split_dataset.py
sbatch scripts/train.sh
sbatch scripts/test.sh
```

The CV-assisted scripts are retained for compatibility with the scalar `Cd`
workflow, but the paper transfer result is the plain Transolver replacement
under the same conformal evaluation logic.
