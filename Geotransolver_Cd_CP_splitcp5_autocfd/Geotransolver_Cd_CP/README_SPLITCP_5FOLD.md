# Ordinary Split-CP 5-Fold Notes

This file documents the fold-level baseline implemented by
`scripts/run_splitcp_5fold.sh`.

The script creates five independent ordinary split-CP runs inside the official
400-case AutoCFD training pool. Each run trains on four folds, calibrates on the
remaining fold, and evaluates the calibrated interval on the official 50-case
test split. It intentionally does not merge OOF predictions and does not train a
final refit model.

Default outputs are under `results/splitcp_5fold/`.
