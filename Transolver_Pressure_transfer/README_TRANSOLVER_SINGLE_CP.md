# Transolver Single-Run Split CP

This note documents the single-run split-CP workflow implemented by
`scripts/run_transolver_single_cp.sh`.

The official 400-case AutoCFD training pool is split into 300 model-training
cases and 100 conformal-calibration cases. The model is selected using the
official 34-case validation split and evaluated on the official 50-case test
split.

After inference outputs exist, final calibrated intervals can be generated with:

```bash
bash scripts/generate_final_intervals_global_point_case.sh
```

The interval writer produces one folder for each CP mode:

```text
global_abs/
point_sigma/
case_sigma/
```

Each interval file contains `pred`, `sigma`, `lower`, `upper`, `width`, and
`qhat`. If labels are present, it also includes `target` and `covered`.
