#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo evaluation for CP/CQR.

Preferred leakage-safe mode:
  Read existing CV+ out-of-fold predictions from oof_fold_*.npz, then randomly
  split only the cached OOF predictions into calibration/evaluation subsets.

Legacy mode:
  Run one checkpoint over one CSV, cache those predictions, then randomly split
  the cached predictions. This is useful for diagnostics, but if the checkpoint
  was trained on the same CSV it is an in-sample analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from cqr_common import apply_asymmetric_cqr, asymmetric_cqr_hat_q


def _load_model_and_config(checkpoint_path: str, device):
    import torch
    from train import CdPredictionModel, Config

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = Config()
    model_spec = checkpoint.get("model_spec", {})
    for key in [
        "QUANTILES",
        "OUT_DIM",
        "FUNCTIONAL_DIM",
        "GEOTRANS_OUT_DIM",
        "GEOMETRY_DIM",
        "GLOBAL_DIM",
        "N_HIDDEN",
        "N_LAYERS",
        "N_HEAD",
        "DROPOUT",
        "SLICE_NUM",
        "POOLING_TYPE",
        "NUM_POINTS",
    ]:
        if key in model_spec:
            setattr(config, key, model_spec[key] if key != "QUANTILES" else tuple(model_spec[key]))

    model = CdPredictionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ckpt_cfg = checkpoint.get("config", {})
    config.TARGET_MEAN = float(ckpt_cfg.get("TARGET_MEAN", 0.0))
    config.TARGET_STD = float(ckpt_cfg.get("TARGET_STD", 1.0))
    config.GLOBAL_DESCRIPTOR_MEAN = torch.tensor(
        ckpt_cfg.get("GLOBAL_DESCRIPTOR_MEAN", [0.0] * 15),
        dtype=torch.float32,
    )
    config.GLOBAL_DESCRIPTOR_STD = torch.tensor(
        ckpt_cfg.get("GLOBAL_DESCRIPTOR_STD", [1.0] * 15),
        dtype=torch.float32,
    )
    model.eval()
    return model, config


def _evaluate_collect(model, config, csv_file: str, device):
    from torch.utils.data import DataLoader

    from draivernet_dataset import DrivAerNetDataset
    from test import evaluate_model

    ds = DrivAerNetDataset(
        root_dir=config.STL_ROOT_DIR,
        csv_file=csv_file,
        num_points=config.NUM_POINTS,
        transform=None,
        apply_augmentations=False,
        normalize=True,
        design_column=config.DESIGN_COLUMN,
        target_column=config.TARGET_COLUMN,
        file_suffix=config.FILE_SUFFIX,
        normalize_target=True,
        target_mean=config.TARGET_MEAN,
        target_std=config.TARGET_STD,
        global_descriptor_mean=config.GLOBAL_DESCRIPTOR_MEAN,
        global_descriptor_std=config.GLOBAL_DESCRIPTOR_STD,
        deterministic_sampling=True,
        deterministic_seed_base=config.SEED,
    )
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    pred_norm, y_norm = evaluate_model(model, loader, device)
    q05 = pred_norm[0.05] * config.TARGET_STD + config.TARGET_MEAN
    q50 = pred_norm[0.5] * config.TARGET_STD + config.TARGET_MEAN
    q95 = pred_norm[0.95] * config.TARGET_STD + config.TARGET_MEAN
    y = y_norm * config.TARGET_STD + config.TARGET_MEAN
    return (
        y.reshape(-1).astype(np.float64),
        q05.astype(np.float64),
        q50.astype(np.float64),
        q95.astype(np.float64),
    )


def _load_oof_collect(oof_dir: str):
    pattern = os.path.join(oof_dir, "oof_fold_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No OOF files found: %s" % pattern)

    y_parts, q05_parts, q95_parts = [], [], []
    for path in files:
        z = np.load(path)
        required = ("y_true", "q05_pred", "q95_pred")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError("%s missing keys: %s" % (path, ", ".join(missing)))
        y_parts.append(z["y_true"].astype(np.float64).ravel())
        q05_parts.append(z["q05_pred"].astype(np.float64).ravel())
        q95_parts.append(z["q95_pred"].astype(np.float64).ravel())

    y = np.concatenate(y_parts, axis=0)
    q05 = np.concatenate(q05_parts, axis=0)
    q95 = np.concatenate(q95_parts, axis=0)
    q50 = np.full_like(y, np.nan, dtype=np.float64)
    return y, q05, q50, q95, files


def _interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> float:
    width = upper - lower
    under = (lower - y) * (y < lower)
    over = (y - upper) * (y > upper)
    return float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))


def _one_replicate(
    rng: np.random.Generator,
    y: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    n_cal: int,
    alpha: float,
    include_q50: bool,
):
    n = y.size
    perm = rng.permutation(n)
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:]

    q_l, q_u = asymmetric_cqr_hat_q(q05[cal_idx], q95[cal_idx], y[cal_idx], alpha)

    yv = y[val_idx]
    q05v, q95v = q05[val_idx], q95[val_idx]
    lo, hi = apply_asymmetric_cqr(q05v, q95v, q_l, q_u)
    raw_width = float(np.mean(q95v - q05v))
    raw_interval_score = _interval_score(q05v, q95v, yv, alpha)

    if np.isfinite(q_l) and np.isfinite(q_u):
        cqr_cover = float(np.mean((yv >= lo) & (yv <= hi)))
        cqr_tail_low = float(np.mean(yv < lo))
        cqr_tail_high = float(np.mean(yv > hi))
        cqr_width = float(np.mean(hi - lo))
        cqr_width_expansion = cqr_width - raw_width
        cqr_interval_score = _interval_score(lo, hi, yv, alpha)
        cqr_interval_score_delta = cqr_interval_score - raw_interval_score
    else:
        cqr_cover = float("nan")
        cqr_tail_low = float("nan")
        cqr_tail_high = float("nan")
        cqr_width = float("nan")
        cqr_width_expansion = float("nan")
        cqr_interval_score = float("nan")
        cqr_interval_score_delta = float("nan")

    out = {
        "cqr_coverage": cqr_cover,
        "cqr_tail_below_lower": cqr_tail_low,
        "cqr_tail_above_upper": cqr_tail_high,
        "raw_pi90_width": raw_width,
        "cqr_pi90_width": cqr_width,
        "cqr_width_expansion": cqr_width_expansion,
        "raw_pi90_interval_score": raw_interval_score,
        "cqr_pi90_interval_score": cqr_interval_score,
        "cqr_interval_score_delta": cqr_interval_score_delta,
        "pi90_uncalibrated": float(np.mean((yv >= q05v) & (yv <= q95v))),
        "tail_below_q05": float(np.mean(yv < q05v)),
        "tail_above_q95": float(np.mean(yv > q95v)),
    }
    if include_q50:
        out["frac_y_le_q50"] = float(np.mean(yv <= q50[val_idx]))
    return out


def _load_cached_npz(cache_path: str, expected_source: str | None = None):
    z = np.load(cache_path)
    source = str(z["prediction_source"]) if "prediction_source" in z.files else "legacy_unknown"
    if expected_source is not None and source != expected_source:
        raise ValueError(
            "cache prediction_source=%s, expected %s; use a different --cache path"
            % (source, expected_source)
        )
    y = z["y"].astype(np.float64)
    q05 = z["q05"].astype(np.float64)
    q95 = z["q95"].astype(np.float64)
    q50 = z["q50"].astype(np.float64) if "q50" in z.files else np.full_like(y, np.nan)
    return y, q05, q50, q95, source


def _save_cached_npz(cache_path: str, y, q05, q50, q95, prediction_source: str, extra=None):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "y": y,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "prediction_source": np.array(prediction_source),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(cache_path, **payload)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Monte Carlo random calibration/evaluation splits for CP/CQR."
    )
    ap.add_argument("--checkpoint", default="", help="best_model.pth path for legacy mode")
    ap.add_argument("--csv", default="", help="evaluation CSV for legacy mode")
    ap.add_argument(
        "--oof-dir",
        default="",
        help="directory containing existing CV+ oof_fold_*.npz files; preferred leakage-safe mode",
    )
    ap.add_argument("--n-cal", type=int, required=True, help="calibration size per replicate")
    ap.add_argument("--R", type=int, default=1000, dest="R", help="number of random replicates")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="", help="prediction cache npz path")
    ap.add_argument("--device", default="", help="cuda or cpu; auto if empty")
    ap.add_argument("--out-json", default="", help="write summary JSON")
    ap.add_argument("--batch-size", type=int, default=0, help="legacy mode batch size override")
    args = ap.parse_args()

    cache_path = args.cache.strip()
    oof_dir = args.oof_dir.strip()
    prediction_source = "cached_out_of_fold" if oof_dir else "checkpoint"

    if oof_dir:
        if cache_path and os.path.isfile(cache_path):
            print("[INFO] Loading OOF prediction cache: %s" % cache_path)
            try:
                y, q05, q50, q95, _ = _load_cached_npz(cache_path, prediction_source)
            except Exception as exc:
                print("[ERROR] %s" % exc, file=sys.stderr)
                return 1
            oof_files = []
        else:
            print("[INFO] Loading cached out-of-fold predictions from: %s" % oof_dir)
            try:
                y, q05, q50, q95, oof_files = _load_oof_collect(oof_dir)
            except Exception as exc:
                print("[ERROR] failed to load OOF predictions: %s" % exc, file=sys.stderr)
                return 1
            if cache_path:
                _save_cached_npz(
                    cache_path,
                    y,
                    q05,
                    q50,
                    q95,
                    prediction_source,
                    {"oof_files": np.array(oof_files)},
                )
                print("[INFO] Wrote OOF prediction cache: %s" % cache_path)
        print("[INFO] OOF mode avoids in-sample prediction leakage.")
    elif cache_path and os.path.isfile(cache_path):
        print("[INFO] Loading prediction cache: %s" % cache_path)
        try:
            y, q05, q50, q95, prediction_source = _load_cached_npz(cache_path)
        except Exception as exc:
            print("[ERROR] %s" % exc, file=sys.stderr)
            return 1
    else:
        if not args.checkpoint or not os.path.isfile(args.checkpoint):
            print("[ERROR] checkpoint not found: %s" % args.checkpoint, file=sys.stderr)
            return 1
        if not args.csv or not os.path.isfile(args.csv):
            print("[ERROR] csv not found: %s" % args.csv, file=sys.stderr)
            return 1
        print("[INFO] Legacy mode: running one checkpoint over one CSV.")
        print("[WARN] If this CSV was used to train the checkpoint, this is in-sample.")
        import torch

        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model, config = _load_model_and_config(args.checkpoint, device)
        config.STL_ROOT_DIR = os.getenv("STL_ROOT_DIR", config.STL_ROOT_DIR)
        if int(args.batch_size or 0) > 0:
            config.BATCH_SIZE = int(args.batch_size)
        y, q05, q50, q95 = _evaluate_collect(model, config, args.csv, device)
        if cache_path:
            _save_cached_npz(cache_path, y, q05, q50, q95, prediction_source)
            print("[INFO] Wrote prediction cache: %s" % cache_path)

    n = int(y.size)
    n_cal = int(args.n_cal)
    if n_cal <= 0 or n_cal >= n:
        print("[ERROR] need 0 < n_cal < N; N=%d, n_cal=%d" % (n, n_cal), file=sys.stderr)
        return 1

    R = int(args.R)
    alpha = float(args.alpha)
    rng = np.random.default_rng(args.seed)
    include_q50 = bool(np.all(np.isfinite(q50)))

    names = [
        "cqr_coverage",
        "cqr_tail_below_lower",
        "cqr_tail_above_upper",
        "raw_pi90_width",
        "cqr_pi90_width",
        "cqr_width_expansion",
        "raw_pi90_interval_score",
        "cqr_pi90_interval_score",
        "cqr_interval_score_delta",
        "pi90_uncalibrated",
        "tail_below_q05",
        "tail_above_q95",
    ]
    if include_q50:
        names.append("frac_y_le_q50")
    elif prediction_source == "cached_out_of_fold":
        print("[INFO] q50_pred is absent in existing OOF files; skipping frac_y_le_q50.")

    acc = {k: np.zeros(R, dtype=np.float64) for k in names}

    print(
        "[INFO] source=%s, R=%d, N=%d, n_cal=%d, n_val=%d, alpha=%s, seed=%d"
        % (prediction_source, R, n, n_cal, n - n_cal, alpha, args.seed)
    )

    for r in range(R):
        out = _one_replicate(rng, y, q05, q50, q95, n_cal, alpha, include_q50)
        for k in names:
            acc[k][r] = out[k]

    summary = {
        "prediction_source": prediction_source,
        "N": n,
        "n_cal": n_cal,
        "n_val": n - n_cal,
        "R": R,
        "alpha": alpha,
        "nominal": {
            "pi90_coverage_target": 1.0 - alpha,
            "cqr_tail_below_lower_target": alpha / 2.0,
            "cqr_tail_above_upper_target": alpha / 2.0,
            "tail_below_q05_target": alpha / 2.0,
            "tail_above_q95_target": alpha / 2.0,
        },
        "mean": {k: float(np.mean(acc[k])) for k in names},
        "std": {k: float(np.std(acc[k], ddof=0)) for k in names},
    }
    if include_q50:
        summary["nominal"]["frac_y_le_q50_target"] = 0.5

    print("\n=== Monte Carlo random split summary on evaluation subsets ===")
    print(
        "Nominal: CQR coverage ~= %.4f | CQR lower/upper tail ~= %.4f | "
        "raw lower/upper tail ~= %.4f"
        % (1.0 - alpha, alpha / 2.0, alpha / 2.0)
    )
    for k in names:
        print("  %-24s %.6f +/- %.6f" % (k, summary["mean"][k], summary["std"][k]))

    out_json = args.out_json.strip()
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\n[INFO] Wrote summary: %s" % out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
