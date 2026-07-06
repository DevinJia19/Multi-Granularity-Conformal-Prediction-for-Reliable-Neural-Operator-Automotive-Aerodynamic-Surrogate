#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo evaluation for CP/CQR.

Preferred leakage-safe split-CP mode:
  Read an existing per-fold prediction CSV produced by test.py, cache those
  predictions, then randomly split only the cached predictions into
  calibration/evaluation subsets. This mode does not load a checkpoint and does
  not run model inference.

There is intentionally no checkpoint/model path here. Monte Carlo uses only
already-saved predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from cqr_common import apply_asymmetric_cqr, asymmetric_cqr_hat_q


def _read_float_column(rows, names, required=True):
    for name in names:
        if rows and name in rows[0]:
            values = []
            for row in rows:
                text = str(row.get(name, "")).strip()
                values.append(float(text) if text else np.nan)
            return np.array(values, dtype=np.float64)
    if required:
        raise ValueError("missing required column; tried: %s" % ", ".join(names))
    return None


def _load_prediction_csv(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError("prediction CSV not found: %s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("prediction CSV is empty: %s" % path)

    y = _read_float_column(rows, ["Ground_Truth_Cd", "y_true", "ground_truth"])
    q05 = _read_float_column(rows, ["Predicted_Cd_Q05", "q05_pred", "Q05"])
    q95 = _read_float_column(rows, ["Predicted_Cd_Q95", "q95_pred", "Q95"])
    q50 = _read_float_column(
        rows,
        ["Predicted_Cd_Q50", "q50_pred", "Q50", "Predicted_Cd"],
        required=False,
    )
    if q50 is None:
        q50 = np.full_like(y, np.nan, dtype=np.float64)
    return y.ravel(), q05.ravel(), q50.ravel(), q95.ravel()


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
        cqr_interval_score = _interval_score(lo, hi, yv, alpha)
    else:
        cqr_cover = float("nan")
        cqr_tail_low = float("nan")
        cqr_tail_high = float("nan")
        cqr_width = float("nan")
        cqr_interval_score = float("nan")

    out = {
        "cqr_coverage": cqr_cover,
        "cqr_tail_below_lower": cqr_tail_low,
        "cqr_tail_above_upper": cqr_tail_high,
        "cqr_width": cqr_width,
        "cqr_interval_score": cqr_interval_score,
        "pi90_uncalibrated": float(np.mean((yv >= q05v) & (yv <= q95v))),
        "pi90_uncalibrated_width": raw_width,
        "pi90_uncalibrated_interval_score": raw_interval_score,
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
    ap.add_argument(
        "--predictions-csv",
        default="",
        help="existing per-fold predictions.csv or per_sample_cd_cp_intervals.csv from test.py",
    )
    ap.add_argument("--n-cal", type=int, default=0, help="calibration size per replicate")
    ap.add_argument("--R", type=int, default=1000, dest="R", help="number of random replicates")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="", help="prediction cache npz path")
    ap.add_argument("--out-json", default="", help="write summary JSON")
    args = ap.parse_args()

    cache_path = args.cache.strip()
    predictions_csv = args.predictions_csv.strip()
    prediction_source = "splitcp_prediction_csv"

    if not predictions_csv:
        print("[ERROR] --predictions-csv is required.", file=sys.stderr)
        return 1

    if cache_path and os.path.isfile(cache_path):
        print("[INFO] Loading prediction CSV cache: %s" % cache_path)
        try:
            y, q05, q50, q95, _ = _load_cached_npz(cache_path, prediction_source)
        except Exception as exc:
            print("[ERROR] %s" % exc, file=sys.stderr)
            return 1
    else:
        print("[INFO] Loading existing prediction CSV: %s" % predictions_csv)
        try:
            y, q05, q50, q95 = _load_prediction_csv(predictions_csv)
        except Exception as exc:
            print("[ERROR] failed to load prediction CSV: %s" % exc, file=sys.stderr)
            return 1
        if cache_path:
            _save_cached_npz(
                cache_path,
                y,
                q05,
                q50,
                q95,
                prediction_source,
                {"predictions_csv": np.array(os.path.abspath(predictions_csv))},
            )
            print("[INFO] Wrote prediction CSV cache: %s" % cache_path)
    print("[INFO] Prediction-CSV mode: no checkpoint loading and no model inference.")

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
        "cqr_width",
        "cqr_interval_score",
        "pi90_uncalibrated",
        "pi90_uncalibrated_width",
        "pi90_uncalibrated_interval_score",
        "tail_below_q05",
        "tail_above_q95",
    ]
    if include_q50:
        names.append("frac_y_le_q50")
    else:
        print("[INFO] q50 is absent in existing predictions; skipping frac_y_le_q50.")

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
