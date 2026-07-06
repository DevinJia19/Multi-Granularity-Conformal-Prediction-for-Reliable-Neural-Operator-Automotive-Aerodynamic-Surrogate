#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration-set-size sensitivity analysis for Cd conformal prediction.

This script only consumes cached OOF predictions on the 400 official
training-calibration cases. It never imports training code, never loads a
checkpoint, and never evaluates the official 50 test cases.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_M_LIST = (40, 80, 120, 160, 200)
PROJECT_ROOT = os.path.abspath(
    os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_OOF_DIR = os.path.join(PROJECT_ROOT, "results", "cvplus")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "cd_calibration_size_sensitivity")


def _first_available_npz(z: np.lib.npyio.NpzFile, names: Iterable[str], path: str) -> np.ndarray:
    for name in names:
        if name in z.files:
            return z[name]
    raise ValueError("%s missing any of keys: %s" % (path, ", ".join(names)))


def _load_npz_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        y = _first_available_npz(z, ("y", "cd_true", "y_true", "Ground_Truth_Cd"), path)
        q05 = _first_available_npz(z, ("q05", "q05_pred", "Predicted_Cd_Q05"), path)
        q95 = _first_available_npz(z, ("q95", "q95_pred", "Predicted_Cd_Q95"), path)
        q50 = None
        for name in ("q50", "q50_pred", "Predicted_Cd_Q50"):
            if name in z.files:
                q50 = z[name]
                break
    return (
        np.asarray(y, dtype=np.float64).ravel(),
        np.asarray(q05, dtype=np.float64).ravel(),
        None if q50 is None else np.asarray(q50, dtype=np.float64).ravel(),
        np.asarray(q95, dtype=np.float64).ravel(),
    )


def _load_csv_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    df = pd.read_csv(path)

    def pick(names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        raise ValueError("%s missing any of columns: %s" % (path, ", ".join(names)))

    y = pick(("y", "cd_true", "y_true", "Ground_Truth_Cd"))
    q05 = pick(("q05", "q05_pred", "Predicted_Cd_Q05"))
    q95 = pick(("q95", "q95_pred", "Predicted_Cd_Q95"))
    q50 = None
    for name in ("q50", "q50_pred", "Predicted_Cd_Q50"):
        if name in df.columns:
            q50 = df[name]
            break
    return (
        pd.to_numeric(y, errors="raise").to_numpy(dtype=np.float64),
        pd.to_numeric(q05, errors="raise").to_numpy(dtype=np.float64),
        None if q50 is None else pd.to_numeric(q50, errors="raise").to_numpy(dtype=np.float64),
        pd.to_numeric(q95, errors="raise").to_numpy(dtype=np.float64),
    )


def load_cached_oof_predictions(oof_dir: str, cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    paths: list[str]
    if cache_path:
        if not os.path.isfile(cache_path):
            raise FileNotFoundError("cached prediction file not found: %s" % cache_path)
        paths = [cache_path]
    else:
        pattern = os.path.join(oof_dir, "oof_fold_*.npz")
        paths = sorted(glob.glob(pattern))
        if not paths:
            merged = os.path.join(oof_dir, "oof_merged.npz")
            if os.path.isfile(merged):
                paths = [merged]
        if not paths:
            raise FileNotFoundError("no cached OOF predictions found under: %s" % oof_dir)

    y_parts, q05_parts, q50_parts, q95_parts = [], [], [], []
    have_q50 = True
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            y, q05, q50, q95 = _load_npz_file(path)
        elif ext == ".csv":
            y, q05, q50, q95 = _load_csv_file(path)
        else:
            raise ValueError("unsupported cached prediction format: %s" % path)

        sizes = {arr.size for arr in (y, q05, q95)}
        if q50 is not None:
            sizes.add(q50.size)
        if len(sizes) != 1:
            raise ValueError("array length mismatch in %s" % path)

        y_parts.append(y)
        q05_parts.append(q05)
        q95_parts.append(q95)
        if q50 is None:
            have_q50 = False
        else:
            q50_parts.append(q50)

    y_all = np.concatenate(y_parts)
    q05_all = np.concatenate(q05_parts)
    q95_all = np.concatenate(q95_parts)
    q50_all = np.concatenate(q50_parts) if have_q50 and q50_parts else None

    finite_arrays = [y_all, q05_all, q95_all]
    if q50_all is not None:
        finite_arrays.append(q50_all)
    if not all(np.all(np.isfinite(arr)) for arr in finite_arrays):
        raise ValueError("cached predictions contain NaN or infinite values")

    return y_all, q05_all, q50_all, q95_all, paths


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    n_cal = int(scores.size)
    if n_cal <= 0:
        raise ValueError("calibration scores are empty")
    k = int(np.ceil((n_cal + 1) * float(level)))
    k = min(max(k, 1), n_cal)
    return float(np.sort(scores)[k - 1])


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return (
        (upper - lower)
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def one_replicate(
    rng: np.random.Generator,
    y: np.ndarray,
    q05: np.ndarray,
    q95: np.ndarray,
    m: int,
    n_eval: int,
    alpha: float,
) -> dict[str, float | int]:
    n = int(y.size)
    perm = rng.permutation(n)
    cal_idx = perm[:m]
    eval_idx = perm[m : m + n_eval]

    s_l = np.maximum(q05[cal_idx] - y[cal_idx], 0.0)
    s_u = np.maximum(y[cal_idx] - q95[cal_idx], 0.0)
    level = 1.0 - alpha / 2.0
    q_l = conformal_quantile(s_l, level)
    q_u = conformal_quantile(s_u, level)

    y_eval = y[eval_idx]
    lower = q05[eval_idx] - q_l
    upper = q95[eval_idx] + q_u

    return {
        "n_cal": int(m),
        "n_eval": int(n_eval),
        "q_L": q_l,
        "q_U": q_u,
        "coverage": float(np.mean((y_eval >= lower) & (y_eval <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "lower_tail": float(np.mean(y_eval < lower)),
        "upper_tail": float(np.mean(y_eval > upper)),
        "mean_interval_score": float(np.mean(interval_score(lower, upper, y_eval, alpha))),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("calibration_size", sort=True):
        rows.append(
            {
                "calibration_size": int(m),
                "mean_coverage": float(g["coverage"].mean()),
                "std_coverage": float(g["coverage"].std(ddof=0)),
                "mean_width": float(g["mean_width"].mean()),
                "std_width": float(g["mean_width"].std(ddof=0)),
                "mean_interval_score": float(g["mean_interval_score"].mean()),
                "std_interval_score": float(g["mean_interval_score"].std(ddof=0)),
                "mean_lower_tail": float(g["lower_tail"].mean()),
                "std_lower_tail": float(g["lower_tail"].std(ddof=0)),
                "mean_upper_tail": float(g["upper_tail"].mean()),
                "std_upper_tail": float(g["upper_tail"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def print_latex_table(summary_df: pd.DataFrame) -> None:
    print("\nLaTeX table:")
    print(r"\begin{tabular}{rrrrr}")
    print(r"\toprule")
    print(r"Calibration size & Mean coverage & Std. coverage & Mean width & Std. width \\")
    print(r"\midrule")
    for row in summary_df.itertuples(index=False):
        print(
            "%d & %.4f & %.4f & %.6f & %.6f \\\\"
            % (
                row.calibration_size,
                row.mean_coverage,
                row.std_coverage,
                row.mean_width,
                row.std_width,
            )
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def parse_m_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cd calibration-size sensitivity analysis using cached OOF predictions only."
    )
    ap.add_argument(
        "--oof-dir",
        default=os.getenv("CVPLUS_OOF_DIR", DEFAULT_OOF_DIR),
        help="directory containing cached OOF oof_fold_*.npz files",
    )
    ap.add_argument(
        "--cache",
        default="",
        help="single cached OOF prediction file (.npz or .csv); overrides --oof-dir",
    )
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--m-list", default=",".join(str(x) for x in DEFAULT_M_LIST))
    ap.add_argument("--R", type=int, default=500, dest="R")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--expected-n", type=int, default=400)
    ap.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    try:
        m_list = parse_m_list(args.m_list)
        y, q05, q50, q95, source_paths = load_cached_oof_predictions(
            args.oof_dir.strip() or DEFAULT_OOF_DIR,
            args.cache.strip(),
        )
    except Exception as exc:
        print("[ERROR] %s" % exc, file=sys.stderr)
        return 1

    n = int(y.size)
    if n != int(args.expected_n):
        print(
            "[ERROR] expected exactly %d cached official training-calibration OOF cases, got %d. "
            "Refusing to run to avoid accidentally using official test cases."
            % (int(args.expected_n), n),
            file=sys.stderr,
        )
        return 1

    if int(args.R) <= 0:
        print("[ERROR] R must be positive", file=sys.stderr)
        return 1
    if not (0.0 < float(args.alpha) < 1.0):
        print("[ERROR] alpha must be in (0, 1)", file=sys.stderr)
        return 1
    for m in m_list:
        if m <= 0:
            print("[ERROR] calibration sizes must be positive: %s" % m_list, file=sys.stderr)
            return 1
        if m + int(args.n_eval) > n:
            print(
                "[ERROR] calibration size %d plus n_eval %d exceeds N=%d"
                % (m, int(args.n_eval), n),
                file=sys.stderr,
            )
            return 1

    os.makedirs(args.out_dir, exist_ok=True)
    replicates_csv = os.path.join(args.out_dir, "cd_calibration_size_replicates.csv")
    summary_csv = os.path.join(args.out_dir, "cd_calibration_size_summary.csv")
    summary_json = os.path.join(args.out_dir, "cd_calibration_size_summary.json")

    print("[INFO] Loaded %d cached OOF cases from %d file(s)." % (n, len(source_paths)))
    print("[INFO] Source files:")
    for path in source_paths:
        print("  %s" % path)
    if q50 is None:
        print("[INFO] q50 is absent; it is not needed for this sensitivity analysis.")
    print(
        "[INFO] alpha=%.3f, m_list=%s, R=%d, n_eval=%d, seed=%d"
        % (float(args.alpha), m_list, int(args.R), int(args.n_eval), int(args.seed))
    )

    rng = np.random.default_rng(int(args.seed))
    rows = []
    for m in m_list:
        for r in range(int(args.R)):
            out = one_replicate(rng, y, q05, q95, int(m), int(args.n_eval), float(args.alpha))
            rows.append(
                {
                    "calibration_size": int(m),
                    "repeat": int(r),
                    "n_total": n,
                    "alpha": float(args.alpha),
                    "seed": int(args.seed),
                    **out,
                }
            )

    replicates_df = pd.DataFrame(rows)
    summary_df = summarize(replicates_df)

    replicates_df.to_csv(replicates_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "protocol": "cd_calibration_size_sensitivity_cached_oof_only",
        "alpha": float(args.alpha),
        "R": int(args.R),
        "seed": int(args.seed),
        "n_total": n,
        "n_eval": int(args.n_eval),
        "m_list": [int(m) for m in m_list],
        "source_files": source_paths,
        "replicates_csv": replicates_csv,
        "summary_csv": summary_csv,
        "summary": summary_df.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[OK] Wrote replicate results: %s" % replicates_csv)
    print("[OK] Wrote summary CSV: %s" % summary_csv)
    print("[OK] Wrote summary JSON: %s" % summary_json)
    print_latex_table(summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
