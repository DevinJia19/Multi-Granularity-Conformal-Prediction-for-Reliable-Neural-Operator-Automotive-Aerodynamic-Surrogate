#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate 5-fold ordinary split conformal prediction results.

This is intentionally NOT CV+ / OOF merging. Each fold is evaluated with:
  - its own fold checkpoint trained on K-1 folds;
  - its own held-out calibration fold for q_l/q_u;
  - the same AutoCFD official test set.

The script averages the metrics produced by test.py and, optionally, the
Monte-Carlo summaries produced by evaluate_cp_replicates.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any

import numpy as np


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out[prefix] = float(obj)
    return out


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _aggregate_json_files(files: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    flat_rows = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append({"path": path, "payload": payload})
        flat = _flatten_numeric(payload)
        flat["__path__"] = path  # type: ignore[assignment]
        flat_rows.append(flat)

    keys = sorted({k for row in flat_rows for k in row.keys() if k != "__path__"})
    agg = {
        "n_folds_found": len(files),
        "files": [os.path.abspath(f) for f in files],
        "mean": {},
        "std": {},
    }
    for k in keys:
        vals = [float(row[k]) for row in flat_rows if k in row]
        mean, std = _safe_mean_std(vals)
        agg["mean"][k] = mean
        agg["std"][k] = std
    return agg, flat_rows


def _write_flat_csv(path: str, flat_rows: list[dict[str, Any]]) -> None:
    if not flat_rows:
        return
    keys = sorted({k for row in flat_rows for k in row.keys()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate 5-fold split CP metrics.")
    ap.add_argument(
        "--root",
        default="./results/splitcp_5fold",
        help="root result directory containing fold_0 ... fold_4",
    )
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    test_files = [os.path.join(root, f"fold_{i}", "metrics_summary.json") for i in range(args.n_folds)]
    test_files = [f for f in test_files if os.path.isfile(f)]
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(root, "fold_*", "metrics_summary.json")))

    mc_files = [os.path.join(root, f"fold_{i}", "mc_replicates_summary.json") for i in range(args.n_folds)]
    mc_files = [f for f in mc_files if os.path.isfile(f)]
    if not mc_files:
        mc_files = sorted(glob.glob(os.path.join(root, "fold_*", "mc_replicates_summary.json")))

    if not test_files and not mc_files:
        print(f"[ERROR] No fold metrics found under {root}")
        return 1

    summary: dict[str, Any] = {
        "protocol": "ordinary_split_cp_5fold_mean_no_oof_no_final_refit",
        "root": os.path.abspath(root),
        "requested_n_folds": int(args.n_folds),
    }
    all_flat_rows: list[dict[str, Any]] = []

    if test_files:
        test_agg, test_flat = _aggregate_json_files(test_files)
        summary["test"] = test_agg
        for row in test_flat:
            row = dict(row)
            row["kind"] = "test"
            all_flat_rows.append(row)
        print(f"[OK] Aggregated test metrics from {len(test_files)} folds")

    if mc_files:
        mc_agg, mc_flat = _aggregate_json_files(mc_files)
        summary["monte_carlo"] = mc_agg
        for row in mc_flat:
            row = dict(row)
            row["kind"] = "monte_carlo"
            all_flat_rows.append(row)
        print(f"[OK] Aggregated Monte-Carlo metrics from {len(mc_files)} folds")

    out_json = args.out_json or os.path.join(root, "splitcp_5fold_summary.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Summary JSON: {out_json}")

    out_csv = args.out_csv or os.path.join(root, "splitcp_5fold_fold_metrics_flat.csv")
    _write_flat_csv(out_csv, all_flat_rows)
    print(f"[OK] Per-fold flat CSV: {out_csv}")

    # Concise console highlights for common Cd metrics.
    common_keys = [
        "point_metrics.MAE",
        "point_metrics.MSE",
        "point_metrics.RMSE",
        "point_metrics.R2_Score",
        "cqr_metrics.calibrated_test_coverage_pct",
        "cqr_metrics.calibrated_test_width",
        "cqr_metrics.calibrated_test_interval_score",
        "cqr_metrics.raw_test_coverage_pct",
        "cqr_metrics.raw_test_width",
    ]
    if "test" in summary:
        print("\n=== 5-fold split CP test mean ± std ===")
        means = summary["test"]["mean"]
        stds = summary["test"]["std"]
        for k in common_keys:
            if k in means:
                print(f"{k:48s} {means[k]:.8f} ± {stds[k]:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
