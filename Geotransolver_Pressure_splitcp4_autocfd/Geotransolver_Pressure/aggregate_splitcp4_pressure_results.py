#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate 4-fold ordinary split conformal prediction results for pressure/WSS.

This is intentionally NOT CV+ / OOF merging. Each fold is evaluated with:
  - its own fold checkpoint trained on 3/4 of the AutoCFD official train pool;
  - its own held-out 1/4 fold for qhat calibration;
  - the same AutoCFD official test set.

The script averages qhat and metrics from cp_compare_global_point_case.py. It can
also build an averaged test-npz directory and averaged_qhat.json so that the
existing cp_write_vtp_global_point_case.py can directly generate VTP files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

CHANNELS = ["pressure", "wss_x", "wss_y", "wss_z"]
MODES = ["global_abs", "point_sigma", "case_sigma"]


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
        else:
            out[prefix] = float(obj)
    return out


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_fold_dirs(root: Path, n_folds: int) -> list[Path]:
    fold_dirs = [root / f"fold_{i}" for i in range(n_folds)]
    existing = [p for p in fold_dirs if (p / "summary.json").is_file()]
    if not existing:
        existing = sorted(p for p in root.glob("fold_*") if (p / "summary.json").is_file())
    if not existing:
        raise FileNotFoundError(f"No fold summary.json files found under {root}")
    return existing


def _aggregate_flat_summaries(fold_dirs: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat_rows: list[dict[str, Any]] = []
    for fd in fold_dirs:
        payload = _read_json(fd / "summary.json")
        flat = _flatten_numeric(payload)
        flat["fold"] = fd.name
        flat["__path__"] = str((fd / "summary.json").resolve())
        flat_rows.append(flat)

    keys = sorted(k for row in flat_rows for k in row.keys() if k not in {"fold", "__path__"})
    agg = {
        "n_folds_found": len(fold_dirs),
        "files": [str((fd / "summary.json").resolve()) for fd in fold_dirs],
        "mean": {},
        "std": {},
    }
    for k in keys:
        vals = [float(row[k]) for row in flat_rows if k in row]
        mean, std = _mean_std(vals)
        agg["mean"][k] = mean
        agg["std"][k] = std
    return agg, flat_rows


def _aggregate_qhats(fold_dirs: list[Path]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        mode: {ch: [] for ch in CHANNELS} for mode in MODES
    }
    alpha = None
    eps = None
    case_score = None
    for fd in fold_dirs:
        obj = _read_json(fd / "qhat.json")
        qobj = obj.get("qhat", obj)
        alpha = obj.get("alpha", alpha)
        eps = obj.get("eps", eps)
        case_score = obj.get("case_score", case_score)
        for mode in MODES:
            if mode not in qobj:
                continue
            for ch in CHANNELS:
                values[mode][ch].append(float(qobj[mode][ch]))

    qhat_mean = {mode: {} for mode in MODES}
    qhat_std = {mode: {} for mode in MODES}
    for mode in MODES:
        for ch in CHANNELS:
            mean, std = _mean_std(values[mode][ch])
            qhat_mean[mode][ch] = mean
            qhat_std[mode][ch] = std

    return {
        "alpha": alpha,
        "target_coverage": None if alpha is None else 1.0 - float(alpha),
        "eps": eps,
        "case_score": case_score,
        "protocol": "ordinary_split_cp_4fold_mean_no_oof_no_final_refit",
        "qhat": qhat_mean,
        "qhat_std": qhat_std,
        "note": (
            "qhat values are arithmetic means across the four independent split-CP folds. "
            "They are provided for visualization convenience; official reported metrics "
            "should use the per-fold metric mean/std in splitcp_4fold_summary.json."
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate_comparison_tables(fold_dirs: list[Path]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str], dict[str, list[float]]] = {}
    all_metric_names: set[str] = set()
    for fd in fold_dirs:
        for row in _read_csv(fd / "comparison_table.csv"):
            mode = row.get("mode", "")
            channel = row.get("channel", "")
            key = (mode, channel)
            bucket.setdefault(key, {})
            for k, v in row.items():
                if k in {"mode", "channel"} or v in {"", None}:
                    continue
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    continue
                bucket[key].setdefault(k, []).append(x)
                all_metric_names.add(k)

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for ch in CHANNELS:
            vals_by_metric = bucket.get((mode, ch), {})
            if not vals_by_metric:
                continue
            out: dict[str, Any] = {"mode": mode, "channel": ch}
            for metric in sorted(all_metric_names):
                vals = vals_by_metric.get(metric, [])
                mean, std = _mean_std(vals)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            rows.append(out)
    return rows


def _copy_or_average_npz_arrays(fold_test_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Average pred/sigma over folds case-by-case for VTP visualization.

    The existing cp_write_vtp_global_point_case.py expects a single test directory
    containing pred, target, sigma, surface_mesh_centers. We average pred and sigma
    across the four fold models; target and mesh arrays are copied from fold 0.
    """
    if not fold_test_dirs:
        raise ValueError("No fold test directories provided.")
    for p in fold_test_dirs:
        if not p.is_dir():
            raise FileNotFoundError(f"Missing test npz directory: {p}")

    file_lists = [sorted(p.glob("*.npz")) for p in fold_test_dirs]
    if not file_lists[0]:
        raise FileNotFoundError(f"No .npz files found in {fold_test_dirs[0]}")
    n_cases = len(file_lists[0])
    for idx, files in enumerate(file_lists):
        if len(files) != n_cases:
            raise ValueError(
                f"Fold test dir {fold_test_dirs[idx]} has {len(files)} files, expected {n_cases}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for case_idx in range(n_cases):
        first_file = file_lists[0][case_idx]
        pred_sum = None
        sigma_sum = None
        first_payload: dict[str, Any] = {}

        for fold_idx, files in enumerate(file_lists):
            f = files[case_idx]
            # Prefer matching sorted order; warn by recording names if stems differ.
            with np.load(f, allow_pickle=True) as data:
                if "pred" not in data or "sigma" not in data:
                    raise KeyError(f"{f} must contain pred and sigma for averaged visualization.")
                pred = np.asarray(data["pred"], dtype=np.float32)
                sigma = np.asarray(data["sigma"], dtype=np.float32)
                pred_sum = pred if pred_sum is None else pred_sum + pred
                sigma_sum = sigma if sigma_sum is None else sigma_sum + sigma

                if fold_idx == 0:
                    for key in data.files:
                        if key not in {"pred", "sigma"}:
                            first_payload[key] = data[key]

        assert pred_sum is not None and sigma_sum is not None
        save_dict = dict(first_payload)
        save_dict["pred"] = pred_sum / float(len(file_lists))
        save_dict["sigma"] = sigma_sum / float(len(file_lists))
        save_dict["splitcp4_average_note"] = np.array(
            "pred and sigma are arithmetic means across the four split-CP fold models; "
            "target and mesh arrays are copied from fold_0.",
            dtype=object,
        )
        save_dict["source_fold_files"] = np.array(
            [str(files[case_idx].resolve()) for files in file_lists], dtype=object
        )
        out_path = out_dir / first_file.name
        np.savez_compressed(out_path, **save_dict)
        written.append(str(out_path.resolve()))

    return {
        "averaged_test_dir": str(out_dir.resolve()),
        "n_cases": n_cases,
        "files": written,
        "source_test_dirs": [str(p.resolve()) for p in fold_test_dirs],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate 4-fold ordinary split CP pressure results.")
    ap.add_argument("--root", type=Path, default=Path("results/pressure_splitcp_4fold"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs/splitcp_4fold"))
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--make-average-npz", action="store_true")
    ap.add_argument("--average-npz-out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    fold_dirs = _collect_fold_dirs(root, args.n_folds)

    summary_agg, flat_rows = _aggregate_flat_summaries(fold_dirs)
    qhat_agg = _aggregate_qhats(fold_dirs)
    table_rows = _aggregate_comparison_tables(fold_dirs)

    summary: dict[str, Any] = {
        "protocol": "ordinary_split_cp_4fold_mean_no_oof_no_final_refit",
        "root": str(root),
        "requested_n_folds": int(args.n_folds),
        "folds_found": [fd.name for fd in fold_dirs],
        "metrics": summary_agg,
        "qhat_mean_for_visualization": qhat_agg,
        "important_note": (
            "This is NOT CV+. Each fold uses only its own held-out calibration fold "
            "to compute qhat. No OOF calibration scores are merged, and no final "
            "full-400 refit model is evaluated in this protocol."
        ),
    }

    _write_json(root / "splitcp_4fold_summary.json", summary)
    _write_json(root / "averaged_qhat.json", qhat_agg)
    _write_csv(root / "splitcp_4fold_fold_metrics_flat.csv", flat_rows)
    _write_csv(root / "comparison_table_4fold_mean.csv", table_rows)

    all_per_case: list[dict[str, Any]] = []
    for fd in fold_dirs:
        for row in _read_csv(fd / "per_case_metrics.csv"):
            row = dict(row)
            row["fold"] = fd.name
            all_per_case.append(row)
    _write_csv(root / "per_case_metrics_all_folds.csv", all_per_case)

    if args.make_average_npz:
        out_dir = args.average_npz_out or (root / "averaged_test_npz")
        fold_test_dirs = [args.runs_root.resolve() / f"fold_{i}" / "splitcp_test" / f"fold_{i}" for i in range(args.n_folds)]
        avg_info = _copy_or_average_npz_arrays(fold_test_dirs, out_dir.resolve())
        summary["averaged_test_npz"] = avg_info
        _write_json(root / "splitcp_4fold_summary.json", summary)
        _write_json(root / "averaged_test_npz_manifest.json", avg_info)

    print(f"[OK] Aggregated {len(fold_dirs)} split-CP folds under {root}")
    print(f"[OK] Summary JSON: {root / 'splitcp_4fold_summary.json'}")
    print(f"[OK] Mean table:    {root / 'comparison_table_4fold_mean.csv'}")
    print(f"[OK] Qhat JSON for VTP: {root / 'averaged_qhat.json'}")
    if args.make_average_npz:
        print(f"[OK] Averaged test NPZ dir: {args.average_npz_out or (root / 'averaged_test_npz')}")
    print("\nCommon VTP command:")
    print("python cp_write_vtp_global_point_case.py \\")
    print(f"  --test-dir {root / 'averaged_test_npz'} \\")
    print(f"  --qhat-json {root / 'averaged_qhat.json'} \\")
    print(f"  --out {root / 'vtp_averaged'} \\")
    print("  --modes global_abs point_sigma case_sigma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
