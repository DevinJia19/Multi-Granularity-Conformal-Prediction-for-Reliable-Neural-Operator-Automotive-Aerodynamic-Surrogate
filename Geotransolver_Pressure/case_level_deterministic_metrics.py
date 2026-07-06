#!/usr/bin/env python3
"""Compute case-level deterministic metrics for surface channels.

This script intentionally reports mean case-level metrics. For every vehicle
case, it computes MAE, RMSE, R2, and Pearson per channel, then averages those
case metrics. This avoids letting cases with more surface points dominate the
reported deterministic performance.

Expected .npz keys:
    pred, target

Expected channel layout:
    0 pressure, 1 wss_x, 2 wss_y, 3 wss_z
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

CHANNELS = ["pressure", "wss_x", "wss_y", "wss_z"]
METRICS = ["mae", "rmse", "r2", "pearson"]


def as_points(x: np.ndarray, n_channels: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected (N,C) or (1,N,C), got {x.shape}")
    if x.shape[1] < n_channels:
        raise ValueError(f"Expected at least {n_channels} channels, got {x.shape}")
    return x[:, :n_channels]


def list_npz(path: Path) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {path}")
    return files


def list_npz_from_glob(pattern: str) -> list[Path]:
    files: list[Path] = []
    for item in sorted(glob.glob(pattern)):
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No .npz files matched: {pattern}")
    return files


def load_pred_target(path: Path, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        for key in ("pred", "target"):
            if key not in data:
                raise KeyError(f"{path} does not contain key '{key}'")
        pred = as_points(data["pred"], n_channels).astype(np.float64, copy=False)
        target = as_points(data["target"], n_channels).astype(np.float64, copy=False)
    if pred.shape != target.shape:
        raise ValueError(f"{path} pred shape {pred.shape} != target shape {target.shape}")
    return pred, target


def deterministic_metrics_1d(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    n = pred.size
    if n == 0:
        return {m: float("nan") for m in METRICS}

    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    target_centered = target - np.mean(target)
    ss_tot = float(np.sum(target_centered * target_centered))
    ss_res = float(np.sum(err * err))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    if n < 2:
        pearson = float("nan")
    else:
        pred_centered = pred - np.mean(pred)
        denom = float(
            np.sqrt(
                np.sum(pred_centered * pred_centered)
                * np.sum(target_centered * target_centered)
            )
        )
        pearson = (
            float(np.sum(pred_centered * target_centered) / denom)
            if denom > 0.0
            else float("nan")
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def read_vehicle_map(path: Path | None) -> dict[str, dict[str, str]]:
    """Read optional batch-to-vehicle metadata.

    The split-CP helper writes columns like fold_0_batch_file and
    fold_0_batch_index. We map each batch filename to the original row so the
    per-case output can include run_id / vehicle when available.
    """
    if path is None:
        return {}
    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key.endswith("_batch_file") and value:
                    mapping[value] = row
            if row.get("batch_file"):
                mapping[row["batch_file"]] = row
    return mapping


def case_id_from_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("batch_"):
        return stem.removeprefix("batch_")
    return stem


def evaluate_dataset(
    label: str,
    files: list[Path],
    channels: list[str],
    vehicle_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in files:
        pred, target = load_pred_target(file_path, len(channels))
        meta = vehicle_map.get(file_path.name, {})
        for ch_idx, ch_name in enumerate(channels):
            vals = deterministic_metrics_1d(pred[:, ch_idx], target[:, ch_idx])
            row: dict[str, Any] = {
                "dataset": label,
                "file": file_path.name,
                "case_id": case_id_from_file(file_path),
                "run_id": meta.get("run_id", ""),
                "vehicle": meta.get("vehicle", ""),
                "n_points": int(pred.shape[0]),
                "channel": ch_name,
            }
            row.update(vals)
            rows.append(row)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def summarize_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({str(r["dataset"]) for r in rows})
    channels = sorted(
        {str(r["channel"]) for r in rows},
        key=lambda x: CHANNELS.index(x) if x in CHANNELS else x,
    )
    summary: list[dict[str, Any]] = []
    for label in labels:
        for ch in channels:
            subset = [r for r in rows if r["dataset"] == label and r["channel"] == ch]
            out: dict[str, Any] = {
                "dataset": label,
                "channel": ch,
                "n_cases": len(subset),
                "mean_n_points": (
                    float(np.mean([float(r["n_points"]) for r in subset]))
                    if subset
                    else float("nan")
                ),
            }
            for metric in METRICS:
                vals = [float(r[metric]) for r in subset]
                mean, std = mean_std(vals)
                out[f"mean_case_{metric}"] = mean
                out[f"std_case_{metric}"] = std
            summary.append(out)
    return summary


def append_mean_over_datasets(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({str(r["dataset"]) for r in summary_rows})
    if len(datasets) <= 1:
        return summary_rows

    channels = sorted(
        {str(r["channel"]) for r in summary_rows},
        key=lambda x: CHANNELS.index(x) if x in CHANNELS else x,
    )
    out_rows = list(summary_rows)
    for ch in channels:
        subset = [r for r in summary_rows if r["channel"] == ch]
        out: dict[str, Any] = {
            "dataset": "mean_over_folds",
            "channel": ch,
            "n_cases": int(sum(int(r["n_cases"]) for r in subset)),
            "mean_n_points": float(np.nanmean([float(r["mean_n_points"]) for r in subset])),
        }
        for metric in METRICS:
            out[f"mean_case_{metric}"] = float(
                np.nanmean([float(r[f"mean_case_{metric}"]) for r in subset])
            )
            out[f"std_case_{metric}"] = ""
        out_rows.append(out)
    return out_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    extra = sorted({k for row in rows for k in row.keys()} - set(fieldnames))
    fieldnames.extend(extra)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def collect_inputs(args: argparse.Namespace) -> list[tuple[str, list[Path]]]:
    datasets: list[tuple[str, list[Path]]] = []
    checked_paths: list[Path] = []

    for test_dir in args.test_dir or []:
        datasets.append((test_dir.name, list_npz(test_dir)))

    for pattern in args.test_glob or []:
        files = list_npz_from_glob(pattern)
        datasets.append((Path(pattern).name or "glob", files))

    if args.splitcp4_runs_root is not None:
        for fold in range(args.n_folds):
            test_dir = args.splitcp4_runs_root / f"fold_{fold}" / "splitcp_test" / f"fold_{fold}"
            checked_paths.append(test_dir)
            if test_dir.is_dir():
                datasets.append((f"fold_{fold}", list_npz(test_dir)))

    if not datasets:
        msg = "No test .npz inputs found. Provide --test-dir, --test-glob, or --splitcp4-runs-root."
        if checked_paths:
            msg += "\nChecked splitcp4 test directories:\n  " + "\n  ".join(
                str(p) for p in checked_paths
            )
        raise ValueError(msg)
    return datasets


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute mean case-level MAE/RMSE/R2/Pearson per surface channel."
    )
    ap.add_argument(
        "--test-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing test .npz files. Can be repeated.",
    )
    ap.add_argument(
        "--test-glob",
        type=str,
        action="append",
        default=None,
        help="Glob matching test .npz files or directories. Can be repeated.",
    )
    ap.add_argument(
        "--splitcp4-runs-root",
        type=Path,
        default=None,
        help="Shortcut for runs/splitcp_4fold/fold_i/splitcp_test/fold_i.",
    )
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument(
        "--vehicle-map",
        type=Path,
        default=None,
        help="Optional CSV such as results/pressure_splitcp_4fold/batch_vehicle_map_test.csv.",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=CHANNELS,
        help="Channel names in pred/target order. Defaults to pressure wss_x wss_y wss_z.",
    )
    ap.add_argument("--out", type=Path, default=Path("results/case_level_deterministic_metrics"))
    args = ap.parse_args()

    datasets = collect_inputs(args)
    vehicle_map = read_vehicle_map(args.vehicle_map)

    all_case_rows: list[dict[str, Any]] = []
    for label, files in datasets:
        print(f"[INFO] Evaluating {label}: {len(files)} cases")
        all_case_rows.extend(evaluate_dataset(label, files, args.channels, vehicle_map))

    summary_rows = append_mean_over_datasets(summarize_case_rows(all_case_rows))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "case_level_deterministic_per_case.csv", all_case_rows)
    write_csv(args.out / "case_level_deterministic_summary.csv", summary_rows)

    payload = {
        "definition": (
            "MAE, RMSE, R2, and Pearson are computed separately for each case and "
            "channel; reported summary values are arithmetic means over cases."
        ),
        "channels": args.channels,
        "metrics": METRICS,
        "datasets": [
            {
                "label": label,
                "n_cases": len(files),
                "path_examples": [str(p) for p in files[:3]],
            }
            for label, files in datasets
        ],
        "summary": summary_rows,
    }
    write_json(args.out / "case_level_deterministic_summary.json", payload)

    print("[OK] Wrote:")
    print(f"  {args.out / 'case_level_deterministic_per_case.csv'}")
    print(f"  {args.out / 'case_level_deterministic_summary.csv'}")
    print(f"  {args.out / 'case_level_deterministic_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
