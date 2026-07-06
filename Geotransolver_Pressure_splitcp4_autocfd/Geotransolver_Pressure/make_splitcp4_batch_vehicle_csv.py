#!/usr/bin/env python3
"""Write a CSV mapping AutoCFD test run IDs to split-CP batch npz names.

For the ordinary 4-fold split-CP workflow, the official test set is evaluated
once by each fold model. This script writes one row per vehicle/run ID, with
the corresponding batch file for fold_0 ... fold_3 on that row.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from split_dataset import AUTO_CFD_TEST_IDS

RUN_RE = re.compile(r"run_(\d+)(?:\.zarr)?$", re.IGNORECASE)


def _batch_name(idx: int) -> str:
    return f"batch_{idx:05d}.npz"


def _parse_run_id(path_or_name: str) -> int:
    name = Path(path_or_name.strip()).name
    match = RUN_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse run ID from: {path_or_name}")
    return int(match.group(1))


def _ids_from_manifest(manifest: Path) -> list[int]:
    ids: list[int] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(_parse_run_id(line))
    return ids


def _ids_from_dir(split_dir: Path) -> list[int]:
    ids = [_parse_run_id(p.name) for p in split_dir.iterdir() if RUN_RE.match(p.name)]
    return sorted(ids)


def _ids_from_current_data_splits(split_name: str, data_splits_dir: Path) -> list[int] | None:
    manifest = data_splits_dir / f"{split_name}_files.txt"
    if manifest.is_file():
        return _ids_from_manifest(manifest)
    split_dir = data_splits_dir / split_name
    if split_dir.is_dir():
        return _ids_from_dir(split_dir)
    return None


def _npz_dir(runs_root: Path, split_name: str, fold: int) -> Path:
    return runs_root / f"fold_{fold}" / f"splitcp_{split_name}" / f"fold_{fold}"


def build_rows(
    n_folds: int,
    runs_root: Path,
    data_splits_dir: Path,
    prefer_current_split: bool,
) -> list[dict[str, str]]:
    ordered_ids = sorted(AUTO_CFD_TEST_IDS)

    # A current data_splits/test manifest may exist after the run and gives the
    # exact local dataset order used by inference.
    current_ids = _ids_from_current_data_splits("test", data_splits_dir) if prefer_current_split else None
    if current_ids:
        ordered_ids = current_ids

    per_fold_ids = {fold: ordered_ids for fold in range(n_folds)}
    run_ids = sorted(ordered_ids)
    rows_by_id: dict[int, dict[str, str]] = {
        rid: {"run_id": str(rid), "vehicle": f"run_{rid}.zarr", "split": "test"}
        for rid in run_ids
    }

    for fold in range(n_folds):
        ids = per_fold_ids[fold]
        id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
        folder = _npz_dir(runs_root, "test", fold)
        for rid, row in rows_by_id.items():
            idx = id_to_idx.get(rid)
            if idx is None:
                row[f"fold_{fold}_batch_index"] = ""
                row[f"fold_{fold}_batch_file"] = ""
                row[f"fold_{fold}_npz_exists"] = ""
                continue
            batch_file = _batch_name(idx)
            row[f"fold_{fold}_batch_index"] = str(idx)
            row[f"fold_{fold}_batch_file"] = batch_file
            row[f"fold_{fold}_npz_exists"] = str((folder / batch_file).is_file())

    return [rows_by_id[rid] for rid in run_ids]


def write_csv(path: Path, rows: list[dict[str, str]], n_folds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "vehicle", "split"]
    for fold in range(n_folds):
        fieldnames.extend(
            [
                f"fold_{fold}_batch_index",
                f"fold_{fold}_batch_file",
                f"fold_{fold}_npz_exists",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Map split-CP 4-fold test batch_XXXXX.npz names back to AutoCFD run IDs."
    )
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=repo_root / "runs" / "splitcp_4fold",
        help="Root containing fold_*/splitcp_test outputs.",
    )
    parser.add_argument(
        "--data-splits-dir",
        type=Path,
        default=repo_root / "data_splits",
        help="Optional current data_splits directory used to refine test order.",
    )
    parser.add_argument(
        "--ignore-current-data-splits",
        action="store_true",
        help="Use canonical AutoCFD test ordering instead of current test manifest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: results/pressure_splitcp_4fold/batch_vehicle_map_test.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    out = args.out
    if out is None:
        out = (
            repo_root
            / "results"
            / "pressure_splitcp_4fold"
            / "batch_vehicle_map_test.csv"
        )
    rows = build_rows(
        n_folds=args.n_folds,
        runs_root=args.runs_root,
        data_splits_dir=args.data_splits_dir,
        prefer_current_split=not args.ignore_current_data_splits,
    )
    write_csv(out, rows, args.n_folds)
    print(f"[OK] Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
