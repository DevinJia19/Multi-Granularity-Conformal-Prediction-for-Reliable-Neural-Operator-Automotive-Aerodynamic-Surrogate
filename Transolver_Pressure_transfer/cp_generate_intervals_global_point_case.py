#!/usr/bin/env python3
"""Generate calibrated prediction intervals with global, point-wise, and case-wise CP.

This is the final interval writer for trained Transolver transfer runs. It uses
the same three definitions as cp_compare_global_point_case.py:

1) global_abs:
   score = abs(target - pred)
   interval = pred +/- q_abs

2) point_sigma:
   score = abs(target - pred) / max(sigma, eps)
   interval = pred +/- q_sigma * sigma

3) case_sigma:
   reduce normalized point scores inside each calibration case, then take a
   conformal quantile across cases
   interval = pred +/- q_case * sigma

Plain Transolver outputs may not contain sigma. In that case sigma=1 is used,
so point_sigma and case_sigma become absolute-residual CP variants.

Example:
    python cp_generate_intervals_global_point_case.py \
      --calib-dir runs/transolver_single_cp/split_0/single_cp_calib/split_0 \
      --test-dir runs/transolver_single_cp/split_0/single_cp_test/split_0 \
      --out results/pressure_transolver_single_cp/split_0/intervals

If qhat.json has already been computed:
    python cp_generate_intervals_global_point_case.py \
      --test-dir runs/transolver_single_cp/split_0/single_cp_test/split_0 \
      --qhat-json results/pressure_transolver_single_cp/split_0/qhat.json \
      --out results/pressure_transolver_single_cp/split_0/intervals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from cp_compare_global_point_case import (
    CHANNELS,
    MODES,
    compute_case_qhat,
    compute_point_or_global_qhat,
    list_npz,
    list_npz_from_glob,
)
from cp_write_vtp_global_point_case import load_qhats


def as_points(x: np.ndarray, last_dim: int | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array or (1,N,C), got shape={x.shape}")
    if last_dim is not None and x.shape[1] < last_dim:
        raise ValueError(f"Expected at least {last_dim} columns, got shape={x.shape}")
    return x


def load_or_compute_qhats(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    if args.qhat_json is not None:
        return load_qhats(args.qhat_json)

    if args.calib_glob:
        calib_files = list_npz_from_glob(args.calib_glob)
    elif args.calib_dir:
        calib_files = list_npz(args.calib_dir)
    else:
        raise ValueError("Provide --qhat-json, --calib-dir, or --calib-glob.")

    print(f"[INFO] calibration files: {len(calib_files)}")
    qhats: Dict[str, np.ndarray] = {}
    qhat_meta: Dict[str, Dict] = {}

    print("[INFO] Computing global_abs qhat...")
    qhats["global_abs"], qhat_meta["global_abs"] = compute_point_or_global_qhat(
        calib_files,
        args.alpha,
        args.eps,
        mode="global_abs",
        score_sample_per_file=args.score_sample_per_file,
        score_sample_seed=args.score_sample_seed,
    )
    print("[INFO] Computing point_sigma qhat...")
    qhats["point_sigma"], qhat_meta["point_sigma"] = compute_point_or_global_qhat(
        calib_files,
        args.alpha,
        args.eps,
        mode="point_sigma",
        score_sample_per_file=args.score_sample_per_file,
        score_sample_seed=args.score_sample_seed,
    )
    print("[INFO] Computing case_sigma qhat...")
    qhats["case_sigma"], qhat_meta["case_sigma"] = compute_case_qhat(
        calib_files, args.alpha, args.eps, args.case_score
    )

    qhat_json = {
        mode: {CHANNELS[i]: float(qhats[mode][i]) for i in range(4)}
        for mode in MODES
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "qhat.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": args.alpha,
                "eps": args.eps,
                "case_score": args.case_score,
                "score_sample_per_file": args.score_sample_per_file,
                "score_sample_seed": args.score_sample_seed,
                "calib_dir": str(args.calib_dir) if args.calib_dir else None,
                "calib_glob": args.calib_glob,
                "qhat": qhat_json,
                "meta": qhat_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK] Wrote qhat: {args.out / 'qhat.json'}")
    return qhats


def compute_half_width(mode: str, qhat: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if mode == "global_abs":
        return np.broadcast_to(qhat.reshape(1, 4), sigma.shape).astype(np.float32, copy=False)
    if mode in ("point_sigma", "case_sigma"):
        return (qhat.reshape(1, 4) * sigma).astype(np.float32, copy=False)
    raise ValueError(f"Unknown mode: {mode}")


def copy_optional_arrays(src: np.lib.npyio.NpzFile, save_dict: Dict[str, np.ndarray]) -> None:
    for key in ("surface_mesh_centers", "surface_normals", "surface_areas"):
        if key in src:
            save_dict[key] = np.asarray(src[key])


def write_intervals_for_file(
    npz_path: Path,
    out_dir: Path,
    qhats: Dict[str, np.ndarray],
    modes: Iterable[str],
    eps: float,
    compressed: bool,
    include_mesh: bool,
) -> List[Dict]:
    print(f"[INFO] Loading {npz_path}")
    with np.load(npz_path, allow_pickle=True) as d:
        pred = as_points(d["pred"], 4)[:, :4].astype(np.float32, copy=False)
        target = None
        if "target" in d:
            target = as_points(d["target"], 4)[:, :4].astype(np.float32, copy=False)
        if "sigma" in d:
            sigma = as_points(d["sigma"], 4)[:, :4].astype(np.float32, copy=False)
        else:
            sigma = np.ones_like(pred, dtype=np.float32)
        sigma = np.maximum(sigma, eps)

        common = {
            "pred": pred,
            "sigma": sigma,
        }
        if target is not None:
            common["target"] = target
        if include_mesh:
            copy_optional_arrays(d, common)

        rows: List[Dict] = []
        for mode in modes:
            if mode not in qhats:
                raise KeyError(f"Mode '{mode}' not found in qhat data.")
            half_width = compute_half_width(mode, qhats[mode], sigma)
            lower = pred - half_width
            upper = pred + half_width
            width = 2.0 * half_width
            covered = None
            if target is not None:
                covered = ((target >= lower) & (target <= upper)).astype(np.uint8)

            save_dict = dict(common)
            save_dict.update(
                {
                    "lower": lower.astype(np.float32, copy=False),
                    "upper": upper.astype(np.float32, copy=False),
                    "width": width.astype(np.float32, copy=False),
                    "qhat": qhats[mode].astype(np.float32, copy=False),
                    "mode": np.asarray(mode),
                    "channels": np.asarray(CHANNELS),
                }
            )
            if covered is not None:
                save_dict["covered"] = covered

            mode_dir = out_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            out_path = mode_dir / f"{npz_path.stem}_{mode}_intervals.npz"
            if compressed:
                np.savez_compressed(out_path, **save_dict)
            else:
                np.savez(out_path, **save_dict)
            print(f"[INFO] Wrote {out_path}")

            mean_width = width.mean(axis=0)
            row = {
                "mode": mode,
                "file": npz_path.name,
                "n_points": int(pred.shape[0]),
            }
            for i, ch in enumerate(CHANNELS):
                if covered is not None:
                    row[f"coverage_{ch}"] = float(covered.mean(axis=0)[i])
                row[f"mean_width_{ch}"] = float(mean_width[i])
            rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/cp_intervals_global_point_case"))
    ap.add_argument("--qhat-json", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=None)
    ap.add_argument("--calib-glob", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--case-score", choices=["quantile", "max", "mean"], default="quantile")
    ap.add_argument("--score-sample-per-file", type=int, default=None)
    ap.add_argument("--score-sample-seed", type=int, default=42)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--uncompressed", action="store_true")
    ap.add_argument("--no-mesh", action="store_true", help="Do not copy mesh arrays into interval files.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    qhats = load_or_compute_qhats(args)

    test_files = list_npz(args.test_dir)
    if args.files:
        selected = set(args.files)
        test_files = [p for p in test_files if p.name in selected or str(p) in selected]
        if not test_files:
            raise FileNotFoundError(f"No requested --files were found in {args.test_dir}")
    if args.max_cases is not None:
        test_files = test_files[: args.max_cases]

    print(f"[INFO] test files: {len(test_files)}")
    print(f"[INFO] modes     : {args.modes}")
    print("[INFO] qhats:")
    for mode in args.modes:
        q = qhats[mode]
        print("  " + mode + ": " + ", ".join(f"{ch}={q[i]:.6g}" for i, ch in enumerate(CHANNELS)))

    rows: List[Dict] = []
    for f in test_files:
        rows.extend(
            write_intervals_for_file(
                f,
                args.out,
                qhats,
                args.modes,
                eps=args.eps,
                compressed=not args.uncompressed,
                include_mesh=not args.no_mesh,
            )
        )

    summary = {
        "test_dir": str(args.test_dir),
        "n_test_cases": len(test_files),
        "modes": args.modes,
        "qhat": {
            mode: {CHANNELS[i]: float(qhats[mode][i]) for i in range(4)}
            for mode in args.modes
        },
        "interval_definition": {
            "global_abs": "lower/upper = pred +/- q_abs",
            "point_sigma": "lower/upper = pred +/- q_sigma * sigma",
            "case_sigma": "lower/upper = pred +/- q_case * sigma",
        },
        "per_case_metrics": rows,
    }
    with open(args.out / "interval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Interval files written under: {args.out}")
    print(f"[OK] Summary: {args.out / 'interval_summary.json'}")


if __name__ == "__main__":
    main()
