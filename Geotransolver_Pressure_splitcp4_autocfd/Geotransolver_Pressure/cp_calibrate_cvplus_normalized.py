#!/usr/bin/env python3
"""Merge 4-fold CV+ OOF pointwise npz and compute channel-wise conformal q_hat.

Uses a streaming score pass per channel to avoid loading all pred/target/sigma
arrays into memory at once (important for large surface point-wise CP).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cp_calibrate_normalized import CHANNELS, _as_points


def _collect_npz_files(pattern: str) -> list[Path]:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No paths matched glob: {pattern}")

    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)

    if not files:
        raise FileNotFoundError(f"No .npz files found for glob: {pattern}")
    return files


def compute_qhat_streaming(
    pattern: str,
    alpha: float,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Compute channel-wise q_hat by streaming one channel at a time."""
    files = _collect_npz_files(pattern)

    q_hat: list[float] = []
    n_total = 0

    for ch in range(len(CHANNELS)):
        scores_ch: list[np.ndarray] = []

        for f in files:
            with np.load(f) as data:
                for key in ("pred", "target", "sigma"):
                    if key not in data:
                        raise KeyError(f"{f} has no key '{key}'.")

                pred = _as_points(data["pred"])[:, ch].astype(np.float32, copy=False)
                target = _as_points(data["target"])[:, ch].astype(np.float32, copy=False)
                sigma = _as_points(data["sigma"])[:, ch].astype(np.float32, copy=False)
                sigma = np.maximum(sigma, eps)

                score = np.abs(target - pred) / sigma
                scores_ch.append(score.astype(np.float32, copy=False))

        scores_ch_arr = np.concatenate(scores_ch, axis=0)

        if ch == 0:
            n_total = int(scores_ch_arr.shape[0])

        n = scores_ch_arr.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)

        q = np.partition(scores_ch_arr, k - 1)[k - 1]
        q_hat.append(float(q))

        del scores_ch
        del scores_ch_arr

    return np.array(q_hat, dtype=np.float32), n_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calib-glob",
        type=str,
        default="runs/cvplus/fold_*/cvplus_oof/fold_*",
        help="Glob for per-fold OOF npz directories.",
    )
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--out", type=Path, default=Path("results/pressure_cvplus_cp"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    q_hat, n_calib = compute_qhat_streaming(
        args.calib_glob,
        args.alpha,
        args.eps,
    )

    summary = {
        "protocol": "pressure_4fold_cvplus_oof_normalized_cp",
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "calib_glob": args.calib_glob,
        "n_calibration_points": int(n_calib),
        "q_hat": {c: float(v) for c, v in zip(CHANNELS, q_hat)},
    }

    qhat_path = args.out / "q_hat.json"
    with open(qhat_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
