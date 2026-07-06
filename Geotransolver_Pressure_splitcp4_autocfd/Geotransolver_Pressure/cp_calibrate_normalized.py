#!/usr/bin/env python3
"""Normalized conformal prediction for surface pressure/WSS fields.

Expected input files are produced by inference_on_zarr.py when

    cp_output.save_pointwise_npz: true

Each .npz should contain at least (physical units when saved from inference_on_zarr.py):
    pred   : (N, 4) or (1, N, 4)  — channels: pressure, wss_x, wss_y, wss_z
    target : (N, 4) or (1, N, 4)
    sigma  : (N, 4) or (1, N, 4)  — sigma_phys = sigma_norm * surface_std * rho*U^2

The conformal score is

    s = |target - pred| / (sigma + eps)

and the interval is

    pred ± q_hat * sigma

where q_hat is the finite-sample conformal quantile computed channel-wise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ["pressure", "wss_x", "wss_y", "wss_z"]


def _as_points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected (N,C) or (1,N,C), got {x.shape}")
    return x


def load_npz_dir(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {path}")

    preds, targets, sigmas = [], [], []
    for f in files:
        data = np.load(f)
        for key in ("pred", "target", "sigma"):
            if key not in data:
                raise KeyError(f"{f} has no key '{key}'.")
        preds.append(_as_points(data["pred"]))
        targets.append(_as_points(data["target"]))
        sigmas.append(_as_points(data["sigma"]))

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(sigmas, axis=0)


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample split conformal quantile, channel-wise."""
    scores = np.asarray(scores)
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    sorted_scores = np.sort(scores, axis=0)
    return sorted_scores[k - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--out", type=Path, default=Path("normalized_cp_results"))
    ap.add_argument("--save-intervals", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pred_c, true_c, sigma_c = load_npz_dir(args.calib_dir)
    pred_t, true_t, sigma_t = load_npz_dir(args.test_dir)

    sigma_c = np.maximum(sigma_c, args.eps)
    sigma_t = np.maximum(sigma_t, args.eps)

    scores = np.abs(true_c - pred_c) / sigma_c
    q_hat = conformal_quantile(scores, args.alpha)

    lower = pred_t - q_hat.reshape(1, -1) * sigma_t
    upper = pred_t + q_hat.reshape(1, -1) * sigma_t

    covered = (true_t >= lower) & (true_t <= upper)
    width = upper - lower

    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_calibration_points": int(pred_c.shape[0]),
        "n_test_points": int(pred_t.shape[0]),
        "q_hat": {c: float(v) for c, v in zip(CHANNELS, q_hat)},
        "coverage": {c: float(v) for c, v in zip(CHANNELS, covered.mean(axis=0))},
        "mean_interval_width": {c: float(v) for c, v in zip(CHANNELS, width.mean(axis=0))},
        "median_interval_width": {c: float(v) for c, v in zip(CHANNELS, np.median(width, axis=0))},
    }

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_intervals:
        np.savez_compressed(
            args.out / "test_intervals.npz",
            pred=pred_t,
            target=true_t,
            sigma=sigma_t,
            lower=lower,
            upper=upper,
            q_hat=q_hat,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
