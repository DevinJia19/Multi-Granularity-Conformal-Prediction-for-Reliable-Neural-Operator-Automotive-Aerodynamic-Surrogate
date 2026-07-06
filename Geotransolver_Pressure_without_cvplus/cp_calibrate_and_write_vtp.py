#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

CHANNELS = ["pressure", "wss_x", "wss_y", "wss_z"]


def as_points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected shape (N, C) or (1, N, C), got {x.shape}")
    return x


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    scores = np.asarray(scores)
    n = scores.shape[0]
    if n < 1:
        raise ValueError("No calibration scores found.")

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)

    sorted_scores = np.sort(scores, axis=0)
    return sorted_scores[k - 1]


def load_calibration_scores(calib_dir: Path, eps: float) -> tuple[np.ndarray, int]:
    files = sorted(calib_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {calib_dir}")

    all_scores = []
    total_points = 0

    for f in files:
        data = np.load(f)

        pred = as_points(data["pred"])
        target = as_points(data["target"])
        sigma = np.maximum(as_points(data["sigma"]), eps)

        score = np.abs(target - pred) / sigma

        all_scores.append(score)
        total_points += pred.shape[0]

    return np.concatenate(all_scores, axis=0), total_points


def write_case_vtp(
    npz_path: Path,
    out_path: Path,
    q_hat: np.ndarray,
    eps: float,
) -> dict:
    data = np.load(npz_path)

    pred = as_points(data["pred"])
    target = as_points(data["target"])
    sigma = np.maximum(as_points(data["sigma"]), eps)

    lower = pred - q_hat.reshape(1, -1) * sigma
    upper = pred + q_hat.reshape(1, -1) * sigma
    width = upper - lower
    covered = (target >= lower) & (target <= upper)

    if "surface_mesh_centers" not in data:
        raise KeyError(
            f"{npz_path} does not contain surface_mesh_centers. "
            "Please save surface_mesh_centers in inference_on_zarr.py."
        )

    points = as_points(data["surface_mesh_centers"])[:, :3]

    mesh = pv.PolyData(points)

    # Pressure channel
    mesh.point_data["PredictedPressure"] = pred[:, 0]
    mesh.point_data["TruePressure"] = target[:, 0]
    mesh.point_data["SigmaPressure"] = sigma[:, 0]
    mesh.point_data["CPPressureLower"] = lower[:, 0]
    mesh.point_data["CPPressureUpper"] = upper[:, 0]
    mesh.point_data["CPPressureWidth"] = width[:, 0]
    mesh.point_data["PressureCovered"] = covered[:, 0].astype(np.float32)

    # WSS channels, optional but useful
    mesh.point_data["PredictedWSS"] = pred[:, 1:4]
    mesh.point_data["TrueWSS"] = target[:, 1:4]
    mesh.point_data["SigmaWSS"] = sigma[:, 1:4]
    mesh.point_data["CPWSSLower"] = lower[:, 1:4]
    mesh.point_data["CPWSSUpper"] = upper[:, 1:4]
    mesh.point_data["CPWSSWidth"] = width[:, 1:4]

    if "surface_normals" in data:
        mesh.point_data["SurfaceNormals"] = as_points(data["surface_normals"])[:, :3]

    if "surface_areas" in data:
        area = np.asarray(data["surface_areas"])
        if area.ndim == 3 and area.shape[0] == 1:
            area = area[0]
        area = area.reshape(-1)
        if area.shape[0] == points.shape[0]:
            mesh.point_data["SurfaceArea"] = area

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(out_path)

    return {
        "file": npz_path.name,
        "n_points": int(pred.shape[0]),
        "coverage": {
            name: float(covered[:, i].mean())
            for i, name in enumerate(CHANNELS)
        },
        "mean_width": {
            name: float(width[:, i].mean())
            for i, name in enumerate(CHANNELS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=Path("cp_vtp_results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    scores, n_calib_points = load_calibration_scores(args.calib_dir, args.eps)
    q_hat = conformal_quantile(scores, args.alpha)

    test_files = sorted(args.test_dir.glob("*.npz"))
    if not test_files:
        raise FileNotFoundError(f"No .npz files found in {args.test_dir}")

    per_case = []

    for f in test_files:
        out_vtp = args.out / "vtp" / f"{f.stem}_cp_pressure.vtp"
        info = write_case_vtp(f, out_vtp, q_hat, args.eps)
        per_case.append(info)

    summary = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_calibration_points": int(n_calib_points),
        "n_test_cases": len(test_files),
        "q_hat": {
            name: float(q_hat[i])
            for i, name in enumerate(CHANNELS)
        },
        "mean_case_coverage": {
            name: float(np.mean([case["coverage"][name] for case in per_case]))
            for name in CHANNELS
        },
        "mean_case_interval_width": {
            name: float(np.mean([case["mean_width"][name] for case in per_case]))
            for name in CHANNELS
        },
        "per_case": per_case,
    }

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] Wrote VTP files to: {args.out / 'vtp'}")


if __name__ == "__main__":
    main()
