#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate scalar Cd quantile predictions and calibrated CQR intervals."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cqr_common import apply_asymmetric_cqr, asymmetric_cqr_hat_q
from train import CdPredictionModel, Config, create_eval_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestConfig(Config):
    """Runtime defaults for scalar Cd evaluation."""

    TEST_CSV = os.getenv("TEST_CSV", "./data_splits/test_split.csv").strip()
    CALIBRATION_CSV = os.getenv("CALIBRATION_CSV", "./data_splits/calibration_split.csv").strip()
    CQR_HAT_Q_JSON = os.getenv("CQR_HAT_Q_JSON", "").strip()
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoints/best_model.pth").strip()
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results").strip() or "./results"
    CQR_ALPHA = float(os.getenv("CQR_ALPHA", "0.1"))


def _load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _restore_config_from_checkpoint(config: Config, checkpoint: dict) -> None:
    for key, value in checkpoint.get("model_spec", {}).items():
        setattr(config, key, tuple(value) if key == "QUANTILES" else value)
    for key, value in checkpoint.get("config", {}).items():
        if key in {
            "TARGET_MEAN",
            "TARGET_STD",
            "GLOBAL_DESCRIPTOR_MEAN",
            "GLOBAL_DESCRIPTOR_STD",
            "NORMALIZE_TARGET",
            "NORMALIZE_GLOBAL_DESCRIPTOR",
        }:
            setattr(config, key, value)


def _denormalize(config: Config, values: np.ndarray) -> np.ndarray:
    if getattr(config, "NORMALIZE_TARGET", True):
        return values * float(config.TARGET_STD) + float(config.TARGET_MEAN)
    return values


def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device):
    """Collect quantile predictions and targets from a dataloader."""
    model.eval()
    quantiles = tuple(float(q) for q in model.quantiles)
    predictions = {q: [] for q in quantiles}
    targets: list[float] = []
    with torch.no_grad():
        for point_clouds, global_geometry_descriptors, cd_values in dataloader:
            pred = model(
                point_clouds.to(device),
                global_geometry_descriptors.to(device),
            ).detach().cpu().numpy()
            for idx, q in enumerate(quantiles):
                predictions[q].extend(pred[:, idx].tolist())
            targets.extend(cd_values.detach().cpu().numpy().reshape(-1).tolist())
    return {q: np.asarray(v, dtype=np.float64) for q, v in predictions.items()}, np.asarray(targets, dtype=np.float64)


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(score))


def calculate_metrics(predictions: dict[float, np.ndarray], y: np.ndarray) -> dict[str, float]:
    q05, q50, q95 = predictions[0.05], predictions[0.5], predictions[0.95]
    error = q50 - y
    mse = float(np.mean(error**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(np.mean(np.abs(error) / np.maximum(np.abs(y), 1.0e-12)) * 100.0),
        "R2_Score": float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "Correlation": float(np.corrcoef(q50, y)[0, 1]) if y.size > 1 else float("nan"),
        "Raw_Coverage_90": float(np.mean((y >= q05) & (y <= q95)) * 100.0),
        "Raw_PI90_Width": float(np.mean(q95 - q05)),
        "Raw_PI90_IntervalScore": interval_score(q05, q95, y, alpha=0.1),
    }


def _load_cqr_hat_q(path: str) -> tuple[float, float, int | None, str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["q_l"]), float(payload["q_u"]), payload.get("n_cal"), payload.get("protocol", "precomputed_cqr")


def _calibrate_cqr(model, config: TestConfig, device: torch.device) -> tuple[float, float, int, str]:
    if config.CQR_HAT_Q_JSON:
        q_l, q_u, n_cal, protocol = _load_cqr_hat_q(config.CQR_HAT_Q_JSON)
        return q_l, q_u, int(n_cal or -1), protocol
    cal_loader, _ = create_eval_dataloader(config, config.CALIBRATION_CSV)
    if cal_loader is None:
        raise FileNotFoundError("Calibration CSV is missing. Set CALIBRATION_CSV or CQR_HAT_Q_JSON.")
    pred_norm, y_norm = evaluate_model(model, cal_loader, device)
    q_l, q_u = asymmetric_cqr_hat_q(
        _denormalize(config, pred_norm[0.05]),
        _denormalize(config, pred_norm[0.95]),
        _denormalize(config, y_norm),
        alpha=config.CQR_ALPHA,
    )
    return q_l, q_u, int(y_norm.size), "single_calibration_split"


def _read_design_ids(csv_path: str, design_column: str, n: int) -> np.ndarray:
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if design_column in df.columns and len(df) == n:
            return df[design_column].to_numpy()
    return np.arange(n)


def main() -> None:
    config = TestConfig()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    device = torch.device(config.DEVICE)

    checkpoint = _load_checkpoint(config.CHECKPOINT_PATH, device)
    _restore_config_from_checkpoint(config, checkpoint)
    model = CdPredictionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    q_l, q_u, n_cal, protocol = _calibrate_cqr(model, config, device)
    test_loader, _ = create_eval_dataloader(config, config.TEST_CSV)
    if test_loader is None:
        raise FileNotFoundError(f"Test CSV is missing: {config.TEST_CSV}")

    pred_norm, y_norm = evaluate_model(model, test_loader, device)
    predictions = {q: _denormalize(config, arr) for q, arr in pred_norm.items()}
    y = _denormalize(config, y_norm)
    q05, q50, q95 = predictions[0.05], predictions[0.5], predictions[0.95]
    cqr_lower, cqr_upper = apply_asymmetric_cqr(q05, q95, q_l, q_u)

    metrics = calculate_metrics(predictions, y)
    metrics.update(
        {
            "CQR_Coverage_90": float(np.mean((y >= cqr_lower) & (y <= cqr_upper)) * 100.0),
            "CQR_PI90_Width": float(np.mean(cqr_upper - cqr_lower)),
            "CQR_PI90_IntervalScore": interval_score(cqr_lower, cqr_upper, y, config.CQR_ALPHA),
            "CQR_q_l": float(q_l),
            "CQR_q_u": float(q_u),
            "CQR_n_cal": int(n_cal),
            "CQR_protocol": protocol,
        }
    )

    out_df = pd.DataFrame(
        {
            "Design_ID": _read_design_ids(config.TEST_CSV, config.DESIGN_COLUMN, y.size),
            "Predicted_Cd_Q05": q05,
            "Predicted_Cd_Q50": q50,
            "Predicted_Cd_Q95": q95,
            "Ground_Truth_Cd": y,
            "Absolute_Error_Q50": np.abs(q50 - y),
            "CQR_Lower": cqr_lower,
            "CQR_Upper": cqr_upper,
            "CQR_Covered": (y >= cqr_lower) & (y <= cqr_upper),
        }
    )

    results_dir = Path(config.RESULTS_DIR)
    out_df.to_csv(results_dir / "predictions.csv", index=False, encoding="utf-8")
    out_df.to_csv(results_dir / "per_sample_cd_cp_intervals.csv", index=False, encoding="utf-8")
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Scalar Cd Evaluation Results\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    logger.info("Raw coverage: %.2f%%", metrics["Raw_Coverage_90"])
    logger.info("CQR coverage: %.2f%%", metrics["CQR_Coverage_90"])


if __name__ == "__main__":
    main()
