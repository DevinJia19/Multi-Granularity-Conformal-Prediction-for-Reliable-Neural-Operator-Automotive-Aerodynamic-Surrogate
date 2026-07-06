#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save out-of-fold predictions for CV-assisted Cd CQR calibration."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _collect_predictions(model, dataloader, device):
    """Collect predictions without importing test.py and creating cycles."""
    model.eval()
    quantiles = tuple(float(q) for q in model.quantiles)
    predictions_by_quantile = {q: [] for q in quantiles}
    ground_truth = []
    with torch.no_grad():
        for point_clouds, global_geometry_descriptors, cd_values in dataloader:
            point_clouds = point_clouds.to(device)
            global_geometry_descriptors = global_geometry_descriptors.to(device)
            cd_values = cd_values.to(device)
            cd_pred = model(point_clouds, global_geometry_descriptors)
            cd_pred_np = cd_pred.cpu().numpy()
            for idx, q in enumerate(quantiles):
                predictions_by_quantile[q].extend(cd_pred_np[:, idx].tolist())
            ground_truth.extend(cd_values.cpu().numpy().flatten().tolist())
    predictions_by_quantile = {
        q: np.array(v, dtype=np.float64) for q, v in predictions_by_quantile.items()
    }
    return predictions_by_quantile, np.array(ground_truth, dtype=np.float64)


def save_cvplus_oof_scores(
    best_checkpoint_path: str,
    fold_idx: int,
    config,
    device: torch.device,
    model: torch.nn.Module,
) -> str | None:
    """Reload the best checkpoint and save OOF calibration predictions."""
    cal_csv = (getattr(config, "CALIBRATION_CSV", "") or "").strip()
    if not cal_csv or not os.path.isfile(cal_csv):
        logger.warning("[CV+] Skipping OOF save: invalid CALIBRATION_CSV: %s", cal_csv)
        return None

    base = model.module if hasattr(model, "module") else model
    try:
        ckpt = torch.load(best_checkpoint_path, map_location=device)
    except Exception:
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()

    from train import create_eval_dataloader

    old_distributed = config.DISTRIBUTED
    config.DISTRIBUTED = False
    cal_loader, _ = create_eval_dataloader(config, cal_csv)
    config.DISTRIBUTED = old_distributed
    if cal_loader is None:
        logger.warning("[CV+] Could not build calibration DataLoader: %s", cal_csv)
        return None

    pred_norm, y_norm = _collect_predictions(base, cal_loader, device)
    y_cal = y_norm * config.TARGET_STD + config.TARGET_MEAN
    q05 = pred_norm[0.05] * config.TARGET_STD + config.TARGET_MEAN
    q95 = pred_norm[0.95] * config.TARGET_STD + config.TARGET_MEAN

    oof_dir = os.getenv("CVPLUS_OOF_DIR", "./results/cvplus").strip() or "./results/cvplus"
    os.makedirs(oof_dir, exist_ok=True)
    out_path = os.path.join(oof_dir, "oof_fold_%d.npz" % int(fold_idx))
    np.savez_compressed(
        out_path,
        y_true=y_cal.astype(np.float64),
        q05_pred=q05.astype(np.float64),
        q95_pred=q95.astype(np.float64),
        fold=np.int32(fold_idx),
    )
    logger.info("[CV+] Saved fold %d OOF calibration data: %s (n=%d)", fold_idx, out_path, y_cal.size)
    return out_path
