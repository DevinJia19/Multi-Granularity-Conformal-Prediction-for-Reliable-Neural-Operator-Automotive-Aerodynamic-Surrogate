#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共形 / 分位数区间诊断：预测缓存 + R 次随机校准/验证划分（不重跑模型）。

流程（与「反复划分评估 CP」文献一致）：
  1）对固定 checkpoint，在全数据 CSV 上只做一次推理，缓存每条的 q05,q50,q95,y；
  2）重复 R 次：随机打乱**索引**，前 n_cal 条作校准集算 q_l/q_u，其余作验证集；
  3）在验证集上统计：
       - 非对称 CQR 校准后区间是否覆盖 y，及 P(y<下界)、P(y>上界)；
       - 未校准 PI90：y 是否在 [q05,q95]；raw tail：P(y<q05), P(y>q95)；
       - P(y<=q50) 是否接近 0.5。

用法示例：
  python evaluate_cp_replicates.py \\
    --checkpoint ./checkpoints/final/best_model.pth \\
    --csv ./data_splits/train_pool_90_with_cv_fold.csv \\
    --n-cal 800 \\
    --R 1000 \\
    --cache ./results/cp_eval_cache.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from cqr_common import apply_asymmetric_cqr, asymmetric_cqr_hat_q
from draivernet_dataset import DrivAerNetDataset
from train import CdPredictionModel, Config


def _load_model_and_config(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = Config()
    model_spec = checkpoint.get("model_spec", {})
    for key in [
        "QUANTILES",
        "OUT_DIM",
        "FUNCTIONAL_DIM",
        "GEOTRANS_OUT_DIM",
        "GEOMETRY_DIM",
        "GLOBAL_DIM",
        "BACKBONE_TYPE",
        "BACKBONE_OUT_DIM",
        "N_HIDDEN",
        "N_LAYERS",
        "N_HEAD",
        "DROPOUT",
        "SLICE_NUM",
        "POOLING_TYPE",
        "NUM_POINTS",
        "POINT_SURFACE_FEATURES",
        "POINT_USE_CURVATURE",
        "USE_AREA_WEIGHTED_POOLING",
    ]:
        if key in model_spec:
            setattr(config, key, model_spec[key] if key != "QUANTILES" else tuple(model_spec[key]))

    model = CdPredictionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ckpt_cfg = checkpoint.get("config", {})
    config.TARGET_MEAN = float(ckpt_cfg.get("TARGET_MEAN", 0.0))
    config.TARGET_STD = float(ckpt_cfg.get("TARGET_STD", 1.0))
    config.GLOBAL_DESCRIPTOR_MEAN = torch.tensor(
        ckpt_cfg.get("GLOBAL_DESCRIPTOR_MEAN", [0.0] * 15),
        dtype=torch.float32,
    )
    config.GLOBAL_DESCRIPTOR_STD = torch.tensor(
        ckpt_cfg.get("GLOBAL_DESCRIPTOR_STD", [1.0] * 15),
        dtype=torch.float32,
    )
    model.eval()
    return model, config


def _evaluate_collect(model, config, csv_file: str, device: torch.device):
    """返回物理尺度下的 y, q05, q50, q95。"""
    from test import evaluate_model

    from train import _eval_dataset_kwargs

    ds = DrivAerNetDataset(**_eval_dataset_kwargs(config, csv_file))
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    pred_norm, y_norm = evaluate_model(model, loader, device)
    q05 = pred_norm[0.05] * config.TARGET_STD + config.TARGET_MEAN
    q50 = pred_norm[0.5] * config.TARGET_STD + config.TARGET_MEAN
    q95 = pred_norm[0.95] * config.TARGET_STD + config.TARGET_MEAN
    y = y_norm * config.TARGET_STD + config.TARGET_MEAN
    y = y.reshape(-1)
    return (
        y.astype(np.float64),
        q05.astype(np.float64),
        q50.astype(np.float64),
        q95.astype(np.float64),
    )


def _one_replicate(
    rng: np.random.Generator,
    y: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    n_cal: int,
    alpha: float,
):
    n = y.size
    perm = rng.permutation(n)
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:]

    q_l, q_u = asymmetric_cqr_hat_q(
        q05[cal_idx], q95[cal_idx], y[cal_idx], alpha
    )

    yv = y[val_idx]
    q05v, q50v, q95v = q05[val_idx], q50[val_idx], q95[val_idx]
    lo, hi = apply_asymmetric_cqr(q05v, q95v, q_l, q_u)

    if np.isfinite(q_l) and np.isfinite(q_u):
        cqr_cover = float(np.mean((yv >= lo) & (yv <= hi)))
        cqr_tail_low = float(np.mean(yv < lo))
        cqr_tail_high = float(np.mean(yv > hi))
    else:
        cqr_cover = float("nan")
        cqr_tail_low = float("nan")
        cqr_tail_high = float("nan")

    pi90_raw = float(np.mean((yv >= q05v) & (yv <= q95v)))
    tail_low = float(np.mean(yv < q05v))
    tail_high = float(np.mean(yv > q95v))
    frac_le_q50 = float(np.mean(yv <= q50v))

    return {
        "cqr_coverage": cqr_cover,
        "cqr_tail_below_lower": cqr_tail_low,
        "cqr_tail_above_upper": cqr_tail_high,
        "pi90_uncalibrated": pi90_raw,
        "tail_below_q05": tail_low,
        "tail_above_q95": tail_high,
        "frac_y_le_q50": frac_le_q50,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CP / CQR 随机划分蒙特卡洛评估（预测缓存，仅打乱索引）"
    )
    ap.add_argument("--checkpoint", required=True, help="best_model.pth 路径")
    ap.add_argument(
        "--csv",
        required=True,
        help="用于评估的数据 CSV（如 train_pool_90 或 test_split）",
    )
    ap.add_argument("--n-cal", type=int, required=True, help="每次重复中校准集样本数 n_cal")
    ap.add_argument("--R", type=int, default=1000, dest="R", help="随机划分重复次数")
    ap.add_argument("--alpha", type=float, default=0.1, help="CQR miscoverage（名义 90%% 区间对应 alpha=0.1）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache",
        default="",
        help="缓存 npz 路径；存在则跳过推理直接加载（大幅加速重复试验）",
    )
    ap.add_argument("--device", default="", help="cuda 或 cpu，默认自动")
    ap.add_argument("--out-json", default="", help="汇总结果写入 JSON")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="DataLoader 批大小；0 则使用 Config 默认",
    )
    args = ap.parse_args()

    cache_path = args.cache.strip()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if cache_path and os.path.isfile(cache_path):
        print("[INFO] 加载缓存: %s" % cache_path)
        z = np.load(cache_path)
        y = z["y"]
        q05 = z["q05"]
        q50 = z["q50"]
        q95 = z["q95"]
    else:
        if not os.path.isfile(args.checkpoint):
            print("[ERROR] checkpoint 不存在: %s" % args.checkpoint, file=sys.stderr)
            return 1
        if not os.path.isfile(args.csv):
            print("[ERROR] csv 不存在: %s" % args.csv, file=sys.stderr)
            return 1
        print("[INFO] 单次推理（可写入 --cache 下次秒开）...")
        model, config = _load_model_and_config(args.checkpoint, device)
        config.STL_ROOT_DIR = os.getenv("STL_ROOT_DIR", config.STL_ROOT_DIR)
        if int(args.batch_size or 0) > 0:
            config.BATCH_SIZE = int(args.batch_size)
        y, q05, q50, q95 = _evaluate_collect(model, config, args.csv, device)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(cache_path, y=y, q05=q05, q50=q50, q95=q95)
            print("[INFO] 已写入缓存: %s" % cache_path)

    n = int(y.size)
    n_cal = int(args.n_cal)
    if n_cal <= 0 or n_cal >= n:
        print("[ERROR] 需要 0 < n_cal < N，当前 N=%d, n_cal=%d" % (n, n_cal), file=sys.stderr)
        return 1

    R = int(args.R)
    alpha = float(args.alpha)
    rng = np.random.default_rng(args.seed)

    names = [
        "cqr_coverage",
        "cqr_tail_below_lower",
        "cqr_tail_above_upper",
        "pi90_uncalibrated",
        "tail_below_q05",
        "tail_above_q95",
        "frac_y_le_q50",
    ]
    acc = {k: np.zeros(R, dtype=np.float64) for k in names}

    print("[INFO] R=%d, n_cal=%d, n_val=%d, alpha=%s, seed=%d" % (R, n_cal, n - n_cal, alpha, args.seed))

    for r in range(R):
        out = _one_replicate(rng, y, q05, q50, q95, n_cal, alpha)
        for k in names:
            acc[k][r] = out[k]

    summary = {
        "N": n,
        "n_cal": n_cal,
        "n_val": n - n_cal,
        "R": R,
        "alpha": alpha,
        "nominal": {
            "pi90_coverage_target": 1.0 - alpha,
            "cqr_tail_below_lower_target": alpha / 2.0,
            "cqr_tail_above_upper_target": alpha / 2.0,
            "tail_below_q05_target": alpha / 2.0,
            "tail_above_q95_target": alpha / 2.0,
            "frac_y_le_q50_target": 0.5,
        },
        "mean": {k: float(np.nanmean(acc[k])) for k in names},
        "std": {k: float(np.nanstd(acc[k], ddof=0)) for k in names},
    }

    print("\n=== 随机划分蒙特卡洛均值 ± 标准差（验证集上）===")
    print(
        "名义目标: CQR 覆盖率≈%.4f | CQR P(y<下界)≈P(y>上界)≈%.4f | "
        "raw P(y<q05)≈P(y>q95)≈%.4f | P(y<=q50)≈0.5"
        % (1.0 - alpha, alpha / 2.0, alpha / 2.0)
    )
    for k in names:
        print("  %-22s  %.6f ± %.6f" % (k, summary["mean"][k], summary["std"][k]))

    out_json = args.out_json.strip()
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\n[INFO] 汇总已写入 %s" % out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
