#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cd prediction training script
使用DrivAerNet数据集训练模型以预测Cd值
"""

import os
import json
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import trimesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from physicsnemo.models.transolver import Transolver
except Exception:  # Some PhysicsNeMo versions expose it from the module path below.
    from physicsnemo.models.transolver.transolver import Transolver

from draivernet_dataset import DrivAerNetDataset  # pyright: ignore[reportMissingImports]

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """是否由 torchrun 启动为分布式模式。"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_main_process() -> bool:
    """仅主进程负责日志和写文件。"""
    return (not is_distributed()) or int(os.environ["RANK"]) == 0


def setup_distributed():
    """初始化分布式环境并返回 rank/local_rank/world_size。"""
    if not is_distributed():
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def cleanup_distributed():
    """清理分布式进程组。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _dataset_kwargs(config) -> dict:
    return {
        "root_dir": config.STL_ROOT_DIR,
        "num_points": config.NUM_POINTS,
        "transform": None,
        "design_column": config.DESIGN_COLUMN,
        "target_column": config.TARGET_COLUMN,
        "file_suffix": config.FILE_SUFFIX,
        "normalize_target": config.NORMALIZE_TARGET,
        "target_mean": config.TARGET_MEAN,
        "target_std": config.TARGET_STD,
        "global_descriptor_mean": config.GLOBAL_DESCRIPTOR_MEAN,
        "global_descriptor_std": config.GLOBAL_DESCRIPTOR_STD,
        "enable_point_cache": config.ENABLE_POINT_CACHE,
        "point_cache_dir": config.POINT_CACHE_DIR,
        "point_cache_version": config.POINT_CACHE_VERSION,
        "enable_mesh_cache": config.ENABLE_MESH_CACHE,
        "mesh_cache_dir": config.MESH_CACHE_DIR,
        "mesh_cache_version": config.MESH_CACHE_VERSION,
        "surface_features": getattr(config, "POINT_SURFACE_FEATURES", config.FUNCTIONAL_DIM > 3),
        "use_curvature": getattr(config, "POINT_USE_CURVATURE", False),
    }


def _eval_dataset_kwargs(config, csv_file: str) -> dict:
    """评估/测试用 Dataset 参数（强制 normalize_target=True，避免与 _dataset_kwargs 冲突）。"""
    ds_kwargs = _dataset_kwargs(config)
    ds_kwargs["normalize_target"] = True
    return {
        **ds_kwargs,
        "csv_file": csv_file,
        "apply_augmentations": False,
        "normalize": True,
        "deterministic_sampling": True,
        "deterministic_seed_base": config.SEED,
    }


def _build_eval_dataset(config, csv_file: str):
    return DrivAerNetDataset(**_eval_dataset_kwargs(config, csv_file))


def create_eval_dataloader(config, csv_file: str):
    """非训练集 DataLoader（无增强）；用于 OOF/校准推理等，不作训练期验证集。

    文件不存在则返回 (None, None)。
    """
    if not csv_file or not os.path.isfile(csv_file):
        return None, None
    if is_main_process():
        logger.info(f"评估数据 CSV: {csv_file}")
    dataset = _build_eval_dataset(config, csv_file)
    eval_sampler = None
    shuffle = False
    if config.DISTRIBUTED:
        eval_sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        sampler=eval_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
        persistent_workers=(config.NUM_WORKERS > 0 and config.DATALOADER_PERSISTENT_WORKERS),
        prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
    )
    if is_main_process():
        logger.info(f"  -> {len(dataset)} 样本")
    return loader, eval_sampler


def save_loss_curves_figure(
    training_history: dict,
    out_path: str,
) -> None:
    """将训练 loss 存为折线图（仅主进程调用；不再绘制验证/测试集曲线）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    epochs = training_history["epoch"]
    train_loss = training_history["train_loss"]
    val_loss = training_history.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    ax.plot(epochs, train_loss, label="训练损失", color="#1f77b4", linewidth=1.8)
    if val_loss:
        ax.plot(epochs, val_loss, label="验证损失", color="#ff7f0e", linewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball loss")
    title = "训练/验证损失曲线" if val_loss else "训练损失曲线"
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"损失曲线已保存: {out_path}")


# ==================== 配置参数 ====================
class Config:
    # 数据集路径
    STL_ROOT_DIR = os.getenv(
        "STL_ROOT_DIR",
        "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/stl",
    )
    TRAIN_CSV = os.getenv("TRAIN_CSV", "./data_splits/train_split.csv")
    # 单样本过拟合冒烟测试：OVERFIT_SINGLE=1（固定 1 条样本，关闭增强/验证，便于检查接口）
    OVERFIT_SINGLE = os.getenv("OVERFIT_SINGLE", "0") == "1"
    OVERFIT_MODE = OVERFIT_SINGLE or os.getenv("OVERFIT_MODE", "0") == "1"
    VAL_CSV = os.getenv(
        "VAL_CSV",
        ""
        if OVERFIT_SINGLE or OVERFIT_MODE
        else "./data_splits/validation_split.csv",
    ).strip()
    VALIDATE_EVERY = int(os.getenv("VALIDATE_EVERY", "1"))
    EARLY_STOPPING_PATIENCE = int(
        os.getenv("EARLY_STOPPING_PATIENCE", "0" if OVERFIT_MODE else "80")
    )
    # 仅作环境/元数据保留；test.py 会读 TEST_CSV。
    TEST_CSV = os.getenv("TEST_CSV", "./data_splits/test_split.csv").strip()
    DESIGN_COLUMN = "Design"
    TARGET_COLUMN = "Average Cd"
    FILE_SUFFIX = ".stl"

    OVERFIT_SAMPLE_INDEX = int(os.getenv("OVERFIT_SAMPLE_INDEX", "0"))

    # 模型参数
    NUM_POINTS = int(
        os.getenv("NUM_POINTS", "2048" if (OVERFIT_SINGLE or OVERFIT_MODE) else "8192")
    )  # 点云采样数
    POINT_SURFACE_FEATURES = os.getenv(
        "POINT_SURFACE_FEATURES", "0" if OVERFIT_MODE else "1"
    ) == "1"
    POINT_USE_CURVATURE = os.getenv("POINT_USE_CURVATURE", "0") == "1"
    USE_AREA_WEIGHTED_POOLING = os.getenv("USE_AREA_WEIGHTED_POOLING", "0") == "1"
    FUNCTIONAL_DIM = DrivAerNetDataset.point_feature_dim(
        surface_features=POINT_SURFACE_FEATURES,
        use_curvature=POINT_USE_CURVATURE,
    )
    QUANTILES = (0.05, 0.5, 0.95)  # 90% prediction interval + median
    OUT_DIM = len(QUANTILES)  # 输出维度（q05, q50, q95）

    BACKBONE_TYPE = os.getenv("BACKBONE_TYPE", "transolver").strip().lower()
    BACKBONE_OUT_DIM = int(os.getenv("BACKBONE_OUT_DIM", os.getenv("GEOTRANS_OUT_DIM", "96")))
    GEOTRANS_OUT_DIM = BACKBONE_OUT_DIM  # backward-compatible env name for the pooling head
    GEOMETRY_DIM = 3  # 坐标嵌入维度（Transolver embedding 仍用 xyz）
    POOLING_TYPE = "structured"  # 结构化读出: global stats + geometry descriptors + region pooling
    NUM_LATENTS = 8
    LATENT_HEADS = 4

    N_HIDDEN = 192  # 隐藏层维度（须能被 N_HEAD 整除）
    N_LAYERS = 4  # 层数（baseline）
    N_HEAD = 4  # 注意力头数（baseline）
    DROPOUT = float(os.getenv("DROPOUT", "0.0" if OVERFIT_MODE else "0.05"))
    SLICE_NUM = 32

    # 训练参数
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    NUM_EPOCHS = int(
        os.getenv("NUM_EPOCHS", "500" if OVERFIT_SINGLE else ("1000" if OVERFIT_MODE else "600"))
    )
    LEARNING_RATE = float(
        os.getenv("LEARNING_RATE", "1e-4")
    )
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0" if OVERFIT_MODE else "1e-4"))
    WARMUP_EPOCHS = 5
    STD_REG_WEIGHT = float(os.getenv("STD_REG_WEIGHT", "0.0"))
    Q50_LOSS_WEIGHT = float(
        os.getenv("Q50_LOSS_WEIGHT", "0.0" if OVERFIT_MODE else "0.5")
    )
    USE_COSINE_SCHEDULER = os.getenv(
        "USE_COSINE_SCHEDULER", "0" if OVERFIT_MODE else "1"
    ) == "1"
    NORMALIZE_TARGET = os.getenv(
        "NORMALIZE_TARGET", "0" if OVERFIT_SINGLE else "1"
    ) == "1"
    # 全局几何描述符统计（dataset-level normalization）
    # 说明：首次运行会遍历训练集STL计算 mean/std，可能较慢；后续会读取缓存加速。
    GLOBAL_DESC_NORM = os.getenv(
        "GLOBAL_DESC_NORM", "0" if OVERFIT_SINGLE else "1"
    ) == "1"
    GLOBAL_DESC_STATS_MAX_SAMPLES = 0  # >0 时只随机抽样这么多个STL来估计 mean/std（加速）

    # 其他
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0" if OVERFIT_SINGLE else "2"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    DISTRIBUTED = is_distributed()
    OVERFIT_SUBSET_SIZE = int(
        os.getenv("OVERFIT_SUBSET_SIZE", "1" if OVERFIT_SINGLE else ("8" if OVERFIT_MODE else "0"))
    )  # 0 表示使用完整训练集
    APPLY_AUGMENTATIONS = os.getenv("APPLY_AUGMENTATIONS", "0" if OVERFIT_MODE else "1") == "1"
    DETERMINISTIC_SAMPLING = os.getenv("DETERMINISTIC_SAMPLING", "1" if OVERFIT_MODE else "0") == "1"
    LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "1" if OVERFIT_MODE else "20"))
    LOG_DEBUG_FEATURES = False  # 每个batch输出debug特征会显著拖慢训练
    USE_AMP = os.getenv("USE_AMP", "0" if OVERFIT_MODE else "1") == "1"
    DATALOADER_PERSISTENT_WORKERS = os.getenv(
        "PERSISTENT_WORKERS", "0" if OVERFIT_SINGLE else "1"
    ) == "1"
    DATALOADER_PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "4"))
    ENABLE_POINT_CACHE = os.getenv("ENABLE_POINT_CACHE", "1") == "1"
    POINT_CACHE_DIR = os.getenv("POINT_CACHE_DIR", "./cache/pointclouds")
    POINT_CACHE_VERSION = os.getenv(
        "POINT_CACHE_VERSION", "v1" if OVERFIT_MODE else "v2_surface"
    )
    ENABLE_MESH_CACHE = os.getenv("ENABLE_MESH_CACHE", "1") == "1"
    MESH_CACHE_DIR = os.getenv("MESH_CACHE_DIR", "./cache/meshes")
    MESH_CACHE_VERSION = os.getenv(
        "MESH_CACHE_VERSION", "v1" if OVERFIT_MODE else "v2_faces"
    )

    # 输出目录（CV+ 五折时请按 fold 设置 CHECKPOINT_DIR / LOG_DIR，避免覆盖）
    CHECKPOINT_DIR = os.getenv(
        "CHECKPOINT_DIR",
        "./checkpoints/overfit_single"
        if OVERFIT_SINGLE
        else ("./checkpoints/overfit_8" if OVERFIT_MODE else "./checkpoints"),
    ).strip() or "./checkpoints"
    LOG_DIR = os.getenv(
        "LOG_DIR",
        "./logs/overfit_single"
        if OVERFIT_SINGLE
        else ("./logs/overfit_8" if OVERFIT_MODE else "./logs"),
    ).strip() or "./logs"
    LOSS_CURVE_FILE = os.getenv("LOSS_CURVE_FILE", "").strip()  # 默认写入 LOG_DIR/loss_curves.png
    # 与 split_dataset 生成的校准折 CSV 一致；CV+ 每折训练后在该集合上算 OOF scores
    CALIBRATION_CSV = os.getenv("CALIBRATION_CSV", "./data_splits/calibration_split.csv").strip()
    _gd_env = os.getenv("GLOBAL_DESC_STATS_CACHE", "").strip()
    GLOBAL_DESC_STATS_CACHE = _gd_env or os.path.join(CHECKPOINT_DIR, "global_descriptor_stats.json")



class CdPredictionModel(nn.Module):
    """
    Cd值预测模型：PhysicsNeMo Transolver backbone + 结构化池化 + 分位数回归头。

    点坐标同时作为 functional input 与 unstructured positional embedding：
        output = Transolver(point_features, embedding=geometry)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantiles = tuple(float(q) for q in config.QUANTILES)
        if self.quantiles != (0.05, 0.5, 0.95):
            raise ValueError(
                f"当前实现使用 softplus 区间参数化，仅支持 QUANTILES=(0.05, 0.5, 0.95)，收到: {self.quantiles}"
            )
        self.region_names = ("front", "mid", "rear", "underbody")
        self.backbone_type = getattr(config, "BACKBONE_TYPE", "transolver").strip().lower()
        if self.backbone_type not in ("transolver", "standard_transolver", "ordinary_transolver"):
            raise ValueError(
                f"Unsupported BACKBONE_TYPE={self.backbone_type!r}. "
                "This package only supports PhysicsNeMo Transolver."
            )

        self.backbone = Transolver(
            functional_dim=config.FUNCTIONAL_DIM,
            embedding_dim=config.GEOMETRY_DIM,
            out_dim=config.BACKBONE_OUT_DIM,
            structured_shape=None,
            unified_pos=False,
            n_hidden=config.N_HIDDEN,
            n_layers=config.N_LAYERS,
            n_head=config.N_HEAD,
            dropout=config.DROPOUT,
            slice_num=config.SLICE_NUM,
            use_te=False,
            plus=True,
        )

        self.global_stats_dim = config.BACKBONE_OUT_DIM * 3
        self.region_stats_dim = config.BACKBONE_OUT_DIM * 3 * len(self.region_names)
        self.geometry_descriptor_dim = 15  # global geometry descriptors from full mesh
        pooled_dim = self.global_stats_dim + self.region_stats_dim + self.geometry_descriptor_dim

        # 结构化读出后的MLP回归头（256-128-64 + 输出层）
        self.regression_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.OUT_DIM),
        )
        self.softplus = nn.Softplus()
        self._init_quantile_head_for_stable_intervals()

    def _init_quantile_head_for_stable_intervals(self):
        """Keep initial q05/q95 interval narrow for overfit/debug stability."""
        last_layer = self.regression_head[-1]
        if not isinstance(last_layer, nn.Linear):
            return
        with torch.no_grad():
            nn.init.zeros_(last_layer.weight)
            nn.init.constant_(last_layer.bias[0], 0.0)   # q50
            nn.init.constant_(last_layer.bias[1], -2.0)  # lower width => softplus(-2)=0.127
            nn.init.constant_(last_layer.bias[2], -2.0)  # upper width

    def _stats_pool(self, features, area_weights=None):
        if area_weights is not None:
            w = area_weights / area_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mean_feature = (features * w).sum(dim=1)
            centered = features - mean_feature.unsqueeze(1)
            var_feature = (centered.pow(2) * w).sum(dim=1).clamp_min(1e-12)
            std_feature = torch.sqrt(var_feature)
            max_feature = features.max(dim=1).values
            return torch.cat([mean_feature, max_feature, std_feature], dim=-1)

        mean_feature = features.mean(dim=1)
        max_feature = features.max(dim=1).values
        std_feature = features.std(dim=1, unbiased=False)
        return torch.cat([mean_feature, max_feature, std_feature], dim=-1)

    def _masked_stats_pool(self, features, mask, area_weights=None):
        mask = mask.unsqueeze(-1).to(dtype=features.dtype)
        if area_weights is not None:
            w = mask * area_weights
        else:
            w = mask
        count = w.sum(dim=1).clamp_min(1.0)
        mean_feature = (features * w).sum(dim=1) / count

        masked_for_max = features.masked_fill(mask == 0, float("-inf"))
        max_feature = masked_for_max.max(dim=1).values
        max_feature = torch.where(torch.isfinite(max_feature), max_feature, mean_feature)

        centered = (features - mean_feature.unsqueeze(1)) * w
        var_feature = centered.pow(2).sum(dim=1) / count
        std_feature = torch.sqrt(var_feature.clamp_min(1e-12))
        return torch.cat([mean_feature, max_feature, std_feature], dim=-1)

    def _region_pool(self, features, geometry, area_weights=None):
        x = geometry[..., 0]
        z = geometry[..., 2]
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x_norm = (x - x_min) / (x_max - x_min).clamp_min(1e-6)

        z_min = z.min(dim=1, keepdim=True).values
        z_max = z.max(dim=1, keepdim=True).values
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

        region_masks = {
            "front": x_norm >= (2.0 / 3.0),
            "mid": (x_norm >= (1.0 / 3.0)) & (x_norm < (2.0 / 3.0)),
            "rear": x_norm < (1.0 / 3.0),
            "underbody": z_norm < 0.2,
        }

        region_features = [
            self._masked_stats_pool(features, region_masks[name], area_weights=area_weights)
            for name in self.region_names
        ]
        region_counts = torch.stack(
            [region_masks[name].float().mean(dim=1) for name in self.region_names], dim=-1
        )
        return torch.cat(region_features, dim=-1), region_counts

    def forward(self, point_features, global_geometry_descriptors=None):
        """
        前向传播
        Args:
            point_features: (B, N, C) 点特征；至少含 [x,y,z]，可含法向/面积/曲率
        Returns:
            cd_pred: (B, 3) 形状的Cd分位数预测 [q05, q50, q95]
        """
        geometry = point_features[..., :3]
        area_weights = None
        if (
            getattr(self.config, "USE_AREA_WEIGHTED_POOLING", False)
            and point_features.shape[-1] >= 7
        ):
            area_weights = point_features[..., 6:7]
        output = self.backbone(point_features, embedding=geometry)
        # 输出形状: (B, N, BACKBONE_OUT_DIM)

        global_stats = self._stats_pool(output, area_weights=area_weights)
        region_feature, region_counts = self._region_pool(output, geometry, area_weights=area_weights)
        if global_geometry_descriptors is None:
            global_geometry_descriptors = torch.zeros(
                geometry.size(0),
                self.geometry_descriptor_dim,
                device=geometry.device,
                dtype=geometry.dtype,
            )
        geometry_feature = global_geometry_descriptors
        global_feature = torch.cat([global_stats, region_feature, geometry_feature], dim=-1)

        raw_pred = self.regression_head(global_feature)  # (B, 3)
        q50 = raw_pred[:, 0:1]
        lower_width = self.softplus(raw_pred[:, 1:2])
        upper_width = self.softplus(raw_pred[:, 2:3])
        q05 = q50 - lower_width
        q95 = q50 + upper_width
        cd_pred = torch.cat([q05, q50, q95], dim=-1)
        self.debug_stats = {
            "backbone_type": self.backbone_type,
            "backbone_output_std_all": output.std(unbiased=False).detach().item(),
            "backbone_output_batch_std": output.mean(dim=1).std(dim=0, unbiased=False).mean().detach().item(),
            "global_feature_std": global_feature.std(dim=0, unbiased=False).mean().detach().item(),
            "global_feature_mean_abs": global_feature.abs().mean().detach().item(),
            "global_stats_std": global_stats.std(dim=0, unbiased=False).mean().detach().item(),
            "region_feature_std": region_feature.std(dim=0, unbiased=False).mean().detach().item(),
            "region_occupancy_mean": region_counts.mean(dim=0).detach().cpu().tolist(),
            "geometry_feature_mean_abs": geometry_feature.abs().mean().detach().item(),
        }
        return cd_pred


class QuantilePinballLoss(nn.Module):
    """Pinball loss for multi-quantile regression."""

    def __init__(self, quantiles: tuple[float, ...]):
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles cannot be empty")
        self.quantiles = tuple(float(q) for q in quantiles)
        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32)
        self.register_buffer("q_tensor", q_tensor, persistent=False)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (B, Q), target: (B, 1)
        if preds.ndim != 2:
            raise ValueError(f"preds must be 2D (B, Q), got {tuple(preds.shape)}")
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError(f"target must be (B, 1), got {tuple(target.shape)}")
        if preds.shape[1] != len(self.quantiles):
            raise ValueError(
                f"preds second dim {preds.shape[1]} != number of quantiles {len(self.quantiles)}"
            )

        errors = target - preds  # (B, Q)
        q = self.q_tensor.to(device=preds.device, dtype=preds.dtype).view(1, -1)
        loss = torch.maximum(q * errors, (q - 1.0) * errors)
        return loss.mean()


def set_seed(seed):
    """设置随机种子以保证可重现性。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_target_stats(csv_file: str, target_column: str):
    """从训练集CSV计算标签均值和标准差。"""
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]
    if target_column not in df.columns:
        raise KeyError(f"目标列不存在: {target_column}")
    target = pd.to_numeric(df[target_column], errors="coerce")
    if target.isna().all():
        raise ValueError(f"目标列 {target_column} 全为空或非数值")
    mean = float(target.mean())
    std = float(target.std(ddof=0))
    if std < 1e-8:
        logger.warning(
            f"目标列标准差过小 ({std:.3e})，将使用 std=1.0（常见于单样本过拟合或极小 CSV）"
        )
        std = 1.0
    return mean, std


def compute_global_descriptor_stats(
    root_dir: str,
    csv_file: str,
    design_column: str,
    file_suffix: str,
    max_samples: int = 0,
    seed: int = 42,
):
    """从训练集完整mesh统计全局几何描述符的均值和标准差。"""
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]
    if design_column not in df.columns:
        raise KeyError(f"Design列不存在: {design_column}")

    if max_samples and max_samples > 0 and max_samples < len(df):
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.choice(len(df), size=max_samples, replace=False)]

    descriptor_list = []
    for _, row in df.iterrows():
        design_id = str(row[design_column]).strip()
        geometry_path = os.path.join(root_dir, f"{design_id}{file_suffix}")
        mesh = trimesh.load(geometry_path, force="mesh")
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        descriptor = DrivAerNetDataset.compute_global_geometry_descriptors(vertices)
        descriptor_list.append(descriptor)

    descriptors = torch.stack(descriptor_list, dim=0)
    mean = descriptors.mean(dim=0)
    std = descriptors.std(dim=0, unbiased=False).clamp_min(1e-8)
    return mean, std


def serialize_config(config):
    config_dict = {
        k: v for k, v in dict(config.__dict__).items()
        if not str(k).startswith("_")
    }
    if "GLOBAL_DESCRIPTOR_MEAN" in config_dict and isinstance(config_dict["GLOBAL_DESCRIPTOR_MEAN"], torch.Tensor):
        config_dict["GLOBAL_DESCRIPTOR_MEAN"] = config_dict["GLOBAL_DESCRIPTOR_MEAN"].detach().cpu().tolist()
    if "GLOBAL_DESCRIPTOR_STD" in config_dict and isinstance(config_dict["GLOBAL_DESCRIPTOR_STD"], torch.Tensor):
        config_dict["GLOBAL_DESCRIPTOR_STD"] = config_dict["GLOBAL_DESCRIPTOR_STD"].detach().cpu().tolist()
    return config_dict


def build_model_spec(config):
    """构建可复现实验的模型结构快照，便于 checkpoint 严格对齐。"""
    return {
        "QUANTILES": list(config.QUANTILES),
        "OUT_DIM": int(config.OUT_DIM),
        "FUNCTIONAL_DIM": int(config.FUNCTIONAL_DIM),
        "GEOTRANS_OUT_DIM": int(config.GEOTRANS_OUT_DIM),
        "GEOMETRY_DIM": int(config.GEOMETRY_DIM),
        "N_HIDDEN": int(config.N_HIDDEN),
        "N_LAYERS": int(config.N_LAYERS),
        "N_HEAD": int(config.N_HEAD),
        "DROPOUT": float(config.DROPOUT),
        "SLICE_NUM": int(config.SLICE_NUM),
        "POOLING_TYPE": str(config.POOLING_TYPE),
        "NUM_POINTS": int(config.NUM_POINTS),
        "BACKBONE_TYPE": str(getattr(config, "BACKBONE_TYPE", "transolver")),
        "BACKBONE_OUT_DIM": int(getattr(config, "BACKBONE_OUT_DIM", config.GEOTRANS_OUT_DIM)),
        "POINT_SURFACE_FEATURES": bool(getattr(config, "POINT_SURFACE_FEATURES", False)),
        "POINT_USE_CURVATURE": bool(getattr(config, "POINT_USE_CURVATURE", False)),
        "USE_AREA_WEIGHTED_POOLING": bool(getattr(config, "USE_AREA_WEIGHTED_POOLING", False)),
    }


def load_or_compute_global_descriptor_stats(config):
    """读取缓存或计算全局几何描述符统计量（mean/std）。"""
    cache_path = getattr(config, "GLOBAL_DESC_STATS_CACHE", "")
    use_norm = bool(getattr(config, "GLOBAL_DESC_NORM", True))
    if not use_norm:
        mean = torch.zeros(15, dtype=torch.float32)
        std = torch.ones(15, dtype=torch.float32)
        return mean, std

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        return mean, std

    max_samples = int(getattr(config, "GLOBAL_DESC_STATS_MAX_SAMPLES", 0) or 0)
    mean, std = compute_global_descriptor_stats(
        config.STL_ROOT_DIR,
        config.TRAIN_CSV,
        config.DESIGN_COLUMN,
        config.FILE_SUFFIX,
        max_samples=max_samples,
        seed=config.SEED,
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    return mean, std


def create_dataloaders(config):
    """创建数据加载器。"""
    if is_main_process():
        logger.info("创建数据加载器...")

    dataset = DrivAerNetDataset(
        **_dataset_kwargs(config),
        csv_file=config.TRAIN_CSV,
        apply_augmentations=config.APPLY_AUGMENTATIONS,
        normalize=True,  # center only; keep sample scale information
        deterministic_sampling=config.DETERMINISTIC_SAMPLING,
        deterministic_seed_base=config.SEED,
    )

    subset_size = int(getattr(config, "OVERFIT_SUBSET_SIZE", 0) or 0)
    dataset_size_before_subset = len(dataset)
    if getattr(config, "OVERFIT_SINGLE", False):
        sample_index = int(getattr(config, "OVERFIT_SAMPLE_INDEX", 0) or 0)
        sample_index = sample_index % max(dataset_size_before_subset, 1)
        dataset = Subset(dataset, [sample_index])
        if is_main_process():
            logger.info(
                "单样本过拟合模式: 固定使用索引 "
                f"{sample_index}/{dataset_size_before_subset} 的样本"
            )
    elif subset_size > 0 and subset_size < dataset_size_before_subset:
        rng = np.random.default_rng(config.SEED)
        subset_indices = rng.choice(dataset_size_before_subset, size=subset_size, replace=False).tolist()
        dataset = Subset(dataset, subset_indices)
        if is_main_process():
            logger.info(f"快速过拟合模式: 使用 {subset_size}/{dataset_size_before_subset} 个样本")
    elif subset_size >= dataset_size_before_subset and is_main_process():
        logger.info("OVERFIT_SUBSET_SIZE >= 数据集总量，将使用完整训练集")

    sampler = None
    shuffle = True
    if config.DISTRIBUTED:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
        shuffle = False

    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
        persistent_workers=(config.NUM_WORKERS > 0 and config.DATALOADER_PERSISTENT_WORKERS),
        prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
    )

    if is_main_process():
        logger.info(f"训练集加载器: {len(dataset)} 样本")
        logger.info(
            f"点云缓存: enabled={config.ENABLE_POINT_CACHE}, "
            f"dir={config.POINT_CACHE_DIR}, version={config.POINT_CACHE_VERSION}"
        )
        logger.info(
            f"mesh缓存: enabled={config.ENABLE_MESH_CACHE}, "
            f"dir={config.MESH_CACHE_DIR}, version={config.MESH_CACHE_VERSION}"
        )

    return train_loader, sampler


def train_one_epoch(model, dataloader, optimizer, criterion, device, config):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    pred_std_running = 0.0
    label_std_running = 0.0
    skipped_loss_steps = 0
    skipped_grad_steps = 0

    use_amp = bool(getattr(config, "USE_AMP", True)) and (device.type == "cuda")
    scaler: GradScaler | None = getattr(config, "_grad_scaler", None)

    def _all_ranks_finite(is_finite_flag: torch.Tensor) -> bool:
        finite_tensor = is_finite_flag.detach().to(device=device, dtype=torch.int32)
        if config.DISTRIBUTED:
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        return bool(finite_tensor.item())

    for batch_idx, (point_clouds, global_geometry_descriptors, cd_values) in enumerate(dataloader):
        point_clouds = point_clouds.to(device)
        global_geometry_descriptors = global_geometry_descriptors.to(device)
        cd_values = cd_values.to(device)
        should_log = is_main_process() and (
            (batch_idx + 1) % max(1, int(getattr(config, "LOG_INTERVAL", 1))) == 0
            or (batch_idx + 1) == len(dataloader)
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            cd_pred = model(point_clouds, global_geometry_descriptors)
            base_model = model.module if hasattr(model, "module") else model
            if should_log and getattr(config, "LOG_DEBUG_FEATURES", False) and hasattr(base_model, "debug_stats"):
                logger.info(f"DEBUG_FEATURE: {base_model.debug_stats}")
            # cd_values from dataset is already shaped as (B, 1).
            target = cd_values
            main_loss = criterion(cd_pred, target)
            median_pred = cd_pred[:, 1:2].float()
            q50_loss = F.smooth_l1_loss(median_pred, target.float())
            pred_std_t = median_pred.std(unbiased=False)
            label_std_t = target.float().std(unbiased=False)
            std_ratio = pred_std_t / (label_std_t + 1e-6)
            std_reg_loss = (std_ratio - 1.0).pow(2)
            loss = (
                main_loss
                + config.Q50_LOSS_WEIGHT * q50_loss
                + config.STD_REG_WEIGHT * std_reg_loss
            )
        finite_loss_all = _all_ranks_finite(torch.isfinite(loss))
        if not finite_loss_all:
            skipped_loss_steps += 1
            if should_log:
                logger.warning(
                    f"  Batch [{batch_idx+1}/{len(dataloader)}] non-finite loss detected across ranks, skip optimizer step."
                )
            optimizer.zero_grad(set_to_none=True)
            continue
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            finite_grad_all = _all_ranks_finite(torch.isfinite(grad_norm))
            if not finite_grad_all:
                skipped_grad_steps += 1
                if should_log:
                    logger.warning(
                        f"  Batch [{batch_idx+1}/{len(dataloader)}] non-finite grad_norm detected across ranks, skip optimizer step."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            finite_grad_all = _all_ranks_finite(torch.isfinite(grad_norm))
            if not finite_grad_all:
                skipped_grad_steps += 1
                if should_log:
                    logger.warning(
                        f"  Batch [{batch_idx+1}/{len(dataloader)}] non-finite grad_norm detected across ranks, skip optimizer step."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pred_std_running += pred_std_t.detach().item()
        label_std_running += label_std_t.detach().item()

        if should_log:
            pred_mean = median_pred.mean().detach().item()
            pred_std = median_pred.std(unbiased=False).detach().item()
            label_mean = cd_values.mean().detach().item()
            label_std = cd_values.std(unbiased=False).detach().item()
            q_low_mean = cd_pred[:, 0].mean().detach().item()
            q_mid_mean = cd_pred[:, 1].mean().detach().item()
            q_high_mean = cd_pred[:, -1].mean().detach().item()
            avg_interval_width = (cd_pred[:, -1] - cd_pred[:, 0]).mean().detach().item()
            logger.info(
                f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f} (Main: {main_loss.item():.6f}, Q50: {q50_loss.item():.6f}, StdReg: {std_reg_loss.item():.6f}) | "
                f"Q05/Q50/Q95(mean): {q_low_mean:.6f}/{q_mid_mean:.6f}/{q_high_mean:.6f} | "
                f"PI90 width: {avg_interval_width:.6f} | "
                f"Pred(mean/std): {pred_mean:.6f}/{pred_std:.6f} | "
                f"Label(mean/std): {label_mean:.6f}/{label_std:.6f} | "
                f"GradNorm: {float(grad_norm):.6f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_pred_std = pred_std_running / max(num_batches, 1)
    avg_label_std = label_std_running / max(num_batches, 1)
    if config.DISTRIBUTED:
        stats_tensor = torch.tensor(
            [avg_loss, avg_pred_std, avg_label_std, float(skipped_loss_steps), float(skipped_grad_steps)],
            device=device,
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        stats_tensor = stats_tensor / dist.get_world_size()
        avg_loss, avg_pred_std, avg_label_std, skipped_loss_steps, skipped_grad_steps = stats_tensor.tolist()
        skipped_loss_steps = int(round(skipped_loss_steps))
        skipped_grad_steps = int(round(skipped_grad_steps))

    return avg_loss, avg_pred_std, avg_label_std, skipped_loss_steps, skipped_grad_steps


@torch.no_grad()
def log_overfit_subset_metrics(model, dataloader, device, config, epoch: int):
    """过拟合子集时打印预测 vs 标签，便于确认接口与收敛。"""
    overfit_single = getattr(config, "OVERFIT_SINGLE", False)
    overfit_mode = getattr(config, "OVERFIT_MODE", False) and not overfit_single
    if not (overfit_single or overfit_mode) or not is_main_process():
        return

    eval_model = model.module if hasattr(model, "module") else model
    was_training = eval_model.training
    eval_model.eval()

    tag = "OVERFIT_SINGLE" if overfit_single else "OVERFIT"
    use_amp = bool(getattr(config, "USE_AMP", True)) and (device.type == "cuda")
    sample_idx = 0
    for point_clouds, global_geometry_descriptors, cd_values in dataloader:
        point_clouds = point_clouds.to(device, non_blocking=True)
        global_geometry_descriptors = global_geometry_descriptors.to(device, non_blocking=True)
        cd_values = cd_values.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            cd_pred = eval_model(point_clouds, global_geometry_descriptors)

        target = cd_values.detach().cpu().numpy().reshape(-1)
        pred = cd_pred.detach().cpu().numpy().reshape(-1, cd_pred.shape[-1])
        for row in range(pred.shape[0]):
            q05, q50, q95 = pred[row, 0], pred[row, 1], pred[row, 2]
            y = float(target[row])
            abs_err = abs(q50 - y)
            logger.info(
                f"[{tag}] Epoch {epoch + 1} sample {sample_idx}: "
                f"target={y:.6f}, pred(q05/q50/q95)={q05:.6f}/{q50:.6f}/{q95:.6f}, "
                f"|q50-target|={abs_err:.6f}, interval_width={q95 - q05:.6f}"
            )
            sample_idx += 1
        break

    if was_training:
        eval_model.train()


@torch.no_grad()
def evaluate_loss_rank0(model, dataloader, criterion, device, config):
    """
    只在 rank0 上用 model.module 做完整 validation forward。
    不通过 DDP wrapper，避免其他 rank 不参与 forward 时的同步问题。
    """
    if not is_main_process() or dataloader is None:
        return None

    eval_model = model.module if hasattr(model, "module") else model
    was_training = eval_model.training
    eval_model.eval()

    total_loss = 0.0
    num_batches = 0
    use_amp = bool(getattr(config, "USE_AMP", True)) and (device.type == "cuda")

    for point_clouds, global_geometry_descriptors, cd_values in dataloader:
        point_clouds = point_clouds.to(device, non_blocking=True)
        global_geometry_descriptors = global_geometry_descriptors.to(device, non_blocking=True)
        cd_values = cd_values.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            pred = eval_model(point_clouds, global_geometry_descriptors)
            loss = criterion(pred, cd_values)

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            num_batches += 1

    if was_training:
        eval_model.train()

    if num_batches == 0:
        return float("nan")
    return total_loss / num_batches


def train(config):
    """主训练函数。"""
    rank, local_rank, world_size = setup_distributed()

    if is_main_process():
        logger.info("=" * 50)
        logger.info(f"开始训练 Cd 预测模型，BACKBONE_TYPE={config.BACKBONE_TYPE}")
        if getattr(config, "OVERFIT_SINGLE", False):
            logger.info(
                "单样本过拟合测试已启用 (OVERFIT_SINGLE=1): "
                f"sample_index={config.OVERFIT_SAMPLE_INDEX}, "
                f"num_points={config.NUM_POINTS}, batch_size={config.BATCH_SIZE}, "
                f"normalize_target={config.NORMALIZE_TARGET}, "
                f"global_desc_norm={config.GLOBAL_DESC_NORM}"
            )
        elif getattr(config, "OVERFIT_MODE", False):
            logger.info(
                "快速过拟合测试已启用 (OVERFIT_MODE=1): "
                f"subset_size={config.OVERFIT_SUBSET_SIZE}, "
                f"num_points={config.NUM_POINTS}, batch_size={config.BATCH_SIZE}, "
                f"normalize_target={config.NORMALIZE_TARGET}, "
                f"global_desc_norm={config.GLOBAL_DESC_NORM}"
            )
        if (
            (getattr(config, "OVERFIT_SINGLE", False) or getattr(config, "OVERFIT_MODE", False))
            and config.DISTRIBUTED
            and world_size > 1
        ):
            logger.warning(
                "过拟合冒烟测试不建议多卡 DDP，请使用 NPROC_PER_NODE=1 或直接 python train.py"
            )
        logger.info("=" * 50)

    set_seed(config.SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config.TARGET_MEAN, config.TARGET_STD = compute_target_stats(
        config.TRAIN_CSV, config.TARGET_COLUMN
    )
    config.GLOBAL_DESCRIPTOR_MEAN, config.GLOBAL_DESCRIPTOR_STD = load_or_compute_global_descriptor_stats(
        config
    )

    if config.DISTRIBUTED:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.DEVICE)

    if is_main_process():
        logger.info(f"使用设备: {device}")
        if config.NORMALIZE_TARGET:
            logger.info(
                f"标签标准化启用: mean={config.TARGET_MEAN:.6f}, std={config.TARGET_STD:.6f}"
            )
        logger.info(
            "global geometry descriptor标准化启用: "
            f"mean={config.GLOBAL_DESCRIPTOR_MEAN.tolist()}, "
            f"std={config.GLOBAL_DESCRIPTOR_STD.tolist()}"
        )
        logger.info(f"确定性采样: {config.DETERMINISTIC_SAMPLING}")
        if config.DISTRIBUTED:
            logger.info(f"分布式训练启用: world_size={world_size}, rank={rank}")

    train_loader, train_sampler = create_dataloaders(config)

    val_loader = None
    use_validation = bool(getattr(config, "VAL_CSV", "")) and os.path.isfile(config.VAL_CSV)

    if is_main_process():
        if use_validation:
            logger.info(f"启用官方 validation set 用于 best model selection: {config.VAL_CSV}")
            val_dataset = _build_eval_dataset(config, config.VAL_CSV)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                sampler=None,
                num_workers=config.NUM_WORKERS,
                pin_memory=True if config.DEVICE == "cuda" else False,
                persistent_workers=(
                    config.NUM_WORKERS > 0 and config.DATALOADER_PERSISTENT_WORKERS
                ),
                prefetch_factor=config.DATALOADER_PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
            )
            logger.info(f"验证集加载器: {len(val_dataset)} 样本")
        else:
            logger.info("未找到 validation CSV，best checkpoint 将按 train_loss 保存。")

        if use_validation:
            logger.info("最佳 checkpoint 以官方 validation pinball loss 为准。")
        else:
            logger.info("未启用 validation，最佳 checkpoint 以 train pinball loss 为准。")
        logger.info("初始化模型...")
    model = CdPredictionModel(config).to(device)
    if config.DISTRIBUTED:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if is_main_process():
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = QuantilePinballLoss(config.QUANTILES)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = None
    if config.USE_COSINE_SCHEDULER:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6,
        )
    config._grad_scaler = GradScaler("cuda", enabled=(config.USE_AMP and device.type == "cuda"))

    best_metric = float("inf")
    best_checkpoint = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    epochs_without_improvement = 0
    training_history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    if is_main_process():
        logger.info(f"训练 {config.NUM_EPOCHS} epochs...")
        logger.info(f"Batch Size (per GPU): {config.BATCH_SIZE}")
        logger.info(
            "配置: "
            f"GEOTRANS_OUT_DIM={config.GEOTRANS_OUT_DIM}, "
            f"POOLING_TYPE={config.POOLING_TYPE}, "
            f"QUANTILES={config.QUANTILES}, "
            f"OVERFIT_MODE={config.OVERFIT_MODE}, "
            f"OVERFIT_SINGLE={getattr(config, 'OVERFIT_SINGLE', False)}, "
            f"OVERFIT_SUBSET_SIZE={config.OVERFIT_SUBSET_SIZE}, "
            f"LR={config.LEARNING_RATE}, "
            f"DROPOUT={config.DROPOUT}, "
            f"WEIGHT_DECAY={config.WEIGHT_DECAY}, "
            f"USE_COSINE_SCHEDULER={config.USE_COSINE_SCHEDULER}, "
            f"STD_REG_WEIGHT={config.STD_REG_WEIGHT}, "
            f"Q50_LOSS_WEIGHT={config.Q50_LOSS_WEIGHT}, "
            f"FUNCTIONAL_DIM={config.FUNCTIONAL_DIM}, "
            f"POINT_SURFACE_FEATURES={config.POINT_SURFACE_FEATURES}, "
            f"POINT_USE_CURVATURE={config.POINT_USE_CURVATURE}, "
            f"USE_AREA_WEIGHTED_POOLING={config.USE_AREA_WEIGHTED_POOLING}, "
            f"EARLY_STOPPING_PATIENCE={config.EARLY_STOPPING_PATIENCE}, "
            f"VALIDATE_EVERY={config.VALIDATE_EVERY}"
        )
        if config.DISTRIBUTED:
            logger.info(f"Global Batch Size: {config.BATCH_SIZE * world_size}")

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = datetime.now()
        if config.DISTRIBUTED and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_pred_std, train_label_std, skipped_loss_steps, skipped_grad_steps = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        log_overfit_subset_metrics(model, train_loader, device, config, epoch)

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        val_loss = None
        if use_validation and ((epoch + 1) % max(1, config.VALIDATE_EVERY) == 0):
            val_loss = evaluate_loss_rank0(model, val_loader, criterion, device, config)
        training_history["val_loss"].append(
            float(val_loss) if val_loss is not None else float("nan")
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        metric_for_best = train_loss
        metric_name = "train_loss"
        if is_main_process() and use_validation and val_loss is not None and np.isfinite(val_loss):
            metric_for_best = float(val_loss)
            metric_name = "val_loss"

        improved = False
        if is_main_process():
            improved = metric_for_best < best_metric
            if improved:
                best_metric = metric_for_best
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        stop_now = False
        if is_main_process():
            patience = int(getattr(config, "EARLY_STOPPING_PATIENCE", 0) or 0)
            stop_now = bool(patience > 0 and epochs_without_improvement >= patience)

        if config.DISTRIBUTED:
            stop_tensor = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())

        if is_main_process():
            logger.info(
                f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] "
                f"TrainLoss: {train_loss:.6f} "
                f"ValLoss: {val_loss if val_loss is not None else float('nan'):.6f} "
                f"BestMetric({metric_name}): {best_metric:.6f} "
                f"PredStd: {train_pred_std:.6f} "
                f"LabelStd: {train_label_std:.6f} "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} "
                f"Time: {epoch_time:.1f}s"
            )
            logger.info(
                f"Skipped steps: loss={skipped_loss_steps}, grad={skipped_grad_steps}, total_batches={len(train_loader)}"
            )

        if is_main_process() and improved:
            model_state = (
                model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            )
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss": metric_for_best,
                "best_metric": best_metric,
                "best_metric_name": metric_name,
                "val_csv": config.VAL_CSV,
                "config": serialize_config(config),
                "model_spec": build_model_spec(config),
            }
            torch.save(payload, best_checkpoint)
            logger.info(
                f"✓ 保存最佳模型到 {best_checkpoint} ({metric_name}={metric_for_best:.6f})"
            )

        if stop_now:
            if is_main_process():
                logger.info(
                    f"Early stopping: {epochs_without_improvement} epochs without improvement "
                    f"(patience={config.EARLY_STOPPING_PATIENCE})"
                )
            break

        if is_main_process() and (epoch + 1) % 10 == 0:
            checkpoint = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"
            )
            model_state = (
                model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            )
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "loss": train_loss,
                "config": serialize_config(config),
                "model_spec": build_model_spec(config),
            }
            torch.save(ckpt, checkpoint)

    if is_main_process():
        history_file = os.path.join(config.LOG_DIR, "training_history.json")
        with open(history_file, "w") as f:
            json.dump(training_history, f, indent=2)

        curve_path = getattr(config, "LOSS_CURVE_FILE", "") or ""
        if not curve_path:
            curve_path = os.path.join(config.LOG_DIR, "loss_curves.png")
        save_loss_curves_figure(training_history, curve_path)

        logger.info("\n" + "=" * 50)
        logger.info("训练完成!")
        logger.info(f"最佳模型已保存到: {best_checkpoint}")
        logger.info(f"训练历史已保存到: {history_file}")
        logger.info(f"损失曲线图已保存到: {curve_path}")
        logger.info("=" * 50)

        _cvplus_flag = os.getenv("CVPLUS_SAVE_OOF", "").strip().lower()
        if _cvplus_flag in ("1", "true", "yes"):
            try:
                from cvplus_oof import save_cvplus_oof_scores

                _fold = int(os.getenv("SPLIT_CALIB_FOLD", "0"))
                save_cvplus_oof_scores(
                    best_checkpoint,
                    _fold,
                    config,
                    device,
                    model,
                )
            except Exception:
                logger.exception("[CV+] 保存 OOF conformity scores 失败")

    cleanup_distributed()


if __name__ == "__main__":
    config = Config()
    train(config)
