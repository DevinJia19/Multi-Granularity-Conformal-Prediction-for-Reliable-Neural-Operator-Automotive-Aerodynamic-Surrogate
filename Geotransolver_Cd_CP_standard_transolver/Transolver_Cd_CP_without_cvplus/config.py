#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference configuration for the Transolver Cd transfer experiment.

The runnable scripts import ``Config`` from ``train.py``. This standalone file
documents the English-only paper configuration for the scalar Cd transfer run.
"""

from __future__ import annotations

import os


class DataConfig:
    """Dataset and split locations for the AutoCFD DrivAerML protocol."""

    STL_ROOT_DIR = os.getenv(
        "STL_ROOT_DIR",
        "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/stl",
    )
    CSV_FILE = os.getenv(
        "CSV_FILE",
        "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv",
    )
    TRAIN_CSV = os.getenv("TRAIN_CSV", "./data_splits/train_split.csv")
    CALIBRATION_CSV = os.getenv("CALIBRATION_CSV", "./data_splits/calibration_split.csv")
    VALIDATION_CSV = os.getenv("VAL_CSV", "./data_splits/validation_split.csv")
    TEST_CSV = os.getenv("TEST_CSV", "./data_splits/test_split.csv")

    DESIGN_COLUMN = "Design"
    TARGET_COLUMN = "Average Cd"
    FILE_SUFFIX = ".stl"

    NUM_POINTS = int(os.getenv("NUM_POINTS", "8192"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))


class ModelConfig:
    """Plain Transolver scalar Cd model with the same quantile head."""

    BACKBONE_TYPE = "transolver"
    FUNCTIONAL_DIM = 3
    QUANTILES = (0.05, 0.5, 0.95)
    OUT_DIM = len(QUANTILES)
    BACKBONE_OUT_DIM = 96
    GEOTRANS_OUT_DIM = BACKBONE_OUT_DIM
    GEOMETRY_DIM = 3
    POOLING_TYPE = "structured"
    REGRESSION_HEAD_DIMS = [256, 128, 64]

    N_LAYERS = 4
    N_HIDDEN = 192
    N_HEAD = 4
    SLICE_NUM = 32
    DROPOUT = float(os.getenv("DROPOUT", "0.05"))
    ACT = "gelu"
    MLP_RATIO = 4

    DEVICE = os.getenv("DEVICE", "cuda")


class TrainingConfig:
    """Optimizer and runtime defaults for scalar Cd quantile regression."""

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "600"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    Q50_LOSS_WEIGHT = float(os.getenv("Q50_LOSS_WEIGHT", "0.5"))
    WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "5"))
    USE_COSINE_SCHEDULER = os.getenv("USE_COSINE_SCHEDULER", "1") == "1"
    NORMALIZE_TARGET = os.getenv("NORMALIZE_TARGET", "1") == "1"

    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    LOG_DIR = os.getenv("LOG_DIR", "./logs")


class TestingConfig:
    """Evaluation and CQR calibration defaults."""

    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoints/best_model.pth")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
    CQR_ALPHA = float(os.getenv("CQR_ALPHA", "0.1"))
    CQR_HAT_Q_JSON = os.getenv("CQR_HAT_Q_JSON", "")


class Config:
    """Combined reference configuration."""

    def __init__(self) -> None:
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.testing = TestingConfig()
        self.create_directories()

    def create_directories(self) -> None:
        for path in (
            self.training.CHECKPOINT_DIR,
            self.training.LOG_DIR,
            self.testing.RESULTS_DIR,
            "./data_splits",
        ):
            os.makedirs(path, exist_ok=True)

    def validate(self) -> bool:
        errors: list[str] = []
        if self.model.N_HIDDEN % self.model.N_HEAD != 0:
            errors.append("N_HIDDEN must be divisible by N_HEAD.")
        if self.data.NUM_POINTS <= 0:
            errors.append("NUM_POINTS must be positive.")
        if self.training.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive.")
        if errors:
            print("\nConfiguration errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nConfiguration validation passed.")
        return not errors


if __name__ == "__main__":
    Config().validate()
