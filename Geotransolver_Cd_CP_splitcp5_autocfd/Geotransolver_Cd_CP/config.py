#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration helpers for the GeoTransolver Cd split-CP baseline.

The executable training and evaluation scripts import ``Config`` from
``train.py``. This file is an English-only reference copy that records the
paper settings and the AutoCFD split-CP defaults.
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
    """GeoTransolver scalar Cd model used in the paper."""

    FUNCTIONAL_DIM = 3
    QUANTILES = (0.05, 0.5, 0.95)
    OUT_DIM = len(QUANTILES)
    GEOTRANS_OUT_DIM = 96
    GEOMETRY_DIM = 3
    GLOBAL_DIM = None
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
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "720"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-5"))
    WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "5"))
    USE_COSINE_SCHEDULER = os.getenv("USE_COSINE_SCHEDULER", "1") == "1"
    NORMALIZE_TARGET = os.getenv("NORMALIZE_TARGET", "1") == "1"

    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    LOG_DIR = os.getenv("LOG_DIR", "./logs")
    SAVE_FREQ = int(os.getenv("SAVE_FREQ", "1"))
    LOG_FREQ = int(os.getenv("LOG_FREQ", "20"))


class TestingConfig:
    """Evaluation and CQR calibration defaults."""

    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoints/best_model.pth")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results/splitcp_5fold")
    CQR_ALPHA = float(os.getenv("CQR_ALPHA", "0.1"))
    SAVE_PREDICTIONS = True
    SAVE_METRICS = True


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
