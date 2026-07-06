#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 AutoCFD 官方 Run ID 划分数据集。

生成：
  - train_pool_90_with_cv_fold.csv：官方 train 400 cases + cv_fold（CV+ / OOF）
  - official_train_split.csv：官方 train 400 cases（无 cv_fold）
  - train_split.csv / calibration_split.csv：当前 CV+ fold 的训练 / OOF holdout
  - validation_split.csv：官方 validation 34 cases
  - test_split.csv：官方 test 50 cases
  - split_meta.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AUTO_CFD_TRAIN_IDS = [
    1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 23, 25, 27, 28, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 125, 126, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 143,
    144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162,
    163, 164, 166, 168, 169, 170, 171, 172, 174, 175, 176, 178, 179, 181, 182, 183, 184,
    185, 186, 189, 190, 192, 193, 194, 195, 196, 198, 200, 201, 202, 204, 206, 209, 212,
    213, 214, 216, 217, 219, 220, 223, 224, 225, 227, 229, 230, 231, 232, 233, 235, 236,
    237, 238, 239, 240, 242, 243, 244, 245, 246, 249, 250, 251, 254, 255, 256, 257, 259,
    261, 262, 264, 265, 266, 267, 268, 269, 270, 272, 273, 274, 276, 277, 278, 279, 281,
    283, 285, 286, 287, 288, 289, 292, 293, 294, 296, 297, 299, 300, 301, 302, 304, 305,
    306, 307, 308, 309, 310, 312, 313, 314, 315, 317, 318, 319, 320, 323, 326, 327, 330,
    331, 332, 333, 334, 335, 336, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349,
    351, 353, 355, 356, 357, 358, 359, 360, 361, 362, 365, 367, 368, 369, 371, 373, 374,
    375, 377, 378, 379, 381, 383, 384, 385, 386, 388, 389, 391, 392, 393, 394, 395, 396,
    397, 398, 399, 400, 402, 404, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417,
    418, 419, 420, 421, 422, 425, 426, 427, 430, 431, 432, 433, 434, 435, 437, 438, 439,
    440, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 455, 456, 457, 458, 459,
    460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 474, 475, 476, 477, 478,
    479, 480, 481, 482, 483, 484, 485, 486, 488, 489, 490, 491, 492, 493, 494, 495, 496,
    497, 498, 499, 500,
]

AUTO_CFD_VAL_IDS = [
    4, 22, 56, 109, 150, 165, 177, 191, 228, 234, 241, 247, 252, 253, 260, 271, 275, 298,
    303, 311, 321, 324, 328, 341, 352, 366, 380, 390, 401, 423, 441, 447, 454, 487,
]

AUTO_CFD_TEST_IDS = [
    11, 12, 19, 20, 24, 26, 29, 41, 55, 59, 108, 124, 127, 133, 142, 158, 173, 180,
    187, 188, 197, 199, 203, 205, 207, 208, 210, 215, 222, 226, 258, 263, 280, 284,
    290, 322, 337, 350, 354, 363, 372, 382, 387, 405, 410, 424, 428, 429, 436, 472,
]


def extract_run_id(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        f = float(text)
        if f.is_integer():
            return int(f)
    except Exception:
        pass
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    return int(nums[-1])


def _filter_by_ids(
    df: pd.DataFrame, ids: Iterable[int], design_column: str, split_name: str
) -> pd.DataFrame:
    ids = list(ids)
    wanted = set(int(x) for x in ids)
    part = df[df["_run_id"].isin(wanted)].copy()
    found = set(part["_run_id"].astype(int).tolist())
    missing = sorted(wanted - found)
    duplicate_rows = int(part.duplicated(subset=["_run_id"]).sum())

    if missing:
        raise ValueError(f"{split_name} 有 Run ID 在 CSV 中找不到: {missing}")
    if duplicate_rows > 0:
        raise ValueError(f"{split_name} 中有重复 Run ID 行: duplicate_rows={duplicate_rows}")

    order = {rid: i for i, rid in enumerate(ids)}
    part["_official_order"] = part["_run_id"].map(order)
    part = part.sort_values("_official_order").drop(columns=["_official_order"])
    return part


def _save_without_internal_columns(df: pd.DataFrame, path: str) -> None:
    drop_cols = [c for c in ["_run_id"] if c in df.columns]
    df.drop(columns=drop_cols).to_csv(path, index=False)


def split_dataset(
    csv_file: str,
    output_dir: str = "./data_splits",
    n_splits: int = 5,
    calib_fold: int = 0,
    random_seed: int = 42,
    target_column: str = "Average Cd",
    design_column: str = "Design",
):
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]

    if design_column not in df.columns:
        raise KeyError(f"CSV 中不存在 Design 列: {design_column}")
    if target_column not in df.columns:
        raise KeyError(f"CSV 中不存在目标列: {target_column}")

    df = df.copy()
    df["_run_id"] = df[design_column].map(extract_run_id)
    if df["_run_id"].isna().any():
        bad = df.loc[df["_run_id"].isna(), design_column].head(10).tolist()
        raise ValueError(f"部分 Design 无法解析 Run ID，示例: {bad}")
    df["_run_id"] = df["_run_id"].astype(int)

    train_set = set(AUTO_CFD_TRAIN_IDS)
    val_set = set(AUTO_CFD_VAL_IDS)
    test_set = set(AUTO_CFD_TEST_IDS)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("AutoCFD official split 有重叠 Run ID")

    train_pool_df = _filter_by_ids(df, AUTO_CFD_TRAIN_IDS, design_column, "AutoCFD train")
    validation_df = _filter_by_ids(df, AUTO_CFD_VAL_IDS, design_column, "AutoCFD validation")
    test_df = _filter_by_ids(df, AUTO_CFD_TEST_IDS, design_column, "AutoCFD test")

    used_ids = train_set | val_set | test_set
    csv_ids = set(df["_run_id"].astype(int).tolist())
    unused_csv_ids = sorted(csv_ids - used_ids)
    if unused_csv_ids:
        print(f"[WARN] CSV中有 {len(unused_csv_ids)} 个 Run ID 未被 official split 使用，将忽略。")

    k = int(n_splits)
    cf = int(calib_fold) % k

    train_pool_df = train_pool_df.reset_index(drop=True).copy()
    fold_ids = np.zeros(len(train_pool_df), dtype=np.int64)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    for fold_idx, (_, fold_holdout_idx) in enumerate(kf.split(np.arange(len(train_pool_df)))):
        fold_ids[fold_holdout_idx] = fold_idx
    train_pool_df["cv_fold"] = fold_ids

    train_df = train_pool_df[train_pool_df["cv_fold"] != cf].copy().reset_index(drop=True)
    calibration_df = train_pool_df[train_pool_df["cv_fold"] == cf].copy().reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    train_csv = os.path.join(output_dir, "train_split.csv")
    calibration_csv = os.path.join(output_dir, "calibration_split.csv")
    validation_csv = os.path.join(output_dir, "validation_split.csv")
    test_csv = os.path.join(output_dir, "test_split.csv")
    pool_fold_csv = os.path.join(output_dir, "train_pool_90_with_cv_fold.csv")
    official_train_csv = os.path.join(output_dir, "official_train_split.csv")

    _save_without_internal_columns(train_df, train_csv)
    _save_without_internal_columns(calibration_df, calibration_csv)
    _save_without_internal_columns(validation_df.reset_index(drop=True), validation_csv)
    _save_without_internal_columns(test_df.reset_index(drop=True), test_csv)

    train_pool_df.drop(columns=["_run_id"]).to_csv(pool_fold_csv, index=False)
    train_pool_df.drop(columns=["_run_id", "cv_fold"]).to_csv(official_train_csv, index=False)

    meta = {
        "protocol": "autocfd_official_split_with_cvplus_oof_folds",
        "source_csv": csv_file,
        "n_splits": k,
        "calib_fold": cf,
        "random_seed_for_cv_folds": random_seed,
        "counts": {
            "csv_total": int(len(df)),
            "official_train_pool": int(len(train_pool_df)),
            "cvplus_fold_train_split": int(len(train_df)),
            "cvplus_fold_oof_holdout_calibration_split": int(len(calibration_df)),
            "official_validation": int(len(validation_df)),
            "official_test": int(len(test_df)),
            "unused_csv_ids": int(len(unused_csv_ids)),
        },
        "ids": {
            "train": AUTO_CFD_TRAIN_IDS,
            "validation": AUTO_CFD_VAL_IDS,
            "test": AUTO_CFD_TEST_IDS,
            "unused_csv_ids": unused_csv_ids,
        },
        "files": {
            "train_split": train_csv,
            "calibration_split": calibration_csv,
            "validation_split": validation_csv,
            "test_split": test_csv,
            "train_pool_with_cv_fold": pool_fold_csv,
            "official_train_split": official_train_csv,
        },
    }

    meta_path = os.path.join(output_dir, "split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("AutoCFD split done.")
    print(f"official train pool: {len(train_pool_df)}")
    print(f"current fold train: {len(train_df)}")
    print(f"current fold OOF holdout: {len(calibration_df)}")
    print(f"official validation: {len(validation_df)}")
    print(f"official test: {len(test_df)}")
    print(f"meta: {meta_path}")

    for name, part in (
        ("train", train_df),
        ("OOF holdout", calibration_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        cd = pd.to_numeric(part[target_column], errors="coerce")
        print(f"\n{name} {target_column}: min={cd.min():.6f} max={cd.max():.6f} mean={cd.mean():.6f}")

    return train_df, calibration_df, validation_df, test_df


if __name__ == "__main__":
    CSV_FILE = os.getenv(
        "SPLIT_SOURCE_CSV",
        "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv",
    )
    OUTPUT_DIR = os.getenv("DATA_SPLITS_DIR", "./data_splits")
    N_SPLITS = int(os.getenv("SPLIT_N_SPLITS", "5"))
    CALIB_FOLD = int(os.getenv("SPLIT_CALIB_FOLD", "0"))
    RANDOM_SEED = int(os.getenv("SPLIT_RANDOM_SEED", "42"))

    split_dataset(
        CSV_FILE,
        OUTPUT_DIR,
        n_splits=N_SPLITS,
        calib_fold=CALIB_FOLD,
        random_seed=RANDOM_SEED,
    )
