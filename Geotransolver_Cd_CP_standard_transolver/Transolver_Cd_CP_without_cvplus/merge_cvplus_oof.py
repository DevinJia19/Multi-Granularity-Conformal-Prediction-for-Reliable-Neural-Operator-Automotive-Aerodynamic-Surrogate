#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并五折 OOF 校准数据，计算 CV+ 非对称 CQR 的 q_l / q_u，写入 hat_q.json。

用法:
  python merge_cvplus_oof.py
  python merge_cvplus_oof.py --oof-dir ./results/cvplus --alpha 0.1

依赖: 先完成 5 次 fold 训练且每次 CVPLUS_SAVE_OOF=1，生成 oof_fold_0.npz … oof_fold_4.npz。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from cqr_common import asymmetric_cqr_hat_q


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge CV+ OOF data and compute asymmetric CQR q_l / q_u.")
    ap.add_argument(
        "--oof-dir",
        default=os.getenv("CVPLUS_OOF_DIR", "./results/cvplus"),
        help="directory containing oof_fold_*.npz",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=float(os.getenv("CQR_ALPHA", "0.1")),
        help="CQR miscoverage (nominal coverage = 1-alpha)",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="output calibration JSON path (default: <oof-dir>/hat_q.json)",
    )
    args = ap.parse_args()

    oof_dir = args.oof_dir.strip() or "./results/cvplus"
    pattern = os.path.join(oof_dir, "oof_fold_*.npz")
    files = sorted(glob.glob(pattern))
    if len(files) < 1:
        print("[ERROR] 未找到 %s ，请先完成各 fold 训练并保存 OOF。" % pattern, file=sys.stderr)
        return 1

    expected_folds = int(os.getenv("CVPLUS_N_SPLITS", os.getenv("SPLIT_N_SPLITS", "5")))
    found_folds: list[int] = []
    for path in files:
        z = np.load(path)
        if "fold" in z.files:
            found_folds.append(int(z["fold"]))
        else:
            m = re.search(r"oof_fold_(\d+)\.npz", os.path.basename(path))
            if m:
                found_folds.append(int(m.group(1)))

    found_folds = sorted(set(found_folds))
    expected = list(range(expected_folds))
    if found_folds != expected:
        print(
            "[ERROR] OOF folds 不完整: found=%s, expected=%s (pattern=%s)"
            % (found_folds, expected, pattern),
            file=sys.stderr,
        )
        return 1

    q05_parts, q95_parts, y_parts = [], [], []
    for path in files:
        z = np.load(path)
        if "q05_pred" in z.files:
            q05_parts.append(z["q05_pred"].astype(np.float64).ravel())
            q95_parts.append(z["q95_pred"].astype(np.float64).ravel())
            y_parts.append(z["y_true"].astype(np.float64).ravel())
        else:
            print(
                "[ERROR] %s 缺少 q05_pred/q95_pred/y_true（旧版对称 scores 格式已不再支持）"
                % path,
                file=sys.stderr,
            )
            return 1

    q05 = np.concatenate(q05_parts, axis=0)
    q95 = np.concatenate(q95_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    q_l, q_u = asymmetric_cqr_hat_q(q05, q95, y, float(args.alpha))

    out_json = args.out_json.strip() or os.path.join(oof_dir, "hat_q.json")
    payload = {
        "protocol": "cvplus_oof_merged_asymmetric",
        "alpha": float(args.alpha),
        "n_cal": int(y.size),
        "n_files": len(files),
        "n_splits": expected_folds,
        "folds": found_folds,
        "oof_files": [os.path.basename(f) for f in files],
        "q_l": q_l,
        "q_u": q_u,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    merged_npz = os.path.join(oof_dir, "oof_merged.npz")
    np.savez_compressed(merged_npz, q05_pred=q05, q95_pred=q95, y_true=y)

    print(
        "[OK] merged n=%d from %d folds %s; q_l=%s q_u=%s"
        % (y.size, len(files), found_folds, q_l, q_u)
    )
    print("     JSON: %s" % out_json)
    print("     NPZ:  %s" % merged_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
