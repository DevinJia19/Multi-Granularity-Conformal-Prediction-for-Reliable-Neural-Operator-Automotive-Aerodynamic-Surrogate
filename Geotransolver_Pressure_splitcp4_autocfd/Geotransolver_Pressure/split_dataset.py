#!/usr/bin/env python3
"""Split AutoCFD DrivAerML surface zarr runs into official train/val/test + CV+ folds."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import warnings
from pathlib import Path

from sklearn.model_selection import KFold

RUN_PATTERN = re.compile(r"run_(\d+)\.zarr$", re.IGNORECASE)

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

OFFICIAL_TRAIN_SET = set(AUTO_CFD_TRAIN_IDS)
OFFICIAL_VAL_SET = set(AUTO_CFD_VAL_IDS)
OFFICIAL_TEST_SET = set(AUTO_CFD_TEST_IDS)
ALL_OFFICIAL_IDS = OFFICIAL_TRAIN_SET | OFFICIAL_VAL_SET | OFFICIAL_TEST_SET


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split run_*.zarr entries using AutoCFD official train/val/test IDs, "
            "with optional 4-fold CV+ holdout inside official train."
        )
    )
    _repo_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/"
            "Datasets/DrivAerML/Surface_Field/drivaerml_surface_zarr_all"
        ),
        help="Root directory containing run_*.zarr entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root / "data_splits",
        help="Output root for split directories and manifests.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of CV+ folds inside official train (default: 4).",
    )
    parser.add_argument(
        "--calib-fold",
        type=int,
        default=0,
        help="CV+ fold index used as calib/OOF holdout (0 .. n_splits-1).",
    )
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="Use full official train (400 cases) for data_splits/train.",
    )
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing split entries when symlinking/copying/moving.",
    )
    return parser.parse_args()


def find_zarr_runs_by_id(dataset_dir: Path) -> dict[int, Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    by_id: dict[int, Path] = {}
    for path in sorted(dataset_dir.iterdir(), key=lambda x: x.name):
        if not (path.is_dir() or path.is_file()):
            continue
        match = RUN_PATTERN.match(path.name)
        if match:
            by_id[int(match.group(1))] = path

    if not by_id:
        raise ValueError(
            f"No run_*.zarr entries were found under {dataset_dir}. "
            "Expected names like run_123.zarr."
        )
    return by_id


def _resolve_paths(run_ids: list[int], by_id: dict[int, Path], label: str) -> list[Path]:
    paths = []
    missing = []
    for rid in run_ids:
        if rid not in by_id:
            missing.append(rid)
            continue
        paths.append(by_id[rid])
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} official {label} Run ID(s) under dataset_dir: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    return paths


def cvplus_fold_split(
    official_train_ids: list[int],
    n_splits: int,
    calib_fold: int,
) -> tuple[list[int], list[int]]:
    if calib_fold < 0 or calib_fold >= n_splits:
        raise ValueError(f"calib_fold must be in [0, {n_splits - 1}], got {calib_fold}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(kf.split(official_train_ids))
    train_idx, calib_idx = folds[calib_fold]
    fold_train = [official_train_ids[i] for i in train_idx]
    fold_calib = [official_train_ids[i] for i in calib_idx]
    return fold_train, fold_calib


def write_split_manifest(split_paths: dict[str, list[Path]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, paths in split_paths.items():
        manifest_file = output_dir / f"{split_name}_files.txt"
        with manifest_file.open("w", encoding="utf-8") as f:
            for p in paths:
                f.write(str(p.resolve()) + "\n")
        print(f"Wrote {len(paths)} entries to {manifest_file}")


def _same_symlink_target(dest: Path, src: Path) -> bool:
    if not dest.is_symlink():
        return False
    try:
        return dest.resolve() == src.resolve()
    except OSError:
        return False


def _replace_dest_if_needed(dest: Path, force: bool, src: Path) -> bool:
    if not dest.exists() and not dest.is_symlink():
        return False
    if _same_symlink_target(dest, src):
        print(f"Skip (unchanged symlink): {dest}")
        return True
    if not force:
        raise FileExistsError(
            f"Destination already exists: {dest}. "
            "Remove it or re-run with --force."
        )
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)
    else:
        dest.unlink()
    return False


def copy_or_move_split(split_paths, output_dir: Path, action: str, force: bool = False):
    for split_name, paths in split_paths.items():
        target_dir = output_dir / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dest = target_dir / src.name
            if dest.exists() or dest.is_symlink():
                if force:
                    if dest.is_symlink() or dest.is_file():
                        dest.unlink()
                    elif dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                else:
                    raise FileExistsError(f"Destination already exists: {dest}")
            if src.is_dir():
                if action == "copy":
                    shutil.copytree(src, dest)
                else:
                    shutil.move(str(src), str(dest))
            else:
                if action == "copy":
                    shutil.copy2(src, dest)
                else:
                    shutil.move(str(src), str(dest))
        print(f"{action.capitalize()}ed {len(paths)} entries to {target_dir}")


def symlink_split(split_paths, output_dir: Path, force: bool = False):
    for split_name, paths in split_paths.items():
        target_dir = output_dir / split_name

        if force and target_dir.exists():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        for src in paths:
            dest = target_dir / src.name
            dest.symlink_to(src, target_is_directory=src.is_dir())
            created += 1
        print(
            f"Symlinked {created} new entries under {target_dir} "
            f"({len(paths)} total in split)."
        )


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_dir.resolve()

    by_id = find_zarr_runs_by_id(dataset_dir)

    extra_ids = sorted(set(by_id) - ALL_OFFICIAL_IDS)
    if extra_ids:
        warnings.warn(
            f"Found {len(extra_ids)} Run ID(s) outside official AutoCFD split; "
            f"ignored (first few: {extra_ids[:10]}).",
            stacklevel=2,
        )

    official_train_paths = _resolve_paths(
        sorted(AUTO_CFD_TRAIN_IDS), by_id, "train"
    )
    official_val_paths = _resolve_paths(sorted(AUTO_CFD_VAL_IDS), by_id, "validation")
    official_test_paths = _resolve_paths(sorted(AUTO_CFD_TEST_IDS), by_id, "test")

    if args.final_train:
        train_paths = official_train_paths
        calib_paths: list[Path] = []
    else:
        fold_train_ids, fold_calib_ids = cvplus_fold_split(
            AUTO_CFD_TRAIN_IDS, args.n_splits, args.calib_fold
        )
        train_paths = _resolve_paths(fold_train_ids, by_id, "CV+ train")
        calib_paths = _resolve_paths(fold_calib_ids, by_id, "CV+ calib/OOF")

    split_paths = {
        "train": train_paths,
        "calib": calib_paths,
        "val": official_val_paths,
        "test": official_test_paths,
    }

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_root}")
    print(f"official train total: {len(official_train_paths)}")
    if args.final_train:
        print(f"train (final, full official train): {len(train_paths)}")
        print("current fold calib / OOF holdout: 0 (disabled with --final-train)")
    else:
        print(f"current fold train: {len(train_paths)}")
        print(f"current fold calib / OOF holdout: {len(calib_paths)}")
        print(f"CV+ n_splits={args.n_splits}, calib_fold={args.calib_fold}")
    print(f"official validation: {len(official_val_paths)}")
    print(f"official test: {len(official_test_paths)}")

    write_split_manifest(split_paths, output_root)

    official_manifest = output_root / "official_train_files.txt"
    with official_manifest.open("w", encoding="utf-8") as f:
        for p in official_train_paths:
            f.write(str(p.resolve()) + "\n")
    print(f"Wrote {len(official_train_paths)} entries to {official_manifest}")

    meta = {
        "protocol": "autocfd_official_split_cvplus",
        "dataset_dir": str(dataset_dir),
        "n_splits": args.n_splits,
        "calib_fold": args.calib_fold,
        "final_train": args.final_train,
        "counts": {k: len(v) for k, v in split_paths.items()},
        "official_train_count": len(official_train_paths),
        "official_val_count": len(official_val_paths),
        "official_test_count": len(official_test_paths),
    }
    meta_path = output_root / "split_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote split metadata to {meta_path}")

    actions = sum([args.move, args.copy, args.symlink])
    if actions > 1:
        raise ValueError("Choose only one of --move, --copy, or --symlink.")

    if args.move:
        copy_or_move_split(split_paths, output_root, "move", force=args.force)
    elif args.copy:
        copy_or_move_split(split_paths, output_root, "copy", force=args.force)
    elif args.symlink:
        symlink_split(split_paths, output_root, force=args.force)

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
