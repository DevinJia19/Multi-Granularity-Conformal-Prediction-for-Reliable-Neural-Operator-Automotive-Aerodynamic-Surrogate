#!/usr/bin/env python3
import argparse
import random
import re
import shutil
from pathlib import Path

RUN_PATTERN = re.compile(r"run_(\d+)\.zarr$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a directory of run_*.zarr dataset entries into train/calib/test sets."
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
        help=(
            "Output root for split text manifests or train/calib/test directories. "
            "Defaults to the project work root data_splits folder."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for reproducible shuffling.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move matched zarr entries into train/calib/test subdirectories.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy matched zarr entries into train/calib/test subdirectories."
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links to matched zarr entries in train/calib/test subdirectories."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing train/calib/test entries when symlinking (or copying/moving).",
    )
    return parser.parse_args()


def find_zarr_runs(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    candidates = []
    for path in sorted(dataset_dir.iterdir(), key=lambda x: x.name):
        if not (path.is_dir() or path.is_file()):
            continue
        match = RUN_PATTERN.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise ValueError(
            f"No run_*.zarr entries were found under {dataset_dir}. "
            "Make sure the directory contains files or folders named like run_123.zarr."
        )

    return [path for _, path in sorted(candidates, key=lambda item: item[0])]


def write_split_manifest(split_paths, output_dir: Path):
    manifest_dir = output_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split_name, paths in split_paths.items():
        manifest_file = manifest_dir / f"{split_name}_files.txt"
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
    """Return True if caller should skip creating dest (already correct symlink)."""
    if not dest.exists() and not dest.is_symlink():
        return False
    if _same_symlink_target(dest, src):
        print(f"Skip (unchanged symlink): {dest}")
        return True
    if not force:
        raise FileExistsError(
            f"Destination already exists: {dest}. "
            "Remove it or the whole split folder, or re-run with --force."
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
                shutil.copytree(src, dest) if action == "copy" else shutil.move(str(src), str(dest))
            else:
                shutil.copy2(src, dest) if action == "copy" else shutil.move(str(src), str(dest))
        print(f"{action.capitalize()}ed {len(paths)} entries to {target_dir}")


def symlink_split(split_paths, output_dir: Path, force: bool = False):
    for split_name, paths in split_paths.items():
        target_dir = output_dir / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        for src in paths:
            dest = target_dir / src.name
            if _replace_dest_if_needed(dest, force, src):
                continue
            dest.symlink_to(src, target_is_directory=src.is_dir())
            created += 1
        print(f"Symlinked {created} new entries under {target_dir} ({len(paths)} total in split).")


def main():
    args = parse_args()
    output_root = args.output_dir or args.dataset_dir
    dataset_dir = args.dataset_dir.resolve()
    output_root = output_root.resolve()

    zarr_paths = find_zarr_runs(dataset_dir)
    total = len(zarr_paths)
    if total < 10:
        raise ValueError("Need at least 10 run_*.zarr items for a sensible 80/10/10 split.")

    random.seed(args.seed)
    shuffled = zarr_paths.copy()
    random.shuffle(shuffled)

    n_train = int(total * 0.8)
    n_calib = int(total * 0.1)
    n_test = total - n_train - n_calib
    if n_test == 0:
        n_test = 1
        if n_calib > 1:
            n_calib -= 1
        else:
            n_train -= 1

    train_paths = shuffled[:n_train]
    calib_paths = shuffled[n_train : n_train + n_calib]
    test_paths = shuffled[n_train + n_calib :]

    print(f"Found {total} run_*.zarr entries.")
    print(
        f"Split into {len(train_paths)} train, "
        f"{len(calib_paths)} calib, {len(test_paths)} test."
    )

    split_paths = {
        "train": train_paths,
        "calib": calib_paths,
        "test": test_paths,
    }

    manifest_dir = output_root
    write_split_manifest(split_paths, manifest_dir)

    if args.move and args.copy:
        raise ValueError("Please choose only one of --move or --copy.")
    if args.move and args.symlink:
        raise ValueError("Please choose only one of --move, --copy, or --symlink.")
    if args.copy and args.symlink:
        raise ValueError("Please choose only one of --move, --copy, or --symlink.")

    if args.move:
        copy_or_move_split(split_paths, output_root, "move", force=args.force)
    elif args.copy:
        copy_or_move_split(split_paths, output_root, "copy", force=args.force)
    elif args.symlink:
        symlink_split(split_paths, output_root, force=args.force)

    print("Dataset split complete.")
    print("Use the generated *_files.txt manifests to inspect or configure train/calib/test data sources.")


if __name__ == "__main__":
    main()
