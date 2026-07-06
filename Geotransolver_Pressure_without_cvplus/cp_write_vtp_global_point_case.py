#!/usr/bin/env python3
"""Write VTP files for Global CP, Point-based CP, and Case-based CP visualization.

Input .npz files are expected to contain:
    pred, target, sigma, surface_mesh_centers
Optional:
    surface_normals, surface_areas

Channels are assumed to be:
    0: pressure
    1: wss_x
    2: wss_y
    3: wss_z

This script reads qhat.json produced by cp_compare_global_point_case.py and writes
one VTP file per test case per CP mode.

Example:
    python cp_write_vtp_global_point_case.py \
      --test-dir runs/debug/geotransolver_sigma_multisample/cp_pointwise_test \
      --qhat-json results/cp_compare_global_point_case/qhat.json \
      --out results/cp_compare_global_point_case/vtp \
      --modes global_abs point_sigma case_sigma \
      --files batch_00000.npz batch_00038.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # type: ignore
except Exception as e:  # pragma: no cover
    vtk = None
    _VTK_IMPORT_ERROR = e

CHANNELS = ["pressure", "wss_x", "wss_y", "wss_z"]
MODES = ["global_abs", "point_sigma", "case_sigma"]


def as_points(x: np.ndarray, last_dim: int | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array or (1,N,C), got shape={x.shape}")
    if last_dim is not None and x.shape[1] < last_dim:
        raise ValueError(f"Expected at least {last_dim} columns, got shape={x.shape}")
    return x


def load_qhats(qhat_json: Path) -> Dict[str, np.ndarray]:
    with open(qhat_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    qobj = obj.get("qhat", obj)
    out: Dict[str, np.ndarray] = {}

    for mode in MODES:
        if mode not in qobj:
            continue
        m = qobj[mode]
        if isinstance(m, dict):
            out[mode] = np.asarray([float(m[ch]) for ch in CHANNELS], dtype=np.float32)
        elif isinstance(m, (list, tuple)):
            if len(m) < 4:
                raise ValueError(f"qhat for {mode} has length {len(m)}, expected 4")
            out[mode] = np.asarray(m[:4], dtype=np.float32)
        else:
            raise TypeError(f"Unsupported qhat format for mode {mode}: {type(m)}")

    if not out:
        raise ValueError(f"No qhat found in {qhat_json}. Expected keys: {MODES}")
    return out


def list_test_files(test_dir: Path, files: List[str] | None, max_cases: int | None) -> List[Path]:
    if files:
        out = []
        for name in files:
            p = Path(name)
            if not p.is_absolute():
                p = test_dir / name
            if not p.exists():
                raise FileNotFoundError(p)
            out.append(p)
    else:
        out = sorted(test_dir.glob("*.npz"))

    if not out:
        raise FileNotFoundError(f"No .npz files found in {test_dir}")
    if max_cases is not None:
        out = out[:max_cases]
    return out


def add_array(point_data, name: str, arr: np.ndarray, dtype=np.float32) -> None:
    arr = np.asarray(arr)
    arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
    vtk_arr = numpy_to_vtk(arr, deep=True)
    vtk_arr.SetName(name)
    point_data.AddArray(vtk_arr)


def add_vertices(poly, n: int) -> None:
    # One vertex cell per point. This makes ParaView render dense point clouds reliably.
    connectivity = np.arange(n, dtype=np.int64)
    offsets = np.arange(n + 1, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    poly.SetVerts(cells)


def write_vtp(
    out_path: Path,
    points_np: np.ndarray,
    arrays: Dict[str, np.ndarray],
    add_vertex_cells: bool = True,
    binary: bool = True,
) -> None:
    if vtk is None:
        raise ImportError(
            "Python package 'vtk' is required to write .vtp files. "
            f"Original import error: {_VTK_IMPORT_ERROR}"
        )

    points_np = np.ascontiguousarray(points_np.astype(np.float32, copy=False))
    n = points_np.shape[0]

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points_np, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    if add_vertex_cells:
        add_vertices(poly, n)

    pd = poly.GetPointData()
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(f"Array {name} has first dim {arr.shape[0]}, expected {n}")
        if arr.dtype == np.bool_:
            add_array(pd, name, arr.astype(np.uint8), dtype=np.uint8)
        elif arr.dtype == np.uint8:
            add_array(pd, name, arr, dtype=np.uint8)
        else:
            add_array(pd, name, arr, dtype=np.float32)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    if binary:
        writer.SetDataModeToBinary()
        if hasattr(writer, "SetCompressorTypeToZLib"):
            writer.SetCompressorTypeToZLib()
    else:
        writer.SetDataModeToAscii()

    ok = writer.Write()
    if ok != 1:
        raise RuntimeError(f"Failed to write {out_path}")


def compute_half_width(mode: str, qhat: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if mode == "global_abs":
        return np.broadcast_to(qhat.reshape(1, 4), sigma.shape).astype(np.float32, copy=False)
    if mode in ("point_sigma", "case_sigma"):
        return (qhat.reshape(1, 4) * sigma).astype(np.float32, copy=False)
    raise ValueError(f"Unknown mode: {mode}")


def make_arrays_for_mode(
    pred: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    half_width: np.ndarray,
    include_bounds: bool,
    include_pred_target: bool,
    include_wss: bool,
) -> Dict[str, np.ndarray]:
    lower = pred - half_width
    upper = pred + half_width
    width = 2.0 * half_width
    abs_err = np.abs(target - pred)
    covered = ((target >= lower) & (target <= upper)).astype(np.uint8)

    arrays: Dict[str, np.ndarray] = {}

    # Pressure scalar arrays.
    if include_pred_target:
        arrays["PredPressure"] = pred[:, 0]
        arrays["TargetPressure"] = target[:, 0]
    arrays["SigmaPressure"] = sigma[:, 0]
    arrays["AbsErrorPressure"] = abs_err[:, 0]
    arrays["CPWidthPressure"] = width[:, 0]
    arrays["CPCoveredPressure"] = covered[:, 0]
    if include_bounds:
        arrays["CPLowerPressure"] = lower[:, 0]
        arrays["CPUpperPressure"] = upper[:, 0]

    if include_wss:
        # WSS vector arrays. ParaView can show X/Y/Z components from these.
        if include_pred_target:
            arrays["PredWSS"] = pred[:, 1:4]
            arrays["TargetWSS"] = target[:, 1:4]
        arrays["SigmaWSS"] = sigma[:, 1:4]
        arrays["AbsErrorWSS"] = abs_err[:, 1:4]
        arrays["CPWidthWSS"] = width[:, 1:4]
        arrays["CPCoveredWSS"] = covered[:, 1:4]
        if include_bounds:
            arrays["CPLowerWSS"] = lower[:, 1:4]
            arrays["CPUpperWSS"] = upper[:, 1:4]

    arrays["CPCoveredAll4"] = np.all(covered == 1, axis=1).astype(np.uint8)

    return arrays


def process_file(
    npz_path: Path,
    out_dir: Path,
    qhats: Dict[str, np.ndarray],
    modes: Iterable[str],
    eps: float,
    stride: int,
    include_bounds: bool,
    include_pred_target: bool,
    include_wss: bool,
    add_vertex_cells: bool,
    binary: bool,
) -> None:
    print(f"[INFO] Loading {npz_path}")
    with np.load(npz_path, allow_pickle=True) as d:
        pred = as_points(d["pred"], 4)[:, :4].astype(np.float32, copy=False)
        target = as_points(d["target"], 4)[:, :4].astype(np.float32, copy=False)
        sigma = as_points(d["sigma"], 4)[:, :4].astype(np.float32, copy=False)
        points = as_points(d["surface_mesh_centers"], 3)[:, :3].astype(np.float32, copy=False)
        normals = None
        areas = None
        if "surface_normals" in d:
            normals = as_points(d["surface_normals"], 3)[:, :3].astype(np.float32, copy=False)
        if "surface_areas" in d:
            areas = np.asarray(d["surface_areas"]).reshape(-1).astype(np.float32, copy=False)

    sigma = np.maximum(sigma, eps)

    if stride > 1:
        sl = slice(None, None, stride)
        pred = pred[sl]
        target = target[sl]
        sigma = sigma[sl]
        points = points[sl]
        if normals is not None:
            normals = normals[sl]
        if areas is not None and areas.shape[0] == pred.shape[0] * stride:
            areas = areas[sl]
        elif areas is not None and areas.shape[0] != pred.shape[0]:
            areas = None

    base = npz_path.stem

    for mode in modes:
        if mode not in qhats:
            raise KeyError(f"Mode '{mode}' not found in qhat file. Available: {list(qhats.keys())}")

        half_width = compute_half_width(mode, qhats[mode], sigma)
        arrays = make_arrays_for_mode(
            pred, target, sigma, half_width,
            include_bounds=include_bounds,
            include_pred_target=include_pred_target,
            include_wss=include_wss,
        )
        if normals is not None:
            arrays["Normals"] = normals
        if areas is not None and areas.shape[0] == points.shape[0]:
            arrays["SurfaceArea"] = areas

        out_path = out_dir / mode / f"{base}_{mode}_cp.vtp"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Writing {out_path} | n_points={points.shape[0]}")
        write_vtp(
            out_path,
            points,
            arrays,
            add_vertex_cells=add_vertex_cells,
            binary=binary,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--qhat-json", type=Path, required=True,
                    help="qhat.json produced by cp_compare_global_point_case.py")
    ap.add_argument("--out", type=Path, default=Path("results/cp_compare_global_point_case/vtp"))
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--files", nargs="*", default=None,
                    help="Specific .npz files to export, e.g. batch_00000.npz batch_00038.npz. Default: all.")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--stride", type=int, default=1,
                    help="Export every k-th point for lighter visualization. Use 1 for full resolution.")
    ap.add_argument("--no-bounds", action="store_true",
                    help="Do not write lower/upper arrays; reduces file size.")
    ap.add_argument("--no-pred-target", action="store_true",
                    help="Do not write pred/target arrays; reduces file size.")
    ap.add_argument("--pressure-only", action="store_true",
                    help="Only write pressure arrays, not WSS vector arrays; greatly reduces file size.")
    ap.add_argument("--no-vertex-cells", action="store_true",
                    help="Write only points without explicit vertex cells. Smaller, but may render less reliably.")
    ap.add_argument("--ascii", action="store_true", help="Write ASCII VTP. Not recommended for large point clouds.")
    args = ap.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)
    qhats = load_qhats(args.qhat_json)
    test_files = list_test_files(args.test_dir, args.files, args.max_cases)

    print(f"[INFO] test files : {len(test_files)}")
    print(f"[INFO] modes      : {args.modes}")
    print(f"[INFO] stride     : {args.stride}")
    print(f"[INFO] qhats      :")
    for mode, q in qhats.items():
        print(f"  {mode}: " + ", ".join(f"{ch}={q[i]:.6g}" for i, ch in enumerate(CHANNELS)))

    for f in test_files:
        process_file(
            f,
            args.out,
            qhats,
            args.modes,
            eps=args.eps,
            stride=args.stride,
            include_bounds=not args.no_bounds,
            include_pred_target=not args.no_pred_target,
            include_wss=not args.pressure_only,
            add_vertex_cells=not args.no_vertex_cells,
            binary=not args.ascii,
        )

    print(f"[OK] VTP files written under: {args.out}")


if __name__ == "__main__":
    main()
