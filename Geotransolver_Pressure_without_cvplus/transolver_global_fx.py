# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Patches for PhysicsNeMo ``TransolverDataPipe`` used by this repo.

1. **Global ``fx``** when zarr has ``global_params_*`` instead of ``air_density`` /
   ``stream_velocity`` (library only builds ``fx`` from the latter).

2. **Fixed-seed surface subsample** for overfit/sanity: when enabled, subsample
   ``k`` points via ``torch.randperm`` with a fixed generator seed (not ``0..k-1``),
   so points cover the surface more broadly than a single local patch.
"""

from __future__ import annotations

import functools

import torch

_patched_call = False
_patched_preprocess = False
_patched_fourier_preprocess = False
_deterministic_subsample: bool = False

_orig_preprocess_surface = None
_orig_preprocess_fourier = None


def set_overfit_deterministic_subsample(enabled: bool) -> None:
    """When True, surface ``resolution`` subsample uses fixed-seed ``torch.randperm`` (see patch)."""
    global _deterministic_subsample
    _deterministic_subsample = bool(enabled)


def _batch_global_vec(t: torch.Tensor) -> torch.Tensor:
    x = t.to(torch.float32)
    while x.dim() > 2:
        x = x.squeeze(1)
    return x


def patch_transolver_datapipe_global_fx() -> None:
    """Idempotent: monkey-patch ``TransolverDataPipe.__call__`` once."""
    global _patched_call
    if _patched_call:
        return

    from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe

    orig_call = TransolverDataPipe.__call__

    @functools.wraps(orig_call)
    def call_with_global_fx(self, data_dict):
        outputs = self.process_data(data_dict)
        if "fx" not in outputs and (
            "global_params_reference" in data_dict
            and "global_params_values" in data_dict
        ):
            emb = outputs["embeddings"]
            dev = emb.device
            ref = _batch_global_vec(data_dict["global_params_reference"]).to(dev)
            val = _batch_global_vec(data_dict["global_params_values"]).to(dev)
            fx = torch.cat([ref, val], dim=-1)
            if getattr(self.config, "broadcast_global_features", True):
                fx = fx.reshape(1, -1).expand(emb.shape[0], -1)
            outputs["fx"] = fx
            for key in outputs.keys():
                if isinstance(outputs[key], list):
                    outputs[key] = [item.unsqueeze(0) for item in outputs[key]]
                else:
                    outputs[key] = outputs[key].unsqueeze(0)
            return outputs
        return orig_call(self, data_dict)

    TransolverDataPipe.__call__ = call_with_global_fx  # type: ignore[method-assign]
    _patched_call = True


def patch_transolver_preprocess_surface_for_overfit() -> None:
    """Wrap ``preprocess_surface_data`` for fixed-seed subsample before library preprocess."""
    global _patched_preprocess, _orig_preprocess_surface
    if _patched_preprocess:
        return

    from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe

    _orig_preprocess_surface = TransolverDataPipe.preprocess_surface_data

    _SUB_KEYS = (
        "surface_mesh_centers",
        "surface_normals",
        "surface_fields",
        "surface_areas",
    )

    def _patched_preprocess_surface_data(
        self,
        data_dict,
        center_of_mass=None,
        scale_factor=None,
    ):
        if (
            not _deterministic_subsample
            or self.config.resolution is None
            or self.config.model_type not in ("surface", "combined")
        ):
            return _orig_preprocess_surface(
                self, data_dict, center_of_mass, scale_factor
            )

        positions = data_dict["surface_mesh_centers"]
        n = int(positions.shape[0])
        k = min(int(self.config.resolution), n)
        g = torch.Generator(device=positions.device)
        g.manual_seed(1234)
        idx = torch.randperm(n, device=positions.device, generator=g)[:k]

        dd = dict(data_dict)
        for key in _SUB_KEYS:
            if key not in dd or dd[key] is None:
                continue
            t = dd[key]
            if not hasattr(t, "shape") or t.shape[0] != n:
                continue
            dd[key] = t[idx]

        old_res = self.config.resolution
        try:
            setattr(self.config, "resolution", None)
        except (AttributeError, TypeError):
            return _orig_preprocess_surface(
                self, data_dict, center_of_mass, scale_factor
            )
        try:
            return _orig_preprocess_surface(
                self, dd, center_of_mass, scale_factor
            )
        finally:
            try:
                setattr(self.config, "resolution", old_res)
            except (AttributeError, TypeError):
                pass

    TransolverDataPipe.preprocess_surface_data = (  # type: ignore[method-assign]
        _patched_preprocess_surface_data
    )
    _patched_preprocess = True


def patch_transolver_preprocess_fourier_surface() -> None:
    """Append multi-band Fourier features to surface embeddings when config enables it.

    Outermost wrapper: call this **after**
    ``patch_transolver_preprocess_surface_for_overfit`` so the call chain is
    (optional fixed-seed subsample) → library ``preprocess_surface_data`` → Fourier
    augments ``embeddings`` in this outer wrapper.

    Expects base surface embeddings of dim 6 (xyz + normals). Output dim 30.

    PhysicsNeMo recent versions return a **dict** (``embeddings``, ``fields``, …);
    older code assumed a 4-tuple. This wrapper handles both.
    """
    global _patched_fourier_preprocess, _orig_preprocess_fourier
    if _patched_fourier_preprocess:
        return

    from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe

    from preprocess import append_fourier_surface_embeddings

    _orig_preprocess_fourier = TransolverDataPipe.preprocess_surface_data

    def _wrapped_preprocess_surface_data(
        self,
        data_dict,
        center_of_mass=None,
        scale_factor=None,
    ):
        out = _orig_preprocess_fourier(
            self, data_dict, center_of_mass, scale_factor
        )
        if not getattr(self.config, "use_fourier_surface_embeddings", False):
            return out
        if self.config.model_type not in ("surface", "combined"):
            return out

        if isinstance(out, dict):
            emb = out.get("embeddings")
            if not torch.is_tensor(emb) or emb.dim() < 2 or emb.shape[-1] != 6:
                return out
            out = dict(out)
            out["embeddings"] = append_fourier_surface_embeddings(emb)
            return out

        if not isinstance(out, (tuple, list)) or len(out) != 4:
            return out
        nf, emb, targets, others = out
        if not torch.is_tensor(emb) or emb.shape[-1] != 6:
            return out
        emb2 = append_fourier_surface_embeddings(emb)
        return nf, emb2, targets, others

    TransolverDataPipe.preprocess_surface_data = (  # type: ignore[method-assign]
        _wrapped_preprocess_surface_data
    )
    _patched_fourier_preprocess = True
