# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Core python imports:
import os

import repo_env

repo_env.ensure_repo_local_caches()

import math
import time
from pathlib import Path
from typing import Literal, Any, Callable
import collections
from contextlib import nullcontext

from collections.abc import Sequence

# Configuration:
import hydra
import omegaconf
from omegaconf import DictConfig

# Pytorch imports:
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

# For metrics and model printouts:
from tabulate import tabulate
import torchinfo

# For loading dataset stats:
import numpy as np

# Physicsnemo imports ...
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import profile, Profiler
from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
    TransolverDataPipe,
)

# Local folder imports for this example
from metrics import metrics_fn

from physicsnemo.nn import collect_concrete_dropout_losses, get_concrete_dropout_rates

# tensorwise is to handle single-point-cloud or multi-point-cloud running.
# it's a decorator that will automatically unzip one or more of a list of tensors,
# run the funtcion, and rezip the results.
from utils import tensorwise

# Transformer Engine 在本仓库中禁用（不探测、不加载；float8 训练将直接报错提示）。
TE_AVAILABLE = False
te, Format, DelayedScaling = None, None, None

# This will go away when checkpointing is refined further below:
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.

    Note:
        This will get upstreamed to physicsnemo shortly.  Don't count on this
        class existing here in the future!

        In other words, this is already marked for deprecation!
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")

        self.optimizers = optimizers

        # Collect parameter groups from all optimizers. We pass an empty
        # *defaults* dict because hyper-parameters are managed by the inner
        # optimizers, not this wrapper.
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        """Nullify gradients"""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        """Execute a single optimization step across all wrapped optimizers."""
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        """Return combined state dict from all wrapped optimizers."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        """Restore state dicts to all wrapped optimizers."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)

        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


def get_autocast_context(precision: str) -> nullcontext:
    """
    Returns the appropriate autocast context for mixed precision training.

    Args:
        precision (str): The desired precision. Supported values are "float16", "bfloat16", or any other string for no autocast.

    Returns:
        Context manager: An autocast context for the specified precision, or a nullcontext if precision is not recognized.
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8" and TE_AVAILABLE:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


@tensorwise
def cast_precisions(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """
    Casts the tensors to the specified precision.

    We are careful to take either a tensor or list of tensors, and return the same format.
    """

    match precision:
        case "float16":
            dtype = torch.float16
        case "bfloat16":
            dtype = torch.bfloat16
        case _:
            dtype = None

    if dtype is not None:
        return tensor.to(dtype)
    else:
        return tensor


@tensorwise
def pad_input_for_fp8(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    geometry: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pads the input features tensor so that the concatenated feature and embedding dimension is a multiple of 16,
    which is required for FP8 operations.  Only the features is updated.

    Args:
        features (torch.Tensor): The input features tensor of shape (..., feature_dim).
        embeddings (torch.Tensor): The embeddings tensor of shape (..., embedding_dim).

    Returns:
        torch.Tensor: The padded features tensor, so that (features.shape[-1] + embeddings.shape[-1]) is a multiple of 16.
    """
    fx_dim = features.shape[-1] + embeddings.shape[-1]
    if fx_dim % 16 != 0:
        pad_size = 16 - (fx_dim % 16)
        features = torch.nn.functional.pad(features, (0, pad_size))
        fx_dim = features.shape[-1] + embeddings.shape[-1]

    if geometry is not None:
        geometry_dim = geometry.shape[-1] if geometry is not None else 0
        if geometry_dim % 16 != 0:
            pad_size = 16 - (geometry_dim % 16)
            geometry = torch.nn.functional.pad(geometry, (0, pad_size))
            geometry_dim = geometry.shape[-1]

    return features, geometry


class _OverfitFixedBatchTrainLoader:
    """Yields one cached batch every step; forwards other attributes to the base loader."""

    __slots__ = ("_base", "_fixed", "_epoch_length")

    def __init__(self, base_loader: Any, fixed_batch: dict, epoch_length: int) -> None:
        object.__setattr__(self, "_base", base_loader)
        object.__setattr__(self, "_fixed", fixed_batch)
        object.__setattr__(self, "_epoch_length", epoch_length)

    def __len__(self) -> int:
        return int(object.__getattribute__(self, "_epoch_length"))

    def __iter__(self):
        fixed = object.__getattribute__(self, "_fixed")
        n = int(object.__getattribute__(self, "_epoch_length"))
        for _ in range(n):
            yield fixed

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base"), name)


def _move_batch_tensors_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list):
            out[k] = [
                x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                for x in v
            ]
        else:
            out[k] = v
    return out


def stl_center_of_mass_from_batch(batch: dict) -> torch.Tensor | None:
    """Same COM as ``preprocess.preprocess_surface_data`` when STL keys exist."""
    if "stl_centers" not in batch or "stl_areas" not in batch:
        return None
    centers = batch["stl_centers"].float()
    sizes = batch["stl_areas"].float()
    if centers.dim() == 2:
        centers = centers.unsqueeze(0)
        sizes = sizes.unsqueeze(0)
    if sizes.dim() == 2:
        sizes = sizes.unsqueeze(-1)
    twp = (sizes * centers).sum(dim=1, keepdim=True)
    ts = sizes.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return twp / ts


def maybe_center_geometry_with_stl_com(
    geometry: torch.Tensor | None,
    batch: dict,
    *,
    enabled: bool,
) -> torch.Tensor | None:
    """Subtract STL COM from geometry xyz so it matches centered ``local_positions``."""
    if not enabled or geometry is None:
        return geometry
    com = stl_center_of_mass_from_batch(batch)
    if com is None:
        return geometry
    com = com.to(device=geometry.device, dtype=geometry.dtype)
    xyz = geometry[..., :3] - com
    if geometry.shape[-1] > 3:
        return torch.cat([xyz, geometry[..., 3:]], dim=-1)
    return xyz


def global_embedding_for_geotransolver(
    features: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Collapse broadcast global features from (B, N, F) to (B, 1, F) when constant over N."""
    gf = features
    if gf.dim() == 3 and gf.shape[1] == embeddings.shape[1]:
        ref = gf[:, :1, :].float()
        diff = (gf.float() - ref).abs().max().item()
        tol = 1e-4 if gf.dtype in (torch.float16, torch.bfloat16) else 1e-8
        if diff < tol:
            gf = gf[:, :1, :]
        else:
            raise RuntimeError(
                "global features look point-wise (B,N,F with N==embeddings) but are "
                f"not constant over N (max abs diff={diff})."
            )
    return gf


def prepare_global_features_for_geotransolver(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    datapipe: TransolverDataPipe,
) -> torch.Tensor:
    """Optional normalize then collapse broadcast global to ``(B, 1, 6)`` for GeoTransolver."""
    g = features
    if bool(getattr(datapipe.config, "normalize_global_features", False)):
        if g.shape[-1] != 6:
            raise RuntimeError(
                "normalize_global_features: expected fx last dim 6 "
                f"(ref+val), got {g.shape[-1]}"
            )
        g = normalize_global_features(g)
    return global_embedding_for_geotransolver(g, embeddings)


@tensorwise
def unpad_output_for_fp8(
    outputs: torch.Tensor, output_pad_size: int | None
) -> torch.Tensor:
    """
    Removes the padding from the output tensor that was added for FP8 compatibility.

    Args:
        outputs (torch.Tensor): The output tensor of shape (..., output_dim + pad_size) if padded.
        output_pad_size (int | None): The number of padded elements to remove from the last dimension. If None, no unpadding is performed.

    Returns:
        torch.Tensor: The unpadded output tensor.
    """
    # Remove the padded outputs:
    if output_pad_size is not None:
        return outputs[:, :, :-output_pad_size]
    return outputs


@tensorwise
def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean-field MSE. If the model returns (mean, sigma), only the mean is used.
    """
    mean, _ = split_mean_sigma(outputs)
    return torch.nn.functional.mse_loss(mean, targets)


def split_mean_sigma(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (mean_prediction, sigma_prediction_or_none).

    The sigma-head wrapper returns ``(mean, sigma)``. This helper keeps the rest of
    the training / inference code compatible with plain GeoTransolver outputs.
    """
    if isinstance(outputs, dict):
        mean = outputs.get("mean", outputs.get("y_hat", outputs.get("prediction")))
        sigma = outputs.get("sigma", outputs.get("sigma_hat", None))
        if mean is None:
            raise RuntimeError(
                "Model returned a dict but no mean/y_hat/prediction key was found."
            )
        return mean, sigma
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise RuntimeError("Model returned an empty tuple/list.")
        mean = outputs[0]
        sigma = outputs[1] if len(outputs) > 1 else None
        return mean, sigma
    return outputs, None


def sigma_smoothness_loss(
    sigma: torch.Tensor,
    xyz: torch.Tensor,
    *,
    k: int = 8,
    max_points: int = 2048,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Batch-local KNN smoothness loss on log(sigma).

    This is intentionally *batch local*, not a global calibration-set KNN. It is
    used only to prevent noisy point-to-point sigma oscillations during training.
    """
    if sigma is None or sigma.numel() == 0:
        return torch.zeros((), device=xyz.device, dtype=xyz.dtype)

    B, N, C = sigma.shape
    if N <= 1 or k <= 0:
        return torch.zeros((), device=sigma.device, dtype=sigma.dtype)

    n_use = min(int(max_points), N) if max_points is not None and max_points > 0 else N
    if n_use < N:
        # Same random subset for all channels within this forward; stochastic but cheap.
        idx = torch.randperm(N, device=xyz.device)[:n_use]
        xyz_use = xyz[:, idx, :]
        log_sigma = torch.log(sigma[:, idx, :].clamp_min(eps))
    else:
        xyz_use = xyz
        log_sigma = torch.log(sigma.clamp_min(eps))

    k_eff = min(int(k), n_use - 1)
    if k_eff <= 0:
        return torch.zeros((), device=sigma.device, dtype=sigma.dtype)

    # Compute distances in fp32 for stability even under autocast.
    with torch.no_grad():
        dist_mat = torch.cdist(xyz_use.float(), xyz_use.float())
        nn_idx = dist_mat.topk(k=k_eff + 1, dim=-1, largest=False).indices[:, :, 1:]

    # Gather neighbor log_sigma: (B, N, k, C)
    B2, N2, C2 = log_sigma.shape
    idx_exp = nn_idx.unsqueeze(-1).expand(B2, N2, k_eff, C2)
    expanded = log_sigma.unsqueeze(1).expand(B2, N2, N2, C2)
    nbr = torch.gather(expanded, 2, idx_exp)
    center = log_sigma.unsqueeze(2)
    return (center - nbr).pow(2).mean()


def field_sigma_loss(
    outputs: Any,
    targets: torch.Tensor,
    local_positions: torch.Tensor,
    training_cfg: Any | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Compute field MSE plus optional residual-scale-head losses.

    The scale-head target is the detached absolute residual. Therefore the sigma
    loss trains uncertainty scale without directly pulling the mean prediction.
    All quantities are in the normalized target space; CP intervals in physical
    units should multiply sigma by the same field std / physical scaling used for
    predictions.
    """
    mean, sigma = split_mean_sigma(outputs)
    mean_loss = F.mse_loss(mean, targets)

    parts: dict[str, torch.Tensor] = {"loss/mean": mean_loss.detach()}
    total = mean_loss

    use_sigma = bool(getattr(training_cfg, "use_sigma_loss", False)) if training_cfg is not None else False
    if use_sigma and sigma is not None:
        eps = float(getattr(training_cfg, "sigma_eps", 1.0e-6))
        w_scale = float(getattr(training_cfg, "sigma_loss_weight", 0.05))
        w_smooth = float(getattr(training_cfg, "sigma_smooth_weight", 0.0))

        sigma = sigma.clamp_min(eps)
        residual_floor = float(getattr(training_cfg, "sigma_target_floor", 1.0e-4))
        residual = (targets - mean.detach()).abs().clamp_min(residual_floor)
        pred_log = torch.log(sigma + eps)
        target_log = torch.log(residual + eps)

        scale_loss = F.smooth_l1_loss(pred_log, target_log)
        total = total + w_scale * scale_loss
        parts["loss/sigma_scale"] = scale_loss.detach()

        if w_smooth > 0:
            smooth_loss = sigma_smoothness_loss(
                sigma,
                local_positions,
                k=int(getattr(training_cfg, "sigma_smooth_k", 8)),
                max_points=int(getattr(training_cfg, "sigma_smooth_max_points", 2048)),
                eps=eps,
            )
            total = total + w_smooth * smooth_loss
            parts["loss/sigma_smooth"] = smooth_loss.detach()

        with torch.no_grad():
            parts["sigma/mean"] = sigma.mean().detach()
            parts["sigma/std"] = sigma.std(unbiased=False).detach()
            parts["sigma/min"] = sigma.amin().detach()
            parts["sigma/max"] = sigma.amax().detach()

    return total, parts, mean, sigma


_SURFACE_FIELD_CHANNELS = ("pressure", "wss_x", "wss_y", "wss_z")


def _is_overfit_mode(cfg: DictConfig) -> bool:
    ov = getattr(cfg.data, "overfit_n_samples", None)
    return ov is not None and int(ov) > 0


def _should_log_field_stats(cfg: DictConfig, rank: int) -> bool:
    if rank != 0:
        return False
    if getattr(cfg.training, "debug_forward_io_stats", False):
        return True
    return _is_overfit_mode(cfg)


def _channel_names(n_channels: int) -> tuple[str, ...]:
    if n_channels <= len(_SURFACE_FIELD_CHANNELS):
        return _SURFACE_FIELD_CHANNELS[:n_channels]
    return tuple(f"ch{i}" for i in range(n_channels))


def _log_per_channel_tensor(name: str, t: torch.Tensor, prefix: str = "") -> None:
    channels = _channel_names(t.shape[-1])
    mean_c = t.mean(dim=(0, 1)).detach().cpu().float().numpy()
    std_c = t.std(dim=(0, 1), unbiased=False).detach().cpu().float().numpy()
    min_c = t.amin(dim=(0, 1)).detach().cpu().float().numpy()
    max_c = t.amax(dim=(0, 1)).detach().cpu().float().numpy()
    print(f"[debug]{prefix} {name} (N={int(t.shape[1])}, C={int(t.shape[-1])})")
    for i, ch in enumerate(channels):
        print(
            f"  [{ch}] mean={mean_c[i]:.6g} std={std_c[i]:.6g} "
            f"min={min_c[i]:.6g} max={max_c[i]:.6g}"
        )


def _log_surface_normalization_factors(surface_factors: dict | None) -> None:
    if surface_factors is None:
        return
    mean = surface_factors["mean"].detach().cpu().float().numpy()
    std = surface_factors["std"].detach().cpu().float().numpy()
    print("[debug] surface_fields_normalization:")
    for i, ch in enumerate(_channel_names(len(std))):
        print(f"  [{ch}] mean={mean[i]:.6g} std={std[i]:.6g}")


def _log_forward_io_tensor_stats(
    outputs: torch.Tensor | tuple | list,
    targets: torch.Tensor | tuple | list,
    sigma: torch.Tensor | None = None,
    loss_parts: dict[str, torch.Tensor] | None = None,
    prefix: str = "",
) -> None:
    """Print per-channel mean/std/min/max, std ratios, sigma and loss parts."""
    with torch.no_grad():
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            _log_per_channel_tensor("y_hat (mean)", outputs, prefix=prefix)
            _log_per_channel_tensor("targets", targets, prefix=prefix)

            out_std = outputs.std(dim=(0, 1), unbiased=False)
            tgt_std = targets.std(dim=(0, 1), unbiased=False).clamp_min(1e-12)
            ratio = (out_std / tgt_std).detach().cpu().float().numpy()
            print(f"[debug]{prefix} y_hat_std / target_std:")
            for i, ch in enumerate(_channel_names(outputs.shape[-1])):
                print(f"  [{ch}] ratio={ratio[i]:.6g}")

            residual = (targets - outputs).abs()
            _log_per_channel_tensor("|target - y_hat|", residual, prefix=prefix)

            if sigma is not None:
                _log_per_channel_tensor("sigma_hat", sigma, prefix=prefix)
                score = residual / sigma.clamp_min(1e-12)
                _log_per_channel_tensor("|residual|/sigma", score, prefix=prefix)

            if loss_parts:
                print(f"[debug]{prefix} loss components:")
                for key in (
                    "loss/mean",
                    "loss/sigma_scale",
                    "loss/sigma_smooth",
                    "sigma/mean",
                    "sigma/std",
                    "sigma/min",
                    "sigma/max",
                ):
                    if key not in loss_parts:
                        continue
                    val = loss_parts[key]
                    print(f"  {key}: {float(val.detach().cpu()):.6g}")
        elif isinstance(outputs, (tuple, list)) and isinstance(targets, (tuple, list)):
            for i, (o, t) in enumerate(zip(outputs, targets)):
                if isinstance(o, torch.Tensor) and isinstance(t, torch.Tensor):
                    _log_forward_io_tensor_stats(
                        o, t, sigma=sigma, loss_parts=loss_parts, prefix=f"{prefix}[{i}]"
                    )
        else:
            print(
                "[debug] outputs/targets stats: unexpected types "
                f"{type(outputs)}, {type(targets)}"
            )


def normalize_global_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize global ref/val channels for stable concat with Fourier locals.

    ``features`` shape ``(..., 6)``: ``[ref_u, ref_rho, ref_p, val_u, val_rho, val_p]``.
    Returns 6 channels: ``val / scale`` (3) plus ``(val - ref) / |ref|`` (3).
    """
    ref = features[..., :3]
    val = features[..., 3:6]
    scale = torch.tensor(
        [30.0, 1.205, 101325.0],
        device=features.device,
        dtype=features.dtype,
    ).view(1, 1, 3)
    val_norm = val / scale
    rel = (val - ref) / ref.abs().clamp_min(1e-8)
    return torch.cat([val_norm, rel], dim=-1)


def _log_forward_input_stats(
    global_embedding: torch.Tensor,
    embeddings: torch.Tensor,
    geometry: torch.Tensor | None,
    local_positions: torch.Tensor,
) -> None:
    """Print input shape and per-channel std over (batch, point) for local/global sanity."""
    with torch.no_grad():
        print(
            "[debug] global_embedding shape/std:",
            tuple(global_embedding.shape),
            global_embedding.std(dim=(0, 1), unbiased=False).detach().cpu().float().numpy(),
        )
        print(
            "[debug] embeddings shape/std:",
            tuple(embeddings.shape),
            embeddings.std(dim=(0, 1)).detach().cpu().float().numpy(),
        )
        if geometry is not None:
            print(
                "[debug] geometry shape/std:",
                tuple(geometry.shape),
                geometry.std(dim=(0, 1)).detach().cpu().float().numpy(),
            )
        print(
            "[debug] local_positions std:",
            local_positions.std(dim=(0, 1)).detach().cpu().float().numpy(),
        )
        print(
            "[debug] local_positions min/max:",
            local_positions.amin(dim=(0, 1)).detach().cpu().float().numpy(),
            local_positions.amax(dim=(0, 1)).detach().cpu().float().numpy(),
        )
        if geometry is not None:
            print(
                "[debug] geometry min/max:",
                geometry.amin(dim=(0, 1)).detach().cpu().float().numpy(),
                geometry.amax(dim=(0, 1)).detach().cpu().float().numpy(),
            )


def _log_fx_split_stats(fx_in: torch.Tensor, embeddings: torch.Tensor) -> None:
    """Debug: global vs local channels in ``fx_in`` (global is constant over points; std≈0)."""
    with torch.no_grad():
        g = fx_in[..., :6]
        l = fx_in[..., 6:] if fx_in.shape[-1] > 6 else None

        print(
            "[debug] fx_global mean/std/min/max:",
            g.mean(dim=(0, 1)).detach().cpu().numpy(),
            g.std(dim=(0, 1), unbiased=False).detach().cpu().numpy(),
            g.amin(dim=(0, 1)).detach().cpu().numpy(),
            g.amax(dim=(0, 1)).detach().cpu().numpy(),
        )

        if l is not None:
            print(
                "[debug] fx_local mean/std/min/max:",
                l.mean(dim=(0, 1)).detach().cpu().numpy()[:10],
                l.std(dim=(0, 1), unbiased=False).detach().cpu().numpy()[:10],
                l.amin(dim=(0, 1)).detach().cpu().numpy()[:10],
                l.amax(dim=(0, 1)).detach().cpu().numpy()[:10],
            )

        print(
            "[debug] embeddings mean/std/min/max:",
            embeddings.mean(dim=(0, 1)).detach().cpu().numpy()[:10],
            embeddings.std(dim=(0, 1), unbiased=False).detach().cpu().numpy()[:10],
            embeddings.amin(dim=(0, 1)).detach().cpu().numpy()[:10],
            embeddings.amax(dim=(0, 1)).detach().cpu().numpy()[:10],
        )


def sync_use_fourier_datapipe_config(dataloader_or_pipe: Any, cfg: DictConfig) -> None:
    """Mirror Hydra ``data.*`` flags onto the PhysicsNeMo DataPipe ``config``.

    ``TransolverDataConfig`` may not copy arbitrary keys from Hydra; patches such as
    ``transolver_global_fx`` read ``self.config.use_fourier_surface_embeddings``. We
    also set ``concat_embedding_to_fx``, ``legacy_fx_embeddings_only``,
    ``normalize_global_features``, ``include_geometry``, and
    ``center_geometry_with_stl_com`` so the runtime ``config`` matches training
    (same defaults as ``forward_pass``).
    """
    flags = {
        "use_fourier_surface_embeddings": bool(
            getattr(cfg.data, "use_fourier_surface_embeddings", False)
        ),
        "concat_embedding_to_fx": bool(
            getattr(cfg.data, "concat_embedding_to_fx", False)
        ),
        "legacy_fx_embeddings_only": bool(
            getattr(cfg.data, "legacy_fx_embeddings_only", False)
        ),
        "normalize_global_features": bool(
            getattr(cfg.data, "normalize_global_features", False)
        ),
        "include_geometry": bool(getattr(cfg.data, "include_geometry", False)),
        "center_geometry_with_stl_com": bool(
            getattr(cfg.data, "center_geometry_with_stl_com", True)
        ),
    }
    objs: list[Any] = [dataloader_or_pipe]
    ds = getattr(dataloader_or_pipe, "dataset", None)
    if ds is not None:
        objs.append(ds)

    for obj in objs:
        if obj is None:
            continue
        node = getattr(obj, "config", None)
        if node is None:
            continue
        if omegaconf.OmegaConf.is_config(node):
            try:
                omegaconf.OmegaConf.set_struct(node, False)
            except (ValueError, TypeError):
                pass
        for key, val in flags.items():
            try:
                setattr(node, key, val)
            except Exception:
                try:
                    node[key] = val
                except Exception:
                    pass


def forward_pass(
    batch: dict,
    model: torch.nn.Module,
    precision: str,
    output_pad_size: int | None,
    dist_manager: DistributedManager,
    data_mode: Literal["surface", "volume"],
    datapipe: TransolverDataPipe,
    debug_io_stats: bool = False,
    debug_log_prefix: str = "",
    legacy_transolver_forward: bool = False,
    concat_embedding_to_fx: bool = False,
    legacy_fx_embeddings_only: bool = False,
    training_cfg: Any | None = None,
):
    """
    Run the forward pass of the model for one batch, including metrics and loss calculation.

    When ``legacy_transolver_forward`` is True, calls ``model(fx=..., embedding=embeddings)``.
    If ``concat_embedding_to_fx`` is True, ``fx`` is ``torch.cat([g, embeddings], dim=-1)``
    when point counts match, where ``g`` is ``features`` or optionally
    ``normalize_global_features(features)`` if ``datapipe.config.normalize_global_features``.
    If ``legacy_fx_embeddings_only`` is True, ``fx`` is set to
    ``embeddings`` only (mutually exclusive with concat in practice; embeddings-only wins here).

    Otherwise uses GeoTransolver-style ``global_embedding`` / ``local_embedding`` /
    ``local_positions`` and optional ``geometry``.

    """

    features = batch["fx"]
    embeddings = batch["embeddings"]
    targets = batch["fields"]

    # Cast precisions:
    features = cast_precisions(features, precision=precision)
    embeddings = cast_precisions(embeddings, precision=precision)
    if "geometry" in batch.keys():
        geometry = cast_precisions(batch["geometry"], precision=precision)
    else:
        geometry = None

    all_metrics = {}
    if datapipe.config.model_type == "combined":
        # This is hard coded for Typhon.  If you have more point clouds,
        # your mileage may vary.
        modes = ["surface", "volume"]
    elif datapipe.config.model_type == "surface":
        modes = [
            "surface",
        ]
    elif datapipe.config.model_type == "volume":
        modes = [
            "volume",
        ]

    local_positions = embeddings[:, :, :3]

    with get_autocast_context(precision):
        if legacy_transolver_forward:
            fx_in = features
            emb_only = bool(legacy_fx_embeddings_only)
            concat_fx = bool(concat_embedding_to_fx)
            if emb_only:
                fx_in = embeddings
            elif concat_fx:
                if features.dim() == 3 and features.shape[1] == embeddings.shape[1]:
                    g = features
                    if bool(
                        getattr(datapipe.config, "normalize_global_features", False)
                    ):
                        if features.shape[-1] != 6:
                            raise RuntimeError(
                                "normalize_global_features: expected fx last dim 6 "
                                f"(ref+val), got {features.shape[-1]}"
                            )
                        g = normalize_global_features(g)
                    fx_in = torch.cat([g, embeddings], dim=-1)
                else:
                    raise RuntimeError(
                        "concat_embedding_to_fx: need matching N between features and "
                        f"embeddings; got features {tuple(features.shape)}, "
                        f"embeddings {tuple(embeddings.shape)}"
                    )

            if precision == "float8" and TE_AVAILABLE:
                fx_in, geometry = pad_input_for_fp8(fx_in, embeddings, geometry)

            if debug_io_stats and dist_manager.rank == 0:
                _log_fx_split_stats(fx_in, embeddings)
                _log_forward_input_stats(
                    fx_in, embeddings, None, local_positions
                )
            outputs = model(fx=fx_in, embedding=embeddings)
        else:
            if precision == "float8" and TE_AVAILABLE:
                features, geometry = pad_input_for_fp8(
                    features, embeddings, geometry
                )

            center_geo = bool(
                getattr(datapipe.config, "center_geometry_with_stl_com", True)
            )
            geometry = maybe_center_geometry_with_stl_com(
                geometry, batch, enabled=center_geo
            )
            if bool(getattr(datapipe.config, "include_geometry", False)) and geometry is None:
                raise RuntimeError(
                    "data.include_geometry=true but batch has no 'geometry'. "
                    "Ensure stl_coordinates/stl_faces are in data_keys and zarr."
                )

            global_features = prepare_global_features_for_geotransolver(
                features, embeddings, datapipe
            )

            forward_kw: dict = {
                "global_embedding": global_features,
                "local_embedding": embeddings,
                "local_positions": local_positions,
            }
            if geometry is not None:
                forward_kw["geometry"] = geometry

            if debug_io_stats and dist_manager.rank == 0:
                _log_forward_input_stats(
                    global_features, embeddings, geometry, local_positions
                )

            outputs = model(**forward_kw)

        outputs = unpad_output_for_fp8(outputs, output_pad_size)
        full_loss, loss_parts, mean_outputs, sigma_outputs = field_sigma_loss(
            outputs,
            targets,
            local_positions,
            training_cfg=training_cfg,
        )
        all_metrics.update(loss_parts)
        if debug_io_stats and dist_manager.rank == 0:
            prefix = f" {debug_log_prefix}" if debug_log_prefix else ""
            _log_forward_io_tensor_stats(
                mean_outputs,
                targets,
                sigma=sigma_outputs,
                loss_parts=loss_parts,
                prefix=prefix,
            )

        if datapipe.config.model_type == "combined":
            for i, mode in enumerate(modes):
                all_metrics[f"loss/{mode}"] = full_loss.detach()
        else:
            all_metrics[f"loss/{modes[0]}"] = full_loss.detach()

    air_density = batch["air_density"] if "air_density" in batch.keys() else None
    stream_velocity = (
        batch["stream_velocity"] if "stream_velocity" in batch.keys() else None
    )

    unscaled_outputs = tensorwise(datapipe.unscale_model_targets)(
        mean_outputs,
        air_density=air_density,
        stream_velocity=stream_velocity,
        factor_type=modes,
    )
    unscaled_targets = tensorwise(datapipe.unscale_model_targets)(
        targets,
        air_density=air_density,
        stream_velocity=stream_velocity,
        factor_type=modes,
    )
    # sigma is a scale, not a shifted target; keep it normalized here. In the
    # zarr inference script we multiply it by the same physical factor as the
    # corresponding field prediction before saving CP arrays.
    unscaled_sigma = sigma_outputs
    metrics = metrics_fn(unscaled_outputs, unscaled_targets, dist_manager, modes)

    # In the combined mode, this is a list of dicts.  Merge them.
    metrics = (
        {k: v for d in metrics for k, v in d.items()}
        if isinstance(metrics, list)
        else metrics
    )
    all_metrics.update(metrics)

    if unscaled_sigma is not None:
        return full_loss, all_metrics, (unscaled_outputs, unscaled_targets, unscaled_sigma)
    return full_loss, all_metrics, (unscaled_outputs, unscaled_targets)


@profile
def train_epoch(
    dataloader,
    epoch_len: int,
    model: torch.nn.Module,
    output_pad_size: int | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: PythonLogger,
    writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        dataloader: Training data loader
        model (torch.nn.Module): The neural network model to train.
        epoch_len (int): Length of the epoch.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        logger (PythonLogger): Logger for training progress.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager from physicsnemo.
        scaler (GradScaler | None, optional): Gradient scaler for mixed precision training.
    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_metrics = {}

    precision = getattr(cfg, "precision", "float32")
    start_time = time.time()
    log_field_stats = _should_log_field_stats(cfg, dist_manager.rank)
    legacy_tf = getattr(cfg.training, "use_legacy_transolver_forward", False)
    concat_fx = getattr(cfg.data, "concat_embedding_to_fx", False)
    emb_only_fx = getattr(cfg.data, "legacy_fx_embeddings_only", False)

    for i, batch in enumerate(dataloader):
        log_prefix = f"[epoch={epoch} train step={i}]" if log_field_stats else ""

        loss, metrics, _ = forward_pass(
            batch,
            model,
            precision,
            output_pad_size,
            dist_manager,
            cfg.data.mode,
            dataloader,
            debug_io_stats=log_field_stats,
            debug_log_prefix=log_prefix,
            legacy_transolver_forward=legacy_tf,
            concat_embedding_to_fx=concat_fx,
            legacy_fx_embeddings_only=emb_only_fx,
            training_cfg=cfg.training,
        )

        # Add concrete dropout regularization loss
        lambda_reg = getattr(cfg.training, "lambda_reg", 0.0)
        if lambda_reg > 0:
            reg_loss = collect_concrete_dropout_losses(model)
            if reg_loss.requires_grad:
                loss = loss + lambda_reg * reg_loss

        debug_osl = (
            getattr(cfg.training, "debug_one_step_loss", False)
            and i == 0
            and dist_manager.rank == 0
        )
        if debug_osl:
            loss_before = float(loss.detach().item())

        optimizer.zero_grad()
        if precision == "float16" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if debug_osl:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                loss_after, _, _ = forward_pass(
                    batch,
                    model,
                    precision,
                    output_pad_size,
                    dist_manager,
                    cfg.data.mode,
                    dataloader,
                    debug_io_stats=False,
                    legacy_transolver_forward=legacy_tf,
                    concat_embedding_to_fx=concat_fx,
                    legacy_fx_embeddings_only=emb_only_fx,
                    training_cfg=cfg.training,
                )
            if was_training:
                model.train()
            la = float(loss_after.detach().item())
            logger.info(
                f"[debug] one_step_loss: before={loss_before:.8f}, after={la:.8f}, "
                f"delta={la - loss_before:.8e}"
            )

        if not isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        end_time = time.time()

        # Logging
        this_loss = loss.detach().item()
        total_loss += this_loss

        if i == 0:
            total_metrics = metrics
        else:
            total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

        duration = end_time - start_time
        start_time = end_time
        images_per_second = 1 / duration

        mem_usage = torch.cuda.memory_reserved() / 1024**3

        logger.info(
            f"Epoch {epoch} [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s Mem: {mem_usage:.2f}GB"
        )
        if dist_manager.rank == 0:
            writer.add_scalar(
                "batch/learning_rate",
                optimizer.param_groups[0]["lr"],
                i + epoch_len * epoch,
            )
            writer.add_scalar("batch/loss", this_loss, i + epoch_len * epoch)
            writer.add_scalar(
                "batch/throughpu_per_gpu", images_per_second, i + epoch_len * epoch
            )
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(
                    f"batch/{metric_name}", metric_value, i + epoch_len * epoch
                )

        if cfg.profile and i >= 10:
            break  # Stop profiling after 10 batches

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)

        # Log concrete dropout rates if enabled
        dropout_rates = get_concrete_dropout_rates(model)
        if dropout_rates:
            for name, rate in dropout_rates.items():
                writer.add_scalar(f"dropout_rates/{name}", rate, epoch)

        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Average Metrics:\n{metrics_table}\n")
    return avg_loss


@profile
def val_epoch(
    dataloader,
    epoch_len: int,
    model: torch.nn.Module,
    output_pad_size: int | None,
    logger: PythonLogger,
    val_writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
) -> float:
    """
    Run validation for one epoch.

    Args:
        dataloader: Validation data loader.
        epoch_len (int): Length of the epoch.
        model (torch.nn.Module): The model to evaluate.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        logger (PythonLogger): Logger for validation progress.
        val_writer (SummaryWriter): TensorBoard writer for logging validation metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager instance.
    Returns:
        float: The average validation loss for the epoch.
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_metrics = {}

    precision = getattr(cfg, "precision", "float32")
    log_field_stats = _should_log_field_stats(cfg, dist_manager.rank)
    legacy_tf = getattr(cfg.training, "use_legacy_transolver_forward", False)
    concat_fx = getattr(cfg.data, "concat_embedding_to_fx", False)
    emb_only_fx = getattr(cfg.data, "legacy_fx_embeddings_only", False)

    start_time = time.time()
    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(dataloader):
            log_prefix = f"[epoch={epoch} val step={i}]" if log_field_stats else ""
            loss, metrics, _ = forward_pass(
                batch,
                model,
                precision,
                output_pad_size,
                dist_manager,
                cfg.data.mode,
                dataloader,
                debug_io_stats=log_field_stats,
                debug_log_prefix=log_prefix,
                legacy_transolver_forward=legacy_tf,
                concat_embedding_to_fx=concat_fx,
                legacy_fx_embeddings_only=emb_only_fx,
                training_cfg=cfg.training,
            )

            if i == 0:
                total_metrics = metrics
            else:
                total_metrics = {
                    k: total_metrics[k] + metrics[k] for k in metrics.keys()
                }

            # Logging
            this_loss = loss.detach().item()
            total_loss += this_loss

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            logger.info(
                f"Val [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s"
            )
            # We don't add individual loss measurements to tensorboard in the validation loop.

            if cfg.profile and i >= 10:
                break  # Stop profiling after 10 batches

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        val_writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            val_writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)
        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Validation Average Metrics:\n{metrics_table}\n")
    return avg_loss


def update_model_params_for_fp8(cfg, logger) -> tuple | None:
    """
    Adjusts model configuration parameters to ensure compatibility with FP8 computations.

    The output shape will be padded to a multiple of 16.  The input shape
    is padded dynamically in the forward pass, but that is printed here
    for information.

    Args:
        cfg: Configuration object with model and training attributes.
        logger: Logger object for info messages.

    Returns:
        tuple: (cfg, output_pad_size) if precision is "float8", where output_pad_size is the amount
               of padding added to the output dimension (or None if no padding was needed).
    """
    # we have to manipulate the output shape
    # to enable fp8 computations with transformer_engine.
    # need the input and output to be divisible by 16.
    # if (cfg.model.embedding_dim + cfg.model.functional_dim) % 16 != 0:

    output_pad_size = None
    if cfg.precision == "float8":
        if cfg.model.out_dim % 16 != 0:
            # pad the output:
            output_pad_size = 16 - (cfg.model.out_dim % 16)
            cfg.model.out_dim += output_pad_size
            logger.info(
                f"Padding output dimension to {cfg.model.out_dim} for fp8 autocast"
            )

        # This part is informational only:
        if (cfg.model.functional_dim + cfg.model.embedding_dim) % 16 != 0:
            input_pad_size = 16 - (
                (cfg.model.functional_dim + cfg.model.embedding_dim) % 16
            )
            cfg.model.functional_dim += input_pad_size
            logger.info(
                f"Padding input dimension to {cfg.model.functional_dim} and {cfg.model.embedding_dim} for fp8 autocast"
            )

    return cfg, output_pad_size


@profile
def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    from transolver_global_fx import (
        patch_transolver_datapipe_global_fx,
        patch_transolver_preprocess_fourier_surface,
        patch_transolver_preprocess_surface_for_overfit,
        set_overfit_deterministic_subsample,
    )

    patch_transolver_datapipe_global_fx()
    patch_transolver_preprocess_surface_for_overfit()
    patch_transolver_preprocess_fourier_surface()
    _ov_s = getattr(cfg.data, "overfit_n_samples", None)
    if (
        _ov_s is not None
        and int(_ov_s) > 0
        and getattr(cfg.data, "overfit_deterministic_subsample", True)
    ):
        set_overfit_deterministic_subsample(True)

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    # Set checkpoint directory - defaults to output_dir if not specified
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = cfg.output_dir

    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/train",
            )
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/val",
            )
        )
    else:
        writer = None
        val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Output directory: {cfg.output_dir}/{cfg.run_id}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}/{cfg.run_id}/checkpoints")

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    if getattr(cfg.training, "use_legacy_transolver_forward", False):
        if dist_manager.rank == 0:
            logger.info(
                "training.use_legacy_transolver_forward=true: forward uses "
                "model(fx=features, embedding=embeddings) (classic Transolver API)."
            )
    elif dist_manager.rank == 0:
        logger.info(
            "training.use_legacy_transolver_forward=false: forward uses "
            "GeoTransolver global_embedding / local_embedding / local_positions "
            "(geometry only if present in batch)."
        )

    # Set up model
    # (Using partial convert to get lists, etc., instead of ListConfigs.)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    model.to(dist_manager.device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Load the normalization file from configured directory (defaults to current dir)
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
        if not Path(norm_file).is_file():
            raise FileNotFoundError(
                f"找不到表面场归一化文件: {Path(norm_file).resolve()}\n"
                "请在项目根目录运行: python compute_normalizations.py\n"
                "或执行 scripts/train.sh 中 RUN_NORM=1 的步骤；若 .npz 在其它目录，请设置 "
                "data.normalization_dir 指向该目录。"
            )
        norm_data = np.load(norm_file)
        surface_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        surface_factors = None

    if cfg.data.mode == "volume" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")
        if not Path(norm_file).is_file():
            raise FileNotFoundError(
                f"找不到体场归一化文件: {Path(norm_file).resolve()}\n"
                "请用 data.mode=volume 运行 python compute_normalizations.py，或设置 "
                "data.normalization_dir。"
            )
        norm_data = np.load(norm_file)
        volume_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        volume_factors = None

    # Training dataset
    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    skip_validation = bool(getattr(cfg.training, "skip_validation", False))

    val_dataloader = None
    if not skip_validation:
        val_dataloader = create_transolver_dataset(
            cfg.data,
            phase="val",
            surface_factors=surface_factors,
            volume_factors=volume_factors,
        )
    elif dist_manager.rank == 0:
        logger.info(
            "training.skip_validation=true: no val epoch each training loop. "
            "Use data_splits/calib only for CP (inference.phase=val + data.val.data_path=calib)."
        )

    sync_use_fourier_datapipe_config(train_dataloader, cfg)
    if val_dataloader is not None:
        sync_use_fourier_datapipe_config(val_dataloader, cfg)

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = None
    if val_dataloader is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataloader,
            num_replicas=num_replicas,
            rank=data_rank,
            shuffle=False,
            drop_last=True,
        )

    overfit_raw = getattr(cfg.data, "overfit_n_samples", None)
    if overfit_raw is None:
        overfit_indices: list[int] | None = None
    else:
        overfit_n = int(overfit_raw)
        if overfit_n <= 0:
            overfit_indices = None
        else:
            n_cap = min(overfit_n, len(train_dataloader))
            if n_cap < 1:
                raise ValueError(
                    "data.overfit_n_samples is set but train dataloader is empty."
                )
            overfit_indices = list(range(n_cap))
            logger.info(
                f"Overfit mode: using train indices {overfit_indices} "
                f"(overfit_use_train_for_val={getattr(cfg.data, 'overfit_use_train_for_val', True)}). "
                "Multi-GPU: each rank runs the same subset; use 1 GPU if you see sampler issues."
            )
            if dist_manager.rank == 0:
                logger.info(
                    "Overfit mode ON: printing per-channel mean/std/min/max, "
                    "sigma, residual, and loss parts each train/val step."
                )
                _log_surface_normalization_factors(surface_factors)

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]

    # Set up optimizer and scheduler
    if getattr(cfg.training, "simple_adamw_only", False):
        optimizer = hydra.utils.instantiate(
            cfg.training.optimizer,
            params=model.parameters(),
        )
        if dist_manager.rank == 0:
            logger.info(
                "training.simple_adamw_only=true: single AdamW on all parameters (Muon disabled)."
            )
    else:
        optimizer = hydra.utils.instantiate(
            cfg.training.optimizer, params=other_params
        )

        optimizer = CombinedOptimizer(
            optimizers=[
                torch.optim.Muon(
                    muon_params,
                    lr=cfg.training.optimizer.lr,
                    weight_decay=cfg.training.optimizer.weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                optimizer,
            ],
        )

    # Set up learning rate scheduler based on config
    scheduler_cfg = cfg.training.scheduler
    scheduler_name = scheduler_cfg.name
    scheduler_params = dict(scheduler_cfg.params)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    if precision == "float8" and not TE_AVAILABLE:
        raise ImportError(
            "TransformerEngine is not installed.  Please install it to use float8 precision."
        )

    ckpt_args = {
        "path": f"{checkpoint_dir}/{cfg.run_id}/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }

    _ov_ckpt = getattr(cfg.data, "overfit_n_samples", None)
    if (
        _ov_ckpt is not None
        and int(_ov_ckpt) > 0
        and getattr(cfg.data, "overfit_fresh_start", True)
    ):
        loaded_epoch = 0
        if dist_manager.rank == 0:
            logger.info(
                "Overfit mode: skipping load_checkpoint (data.overfit_fresh_start=true). "
                "To resume from runs/<run_id>/checkpoints set data.overfit_fresh_start=false."
            )
    else:
        loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.compile:
        model = torch.compile(model)

    overfit_fixed_batch_cache: dict | None = None
    if (
        overfit_indices is not None
        and getattr(cfg.data, "overfit_fixed_train_batch", False)
    ):
        train_dataloader.dataset.set_indices(overfit_indices)
        overfit_fixed_batch_cache = _move_batch_tensors_to_device(
            next(iter(train_dataloader)), dist_manager.device
        )
        if dist_manager.rank == 0:
            logger.info(
                "data.overfit_fixed_train_batch=true: cached one train batch on device "
                "for all epochs (no per-step datapipe / geometry resample in train)."
            )

    save_best_on_val = bool(getattr(cfg.training, "save_best_on_val", False))
    best_val_loss = float("inf")
    best_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_best"

    # Training loop
    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_dl_eff = None
        val_epoch_len = 0

        if overfit_indices is not None:
            train_dataloader.dataset.set_indices(overfit_indices)
            use_train_for_val = getattr(cfg.data, "overfit_use_train_for_val", True)
            if skip_validation:
                pass
            elif use_train_for_val:
                val_dl_eff = train_dataloader
                val_epoch_len = len(overfit_indices)
            elif val_dataloader is not None:
                nv = min(len(overfit_indices), len(val_dataloader))
                val_dataloader.dataset.set_indices(list(range(nv)))
                val_dl_eff = val_dataloader
                val_epoch_len = max(nv, 1)
            else:
                val_dl_eff = train_dataloader
                val_epoch_len = len(overfit_indices)
            train_epoch_len = len(overfit_indices)
            train_loader_for_epoch = train_dataloader
            if overfit_fixed_batch_cache is not None:
                train_loader_for_epoch = _OverfitFixedBatchTrainLoader(
                    train_dataloader,
                    overfit_fixed_batch_cache,
                    train_epoch_len,
                )
        else:
            train_dataloader.dataset.set_indices(list(train_sampler))
            train_epoch_len = len(list(train_sampler))
            train_loader_for_epoch = train_dataloader
            if val_dataloader is not None and val_sampler is not None:
                val_dataloader.dataset.set_indices(list(val_sampler))
                val_dl_eff = val_dataloader
                val_epoch_len = len(list(val_sampler))
            else:
                val_dl_eff = None
                val_epoch_len = 0

        start_time = time.time()
        # Training phase
        with Profiler():
            train_loss = train_epoch(
                train_loader_for_epoch,
                train_epoch_len,
                model,
                output_pad_size,
                optimizer,
                scheduler,
                logger,
                writer,
                epoch,
                cfg,
                dist_manager,
                scaler,
            )
            end_time = time.time()
            train_duration = end_time - start_time

            if val_dl_eff is not None and val_epoch_len > 0:
                start_time = time.time()
                val_loss = val_epoch(
                    val_dl_eff,
                    val_epoch_len,
                    model,
                    output_pad_size,
                    logger,
                    val_writer,
                    epoch,
                    cfg,
                    dist_manager,
                )
                end_time = time.time()
                val_duration = end_time - start_time
            else:
                val_loss = float("nan")
                val_duration = 0.0

        if val_dl_eff is not None and val_epoch_len > 0:
            logger.info(
                f"Epoch [{epoch}/{cfg.training.num_epochs}] Train Loss: {train_loss:.6f} "
                f"[duration: {train_duration:.2f}s] Val Loss: {val_loss:.6f} "
                f"[duration: {val_duration:.2f}s]"
            )
        else:
            logger.info(
                f"Epoch [{epoch}/{cfg.training.num_epochs}] Train Loss: {train_loss:.6f} "
                f"[duration: {train_duration:.2f}s] (validation skipped)"
            )

        # save checkpoint
        if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch + 1)

        if (
            save_best_on_val
            and val_dl_eff is not None
            and val_epoch_len > 0
            and not math.isnan(val_loss)
            and dist_manager.rank == 0
            and val_loss < best_val_loss
        ):
            best_val_loss = val_loss
            best_ckpt_args = {**ckpt_args, "path": best_ckpt_path}
            save_checkpoint(**best_ckpt_args, epoch=epoch + 1)
            logger.info(
                f"New best val loss {val_loss:.6f} (epoch {epoch}), "
                f"saved to {best_ckpt_path}"
            )

        if scheduler_name == "StepLR":
            scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def launch(cfg: DictConfig):
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """

    # If you want to use `line_profiler` or PyTorch's profiler, enable them here.

    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
        profiler.enable("line_profiler")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
