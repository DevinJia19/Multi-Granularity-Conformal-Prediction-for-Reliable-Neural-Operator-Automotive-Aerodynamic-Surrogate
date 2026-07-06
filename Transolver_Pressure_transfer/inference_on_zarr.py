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

from pathlib import Path

import repo_env

repo_env.ensure_repo_local_caches()

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torchinfo
import typing, csv
import collections
from datetime import datetime

import hydra
import omegaconf
from omegaconf import DictConfig
from physicsnemo.models.transolver.transolver import Transolver
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from sklearn.metrics import r2_score
from metrics import metrics_fn_surface, metrics_fn_volume

from physicsnemo.distributed import DistributedManager

import time

from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
    TransolverDataPipe,
)
from tabulate import tabulate

from inference_utils import (
    batched_inference_loop,
    ensure_single_process_dist_env,
    mc_dropout_inference_loop,
    setup_mc_dropout,
)

# import transformer_engine.pytorch as te
# from transformer_engine.common.recipe import Format, DelayedScaling
from torch.amp import autocast
from contextlib import nullcontext

from train import (
    get_autocast_context,
    pad_input_for_fp8,
    unpad_output_for_fp8,
    update_model_params_for_fp8,
    sync_use_fourier_datapipe_config,
)

# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([typing.Any])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
# torch.serialization.add_safe_globals([omegaconf.base.Metadata])


@torch.no_grad()
def compute_force_coefficients(
    normals: torch.Tensor,
    area: torch.Tensor,
    coeff: float,
    p: torch.Tensor,
    wss: torch.Tensor,
    force_direction: torch.Tensor = np.array([1, 0, 0]),
):
    """
    Computes force coefficients for a given mesh. Output includes the pressure and skin
    friction components. Can be used to compute lift and drag.
    For drag, use the `force_direction` as the direction of the motion,
    e.g. [1, 0, 0] for flow in x direction.
    For lift, use the `force_direction` as the direction perpendicular to the motion,
    e.g. [0, 1, 0] for flow in x direction and weight in y direction.

    Parameters:
    -----------
    normals: torch.Tensor
        The surface normals on cells of the mesh
    area: torch.Tensor
        The surface areas of each cell
    coeff: float
        Reciprocal of dynamic pressure times the frontal area, i.e. 2/(A * rho * U^2)
    p: torch.Tensor
        Pressure distribution on the mesh (on each cell)
    wss: torch.Tensor
        Wall shear stress distribution on the mesh (on each cell)
    force_direction: torch.Tensor
        Direction to compute the force, default is np.array([1, 0, 0])

    Returns:
    --------
    c_total: float
        Computed total force coefficient
    c_p: float
        Computed pressure force coefficient
    c_f: float
        Computed skin friction coefficient
    """

    # Compute coefficients
    c_p = coeff * torch.sum(torch.sum(normals * force_direction, dim=-1) * area * p)
    c_f = -coeff * torch.sum(torch.sum(wss * force_direction, dim=-1) * area)

    # Compute total force coefficients
    c_total = c_p + c_f

    return c_total, c_p, c_f


def _sigma_to_physical_units(
    sigma_norm: torch.Tensor,
    surface_factors: dict | None,
    air_density: torch.Tensor | None,
    stream_velocity: torch.Tensor | None,
) -> torch.Tensor:
    """Map normalized sigma_hat to the same physical scale as saved pred/target.

    ``forward_pass`` returns denormalized predictions but keeps sigma in normalized
    units. CP scores need ``sigma_phys = sigma_norm * std`` (per channel), then the
    same dynamic-pressure factor applied to surface fields in this script.
    """
    sigma = sigma_norm
    if surface_factors is not None:
        std = surface_factors["std"].to(device=sigma.device, dtype=sigma.dtype)
        while std.dim() < sigma.dim():
            std = std.unsqueeze(0)
        sigma = sigma * std
    if stream_velocity is not None and air_density is not None:
        surface_scale = stream_velocity**2.0 * air_density
        sigma = sigma * surface_scale
    return sigma


def _squeeze_leading_batch(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def inference(cfg: DictConfig) -> None:
    """
    Run inference on a validation Zarr dataset using a trained Transolver model.

    Args:
        cfg (DictConfig): Hydra configuration object containing model, data, and training settings.

    Returns:
        None
    """
    ensure_single_process_dist_env()
    DistributedManager.initialize()

    from transolver_global_fx import (
        patch_transolver_datapipe_global_fx,
        patch_transolver_preprocess_fourier_surface,
    )

    patch_transolver_datapipe_global_fx()
    patch_transolver_preprocess_fourier_surface()

    dist_manager = DistributedManager()

    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    if cfg.checkpoint_dir is not None:
        checkpoint_dir = cfg.checkpoint_dir
    else:
        checkpoint_dir = f"{cfg.output_dir}/{cfg.run_id}/checkpoints"

    ckpt_args = {
        "path": checkpoint_dir,
        "models": model,
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    logger.info(f"loaded epoch: {loaded_epoch}")
    model.to(dist_manager.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Load the normalization file from configured directory (defaults to current dir)
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
        norm_data = np.load(norm_file)
        surface_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        surface_factors = None

    if cfg.data.mode == "volume" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")
        norm_data = np.load(norm_file)
        volume_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        volume_factors = None

    if cfg.compile:
        model = torch.compile(model, dynamic=True)

    mc_dropout_samples = setup_mc_dropout(model, cfg, logger)

    # For INFERENCE, we deliberately set the resolution in the data pipe to NONE
    # so there is not downsampling.  We still batch it in the inference script
    # for memory usage constraints.

    batch_resolution = cfg.data.resolution
    cfg.data.resolution = None
    ## Make sure to read the whole data sample for volume:
    if cfg.data.mode == "volume":
        cfg.data.volume_sample_from_disk = False

    # And we need the mesh features for drag, lift in surface data:
    if cfg.data.mode == "surface":
        cfg.data.return_mesh_features = True

    infer_phase = str(
        omegaconf.OmegaConf.select(cfg, "inference.phase", default="val")
    )
    logger.info(f"Inference dataset phase: {infer_phase}")

    infer_dataset = create_transolver_dataset(
        cfg.data,
        phase=infer_phase,
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    sync_use_fourier_datapipe_config(infer_dataset, cfg)

    infer_indices: list[int] | None = None
    if dist_manager.world_size > 1:
        infer_sampler = torch.utils.data.distributed.DistributedSampler(
            infer_dataset,
            num_replicas=dist_manager.world_size,
            rank=dist_manager.rank,
            shuffle=False,
            drop_last=False,
        )
        infer_indices = list(infer_sampler)
        infer_dataset.dataset.set_indices(infer_indices)
        logger.info(
            f"Multi-GPU inference: rank {dist_manager.rank}/{dist_manager.world_size}, "
            f"{len(infer_indices)} samples on this rank."
        )

    legacy_tf = omegaconf.OmegaConf.select(
        cfg, "training.use_legacy_transolver_forward", default=False
    )
    concat_fx = omegaconf.OmegaConf.select(
        cfg, "data.concat_embedding_to_fx", default=False
    )
    emb_only_fx = omegaconf.OmegaConf.select(
        cfg, "data.legacy_fx_embeddings_only", default=False
    )

    results = []
    start = time.time()
    for batch_idx, batch in enumerate(infer_dataset):
        sample_id = (
            infer_indices[batch_idx] if infer_indices is not None else batch_idx
        )
        if mc_dropout_samples > 0:
            # MC-Dropout: run N stochastic forward passes (no torch.no_grad
            # since dropout needs to be active, but we don't need gradients)
            with torch.no_grad():
                (
                    global_predictions,
                    global_std,
                    all_mc_predictions,
                    loss,
                    metrics,
                    global_targets,
                ) = mc_dropout_inference_loop(
                    batch,
                    model,
                    cfg.precision,
                    cfg.data.mode,
                    batch_resolution,
                    output_pad_size,
                    dist_manager,
                    infer_dataset,
                    n_samples=mc_dropout_samples,
                    legacy_transolver_forward=legacy_tf,
                    concat_embedding_to_fx=concat_fx,
                    legacy_fx_embeddings_only=emb_only_fx,
                )
            # Log mean uncertainty for this sample
            mean_std = global_std.mean().item()
            logger.info(f"Batch {batch_idx} mean uncertainty (std): {mean_std:.6f}")
            global_sigmas = None
        else:
            with torch.no_grad():
                loss, metrics, pred_pack = batched_inference_loop(
                    batch,
                    model,
                    cfg.precision,
                    cfg.data.mode,
                    batch_resolution,
                    output_pad_size,
                    dist_manager,
                    infer_dataset,
                    legacy_transolver_forward=legacy_tf,
                    concat_embedding_to_fx=concat_fx,
                    legacy_fx_embeddings_only=emb_only_fx,
                )
                if len(pred_pack) == 3:
                    global_predictions, global_targets, global_sigmas = pred_pack
                else:
                    global_predictions, global_targets = pred_pack
                    global_sigmas = None
        end = time.time()
        elapsed = end - start
        logger.info(f"Finished batch {batch_idx} in {elapsed:.4f} seconds")
        start = time.time()

        air_density = batch["air_density"] if "air_density" in batch.keys() else None
        stream_velocity = (
            batch["stream_velocity"] if "stream_velocity" in batch.keys() else None
        )

        if cfg.data.mode == "surface":
            coeff = 1.0

            if stream_velocity is not None:
                surface_scale = stream_velocity**2.0 * air_density
                global_predictions = global_predictions * surface_scale
                global_targets = global_targets * surface_scale

            if global_sigmas is not None:
                global_sigmas = _sigma_to_physical_units(
                    global_sigmas,
                    surface_factors,
                    air_density,
                    stream_velocity,
                )

            # Optional pointwise dump for CP calibration / testing.
            # Channels: 0 pressure, 1-3 wss_x/y/z; pred/target/sigma in physical units.
            if bool(omegaconf.OmegaConf.select(cfg, "cp_output.save_pointwise_npz", default=False)):
                cp_subdir = omegaconf.OmegaConf.select(cfg, "cp_output.subdir", default=None)
                if not cp_subdir:
                    cp_subdir = f"cp_pointwise_{infer_phase}"
                cp_dir = Path(cfg.output_dir) / cfg.run_id / cp_subdir
                cp_dir.mkdir(parents=True, exist_ok=True)
                if global_sigmas is None:
                    logger.warning(
                        "cp_output.save_pointwise_npz=true but model returned no sigma; "
                        "saving pred/target only. CP scripts can fall back to sigma=1."
                    )
                save_dict = {
                    "pred": _squeeze_leading_batch(global_predictions),
                    "target": _squeeze_leading_batch(global_targets),
                }
                if global_sigmas is not None:
                    save_dict["sigma"] = _squeeze_leading_batch(global_sigmas)

                # Prefer original mesh arrays if available.
                if "surface_mesh_centers" in batch:
                    save_dict["surface_mesh_centers"] = _squeeze_leading_batch(
                        batch["surface_mesh_centers"]
                    )
                else:
                    logger.warning(
                        "surface_mesh_centers missing from batch; "
                        "falling back to embeddings[..., :3] for VTP point coordinates."
                    )
                    save_dict["surface_mesh_centers"] = _squeeze_leading_batch(
                        batch["embeddings"][..., :3]
                    )

                if "surface_normals" in batch:
                    save_dict["surface_normals"] = _squeeze_leading_batch(
                        batch["surface_normals"]
                    )
                else:
                    logger.warning(
                        "surface_normals missing from batch; "
                        "falling back to embeddings[..., 3:6]."
                    )
                    save_dict["surface_normals"] = _squeeze_leading_batch(
                        batch["embeddings"][..., 3:6]
                    )

                if "surface_areas" in batch:
                    save_dict["surface_areas"] = _squeeze_leading_batch(
                        batch["surface_areas"]
                    )
                np.savez_compressed(
                    cp_dir / f"batch_{sample_id:05d}.npz", **save_dict
                )

            metrics = metrics_fn_surface(
                global_predictions, global_targets, dist_manager
            )
            # Compute the drag and loss coefficients:
            # (Index on [0] is to remove the 1 batch index)
            pred_pressure, pred_shear = torch.split(
                global_predictions[0], (1, 3), dim=-1
            )

            pred_pressure = pred_pressure.reshape(-1)
            pred_drag_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                pred_pressure,
                pred_shear,
                torch.tensor([[1, 0, 0]], device=dist_manager.device),
            )

            pred_lift_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                pred_pressure,
                pred_shear,
                torch.tensor([[0, 0, 1]], device=dist_manager.device),
            )

            # true_fields = val_dataset.unscale_model_targets(batch["fields"], air_density=air_density, stream_velocity=stream_velocity)
            true_pressure, true_shear = torch.split(global_targets[0], (1, 3), dim=-1)

            true_pressure = true_pressure.reshape(-1)
            true_drag_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                true_pressure,
                true_shear,
                torch.tensor([[1, 0, 0]], device=dist_manager.device),
            )

            true_lift_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                true_pressure,
                true_shear,
                torch.tensor([[0, 0, 1]], device=dist_manager.device),
            )

            pred_lift_coeff = pred_lift_coeff.item()
            pred_drag_coeff = pred_drag_coeff.item()

            # Extract metric values and convert tensors to floats
            l2_pressure = (
                metrics["l2_pressure_surf"].item()
                if hasattr(metrics["l2_pressure_surf"], "item")
                else metrics["l2_pressure_surf"]
            )
            l1_pressure = (
                metrics["l1_pressure_surf"].item()
                if hasattr(metrics["l1_pressure_surf"], "item")
                else metrics["l1_pressure_surf"]
            )
            mae_pressure = (
                metrics["mae_pressure_surf"].item()
                if hasattr(metrics["mae_pressure_surf"], "item")
                else metrics["mae_pressure_surf"]
            )
            l2_wall_shear_stress = (
                metrics["l2_wall_shear_stress"].item()
                if hasattr(metrics["l2_wall_shear_stress"], "item")
                else metrics["l2_wall_shear_stress"]
            )
            l1_wall_shear_stress = (
                metrics["l1_wall_shear_stress"].item()
                if hasattr(metrics["l1_wall_shear_stress"], "item")
                else metrics["l1_wall_shear_stress"]
            )
            mae_wall_shear_stress = (
                metrics["mae_wall_shear_stress"].item()
                if hasattr(metrics["mae_wall_shear_stress"], "item")
                else metrics["mae_wall_shear_stress"]
            )

            results.append(
                [
                    sample_id,
                    f"{loss:.4f}",
                    f"{l2_pressure:.4f}",
                    f"{l1_pressure:.4f}",
                    f"{mae_pressure:.4f}",
                    f"{l2_wall_shear_stress:.4f}",
                    f"{l1_wall_shear_stress:.4f}",
                    f"{mae_wall_shear_stress:.4f}",
                    f"{pred_drag_coeff:.4f}",
                    f"{pred_lift_coeff:.4f}",
                    f"{true_drag_coeff:.4f}",
                    f"{true_lift_coeff:.4f}",
                    f"{elapsed:.4f}",
                ]
            )

        elif cfg.data.mode == "volume":
            if stream_velocity is not None:
                global_predictions[:, :, 3] = (
                    global_predictions[:, :, 3] * stream_velocity**2.0 * air_density
                )
                global_targets[:, :, 3] = (
                    global_targets[:, :, 3] * stream_velocity**2.0 * air_density
                )
                global_predictions[:, :, 0:3] = (
                    global_predictions[:, :, 0:3] * stream_velocity
                )
                global_targets[:, :, 0:3] = global_targets[:, :, 0:3] * stream_velocity
                global_predictions[:, :, 4] = (
                    global_predictions[:, :, 4] * stream_velocity**2.0 * air_density
                )
                global_targets[:, :, 4] = (
                    global_targets[:, :, 4] * stream_velocity**2.0 * air_density
                )

            metrics = metrics_fn_volume(
                global_predictions, global_targets, dist_manager
            )
            # Extract metric values and convert tensors to floats
            l2_pressure = (
                metrics["l2_pressure_vol"].item()
                if hasattr(metrics["l2_pressure_vol"], "item")
                else metrics["l2_pressure_vol"]
            )
            l1_pressure = (
                metrics["l1_pressure_vol"].item()
                if hasattr(metrics["l1_pressure_vol"], "item")
                else metrics["l1_pressure_vol"]
            )
            mae_pressure = (
                metrics["mae_pressure_vol"].item()
                if hasattr(metrics["mae_pressure_vol"], "item")
                else metrics["mae_pressure_vol"]
            )
            l2_velocity = (
                metrics["l2_velocity"].item()
                if hasattr(metrics["l2_velocity"], "item")
                else metrics["l2_velocity"]
            )
            l1_velocity = (
                metrics["l1_velocity"].item()
                if hasattr(metrics["l1_velocity"], "item")
                else metrics["l1_velocity"]
            )
            mae_velocity = (
                metrics["mae_velocity"].item()
                if hasattr(metrics["mae_velocity"], "item")
                else metrics["mae_velocity"]
            )

            l2_nut = (
                metrics["l2_nut"].item()
                if hasattr(metrics["l2_nut"], "item")
                else metrics["l2_nut"]
            )
            l1_nut = (
                metrics["l1_nut"].item()
                if hasattr(metrics["l1_nut"], "item")
                else metrics["l1_nut"]
            )
            mae_nut = (
                metrics["mae_nut"].item()
                if hasattr(metrics["mae_nut"], "item")
                else metrics["mae_nut"]
            )

            results.append(
                [
                    sample_id,
                    f"{loss:.4f}",
                    f"{l2_pressure:.4f}",
                    f"{l1_pressure:.4f}",
                    f"{mae_pressure:.4f}",
                    f"{l2_velocity:.4f}",
                    f"{l1_velocity:.4f}",
                    f"{mae_velocity:.4f}",
                    f"{l2_nut:.4f}",
                    f"{l1_nut:.4f}",
                    f"{mae_nut:.4f}",
                    f"{elapsed:.4f}",
                ]
            )

    if dist_manager.world_size > 1 and dist.is_initialized():
        gathered: list[list] = [None] * dist_manager.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, results)
        if dist_manager.rank == 0:
            merged: list = []
            for rank_results in gathered:
                merged.extend(rank_results)
            results = sorted(merged, key=lambda row: int(row[0]))

    if dist_manager.rank == 0:
        if cfg.data.mode == "surface":
            pred_drag_coeffs = [r[8] for r in results]
            pred_lift_coeffs = [r[9] for r in results]
            true_drag_coeffs = [r[10] for r in results]
            true_lift_coeffs = [r[11] for r in results]

            # Compute the R2 scores for lift and drag:
            r2_lift = r2_score(true_lift_coeffs, pred_lift_coeffs)
            r2_drag = r2_score(true_drag_coeffs, pred_drag_coeffs)

            headers = [
                "Sample",
                "Loss",
                "L2 Pressure",
                "L1 Pressure",
                "MAE Pressure",
                "L2 Wall Shear Stress",
                "L1 Wall Shear Stress",
                "MAE Wall Shear Stress",
                "Predicted Drag Coefficient",
                "Pred Lift Coefficient",
                "True Drag Coefficient",
                "True Lift Coefficient",
                "Elapsed (s)",
            ]
            logger.info(
                f"Results:\n{tabulate(results, headers=headers, tablefmt='github')}"
            )
            logger.info(f"R2 score for lift: {r2_lift:.4f}")
            logger.info(f"R2 score for drag: {r2_drag:.4f}")
            csv_filename = f"{cfg.output_dir}/{cfg.run_id}/surface_inference_results_{datetime.now()}.csv"
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(results)
            logger.info(f"Results saved to {csv_filename}")

        elif cfg.data.mode == "volume":
            headers = [
                "Sample",
                "Loss",
                "L2 Pressure",
                "L1 Pressure",
                "MAE Pressure",
                "L2 Velocity",
                "L1 Velocity",
                "MAE Velocity",
                "L2 Nut",
                "L1 Nut",
                "MAE Nut",
                "Elapsed (s)",
            ]
            logger.info(
                f"Results:\n{tabulate(results, headers=headers, tablefmt='github')}"
            )
            csv_filename = f"{cfg.output_dir}/{cfg.run_id}/volume_inference_results_{datetime.now()}.csv"
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(results)
            logger.info(f"Results saved to {csv_filename}")

        # Calculate means for each metric (skip sample index column)
        if results:
            # Convert string values back to float for mean calculation
            arr = np.array(results)[:, 1:].astype(float)
            means = arr.mean(axis=0)
            mean_row = ["Mean"] + [f"{m:.4f}" for m in means]
            logger.info(
                f"Summary:\n{tabulate([mean_row], headers=headers, tablefmt='github')}"
            )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def launch(cfg: DictConfig) -> None:
    """
    Launch inference with Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        None
    """
    inference(cfg)


if __name__ == "__main__":
    launch()
