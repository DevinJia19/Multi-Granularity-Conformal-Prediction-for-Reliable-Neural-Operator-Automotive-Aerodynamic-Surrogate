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
import math

import torch

from physicsnemo.domain_parallel.shard_tensor import ShardTensor
from physicsnemo.utils.profiling import profile


def _batch_global_vec(t: torch.Tensor) -> torch.Tensor:
    """Flatten leading singleton dims so we get (B, D) for concat."""
    x = t.to(torch.float32)
    while x.dim() > 2:
        x = x.squeeze(1)
    return x


def append_fourier_surface_embeddings(
    embeddings: torch.Tensor,
    coord_scale: float = 50.0,
    n_bands: int = 4,
) -> torch.Tensor:
    """Append multi-frequency sin/cos of scaled xyz (first 3 channels) to embeddings.

    Base surface embedding is expected to be dim 6 (center xyz + normals). Returns
    dim ``6 + 2 * n_bands * 3`` (e.g. 30 for ``n_bands=4``).
    """
    centers = embeddings[..., :3]
    dtype = embeddings.dtype
    cs = (centers.to(torch.float32) * coord_scale) * math.pi
    sin_feats = [torch.sin(cs * (2**i)) for i in range(n_bands)]
    cos_feats = [torch.cos(cs * (2**i)) for i in range(n_bands)]
    out = torch.cat([embeddings, *sin_feats, *cos_feats], dim=-1)
    return out.to(dtype)


@profile
def preprocess_surface_data(
    batch: dict,
    norm_factors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Preprocess surface data.

    Global / functional input uses ``global_params_reference`` and
    ``global_params_values`` from zarr (typically 3 + 3 = 6), matching
    ``model.functional_dim`` / ``model.global_dim``. Embeddings are mesh
    centers, normals, and Fourier features of scaled coordinates (dim 30);
    targets are normalized surface fields.
    """

    mesh_centers = batch["surface_mesh_centers"]
    normals = batch["surface_normals"]
    targets = batch["surface_fields"]
    if "global_params_reference" in batch and "global_params_values" in batch:
        ref = _batch_global_vec(batch["global_params_reference"])
        val = _batch_global_vec(batch["global_params_values"])
        node_features = torch.cat([ref, val], dim=-1)
    else:
        node_features = torch.stack(
            [batch["air_density"], batch["stream_velocity"]], dim=-1
        ).to(torch.float32)

    # Normalize the surface fields:
    targets = (targets - norm_factors["mean"]) / norm_factors["std"]

    # Calculate center of mass
    sizes = batch["stl_areas"]
    centers = batch["stl_centers"]

    total_weighted_position = torch.einsum("ki,kij->kj", sizes, centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size

    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    embeddings = torch.cat([mesh_centers, normals], dim=-1)
    embeddings = append_fourier_surface_embeddings(embeddings)

    others: dict = {
        "surface_areas": batch["surface_areas"],
        "surface_normals": normals,
    }
    if "global_params_reference" in batch:
        others["global_params_reference"] = batch["global_params_reference"]
    if "global_params_values" in batch:
        others["global_params_values"] = batch["global_params_values"]
    if "stream_velocity" in batch:
        others["stream_velocity"] = batch["stream_velocity"]
    if "air_density" in batch:
        others["air_density"] = batch["air_density"]

    return node_features, embeddings, targets, others


@profile
def downsample_surface(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_keep=1024,
    *,
    deterministic_first: bool = False,
):
    if num_keep == -1:
        features = features.unsqueeze(1).expand(1, embeddings.shape[1], -1)
        return features, embeddings, targets

    """
    Downsample the surface data. We generate one set of indices, and
    use it to sample the same points from the features, embeddings,
    and targets.  Using torch.multinomial to sample without replacement.
    """

    num_samples = embeddings.shape[1]
    k = min(num_keep, num_samples)
    if deterministic_first:
        indices = torch.arange(k, device=features.device, dtype=torch.long)
    else:
        indices = torch.multinomial(
            torch.ones(num_samples, device=features.device),
            num_keep,
            replacement=False,
        )

    # Use the same indices to downsample all tensors
    downsampled_embeddings = embeddings[:, indices]
    downsampled_targets = targets[:, indices]
    # This unsqueezes the features (air density and stream velocity) to
    # the same shape as the embeddings
    downsampled_features = features.unsqueeze(1).expand(
        1, downsampled_embeddings.shape[1], -1
    )

    return downsampled_features, downsampled_embeddings, downsampled_targets
