"""GeoTransolver wrapper with a residual scale head for normalized conformal prediction.

The wrapped backbone predicts the mean surface field y_hat. A lightweight scale
head predicts a positive per-point sigma_hat. During training, train.py can use
sigma_hat to fit residual magnitudes and to add a KNN smoothness regularizer.

Inference returns (y_hat, sigma_hat), so downstream CP code can compute

    score = |y - y_hat| / (sigma_hat + eps)
    interval = y_hat ± q_hat * sigma_hat
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoTransolverWithSigma(nn.Module):
    """Wrap a GeoTransolver backbone and add a pointwise residual-scale head.

    Parameters
    ----------
    backbone:
        Instantiated GeoTransolver module, or a Hydra config with ``_target_``.
    out_dim:
        Number of predicted field channels. For this project: pressure + 3 WSS = 4.
    sigma_input:
        ``"raw_local_mean"`` uses [x,y,z,nx,ny,nz,y_hat_detached] as input.
        ``"raw_local"`` uses [x,y,z,nx,ny,nz].
        ``"full_local"`` uses the full Fourier local embedding.
        ``"full_local_mean"`` uses full local embedding plus y_hat_detached.
    sigma_hidden:
        Hidden width of the sigma MLP.
    sigma_layers:
        Number of Linear layers in the sigma MLP. Minimum 2.
    init_sigma:
        Initial sigma value in normalized target units.
    min_sigma:
        Positive lower bound added after softplus.
    max_sigma:
        Optional upper clamp for sigma. Use null/None to disable.
    """

    def __init__(
        self,
        backbone: nn.Module | Any,
        out_dim: int = 4,
        local_dim: int = 30,
        sigma_input: str = "raw_local_mean",
        sigma_hidden: int = 128,
        sigma_layers: int = 3,
        init_sigma: float = 0.2,
        min_sigma: float = 1.0e-6,
        max_sigma: float | None = None,
        **_: Any,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, nn.Module):
            # Hydra normally instantiates nested configs recursively, but keep this
            # fallback so the class also works if _recursive_=False is used.
            import hydra

            backbone = hydra.utils.instantiate(backbone)

        self.backbone = backbone
        self.out_dim = int(out_dim)
        self.local_dim = int(local_dim)
        self.sigma_input = str(sigma_input)
        self.min_sigma = float(min_sigma)
        self.max_sigma = None if max_sigma is None else float(max_sigma)

        if self.sigma_input == "raw_local":
            sigma_in_dim = 6
        elif self.sigma_input == "raw_local_mean":
            sigma_in_dim = 6 + self.out_dim
        elif self.sigma_input == "full_local":
            sigma_in_dim = self.local_dim
        elif self.sigma_input == "full_local_mean":
            sigma_in_dim = self.local_dim + self.out_dim
        else:
            raise ValueError(
                "Unknown sigma_input. Expected one of: raw_local, raw_local_mean, "
                "full_local, full_local_mean."
            )

        sigma_layers = max(int(sigma_layers), 2)
        layers: list[nn.Module] = []
        in_dim = sigma_in_dim
        for _layer_idx in range(sigma_layers - 1):
            layers.append(nn.Linear(in_dim, int(sigma_hidden)))
            layers.append(nn.GELU())
            in_dim = int(sigma_hidden)
        layers.append(nn.Linear(in_dim, self.out_dim))
        self.sigma_head = nn.Sequential(*layers)
        self._init_sigma_head(float(init_sigma))

    def _init_sigma_head(self, init_sigma: float) -> None:
        # Make initial sigma reasonable and positive. Inverse softplus.
        init_sigma = max(init_sigma - self.min_sigma, 1.0e-8)
        raw_bias = math.log(math.exp(init_sigma) - 1.0)
        last_linear = None
        for mod in self.sigma_head.modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, raw_bias)

    def _sigma_features(
        self,
        local_embedding: torch.Tensor,
        mean_prediction: torch.Tensor,
    ) -> torch.Tensor:
        raw_local = local_embedding[..., :6]
        mean_detached = mean_prediction.detach()
        if self.sigma_input == "raw_local":
            return raw_local
        if self.sigma_input == "raw_local_mean":
            return torch.cat([raw_local, mean_detached], dim=-1)
        if self.sigma_input == "full_local":
            return local_embedding
        if self.sigma_input == "full_local_mean":
            return torch.cat([local_embedding, mean_detached], dim=-1)
        raise RuntimeError(f"Invalid sigma_input={self.sigma_input}")

    def forward(
        self,
        global_embedding: torch.Tensor,
        local_embedding: torch.Tensor,
        local_positions: torch.Tensor,
        geometry: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_kw = {
            "global_embedding": global_embedding,
            "local_embedding": local_embedding,
            "local_positions": local_positions,
            **kwargs,
        }
        if geometry is not None:
            forward_kw["geometry"] = geometry

        mean = self.backbone(**forward_kw)
        sigma_features = self._sigma_features(local_embedding, mean)
        raw_sigma = self.sigma_head(sigma_features)
        sigma = F.softplus(raw_sigma) + self.min_sigma
        if self.max_sigma is not None:
            sigma = sigma.clamp_max(self.max_sigma)
        return mean, sigma
