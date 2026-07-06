"""Transolver wrapper with a residual scale head for normalized CP.

The wrapped backbone predicts the mean surface field. A lightweight pointwise
head predicts positive sigma values, which train.py fits to detached residual
magnitudes and regularizes with the existing KNN smoothness loss.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransolverWithSigma(nn.Module):
    """Wrap a Transolver backbone and add a pointwise residual-scale head.

    The forward method accepts both the classic Transolver API
    ``model(fx=..., embedding=...)`` and the CP/Geo-style API used elsewhere in
    this repo: ``global_embedding``, ``local_embedding``, ``local_positions``.
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
        for _ in range(sigma_layers - 1):
            layers.append(nn.Linear(in_dim, int(sigma_hidden)))
            layers.append(nn.GELU())
            in_dim = int(sigma_hidden)
        layers.append(nn.Linear(in_dim, self.out_dim))
        self.sigma_head = nn.Sequential(*layers)
        self._init_sigma_head(float(init_sigma))

    def _init_sigma_head(self, init_sigma: float) -> None:
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
        fx: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        global_embedding: torch.Tensor | None = None,
        local_embedding: torch.Tensor | None = None,
        local_positions: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_embedding is None:
            local_embedding = embedding
        if local_embedding is None:
            raise RuntimeError(
                "TransolverWithSigma needs `embedding` or `local_embedding` "
                "to compute the sigma head."
            )

        if fx is not None or embedding is not None:
            mean = self.backbone(fx=fx, embedding=embedding, **kwargs)
        else:
            if global_embedding is None:
                raise RuntimeError(
                    "TransolverWithSigma needs `fx` or `global_embedding` for the backbone."
                )
            mean = self.backbone(
                fx=global_embedding,
                embedding=local_embedding,
                **kwargs,
            )

        sigma_features = self._sigma_features(local_embedding, mean)
        raw_sigma = self.sigma_head(sigma_features)
        sigma = F.softplus(raw_sigma) + self.min_sigma
        if self.max_sigma is not None:
            sigma = sigma.clamp_max(self.max_sigma)
        return mean, sigma
