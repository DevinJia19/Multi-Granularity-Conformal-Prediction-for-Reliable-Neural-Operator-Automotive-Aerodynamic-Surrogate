#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared asymmetric conformalized quantile regression utilities."""

from __future__ import annotations

import numpy as np


def one_sided_hat_q(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample conformal quantile for one-sided scores."""
    n = int(scores.size)
    if n <= 0:
        raise ValueError("Calibration scores are empty.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def asymmetric_cqr_hat_q(
    q05_cal: np.ndarray,
    q95_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float = 0.1,
):
    """Compute lower- and upper-tail asymmetric CQR corrections."""
    alpha_l = alpha / 2.0
    alpha_u = alpha / 2.0

    score_l = q05_cal - y_cal
    score_u = y_cal - q95_cal

    q_l = one_sided_hat_q(score_l, alpha_l)
    q_u = one_sided_hat_q(score_u, alpha_u)

    return q_l, q_u


def apply_asymmetric_cqr(
    q05: np.ndarray,
    q95: np.ndarray,
    q_l: float,
    q_u: float,
):
    """Apply asymmetric CQR corrections to raw quantile intervals."""
    lower = q05 - q_l
    upper = q95 + q_u
    return lower, upper
