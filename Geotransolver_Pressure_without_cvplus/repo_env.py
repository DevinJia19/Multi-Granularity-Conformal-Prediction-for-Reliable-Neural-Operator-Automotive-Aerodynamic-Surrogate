# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""必须在导入 physicsnemo / warp 之前调用，把内核与常见缓存目录指到仓库下，避免占满 home。"""

from __future__ import annotations

import os


def ensure_repo_local_caches() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.environ.get("WARP_CACHE_PATH"):
        os.environ["WARP_CACHE_PATH"] = os.path.join(root, ".warp_cache")
    os.makedirs(os.environ["WARP_CACHE_PATH"], exist_ok=True)
    if os.name != "nt" and not os.environ.get("XDG_CACHE_HOME"):
        xdg = os.path.join(root, ".cache")
        os.makedirs(xdg, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg
    # PhysicsNeMo 下载/注册表缓存默认在 ~/.cache/physicsnemo，易占满集群 home 配额。
    # 官方支持用 LOCAL_CACHE 改写（见 physicsnemo.core.filesystem）。
    if not os.environ.get("LOCAL_CACHE"):
        local_cache = os.path.join(root, ".cache", "physicsnemo")
        os.makedirs(local_cache, exist_ok=True)
        os.environ["LOCAL_CACHE"] = local_cache
