# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from .disk import HallDiskConfig
from .spherical import HallConfig
from .torus import HallTorusConfig

type HallSystemConfig = HallConfig | HallDiskConfig | HallTorusConfig

__all__ = ["HallSystemConfig"]
