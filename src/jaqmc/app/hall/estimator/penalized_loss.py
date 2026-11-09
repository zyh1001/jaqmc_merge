# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the sphere penalized_loss module."""

from .sphere.penalized_loss import (
    PenalizedLoss,
    SpherePenalizedLoss,
)

__all__ = ["SpherePenalizedLoss"]
__all__ += ["PenalizedLoss"]
