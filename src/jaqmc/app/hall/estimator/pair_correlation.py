# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the sphere pair_correlation module."""

from .sphere.pair_correlation import (
    PairCorrelation,
    SpherePairCorrelation,
)

__all__ = ["SpherePairCorrelation"]
__all__ += ["PairCorrelation"]
