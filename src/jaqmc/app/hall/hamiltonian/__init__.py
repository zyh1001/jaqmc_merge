# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Geometry-specific Hall potentials."""

from .disk import DiskPotential as DiskPotential
from .sphere import SpherePotential
from .torus import TorusPotential as TorusPotential

__all__ = ["SpherePotential"]
