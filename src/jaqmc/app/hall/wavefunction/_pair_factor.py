# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the sphere _pair_factor module."""

from .sphere._pair_factor import (
    _log_pair_factor_rule,
    log_pair_factor,
)

__all__ = ["log_pair_factor"]
__all__ += ["_log_pair_factor_rule"]
