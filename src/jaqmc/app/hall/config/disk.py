# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from jaqmc.utils.config import configurable_dataclass

__all__ = ["HallDiskConfig"]


@configurable_dataclass
class HallDiskConfig:
    r"""Configuration for a quantum Hall system on a disk.

    Args:
        flux: Magnetic flux :math:`Q` (positive integer).
        nspins: ``(n_up, n_down)`` electron counts.
        radius: Disk radius. Defaults to :math:`\sqrt{2Q}`.
        d: Separation from the neutralizing background disk.
        interaction_type: Interaction potential form.
        interaction_strength: Scaling factor for the potential energy.
        lz_center: Target :math:`L_z` for the penalty method.
        lz_penalty: Penalty strength for
            :math:`(L_z - L_{z,0})^2`.
        l2_penalty: Penalty strength for :math:`L_z^2` on the disk.
    """

    flux: int = 18
    nspins: tuple[int, int] = (6, 0)
    radius: float | None = None
    d: float | int = 0.1
    num_orbitals: int | None = None
    interaction_strength: float = 1.0
    interaction_type: Literal["coulomb"] = "coulomb"
    lz_center: float = 0.0
    lz_penalty: float = 0.0
    l2_penalty: float = 0.0
    lz_var_penalty: float = 0.0
