# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check angular probability weighting and honest grid failure reporting."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/analyze_disk_local_energy.py"
SPEC = importlib.util.spec_from_file_location("disk_local_energy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_angular_weighting_and_independent_radius_scaling() -> None:
    theta = (np.arange(64) + 0.5) * 2 * np.pi / 64
    logpsi = np.tile(0.5 * np.log(1 + 0.6 * np.cos(theta)), (3, 1))
    logpsi += np.array([0, -1e4, 1e4])[:, None]
    values = np.broadcast_to(np.cos(theta)[None, :, None], (3, 64, 1))
    np.testing.assert_allclose(analysis.angular_mean(logpsi, values), 0.3, atol=1e-12)


def test_nonfinite_is_rejected() -> None:
    with pytest.raises(ValueError, match="no points discarded"):
        analysis.angular_mean(np.array([[0, -np.inf]]), np.ones((1, 2, 1)))


def test_curve_comparison_records_failed_radii_and_components() -> None:
    expected = np.ones((2, 6))
    actual = expected.copy()
    actual[1, 1] += 0.1
    result = analysis.compare_curves(expected, actual, np.array([0.1, 1]), 1e-3, 1e-3)
    assert not result["passed"]
    assert result["failed_radii"] == [1]
    assert result["component_passed"][1] == [True, False, True, True, True, True]


def test_jastrow_history_tracks_actual_saved_changes(tmp_path: Path) -> None:
    files = []
    for step, value in ((1, 0.25), (2, 0.2501)):
        path = tmp_path / f"train_ckpt_{step:06d}.npz"
        params = {
            "params/params/jastrow_layer/" + name: np.array([initial])
            for name, initial in analysis.INITIAL.items()
        }
        params["params/params/jastrow_layer/beta_par"] = np.array([value])
        np.savez(path, step=step, **params)
        files.append(path)
    rows = analysis.jastrow_history(files)
    assert rows[0]["beta_par_change"] == 0
    assert rows[1]["beta_par_change"] == pytest.approx(1e-4)
    assert rows[1]["beta_anti_change"] == 0
