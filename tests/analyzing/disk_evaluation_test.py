# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""检查圆盘评估的模型继承、采样数量及产物验证."""

import argparse
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "analyzing/evaluateDisk.py"
SPEC = importlib.util.spec_from_file_location("evaluate_disk_entrypoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.fixture
def train_config() -> dict:
    return {
        "system": {
            "module": "jaqmc.app.hall.config:HallDiskConfig",
            "nspins": [6, 0],
            "flux": 18,
            "num_orbitals": 18,
            "interaction_strength": 1.0,
        },
        "wf": {
            "module": "jaqmc.app.hall.wavefunction.disk.mhpo",
            "ndets": 1,
            "num_layers": 2,
        },
        "sampler": {"steps": 10},
        "workflow": {"batch_size": 4096, "restore_path": "old_training"},
        "estimators": {
            "enabled": {"energy": True, "density": True},
            "energy": {
                "kinetic": {"include_energy": True},
                "potential": {"interaction_strength": 1.0},
            },
            "density": {"bins": 12},
        },
        "train": {"run": {"iterations": 150000}},
    }


def arguments(quantity: str, **updates) -> argparse.Namespace:
    args = ENTRY.parse_args([quantity, "/source"])
    for name, value in updates.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize(
    "quantity", ["energy", "density", "pair_correlation", "wave_scan"]
)
def test_config_preserves_model_and_enables_only_requested_quantity(
    train_config, quantity
):
    original = deepcopy(train_config)
    cfg = ENTRY.build_config(
        train_config, arguments(quantity), Path("/checkpoint.npz"), 8192
    )
    assert cfg["system"] == original["system"]
    assert cfg["wf"] == original["wf"]
    assert cfg["workflow"]["batch_size"] == 8192
    assert cfg["workflow"]["source_path"] == "/checkpoint.npz"
    assert "restore_path" not in cfg["workflow"]
    assert "train" not in cfg
    enabled = {key for key, value in cfg["estimators"]["enabled"].items() if value}
    assert enabled == (
        {"energy", "angular_momentum"}
        if quantity == "energy"
        else {ENTRY.OBSERVABLE_NAMES.get(quantity, quantity)}
    )
    assert train_config == original


def test_actual_batch_and_protected_overrides_are_enforced(train_config):
    with pytest.raises(ValueError, match="8192"):
        ENTRY.build_config(
            train_config, arguments("density"), Path("/checkpoint.npz"), 4096
        )
    for override in (
        "workflow.batch_size=1024",
        "system.num_orbitals=20",
        "estimators.enabled.energy=true",
    ):
        with pytest.raises(ValueError):
            ENTRY.build_config(
                train_config,
                arguments("density", overrides=[override]),
                Path("/checkpoint.npz"),
                8192,
            )
    cfg = ENTRY.build_config(
        train_config,
        arguments("density", overrides=["estimators.density.bins=512"]),
        Path("/checkpoint.npz"),
        8192,
    )
    assert cfg["estimators"]["density"]["bins"] == 512


def test_scan_rejects_insufficient_selection_samples(train_config):
    with pytest.raises(ValueError, match="背景选择"):
        ENTRY.build_config(
            train_config,
            arguments(
                "wave_scan",
                overrides=["estimators.wavefunction_scan.selection_samples=16384"],
            ),
            Path("/checkpoint.npz"),
            8192,
        )


def test_density_validation_rejects_double_normalization(tmp_path):
    coordinates = np.zeros((8192, 6, 2), dtype=np.float32)
    source = tmp_path / "source.npz"
    np.savez(
        source,
        **{
            "step": 149999,
            "batched_data/data/electrons": coordinates,
            "params/weight": np.array([1.0]),
        },
    )
    output = tmp_path / "density"
    output.mkdir()
    np.savez(
        output / "evaluation_ckpt_000001.npz",
        **{
            "step": 1,
            "batched_data/data/electrons": coordinates,
            "params/weight": np.array([1.0]),
        },
    )
    config = {
        "workflow": {"batch_size": 8192},
        "run": {"iterations": 2},
        "system": {"nspins": [6, 0]},
    }
    for factor in (1, 2):
        np.savez(
            output / "evaluation_digest.npz",
            **{
                "density": np.array([[6 * 8192 * 2 * factor]], dtype=float),
                "density:n_steps": 2,
                "density_radial": np.array([6 * 8192 * 2 * factor], dtype=float),
                "density_radial:n_steps": 2,
            },
        )
        if factor == 1:
            result = ENTRY.validate_result(config, "density", output, source)
            assert result["integrated_number_grid"] == 6
            assert result["parameters_unchanged"]
        else:
            with pytest.raises(ValueError, match="密度积分"):
                ENTRY.validate_result(config, "density", output, source)


def test_analysis_only_for_wave_scan(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        ENTRY.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    ENTRY.run_analysis("wave_scan", tmp_path, tmp_path / "wavefunction_scan")
    command, options = commands[0]
    assert "--analysis-only" in command
    assert options["env"]["JAX_PLATFORMS"] == "cpu"
