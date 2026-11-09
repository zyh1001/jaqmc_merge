# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical coordinates, trained-parameter compatibility and conditional energies."""

import gc
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analyzing"))

from conditional_energy import (
    COMPONENTS,
    CURVE_COMPONENTS,
    CUTOFFS,
    PLOT_RANGES,
    directional_curves,
    integrate,
    make_evaluator,
    polar_grid,
    radial_grid,
    scan_positions,
    shortest_periodic_length,
    validate_radius,
)
from conditional_energy_sources import (
    adapt_parameters,
    build_system,
    read_checkpoint_source,
    read_source,
    restore_params,
    sha256,
    verify_compatibility,
)
from evaluateConditionalEnergy import scan_pass
from reportConditionalEnergy import create_report, load_results


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)
    jax.clear_caches()
    gc.collect()


def test_native_distances_and_fixed_electrons():
    radii = np.array([0.001, 0.01, 0.1, 1.0, 2.0])
    angles = np.array([0.13, 0.7, 1.4, 2.3, 5.1])
    original = np.array(
        [[0.1, 0.2], [0.97, 0.98], [0.3, 0.4], [0.4, 0.6], [0.6, 0.7], [0.8, 0.2]]
    )
    disk = scan_positions(original, radii, angles, "disk", mobile=3, reference=1)
    np.testing.assert_allclose(
        np.linalg.norm(disk[:, 3] - original[1], axis=-1), radii, atol=1e-14
    )
    lattice = np.array([[12.0, 0.0], [8.0, 10.0]])
    torus = scan_positions(
        original, radii, angles, "torus", mobile=3, reference=1, lattice=lattice
    )
    from jaqmc.geometry.pbc import build_distance_fn

    _, distance = build_distance_fn(jnp.asarray(lattice))(
        jnp.asarray(torus[:, 3:4] @ lattice), jnp.asarray(original[1:2] @ lattice)
    )
    np.testing.assert_allclose(np.asarray(distance).ravel(), radii, atol=1e-13)
    assert (torus[:, 3] >= 0).all() and (torus[:, 3] < 1).all()
    sphere = scan_positions(
        original, radii, angles, "sphere", mobile=3, reference=1, sphere_radius=3.0
    )

    def xyz(x):
        t, p = x[..., 0], x[..., 1]
        return np.stack(
            [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1
        )

    np.testing.assert_allclose(
        3 * np.linalg.norm(xyz(sphere[:, 3]) - xyz(original[1]), axis=-1),
        radii,
        atol=1e-14,
    )
    for positions in (disk, torus, sphere):
        np.testing.assert_array_equal(
            positions[:, np.arange(6) != 3],
            np.broadcast_to(original[np.arange(6) != 3], (len(radii), 5, 2)),
        )
    # The shortest vector need not be a primitive vector of the supplied basis.
    assert shortest_periodic_length(
        np.array([[100.0, 0.0], [99.0, 1.0]])
    ) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="smaller"):
        validate_radius("torus", 2.0, lattice=np.diag([3.0, 20.0]))
    with pytest.raises(ValueError, match="smaller"):
        validate_radius("sphere", 2.0, sphere_radius=1.0)


def test_angular_weights_and_interval_boundaries():
    grid = radial_grid(8, 32)
    for boundary in [0.001, 0.01, 0.1, 1.0, 2.0]:
        assert boundary in grid["radius"]
    angle = grid["theta"]
    energy = np.cos(angle) + 0.2j * np.sin(angle)
    raw = dict(
        logpsi=0.5 * np.log(1 + 0.6 * np.cos(angle)) - 10000 * np.log(grid["radius"]),
        energy=energy,
        kinetic=energy - 2,
        potential=np.full(len(angle), 2.0),
    )
    data = directional_curves(grid["radius"], raw)
    assert tuple(data["components"]) == CURVE_COMPONENTS
    np.testing.assert_allclose(data["node_conditional"][:, 0], 0.3, atol=2e-10)
    np.testing.assert_allclose(
        data["node_conditional"][:, 1:3].sum(axis=1), 0.3, atol=2e-10
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        directional_curves(
            grid["radius"], raw | {"energy": np.full(len(angle), np.nan)}
        )


def test_parameter_adaptation_is_narrow():
    kernel = "params/params/backbone_layer/Dense_0/kernel"
    saved = {kernel: np.arange(15.0).reshape(5, 3)}
    actual, changes = adapt_parameters(saved, {kernel: np.zeros((9, 3))}, "torus")
    np.testing.assert_array_equal(actual[kernel][:4], saved[kernel][:4])
    np.testing.assert_array_equal(actual[kernel][4:8], 0.0)
    np.testing.assert_array_equal(actual[kernel][8], saved[kernel][4])
    assert changes[0]["kind"] == "zero_center_of_mass_weights"
    with pytest.raises(ValueError, match="Invalid parameter"):
        adapt_parameters(saved, {kernel: np.zeros((8, 3))}, "torus")
    with pytest.raises(ValueError, match="names differ"):
        adapt_parameters(
            saved | {"unexpected": np.ones(1)}, {kernel: np.zeros((9, 3))}, "torus"
        )


@pytest.mark.parametrize("geometry", ["disk", "sphere", "torus"])
def test_existing_six_electron_energy_and_compatibility(geometry, tmp_path):
    from jaqmc.app.hall.data import HallData

    modules = {
        "disk": "jaqmc.app.hall.config:HallDiskConfig",
        "sphere": "jaqmc.app.hall.config:HallConfig",
        "torus": "jaqmc.app.hall.config:HallTorusConfig",
    }
    wavefunctions = {
        "disk": "jaqmc.app.hall.wavefunction.disk.mhpo",
        "sphere": "jaqmc.app.hall.wavefunction.sphere.mhpo",
        "torus": "jaqmc.app.hall.wavefunction.torus.mhpo",
    }
    system = dict(
        module=modules[geometry], nspins=[6, 0], flux=18, interaction_strength=1.0
    )
    if geometry == "torus":
        system["tau"] = [0.31, 1.07]
    source = dict(
        geometry=geometry,
        config=dict(
            system=system,
            wf=dict(
                module=wavefunctions[geometry],
                num_heads=1,
                heads_dim=4,
                num_layers=1,
                ndets=1,
            ),
            estimators=dict(energy={}),
        ),
    )
    _, wf, estimators, _ = build_system(source)
    positions = jnp.array(
        [
            [0.17, 0.23],
            [0.29, 0.31],
            [0.41, 0.43],
            [0.53, 0.59],
            [0.67, 0.71],
            [0.79, 0.83],
        ]
    )
    example = HallData(electrons=positions)
    params = wf.init_params(example, jax.random.PRNGKey(7))
    from jaqmc.utils.checkpoint import tree_to_npz

    saved = {
        key: np.asarray(value) for key, value in tree_to_npz({"params": params}).items()
    }
    if geometry == "disk":
        del saved["params/params/orbital_layer/amplitude_imag"]
    if geometry == "torus":
        key = "params/params/backbone_layer/Dense_0/kernel"
        saved[key] = np.r_[saved[key][:4], saved[key][8:9]]
    checkpoint = tmp_path / "trained.npz"
    np.savez(checkpoint, **saved)
    restored, info = restore_params(wf, example, checkpoint, geometry)
    assert verify_compatibility(wf, restored, positions, info["compatibility_changes"])[
        "passed"
    ]
    key = jax.random.PRNGKey(0)
    potential = estimators["potential"]
    potential.init(example, key)
    single, batch = make_evaluator(wf.logpsi, potential, kinetic=estimators["kinetic"])
    single = jax.jit(single)
    result = single(restored, positions)

    @jax.jit
    def original(p, x):
        data = HallData(electrons=x)
        k, _ = estimators["kinetic"].evaluate_single_walker(p, data, {}, None, key)
        v, _ = potential.evaluate_single_walker(p, data, {}, None, key)
        return k["energy:kinetic"] + v["energy:potential"]

    np.testing.assert_allclose(
        result["energy"], original(restored, positions), atol=1e-8
    )
    np.testing.assert_allclose(
        batch(restored, positions[None])["energy"][0],
        result["energy"],
        rtol=1e-8,
        atol=1e-8,
    )
    if geometry == "torus":
        shifted = positions.at[0].add(jnp.array([1.0, -1.0]))
        translated = single(restored, shifted)
        np.testing.assert_allclose(
            translated["energy"], result["energy"], rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            translated["logpsi"].real, result["logpsi"].real, atol=1e-10
        )


def test_nonfinite_scan_is_saved(tmp_path):
    source = dict(
        geometry="disk",
        background=np.arange(12.0).reshape(6, 2).tolist(),
        mobile_electron=0,
        probe_electron=1,
        config=dict(system={}),
    )

    def invalid(params, positions):
        return {
            key: jnp.full(
                len(positions),
                jnp.nan,
                dtype=(jnp.complex128 if key != "potential" else jnp.float64),
            )
            for key in (
                "logpsi",
                "energy",
                "kinetic",
                "potential",
                "electron_electron",
                "background",
            )
        }

    # Match the real disk-only component dtypes.
    def evaluate(params, positions):
        data = invalid(params, positions)
        for key in ("electron_electron", "background"):
            data[key] = data[key].real
        return data

    with pytest.raises(ValueError, match="Nonfinite"):
        scan_pass(
            tmp_path,
            "invalid",
            dict(radius=np.array([0.1]), theta=np.array([0.2])),
            source,
            {},
            evaluate,
            1,
        )
    with h5py.File(tmp_path / "invalid_grid.h5", "r") as saved:
        assert not saved.attrs["complete"]
        np.testing.assert_array_equal(saved["error_indices"][:], [0])
        assert np.isfinite(saved["electrons"][:]).all()
        assert np.isnan(saved["energy"][:]).all()


def test_merged_source_and_checkpoint_hash(tmp_path):
    import yaml

    checkpoint = tmp_path / "train_ckpt_000010.npz"
    np.savez(
        checkpoint,
        step=10,
        **{"batched_data/data/electrons": np.arange(24.0).reshape(2, 6, 2)},
    )
    cfg = dict(
        system=dict(module="disk", flux=18, nspins=[6, 0], interaction_strength=1.0),
        wf={},
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                source_root=str(tmp_path),
                files=[
                    dict(
                        file_id="C",
                        geometry="disk",
                        category="checkpoint",
                        path=checkpoint.name,
                        sha256=sha256(checkpoint),
                    )
                ],
            )
        )
    )
    (tmp_path / "schema.json").write_text(json.dumps(dict(geometry_groups=["disk"])))
    with h5py.File(tmp_path / "unified.h5", "w") as h:
        meta = h.create_group("disk/metadata")
        for key, value in dict(
            config_yaml=yaml.safe_dump(cfg),
            final_checkpoint_step=10,
            flux=18,
            nspins=[6, 0],
            interaction_strength=1.0,
            n_electrons=6,
            run_id="fixture",
            coordinate_system="x,y",
        ).items():
            meta.create_dataset(key, data=value)
        meta["config_yaml"].attrs.update(status="available", source_files='["C"]')
    source = read_source(tmp_path, "disk", walker_index=1)
    np.testing.assert_array_equal(
        source["background"], np.arange(12.0, 24.0).reshape(6, 2)
    )
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    standalone = read_checkpoint_source(config_path, checkpoint, "disk", walker_index=1)
    np.testing.assert_array_equal(standalone["background"], source["background"])
    assert standalone["config_sha256"] == sha256(config_path)
    assert standalone["checkpoint_sha256"] == source["checkpoint_sha256"]
    assert standalone["checkpoint_step"] == 10
    assert "unified_sha256" not in standalone
    with pytest.raises(ValueError, match="geometry disagrees"):
        read_checkpoint_source(config_path, checkpoint, "sphere")
    np.savez(checkpoint, step=11)
    with pytest.raises(ValueError, match="filename disagrees"):
        read_checkpoint_source(config_path, checkpoint, "disk")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        read_source(tmp_path, "disk")


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="Plot environment requires matplotlib",
)
@pytest.mark.parametrize("geometry", ["disk", "sphere", "torus"])
def test_all_segment_plots_preserve_data(tmp_path, geometry):
    directory = tmp_path / "scan"
    directory.mkdir()
    radius = np.unique(
        np.r_[np.geomspace(0.001, 1.0, 33), np.linspace(1.0, 2.0, 17), 0.01, 0.1]
    )
    values = np.stack(
        [1 / radius - 2, 0.4 / radius - 3, 0.6 / radius + 1, np.zeros(len(radius))],
        axis=-1,
    )
    np.savez(
        directory / "curves_refined.npz",
        node_radius=radius,
        node_conditional=values,
        components=np.asarray(CURVE_COMPONENTS),
    )
    manifest = dict(
        geometry=geometry,
        model="fixture",
        config=dict(system=dict(interaction_strength=1.0)),
        settings=dict(system=dict(flux=18), estimators=dict(energy={})),
        mobile_electron=0,
        probe_electron=1,
        background=np.zeros((6, 2)).tolist(),
    )
    if geometry == "disk":
        edges = np.unique(np.r_[0.0, CUTOFFS, 4.0])
        grid = polar_grid(edges, 8)
        r = grid["radius"]
        integrals = integrate(
            edges,
            grid,
            -(r**2) / 4,
            np.stack(
                [
                    1 / r - 2,
                    0.4 / r - 3,
                    0.6 / r + 2,
                    -np.ones(len(r)),
                    np.zeros(len(r)),
                ],
                axis=-1,
            ),
        )
        np.savez(
            directory / "integrals.npz", components=np.asarray(COMPONENTS), **integrals
        )
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    (directory / "validation.json").write_text(
        json.dumps(
            dict(
                complete=True,
                identities_passed=True,
                convergence=dict(
                    angular=dict(passed=True), disk_integrals=dict(passed=True)
                ),
                files_sha256={
                    path.name: sha256(path) for path in directory.glob("*.npz")
                },
            )
        )
    )
    groups = load_results([directory])[geometry]
    output = tmp_path / "report"
    create_report(groups, output)
    scales = json.loads((output / "plot_scales.json").read_text())
    for low, high, suffix, xscale in PLOT_RANGES:
        key = f"conditional_energy_r_{suffix}"
        for ext in ("png", "pdf"):
            assert (output / "figures" / f"{key}.{ext}").stat().st_size > 100
        assert scales[key][0]["x_scale"] == xscale
        np.testing.assert_array_equal(scales[key][0]["x_limits"], [low, high])
    np.testing.assert_array_equal(groups[0]["curves"]["node_conditional"], values)
    text = (output / "固定背景能量分析报告.md").read_text()
    if geometry == "disk":
        for key in (
            "probability",
            "actual_contribution",
            "cumulative_energy",
            "conditional_energy_details",
            "bulk_linear",
            "differences",
            "components",
            "background",
        ):
            assert (output / "figures" / f"{key}.png").exists()
    else:
        assert "没有计算完整空间概率" in text
        assert not (output / "figures/cumulative_energy.png").exists()
    import csv

    with (output / "逐距离曲线.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_allclose([float(row["总能量"]) for row in rows], values[:, 0])
