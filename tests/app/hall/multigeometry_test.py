# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Small compute-node checks for migrated Hall calculations."""

import copy
import dataclasses
import json
import os
import sys
import tarfile
import types
from operator import itemgetter
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig, HallDiskConfig, HallTorusConfig
from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.overlap import HallOverlapWorkflow
from jaqmc.app.hall.workflow import (
    HallEvalWorkflow,
    HallTrainWorkflow,
    configure_system,
    make_estimators,
)
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.func_transform import grad_maybe_complex, hessian_maybe_complex

ROOT = Path(__file__).resolve().parents[3]
DISK_STATES = (
    [
        (
            "mhpo",
            {"envelope_type": mode, "num_heads": 1, "heads_dim": 4, "num_layers": 1},
        )
        for mode in (
            "no_m_learnable",
            "m_learnable",
            "m_learnable2",
            "m_learnable2_old",
        )
    ]
    + [("laughlin", {})]
    + [
        (f"{module}:{name}", {"kappa": 1.0})
        for module, names in (
            ("symbolic", ("DiskPhi4", "DiskPhi7", "DiskPhi8")),
            (
                "symbolic_20261021",
                ("DiskP6", "DiskP12", "DiskPsi5", "DiskPsi6", "DiskPsi7"),
            ),
            ("symbolic_round13", ("DiskP05C330", "DiskP10C459", "DiskP11C476")),
            ("symbolic_phi7_refit", ("DiskPhi7RefitC363",)),
        )
        for name in names
    ]
)


def source_module(module_name, origin):
    path = "src/" + module_name.replace(".", "/") + ".py"
    manifest = json.loads((ROOT / "integration_sources.json").read_text())
    with tarfile.open(ROOT / manifest[origin]["archive"]) as archive:
        code = archive.extractfile(path).read().decode()
    name = module_name.rsplit(".", 1)[0] + "._source_" + module_name.rsplit(".", 1)[1]
    module = types.ModuleType(name)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    exec(compile(code, f"{origin}:{path}", "exec"), module.__dict__)
    return module


def finite_summary(label, values):
    array = np.asarray(values)
    assert np.isfinite(array).all(), label
    print(
        f"{label}: shape={array.shape}, abs_max={np.max(np.abs(array), initial=0):.8g}"
    )


def run_config(tmp_path, geometry, module=None):
    config = {
        "workflow": {
            "save_path": str(tmp_path / "train"),
            "batch_size": 32,
            "seed": 21,
        },
        "system": {"flux": 3, "nspins": [2, 0]},
        "wf": {"num_heads": 1, "heads_dim": 4, "num_layers": 1},
        "train": {"run": {"iterations": 2, "burn_in": 2, "save_step_interval": 1}},
    }
    if geometry != "sphere":
        config["system"]["module"] = f"jaqmc.app.hall.config.{geometry}"
    if module:
        config["wf"]["module"] = f"jaqmc.app.hall.wavefunction.{geometry}.{module}"
    if geometry == "disk":
        config["estimators"] = {"energy": {"potential": {"radial_grid_size": 64}}}
    if geometry == "torus":
        config["system"]["tau"] = [0.0, 1.0 if module == "sigma_mhpo" else 1.2]
        config["estimators"] = {"energy": {"potential": {"ewald_gmax": 8}}}
    return config


def evaluation_config(config, tmp_path):
    result = copy.deepcopy(config)
    del result["train"]
    result["workflow"]["source_path"] = result["workflow"]["save_path"]
    result["workflow"]["save_path"] = str(tmp_path / "evaluation")
    result["run"] = {"iterations": 2, "burn_in": 2}
    return result


@pytest.mark.parametrize(
    "geometry,module",
    [("sphere", None), ("disk", "mhpo"), ("torus", "mhpo"), ("torus", "sigma_mhpo")],
)
def test_train_restore_evaluate(tmp_path, geometry, module):
    config = run_config(tmp_path, geometry, module)
    print("Configuration:", json.dumps(config))
    workflow = HallTrainWorkflow(ConfigManager(config))
    estimators = workflow.train_stage.estimators.estimators
    print("Wavefunction:", type(workflow.train_stage.wavefunction).__name__)
    print(
        "Hamiltonian:",
        type(estimators["kinetic"]).__name__,
        type(estimators["potential"]).__name__,
    )
    assert type(estimators["potential"]).__name__ == geometry.capitalize() + "Potential"
    workflow()
    assert list((tmp_path / "train").glob("train_ckpt_*.npz"))
    eval_config = evaluation_config(config, tmp_path)
    if geometry == "torus":
        eval_config["estimators"]["enabled"] = {"fubini": True}
    HallEvalWorkflow(ConfigManager(eval_config))()
    with np.load(tmp_path / "evaluation/evaluation_digest.npz") as digest:
        for key in (
            "energy:kinetic",
            "energy:potential",
            "total_energy_real",
            "total_energy_real_var",
        ):
            finite_summary(key, digest[key])
        if geometry == "torus":
            finite_summary("fubini:distance", digest["fubini:distance"])
            np.testing.assert_allclose(digest["fubini:cosine"][-1], 1, atol=2e-6)
    if geometry == "disk":
        overlap = copy.deepcopy(eval_config)
        overlap.pop("estimators")
        overlap["workflow"]["save_path"] = str(tmp_path / "overlap")
        overlap["workflow"]["reference_path"] = str(tmp_path / "train")
        HallOverlapWorkflow(ConfigManager(overlap))()
        with np.load(tmp_path / "overlap/overlap_digest.npz") as digest:
            np.testing.assert_allclose(digest["cosine"], 1, atol=2e-6)
            np.testing.assert_allclose(digest["fubini_distance"], 0, atol=2e-6)


@pytest.mark.parametrize(
    "geometry,module,options",
    [("disk", name, options) for name, options in DISK_STATES]
    + [
        ("torus", name, {"num_heads": 1, "heads_dim": 4, "num_layers": 1})
        for name in ("mhpo", "sigma_mhpo")
    ],
)
def test_source_wavefunctions_and_hamiltonians(geometry, module, options):
    nelec = 6 if "symbolic" in module else 2
    cfg = ConfigManager(
        {
            "system": {
                "module": f"jaqmc.app.hall.config.{geometry}",
                "nspins": [nelec, 0],
                "flux": 3 * nelec,
            },
            "wf": {
                "module": f"jaqmc.app.hall.wavefunction.{geometry}.{module}",
                **options,
            },
            "estimators": {
                "energy": {
                    "potential": {"radial_grid_size": 64}
                    if geometry == "disk"
                    else {"ewald_gmax": 8}
                }
            },
        }
    )
    system, wf = configure_system(cfg)
    origin = "local" if geometry == "disk" else "torus"
    original = source_module(type(wf).__module__, origin)
    kwargs = {
        f.name: getattr(wf, f.name)
        for f in dataclasses.fields(wf)
        if f.init and f.name not in ("parent", "name")
    }
    before = getattr(original, type(wf).__name__)(**kwargs)
    xy = jnp.array(
        [
            [0.13, 0.27],
            [0.61, 0.43],
            [0.36, 0.89],
            [0.82, 0.16],
            [0.94, 0.73],
            [0.21, 0.64],
        ]
    )[:nelec]
    if geometry == "disk":
        xy = 4 * xy - 2
    data = HallData(electrons=xy)
    key = jax.random.key(31)
    params = wf.init_params(data, key)
    old_template = before.init_params(data, key)
    assert jax.tree.structure(params) == jax.tree.structure(old_template)
    assert [x.shape for x in jax.tree.leaves(params)] == [
        x.shape for x in jax.tree.leaves(old_template)
    ]
    a, b = wf.logpsi(params, data), before.logpsi(params, data)
    finite_summary(type(wf).__name__, a)
    np.testing.assert_allclose(a.real, b.real, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(jnp.exp(1j * (a.imag - b.imag)), 1, atol=2e-6)
    for transform in (grad_maybe_complex, hessian_maybe_complex):
        current = transform(lambda x: wf.logpsi(params, HallData(electrons=x)))(xy)
        previous = transform(lambda x: before.logpsi(params, HallData(electrons=x)))(xy)
        finite_summary(transform.__name__, current)
        np.testing.assert_allclose(current, previous, rtol=2e-5, atol=2e-5)
    swapped = data.merge({"electrons": xy.at[0].set(xy[1]).at[1].set(xy[0])})
    np.testing.assert_allclose(jnp.exp(wf.logpsi(params, swapped) - a), -1, atol=2e-5)
    for estimator in list(make_estimators(cfg, wf, system).values())[:2]:
        source_class = getattr(
            source_module(type(estimator).__module__, origin), type(estimator).__name__
        )
        kwargs = {
            f.name: getattr(estimator, f.name)
            for f in dataclasses.fields(estimator)
            if f.init
        }
        reference = source_class(**kwargs)
        estimator.init(data, key)
        reference.init(data, key)
        current, _ = jax.jit(estimator.evaluate_single_walker)(
            params, data, {}, None, key
        )
        previous, _ = reference.evaluate_single_walker(params, data, {}, None, key)
        for name in current:
            finite_summary(name, current[name])
            np.testing.assert_allclose(
                current[name], previous[name], rtol=2e-5, atol=2e-5
            )
    jax.clear_caches()


def test_geometry_config_roundtrip():
    from serde import from_dict, to_dict

    from jaqmc.utils.module_resolver import resolve_object

    for cls in (HallConfig, HallDiskConfig, HallTorusConfig):
        cfg = cls()
        restored = from_dict(cls, to_dict(cfg))
        assert restored.flux == cfg.flux
        assert restored.nspins == cfg.nspins
    assert resolve_object("jaqmc.app.hall.config:HallConfig") is HallConfig
    assert resolve_object("jaqmc.app.hall.config.disk") is HallDiskConfig
    assert resolve_object("jaqmc.app.hall.config:HallTorusConfig") is HallTorusConfig
    assert resolve_object("jaqmc.estimator.kinetic").__name__ == "EuclideanKinetic"


def test_split_channel_defaults_match_upstream():
    from jaqmc.wavefunction.output.orbital import SplitChannelDense

    original = source_module("jaqmc.wavefunction.output.orbital", "upstream")
    current = SplitChannelDense(channels=(2, 1), features=[2, 3])
    previous = original.SplitChannelDense(channels=(2, 1), features=[2, 3])
    features = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 12
    key = jax.random.key(45)
    params = current.init(key, features)
    old_params = previous.init(key, features)
    assert jax.tree.structure(params) == jax.tree.structure(old_params)
    np.testing.assert_allclose(
        current.apply(params, features), previous.apply(params, features), atol=1e-7
    )


def evaluate_pipeline(estimators, params, electrons):
    pipeline = EstimatorPipeline(estimators)
    batched = BatchedData(
        data=HallData(electrons=electrons), fields_with_batch=["electrons"]
    )
    state = pipeline.init(batched, jax.random.key(5))
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    params = jax.device_put(params, shared)
    state = jax.device_put(state, shared)
    batched = jax.device_put(
        batched, parallel_jax.make_sharding(batched.partition_spec)
    )
    evaluate = parallel_jax.jit_sharded(
        pipeline.evaluate,
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            batched.partition_spec,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
    )
    stats, state = evaluate(
        params, batched, state, parallel_jax.shard_rngs(jax.random.key(6))
    )
    return pipeline.digest(jax.tree.map(itemgetter(None), stats), state, n_steps=1)


@pytest.mark.parametrize("geometry", ["disk", "torus"])
def test_histograms_and_distributed_counts(geometry):
    from jaqmc.app.hall.estimator.disk.pair_correlation import DiskPairCorrelation
    from jaqmc.app.hall.estimator.disk.structure_factor import DiskStructureFactor
    from jaqmc.app.hall.estimator.torus.pair_correlation import TorusPairCorrelation
    from jaqmc.estimator.density.disk import DiskDensity
    from jaqmc.estimator.density.torus import TorusDensity

    electrons = jax.random.uniform(
        jax.random.key(7), (32, 3, 2), minval=0.1, maxval=0.9
    )
    if geometry == "disk":
        estimators = {
            "density": DiskDensity(bins=8, range_max=2),
            "pair": DiskPairCorrelation(
                bins=4, max_r=2, r=0.6, reference_bin_width=1.0
            ),
            "structure": DiskStructureFactor(num_q=4, num_angles=2),
        }
    else:
        estimators = {
            "density": TorusDensity(bins_u=4, bins_v=4),
            "pair": TorusPairCorrelation(bins_u=4, bins_v=4),
        }
    digest = evaluate_pipeline(estimators, {}, electrons)
    np.testing.assert_allclose(np.sum(digest["density"]), 96)
    if geometry == "disk":
        np.testing.assert_allclose(np.sum(digest["density_radial"]), 96)
        np.testing.assert_allclose(digest["pair_correlation:sample_count"], 32)
        np.testing.assert_allclose(digest["structure_factor:sample_count"], 32)
        np.testing.assert_allclose(digest["structure_factor"][0], 3, atol=1e-6)
        np.testing.assert_allclose(
            digest["structure_factor:connected"][0], 0, atol=1e-6
        )
        valid = (np.asarray(digest["pair_correlation:rho1_target"]) > 0) & (
            np.asarray(digest["pair_correlation:rho1_reference"]) > 0
        )
        assert valid.any()
        finite_summary(
            "pair correlation in populated bins", digest["pair_correlation"][valid]
        )
        assert np.isnan(np.asarray(digest["pair_correlation"])[~valid]).all()
    else:
        np.testing.assert_allclose(
            np.mean(digest["pair_correlation"]), 2 / 3, atol=1e-6
        )
    print("Histogram counts checked on", jax.device_count(), "devices")


@pytest.mark.parametrize(
    "observable", ["one_rdm", "edge_green", "short_range", "wavefunction_scan"]
)
def test_disk_observable_workflow(tmp_path, observable):
    config = {
        "workflow": {"save_path": str(tmp_path), "batch_size": 32, "seed": 9},
        "system": {
            "module": "jaqmc.app.hall.config.disk",
            "flux": 6,
            "nspins": [2, 0],
            "interaction_strength": 0.5,
        },
        "wf": {"module": "jaqmc.app.hall.wavefunction.disk.laughlin"},
        "run": {"iterations": 2, "burn_in": 2},
        "estimators": {
            "enabled": {observable: True},
            "energy": {"potential": {"radial_grid_size": 64}},
        },
    }
    extras = {
        "one_rdm": {"num_orbitals": 3, "num_landau_levels": 1},
        "edge_green": {"n_theta": 8, "theta_min": 0.1, "theta_max": 5.9},
        "short_range": {
            "distance_edges": [0.0, 1.0, float("inf")],
            "radial_edges": [0.0, 4.0, float("inf")],
        },
        "wavefunction_scan": {
            "n_x": 4,
            "n_y": 4,
            "radial_points": 4,
            "chunk_size": 8,
            "selection_samples": 32,
        },
    }
    config["estimators"][observable] = extras[observable]
    HallEvalWorkflow(ConfigManager(config))()
    with np.load(tmp_path / "evaluation_digest.npz") as digest:
        names = [k for k in digest.files if k.startswith(observable)]
        assert names, digest.files
        for key in names:
            finite_summary(key, digest[key])
        if observable == "short_range":
            np.testing.assert_allclose(np.sum(digest["short_range:count"]), 32)
            np.testing.assert_allclose(
                np.sum(digest["short_range:sum"], axis=0),
                digest["short_range:global_sum"],
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_array_equal(digest["short_range:nonfinite_count"], 0)


def test_disk_overlap_analytic_self(tmp_path):
    config = {
        "workflow": {"save_path": str(tmp_path), "batch_size": 32, "seed": 9},
        "system": {"flux": 6, "nspins": [2, 0]},
        "wf": {"module": "jaqmc.app.hall.wavefunction.disk.laughlin"},
        "run": {"iterations": 2, "burn_in": 2},
    }
    HallOverlapWorkflow(ConfigManager(config))()
    with np.load(tmp_path / "overlap_digest.npz") as digest:
        np.testing.assert_allclose(digest["cosine"], 1, atol=2e-6)
        np.testing.assert_allclose(digest["fubini_distance"], 0, atol=2e-6)


def test_environment_report():
    from importlib.metadata import version

    print("Working directory:", Path.cwd())
    revision = json.loads(
        (ROOT / "integration_test_results/prepared_revision.json").read_text()
    )
    print("Prepared commit:", revision["commit"])
    print("Revision recorded at:", revision["prepared_at"])
    print("Working tree at preparation:", revision["working_tree"])
    print("Python:", sys.executable, sys.version)
    print(
        "Packages:",
        {
            name: version(name)
            for name in ("jax", "jaxlib", "flax", "numpy", "kfac-jax")
        },
    )
    print("Devices:", jax.devices())
    assert os.environ.get("SLURM_JOB_ID"), (
        "Run numerical integration tests through the supplied Slurm script."
    )


def test_estimator_source_outputs():
    from jaqmc.app.hall.estimator.disk.edge_green import DiskEdgeGreenDirect
    from jaqmc.app.hall.estimator.disk.fubini import DiskFubini, FubiniParams
    from jaqmc.app.hall.estimator.disk.one_rdm import DiskOneRDM
    from jaqmc.app.hall.estimator.disk.short_range import ShortRangeEstimator
    from jaqmc.app.hall.wavefunction.disk.laughlin import DiskLaughlin
    from jaqmc.estimator.kinetic.disk import DiskKinetic
    from jaqmc.utils.wiring import wire

    wf = wire(DiskLaughlin(), nspins=(2, 0), flux=6)
    data = HallData(electrons=jnp.array([[2.0, 0.0], [0.4, 0.7]]))
    params = wf.init_params(data, jax.random.key(15))
    for estimator in (
        DiskOneRDM(num_orbitals=3, num_landau_levels=1, f_log_psi=wf.logpsi),
        DiskEdgeGreenDirect(edge_radius=2.0, n_theta=8, f_log_psi=wf.logpsi),
        DiskFubini(f_log_psi=wf.logpsi),
    ):
        original = source_module(type(estimator).__module__, "local")
        cls = getattr(original, type(estimator).__name__)
        before = cls(
            **{
                f.name: getattr(estimator, f.name)
                for f in dataclasses.fields(estimator)
                if f.init
            }
        )
        estimator.init(data, jax.random.key(16))
        before.init(data, jax.random.key(16))
        p = (
            FubiniParams(params, params)
            if isinstance(estimator, DiskFubini)
            else params
        )
        current, _ = estimator.evaluate_single_walker(
            p, data, {}, None, jax.random.key(17)
        )
        previous, _ = before.evaluate_single_walker(
            p, data, {}, None, jax.random.key(17)
        )
        for name in current:
            finite_summary(name, current[name])
            np.testing.assert_allclose(
                current[name], previous[name], rtol=2e-6, atol=2e-6
            )

    batch = BatchedData(
        data=HallData(electrons=jnp.stack([data.electrons, data.electrons * 1.1])),
        fields_with_batch=["electrons"],
    )
    kinetic = DiskKinetic(f_log_psi=wf.logpsi)
    stats, _ = kinetic.evaluate_batch_walkers(
        params, batch, {}, None, jax.random.key(18)
    )
    np.testing.assert_allclose(stats["energy:kinetic"], 1, atol=2e-6)
    stats["energy:potential"] = jnp.array([-1.0, -1.1])
    stats["total_energy"] = stats["energy:kinetic"] + stats["energy:potential"]
    estimator = ShortRangeEstimator(
        kappa=0.5,
        distance_edges=[0, 1, float("inf")],
        radial_edges=[0, 4, float("inf")],
    )
    original = source_module(type(estimator).__module__, "local")
    before = original.ShortRangeEstimator(
        kappa=0.5,
        distance_edges=[0, 1, float("inf")],
        radial_edges=[0, 4, float("inf")],
    )
    current, _ = estimator.evaluate_batch_walkers(
        params, batch, stats, None, jax.random.key(19)
    )
    previous, _ = before.evaluate_batch_walkers(
        params, batch, stats, None, jax.random.key(19)
    )
    for name in current:
        np.testing.assert_allclose(current[name], previous[name], rtol=2e-6, atol=2e-6)


def test_scan_selection_global_winner():
    from jaqmc.app.hall.estimator.disk.wavefunction_scan import DiskWavefunctionScan
    from jaqmc.app.hall.wavefunction.disk.laughlin import DiskLaughlin
    from jaqmc.utils.wiring import wire

    wf = wire(DiskLaughlin(), nspins=(2, 0), flux=6)
    electrons = jax.random.normal(jax.random.key(40), (32, 2, 2))
    params = wf.init_params(HallData(electrons=electrons[0]), jax.random.key(41))
    scan = DiskWavefunctionScan(
        n_x=4,
        n_y=4,
        radial_points=4,
        chunk_size=8,
        selection_samples=32,
        f_log_psi=wf.logpsi,
    )
    digest = evaluate_pipeline({"scan": scan}, params, electrons)
    scores = jax.vmap(wf.logpsi, in_axes=(None, 0))(
        params, HallData(electrons=electrons)
    ).real
    expected = np.asarray(electrons)[np.argmax(np.asarray(scores))]
    np.testing.assert_allclose(
        digest["wavefunction_scan:fixed_electrons"], expected[1:], atol=1e-7
    )
    finite_summary("scan log amplitude", digest["wavefunction_scan:log_abs"])


def test_disk_penalty_preserves_definition():
    from jaqmc.app.hall.estimator.disk.penalized_loss import DiskPenalizedLoss

    estimator = DiskPenalizedLoss(
        lz_center=2, lz_penalty=0.3, l2_penalty=0.4, lz_var_penalty=0.2
    )
    batch = BatchedData(
        data=HallData(electrons=jnp.zeros((2, 2, 2))), fields_with_batch=["electrons"]
    )
    stats = {
        "total_energy": jnp.array([1.0, 2.0]),
        "observable:angular_momentum_z": jnp.array([1.0, 3.0]),
        "observable:angular_momentum_z_square": jnp.array([1.0, 9.0]),
    }
    current, _ = estimator.evaluate_batch_walkers(
        {}, batch, stats, None, jax.random.key(44)
    )
    expected = (
        stats["total_energy"]
        + 0.3
        * (
            stats["observable:angular_momentum_z_square"]
            - 4 * stats["observable:angular_momentum_z"]
            + 4
        )
        + 0.4 * stats["observable:angular_momentum_z_square"]
        + 0.2
        * (
            stats["observable:angular_momentum_z_square"]
            - 4 * stats["observable:angular_momentum_z"]
            + 4
        )
    )
    np.testing.assert_allclose(current["penalized_loss"], expected, atol=1e-7)
