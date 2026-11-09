# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public geometry selection, configuration and compatibility checks."""

from dataclasses import asdict

import pytest
import yaml

from jaqmc.app.hall.config import (
    HallConfig,
    HallDiskConfig,
    HallGeometryConfig,
    HallSphereConfig,
    HallSphericalConfig,
    HallTorusConfig,
)
from jaqmc.app.hall.workflow import configure_system, make_estimators, make_sampler
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.module_resolver import resolve_object


@pytest.mark.parametrize(
    "geometry,config_class,flux,tau",
    [
        ("sphere", HallSphereConfig, 3, None),
        ("disk", HallDiskConfig, 6, None),
        ("torus", HallTorusConfig, 6, [0.0, 1.0]),
        ("torus", HallTorusConfig, 6, [0.5, 0.8660254037844386]),
    ],
)
def test_geometry_selection_and_roundtrip(geometry, config_class, flux, tau):
    system = {
        "module": f"jaqmc.app.hall.config:{config_class.__name__}",
        "nspins": [2, 0],
        "flux": flux,
        "interaction_strength": 1.25,
    }
    if geometry == "disk":
        system.update(radius=4.0, d=0.3, num_orbitals=8, lz_var_penalty=0.2)
    if tau is not None:
        system["tau"] = tau
    config = ConfigManager({"system": system})
    actual, wavefunction = configure_system(config)
    assert type(actual) is config_class
    assert isinstance(actual, HallGeometryConfig)
    assert type(wavefunction).__name__ == geometry.capitalize() + "MHPO"
    assert wavefunction.nspins == (2, 0)
    assert wavefunction.flux == flux
    if geometry == "sphere":
        assert wavefunction.monopole_strength == pytest.approx(flux / 2)
        assert wavefunction.interaction_strength == pytest.approx(1.25)
    elif geometry == "disk":
        assert wavefunction.num_orbitals == 8
        assert actual.lz_var_penalty == pytest.approx(0.2)
    else:
        assert wavefunction.tau == pytest.approx(complex(*tau))

    estimators = make_estimators(
        config, wavefunction, actual, always_enable_energy=True
    )
    assert type(estimators["potential"]).__name__ == geometry.capitalize() + "Potential"
    assert estimators["potential"].interaction_strength == pytest.approx(1.25)
    make_sampler(config, actual)
    restored = ConfigManager(yaml.safe_load(config.to_yaml())).get_module(
        "system", "jaqmc.app.hall.config:HallSphereConfig"
    )
    assert type(restored) is config_class
    assert asdict(restored) == asdict(actual)


def test_default_sphere_and_legacy_config_names():
    assert HallConfig is HallSphericalConfig is HallSphereConfig
    system, wavefunction = configure_system(ConfigManager({}))
    assert type(system) is HallSphereConfig
    assert type(wavefunction).__name__ == "SphereMHPO"
    assert HallDiskConfig().flux == 18
    assert HallDiskConfig().nspins == (6, 0)
    assert HallDiskConfig().d == pytest.approx(0.1)
    assert HallTorusConfig().tau == pytest.approx(1j)
    assert resolve_object("jaqmc.app.hall.config") is HallSphereConfig


@pytest.mark.parametrize(
    "old,new",
    [
        ("wavefunction.mhpo:MHPO", "wavefunction.sphere.mhpo:SphereMHPO"),
        (
            "wavefunction.laughlin:Laughlin",
            "wavefunction.sphere.laughlin:SphereLaughlin",
        ),
        ("wavefunction.free:Free", "wavefunction.sphere.free:SphereFree"),
        ("wavefunction:MHPO", "wavefunction.sphere.mhpo:SphereMHPO"),
        ("wavefunction.sphere.mhpo:MHPO", "wavefunction.sphere.mhpo:SphereMHPO"),
        (
            "wavefunction.sphere.base:HallWavefunction",
            "wavefunction.base:HallWavefunction",
        ),
        ("estimator.one_rdm:OneRDM", "estimator.sphere.one_rdm:SphereOneRDM"),
        (
            "estimator.pair_correlation:PairCorrelation",
            "estimator.sphere.pair_correlation:SpherePairCorrelation",
        ),
        (
            "estimator.penalized_loss:PenalizedLoss",
            "estimator.sphere.penalized_loss:SpherePenalizedLoss",
        ),
        (
            "wavefunction._dressing:DressingConfig",
            "wavefunction.sphere._dressing:DressingConfig",
        ),
    ],
)
def test_old_entrypoints_resolve_to_same_implementation(old, new):
    prefix = "jaqmc.app.hall."
    assert resolve_object(prefix + old) is resolve_object(prefix + new)


@pytest.mark.parametrize(
    "module,name",
    [
        ("wavefunction.sphere", "SphereMHPO"),
        ("wavefunction.sphere.mhpo", "SphereMHPO"),
        ("wavefunction.sphere.laughlin", "SphereLaughlin"),
        ("wavefunction.sphere.free", "SphereFree"),
        ("wavefunction.disk.mhpo", "DiskMHPO"),
        ("wavefunction.torus.mhpo", "TorusMHPO"),
        ("wavefunction.torus.sigma_mhpo", "TorusSigmaMHPO"),
        ("estimator.sphere.one_rdm", "SphereOneRDM"),
        ("estimator.sphere.pair_correlation", "SpherePairCorrelation"),
        ("estimator.sphere.penalized_loss", "SpherePenalizedLoss"),
    ],
)
def test_shorthand_selects_public_geometry_component(module, name):
    reference = "jaqmc.app.hall." + module
    assert resolve_object(reference).__name__ == name
