# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check polar measure, weighted means, full kinetic derivatives and closure."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analyzing"))
from conditional_energy import (
    CUTOFFS,
    at_cutoffs,
    convergence,
    integrate,
    make_evaluator,
    polar_grid,
    refine_edges,
)


class ConditionalIntegralTest(unittest.TestCase):
    def setUp(self):
        self.edges = np.unique(np.r_[0.0, CUTOFFS, 4.0])
        self.grid = polar_grid(self.edges, 32)

    def test_polar_area_and_constant_energy(self):
        grid = self.grid
        np.testing.assert_allclose(grid["area_weight"].sum(), np.pi * 4**2, rtol=1e-14)
        energy = np.tile([7.0, 2.0, 3.0, 2.0, 0.0], (len(grid["radius"]), 1))
        result = integrate(self.edges, grid, np.zeros(len(energy), complex), energy)
        np.testing.assert_allclose(result["mean"], energy[0], atol=1e-13)
        np.testing.assert_allclose(result["node_conditional"][:, 0], 7.0, atol=1e-13)
        np.testing.assert_allclose(
            result["actual"][:, 0], 7 * result["probability"], atol=1e-14
        )
        np.testing.assert_allclose(
            at_cutoffs(result)[:, 0], 7 * CUTOFFS**2 / 16.0, atol=1e-14
        )
        np.testing.assert_allclose(result["cumulative"][-1], result["mean"], atol=1e-14)

    def test_direction_is_wavefunction_weighted(self):
        theta = self.grid["theta"]
        logpsi = 0.5 * np.log(1 + 0.6 * np.cos(theta))
        energy = np.tile(np.cos(theta)[:, None], (1, 5))
        result = integrate(self.edges, self.grid, logpsi, energy)
        np.testing.assert_allclose(result["node_conditional"], 0.3, atol=1e-14)
        np.testing.assert_allclose(result["mean"], 0.3, atol=1e-14)

    def test_gaussian_normalization_and_amplitude_scale(self):
        edges = np.linspace(0, 8, 801)
        grid = polar_grid(edges, 16)
        logpsi = -0.25 * grid["radius"] ** 2
        values = np.ones((len(logpsi), 5))
        first = integrate(edges, grid, logpsi, values)
        second = integrate(edges, grid, logpsi + 10000 + 3j, values)
        np.testing.assert_allclose(
            first["log_normalization"], np.log(2 * np.pi * (1 - np.exp(-32))), atol=1e-8
        )
        np.testing.assert_allclose(
            first["probability"], second["probability"], atol=1e-12
        )

    def test_nonfinite_is_not_discarded(self):
        logpsi = np.zeros(len(self.grid["radius"]))
        logpsi[3] = -np.inf
        with self.assertRaises(ValueError):
            integrate(self.edges, self.grid, logpsi, np.ones((len(logpsi), 5)))

    def test_convergence_and_tail(self):
        def uniform(edges, angles):
            grid = polar_grid(edges, angles)
            return integrate(
                edges,
                grid,
                np.zeros(len(grid["radius"])),
                np.ones((len(grid["radius"]), 5)),
            )

        base = uniform(self.edges, 16)
        fine = uniform(refine_edges(self.edges), 32)
        self.assertTrue(convergence(base, fine, fine)["passed"])
        self.assertFalse(
            convergence(base, fine, uniform(np.r_[refine_edges(self.edges), 6.0], 32))[
                "passed"
            ]
        )


class ExistingEnergyTest(unittest.TestCase):
    def test_gaussian_keeps_all_six_electron_derivatives(self):
        import jax
        from jax import numpy as jnp

        from jaqmc.app.hall.data import HallData
        from jaqmc.app.hall.hamiltonian.disk import DiskPotential
        from jaqmc.estimator.kinetic.disk import DiskKinetic
        from jaqmc.estimator.total_energy import TotalEnergy

        jax.config.update("jax_enable_x64", True)
        positions = jnp.array(
            [
                [0.2, 0.3],
                [1.0, 0.4],
                [-1.0, 1.0],
                [2.0, -1.0],
                [-2.0, -0.5],
                [0.5, -2.0],
            ]
        )
        logpsi = lambda params, data: (-jnp.sum(data.electrons**2) / 4).astype(
            jnp.complex128
        )
        potential = DiskPotential(
            radius=6.0,
            N=6,
            flux=18,
            d=0.1,
            interaction_strength=12.0,
            radial_grid_size=32,
        )
        data = HallData(electrons=positions)
        key = jax.random.PRNGKey(0)
        potential.init(data, key)
        _, batch = make_evaluator(logpsi, potential)
        result = jax.device_get(
            batch({}, jnp.stack([positions, positions.at[0, 0].add(0.1)]))
        )
        # Each Gaussian electron contributes 1/2, including the five fixed ones.
        np.testing.assert_allclose(result["kinetic"], 3.0, atol=1e-12)
        kinetic = DiskKinetic(f_log_psi=logpsi)
        k, _ = kinetic.evaluate_single_walker({}, data, {}, None, key)
        v, _ = potential.evaluate_single_walker({}, data, {}, None, key)
        e, _ = TotalEnergy().evaluate_single_walker({}, data, k | v, None, key)
        np.testing.assert_allclose(result["energy"][0], e["total_energy"], atol=1e-12)
        np.testing.assert_allclose(
            result["energy"],
            result["kinetic"] + result["electron_electron"] + result["background"],
            atol=1e-12,
        )
        self.assertEqual(result["energy"].dtype, np.complex128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
