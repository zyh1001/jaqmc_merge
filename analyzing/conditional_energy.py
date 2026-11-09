# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Conditional local energies and the existing disk polar quadrature."""

from typing import Any

import numpy as np

COMPONENTS = (
    "energy",
    "kinetic",
    "electron_electron",
    "background",
    "energy_imaginary",
)
CUTOFFS = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
CURVE_COMPONENTS = ("energy", "kinetic", "potential", "energy_imaginary")
PLOT_RANGES = (
    (1e-3, 1e-2, "1e-3_1e-2", "log"),
    (1e-2, 1e-1, "1e-2_1e-1", "log"),
    (1e-1, 1.0, "1e-1_1", "log"),
    (1.0, 2.0, "1_2", "linear"),
)


def radial_edges(center, radial_bins=200, outer_step=0.2, outer_margin=10.0):
    maximum = np.linalg.norm(center) + outer_margin
    if radial_bins < 2 or outer_step <= 0 or outer_margin <= 3:
        raise ValueError("Invalid radial grid settings")
    outer = np.linspace(3.0, maximum, int(np.ceil((maximum - 3.0) / outer_step)) + 1)
    return np.unique(
        np.r_[
            0.0, np.geomspace(1e-3, 3.0, radial_bins + 1), outer, CUTOFFS, 0.001, 0.01
        ]
    )


def refine_edges(edges):
    """Refine the polar integration grid.

    Returns:
        Sorted edges of two equal-area subannuli per input annulus.
    """
    middle = np.sqrt((edges[:-1] ** 2 + edges[1:] ** 2) / 2)
    return np.sort(np.r_[edges, middle])


def polar_grid(edges, angles):
    """Place two Gauss nodes in each annulus using the area coordinate.

    Returns:
        Flattened radii, angles, annular indices and polar area weights.

    Raises:
        ValueError: If annular edges or the angular resolution are invalid.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if angles < 4 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Invalid annulus edges or angular grid")
    t = edges**2 / 2
    midpoint = (t[:-1] + t[1:]) / 2
    halfwidth = np.diff(t) / 2
    nodes = midpoint[:, None] + halfwidth[:, None] * np.array([-1.0, 1.0]) / np.sqrt(3)
    radii = np.sqrt(2 * nodes)
    theta = (np.arange(angles) + 0.5) * (2 * np.pi / angles)
    shape = (len(edges) - 1, 2, angles)
    return dict(
        radius=np.broadcast_to(radii[:, :, None], shape).copy().ravel(),
        theta=np.broadcast_to(theta, shape).copy().ravel(),
        area_weight=np.broadcast_to(
            halfwidth[:, None, None] * (2 * np.pi / angles), shape
        )
        .copy()
        .ravel(),
        bin_index=np.broadcast_to(np.arange(shape[0])[:, None, None], shape)
        .copy()
        .ravel(),
    )


def integrate(edges, grid, logpsi, values):
    """Integrate disk observables using one amplitude scale for the whole domain.

    Returns:
        Normalized probabilities, energy contributions and conditional curves.

    Raises:
        ValueError: If an evaluation is nonfinite or normalization is not positive.
    """
    logpsi, values = np.asarray(logpsi), np.asarray(values)
    if not (np.isfinite(logpsi).all() and np.isfinite(values).all()):
        raise ValueError(
            "A node or nonfinite local energy was found; do not discard the point"
        )
    shift = float(np.max(logpsi.real))
    weight = grid["area_weight"] * np.exp(2 * (logpsi.real - shift))
    normalization = weight.sum()
    if not normalization > 0:
        raise ValueError("Conditional normalization must be positive")
    bins = len(edges) - 1
    mass = np.bincount(grid["bin_index"], weights=weight, minlength=bins)
    sums = np.stack(
        [
            np.bincount(grid["bin_index"], weights=weight * column, minlength=bins)
            for column in values.T
        ],
        axis=-1,
    )
    probability = mass / normalization
    actual = sums / normalization
    conditional = np.full_like(sums, np.nan)
    np.divide(sums, mass[:, None], out=conditional, where=mass[:, None] > 0)
    # Keep the two radial nodes separate for the curve at a specified radius.
    nodes = np.unique(grid["radius"])
    node_index = np.searchsorted(nodes, grid["radius"])
    angular_weight = np.exp(2 * (logpsi.real - shift))
    angular_mass = np.bincount(node_index, weights=angular_weight)
    angular_sums = np.stack(
        [
            np.bincount(node_index, weights=angular_weight * column)
            for column in values.T
        ],
        axis=-1,
    )
    node_mean = np.full_like(angular_sums, np.nan)
    np.divide(
        angular_sums,
        angular_mass[:, None],
        out=node_mean,
        where=angular_mass[:, None] > 0,
    )
    angles = len(np.unique(grid["theta"]))
    density = 2 * np.pi * nodes * angular_mass / angles / normalization
    node_actual = 2 * np.pi * nodes[:, None] * angular_sums / angles / normalization
    return dict(
        edges=np.asarray(edges),
        radius=(edges[:-1] + edges[1:]) / 2,
        probability=probability,
        probability_density=probability / np.diff(edges),
        conditional=conditional,
        actual=actual,
        contribution_density=actual / np.diff(edges)[:, None],
        cumulative=np.cumsum(actual, axis=0),
        cumulative_probability=np.cumsum(probability),
        mean=actual.sum(axis=0),
        log_normalization=2 * shift + np.log(normalization),
        node_radius=nodes,
        node_conditional=node_mean,
        node_probability_density=density,
        node_contribution_density=node_actual,
    )


def at_cutoffs(result):
    indices = np.searchsorted(result["edges"], CUTOFFS)
    if not np.allclose(result["edges"][indices], CUTOFFS, atol=1e-13, rtol=0):
        raise ValueError("Exact cutoff edges are required")
    return result["cumulative"][indices - 1]


def convergence(base, refined, extended, norm_tolerance=1e-5, energy_tolerance=1e-3):
    grid_norm = abs(np.expm1(base["log_normalization"] - refined["log_normalization"]))
    tail = extended["edges"][:-1] >= refined["edges"][-1]
    tail_mass = extended["probability"][tail].sum()
    # Test energy components separately so cancellation cannot hide an error.
    grid_mean = np.max(np.abs(base["mean"] - refined["mean"]))
    tail_mean = np.max(np.abs(extended["mean"] - refined["mean"]))
    grid_curve = np.max(np.abs(at_cutoffs(base) - at_cutoffs(refined)))
    tail_curve = np.max(np.abs(at_cutoffs(extended) - at_cutoffs(refined)))
    passed = (
        grid_norm < norm_tolerance
        and 0 <= tail_mass < norm_tolerance
        and max(grid_mean, tail_mean, grid_curve, tail_curve) < energy_tolerance
    )
    return dict(
        passed=bool(passed),
        relative_normalization_change=float(grid_norm),
        added_tail_probability=float(tail_mass),
        max_component_mean_change=float(grid_mean),
        max_tail_mean_change=float(tail_mean),
        max_cutoff_change=float(grid_curve),
        max_tail_cutoff_change=float(tail_curve),
        normalization_tolerance=norm_tolerance,
        energy_tolerance=energy_tolerance,
        interpretation=(
            "deterministic_grid_and_finite_domain_checks; not statistical errors"
        ),
    )


def make_evaluator(logpsi, potential, *, kinetic=None):
    """Retain pointwise values from the existing six-electron energy routines.

    Returns:
        Single-configuration evaluator and its JIT-compiled batched form.
    """
    import jax
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData
    from jaqmc.app.hall.hamiltonian.disk import DiskPotential
    from jaqmc.estimator.kinetic.disk import DiskKinetic
    from jaqmc.estimator.total_energy import TotalEnergy

    if kinetic is None:
        kinetic = DiskKinetic(f_log_psi=logpsi)
    disk = isinstance(potential, DiskPotential)
    total = TotalEnergy(components=["energy:kinetic", "energy:potential"])
    key = jax.random.PRNGKey(0)

    def single(params, positions):
        data = HallData(electrons=positions)
        k, _ = kinetic.evaluate_single_walker(params, data, {}, None, key)
        v, _ = potential.evaluate_single_walker(params, data, {}, None, key)
        e, _ = total.evaluate_single_walker(params, data, k | v, None, key)
        energy = e["total_energy"]
        result = dict(
            logpsi=logpsi(params, data),
            energy=energy,
            kinetic=k["energy:kinetic"],
            potential=v["energy:potential"],
        )
        if disk:
            i, j = jnp.triu_indices(positions.shape[0], 1)
            vee = potential.interaction_strength * jnp.sum(
                1 / jnp.linalg.norm(positions[i] - positions[j], axis=-1)
            )
            result.update(electron_electron=vee, background=v["energy:potential"] - vee)
        return result

    return single, jax.jit(jax.vmap(single, in_axes=(None, 0)))


def shortest_periodic_length(lattice: np.ndarray) -> float:
    """Reduce a two-dimensional lattice to its shortest nonzero vector.

    Returns:
        Length of the shortest nonzero periodic displacement.

    Raises:
        ValueError: If the lattice is invalid or reduction does not converge.
    """
    basis = np.asarray(lattice, dtype=np.float64).copy()
    if (
        basis.shape != (2, 2)
        or not np.isfinite(basis).all()
        or np.linalg.det(basis) == 0
    ):
        raise ValueError("Expected a finite, nonsingular two-dimensional lattice")
    for _ in range(100):
        if np.dot(basis[1], basis[1]) < np.dot(basis[0], basis[0]):
            basis = basis[::-1].copy()
        multiple = np.rint(np.dot(basis[0], basis[1]) / np.dot(basis[0], basis[0]))
        if multiple == 0:
            return float(np.linalg.norm(basis[0]))
        basis[1] -= multiple * basis[0]
    raise ValueError("Lattice reduction did not converge")


def validate_radius(
    geometry: str, maximum: float, *, lattice=None, sphere_radius=None
) -> None:
    """Reject radial circles that cannot represent the requested distance.

    Raises:
        ValueError: If the geometry, cell size or radius is invalid.
    """
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Scan radius must be positive and finite")
    if geometry == "torus":
        limit = shortest_periodic_length(lattice) / 2
    elif geometry == "sphere":
        if (
            sphere_radius is None
            or not np.isfinite(sphere_radius)
            or sphere_radius <= 0
        ):
            raise ValueError("Sphere radius must be positive and finite")
        limit = 2 * sphere_radius
    elif geometry == "disk":
        return
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    if maximum >= limit:
        raise ValueError(
            f"Requested radius {maximum:g} must be smaller than "
            f"{limit:g} for {geometry}"
        )


def scan_positions(
    background: np.ndarray,
    radius: np.ndarray,
    angle: np.ndarray,
    geometry: str,
    *,
    mobile: int = 0,
    reference: int = 1,
    lattice=None,
    sphere_radius=None,
) -> np.ndarray:
    """Generate native-coordinate configurations with five electrons fixed.

    Returns:
        Full six-electron configurations in the original coordinate convention.

    Raises:
        ValueError: If positions, indices or requested distances are invalid.
    """
    background = np.asarray(background, dtype=np.float64)
    radius, angle = np.asarray(radius), np.asarray(angle)
    if background.shape != (6, 2) or not np.isfinite(background).all():
        raise ValueError("Expected six finite native-coordinate electron positions")
    if not (0 <= mobile < 6 and 0 <= reference < 6 and mobile != reference):
        raise ValueError("Mobile and reference indices must be distinct, from 0 to 5")
    if radius.ndim != 1 or radius.shape != angle.shape or not len(radius):
        raise ValueError(
            "Radii and angles must be nonempty one-dimensional arrays of equal size"
        )
    if (
        not np.isfinite(radius).all()
        or not np.isfinite(angle).all()
        or np.any(radius <= 0)
    ):
        raise ValueError("Radii must be positive and all scan coordinates finite")
    validate_radius(
        geometry, float(radius.max()), lattice=lattice, sphere_radius=sphere_radius
    )
    center = background[reference]
    offset = radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    if geometry == "disk":
        moving = center + offset
    elif geometry == "torus":
        moving = (center + offset @ np.linalg.inv(lattice)) % 1.0
    else:
        theta, phi = center
        normal = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = np.cross(normal, axis)
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        gamma = 2 * np.arcsin(radius / (2 * sphere_radius))
        direction = np.cos(angle)[:, None] * tangent + np.sin(angle)[:, None] * other
        xyz = np.cos(gamma)[:, None] * normal + np.sin(gamma)[:, None] * direction
        moving = np.stack(
            [
                np.arctan2(np.linalg.norm(xyz[:, :2], axis=-1), xyz[:, 2]),
                np.arctan2(xyz[:, 1], xyz[:, 0]),
            ],
            axis=-1,
        )
    positions = np.broadcast_to(background, (len(radius), 6, 2)).copy()
    positions[:, mobile] = moving
    return positions


def radial_grid(radial_bins: int = 200, angles: int = 64) -> dict[str, np.ndarray]:
    """Build radial curves containing every requested interval boundary.

    Returns:
        Flattened radius and angular nodes excluding coincident electrons.

    Raises:
        ValueError: If fewer than four radial bins or angular directions are requested.
    """
    if radial_bins < 4 or angles < 4:
        raise ValueError("At least four radial bins and four angles are required")
    radii = np.unique(
        np.r_[
            np.geomspace(0.001, 1.0, radial_bins + 1),
            np.linspace(1.0, 2.0, max(4, radial_bins // 3) + 1),
            0.001,
            0.01,
            0.1,
            1.0,
            2.0,
        ]
    )
    theta = (np.arange(angles) + 0.5) * 2 * np.pi / angles
    return dict(radius=np.repeat(radii, angles), theta=np.tile(theta, len(radii)))


def directional_curves(
    radius: np.ndarray, raw: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Average complex local energies with separately scaled angular weights.

    Returns:
        Sorted radii and probability-weighted directional energy means.

    Raises:
        ValueError: If the input contains any nonfinite values.
    """
    radius = np.asarray(radius)
    if not np.isfinite(radius).all() or any(
        not np.isfinite(x).all() for x in raw.values()
    ):
        raise ValueError(
            "Nonfinite scan data cannot be discarded from angular averages"
        )
    nodes, inverse = np.unique(radius, return_inverse=True)
    shift = np.full(len(nodes), -np.inf)
    np.maximum.at(shift, inverse, raw["logpsi"].real)
    weight = np.exp(2 * (raw["logpsi"].real - shift[inverse]))
    mass = np.bincount(inverse, weights=weight)
    values = np.stack(
        [
            raw["energy"].real,
            raw["kinetic"].real,
            raw["potential"].real,
            raw["energy"].imag,
        ],
        axis=-1,
    )
    sums = np.stack(
        [np.bincount(inverse, weights=weight * column) for column in values.T], axis=-1
    )
    return dict(
        node_radius=nodes,
        node_conditional=sums / mass[:, None],
        components=np.asarray(CURVE_COMPONENTS),
    )
