# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compute six-electron conditional energies from merged data or a training state."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from conditional_energy import (
    COMPONENTS,
    convergence,
    directional_curves,
    integrate,
    make_evaluator,
    polar_grid,
    radial_edges,
    radial_grid,
    refine_edges,
    scan_positions,
    validate_radius,
)
from conditional_energy_sources import (
    GEOMETRIES,
    build_system,
    read_checkpoint_source,
    read_source,
    restore_params,
    sha256,
    verify_compatibility,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def geometry_settings(source: dict[str, Any]) -> dict[str, Any]:
    """Get the physical coordinate mapping without importing JAX.

    Returns:
        Physical lattice or sphere radius used for scan coordinates.
    """
    system = source["config"]["system"]
    if source["geometry"] == "torus":
        tau = complex(*system["tau"])
        l1 = np.sqrt(2 * np.pi * system["flux"] / tau.imag)
        return {"lattice": np.array([[l1, 0.0], [l1 * tau.real, l1 * tau.imag]])}
    if source["geometry"] == "sphere":
        potential = source["config"]["estimators"]["energy"]["potential"]
        radius = potential.get("radius", system.get("radius"))
        if radius is None:
            radius = np.sqrt(system["flux"] / 2)
        return {"sphere_radius": float(radius)}
    return {}


def scan_pass(
    directory: Path,
    name: str,
    grid: dict[str, np.ndarray],
    source: dict,
    params: Any,
    evaluate: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Save every evaluated point before checking for nonfinite results.

    Returns:
        Raw complex log wavefunctions and energy components at every point.

    Raises:
        ValueError: If an evaluated value is nonfinite or has an unexpected dtype.
    """
    import jax
    from jax import numpy as jnp

    size = len(grid["radius"])
    keys = ["logpsi", "energy", "kinetic", "potential"]
    if source["geometry"] == "disk":
        keys += ["electron_electron", "background"]
    raw = {
        key: np.full(
            size, np.nan, dtype=np.complex128 if key in keys[:3] else np.float64
        )
        for key in keys
    }
    with h5py.File(directory / f"{name}_grid.h5", "w") as output:
        output.attrs.update(
            complete=False, geometry=source["geometry"], coordinate_dtype="float64"
        )
        for key, value in grid.items():
            output.create_dataset(key, data=value)
        coordinates = output.create_dataset(
            "electrons",
            shape=(size, 6, 2),
            dtype="f8",
            chunks=(min(batch_size, size), 6, 2),
            compression="gzip",
            fillvalue=np.nan,
        )
        datasets = {
            key: output.create_dataset(
                key,
                shape=(size,),
                dtype=value.dtype,
                fillvalue=np.asarray(np.nan, dtype=value.dtype)[()],
            )
            for key, value in raw.items()
        }
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            positions = scan_positions(
                source["background"],
                grid["radius"][start:stop],
                grid["theta"][start:stop],
                source["geometry"],
                mobile=source["mobile_electron"],
                reference=source["probe_electron"],
                **geometry_settings(source),
            )
            coordinates[start:stop] = positions
            padded = np.pad(
                positions,
                ((0, batch_size - len(positions)), (0, 0), (0, 0)),
                mode="edge",
            )
            try:
                result = jax.device_get(evaluate(params, jnp.asarray(padded)))
                invalid = np.zeros(stop - start, dtype=bool)
                for key in keys:
                    value = np.asarray(result[key])[: stop - start]
                    if value.dtype != raw[key].dtype:
                        raise ValueError(f"Unexpected dtype for {key}: {value.dtype}")
                    datasets[key][start:stop] = value
                    raw[key][start:stop] = value
                    invalid |= ~np.isfinite(value)
                if invalid.any():
                    output.create_dataset(
                        "error_indices", data=start + np.flatnonzero(invalid)
                    )
                    indices = (start + np.flatnonzero(invalid)).tolist()
                    raise ValueError(
                        f"Nonfinite local energies at saved rows {indices}"
                    )
                np.testing.assert_allclose(
                    raw["energy"][start:stop],
                    raw["kinetic"][start:stop] + raw["potential"][start:stop],
                    atol=1e-9,
                    rtol=2e-12,
                    err_msg="Local energy must equal kinetic plus potential",
                )
            except Exception as error:
                output.attrs["error"] = str(error)
                output.attrs["error_batch_start"] = start
                output.flush()
                raise
            if start == 0 or stop == size or (start // batch_size) % 100 == 0:
                print(f"{source['geometry']} {name}: {stop}/{size}", flush=True)
                output.flush()
        output.attrs["complete"] = True
    return raw


def disk_integrals(
    directory: Path, source: dict, params: Any, evaluate: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Retain the existing base, refined and tail disk integrations.

    Returns:
        Normalization, energy and tail convergence checks.
    """
    center = np.asarray(source["background"])[source["probe_electron"]]
    edges = radial_edges(center, args.radial_bins, args.outer_step, args.outer_margin)
    fine_edges = refine_edges(edges)
    tail_edges = np.linspace(
        edges[-1],
        edges[-1] + args.tail_extension,
        int(np.ceil(args.tail_extension / (args.outer_step / 2))) + 1,
    )

    def one(name, selected_edges, angles):
        grid = polar_grid(selected_edges, angles)
        raw = scan_pass(
            directory, name, grid, source, params, evaluate, args.batch_size
        )
        return grid, raw

    base_grid, base_raw = one("base", edges, args.angles)
    fine_grid, fine_raw = one("refined", fine_edges, args.angles * 2)
    tail_grid, tail_raw = one("tail", tail_edges, args.angles * 2)

    def integral(selected_edges, grid, raw):
        values = np.stack(
            [
                raw["energy"].real,
                raw["kinetic"].real,
                raw["electron_electron"],
                raw["background"],
                raw["energy"].imag,
            ],
            axis=-1,
        )
        return integrate(selected_edges, grid, raw["logpsi"], values)

    base = integral(edges, base_grid, base_raw)
    fine = integral(fine_edges, fine_grid, fine_raw)
    tail_grid["bin_index"] += len(fine_edges) - 1
    full_grid = {key: np.r_[fine_grid[key], tail_grid[key]] for key in fine_grid}
    full_raw = {key: np.r_[fine_raw[key], tail_raw[key]] for key in fine_raw}
    final = integral(np.r_[fine_edges, tail_edges[1:]], full_grid, full_raw)
    checks = convergence(base, fine, final, args.norm_tolerance, args.energy_tolerance)
    for name, values in (("base", base), ("refined", fine), ("integrals", final)):
        np.savez_compressed(
            directory / f"{name}.npz", components=np.asarray(COMPONENTS), **values
        )
    np.testing.assert_allclose(final["probability"].sum(), 1.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        final["actual"][:, 0], final["actual"][:, 1:4].sum(axis=1), atol=1e-8
    )
    np.testing.assert_allclose(final["cumulative"][-1], final["mean"], atol=1e-8)
    return checks


def prepare(source: dict) -> tuple[Any, Any, dict[str, Any]]:
    """Restore and validate the state against the existing energy objects.

    Returns:
        Parameters, batched evaluator and restoration verification metadata.

    Raises:
        ValueError: If the validation configuration produces nonfinite values.
    """
    import jax
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData

    jax.config.update("jax_enable_x64", True)
    _, wavefunction, estimators, settings = build_system(source)
    background = np.asarray(source["background"], dtype=np.float64)
    # A supplied fixed background may replace the reference walker. Verify at a
    # noncoincident scan position instead of its otherwise unused mobile value.
    positions = scan_positions(
        background,
        np.array([0.2]),
        np.array([0.37]),
        source["geometry"],
        mobile=source["mobile_electron"],
        reference=source["probe_electron"],
        **geometry_settings(source),
    )[0]
    example = HallData(electrons=jnp.asarray(positions))
    params, restore = restore_params(
        wavefunction, example, Path(source["checkpoint"]), source["geometry"]
    )
    compatibility = verify_compatibility(
        wavefunction, params, positions, restore["compatibility_changes"]
    )
    key = jax.random.PRNGKey(0)
    potential = estimators["potential"]
    potential.init(example, key)
    single, batch = make_evaluator(
        wavefunction.logpsi, potential, kinetic=estimators["kinetic"]
    )
    direct = jax.jit(single)(params, example.electrons)

    @jax.jit
    def original_energy(p, positions):
        data = HallData(electrons=positions)
        k, _ = estimators["kinetic"].evaluate_single_walker(p, data, {}, None, key)
        v, _ = potential.evaluate_single_walker(p, data, {}, None, key)
        return k["energy:kinetic"] + v["energy:potential"]

    expected = original_energy(params, example.electrons)
    np.testing.assert_allclose(direct["energy"], expected, atol=1e-9, rtol=2e-12)
    if any(not np.isfinite(np.asarray(value)).all() for value in direct.values()):
        raise ValueError(
            "Nonfinite wavefunction or energy in the validation configuration"
        )
    return (
        params,
        batch,
        dict(
            settings=settings,
            compatibility=compatibility,
            **restore,
            energy_interface_check=True,
            jax_version=jax.__version__,
            backend=jax.default_backend(),
            device_kinds=[device.device_kind for device in jax.devices()],
        ),
    )


def run(source: dict, args: argparse.Namespace) -> None:
    """Evaluate one geometry in a new output directory.

    Raises:
        ValueError: If the output directory is already populated.
    """
    geometry = source["geometry"]
    directory = args.output_root.resolve() / geometry
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output exists; choose another --output-root: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = dict(
        **source,
        model=args.label or source["run_id"],
        coordinate_dtype="float64",
        full_six_electron_derivatives=True,
        mode="disk_integrals_and_radial"
        if geometry == "disk" and not args.radial_only
        else "radial_only",
        grid=dict(
            radial_bins=args.radial_bins,
            angles=args.angles,
            batch_size=args.batch_size,
            outer_step=args.outer_step,
            outer_margin=args.outer_margin,
            tail_extension=args.tail_extension,
            norm_tolerance=args.norm_tolerance,
            energy_tolerance=args.energy_tolerance,
            relative_tolerance=args.relative_tolerance,
        ),
        started_at=started,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        analysis_source_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "conditional_energy.py",
                "conditional_energy_sources.py",
                "evaluateConditionalEnergy.py",
            )
        },
    )
    write_json(directory / "run_manifest.json", manifest)
    try:
        params, evaluate, validation = prepare(source)
        manifest.update(validation)
        write_json(directory / "run_manifest.json", manifest)
        if args.validate_only:
            write_json(
                directory / "validation.json",
                dict(complete=False, validation_only=True, **validation),
            )
            print(
                f"{geometry}: parameter, compatibility and energy checks passed",
                flush=True,
            )
            return
        grid = radial_grid(args.radial_bins, args.angles)
        raw = scan_pass(
            directory, "radial", grid, source, params, evaluate, args.batch_size
        )
        curves = directional_curves(grid["radius"], raw)
        np.savez_compressed(directory / "curves.npz", allow_pickle=False, **curves)
        # Compare identical radii with doubled angular resolution. This is a
        # numerical quadrature check, not a Monte Carlo standard error.
        angles = args.angles * 2
        radii = curves["node_radius"]
        refined_grid = dict(
            radius=np.repeat(radii, angles),
            theta=np.tile((np.arange(angles) + 0.5) * 2 * np.pi / angles, len(radii)),
        )
        refined_raw = scan_pass(
            directory,
            "radial_refined",
            refined_grid,
            source,
            params,
            evaluate,
            args.batch_size,
        )
        refined = directional_curves(refined_grid["radius"], refined_raw)
        np.savez_compressed(
            directory / "curves_refined.npz", allow_pickle=False, **refined
        )
        a, b = curves["node_conditional"], refined["node_conditional"]
        error = np.abs(a - b)
        angular = dict(
            passed=bool(
                np.all(
                    error
                    <= args.energy_tolerance
                    + args.relative_tolerance * np.maximum(np.abs(a), np.abs(b))
                )
            ),
            max_absolute_error=error.max(axis=0).tolist(),
            absolute_tolerance=args.energy_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        checks = dict(angular=angular)
        if geometry == "disk" and not args.radial_only:
            checks["disk_integrals"] = disk_integrals(
                directory, source, params, evaluate, args
            )
        write_json(
            directory / "validation.json",
            dict(
                complete=True,
                identities_passed=True,
                convergence=checks,
                **validation,
                files_sha256={
                    path.name: sha256(path)
                    for path in directory.iterdir()
                    if path.suffix in (".h5", ".npz")
                },
            ),
        )
        print(
            f"Completed {geometry}: {directory}; "
            f"angular convergence={angular['passed']}",
            flush=True,
        )
    except Exception as error:
        write_json(
            directory / "validation.json", dict(complete=False, error=str(error))
        )
        raise


def select_source(args: argparse.Namespace, geometry: str) -> dict[str, Any]:
    """Select merged metadata or an explicitly supplied training state.

    Returns:
        Verified configuration and checkpoint provenance for one geometry.
    """
    if args.config:
        return read_checkpoint_source(
            args.config,
            args.checkpoint,
            geometry,
            walker_index=args.walker_index,
            mobile=args.mobile_electron,
            reference=args.probe_electron,
            background_manifest=args.background_manifest,
        )
    return read_source(
        args.data_root,
        geometry,
        walker_index=args.walker_index,
        mobile=args.mobile_electron,
        reference=args.probe_electron,
        background_manifest=args.background_manifest,
        merged_background=args.merged_background,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse merged-data selectors and run input checks or conditional scans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data/merge")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--geometry", choices=(*GEOMETRIES, "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--walker-index", type=int, default=0)
    parser.add_argument("--mobile-electron", type=int, default=0)
    parser.add_argument("--probe-electron", type=int, default=1)
    background = parser.add_mutually_exclusive_group()
    background.add_argument("--background-manifest", type=Path)
    background.add_argument("--merged-background", action="store_true")
    parser.add_argument("--angles", type=int, default=64)
    parser.add_argument("--radial-bins", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--outer-step", type=float, default=0.2)
    parser.add_argument("--outer-margin", type=float, default=10.0)
    parser.add_argument("--tail-extension", type=float, default=4.0)
    parser.add_argument("--norm-tolerance", type=float, default=1e-5)
    parser.add_argument("--energy-tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--radial-only", action="store_true", help="Skip full disk integration"
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data, hashes and geometry without creating output",
    )
    checks.add_argument(
        "--validate-only",
        action="store_true",
        help="Also verify network restoration, derivatives and energy at one point",
    )
    args = parser.parse_args(argv)
    if (args.config is None) != (args.checkpoint is None):
        parser.error("Provide --config and --checkpoint together")
    if args.config and (args.geometry == "all" or args.merged_background):
        parser.error(
            "A standalone checkpoint requires one geometry and its own background"
        )
    if args.radial_bins < 4 or args.angles < 4 or args.batch_size < 1:
        parser.error("Require radial-bins >= 4, angles >= 4 and batch-size >= 1")
    if (
        min(
            args.outer_step,
            args.tail_extension,
            args.norm_tolerance,
            args.energy_tolerance,
            args.relative_tolerance,
        )
        <= 0
        or args.outer_margin <= 3
    ):
        parser.error("Grid steps and tolerances must be positive and outer-margin > 3")
    if args.geometry == "all" and (args.background_manifest or args.merged_background):
        parser.error("An explicit background requires selecting a single geometry")
    if args.output_root is None:
        stamp = os.environ.get("SLURM_JOB_ID") or datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).strftime("%Y%m%d_%H%M%S_%f")
        args.output_root = ROOT / "runs" / f"conditional_energy_{stamp}"
    selected = GEOMETRIES if args.geometry == "all" else (args.geometry,)
    sources = []
    for geometry in selected:
        source = select_source(args, geometry)
        validate_radius(geometry, 2.0, **geometry_settings(source))
        sources.append(source)
    for source in sources:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        key: source[key]
                        for key in (
                            "geometry",
                            "run_id",
                            "checkpoint",
                            "checkpoint_step",
                            "checkpoint_sha256",
                            "background_origin",
                        )
                    }
                    | dict(
                        output=str(args.output_root / source["geometry"]),
                        system=source["config"]["system"],
                        wf=source["config"]["wf"],
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            run(source, args)


if __name__ == "__main__":
    main()
