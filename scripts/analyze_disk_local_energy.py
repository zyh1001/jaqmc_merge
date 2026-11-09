# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan full-system local energies with one mobile electron on a disk."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT.parent / "Runs/n4_Q12_kapa3_8192_final"
DEFAULT_OUTPUT = ROOT.parent / "Runs/n4_Q12_kapa3_8192_final_local_energy_20261101"
FIELDS = ("energy", "kinetic", "potential", "electron_electron", "background")
CURVE_FIELDS = (*FIELDS, "energy_imaginary")
INITIAL = {"beta_par": 0.25, "ee_par": 1.0, "beta_anti": 0.5, "ee_anti": 1.0}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_run(run_dir: Path, checkpoint: Path | None) -> tuple[dict, Path, list]:
    config = yaml.safe_load((run_dir / "train_config.yaml").read_text())
    if checkpoint is None:
        files = sorted(run_dir.glob("train_ckpt_*.npz"))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {run_dir}")
        checkpoint = files[-1]
    checkpoint = checkpoint.resolve()
    system = config["system"]
    if system["module"] not in (
        "jaqmc.app.hall.config:HallDiskConfig",
        "jaqmc.app.hall.config.disk",
        "jaqmc.app.hall.config.disk:HallDiskConfig",
    ):
        raise ValueError("Expected the disk system configuration")
    system["module"] = "jaqmc.app.hall.config:HallDiskConfig"
    potential = config["estimators"]["energy"]["potential"]
    for key, value in {
        "N": sum(system["nspins"]),
        "flux": system["flux"],
        "d": system["d"],
        "interaction_strength": system["interaction_strength"],
        "interaction_type": system["interaction_type"],
    }.items():
        if potential[key] != value:
            raise ValueError(f"Saved system and potential disagree: {key}")
    if min(system["nspins"]) < 0 or sum(system["nspins"]) != 4:
        raise ValueError("This analysis report expects four electrons")
    files = sorted(set(run_dir.glob("train_ckpt_*.npz")) | {checkpoint})
    history = jastrow_history(files)
    with np.load(checkpoint, allow_pickle=False) as saved:
        coordinates = saved["batched_data/data/electrons"]
        if coordinates.ndim != 3 or coordinates.shape[1:] != (sum(system["nspins"]), 2):
            raise ValueError(f"Unexpected coordinate shape: {coordinates.shape}")
        if not np.isfinite(coordinates).all():
            raise ValueError("Nonfinite saved coordinates")
    return config, checkpoint, history


def jastrow_history(files: list[Path]) -> list[dict[str, Any]]:
    history = []
    for file in files:
        with np.load(file, allow_pickle=False) as saved:
            row: dict[str, Any] = {
                "checkpoint": str(file.resolve()),
                "step": int(saved["step"]),
            }
            for name in INITIAL:
                key = "params/params/jastrow_layer/" + name
                if key not in saved or saved[key].shape != (1,):
                    raise ValueError(
                        f"Missing or invalid Jastrow parameter: {file}: {key}"
                    )
                value = float(saved[key][0])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite Jastrow parameter: {file}: {key}")
                row[name] = value
                row[name + "_change"] = value - INITIAL[name]
            history.append(row)
    return history


def restore_params(
    wavefunction: Any, example: Any, checkpoint: Path
) -> tuple[Any, dict]:
    """Restore complete parameters, then promote precision.

    Returns:
        Restored parameters and checkpoint step and original dtype metadata.

    Raises:
        ValueError: If a parameter is missing, extra, nonfinite or misshaped.
    """
    import jax
    from jax import numpy as jnp
    from upath import UPath

    from jaqmc.utils.checkpoint import NumPyCheckpointManager, tree_to_npz

    template = wavefunction.init_params(example, jax.random.PRNGKey(1))
    expected = tree_to_npz({"params": template})
    with np.load(checkpoint, allow_pickle=False) as saved:
        actual = {key for key in saved.files if key.startswith("params/")}
        if actual != set(expected):
            raise ValueError(
                f"Parameter mismatch: missing={sorted(set(expected) - actual)}, "
                f"extra={sorted(actual - set(expected))}"
            )
        for key, value in expected.items():
            if saved[key].shape != value.shape or not np.isfinite(saved[key]).all():
                raise ValueError(f"Invalid saved parameter: {key}")
        original_dtypes = sorted({str(saved[key].dtype) for key in actual})
    step, restored = NumPyCheckpointManager.restore_from_file(
        UPath(str(checkpoint)), {"params": template}
    )

    def promote(value: Any) -> Any:
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            return value.astype(jnp.complex128)
        if jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(jnp.float64)
        return value

    params = jax.tree.map(promote, restored["params"])
    return params, {
        "checkpoint_step": step - 1,
        "saved_parameter_dtypes": original_dtypes,
    }


def make_evaluator(logpsi: Callable, potential: Any) -> Callable:
    """Adapt the previous conditional-energy evaluator to any electron count.

    Returns:
        Compiled batched evaluator of log amplitude and energy components.
    """
    import jax
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData
    from jaqmc.estimator.kinetic.disk import DiskKinetic
    from jaqmc.estimator.total_energy import TotalEnergy

    kinetic = DiskKinetic(f_log_psi=logpsi)
    total = TotalEnergy(components=["energy:kinetic", "energy:potential"])
    key = jax.random.PRNGKey(0)

    def single(params: Any, positions: Any) -> dict:
        data = HallData(electrons=positions)
        k, _ = kinetic.evaluate_single_walker(params, data, {}, None, key)
        v, _ = potential.evaluate_single_walker(params, data, {}, None, key)
        e, _ = total.evaluate_single_walker(params, data, k | v, None, key)
        i, j = jnp.triu_indices(positions.shape[0], 1)
        vee = potential.interaction_strength * jnp.sum(
            1 / jnp.linalg.norm(positions[i] - positions[j], axis=-1)
        )
        return {
            "logpsi": logpsi(params, data),
            "energy": e["total_energy"],
            "kinetic": k["energy:kinetic"],
            "potential": v["energy:potential"],
            "electron_electron": vee,
            "background": v["energy:potential"] - vee,
        }

    return jax.jit(jax.vmap(single, in_axes=(None, 0)))


def angular_mean(logpsi: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not np.isfinite(logpsi).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite angular values; no points discarded")
    log_weight = 2 * logpsi.real
    weights = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
    return np.sum(weights[..., None] * values, axis=1) / weights.sum(axis=1)[:, None]


def radial_grid(center: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    maximum = float(np.linalg.norm(center)) + args.outer_margin
    outer = np.linspace(3, maximum, int(np.ceil((maximum - 3) / args.outer_step)) + 1)
    return np.unique(np.r_[np.geomspace(args.radial_min, 3, args.radial_points), outer])


def compare_curves(
    reference: np.ndarray,
    actual: np.ndarray,
    radii: np.ndarray,
    atol: float,
    rtol: float,
) -> dict:
    error = np.abs(reference - actual)
    tolerance = atol + rtol * np.maximum(np.abs(reference), np.abs(actual))
    passed = error <= tolerance
    return {
        "passed": bool(passed.all()),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": dict(
            zip(CURVE_FIELDS, error.max(axis=0).tolist(), strict=True)
        ),
        "failed_radii": radii[~passed.all(axis=1)].tolist(),
        "component_passed": passed.tolist(),
        "max_tolerance_ratio": float(np.max(error / tolerance)),
    }


def scan(
    output: Path,
    name: str,
    radii: np.ndarray,
    angles: int,
    fixed: np.ndarray,
    params: Any,
    evaluate: Callable,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    import h5py
    import jax
    from jax import numpy as jnp

    theta = (np.arange(angles) + 0.5) * (2 * np.pi / angles)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    rr, tt = rr.ravel(), tt.ravel()
    nelec = len(fixed) + 1
    size = len(rr)
    keys = ("logpsi", *FIELDS)
    raw = {
        key: np.empty(size, dtype=np.complex128 if key in keys[:3] else np.float64)
        for key in keys
    }
    residuals = {"energy": 0.0, "potential": 0.0}
    with h5py.File(output / f"{name}_grid.h5", "w") as stream:
        stream.attrs.update(complete=False, angles=angles, coordinate_dtype="float64")
        stream.create_dataset("radius", data=rr)
        stream.create_dataset("theta", data=tt)
        coordinates = stream.create_dataset(
            "electrons", shape=(size, nelec, 2), dtype="f8", compression="gzip"
        )
        datasets = {
            key: stream.create_dataset(
                key, shape=(size,), dtype=value.dtype, compression="gzip"
            )
            for key, value in raw.items()
        }
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            positions = np.empty((stop - start, nelec, 2), dtype=np.float64)
            positions[:, 1:] = fixed
            positions[:, 0] = fixed[0] + rr[start:stop, None] * np.stack(
                [np.cos(tt[start:stop]), np.sin(tt[start:stop])], axis=-1
            )
            padded = np.pad(
                positions,
                ((0, batch_size - len(positions)), (0, 0), (0, 0)),
                mode="edge",
            )
            result = jax.device_get(evaluate(params, jnp.asarray(padded)))
            for key in keys:
                values = np.asarray(result[key])[: len(positions)]
                if values.dtype != raw[key].dtype:
                    raise ValueError(
                        f"Expected {raw[key].dtype}, got {values.dtype}: {key}"
                    )
                if not np.isfinite(values).all():
                    index = int(np.flatnonzero(~np.isfinite(values))[0])
                    write_json(
                        output / "failed_point.json",
                        {
                            "pass": name,
                            "row": start + index,
                            "field": key,
                            "value": str(values[index]),
                            "radius": float(rr[start + index]),
                            "theta": float(tt[start + index]),
                            "electrons": positions[index].tolist(),
                        },
                    )
                    raise ValueError(f"Nonfinite {key}, see failed_point.json")
                raw[key][start:stop] = values
                datasets[key][start:stop] = values
            current = {key: value[start:stop] for key, value in raw.items()}
            for label, lhs, rhs in (
                (
                    "energy",
                    current["energy"],
                    current["kinetic"] + current["potential"],
                ),
                (
                    "potential",
                    current["potential"],
                    current["electron_electron"] + current["background"],
                ),
            ):
                residuals[label] = max(
                    residuals[label], float(np.max(np.abs(lhs - rhs)))
                )
                if not np.allclose(lhs, rhs, rtol=2e-12, atol=1e-9):
                    raise ValueError(
                        f"Energy identity failed: {label}, rows {start}:{stop}"
                    )
            coordinates[start:stop] = positions
            if start == 0 or stop == size or start % (batch_size * 200) == 0:
                print(f"{name}: {stop}/{size}", flush=True)
                stream.flush()
        stream.attrs["complete"] = True
    values = np.stack([*(raw[key].real for key in FIELDS), raw["energy"].imag], axis=-1)
    means = angular_mean(
        raw["logpsi"].reshape(-1, angles), values.reshape(-1, angles, len(CURVE_FIELDS))
    )
    np.savez_compressed(
        output / f"{name}_curves.npz",
        radius=radii,
        values=means,
        fields=np.asarray(CURVE_FIELDS),
    )
    # This gives relative angular probability density, without a radial normalization.
    amplitudes = raw["logpsi"].real.reshape(-1, angles)
    log_mass = 2 * amplitudes.max(axis=1) + np.log(
        np.exp(2 * (amplitudes - amplitudes.max(axis=1, keepdims=True))).mean(axis=1)
    )
    rows = [
        dict(
            radius=float(r),
            **dict(zip(CURVE_FIELDS, value.tolist(), strict=True)),
            log_angular_probability=float(mass),
        )
        for r, value, mass in zip(radii, means, log_mass, strict=True)
    ]
    write_csv(output / f"{name}_curves.csv", rows)
    return means, residuals


def gaussian_check() -> dict:
    import jax
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData
    from jaqmc.app.hall.hamiltonian.disk import DiskPotential

    positions = jnp.array(
        [[0.2, 0.3], [1.0, 0.4], [-1.0, 1.0], [2.0, -1.0]], dtype=jnp.float64
    )

    def logpsi(params: Any, data: Any) -> Any:
        del params
        return (-jnp.sum(data.electrons**2) / 4).astype(jnp.complex128)

    potential = DiskPotential(
        N=4,
        flux=12,
        radius=np.sqrt(24),
        d=0.1,
        interaction_strength=3,
        radial_grid_size=16,
    )
    potential.init(HallData(electrons=positions), jax.random.PRNGKey(0))
    result = jax.device_get(make_evaluator(logpsi, potential)({}, positions[None]))
    error = float(np.max(np.abs(result["kinetic"] - 2)))
    if error > 1e-12:
        raise ValueError(f"Four-electron Gaussian kinetic test failed: {error}")
    theta = (np.arange(64) + 0.5) * 2 * np.pi / 64
    logpsi_array = (0.5 * np.log(1 + 0.6 * np.cos(theta)))[None]
    weighted = angular_mean(logpsi_array, np.cos(theta)[None, :, None])[0, 0]
    if abs(weighted - 0.3) > 1e-13:
        raise ValueError("Analytic angular weighting test failed")
    return {
        "gaussian_total_kinetic": 2.0,
        "max_kinetic_error": error,
        "analytic_angular_mean": float(weighted),
        "passed": True,
    }


def run(
    args: argparse.Namespace, config: dict, checkpoint: Path, history: list
) -> None:
    import jax
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData
    from jaqmc.app.hall.hamiltonian.disk import DiskPotential
    from jaqmc.app.hall.workflow import configure_system
    from jaqmc.utils.config import ConfigManager

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output exists; choose a new --output: {output}")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Submit the numerical scan with the supplied .sh script")
    if (
        jax.device_count() != 1
        or jax.process_count() != 1
        or jax.default_backend() != "gpu"
    ):
        raise ValueError("The scan requires one process and one allocated GPU")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "jastrow_history.csv", history)
    write_json(output / "validation.json", {"complete": False, "stage": "initializing"})
    try:
        checks = gaussian_check()
        _, wavefunction = configure_system(
            ConfigManager({"system": config["system"], "wf": config["wf"]})
        )
        with np.load(checkpoint, allow_pickle=False) as saved:
            walkers = saved["batched_data/data/electrons"].astype(np.float64)
        params, restored = restore_params(
            wavefunction, HallData(electrons=jnp.asarray(walkers[0])), checkpoint
        )
        batch_logpsi = jax.jit(
            jax.vmap(
                lambda p, x: wavefunction.logpsi(p, HallData(electrons=x)),
                in_axes=(None, 0),
            )
        )
        scores = []
        for start in range(0, len(walkers), args.batch_size):
            part = walkers[start : start + args.batch_size]
            padded = np.pad(
                part, ((0, args.batch_size - len(part)), (0, 0), (0, 0)), mode="edge"
            )
            values = np.asarray(
                jax.device_get(batch_logpsi(params, jnp.asarray(padded)))
            )[: len(part)]
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nonfinite saved-configuration logpsi at batch {start}"
                )
            scores.extend(values.real.tolist())
        selected = int(np.argmax(scores))
        fixed = walkers[selected, 1:]
        potential = DiskPotential(**config["estimators"]["energy"]["potential"])
        potential.init(
            HallData(electrons=jnp.asarray(walkers[selected])), jax.random.PRNGKey(0)
        )
        evaluate = make_evaluator(wavefunction.logpsi, potential)
        radii = radial_grid(fixed[0], args)
        midpoint = np.sqrt(radii[:-1] * radii[1:])
        sources = {
            Path(__file__).resolve(),
            args.run_dir / "train_config.yaml",
            checkpoint,
        }
        for module in list(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if (
                path
                and Path(path).is_relative_to(ROOT / "src/jaqmc")
                and Path(path).is_file()
            ):
                sources.add(Path(path))
        manifest = {
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source_checkpoint": str(checkpoint),
            "run_dir": str(args.run_dir),
            "settings": config,
            "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
            "selected_sample": selected,
            "samples_considered": len(walkers),
            "selected_log_abs": scores[selected],
            "selected_electrons": walkers[selected].tolist(),
            "fixed_electrons": fixed.tolist(),
            "mobile_electron": 0,
            "center_electron": 1,
            "background_selection": (
                "maximum_log_abs_among_saved_walkers_first_index_on_ties"
            ),
            "angular_average": "wavefunction_probability_weighted_at_each_radius",
            "full_electron_derivatives": True,
            "background_self_energy_included": False,
            "coordinate_dtype": "float64",
            "evaluation_parameter_dtype": "float64_or_complex128",
            "precision_note": (
                "Promoting saved float32 values "
                "does not recover lost training precision."
            ),
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "radial_points": len(radii),
            "radius_max": float(radii[-1]),
            **restored,
        }
        write_json(output / "run_manifest.json", manifest)
        print(
            f"Selected sample {selected}/{len(walkers)}, "
            f"center={fixed[0]}, radii={len(radii)}",
            flush=True,
        )
        base, base_residual = scan(
            output, "base", radii, args.angles, fixed, params, evaluate, args.batch_size
        )
        fine, fine_residual = scan(
            output,
            "refined",
            radii,
            2 * args.angles,
            fixed,
            params,
            evaluate,
            args.batch_size,
        )
        mid, mid_residual = scan(
            output,
            "midpoint",
            midpoint,
            2 * args.angles,
            fixed,
            params,
            evaluate,
            args.batch_size,
        )
        angular = compare_curves(base, fine, radii, args.atol, args.rtol)
        radial = compare_curves(
            (fine[:-1] + fine[1:]) / 2, mid, midpoint, args.atol, args.rtol
        )
        angular_rows = [
            dict(radius=float(r), **dict(zip(CURVE_FIELDS, row, strict=True)))
            for r, row in zip(radii, angular["component_passed"], strict=True)
        ]
        radial_rows = [
            dict(
                radius_left=float(left),
                radius_right=float(right),
                radius_mid=float(r),
                **dict(zip(CURVE_FIELDS, row, strict=True)),
            )
            for left, right, r, row in zip(
                radii[:-1], radii[1:], midpoint, radial["component_passed"], strict=True
            )
        ]
        write_csv(output / "angular_convergence.csv", angular_rows)
        write_csv(output / "radial_convergence.csv", radial_rows)
        write_json(
            output / "validation.json",
            {
                "complete": True,
                "identities_passed": True,
                "analytic_checks": checks,
                "identity_max_absolute_residuals": {
                    "base": base_residual,
                    "refined": fine_residual,
                    "midpoint": mid_residual,
                },
                "angular_convergence": angular,
                "radial_interpolation_check": radial,
                "interpretation": (
                    "Pointwise grid checks for one fixed configuration; "
                    "not ensemble or derivative convergence."
                ),
                "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "output_hashes": {
                    p.name: sha256(p)
                    for p in output.iterdir()
                    if p.suffix in (".h5", ".npz", ".csv")
                },
            },
        )
        print(
            f"Completed. Angular passed={angular['passed']}; "
            f"radial passed={radial['passed']}",
            flush=True,
        )
    except Exception as error:
        write_json(output / "validation.json", {"complete": False, "error": str(error)})
        raise


def configure_plot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    for font in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
    ):
        if any(f.name == font for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    else:
        raise ValueError("A Chinese font is required to render the figures")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 3
    return plt


def load_metadata(output: Path) -> tuple[dict, dict]:
    manifest = json.loads((output / "run_manifest.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    if not validation["complete"] or not validation["identities_passed"]:
        raise ValueError("Cannot plot an incomplete or inconsistent scan")
    for name, digest in validation["output_hashes"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Saved result changed: {name}")
    return manifest, validation


def plot(args: argparse.Namespace) -> None:
    plt = configure_plot()
    output = args.output.resolve()
    manifest, validation = load_metadata(output)
    with np.load(output / "refined_curves.npz", allow_pickle=False) as saved:
        radii, values = saved["radius"], saved["values"]
    with np.load(output / "midpoint_curves.npz", allow_pickle=False) as saved:
        midr, midv = saved["radius"], saved["values"]
    allr, allv = np.r_[radii, midr], np.r_[values, midv]
    order = np.argsort(allr)
    allr, allv = allr[order], allv[order]
    failed_angle = ~np.asarray(validation["angular_convergence"]["component_passed"])
    failed_radial = ~np.asarray(
        validation["radial_interpolation_check"]["component_passed"]
    )
    colors = ("#263d66", "#d17c27", "#2c8868")
    for name, maximum in (
        ("local_energy_full", float(allr[-1])),
        ("local_energy_short_range", 1.0),
    ):
        figure, axes = plt.subplots(
            3, 1, figsize=(10, 10), sharex=True, constrained_layout=True
        )
        for column, (axis, label, color) in enumerate(
            zip(axes, ("总局域能量", "总局域动能", "总局域势能"), colors, strict=True)
        ):
            visible = allr <= maximum
            y = allv[visible, column]
            axis.plot(allr[visible], y, color=color, linewidth=1.1, label=label)
            ylabel = label
            if np.max(np.abs(y)) > 1e3 * max(1.0, float(np.min(np.abs(y)))):
                axis.set_yscale("symlog", linthresh=1)
                ylabel += "（对称对数）"
            mask = failed_angle[:, column] & (radii <= maximum)
            mark_angular_failures(axis, radii[mask], values[mask, column])
            for left, right in zip(
                radii[:-1][failed_radial[:, column]],
                radii[1:][failed_radial[:, column]],
                strict=True,
            ):
                if left <= maximum:
                    axis.axvspan(left, min(right, maximum), color="#b22222", alpha=0.07)
            axis.set_ylabel(ylabel + "\n（磁回旋能量单位）")
            axis.grid(alpha=0.2)
            axis.legend(loc="best", fontsize=9)
            axis.set_xscale("log")
            axis.set_xlim(float(allr[0]), maximum)
        axes[-1].set_xlabel("到中心电子的距离 r（磁长度单位）")
        figure.suptitle(
            "四电子局域能量：固定三个电子，各半径按波函数概率平均\n"
            "各图淡红区间表示该能量分量的径向中点检查未通过"
        )
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    coordinates = np.asarray(manifest["selected_electrons"])
    axis.scatter(
        coordinates[1:, 0], coordinates[1:, 1], color="#263d66", label="固定电子"
    )
    axis.scatter(
        *coordinates[0],
        facecolors="none",
        edgecolors="#d17c27",
        label="移动电子的原始位置",
    )
    for index, xy in enumerate(coordinates):
        axis.annotate(str(index), xy, xytext=(5, 5), textcoords="offset points")
    for radius in (0.1, 1.0):
        axis.add_patch(
            plt.Circle(coordinates[1], radius, fill=False, linestyle="--", alpha=0.5)
        )
    axis.set_aspect("equal")
    axis.set_xlabel("横坐标（磁长度单位）")
    axis.set_ylabel("纵坐标（磁长度单位）")
    axis.set_title("固定构型与绕行中心（电子 1）")
    axis.legend()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"fixed_configuration.{extension}", dpi=180)
    plt.close(figure)
    table, parameter_text = summarize_jastrow(output, manifest)
    angle_count = int(failed_angle.any(axis=1).sum())
    radial_count = int(failed_radial.any(axis=1).sum())
    component_failures = "、".join(
        f"{label} {int(failed_radial[:, column].sum())} 个"
        for column, label in enumerate(("总能量", "动能", "势能"))
    )
    text = f"""# 四电子波函数的参数与局域能量\n
本次检查使用训练目录 `{manifest["run_dir"]}`
中的第 {manifest["checkpoint_step"]} 步检查点。
相互作用强度为 {manifest["settings"]["system"]["interaction_strength"]}，
磁通参数为 {manifest["settings"]["system"]["flux"]}。
训练保存了 {manifest["samples_considered"]} 个构型，实际批量以保存数据为准。

## Jastrow 参数

| 参数 | 代码初始值 | 最终保存值 | 变化量 |
|---|---:|---:|---:|
{table}

{parameter_text}
完整变化记录见 `jastrow_history.csv`。参数变化本身不能证明满足尖点条件。

## 固定构型与计算方法

从保存构型中选择波函数幅度最大的第 {manifest["selected_sample"]} 个构型，编号从零开始。
固定电子 1、2、3，电子 0 绕电子 1 移动。各电子坐标如下。

```text
{np.array2string(coordinates, precision=12)}
```

每个半径绕行一周，用波函数模平方对各角度加权。图中展示总局域能量、总局域动能和总局域势能的实部。
动能包含对全部四个电子坐标的求导，势能包含全部六对电子相互作用和全部电子的背景势能。
背景圆盘半径沿用训练配置，不计背景自能常数。长度单位为磁长度，能量单位为磁回旋能量。
能量虚部、电子间势能和背景势能保存在曲线表中，各角度数据及完整坐标保存在三个原始数据文件中。

![固定构型](fixed_configuration.png)
![全范围曲线](local_energy_full.png)
![小距离曲线](local_energy_short_range.png)

在最小扫描距离 {radii[0]:.6g} 处，总局域能量约为 {values[0, 0]:.6f}，
总局域动能约为 {values[0, 1]:.6f}，总局域势能约为 {values[0, 2]:.6f}。
这表明在本次固定构型下，两个电子接近时出现很大的正局域能量。

## 数值检查与限制

四电子高斯波函数的总动能检查通过，预期值为 2。
可解析的角度概率加权检查通过，预期值为 0.3。
逐点检查了总能量等于动能加势能、总势能等于电子间势能加背景势能，检查通过。
角度数从 {manifest["arguments"]["angles"]} 加倍后，
有 {angle_count}/{len(radii)} 个半径未满足给定容差。
增加径向中点后，有 {radial_count}/{len(midr)} 个区间的实际中点值
与端点在对数距离上的线性插值不满足容差。
三个主图分量的径向中点检查未通过数量为：{component_failures}。
各图只标明对应能量分量未通过的位置，具体结果见两个收敛检查表。
绝对容差为 {manifest["arguments"]["atol"]}，
相对容差为 {manifest["arguments"]["rtol"]}。
径向中点检查衡量曲线采样是否足够密，不能单独证明局域能量导数准确。

计算将已保存的单精度参数转换为双精度，这不会恢复训练时未保存的精度。
曲线对应一个固定三电子构型下的条件结果，不能直接作为整个体系的能量期望值。
小距离局域能量很大时，应结合波函数概率和动能、势能分量判断，不能仅凭数值大小认定计算错误。
来源文件、校验值、作业编号和完整检查结果见 `run_manifest.json` 与 `validation.json`。
"""
    (output / "说明.md").write_text(text, encoding="utf-8")
    write_json(
        output / "plot_validation.json",
        {
            "complete": True,
            "angular_failed_radii": angle_count,
            "radial_failed_intervals": radial_count,
            "plotting_script_sha256": sha256(Path(__file__).resolve()),
            "files_sha256": {
                p.name: sha256(p)
                for p in output.iterdir()
                if p.suffix in (".png", ".pdf", ".md")
            },
        },
    )
    print(f"Figures and Chinese report: {output}", flush=True)


def mark_angular_failures(axis: Any, radius: np.ndarray, value: np.ndarray) -> None:
    if len(radius):
        axis.scatter(
            radius,
            value,
            facecolors="none",
            edgecolors="#b22222",
            s=16,
            label="角度加密检查未通过",
            zorder=3,
        )


def summarize_jastrow(output: Path, manifest: dict) -> tuple[str, str]:
    with (output / "jastrow_history.csv").open(encoding="utf-8-sig") as stream:
        history = list(csv.DictReader(stream))
    final = next(
        row
        for row in history
        if Path(row["checkpoint"]) == Path(manifest["source_checkpoint"])
    )
    table = "\n".join(
        f"| `{key}` | {value:.12g} | {float(final[key]):.12g} | "
        f"{float(final[key]) - value:.12g} |"
        for key, value in INITIAL.items()
    )
    same_spin = ("beta_par", "ee_par")
    changed = not np.array_equal(
        [float(final[key]) for key in same_spin], [INITIAL[key] for key in same_spin]
    )
    text = "同自旋参数发生了更新。" if changed else "同自旋参数与代码初始值相同。"
    preceding = sorted(
        (row for row in history if int(row["step"]) <= int(final["step"])),
        key=lambda row: int(row["step"]),
        reverse=True,
    )
    plateau = []
    for row in preceding:
        if any(row[key] != final[key] for key in same_spin):
            break
        plateau.append(row)
    if len(plateau) > 1:
        text += (
            f"最后连续 {len(plateau)} 个检查点中的两个同自旋参数值相同，"
            f"最早为第 {plateau[-1]['step']} 步。"
        )
    if min(manifest["settings"]["system"]["nspins"]) == 0:
        text += "全部电子同自旋，异自旋项不影响波函数。"
    else:
        text += "本次包含两种自旋，两个自旋通道的参数都参与波函数计算。"
    return table, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定其余电子，计算一个电子绕另一电子移动时的完整体系局域能量。"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angles", type=int, default=64)
    parser.add_argument("--radial-points", type=int, default=200)
    parser.add_argument("--radial-min", type=float, default=1e-4)
    parser.add_argument("--outer-step", type=float, default=0.2)
    parser.add_argument("--outer-margin", type=float, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.angles < 4 or args.radial_points < 2 or args.batch_size < 1:
        parser.error("Grid counts and batch size must be positive")
    if not (
        0 < args.radial_min < 1
        and args.outer_margin > 3
        and min(args.outer_step, args.atol, args.rtol) > 0
    ):
        parser.error("Invalid radius range, step or tolerances")
    return args


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot(args)
        return
    args.run_dir = args.run_dir.resolve()
    if not args.self_test:
        config, checkpoint, history = inspect_run(args.run_dir, args.checkpoint)
        if args.dry_run:
            with np.load(checkpoint, allow_pickle=False) as saved:
                shape = saved["batched_data/data/electrons"].shape
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "output": str(args.output.resolve()),
                        "saved_coordinate_shape": shape,
                        "system": config["system"],
                        "potential": config["estimators"]["energy"]["potential"],
                        "jastrow_history": history,
                        "arguments": {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    import jax

    jax.config.update("jax_enable_x64", True)
    if args.self_test:
        print(json.dumps(gaussian_check(), indent=2))
    else:
        run(args, config, checkpoint, history)


if __name__ == "__main__":
    main()
