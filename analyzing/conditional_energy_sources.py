# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Read Hall training sources and restore the trained wavefunction."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

GEOMETRIES = ("disk", "sphere", "torus")


def sha256(path: Path) -> str:
    """Return a file's SHA-256 without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write JSON without silently accepting nonfinite values."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def training_metadata(metadata: Any) -> tuple[dict, str]:
    """Read and cross-check the merged training configuration.

    Returns:
        Parsed configuration and its original YAML text.

    Raises:
        ValueError: If metadata are unavailable or disagree with the configuration.
    """
    if metadata["config_yaml"].attrs.get("status") != "available":
        raise ValueError("Training configuration unavailable")
    text = metadata["config_yaml"].asstr()[()]
    config = yaml.safe_load(text)
    for key in ("flux", "interaction_strength", "nspins"):
        if not np.array_equal(config["system"][key], metadata[key][()]):
            raise ValueError(f"Configuration disagrees with merged metadata: {key}")
    if sum(config["system"]["nspins"]) != 6 or int(metadata["n_electrons"][()]) != 6:
        raise ValueError("Conditional scans require six total electrons")
    return config, text


def training_checkpoint(
    metadata: Any, manifest: dict, geometry: str, source_root: Path
) -> tuple[Path, dict]:
    """Resolve the exact checkpoint named by the metadata's source references.

    Returns:
        Checkpoint path and its manifest entry.

    Raises:
        ValueError: If the reference is ambiguous or the checksum differs.
    """
    files = {item["file_id"]: item for item in manifest["files"]}
    source_ids = json.loads(metadata["config_yaml"].attrs["source_files"])
    candidates = [
        files[key]
        for key in source_ids
        if files[key]["geometry"] == geometry and files[key]["category"] == "checkpoint"
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one referenced training checkpoint for {geometry}")
    item = candidates[0]
    checkpoint = (source_root / item["path"]).resolve()
    if sha256(checkpoint) != item["sha256"]:
        raise ValueError(f"Checkpoint hash differs from merged manifest: {checkpoint}")
    return checkpoint, item


def checkpoint_background(checkpoint: Path, step: int, index: int) -> np.ndarray:
    """Read one saved walker without resuming the sampler.

    Returns:
        Six positions in the geometry's native coordinates.

    Raises:
        ValueError: If the checkpoint step, coordinate shape or index is invalid.
    """
    with np.load(checkpoint, allow_pickle=False) as saved:
        if int(saved["step"]) != step:
            raise ValueError("Checkpoint step disagrees with merged metadata")
        positions = saved["batched_data/data/electrons"]
        if positions.ndim != 3 or positions.shape[1:] != (6, 2):
            raise ValueError(f"Invalid saved positions: {positions.shape}")
        if not 0 <= index < len(positions):
            raise ValueError(
                f"Walker index must be between zero and {len(positions) - 1}"
            )
        return np.asarray(positions[index])


def saved_scan_background(
    scan: Any, digest: str, mobile: int, reference: int
) -> np.ndarray:
    """Read a previously saved fixed background for the same trained state.

    Returns:
        Five fixed native-coordinate positions.

    Raises:
        ValueError: If the background is absent or refers to another scan.
    """
    if scan["fixed_electrons"].attrs.get("status") != "available":
        raise ValueError("Merged scan background is unavailable")
    if scan["source_checkpoint_sha256"].asstr()[()] != digest:
        raise ValueError("Merged background refers to another checkpoint")
    if (
        int(scan["mobile_electron"][()]) != mobile
        or int(scan["probe_electron"][()]) != reference
    ):
        raise ValueError("Requested indices differ from the merged scan")
    return np.asarray(scan["fixed_electrons"][()], dtype=np.float64)


def supplied_background(
    path: Path, geometry: str, digest: str, mobile: int, reference: int
) -> np.ndarray:
    """Read five fixed positions from a standalone or full result manifest.

    Returns:
        Five positions in native electron order.

    Raises:
        ValueError: If the background has another state, geometry or shape.
    """
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if supplied.get("geometry", geometry) != geometry:
        raise ValueError("Background manifest has another geometry")
    if (
        supplied.get(
            "checkpoint_sha256", supplied.get("source_checkpoint_sha256", digest)
        )
        != digest
    ):
        raise ValueError("Background manifest has another checkpoint")
    if (
        supplied.get("mobile_electron", mobile) != mobile
        or supplied.get("probe_electron", reference) != reference
    ):
        raise ValueError("Background manifest uses different electron indices")
    selected = supplied.get("background", supplied)
    if isinstance(selected, list):
        selected = dict(fixed_electrons=np.delete(selected, mobile, axis=0))
    if (
        selected.get("mobile_electron", mobile) != mobile
        or selected.get("probe_electron", reference) != reference
    ):
        raise ValueError("Background manifest uses different electron indices")
    fixed = np.asarray(selected["fixed_electrons"], dtype=np.float64)
    if fixed.shape != (5, 2):
        raise ValueError("Background manifest must contain five native positions")
    return fixed


def read_source(
    data_root: Path,
    geometry: str,
    *,
    walker_index: int = 0,
    mobile: int = 0,
    reference: int = 1,
    background_manifest: Path | None = None,
    merged_background: bool = False,
) -> dict[str, Any]:
    """Resolve a merged geometry to the exact training state and background.

    Returns:
        Configuration, provenance and the six-position background.

    Raises:
        ValueError: If the geometry, metadata, checkpoint or coordinates are invalid.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry: {geometry}")
    if not (0 <= mobile < 6 and 0 <= reference < 6 and mobile != reference):
        raise ValueError("Mobile and reference indices must be distinct, from 0 to 5")
    data_root = data_root.resolve()
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads((data_root / "schema.json").read_text())
    if geometry not in schema["geometry_groups"]:
        raise ValueError(f"Geometry missing from schema: {geometry}")
    source_root = Path(manifest["source_root"])
    if not source_root.is_absolute():
        source_root = (data_root / source_root).resolve()
    with h5py.File(data_root / "unified.h5", "r") as merged:
        group = merged[geometry]
        metadata = group["metadata"]
        config, config_text = training_metadata(metadata)
        step = int(metadata["final_checkpoint_step"][()])
        checkpoint, item = training_checkpoint(
            metadata, manifest, geometry, source_root
        )
        saved_background = checkpoint_background(checkpoint, step, walker_index)
        original_coordinate_dtype = str(saved_background.dtype)
        background = np.asarray(saved_background, dtype=np.float64)
        background_origin = f"checkpoint walker {walker_index}"
        if merged_background:
            background[np.arange(6) != mobile] = saved_scan_background(
                group["wavefunction_scan"], item["sha256"], mobile, reference
            )
            background_origin = "merged wavefunction_scan/fixed_electrons"
        run_id = metadata["run_id"].asstr()[()]
        coordinate_system = metadata["coordinate_system"].asstr()[()]
    if background_manifest is not None:
        background[np.arange(6) != mobile] = supplied_background(
            background_manifest, geometry, item["sha256"], mobile, reference
        )
        background_origin = str(background_manifest.resolve())
    if not np.isfinite(background).all():
        raise ValueError("Background contains nonfinite coordinates")
    return dict(
        geometry=geometry,
        run_id=run_id,
        config=config,
        config_yaml=config_text,
        checkpoint=str(checkpoint),
        checkpoint_sha256=item["sha256"],
        checkpoint_step=step,
        checkpoint_file_id=item["file_id"],
        background=background.tolist(),
        background_origin=background_origin,
        walker_index=walker_index,
        mobile_electron=mobile,
        probe_electron=reference,
        coordinate_system=coordinate_system,
        original_coordinate_dtype=original_coordinate_dtype,
        data_root=str(data_root),
        manifest_sha256=sha256(manifest_path),
        unified_sha256=sha256(data_root / "unified.h5"),
        schema_sha256=sha256(data_root / "schema.json"),
    )


def read_checkpoint_source(
    config_path: Path,
    checkpoint: Path,
    geometry: str,
    *,
    walker_index: int = 0,
    mobile: int = 0,
    reference: int = 1,
    background_manifest: Path | None = None,
) -> dict[str, Any]:
    """Read a standalone training configuration and its explicit checkpoint.

    Returns:
        Configuration, file checksums and the selected six-electron background.

    Raises:
        ValueError: If geometry, electron count, checkpoint step or indices differ.
    """
    modules = {
        "disk": (
            "disk",
            "jaqmc.app.hall.config:HallDiskConfig",
            "jaqmc.app.hall.config.disk",
            "jaqmc.app.hall.config.disk:HallDiskConfig",
        ),
        "sphere": (
            "sphere",
            "spherical",
            "jaqmc.app.hall.config",
            "jaqmc.app.hall.config:HallConfig",
            "jaqmc.app.hall.config:HallSphereConfig",
        ),
        "torus": ("torus", "jaqmc.app.hall.config:HallTorusConfig"),
    }
    if geometry not in modules:
        raise ValueError(f"Unknown geometry: {geometry}")
    if not (0 <= mobile < 6 and 0 <= reference < 6 and mobile != reference):
        raise ValueError("Mobile and reference indices must be distinct, from 0 to 5")
    config_path, checkpoint = config_path.resolve(), checkpoint.resolve()
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    if config["system"]["module"] not in modules[geometry]:
        raise ValueError("Requested geometry disagrees with the training configuration")
    if sum(config["system"]["nspins"]) != 6:
        raise ValueError("Conditional scans require six total electrons")
    digest = sha256(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as saved:
        step = int(saved["step"])
    if checkpoint.stem != f"train_ckpt_{step:06d}":
        raise ValueError("Checkpoint filename disagrees with its saved step")
    saved_background = checkpoint_background(checkpoint, step, walker_index)
    background = np.asarray(saved_background, dtype=np.float64)
    background_origin = f"checkpoint walker {walker_index}"
    if background_manifest is not None:
        background[np.arange(6) != mobile] = supplied_background(
            background_manifest, geometry, digest, mobile, reference
        )
        background_origin = str(background_manifest.resolve())
    if not np.isfinite(background).all():
        raise ValueError("Background contains nonfinite coordinates")
    return dict(
        source_type="training_checkpoint",
        geometry=geometry,
        run_id=config_path.parent.name,
        config=config,
        config_yaml=text,
        config_path=str(config_path),
        config_sha256=sha256(config_path),
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        checkpoint_step=step,
        background=background.tolist(),
        background_origin=background_origin,
        walker_index=walker_index,
        mobile_electron=mobile,
        probe_electron=reference,
        coordinate_system={
            "disk": "Cartesian (x,y)",
            "sphere": "spherical angles (theta,phi)",
            "torus": "fractional (u,v)",
        }[geometry],
        original_coordinate_dtype=str(saved_background.dtype),
        data_root=str(config_path.parent),
    )


def build_system(
    source: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Use workflow factories with only legacy module aliases normalized.

    Returns:
        System, wavefunction, estimators and normalized assembly settings.
    """
    from jaqmc.app.hall.workflow import configure_system, make_estimators
    from jaqmc.utils.config import ConfigManager

    config = deepcopy(source["config"])
    aliases = {
        "disk": "jaqmc.app.hall.config:HallDiskConfig",
        "sphere": "jaqmc.app.hall.config:HallSphereConfig",
        "torus": "jaqmc.app.hall.config:HallTorusConfig",
    }
    config["system"]["module"] = aliases[source["geometry"]]
    if config["wf"]["module"] == "mhpo":
        config["wf"]["module"] = (
            f"jaqmc.app.hall.wavefunction.{source['geometry']}.mhpo"
        )
    # Evaluation requires only energy objects, not the training sampler or writers.
    settings = {
        "system": config["system"],
        "wf": config["wf"],
        "estimators": {
            "energy": config["estimators"]["energy"],
            "enabled": {"energy": True, "angular_momentum": True},
        },
    }
    cfg = ConfigManager(settings)
    system, wavefunction = configure_system(cfg)
    estimators = make_estimators(cfg, wavefunction, system, always_enable_energy=True)
    return system, wavefunction, estimators, settings


def adapt_parameters(
    actual: dict[str, np.ndarray], expected: dict[str, Any], geometry: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Apply only the two authorized checkpoint conversions.

    Returns:
        Strictly matched parameters and a record of compatibility conversions.

    Raises:
        ValueError: If another parameter is missing, unexpected, misshaped or nonfinite.
    """
    actual = dict(actual)
    changes = []
    imag = "params/params/orbital_layer/amplitude_imag"
    real = "params/params/orbital_layer/amplitude"
    kernel = "params/params/backbone_layer/Dense_0/kernel"
    if (
        geometry == "disk"
        and imag in expected
        and imag not in actual
        and real in actual
    ):
        actual[imag] = np.zeros_like(actual[real])
        changes.append(
            dict(
                kind="zero_orbital_imaginary_amplitude",
                key=imag,
                before=None,
                after=list(actual[imag].shape),
            )
        )
    if (
        geometry == "torus"
        and kernel in actual
        and kernel in expected
        and actual[kernel].shape == (5, expected[kernel].shape[1])
        and expected[kernel].shape[0] == 9
    ):
        old = actual[kernel]
        new = np.zeros(expected[kernel].shape, dtype=old.dtype)
        new[:4], new[8] = old[:4], old[4]
        actual[kernel] = new
        changes.append(
            dict(
                kind="zero_center_of_mass_weights",
                key=kernel,
                before=list(old.shape),
                after=list(new.shape),
            )
        )
    if set(actual) != set(expected):
        raise ValueError(
            f"Parameter names differ: missing={sorted(set(expected) - set(actual))}; "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for key, value in actual.items():
        if value.shape != expected[key].shape or not np.isfinite(value).all():
            raise ValueError(
                f"Invalid parameter shape or value: {key}: {value.shape}, "
                f"expected {expected[key].shape}"
            )
    return actual, changes


def restore_params(
    wavefunction: Any, example: Any, checkpoint: Path, geometry: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly restore trained arrays and promote their precision.

    Returns:
        Restored parameter tree, original dtypes and conversion records.

    Raises:
        ValueError: If double precision is disabled or trained parameters do not match.
    """
    import jax
    from flax.traverse_util import unflatten_dict
    from jax import numpy as jnp

    from jaqmc.utils.checkpoint import tree_to_npz

    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "Double precision must be enabled before parameter restoration"
        )
    template = wavefunction.init_params(example, jax.random.PRNGKey(1))
    # Initialization determines the schema. Every returned leaf must be restored.
    expected = tree_to_npz({"params": template})
    with np.load(checkpoint, allow_pickle=False) as saved:
        actual = {key: saved[key] for key in saved.files if key.startswith("params/")}
    original_dtypes = sorted({str(value.dtype) for value in actual.values()})
    original_structure = {
        key: dict(shape=list(value.shape), dtype=str(value.dtype))
        for key, value in actual.items()
    }
    actual, changes = adapt_parameters(actual, expected, geometry)
    promoted = {
        tuple(key.split("/")[1:]): jnp.asarray(
            value, dtype=(jnp.complex128 if np.iscomplexobj(value) else jnp.float64)
        )
        for key, value in actual.items()
    }
    return unflatten_dict(promoted), dict(
        original_parameter_dtypes=original_dtypes,
        compatibility_changes=changes,
        original_parameter_structure=original_structure,
        restored_parameter_structure={
            "params/" + "/".join(key): dict(
                shape=list(value.shape), dtype=str(value.dtype)
            )
            for key, value in promoted.items()
        },
    )


def legacy_interceptor(torus: bool) -> Any:
    """Construct the legacy feature or real-orbital reference used for validation.

    Returns:
        A Flax method interceptor with the original checkpoint conventions.
    """
    from jax import numpy as jnp
    from jax.scipy.special import gammaln

    from jaqmc.app.hall.wavefunction.disk.mhpo import DiskOrbitals
    from jaqmc.wavefunction.backbone.psiformer import PsiformerBackbone

    def intercept(next_fun, args, kwargs, context):
        if (
            torus
            and isinstance(context.module, PsiformerBackbone)
            and context.method_name == "__call__"
        ):
            return next_fun(args[0][..., :4], **kwargs)
        if (
            not torus
            and isinstance(context.module, DiskOrbitals)
            and context.method_name == "__call__"
        ):
            # Independent real-amplitude orbitals for derivative comparisons.
            layer = context.module
            h_one, x, y = args
            coefficients = layer.orbitals_real(h_one) + 1j * layer.orbitals_imag(h_one)
            sigma = jnp.abs(layer.sigma).T
            amplitude = layer.amplitude.T
            basis = jnp.exp(-(x * x + y * y)[:, None, None] * sigma[None])
            if str(layer.envelope_type) != "no_m_learnable":
                m = jnp.arange(layer.num_orbitals)
                norm = -0.5 * gammaln(m + 1.0)
                if str(layer.envelope_type) == "m_learnable2":
                    norm -= 0.5 * m * jnp.log(2.0)
                basis = (
                    basis
                    * (x - 1j * y)[:, None, None] ** m[None, :, None]
                    * jnp.exp(norm)[None, :, None]
                )
            orbitals = jnp.sum(
                coefficients * (amplitude[None] * basis)[..., None], axis=1
            )
            return jnp.moveaxis(orbitals, -1, 0)
        return next_fun(*args, **kwargs)

    return intercept


def verify_compatibility(
    wavefunction: Any, params: Any, electrons: np.ndarray, changes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the full wavefunction and its derivatives to a legacy reference.

    Returns:
        Compatibility status and maximum derivative differences.

    Raises:
        ValueError: If wavefunction values differ beyond the comparison tolerance.
    """
    if not changes:
        return dict(required=False, passed=True)
    import jax
    from flax import linen as nn
    from jax import numpy as jnp

    from jaqmc.app.hall.data import HallData
    from jaqmc.utils.func_transform import grad_maybe_complex

    torus = any(change["kind"] == "zero_center_of_mass_weights" for change in changes)
    legacy_params = jax.tree.map(lambda value: value, params)
    if torus:
        kernel = params["params"]["backbone_layer"]["Dense_0"]["kernel"]
        legacy_params["params"]["backbone_layer"]["Dense_0"]["kernel"] = (
            jnp.concatenate([kernel[:4], kernel[8:9]])
        )

    intercept = legacy_interceptor(torus)

    def current(p, x):
        return wavefunction.logpsi(p, HallData(electrons=x))

    def legacy(p, x):
        with nn.intercept_methods(intercept):
            return wavefunction.logpsi(p, HallData(electrons=x))

    x = jnp.asarray(electrons, dtype=jnp.float64)
    value_error = float(
        jnp.abs(
            jnp.exp(jax.jit(current)(params, x) - jax.jit(legacy)(legacy_params, x)) - 1
        )
    )
    errors = {"wavefunction_relative_error": value_error}
    current_gradient = grad_maybe_complex(current, argnums=1)
    legacy_gradient = grad_maybe_complex(legacy, argnums=1)
    a = np.asarray(jax.jit(current_gradient)(params, x))
    b = np.asarray(jax.jit(legacy_gradient)(legacy_params, x))
    np.testing.assert_allclose(
        a,
        b,
        atol=2e-7,
        rtol=2e-6,
        equal_nan=False,
        err_msg="Legacy gradient changed",
    )
    errors["gradient_max_absolute_error"] = float(np.max(np.abs(a - b)))

    # Evaluate one Hessian column at a time instead of vectorizing all twelve
    # derivative directions through the full trained network simultaneously.
    @jax.jit
    def current_column(p, y, direction):
        return jax.jvp(lambda z: current_gradient(p, z), (y,), (direction,))[1]

    @jax.jit
    def legacy_column(p, y, direction):
        return jax.jvp(lambda z: legacy_gradient(p, z), (y,), (direction,))[1]

    hessian_error = 0.0
    for direction in np.eye(x.size).reshape((x.size, *x.shape)):
        a = np.asarray(current_column(params, x, jnp.asarray(direction)))
        b = np.asarray(legacy_column(legacy_params, x, jnp.asarray(direction)))
        np.testing.assert_allclose(
            a,
            b,
            atol=2e-7,
            rtol=2e-6,
            equal_nan=False,
            err_msg="Legacy Hessian changed",
        )
        hessian_error = max(hessian_error, float(np.max(np.abs(a - b))))
    errors["hessian_max_absolute_error"] = hessian_error
    if not np.isfinite(value_error) or value_error > 2e-7:
        raise ValueError(f"Legacy wavefunction changed: relative error={value_error}")
    return dict(required=True, passed=True, **errors)
