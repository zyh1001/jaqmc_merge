# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run Hall CLI configuration checks and small training jobs in Slurm."""

import argparse
import copy
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "sphere": ("sphere", 3, None),
    "disk": ("disk", 6, None),
    "torus-square": ("torus", 6, [0.0, 1.0]),
    "torus-hex": ("torus", 6, [0.5, 0.8660254037844386]),
}


def training_config(case: str, destination: Path) -> dict[str, Any]:
    """Return the two-electron configuration for one geometry.

    Returns:
        CLI configuration with the geometry's one-third filling convention.
    """
    geometry, flux, tau = CASES[case]
    config: dict[str, Any] = {
        "jax": {"enable_x64": False},
        "workflow": {"save_path": str(destination), "batch_size": 32, "seed": 21},
        "system": {
            "module": f"jaqmc.app.hall.config:Hall{geometry.capitalize()}Config",
            "nspins": [2, 0],
            "flux": flux,
            "interaction_strength": 1.0,
        },
        "wf": {
            "module": (
                f"jaqmc.app.hall.wavefunction.{geometry}.mhpo:"
                f"{geometry.capitalize()}MHPO"
            ),
            "ndets": 1,
            "num_layers": 1,
            "num_heads": 1,
            "heads_dim": 8,
        },
        "train": {
            "optim": {
                "module": "jaqmc.optimizer.kfac",
                "learning_rate": {"rate": 0.05, "warmup": 10},
            },
            "run": {
                "iterations": 100,
                "burn_in": 10,
                "save_step_interval": 50,
                "save_time_interval": 0,
            },
        },
    }
    if geometry == "disk":
        config["system"]["num_orbitals"] = 6
    if tau is not None:
        config["system"]["tau"] = tau
    return config


def run_cli(
    action: str, config: Path, overrides: list[str], log: Path, *, dry_run: bool = False
) -> None:
    """Run the installed CLI and retain its full output.

    Raises:
        RuntimeError: If the CLI fails.
    """
    command = [str(ROOT / ".venv/bin/jaqmc"), "hall", action, "--yml", str(config)]
    if dry_run:
        command += ["--dry-run", "workflow.config.verbose=true"]
    command += overrides
    print(shlex.join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"CLI exited {result.returncode}; see {log}")


def check_training(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check completed updates, finite results and saved parameter changes.

    Returns:
        The resolved configuration and measured training results.
    """
    resolved = yaml.safe_load((directory / "train_config.yaml").read_text())
    iterations = resolved["train"]["run"]["iterations"]
    with (directory / "train_stats.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_array_equal(
        [int(row["step"]) for row in rows], np.arange(iterations)
    )
    for field in ("total_energy_real", "total_energy_real_var"):
        assert np.isfinite([float(row[field]) for row in rows]).all(), field
    checkpoints = sorted(directory.glob("train_ckpt_*.npz"))
    assert len(checkpoints) >= 2, "Save at least two checkpoints to verify updates"
    final = checkpoints[-1]
    with np.load(final, allow_pickle=False) as saved:
        assert int(saved["step"]) == iterations - 1
        assert int(saved["opt_state/kfac/count"]) == iterations
        for key in saved.files:
            if saved[key].dtype.kind in "biufc":
                assert np.isfinite(saved[key]).all(), key
        parameter_keys = [key for key in saved.files if key.startswith("params/")]
        assert parameter_keys, "MHPO must have trainable parameters"
        with np.load(checkpoints[0], allow_pickle=False) as first:
            changed = sum(
                not np.array_equal(first[key], saved[key]) for key in parameter_keys
            )
        assert changed > 0, "Saved parameters did not change"
    return resolved, {
        "iterations": iterations,
        "checkpoint": str(final),
        "wavefunction": resolved["wf"]["module"],
        "energy_configuration": resolved["estimators"]["energy"],
        "final_energy": float(rows[-1]["total_energy_real"]),
        "final_energy_variance": float(rows[-1]["total_energy_real_var"]),
        "changed_parameter_arrays": changed,
    }


def check_evaluation(directory: Path, training_checkpoint: Path) -> dict[str, Any]:
    """Verify finite energy statistics and unchanged restored parameters.

    Returns:
        The measured evaluation energies.
    """
    values = {}
    with np.load(directory / "evaluation_digest.npz", allow_pickle=False) as digest:
        for field in ("total_energy_real", "total_energy_real_var"):
            assert np.isfinite(digest[field]).all(), field
            values[field] = digest[field].tolist()
    with (
        np.load(training_checkpoint, allow_pickle=False) as trained,
        np.load(
            directory / "evaluation_ckpt_000001.npz", allow_pickle=False
        ) as evaluated,
    ):
        for key in trained.files:
            if key.startswith("params/"):
                np.testing.assert_array_equal(trained[key], evaluated[key], err_msg=key)
    return values


def source_revision() -> str:
    """Read the checkout revision without requiring Git on compute nodes.

    Returns:
        The commit ID or an explicit unavailable marker.
    """
    git_dir = ROOT / ".git"
    if not git_dir.is_dir():
        return "unavailable"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.exists():
        return loose.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return "unavailable"


def main() -> int:
    """Run selected cases and write pass, fail or not-run results.

    Returns:
        A nonzero exit code if any required check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "train", "all"), default="all")
    parser.add_argument("--geometry", choices=("all", *CASES), default="all")
    parser.add_argument("--output", type=Path)
    args, overrides = parser.parse_known_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the Slurm shell entry point on a compute node")
    if any("=" not in value or value.startswith("--") for value in overrides):
        parser.error("Additional arguments must use key=value notation")
    job_id = os.environ["SLURM_JOB_ID"]
    output = (
        args.output or ROOT / "integration_test_results" / f"hall-structure-{job_id}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "job_id": job_id,
        "revision": source_revision(),
        "integration_sources": json.loads(
            (ROOT / "integration_sources.json").read_text()
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *(ROOT / "src/jaqmc/app/hall").rglob("*.py"),
                    *(ROOT / "src/jaqmc/optimizer/kfac").rglob("*.py"),
                ]
            )
        },
        "python": sys.version,
        "nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "devices": [
            {"platform": device.platform, "kind": device.device_kind}
            for device in jax.devices()
        ],
        "packages": {
            name: version(name)
            for name in ("jax", "jaxlib", "flax", "optax", "kfac-jax", "numpy")
        },
        "mode": args.mode,
        "overrides": overrides,
        "cases": {},
    }
    selected = list(CASES) if args.geometry == "all" else [args.geometry]
    for case in selected:
        report["cases"][case] = {
            "dry-run": "not-run",
            "train": "not-run",
            "evaluate": "not-run",
        }
    manifest = output / "summary.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    failures = 0
    for case in selected:
        result = report["cases"][case]
        directory = output / case
        directory.mkdir()
        config = training_config(case, directory / "train")
        config_path = directory / "input.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        # Output directories are managed by this script for every selected case.
        cli_overrides = [
            *overrides,
            f"workflow.save_path={directory / 'train'}",
            "workflow.restore_path=null",
        ]
        phase = "dry-run" if args.mode != "train" else "train"
        try:
            if args.mode in ("dry-run", "all"):
                run_cli(
                    "train",
                    config_path,
                    cli_overrides,
                    directory / "dry-run.log",
                    dry_run=True,
                )
                result["dry-run"] = "pass"
            if args.mode in ("train", "all"):
                phase = "train"
                run_cli("train", config_path, cli_overrides, directory / "train.log")
                resolved, measurements = check_training(directory / "train")
                result["training_results"] = measurements
                result["train"] = "pass"
                phase = "evaluate"
                evaluation = copy.deepcopy(resolved)
                evaluation.pop("train")
                evaluation["workflow"].update(
                    save_path=str(directory / "evaluation"),
                    restore_path=None,
                    source_path=measurements["checkpoint"],
                )
                evaluation["run"] = {"iterations": 2, "burn_in": 0}
                eval_path = directory / "evaluation.yaml"
                eval_path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
                run_cli("evaluate", eval_path, [], directory / "evaluate.log")
                result["evaluation_results"] = check_evaluation(
                    directory / "evaluation", Path(measurements["checkpoint"])
                )
                result["evaluate"] = "pass"
            print(f"PASS: {case}", flush=True)
        except Exception as error:
            failures += 1
            result[phase] = "fail"
            result["error"] = str(error)
            traceback.print_exc()
            print(f"FAIL: {case}: {error}", flush=True)
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results: {manifest}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
