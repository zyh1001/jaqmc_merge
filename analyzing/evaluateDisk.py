# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""准备圆盘评估配置，调用已有计算程序并分析结果."""

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

DEFAULT_ITERATIONS = {
    "energy": 10000,
    "density": 3000,
    "pair_correlation": 4000,
    "wave_scan": 1,
}
OBSERVABLE_NAMES = {"wave_scan": "wavefunction_scan"}
ANALYSIS_SCRIPTS = {
    "energy": "analyzingEnergy.py",
    "density": "analyzingDensity.py",
    "pair_correlation": "analyzingPairCorrelation.py",
    "wave_scan": "evaluateWaveScan.py",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取提交参数及末尾的点分配置.

    Returns:
        解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quantity", choices=tuple(DEFAULT_ITERATIONS))
    parser.add_argument("working_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reference-r", type=float, default=3.0)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    return parser.parse_intermixed_args(argv)


def checksum(path: Path) -> str:
    """计算文件校验值，分块读取检查点.

    Returns:
        十六进制校验值。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """完整写入后替换说明文件."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """允许调整评估设置，保持来源模型和采样配置数不变.

    Raises:
        ValueError: 参数格式错误或试图覆盖受保护的配置。
    """
    for item in overrides:
        key, separator, value = item.partition("=")
        parts = key.split(".")
        if not separator or not all(parts):
            raise ValueError(f"参数应为 key=value：{item}")
        if parts[0] not in {"run", "sampler", "jax", "logging", "estimators"}:
            raise ValueError(f"不能通过末尾参数改动模型或来源配置：{key}")
        if parts[:2] == ["estimators", "enabled"]:
            raise ValueError("每个作业只启用所选物理量，不能覆盖 estimators.enabled。")
        section = config
        for part in parts[:-1]:
            child = section.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"配置不是可继续分层的字典：{key}")
            section = child
        section[parts[-1]] = yaml.safe_load(value)


def build_config(
    train: dict[str, Any],
    args: argparse.Namespace,
    checkpoint: Path,
    actual_batch_size: int,
) -> dict[str, Any]:
    """仅保留当前评估使用的配置，不继承训练或旧评估状态.

    Returns:
        可交给原有评估程序的配置。

    Raises:
        ValueError: 来源几何或实际采样配置数不符合要求。
    """
    if actual_batch_size != 8192:
        raise ValueError(
            f"本脚本要求检查点实际包含8192个配置，当前为{actual_batch_size}。"
        )
    if "disk" not in str(train["system"].get("module", "")).lower():
        raise ValueError("训练配置不是圆盘几何。")
    config = {
        key: deepcopy(train[key])
        for key in ("system", "wf", "sampler", "jax", "logging")
        if key in train
    }
    config["system"]["module"] = "jaqmc.app.hall.config:HallDiskConfig"
    observable = OBSERVABLE_NAMES.get(args.quantity, args.quantity)
    config["workflow"] = {
        "source_path": str(checkpoint),
        "batch_size": actual_batch_size,
        "seed": args.seed,
    }
    config["run"] = {
        "iterations": args.iterations
        if args.iterations is not None
        else DEFAULT_ITERATIONS[args.quantity],
        "burn_in": args.burn_in,
        "digest_step_interval": 0,
    }
    config["estimators"] = {
        "enabled": {
            "energy": args.quantity == "energy",
            "angular_momentum": args.quantity == "energy",
            "density": args.quantity == "density",
            "pair_correlation": args.quantity == "pair_correlation",
            "structure_factor": False,
            "one_rdm": False,
            "edge_green": False,
            "wavefunction_scan": args.quantity == "wave_scan",
            "short_range": False,
        }
    }
    if args.quantity == "energy":
        energy = deepcopy(train.get("estimators", {}).get("energy", {}))
        energy.setdefault("kinetic", {})["vmap_chunk_size"] = 64
        config["estimators"]["energy"] = energy
    elif args.quantity == "density":
        config["estimators"][observable] = {"bins": 400, "range_max": 10.0}
    elif args.quantity == "pair_correlation":
        config["estimators"][observable] = {
            "bins": 800,
            "max_r": 15.0,
            "r": args.reference_r,
        }
    else:
        config["estimators"][observable] = {
            "mobile_electron": 0,
            "selection_samples": 8192,
            "x_min": -8.0,
            "x_max": 8.0,
            "y_min": -8.0,
            "y_max": 8.0,
            "n_x": 301,
            "n_y": 301,
            "chunk_size": 256,
            "probe_electron": 1,
            "radial_direction": [1.0, 0.0],
            "radial_min": 0.001,
            "radial_max": 3.0,
            "radial_points": 200,
        }
    apply_overrides(config, args.overrides)
    validate_settings(config, args.quantity, actual_batch_size)
    return config


def validate_settings(config: dict[str, Any], quantity: str, batch: int) -> None:
    """检查评估步数和波函数背景选择数量.

    Raises:
        ValueError: 评估设置不合法。
    """
    for key in ("iterations", "burn_in"):
        value = config["run"][key]
        minimum = 1 if key == "iterations" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"run.{key} 必须为不小于{minimum}的整数。")
    if config["estimators"]["enabled"].get("energy") != (quantity == "energy"):
        raise ValueError("启用的估计器与请求物理量不符。")
    if quantity == "wave_scan":
        scan = config["estimators"]["wavefunction_scan"]
        if scan["selection_samples"] > batch * config["run"]["iterations"]:
            raise ValueError("波函数背景选择数量超过本次评估可提供的配置数。")


def read_checkpoint(path: Path, n_electrons: int) -> tuple[int, int]:
    """确认实际配置数和电子坐标形状，读取来源步数.

    Returns:
        来源步数和实际采样配置数。

    Raises:
        ValueError: 电子坐标形状不符合训练配置。
    """
    with np.load(path, allow_pickle=False) as data:
        coordinates = data["batched_data/data/electrons"]
        if coordinates.ndim != 3 or coordinates.shape[1:] != (n_electrons, 2):
            raise ValueError(f"检查点电子坐标形状不符：{coordinates.shape}")
        return int(data["step"]), coordinates.shape[0]


def validate_digest(
    config: dict[str, Any], quantity: str, save_path: Path
) -> dict[str, Any]:
    """检查正式评估汇总文件中的物理量和累计次数.

    Returns:
        物理量检查结果。

    Raises:
        ValueError: 物理量缺失或累计数量不符合设置。
    """
    result: dict[str, Any] = {"passed": True, "quantity": quantity}
    digest_path = save_path / "evaluation_digest.npz"
    batch = config["workflow"]["batch_size"]
    expected_steps = config["run"]["iterations"]
    with np.load(digest_path, allow_pickle=False) as digest:
        if quantity == "energy":
            keys = ("total_energy", "total_energy_real", "total_energy_real_var")
            if not all(
                key in digest and np.isfinite(digest[key]).all() for key in keys
            ):
                raise ValueError("能量输出缺少有效字段。")
            result["energy_real"] = float(digest["total_energy_real"])
        elif quantity == "density":
            steps = int(digest["density:n_steps"])
            radial_steps = int(digest["density_radial:n_steps"])
            if steps != expected_steps or radial_steps != expected_steps:
                raise ValueError("密度累计步数与请求步数不一致。")
            integrated = float(
                np.sum(digest["density"], dtype=np.float64) / (steps * batch)
            )
            radial_integrated = float(
                np.sum(digest["density_radial"], dtype=np.float64)
                / (radial_steps * batch)
            )
            number = sum(config["system"]["nspins"])
            result.update(
                integrated_number_grid=integrated,
                integrated_number_radial=radial_integrated,
            )
            if not (
                abs(integrated - number) <= 0.01
                and abs(radial_integrated - number) <= 0.01
            ):
                raise ValueError(f"密度积分与电子数不符，请检查范围与归一化：{result}")
        elif quantity == "pair_correlation":
            requested = config["estimators"]["pair_correlation"]["r"]
            result["reference_r"] = float(digest["pair_correlation:reference_r"])
            if not np.isclose(result["reference_r"], requested):
                raise ValueError("对关联参考半径与请求值不一致。")
            if not np.isclose(
                float(digest["pair_correlation:sample_count"]), batch * expected_steps
            ):
                raise ValueError("对关联累计配置次数与请求值不一致。")
        else:
            scan = config["estimators"]["wavefunction_scan"]
            expected_shape = (scan["n_y"], scan["n_x"])
            if digest["wavefunction_scan:log_abs"].shape != expected_shape:
                raise ValueError("波函数扫描网格与请求值不一致。")
            result["grid_shape"] = list(expected_shape)
    return result


def validate_result(
    config: dict[str, Any], quantity: str, save_path: Path, checkpoint: Path
) -> dict[str, Any]:
    """核对正式评估产物，不额外运行采样或导数计算.

    Returns:
        物理量、参数和实际采样配置数的检查结果。

    Raises:
        FileNotFoundError: 缺少最终评估检查点。
        ValueError: 模型参数或实际采样配置数发生改变。
    """
    result = validate_digest(config, quantity, save_path)
    batch = config["workflow"]["batch_size"]
    paths = list(save_path.glob("evaluation_ckpt_*.npz"))
    if not paths:
        raise FileNotFoundError("缺少正式评估结束时的检查点。")
    latest = max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    with (
        np.load(checkpoint, allow_pickle=False) as source,
        np.load(latest, allow_pickle=False) as evaluated,
    ):
        keys = {key for key in source.files if key.startswith("params/")}
        if keys != {key for key in evaluated.files if key.startswith("params/")}:
            raise ValueError("评估检查点的参数结构与来源不一致。")
        if not all(np.array_equal(source[key], evaluated[key]) for key in keys):
            raise ValueError("评估过程中网络参数发生了改变。")
        actual = evaluated["batched_data/data/electrons"].shape[0]
        if actual != batch:
            raise ValueError(f"实际配置数为{actual}，配置记录为{batch}。")
    result.update(
        actual_batch_size=actual,
        parameters_unchanged=True,
        evaluation_checkpoint=str(latest),
    )
    return result


def run_analysis(quantity: str, working_dir: Path, save_path: Path) -> None:
    """复用原有分析程序，波函数只分析已有结果."""
    project = Path(os.environ.get("DISK_ANALYSIS_ROOT", "/data/home/zyh/work/jaqmc"))
    python = Path(
        os.environ.get("DISK_ANALYSIS_PYTHON", str(project / ".venv/bin/python"))
    )
    script = project / "analyzing" / ANALYSIS_SCRIPTS[quantity]
    command = [str(python), str(script)]
    if quantity == "wave_scan":
        command += [
            "--analysis-only",
            "--working-dir",
            str(working_dir),
            "--save-path",
            str(save_path),
        ]
    else:
        command.append(str(save_path))
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
    subprocess.run(command, env=env, check=True)


def main() -> None:
    args = parse_args()
    working_dir = args.working_dir.expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint or Path("train_ckpt_149999.npz")
    if not checkpoint.is_absolute():
        checkpoint = working_dir / checkpoint
    checkpoint = checkpoint.resolve()
    config_path = working_dir / "train_config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        train = yaml.safe_load(stream)
    step, batch = read_checkpoint(checkpoint, sum(train["system"]["nspins"]))
    config = build_config(train, args, checkpoint, batch)
    label = OBSERVABLE_NAMES.get(args.quantity, args.quantity)
    if args.quantity == "pair_correlation":
        label += f"_r{float(config['estimators']['pair_correlation']['r'])}"
    save_path = (args.save_path or working_dir / "eval" / label).expanduser().resolve()
    if save_path.exists():
        raise FileExistsError(
            f"输出目录已存在，请用 --save-path 指定新目录：{save_path}"
        )
    config["workflow"]["save_path"] = str(save_path)
    if not args.skip_analysis:
        analysis = Path(
            os.environ.get("DISK_ANALYSIS_ROOT", "/data/home/zyh/work/jaqmc")
        )
        python = Path(
            os.environ.get("DISK_ANALYSIS_PYTHON", str(analysis / ".venv/bin/python"))
        )
        if (
            not python.is_file()
            or not (analysis / "analyzing" / ANALYSIS_SCRIPTS[args.quantity]).is_file()
        ):
            raise FileNotFoundError("缺少已有分析程序或其虚拟环境。")
    manifest: dict[str, Any] = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "quantity": args.quantity,
        "project_root": str(project),
        "working_dir": str(working_dir),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_step": step,
        "source_checkpoint_sha256": checksum(checkpoint),
        "training_config": str(config_path),
        "training_config_sha256": checksum(config_path),
        "actual_batch_size": batch,
        "save_path": str(save_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "prepared",
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    cli = project / ".venv/bin/jaqmc"
    if args.dry_run:
        with TemporaryDirectory(prefix="disk-eval-config-") as temporary:
            requested = Path(temporary) / "requested_evaluation.yaml"
            requested.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            subprocess.run(
                [str(cli), "hall", "evaluate", "--yaml", str(requested), "--dry-run"],
                check=True,
            )
        return
    save_path.mkdir(parents=True, exist_ok=False)
    requested = save_path / "requested_evaluation.yaml"
    requested.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest["requested_config_sha256"] = checksum(requested)
    manifest_path = save_path / "run_manifest.json"
    write_json(manifest_path, manifest)
    try:
        manifest["status"] = "evaluating"
        write_json(manifest_path, manifest)
        subprocess.run(
            [str(cli), "hall", "evaluate", "--yaml", str(requested)], check=True
        )
        result = validate_result(config, args.quantity, save_path, checkpoint)
        if (
            checksum(checkpoint) != manifest["source_checkpoint_sha256"]
            or checksum(config_path) != manifest["training_config_sha256"]
        ):
            raise ValueError("来源训练检查点或训练配置发生改变。")
        write_json(save_path / "validation.json", result)
        manifest["status"] = "evaluated"
        write_json(manifest_path, manifest)
        if not args.skip_analysis:
            run_analysis(args.quantity, working_dir, save_path)
        manifest["status"] = "completed"
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
