# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""从已完成采样的圆盘检查点恢复对关联汇总文件."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from evaluateDisk import checksum, run_analysis, validate_result, write_json

from jaqmc.app.hall.estimator.disk.pair_correlation import DiskPairCorrelation


def recover(save_path: Path, *, skip_analysis: bool = False) -> None:
    """恢复完整采样的汇总结果，并保留原始失败记录.

    Raises:
        FileExistsError: 汇总文件或恢复前记录已经存在。
        ValueError: 来源文件改变、采样未完成或累计次数不一致。
    """
    digest_path = save_path / "evaluation_digest.npz"
    manifest_path = save_path / "run_manifest.json"
    backup_path = save_path / "run_manifest.before_recovery.json"
    if digest_path.exists() or backup_path.exists():
        raise FileExistsError("汇总文件或恢复前记录已存在，请先检查已有恢复结果。")
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    if manifest["quantity"] != "pair_correlation":
        raise ValueError("此程序只恢复对关联结果。")
    config = yaml.safe_load((save_path / "evaluation_config.yaml").read_text())
    source = Path(manifest["source_checkpoint"])
    training = Path(manifest["training_config"])
    if (
        checksum(source) != manifest["source_checkpoint_sha256"]
        or checksum(training) != manifest["training_config_sha256"]
    ):
        raise ValueError("来源训练检查点或训练配置已经改变。")
    checkpoint = max(
        save_path.glob("evaluation_ckpt_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    with np.load(checkpoint, allow_pickle=False) as data:
        steps = int(data["step"]) + 1
        batch = data["batched_data/data/electrons"].shape[0]
        prefix = "estimator_state/pair_correlation/"
        state = {
            key.removeprefix(prefix): data[key]
            for key in data.files
            if key.startswith(prefix)
        }
    if steps != config["run"]["iterations"]:
        raise ValueError("检查点尚未完成请求的全部采样步数。")
    if (
        batch != config["workflow"]["batch_size"]
        or batch != manifest["actual_batch_size"]
    ):
        raise ValueError("检查点实际配置数与评估记录不一致。")
    if not np.isclose(
        np.sum(state["sample_counts"], dtype=np.float64),
        steps * batch,
        rtol=0,
        atol=0.5,
    ):
        raise ValueError("检查点累计配置次数与采样步数不一致。")
    estimator = DiskPairCorrelation(**config["estimators"]["pair_correlation"])
    digest = estimator.finalize_state(state, n_steps=steps)
    with backup_path.open("x", encoding="utf-8") as stream:
        stream.write(original)
    manifest["recovery"] = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checksum(checkpoint),
        "steps": steps,
        "sample_count": steps * batch,
        "resampled": False,
        "recovery_script_sha256": checksum(Path(__file__).resolve()),
        "finalizer_sha256": checksum(
            Path(__file__).resolve().parents[1]
            / "src/jaqmc/app/hall/estimator/disk/pair_correlation.py"
        ),
    }
    manifest.pop("error", None)
    try:
        manifest["status"] = "recovering"
        write_json(manifest_path, manifest)
        temporary = digest_path.with_suffix(".npz.tmp")
        with temporary.open("xb") as stream:
            np.savez(stream, **digest)
        temporary.replace(digest_path)
        result = validate_result(config, "pair_correlation", save_path, source)
        result.update(recovered_from_checkpoint=True, resampled=False)
        write_json(save_path / "validation.json", result)
        manifest["status"] = "evaluated"
        write_json(manifest_path, manifest)
        if not skip_analysis:
            run_analysis("pair_correlation", Path(manifest["working_dir"]), save_path)
        manifest["status"] = "completed"
        manifest["recovery"]["completed_at"] = datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(timespec="seconds")
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(manifest_path, manifest)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    recover(
        args.evaluation_dir.expanduser().resolve(), skip_analysis=args.skip_analysis
    )


if __name__ == "__main__":
    main()
