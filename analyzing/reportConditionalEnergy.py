# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Plot probability-weighted conditional energies separately for each geometry."""

import argparse
import base64
import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from conditional_energy import COMPONENTS, CURVE_COMPONENTS, PLOT_RANGES
from conditional_energy_sources import sha256, write_json

GEOMETRY_NAMES = {"disk": "圆盘", "sphere": "球面", "torus": "圆环面"}


def configure_value_scale(
    axis: Any,
    *,
    positive_log: bool = False,
    linthresh: float = 1e-14,
    force_linear: bool = False,
) -> dict:
    """Keep the original automatic scale rule, preserving negative values and zero.

    Returns:
        Scale names and the threshold used for a symmetric logarithmic axis.
    """
    low, high = sorted(axis.get_xlim())
    visible = []
    for line in axis.lines:
        x, y = (
            np.asarray(line.get_xdata(), dtype=float),
            np.asarray(line.get_ydata(), dtype=float),
        )
        visible.append(y[np.isfinite(x) & np.isfinite(y) & (x >= low) & (x <= high)])
    values = np.concatenate(visible) if visible else np.array([])
    nonzero = np.abs(values[values != 0])
    orders = (
        float(np.log10(nonzero.max()) - np.log10(nonzero.min()))
        if len(nonzero)
        else 0.0
    )
    scale = "linear"
    if not force_linear and orders >= 3:
        scale = "log" if positive_log and np.all(values >= 0) else "symlog"
    if scale == "log":
        axis.set_yscale("log", nonpositive="mask")
    elif scale == "symlog":
        axis.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        axis.axhspan(-linthresh, linthresh, color="gray", alpha=0.08, zorder=0)
    return dict(
        x_scale=axis.get_xscale(),
        x_limits=list(axis.get_xlim()),
        y_scale=scale,
        linear_threshold=linthresh if scale == "symlog" else None,
    )


def load_disk_integrals(directory: Path, geometry: str) -> dict:
    """Validate the retained disk conditional integrals.

    Returns:
        All original integral arrays.

    Raises:
        ValueError: If a different geometry or component schema is supplied.
    """
    if geometry != "disk":
        raise ValueError("Full-space integrals are only supported for disk results")
    with np.load(directory / "integrals.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    if tuple(data["components"]) != COMPONENTS:
        raise ValueError("Unexpected disk integral components")
    np.testing.assert_allclose(data["probability"].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(data["cumulative"][-1], data["mean"], atol=1e-8)
    np.testing.assert_allclose(
        data["actual"][:, 0], data["actual"][:, 1:4].sum(axis=1), atol=1e-8
    )
    return data


def load_result(path: Path) -> dict:
    """Validate one completed scan and load its radial and optional integral data.

    Returns:
        Result data and provenance for one model.

    Raises:
        ValueError: If results are incomplete, changed or physically inconsistent.
    """
    directory = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("complete") or not validation.get("identities_passed"):
        raise ValueError(f"Incomplete scan: {directory}")
    for name, digest in validation["files_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Result file changed: {directory / name}")
    with np.load(directory / "curves_refined.npz", allow_pickle=False) as data:
        curves = {key: data[key] for key in data.files}
    if tuple(curves["components"]) != CURVE_COMPONENTS:
        raise ValueError(f"Unexpected curve components: {directory}")
    x, y = curves["node_radius"], curves["node_conditional"]
    if (
        y.shape != (len(x), len(CURVE_COMPONENTS))
        or not np.isfinite(y).all()
        or not np.isfinite(x).all()
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(f"Invalid conditional curves: {directory}")
    np.testing.assert_allclose(y[:, 0], y[:, 1] + y[:, 2], rtol=1e-10, atol=1e-8)
    for low, high, _, _ in PLOT_RANGES:
        if np.count_nonzero((x >= low) & (x <= high)) < 2:
            raise ValueError(
                f"Insufficient curve data for [{low}, {high}]: {directory}"
            )
    geometry = manifest["geometry"]
    if geometry not in GEOMETRY_NAMES:
        raise ValueError(f"Unknown result geometry: {geometry}")
    group = dict(
        directory=directory, manifest=manifest, validation=validation, curves=curves
    )
    if (directory / "integrals.npz").is_file():
        group["integrals"] = load_disk_integrals(directory, geometry)
    return group


def check_overlay(reference: dict, candidate: dict) -> None:
    """Require the same physical state and fixed positions for model overlays.

    Raises:
        ValueError: If physical parameters or the conditional background differ.
    """
    fixed_reference = np.delete(
        reference["background"], reference["mobile_electron"], axis=0
    )
    fixed_candidate = np.delete(
        candidate["background"], candidate["mobile_electron"], axis=0
    )
    if (
        reference["settings"]["system"] != candidate["settings"]["system"]
        or reference["settings"]["estimators"]["energy"]
        != candidate["settings"]["estimators"]["energy"]
        or reference["mobile_electron"] != candidate["mobile_electron"]
        or reference["probe_electron"] != candidate["probe_electron"]
        or not np.array_equal(fixed_reference, fixed_candidate)
    ):
        raise ValueError(
            "Overlay requires the same physical parameters and fixed positions"
        )


def load_results(roots: list[Path]) -> dict[str, list[dict]]:
    """Load completed scans, grouping compatible models by geometry.

    Returns:
        One list of validated model results for each geometry.

    Raises:
        ValueError: If no scans exist or an overlay would mix different systems.
    """
    paths = sorted(
        {
            p
            for root in roots
            for p in (
                [root / "run_manifest.json"]
                if (root / "run_manifest.json").is_file()
                else root.rglob("run_manifest.json")
            )
        }
    )
    if not paths:
        raise ValueError("No conditional-energy run manifests found")
    groups: dict[str, list[dict]] = {}
    for path in paths:
        group = load_result(path)
        peers = groups.setdefault(group["manifest"]["geometry"], [])
        if peers:
            check_overlay(peers[0]["manifest"], group["manifest"])
        peers.append(group)
    return groups


def export_curves(groups: list[dict], labels: list[str], output: Path) -> None:
    """Export the plotted angular means without altering their normalization."""
    rows = []
    for label, group in zip(labels, groups, strict=True):
        for radius, values in zip(
            group["curves"]["node_radius"],
            group["curves"]["node_conditional"],
            strict=True,
        ):
            rows.append(
                dict(
                    模型=label,
                    距离=float(radius),
                    总能量=float(values[0]),
                    动能=float(values[1]),
                    势能=float(values[2]),
                    能量虚部=float(values[3]),
                )
            )
    with (output / "逐距离曲线.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_disk_distributions(groups: list[dict], labels: list[str], save: Any) -> None:
    """Preserve the short-range and full-domain disk distribution figures."""
    import matplotlib.pyplot as plt

    maximum = max(group["integrals"]["edges"][-1] for group in groups)
    for filename, field, title in (
        ("probability", "node_probability_density", "两电子距离的条件概率密度"),
        ("conditional_energy", "node_conditional", "给定距离时的局域总能量"),
        (
            "actual_contribution",
            "node_contribution_density",
            "每单位距离的实际能量贡献",
        ),
        ("cumulative_energy", "cumulative", "能量累计贡献"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for label, group in zip(labels, groups, strict=True):
            data = group["integrals"]
            x = data["edges"][1:] if field == "cumulative" else data["node_radius"]
            y = data[field] if filename == "probability" else data[field][:, 0]
            if field == "cumulative":
                x, y = np.r_[0.0, x], np.r_[0.0, y]
            for axis in axes:
                axis.plot(x, y, label=label)
        for axis, bounds in zip(axes, ((0.001, 3.0), (0.0, maximum)), strict=True):
            axis.set(xlim=bounds, xlabel="两电子距离（磁长）", ylabel=title)
            axis.legend()
            axis.grid(alpha=0.25)
        axes[0].set_xscale("log")
        save(
            fig,
            filename,
            title,
            positive_log=filename == "probability",
            threshold=1.0 if filename == "conditional_energy" else 1e-14,
        )


def plot_disk_details(groups: list[dict], labels: list[str], save: Any) -> None:
    """Preserve cumulative components, radial details and fixed positions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, axis in enumerate(axes.flat):
        for label, group in zip(labels, groups, strict=True):
            data = group["integrals"]
            axis.plot(
                np.r_[0.0, data["edges"][1:]],
                np.r_[0.0, data["cumulative"][:, column]],
                label=label,
            )
        axis.set(
            title=("总能量", "动能", "电子间势能", "背景势能")[column],
            xlabel="两电子距离（磁长）",
        )
        axis.legend()
        axis.grid(alpha=0.25)
    save(fig, "components", "动能与势能的累计贡献", threshold=1e-14)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, low, high in ((axes[0], 0.05, 1.0), (axes[1], 1.0, 3.0)):
        for label, group in zip(labels, groups, strict=True):
            data = group["integrals"]
            mask = (data["node_radius"] >= low) & (data["node_radius"] <= high)
            axis.plot(
                data["node_radius"][mask],
                data["node_conditional"][mask, 0],
                label=label,
            )
        axis.set(xlim=(low, high), xlabel="两电子距离（磁长）", ylabel="条件能量")
        axis.legend()
        axis.grid(alpha=0.25)
    save(fig, "conditional_energy_details", "局域能量期望的分段细节")
    manifest = groups[0]["manifest"]
    fig, axis = plt.subplots(figsize=(7, 6))
    fixed = np.delete(
        np.asarray(manifest["background"]), manifest["mobile_electron"], axis=0
    )
    axis.scatter(fixed[:, 0], fixed[:, 1], label="固定电子")
    center = np.asarray(manifest["background"])[manifest["probe_electron"]]
    axis.scatter(*center, marker="*", s=100, label="距离参考电子")
    for radius in (0.5, 1.0, 2.0, 3.0):
        axis.add_patch(plt.Circle(center, radius, fill=False, alpha=0.35))
    axis.set(aspect="equal", xlabel="横坐标（磁长）", ylabel="纵坐标（磁长）")
    axis.autoscale_view()
    axis.legend()
    save(fig, "background", "固定电子位置", linear=True)


def plot_disk_comparisons(groups: list[dict], labels: list[str], save: Any) -> None:
    """Preserve bulk probability plots and common-reference energy differences."""
    import matplotlib.pyplot as plt

    bounds = []
    for group in groups:
        data = group["integrals"]
        low, high = np.searchsorted(data["cumulative_probability"], [0.001, 0.999])
        bounds.append((data["edges"][low], data["edges"][high + 1]))
    limits = (min(b[0] for b in bounds), max(b[1] for b in bounds))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, group in zip(labels, groups, strict=True):
        data = group["integrals"]
        x = data["node_radius"]
        mask = (x >= limits[0]) & (x <= limits[1])
        axes[0].plot(x[mask], data["node_probability_density"][mask], label=label)
        axes[1].plot(x[mask], data["node_contribution_density"][mask, 0], label=label)
    for axis, ylabel in zip(axes, ("概率 / 单位距离", "能量 / 单位距离"), strict=True):
        axis.set(xlim=limits, xlabel="两电子距离（磁长）", ylabel=ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
    save(fig, "bulk_linear", "主要概率区域的线性比较", linear=True)
    reference = groups[0]["integrals"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, group in zip(labels, groups, strict=True):
        data = group["integrals"]
        x = data["edges"][1:]
        # Integrate each piecewise-constant annular contribution to shared radii.
        baseline = np.interp(
            x**2,
            np.r_[0.0, reference["edges"][1:] ** 2],
            np.r_[0.0, reference["cumulative"][:, 0]],
        )
        axes[0].plot(
            np.r_[0.0, x], np.r_[0.0, data["cumulative"][:, 0] - baseline], label=label
        )
        centered = (
            data["cumulative"][:, 0]
            - reference["mean"][0] * data["cumulative_probability"]
        )
        axes[1].plot(np.r_[0.0, x], np.r_[0.0, centered], label=label)
    for axis, title in zip(
        axes,
        (f"相对{labels[0]}的累计贡献差", f"以{labels[0]}条件平均能量为共同基准"),
        strict=True,
    ):
        axis.set(title=title, xlabel="两电子距离（磁长）", ylabel="能量")
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.legend()
        axis.grid(alpha=0.25)
    save(fig, "differences", "能量差与共同基准曲线", threshold=1e-10)


def plot_energy_components(
    groups: list[dict], labels: list[str], distance_name: str, save: Any
) -> None:
    """Plot directional kinetic and full potential energy means."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for column, axis in enumerate(axes, 1):
        for label, group in zip(labels, groups, strict=True):
            data = group["curves"]
            axis.plot(
                data["node_radius"],
                data["node_conditional"][:, column],
                label=label,
            )
        axis.set(
            xscale="log",
            xlabel=f"两电子{distance_name}（磁长）",
            ylabel="加权方向平均",
            title="动能" if column == 1 else "势能",
            xlim=(0.001, 2.0),
        )
        axis.legend()
        axis.grid(alpha=0.25)
    save(fig, "conditional_energy_components", "局域动能与势能")


def report_notes(groups: list[dict], distance_name: str) -> list[str]:
    """Explain the physical meaning and numerical limits of the plotted results.

    Returns:
        Chinese report paragraphs for the selected geometry and calculation mode.
    """
    manifest = groups[0]["manifest"]
    geometry = manifest["geometry"]
    name = GEOMETRY_NAMES[geometry]
    kappa = manifest["config"]["system"]["interaction_strength"]
    notes = [
        f"{name}中的六电子体系，相互作用强度为{kappa:g}。固定五个电子，让电子{manifest['mobile_electron']}围绕电子{manifest['probe_electron']}移动。横轴为两电子的{distance_name}，长度单位为磁长。",
        "每个位置的局域能量包含全部六个电子的动能与完整势能。方向平均按各模型的波函数模平方加权，分段显示不改变平均方式。",
        "图中展示固定背景下的条件能量，不是全体系采样平均能量。局域能量虚部单独保存，不能要求每一点都为零。",
        "能量跨度达到三个数量级时采用对称对数纵轴，零点附近的线性区间为正负一。该变换仅影响显示。",
    ]
    converged = all(
        group["validation"]["convergence"]["angular"]["passed"] for group in groups
    )
    notes.append(
        "方向加密检查通过。"
        if converged
        else "部分曲线尚未通过方向加密检查，应增加方向数后复核。"
    )
    if geometry != "disk" or not all("integrals" in g for g in groups):
        notes.append(
            "本报告仅包含径向扫描，没有计算完整空间概率、累计能量贡献或完整空间条件平均值。"
        )
    else:
        notes.append(
            "圆盘的概率与累计贡献使用完整计算区域的归一化。累计曲线末端仅为此固定背景下的条件平均能量。"
        )
        for group in groups:
            mean = group["integrals"]["mean"]
            label = group["manifest"]["model"]
            passed = group["validation"]["convergence"]["disk_integrals"]["passed"]
            status = "通过" if passed else "尚未通过"
            notes.append(
                f"模型{label}的完整区域条件平均总能量为{mean[0]:.8g}，"
                f"动能为{mean[1]:.8g}，电子间势能为{mean[2]:.8g}，"
                f"背景项为{mean[3]:.8g}。积分网格与尾部检查{status}。"
            )
    return notes


def create_report(groups: list[dict], output: Path) -> None:
    """Create figures, CSV curves, and a Chinese report for one geometry.

    Raises:
        ValueError: If the output directory is already populated.
    """
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Report exists; choose a new --output: {output}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties, fontManager

    font_path = Path("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir()
    manifest = groups[0]["manifest"]
    geometry = manifest["geometry"]
    name = GEOMETRY_NAMES[geometry]
    distance_name = (
        "弦长"
        if geometry == "sphere"
        else "最短周期距离"
        if geometry == "torus"
        else "平面距离"
    )
    notes = report_notes(groups, distance_name)
    title = f"{name}固定五电子背景下的局域能量"
    markdown = f"# {title}\n\n" + "\n\n".join(notes)
    html_blocks = [f"<h1>{title}</h1>"] + [
        f"<p>{html.escape(note)}</p>" for note in notes
    ]
    scales = {}
    labels = [
        group["manifest"].get(
            "display_name", group["manifest"].get("model", f"模型{i + 1}")
        )
        for i, group in enumerate(groups)
    ]
    export_curves(groups, labels, output)
    with PdfPages(output / "固定背景能量分析报告.pdf") as pdf:

        def save(
            fig,
            filename,
            figure_title,
            *,
            positive_log=False,
            threshold=1.0,
            linear=False,
        ):
            nonlocal markdown
            scales[filename] = [
                configure_value_scale(
                    axis,
                    positive_log=positive_log,
                    linthresh=threshold,
                    force_linear=linear,
                )
                for axis in fig.axes
            ]
            fig.suptitle(figure_title)
            fig.tight_layout(rect=(0, 0.02, 1, 0.94))
            png = figures / f"{filename}.png"
            fig.savefig(png, dpi=180)
            fig.savefig(figures / f"{filename}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            markdown += (
                f"\n\n## {figure_title}\n\n![{figure_title}](figures/{filename}.png)\n"
            )
            encoded = base64.b64encode(png.read_bytes()).decode()
            html_blocks.append(
                f"<h2>{figure_title}</h2><img alt='{figure_title}' "
                f"src='data:image/png;base64,{encoded}'>"
            )

        page = plt.figure(figsize=(8.27, 11.69))
        page.text(0.08, 0.94, title, fontsize=16)
        import textwrap

        page.text(
            0.08,
            0.87,
            "\n\n".join("\n".join(textwrap.wrap(note, 40)) for note in notes),
            fontsize=11,
            va="top",
            linespacing=1.5,
        )
        pdf.savefig(page)
        plt.close(page)

        def energy_plot(low, high, xscale, filename, figure_title):
            fig, axis = plt.subplots(figsize=(8, 5))
            for label, group in zip(labels, groups, strict=True):
                data = group["curves"]
                x = data["node_radius"]
                mask = (x >= low) & (x <= high)
                axis.plot(x[mask], data["node_conditional"][mask, 0], label=label)
            axis.set(
                xlim=(low, high),
                xscale=xscale,
                xlabel=f"两电子{distance_name}（磁长）",
                ylabel="局域总能量的加权方向平均",
            )
            axis.grid(alpha=0.25)
            axis.legend()
            save(fig, filename, figure_title)

        if not all("integrals" in group for group in groups):
            energy_plot(
                0.001, 2.0, "log", "conditional_energy", "给定距离时的局域总能量"
            )
        for low, high, suffix, xscale in PLOT_RANGES:
            energy_plot(
                low,
                high,
                xscale,
                f"conditional_energy_r_{suffix}",
                f"局域总能量：距离{low:g}至{high:g}",
            )
        plot_energy_components(groups, labels, distance_name, save)

        if all("integrals" in group for group in groups):
            plot_disk_distributions(groups, labels, save)
            plot_disk_details(groups, labels, save)
            plot_disk_comparisons(groups, labels, save)
    (output / "固定背景能量分析报告.md").write_text(markdown + "\n", encoding="utf-8")
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<title>固定背景能量分析报告</title><style>"
        "body{max-width:1000px;margin:40px auto;font:16px/1.8 sans-serif}"
        "img{width:100%}</style>"
    )
    (output / "固定背景能量分析报告.html").write_text(
        document + "".join(html_blocks) + "</html>", encoding="utf-8"
    )
    write_json(output / "plot_scales.json", scales)
    write_json(
        output / "source_manifest.json",
        dict(
            geometry=geometry,
            sources=[
                dict(
                    directory=str(g["directory"].resolve()),
                    run_manifest_sha256=sha256(g["directory"] / "run_manifest.json"),
                    validation_sha256=sha256(g["directory"] / "validation.json"),
                    curve_sha256=sha256(g["directory"] / "curves_refined.npz"),
                )
                for g in groups
            ],
            report_source_sha256=sha256(Path(__file__)),
            generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """Create separate geometry reports from one or more scan roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output
    if output is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        output = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / f"conditional_energy_{os.environ.get('SLURM_JOB_ID', 'local')}_{stamp}"
        )
    for geometry, groups in load_results(args.root).items():
        create_report(groups, output / geometry)
    print(f"Reports saved: {output}")


if __name__ == "__main__":
    main()
