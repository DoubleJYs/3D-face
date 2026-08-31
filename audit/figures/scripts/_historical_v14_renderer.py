#!/usr/bin/env python3
"""Historical frozen-display renderer dependency.

This file is not an active V15 entry point and is not a second manuscript
package.  It is retained only so the V15 public reconstruction entry can reuse
the exact plotting primitives that produced the frozen quantitative figures.
The supported V15 command is ``rebuild_public_figures.py``. The retained code
performs no model inference and no new hypothesis testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image


PREFIX = "W5B49N_V14_MATCHED_CONTROLS_20260821A"
PAIR_MEMBER = f"{PREFIX}/postprocess_v1_1_formal/PAIR_METRICS.jsonl"
FAMILY_MEMBER = f"{PREFIX}/statistics_v1/FAMILY_RESULTS.csv"
ANALYSIS_TERMINAL_MEMBER = f"{PREFIX}/statistics_v1/ANALYSIS_TERMINAL.json"
FINAL_RECEIPT_MEMBER = f"{PREFIX}/control/FINAL_COMPLETION_RECEIPT.json"
PROVENANCE_MEMBER = (
    f"{PREFIX}/provenance_validation_v1_2/PROVENANCE_VALIDATION_TERMINAL.json"
)
SEEDS = (2026080447, 2026080448, 2026080449, 2026080450, 2026080451)
SEEDED = {"full", "condition0", "b_lite_ft"}
COMPARATORS = (
    "condition0",
    "b_lite_ft",
    "freeuv_conserved",
    "b_lite",
    "lama",
    "zits",
)
DISPLAY = {
    "condition0": "NoCond",
    "b_lite_ft": "B-lite同任务微调\n（B-lite-FT）",
    "freeuv_conserved": "FreeUV\n（已观测纹理保持）",
    "freeuv_native": "FreeUV\n（原始输出）",
    "b_lite": "B-lite",
    "lama": "LaMa-UV",
    "zits": "ZITS-UV",
    "full": "FrugalFace3D-Lite",
}
DATASET_DISPLAY = {"D1": "FaceScape", "D2": "REALY"}

NAVY = "#17365D"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#6F5AA8"
GRAY = "#737B83"
LIGHT_GRAY = "#EEF1F3"
GRID = "#D6DCE1"
CANONICAL_ARCHIVE_NAME = "W5B49N_V14_MATCHED_CONTROLS_20260821A_FINAL_RESULTS_V1.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"],
            "axes.unicode_minus": False,
            "font.size": 8.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#7E8993",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def tar_text(archive: tarfile.TarFile, member: str) -> str:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    return handle.read().decode("utf-8")


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise RuntimeError("empty median")
    return float(statistics.median(materialized))


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["dataset_id"],
            row["pair_id"],
            row["identity_token"],
            row["method_id"],
            row["output_mode"],
            row.get("seed"),
        ): row
        for row in rows
    }


def identity_effects(
    rows: list[dict[str, Any]],
    indexed: dict[tuple[Any, ...], dict[str, Any]],
    dataset: str,
    comparator: str,
    field: str,
    higher_is_better: bool = False,
) -> list[float]:
    identities = sorted({row["identity_token"] for row in rows if row["dataset_id"] == dataset})
    result: list[float] = []
    for identity in identities:
        seed_effects: list[float] = []
        for seed in SEEDS:
            pair_effects: list[float] = []
            for full in rows:
                if (
                    full["dataset_id"] != dataset
                    or full["identity_token"] != identity
                    or full["method_id"] != "full"
                    or full.get("seed") != seed
                ):
                    continue
                comparator_seed = seed if comparator in SEEDED else None
                key = (
                    dataset,
                    full["pair_id"],
                    identity,
                    comparator,
                    "conserved",
                    comparator_seed,
                )
                other = indexed.get(key)
                if other is None or full.get(field) is None or other.get(field) is None:
                    continue
                if higher_is_better:
                    pair_effects.append(float(full[field]) - float(other[field]))
                else:
                    pair_effects.append(float(other[field]) - float(full[field]))
            if pair_effects:
                seed_effects.append(median(pair_effects))
        if seed_effects:
            result.append(median(seed_effects))
    return result


def absolute_identity_median(
    rows: list[dict[str, Any]], dataset: str, method: str, field: str
) -> float:
    identities = sorted({row["identity_token"] for row in rows if row["dataset_id"] == dataset})
    identity_values: list[float] = []
    for identity in identities:
        if method in SEEDED:
            seed_values: list[float] = []
            for seed in SEEDS:
                values = [
                    float(row[field])
                    for row in rows
                    if row["dataset_id"] == dataset
                    and row["identity_token"] == identity
                    and row["method_id"] == method
                    and row.get("seed") == seed
                    and row.get(field) is not None
                ]
                if values:
                    seed_values.append(median(values))
            if seed_values:
                identity_values.append(median(seed_values))
        else:
            values = [
                float(row[field])
                for row in rows
                if row["dataset_id"] == dataset
                and row["identity_token"] == identity
                and row["method_id"] == method
                and row.get(field) is not None
            ]
            if values:
                identity_values.append(median(values))
    return median(identity_values)


def endpoint_identity_effect(
    rows: list[dict[str, Any]],
    indexed: dict[tuple[Any, ...], dict[str, Any]],
    dataset: str,
    field: str,
    higher_is_better: bool,
) -> tuple[float, int]:
    identities = sorted({row["identity_token"] for row in rows if row["dataset_id"] == dataset})
    identity_values: list[float] = []
    for identity in identities:
        pair_values: list[float] = []
        native_rows = [
            row
            for row in rows
            if row["dataset_id"] == dataset
            and row["identity_token"] == identity
            and row["method_id"] == "freeuv_native"
            and row.get(field) is not None
        ]
        for native in native_rows:
            key = (
                dataset,
                native["pair_id"],
                identity,
                "freeuv_conserved",
                "conserved",
                None,
            )
            conserved = indexed.get(key)
            if conserved is None or conserved.get(field) is None:
                continue
            if higher_is_better:
                pair_values.append(float(conserved[field]) - float(native[field]))
            else:
                pair_values.append(float(native[field]) - float(conserved[field]))
        if pair_values:
            identity_values.append(median(pair_values))
    return median(identity_values), len(identity_values)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_figure(fig: plt.Figure, assets: Path, stem: str) -> list[Path]:
    assets.mkdir(parents=True, exist_ok=True)
    outputs = [
        assets / f"{stem}.png",
        assets / f"{stem}.tiff",
        assets / f"{stem}.pdf",
        assets / f"{stem}.svg",
    ]
    fig.savefig(outputs[0], dpi=450)
    fig.savefig(outputs[1], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outputs[2])
    fig.savefig(outputs[3])
    plt.close(fig)
    return outputs


def figure_protocol_compression(compression: list[dict[str, Any]], assets: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 4.75), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 2.55])
    ax0 = fig.add_subplot(grid[0, :])
    ax0.set_axis_off()
    canonical = FancyBboxPatch(
        (0.02, 0.12), 0.96, 0.72, boxstyle="round,pad=0.018", facecolor=LIGHT_GRAY,
        edgecolor=GRAY, linewidth=1.0, transform=ax0.transAxes
    )
    ax0.add_patch(canonical)
    ax0.add_patch(Rectangle((0.10, 0.28), 0.38, 0.38, facecolor=BLUE, alpha=0.86, transform=ax0.transAxes))
    ax0.add_patch(Rectangle((0.48, 0.28), 0.23, 0.38, facecolor=ORANGE, alpha=0.90, transform=ax0.transAxes))
    ax0.add_patch(Rectangle((0.71, 0.28), 0.17, 0.38, facecolor="#B8BEC4", transform=ax0.transAxes))
    ax0.text(0.29, 0.47, "O：源视图已观测且目标可见", ha="center", va="center", color="white", weight="bold", transform=ax0.transAxes)
    ax0.text(0.595, 0.47, "H：源视图未观测\n且目标视图可见", ha="center", va="center", fontsize=7.6, color="#4C3300", weight="bold", transform=ax0.transAxes)
    ax0.text(0.795, 0.47, "目标不可见", ha="center", va="center", color="#343A40", transform=ax0.transAxes)
    ax0.text(0.03, 0.90, "a  可见性区域分解", ha="left", va="bottom", weight="bold", color=NAVY, transform=ax0.transAxes)
    ax0.text(0.50, 0.03, r"$A=O\cup H=M_{canon}V_{tgt}$；$H=M_{canon}(1-V_{src})V_{tgt}$", ha="center", va="bottom", color=NAVY, transform=ax0.transAxes)

    method_order = list(COMPARATORS)
    for axis, dataset, panel in zip((fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])), ("D1", "D2"), ("b", "c")):
        subset = [row for row in compression if row["dataset_id"] == dataset]
        lookup = {row["comparator_id"]: row for row in subset}
        y = np.arange(len(method_order))
        axis.axvline(0.0, color="#4B5560", linewidth=0.9, zorder=0)
        for position, method in enumerate(method_order):
            row = lookup[method]
            h = float(row["h_effect"])
            a = float(row["a_effect"])
            axis.plot([h, a], [position, position], color="#AAB2B9", linewidth=1.6, zorder=1)
            axis.scatter(h, position, s=40, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
            axis.scatter(a, position, s=34, marker="D", color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
            axis.annotate(f"{h:+.4f}", (h, position), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color="#6A4500")
            axis.annotate(f"{a:+.4f}", (a, position), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=6.5, color=NAVY)
        axis.set_yticks(y, [DISPLAY[item] for item in method_order])
        axis.invert_yaxis()
        axis.set_xscale("symlog", linthresh=5e-4, linscale=0.7)
        axis.xaxis.set_major_locator(FixedLocator([-0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1]))
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: "0" if value == 0 else f"{value:.3g}".replace("-", "−")
            )
        )
        axis.tick_params(axis="x", labelsize=6.6)
        axis.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
        axis.set_xlabel("MAE 相对差异：比较方法 − FrugalFace3D-Lite\n（对称对数尺度）")
        ratio = float(subset[0]["median_h_over_a_support_percent"])
        axis.set_title(f"{panel}  {DATASET_DISPLAY[dataset]}：H/A 区域占比中位数 {ratio:.3f}%", loc="left", color=NAVY, weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, label="H 域 MAE 差异", markersize=6),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE, label="A 域 MAE 差异", markersize=5.5),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )
    return save_figure(fig, assets, "v14_protocol_compression_zh")


def figure_endpoint_protocol(endpoint: list[dict[str, Any]], assets: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.75), layout="constrained")
    specs = (
        ("hidden_uv_mae", "H-MAE", "两种输出在 H 域一致"),
        ("all_target_visible_uv_mae", "原始输出 − 保持输出", "A-MAE 差异"),
        ("lpips_alex_v0_1", "原始输出 − 保持输出", "LPIPS 差异"),
        ("sface_source_to_render_cosine", "保持输出 − 原始输出", "SFace 差异"),
    )
    colors = [BLUE, ORANGE]
    for axis, (metric, ylabel, subtitle) in zip(axes, specs):
        subset = [row for row in endpoint if row["metric_id"] == metric]
        subset.sort(key=lambda row: row["dataset_id"])
        values = [float(row["median_identity_effect"]) for row in subset]
        if metric == "hidden_uv_mae":
            axis.set_axis_off()
            axis.add_patch(
                FancyBboxPatch(
                    (0.06, 0.16), 0.88, 0.66, boxstyle="round,pad=0.025",
                    facecolor=LIGHT_GRAY, edgecolor="#AAB2B9", linewidth=0.9,
                    transform=axis.transAxes,
                )
            )
            axis.text(0.5, 0.67, "原始输出与已观测纹理保持\n在 H 区域逐值相同", ha="center", va="center", fontsize=7.6, weight="bold", color=NAVY, transform=axis.transAxes)
            axis.text(0.5, 0.48, "FaceScape  Δ=0.0000", ha="center", va="center", fontsize=7.2, transform=axis.transAxes)
            axis.text(0.5, 0.34, "REALY  Δ=0.0000", ha="center", va="center", fontsize=7.2, transform=axis.transAxes)
            axis.set_title("H-MAE 逐值一致", color=NAVY, weight="bold", fontsize=8.7)
        else:
            bars = axis.bar([0, 1], values, color=colors, width=0.62)
            axis.axhline(0.0, color="#46505A", linewidth=0.8)
            axis.set_xticks([0, 1], ["FaceScape", "REALY"])
            axis.set_ylabel(ylabel)
            axis.set_title(subtitle, color=NAVY, weight="bold", fontsize=8.7)
            axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)
            maximum = max(max(abs(value) for value in values), 1e-4)
            axis.set_ylim(0, max(values) + 0.17 * maximum)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.035 * maximum, f"{value:.4f}", ha="center", va="bottom", fontsize=6.8)
    fig.suptitle("同一次 FreeUV 前向：隐藏预测不变，已观测纹理保留改变整脸指标", color=NAVY, weight="bold", fontsize=10.5)
    fig.text(0.5, -0.01, "A-MAE 与 LPIPS 为“原始输出 − 已观测纹理保持”，SFace 为反向差异；两种输出的 H-MAE 逐值一致。", ha="center", fontsize=7.3, color="#4C5660")
    return save_figure(fig, assets, "v14_endpoint_protocol_zh")


def figure_forest(forest: list[dict[str, Any]], assets: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 4.65))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.82, bottom=0.25, wspace=0.16)
    panels = (
        ("hidden_uv_mae", "H-MAE 差异", "比较方法 −\nFrugalFace3D-Lite"),
        ("lpips_alex_v0_1", "LPIPS 差异", "比较方法 −\nFrugalFace3D-Lite"),
        ("sface_source_to_render_cosine", "SFace 差异", "FrugalFace3D-Lite −\n比较方法"),
    )
    comparator_order = ["condition0", "b_lite_ft", "freeuv_conserved"]
    dataset_order = ["D1", "D2"]
    labels = [f"{DATASET_DISPLAY[dataset]}  {DISPLAY[method]}" for method in comparator_order for dataset in dataset_order]
    for axis, (metric, title, xlabel) in zip(axes, panels):
        lookup = {(row["comparator_id"], row["dataset_id"]): row for row in forest if row["metric_id"] == metric}
        positions = np.arange(len(labels))
        for position, method_dataset in enumerate((item for method in comparator_order for item in ((method, "D1"), (method, "D2")))):
            row = lookup[method_dataset]
            effect = float(row["effect"])
            low, high = float(row["ci95_low"]), float(row["ci95_high"])
            status = row["status"]
            color = BLUE if status == "full_favorable" else VERMILLION if status == "full_unfavorable" else GRAY
            marker = "o" if row["dataset_id"] == "D1" else "s"
            axis.errorbar(effect, position, xerr=[[effect - low], [high - effect]], fmt=marker, color=color, ecolor=color, elinewidth=1.35, capsize=2.6, markersize=5.0, markeredgecolor="white", markeredgewidth=0.5)
        axis.axvline(0.0, color="#46505A", linewidth=0.85)
        axis.set_yticks(positions, labels if metric == "hidden_uv_mae" else [""] * len(labels))
        if metric != "hidden_uv_mae":
            axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        axis.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
        axis.set_title(title, color=NAVY, weight="bold")
        axis.set_xlabel(xlabel)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Holm 校正后差异为正", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=VERMILLION, label="Holm 校正后差异为负", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, label="Holm 校正后未区分", markersize=6),
        ],
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025)
    )
    fig.suptitle("五种子预设主要比较与 95% 身份自助区间", color=NAVY, weight="bold", fontsize=10.8)
    fig.text(0.5, 0.105, "置信区间未作多重校正；颜色由双侧精确符号检验的族内 Holm 结果决定。", ha="center", fontsize=7.2, color="#4C5660")
    return save_figure(fig, assets, "v14_confirmatory_forest_zh")


def figure_tradeoff(
    quality: list[dict[str, Any]], resource: list[dict[str, Any]], assets: Path
) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 4.15), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.22])
    methods = ["b_lite", "condition0", "full", "b_lite_ft"]
    colors = {"b_lite": GRAY, "condition0": ORANGE, "full": BLUE, "b_lite_ft": GREEN}
    offsets = {"b_lite": (3, 5), "condition0": (3, 7), "full": (3, -11), "b_lite_ft": (-42, 5)}
    for panel, dataset in enumerate(("D1", "D2")):
        axis = fig.add_subplot(grid[0, panel])
        lookup = {row["method_id"]: row for row in quality if row["dataset_id"] == dataset}
        for method in methods:
            row = lookup[method]
            x = float(row["trainable_parameters"]) / 1000.0
            y = float(row["absolute_h_mae"])
            axis.scatter(x, y, s=48, color=colors[method], edgecolor="white", linewidth=0.65, zorder=3)
            dx, dy = offsets[method]
            axis.annotate(DISPLAY[method], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.9, color="#303840")
        axis.set_xlim(-7, 135)
        axis.set_xlabel("本任务更新参数（千）")
        axis.set_ylabel("身份级绝对 H-MAE")
        axis.set_title(f"{'a' if panel == 0 else 'b'}  {DATASET_DISPLAY[dataset]}", loc="left", color=NAVY, weight="bold")
        axis.grid(color=GRID, linewidth=0.55, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)

    axis = fig.add_subplot(grid[0, 2])
    labels = [row["display_label"] for row in resource]
    values = [float(row["relative_delta_percent"]) for row in resource]
    bars = axis.barh(np.arange(len(labels)), values, color=[PURPLE, ORANGE, SKY, BLUE, NAVY])
    axis.axvline(0.0, color="#46505A", linewidth=0.8)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("相对固定权重 B-lite 的变化（%）")
    axis.set_title("c  完整流程接入开销", loc="left", color=NAVY, weight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%", va="center", fontsize=6.9)
    fig.suptitle("更新参数量、隐藏区域误差与完整流程资源开销", color=NAVY, weight="bold", fontsize=10.8)
    fig.text(0.5, -0.01, "FrugalFace3D-Lite 更新 89,386 个参数，比 B-lite同任务微调（B-lite-FT）少 26.8%；资源条仅表示接入增量，不表示速度优势。", ha="center", fontsize=7.2, color="#4C5660")
    return save_figure(fig, assets, "v14_parameter_quality_tradeoff_zh")


def figure_qualitative(root: Path, assets: Path) -> list[Path]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    columns = (
        ("input", "输入 / 区域"),
        ("b_lite", "B-lite"),
        ("b_lite_ft", "B-lite同任务微调\n（B-lite-FT）"),
        ("full", "FrugalFace3D-Lite"),
        ("freeuv_conserved", "FreeUV\n（已观测纹理保持）"),
        ("target", "目标参照"),
    )
    fig = plt.figure(figsize=(7.2, 8.35))
    grid = fig.add_gridspec(
        len(cases) * 2, len(columns), height_ratios=[2.35, 1.0] * len(cases),
        left=0.10, right=0.995, top=0.91, bottom=0.105, hspace=0.08, wspace=0.035,
    )
    for row_index, case in enumerate(cases):
        identity = case["identity"]
        display_identity = identity.replace("D2-", "R-")
        case_root = root / identity
        for column_index, (method, title) in enumerate(columns):
            upper = fig.add_subplot(grid[row_index * 2, column_index])
            lower = fig.add_subplot(grid[row_index * 2 + 1, column_index])
            if method == "input":
                upper_path = case_root / "input.png"
                lower_path = case_root / "region_map.png"
            elif method == "target":
                upper_path = case_root / "target.png"
                lower_path = case_root / "target_h_uv.png"
            else:
                upper_path = case_root / f"{method}_render.png"
                lower_path = case_root / f"{method}_h_error.png"
            upper.imshow(Image.open(upper_path).convert("RGB"))
            lower.imshow(Image.open(lower_path).convert("RGB"), interpolation="nearest")
            for axis in (upper, lower):
                axis.set_axis_off()
            if row_index == 0:
                upper.set_title(title, fontsize=8.1, weight="bold", color=NAVY, pad=3)
            if column_index == 0:
                upper.text(
                    -0.08, 0.5,
                    f"{display_identity}\nV04→V01\nH={case['support_texels']}",
                    transform=upper.transAxes, ha="right", va="center", fontsize=7.1, color="#303840"
                )
    fig.suptitle("REALY 目标可见隐藏区域同例比较（有效 texel ≥ 50）", y=0.975, color=NAVY, weight="bold", fontsize=10.8)
    # Reproduce the exact navy -> magenta -> yellow encoding used by
    # materialize_v14_high_support_visuals.py.  The shared bar exposes the
    # fixed normalization and clipping boundary used for every method/case.
    scaled = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    red = np.clip(1.8 * scaled, 0.0, 1.0)
    green = np.clip(2.0 * scaled - 0.8, 0.0, 1.0)
    blue = np.clip(0.35 + 1.2 * scaled - 1.4 * scaled**2, 0.0, 1.0)
    error_cmap = mpl.colors.ListedColormap(np.stack([red, green, blue], axis=1))
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=0.40), cmap=error_cmap
    )
    color_axis = fig.add_axes([0.26, 0.037, 0.53, 0.012])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.2, 0.4], labels=["0.00", "0.20", "0.40"])
    colorbar.ax.tick_params(labelsize=6.8, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(
        "H 域逐 texel 平均绝对 RGB 误差（>0.40 在上限饱和）",
        fontsize=6.8,
        labelpad=1.5,
        color="#4C5660",
    )
    fig.text(
        0.53,
        0.072,
        "上：共享目标几何下的目标视角渲染。下：H 区域图、固定尺度误差图或目标 H 纹理；灰色表示非 H 区域。",
        ha="center",
        fontsize=7.0,
        color="#4C5660",
    )
    return save_figure(fig, assets, "v14_high_support_real_faces_zh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-archive", type=Path, required=True)
    parser.add_argument("--efficiency-delta", type=Path, required=True)
    parser.add_argument("--qualitative-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--figure-data", type=Path, required=True)
    args = parser.parse_args()
    configure()
    if sha256_file(args.v14_archive) != "41d62c40ec3c7959c91eeb896da3a9895128734479413ab42377b2425312e665":
        raise RuntimeError("V14 final archive hash mismatch")

    with tarfile.open(args.v14_archive, "r:gz") as archive:
        receipt = json.loads(tar_text(archive, FINAL_RECEIPT_MEMBER))
        provenance = json.loads(tar_text(archive, PROVENANCE_MEMBER))
        terminal = json.loads(tar_text(archive, ANALYSIS_TERMINAL_MEMBER))
        if receipt.get("status") != "PASS_V14_EXPERIMENT_ANALYSIS_PROVENANCE_COMPLETE":
            raise RuntimeError("V14 completion receipt is not PASS")
        if provenance.get("status") != "PASS_V14_PROVENANCE_CHAIN_COMPLETE_V1_2":
            raise RuntimeError("V14 provenance is not PASS")
        if terminal.get("status") != "PASS_V14_MATCHED_CONTROLS_ANALYSIS_COMPLETE":
            raise RuntimeError("V14 analysis is not PASS")
        pair_rows = [json.loads(line) for line in tar_text(archive, PAIR_MEMBER).splitlines()]
        family_rows = list(csv.DictReader(io.StringIO(tar_text(archive, FAMILY_MEMBER))))
    indexed = index_rows(pair_rows)

    compression: list[dict[str, Any]] = []
    for dataset in ("D1", "D2"):
        pair_support = {}
        for row in pair_rows:
            if row["dataset_id"] == dataset:
                pair_support.setdefault(
                    row["pair_id"],
                    float(row["hidden_support_texels"]) / float(row["all_target_visible_support_texels"]),
                )
        ratio = median(pair_support.values()) * 100.0
        for comparator in COMPARATORS:
            h = median(identity_effects(pair_rows, indexed, dataset, comparator, "hidden_uv_mae"))
            a = median(identity_effects(pair_rows, indexed, dataset, comparator, "all_target_visible_uv_mae"))
            compression.append(
                {
                    "dataset_id": dataset,
                    "comparator_id": comparator,
                    "comparison_label": f"{DISPLAY[comparator].replace(chr(10), '')} 相对 FrugalFace3D-Lite",
                    "h_effect": h,
                    "a_effect": a,
                    "absolute_effect_retained_percent": abs(a / h) * 100.0 if h != 0 else "",
                    "median_h_over_a_support_percent": ratio,
                    "effect_direction": "comparator_minus_full_lower_is_better",
                    "analysis_role": "descriptive_prespecified_no_new_p_test",
                }
            )

    endpoint: list[dict[str, Any]] = []
    for dataset in ("D1", "D2"):
        for metric, field, higher in (
            ("hidden_uv_mae", "hidden_uv_mae", False),
            ("all_target_visible_uv_mae", "all_target_visible_uv_mae", False),
            ("lpips_alex_v0_1", "lpips_alex_v0_1", False),
            ("sface_source_to_render_cosine", "sface_source_to_render_cosine", True),
        ):
            value, identity_count = endpoint_identity_effect(pair_rows, indexed, dataset, field, higher)
            endpoint.append(
                {
                    "dataset_id": dataset,
                    "metric_id": metric,
                    "median_identity_effect": value,
                    "identity_count": identity_count,
                    "effect_definition": "conserved_minus_native" if higher else "native_minus_conserved",
                    "positive_means_conserved_favorable": True,
                }
            )

    forest: list[dict[str, Any]] = []
    for row in family_rows:
        if row["comparator_id"] not in {"condition0", "b_lite_ft", "freeuv_conserved"}:
            continue
        status = (
            "full_favorable" if row["confirmatory_full_favorable"] == "True"
            else "full_unfavorable" if row["confirmatory_full_unfavorable"] == "True"
            else "indeterminate"
        )
        forest.append(
            {
                "family_id": row["family_id"],
                "dataset_id": row["dataset_id"],
                "comparator_id": row["comparator_id"],
                "metric_id": row["metric_id"],
                "identity_count": int(row["identity_count"]),
                "effect": float(row["median_identity_effect"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "p_raw_two_sided": float(row["p_raw_two_sided_exact_sign"]),
                "p_holm": float(row["p_holm_within_family"]),
                "status": status,
                "ci_multiple_comparison_adjusted": False,
                "p_multiple_comparison": "holm_within_prespecified_family",
            }
        )

    trainable = {"b_lite": 0, "condition0": 89386, "full": 89386, "b_lite_ft": 122164}
    quality: list[dict[str, Any]] = []
    for dataset in ("D1", "D2"):
        for method in trainable:
            quality.append(
                {
                    "dataset_id": dataset,
                    "method_id": method,
                    "trainable_parameters": trainable[method],
                    "absolute_h_mae": absolute_identity_median(pair_rows, dataset, method, "hidden_uv_mae"),
                    "aggregation": "pair_median_then_seed_median_then_identity_median",
                }
            )

    with args.efficiency_delta.open(encoding="utf-8", newline="") as handle:
        resource_input = list(csv.DictReader(handle))
    resource_labels = {
        "inference_parameters": "推理参数",
        "macs": "MACs",
        "peak_memory_mib": "峰值显存",
        "p50_ms": "p50 时延",
        "p95_ms": "p95 时延",
    }
    resource = [
        {
            **row,
            "display_label": resource_labels[row["metric"]],
            "source_scope": "V13 frozen RTX4090 FP32 batch1 end-to-end",
        }
        for row in resource_input
    ]

    qualitative_manifest = json.loads((args.qualitative_root / "manifest.json").read_text(encoding="utf-8"))
    candidates = [
        {
            "identity": row["identity"],
            "pair_id": row["pair_id"],
            "source_view": row["source_view"],
            "target_view": row["target_view"],
            "hidden_support_texels": row["support_texels"],
            "selection_used_method_outputs": False,
            "selection_rule": qualitative_manifest["selection_rule"],
        }
        for row in qualitative_manifest["cases"]
    ]

    data_specs = (
        ("v14_region_compression.csv", compression, list(compression[0])),
        ("v14_endpoint_protocol_effects.csv", endpoint, list(endpoint[0])),
        ("v14_confirmatory_forest.csv", forest, list(forest[0])),
        ("v14_parameter_quality_tradeoff.csv", quality, list(quality[0])),
        ("v14_full_vs_blite_resource_delta.csv", resource, list(resource[0])),
        ("v14_high_support_candidates.csv", candidates, list(candidates[0])),
    )
    for filename, source_rows, fields in data_specs:
        write_csv(args.figure_data / filename, source_rows, fields)

    public_selection_manifest = {
        "schema_version": "frugalface3d.v14.high_support_selection.v1",
        "status": "PASS_PRESET_OUTPUT_BLIND_SELECTION",
        "dataset_id": "D2",
        "dataset_name": "REALY",
        "selection_threshold_hidden_support_texels": 50,
        "fixed_view_pair": "V04->V01",
        "eligible_identity_count": 59,
        "eligible_identity_count_by_prespecified_stratum": [15, 17, 14, 13],
        "selection_rule": qualitative_manifest["selection_rule"],
        "tie_rule": "nearest anonymous identity to each prespecified stratum midpoint, then lower identity",
        "method_outputs_used_for_selection": False,
        "publication_scope": "rights-cleared anonymous standardized derivatives",
        "source_binding_sha256": qualitative_manifest["upstream_sha256"]["source_tensors"],
        "cases": [
            {
                "identity": row["identity"],
                "pair_id": row["pair_id"],
                "source_view": row["source_view"],
                "target_view": row["target_view"],
                "hidden_support_texels": row["support_texels"],
                "hidden_uv_crop_box_xyxy": row["hidden_uv_crop_box_xyxy"],
                "seed": row["seed"],
                "derived_file_sha256_by_role": row["output_sha256"],
            }
            for row in qualitative_manifest["cases"]
        ],
    }
    selection_manifest_path = args.figure_data / "v14_high_support_selection_manifest.json"
    selection_temporary = selection_manifest_path.with_suffix(".json.tmp")
    selection_temporary.write_text(
        json.dumps(public_selection_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_temporary.replace(selection_manifest_path)

    outputs: list[Path] = []
    outputs.extend(figure_protocol_compression(compression, args.assets))
    outputs.extend(figure_endpoint_protocol(endpoint, args.assets))
    outputs.extend(figure_forest(forest, args.assets))
    outputs.extend(figure_tradeoff(quality, resource, args.assets))
    outputs.extend(figure_qualitative(args.qualitative_root, args.assets))

    sources = [args.figure_data / filename for filename, _, _ in data_specs]
    manifest = {
        "schema_version": "frugalface3d.v14.figure_manifest.v1",
        "status": "PASS_V14_FIGURES_REBUILT_FROM_FROZEN_DATA",
        "backend": "python_matplotlib_only",
        "v14_archive": {
            "filename": CANONICAL_ARCHIVE_NAME,
            "sha256": sha256_file(args.v14_archive),
            "completion_status": receipt["status"],
            "analysis_status": terminal["status"],
            "provenance_status": provenance["status"],
        },
        "statistical_boundary": {
            "confirmatory_tests": "frozen two-sided exact sign tests with Holm within four prespecified families",
            "confidence_intervals": "95% identity bootstrap, not multiplicity adjusted",
            "h_to_a": "prespecified descriptive decomposition; no new p-values",
            "qualitative_selection_used_method_outputs": False,
        },
        "source_sha256": {path.name: sha256_file(path) for path in sources},
        "asset_sha256": {path.name: sha256_file(path) for path in outputs},
        "qualitative_selection_manifest": {
            "filename": selection_manifest_path.name,
            "sha256": sha256_file(selection_manifest_path),
        },
    }
    manifest_path = args.figure_data / "v14_figure_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"status": manifest["status"], "figures": 5, "assets": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
