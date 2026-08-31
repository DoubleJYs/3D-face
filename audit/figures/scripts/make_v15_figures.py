#!/usr/bin/env python3
"""Build the V15 figure set without changing the frozen V14 evidence.

The frozen archive reader, aggregation code, and fixed qualitative selection
are inherited from the explicitly scoped historical renderer. V15 changes only
the publication display layer: the proposed model is shown first, reader-facing names use
standard paper terminology, and every generated source/asset receives a V15
name.  No training, inference, resampling, or hypothesis testing is performed.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

import _historical_v14_renderer as frozen


SOURCE_NAME_MAP = {
    "v14_region_compression.csv": "v15_visibility_region_compression.csv",
    "v14_endpoint_protocol_effects.csv": "v15_output_form_effects.csv",
    "v14_confirmatory_forest.csv": "v15_multimetric_effects.csv",
}

ASSET_STEM_MAP = {
    "v14_protocol_compression_zh": "v15_visibility_region_evaluation_zh",
    "v14_endpoint_protocol_zh": "v15_output_form_comparison_zh",
    "v14_confirmatory_forest_zh": "v15_multimetric_effects_zh",
}

ORIGINAL_SAVE_FIGURE = frozen.save_figure
ORIGINAL_REGION_FIGURE = frozen.figure_protocol_compression
ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def rounded_box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    color: str = "#25313B",
    fontsize: float = 8.0,
    weight: str = "normal",
) -> None:
    x, y = xy
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            transform=axis.transAxes,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        weight=weight,
        transform=axis.transAxes,
    )


def arrow(axis, start: tuple[float, float], end: tuple[float, float], color: str = "#718096") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color=color,
            transform=axis.transAxes,
        )
    )


def elbow_arrow(
    axis,
    points: tuple[tuple[float, float], ...],
    color: str = "#718096",
    linewidth: float = 1.05,
) -> None:
    """Draw an orthogonal connector without crossing intermediate modules."""
    if len(points) < 2:
        raise ValueError("elbow_arrow_requires_two_points")
    if len(points) > 2:
        axis.plot(
            [point[0] for point in points[:-1]],
            [point[1] for point in points[:-1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            transform=axis.transAxes,
            zorder=2,
        )
    axis.add_patch(
        FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=10,
            linewidth=linewidth, color=color, transform=axis.transAxes, zorder=3,
        )
    )


def figure_model_overview(assets: Path) -> list[Path]:
    """Render the complete frozen-backbone and trainable-residual data flow."""
    fig, axis = plt.subplots(figsize=(7.20, 4.55))
    axis.set_axis_off()

    fixed_face = "#E8EDF1"
    fixed_edge = "#8795A1"
    input_face = "#F5F7F9"
    train_face = "#E7F2F9"
    train_edge = frozen.BLUE
    condition_face = "#FFF2D8"
    condition_edge = "#D7A23A"
    output_face = "#E4F3EB"
    output_edge = "#4F9870"
    muted = "#52616F"

    axis.text(
        0.02,
        0.975,
        "FrugalFace3D-Lite 神经结构与纹理生成路径",
        ha="left",
        va="top",
        fontsize=10.8,
        weight="bold",
        color=frozen.NAVY,
        transform=axis.transAxes,
    )
    axis.text(
        0.02,
        0.925,
        "固定前端和纹理骨干提供表面对应、纹理先验与条件表征；可学习分支预测受掩码约束的有界残差。",
        ha="left",
        va="top",
        fontsize=7.15,
        color=muted,
        transform=axis.transAxes,
    )
    axis.text(0.765, 0.956, "参数固定", ha="left", va="center", fontsize=6.6, color=muted, transform=axis.transAxes)
    axis.add_patch(FancyBboxPatch((0.735, 0.943), 0.022, 0.026, boxstyle="round,pad=0.004", facecolor=fixed_face, edgecolor=fixed_edge, linewidth=0.8, transform=axis.transAxes))
    axis.text(0.895, 0.956, "可学习", ha="left", va="center", fontsize=6.6, color=frozen.NAVY, transform=axis.transAxes)
    axis.add_patch(FancyBboxPatch((0.865, 0.943), 0.022, 0.026, boxstyle="round,pad=0.004", facecolor=train_face, edgecolor=train_edge, linewidth=1.0, transform=axis.transAxes))

    # Frozen feature construction and texture prior.
    axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.655), 0.78, 0.225,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#FAFBFC", edgecolor=fixed_edge, linewidth=1.05,
            linestyle=(0, (4, 2)), transform=axis.transAxes,
        )
    )
    axis.text(0.035, 0.855, "固定特征构建与纹理先验", ha="left", va="center", fontsize=7.25, weight="bold", color=muted, transform=axis.transAxes)
    rounded_box(axis, (0.04, 0.705), 0.105, 0.105, "单幅源图像\n$I_s$", face=input_face, edge=fixed_edge, fontsize=7.4, weight="bold")
    rounded_box(axis, (0.195, 0.695), 0.165, 0.125, "SMIRK / FLAME\n投影与 UV 采样", face=fixed_face, edge=fixed_edge, fontsize=7.1, weight="bold")
    rounded_box(axis, (0.405, 0.695), 0.145, 0.125, "B-lite\n纹理骨干", face=fixed_face, edge=fixed_edge, fontsize=7.25, weight="bold")
    rounded_box(axis, (0.615, 0.695), 0.145, 0.125, "表情辅助编码器", face=fixed_face, edge=fixed_edge, fontsize=7.05, weight="bold")
    arrow(axis, (0.145, 0.758), (0.195, 0.758), fixed_edge)
    elbow_arrow(
        axis,
        ((0.145, 0.790), (0.170, 0.790), (0.170, 0.835), (0.590, 0.835), (0.590, 0.758), (0.615, 0.758)),
        fixed_edge,
    )
    arrow(axis, (0.360, 0.758), (0.405, 0.758), fixed_edge)
    axis.text(0.278, 0.672, r"$T_{obs},\ V_s,\ M_{canon}$" "\n" r"$F_g=[xyz,n]$", ha="center", va="center", fontsize=6.35, color=condition_edge, transform=axis.transAxes)
    axis.text(0.477, 0.672, "$T_{base}$；$F_{tex}$（$16\\times16\\times160$）", ha="center", va="center", fontsize=6.35, color=muted, transform=axis.transAxes)
    axis.text(0.687, 0.672, "$q_e$（128 维）", ha="center", va="center", fontsize=6.35, color=condition_edge, transform=axis.transAxes)
    rounded_box(axis, (0.825, 0.705), 0.145, 0.105, "完整纹理先验\n$T_{base}$", face=output_face, edge=output_edge, fontsize=7.15, weight="bold")
    elbow_arrow(
        axis,
        ((0.550, 0.790), (0.575, 0.790), (0.575, 0.845), (0.805, 0.845), (0.805, 0.758), (0.825, 0.758)),
        output_edge,
    )

    # Trainable residual branch.
    axis.add_patch(
        FancyBboxPatch(
            (0.13, 0.255), 0.67, 0.335,
            boxstyle="round,pad=0.014,rounding_size=0.020",
            facecolor="#F5FAFD", edgecolor=train_edge, linewidth=1.55,
            transform=axis.transAxes,
        )
    )
    axis.text(0.15, 0.565, "可学习残差分支  $F_{\\phi}$", ha="left", va="center", fontsize=7.75, weight="bold", color=frozen.NAVY, transform=axis.transAxes)
    axis.text(0.775, 0.565, "89,386 个更新参数", ha="right", va="center", fontsize=6.85, weight="bold", color=train_edge, transform=axis.transAxes)

    # Condition inputs are shown as inputs, without implying validated causal gain.
    rounded_box(axis, (0.155, 0.485), 0.115, 0.055, "$F_{tex}$", face=condition_face, edge=condition_edge, fontsize=6.6, weight="bold")
    rounded_box(axis, (0.310, 0.485), 0.135, 0.055, r"$F_g,\ V_s,\ M_{canon}$", face=condition_face, edge=condition_edge, fontsize=6.35, weight="bold")
    rounded_box(axis, (0.525, 0.485), 0.115, 0.055, "$q_e$", face=condition_face, edge=condition_edge, fontsize=6.6, weight="bold")
    arrow(axis, (0.477, 0.695), (0.212, 0.540), fixed_edge)
    arrow(axis, (0.278, 0.655), (0.378, 0.540), condition_edge)
    arrow(axis, (0.687, 0.655), (0.582, 0.540), condition_edge)

    nodes = (
        (0.155, "纹理投影\n$160\\rightarrow64$"),
        (0.265, "结构门"),
        (0.375, "局部交叉注意"),
        (0.500, "表情调制"),
        (0.625, "纹理上下文"),
        (0.710, "两级上采样"),
    )
    widths = (0.090, 0.090, 0.105, 0.105, 0.072, 0.072)
    for (x, label), width in zip(nodes, widths):
        rounded_box(axis, (x, 0.350), width, 0.095, label, face=train_face, edge=train_edge, fontsize=6.1, weight="bold")
    for (x0, _), width0, (x1, _) in zip(nodes[:-1], widths[:-1], nodes[1:]):
        arrow(axis, (x0 + width0, 0.397), (x1, 0.397), train_edge)
    arrow(axis, (0.212, 0.485), (0.200, 0.445), condition_edge)
    arrow(axis, (0.378, 0.485), (0.310, 0.445), condition_edge)
    arrow(axis, (0.582, 0.485), (0.552, 0.445), condition_edge)

    axis.text(
        0.465, 0.302,
        "$[R_{\\phi},u_{\\phi}]$ ； $\\Delta T_{\\phi}=0.15\\tanh(R_{\\phi})$ ； $z_{\\phi}=\\operatorname{clip}(u_{\\phi},-6,3)$",
        ha="center", va="center", fontsize=6.25, color=frozen.NAVY, transform=axis.transAxes,
    )

    # Candidate texture and exact observed-texture preservation.
    rounded_box(axis, (0.835, 0.475), 0.135, 0.095, "候选纹理\n$T_{pred}$", face=output_face, edge=output_edge, fontsize=7.1, weight="bold")
    rounded_box(axis, (0.835, 0.325), 0.135, 0.095, "可见性组合\n$V_s / (1-V_s)$", face="#DDECF6", edge=train_edge, fontsize=6.8, weight="bold")
    rounded_box(axis, (0.835, 0.175), 0.135, 0.095, "完整 UV 纹理\n$T_{out}$", face=output_face, edge=output_edge, fontsize=7.1, weight="bold")
    arrow(axis, (0.782, 0.397), (0.835, 0.522), train_edge)
    arrow(axis, (0.898, 0.705), (0.902, 0.570), output_edge)
    arrow(axis, (0.902, 0.475), (0.902, 0.420), output_edge)
    arrow(axis, (0.902, 0.325), (0.902, 0.270), output_edge)
    elbow_arrow(
        axis,
        ((0.360, 0.710), (0.385, 0.710), (0.385, 0.625), (0.815, 0.625), (0.815, 0.372), (0.835, 0.372)),
        fixed_edge,
    )
    axis.text(0.805, 0.607, "$T_{obs},V_s$", ha="right", va="center", fontsize=5.9, color=muted, transform=axis.transAxes)

    axis.text(
        0.49, 0.115,
        "$T_{pred}=\\operatorname{clip}[T_{base}+(1-V_s)\\odot\\Delta T_{\\phi},0,1]$",
        ha="center", va="center", fontsize=6.6, color=muted, transform=axis.transAxes,
    )
    axis.text(
        0.49, 0.066,
        "$M_{canon}\\odot T_{out}=V_s\\odot T_{obs}+(M_{canon}-V_s)\\odot T_{pred}$",
        ha="center", va="center", fontsize=6.6, color=frozen.NAVY, transform=axis.transAxes,
    )
    return ORIGINAL_SAVE_FIGURE(fig, assets, "v15_model_architecture_zh")


def mapped_save_figure(fig: plt.Figure, assets: Path, stem: str) -> list[Path]:
    return ORIGINAL_SAVE_FIGURE(fig, assets, ASSET_STEM_MAP.get(stem, stem.replace("v14_", "v15_", 1)))


def figure_visibility_evaluation(rows: list[dict[str, object]], assets: Path) -> list[Path]:
    """Render the reader-facing visibility figure with manuscript notation."""
    fig = plt.figure(figsize=(7.15, 4.75), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 2.55])
    top = fig.add_subplot(grid[0, :])
    top.set_axis_off()
    top.add_patch(
        frozen.FancyBboxPatch(
            (0.02, 0.12), 0.96, 0.72, boxstyle="round,pad=0.018",
            facecolor=frozen.LIGHT_GRAY, edgecolor=frozen.GRAY, linewidth=1.0,
            transform=top.transAxes,
        )
    )
    top.add_patch(frozen.Rectangle((0.10, 0.28), 0.38, 0.38, facecolor=frozen.BLUE, alpha=0.86, transform=top.transAxes))
    top.add_patch(frozen.Rectangle((0.48, 0.28), 0.23, 0.38, facecolor=frozen.ORANGE, alpha=0.90, transform=top.transAxes))
    top.add_patch(frozen.Rectangle((0.71, 0.28), 0.17, 0.38, facecolor="#B8BEC4", transform=top.transAxes))
    top.text(0.29, 0.47, "O：源视图已观测且目标可见", ha="center", va="center", color="white", weight="bold", transform=top.transAxes)
    top.text(0.595, 0.47, "H：源视图未观测\n且目标视图可见", ha="center", va="center", fontsize=7.6, color="#4C3300", weight="bold", transform=top.transAxes)
    top.text(0.795, 0.47, "目标不可见", ha="center", va="center", color="#343A40", transform=top.transAxes)
    top.text(0.03, 0.90, "a  可见性区域分解", ha="left", va="bottom", weight="bold", color=frozen.NAVY, transform=top.transAxes)
    top.text(0.50, 0.03, r"$A=O\cup H=M_{canon}V_t$；$H=M_{canon}(1-V_s)V_t$", ha="center", va="bottom", color=frozen.NAVY, transform=top.transAxes)

    method_order = list(frozen.COMPARATORS)
    for axis, dataset, panel in zip(
        (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])),
        ("D1", "D2"),
        ("b", "c"),
    ):
        subset = [row for row in rows if row["dataset_id"] == dataset]
        lookup = {row["comparator_id"]: row for row in subset}
        positions = np.arange(len(method_order))
        axis.axvline(0.0, color="#4B5560", linewidth=0.9, zorder=0)
        for position, method in enumerate(method_order):
            row = lookup[method]
            hidden_effect = float(row["h_effect"])
            visible_effect = float(row["a_effect"])
            axis.plot([hidden_effect, visible_effect], [position, position], color="#AAB2B9", linewidth=1.6, zorder=1)
            axis.scatter(hidden_effect, position, s=40, color=frozen.ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
            axis.scatter(visible_effect, position, s=34, marker="D", color=frozen.BLUE, edgecolor="white", linewidth=0.6, zorder=3)
            hidden_label = f"{hidden_effect:+.5f}" if 0.0 < abs(hidden_effect) < 5e-5 else f"{hidden_effect:+.4f}"
            visible_label = f"{visible_effect:+.5f}" if 0.0 < abs(visible_effect) < 5e-5 else f"{visible_effect:+.4f}"
            axis.annotate(hidden_label, (hidden_effect, position), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color="#6A4500")
            axis.annotate(visible_label, (visible_effect, position), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=6.5, color=frozen.NAVY)
        axis.set_yticks(positions, [frozen.DISPLAY[item] for item in method_order])
        axis.invert_yaxis()
        axis.set_xscale("symlog", linthresh=5e-4, linscale=0.7)
        # Keep the largest ZITS effect fully inside the plotting area. The
        # limits are display-only and do not alter any frozen values.
        axis.set_xlim(-0.15, 0.25)
        axis.xaxis.set_major_locator(frozen.FixedLocator([-0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1]))
        axis.xaxis.set_major_formatter(frozen.FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.3g}".replace("-", "−")))
        axis.tick_params(axis="x", labelsize=6.6)
        axis.grid(axis="x", color=frozen.GRID, linewidth=0.6, alpha=0.8)
        axis.set_xlabel("MAE 相对差异：比较方法 − FrugalFace3D-Lite\n（对称对数尺度）")
        ratio = float(subset[0]["median_h_over_a_support_percent"])
        axis.set_title(
            f"{panel}  {frozen.DATASET_DISPLAY[dataset]}：|H|/|A| 区域占比中位数 {ratio:.3f}%",
            loc="left", color=frozen.NAVY, weight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            frozen.Line2D([0], [0], marker="o", color="none", markerfacecolor=frozen.ORANGE, label="H 域 MAE 差异", markersize=6),
            frozen.Line2D([0], [0], marker="D", color="none", markerfacecolor=frozen.BLUE, label="A 域 MAE 差异", markersize=5.5),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    return ORIGINAL_SAVE_FIGURE(fig, assets, "v15_visibility_region_evaluation_zh")


def region_figure_with_model(rows: list[dict[str, object]], assets: Path) -> list[Path]:
    outputs = figure_model_overview(assets)
    outputs.extend(figure_visibility_evaluation(rows, assets))
    return outputs


def figure_multimetric_effects(rows: list[dict[str, object]], assets: Path) -> list[Path]:
    """Render confirmatory effects without duplicating methods text in the panel footer."""
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 4.65))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.82, bottom=0.20, wspace=0.16)
    panels = (
        ("hidden_uv_mae", "H-MAE 差异", "比较方法 −\nFrugalFace3D-Lite"),
        ("lpips_alex_v0_1", "LPIPS 差异", "比较方法 −\nFrugalFace3D-Lite"),
        ("sface_source_to_render_cosine", "SFace 差异", "FrugalFace3D-Lite −\n比较方法"),
    )
    comparator_order = ["condition0", "b_lite_ft", "freeuv_conserved"]
    dataset_order = ["D1", "D2"]
    labels = [
        f"{frozen.DATASET_DISPLAY[dataset]}  {frozen.DISPLAY[method]}"
        for method in comparator_order
        for dataset in dataset_order
    ]
    for axis, (metric, title, xlabel) in zip(axes, panels):
        lookup = {
            (row["comparator_id"], row["dataset_id"]): row
            for row in rows
            if row["metric_id"] == metric
        }
        positions = np.arange(len(labels))
        pairs = (item for method in comparator_order for item in ((method, "D1"), (method, "D2")))
        for position, method_dataset in enumerate(pairs):
            row = lookup[method_dataset]
            effect = float(row["effect"])
            low, high = float(row["ci95_low"]), float(row["ci95_high"])
            status = row["status"]
            color = frozen.BLUE if status == "full_favorable" else frozen.VERMILLION if status == "full_unfavorable" else frozen.GRAY
            marker = "o" if row["dataset_id"] == "D1" else "s"
            axis.errorbar(
                effect, position, xerr=[[effect - low], [high - effect]], fmt=marker,
                color=color, ecolor=color, elinewidth=1.35, capsize=2.6,
                markersize=5.0, markeredgecolor="white", markeredgewidth=0.5,
            )
        axis.axvline(0.0, color="#46505A", linewidth=0.85)
        axis.set_yticks(positions, labels if metric == "hidden_uv_mae" else [""] * len(labels))
        if metric != "hidden_uv_mae":
            axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        axis.grid(axis="x", color=frozen.GRID, linewidth=0.55, alpha=0.75)
        axis.set_title(title, color=frozen.NAVY, weight="bold")
        axis.set_xlabel(xlabel)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            frozen.Line2D([0], [0], marker="o", color="none", markerfacecolor=frozen.BLUE, label="Holm 校正后差异为正", markersize=6),
            frozen.Line2D([0], [0], marker="o", color="none", markerfacecolor=frozen.VERMILLION, label="Holm 校正后差异为负", markersize=6),
            frozen.Line2D([0], [0], marker="o", color="none", markerfacecolor=frozen.GRAY, label="Holm 校正后未区分", markersize=6),
        ],
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025),
    )
    fig.suptitle("五种子预设主要比较与 95% 身份自助区间", color=frozen.NAVY, weight="bold", fontsize=10.8)
    return ORIGINAL_SAVE_FIGURE(fig, assets, "v15_multimetric_effects_zh")


def reuse_frozen_qualitative_plate(_root: Path, assets: Path) -> list[Path]:
    """Recompose the frozen image payload with V15 reader-facing labels."""
    source_svg = ROOT / "assets" / "v14_high_support_real_faces_zh.svg"
    if not source_svg.is_file():
        raise FileNotFoundError(source_svg)
    embedded: list[Image.Image] = []
    for element in ET.parse(source_svg).getroot().iter():
        if not element.tag.endswith("image"):
            continue
        href = element.attrib.get("{http://www.w3.org/1999/xlink}href") or element.attrib.get("href", "")
        if not href.startswith("data:image/png;base64,"):
            continue
        if float(element.attrib.get("height", "0")) <= 20.0:
            continue
        transform = element.attrib.get("transform", "")
        if "scale(1 -1)" not in transform:
            raise RuntimeError(f"unexpected_frozen_panel_transform:{transform}")
        payload = base64.b64decode(href.split(",", 1)[1])
        # Matplotlib stores raster payloads upside down in SVG and restores
        # their display orientation with the SVG transform above. Once the
        # payload is decoded outside the SVG, apply the equivalent vertical
        # flip before composing the V15 plate.
        embedded.append(ImageOps.flip(Image.open(io.BytesIO(payload)).convert("RGB")))
    if len(embedded) != 48:
        raise RuntimeError(f"expected_48_frozen_panel_images:{len(embedded)}")

    selection = json.loads(
        (ROOT / "figure_source_data" / "v15_high_support_selection_manifest.json").read_text(encoding="utf-8")
    )
    columns = (
        "输入 / 区域",
        "固定权重 B-lite",
        "B-lite 同任务微调",
        "FrugalFace3D-Lite",
        "FreeUV\n（已观测纹理保持输出）",
        "目标参照",
    )
    fig = plt.figure(figsize=(7.2, 8.35))
    grid = fig.add_gridspec(
        8, 6, height_ratios=[2.35, 1.0] * 4,
        left=0.10, right=0.995, top=0.91, bottom=0.105,
        hspace=0.08, wspace=0.035,
    )
    cursor = 0
    for row_index, case in enumerate(selection["cases"]):
        for column_index, title in enumerate(columns):
            upper = fig.add_subplot(grid[row_index * 2, column_index])
            lower = fig.add_subplot(grid[row_index * 2 + 1, column_index])
            upper.imshow(embedded[cursor])
            lower.imshow(embedded[cursor + 1], interpolation="nearest")
            cursor += 2
            upper.set_axis_off()
            lower.set_axis_off()
            if row_index == 0:
                upper.set_title(title, fontsize=7.7, weight="bold", color=frozen.NAVY, pad=3)
            if column_index == 0:
                identity = str(case["identity"]).split("-")[-1]
                upper.text(
                    -0.08, 0.5,
                    f"R-{identity}\nV04→V01\nH={case['hidden_support_texels']}",
                    transform=upper.transAxes, ha="right", va="center",
                    fontsize=7.1, color="#303840",
                )
    fig.suptitle(
        "REALY 目标可见隐藏区域跨视角同例比较（有效 texel ≥ 50）",
        y=0.975, color=frozen.NAVY, weight="bold", fontsize=10.3,
    )
    scaled = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    red = np.clip(1.8 * scaled, 0.0, 1.0)
    green = np.clip(2.0 * scaled - 0.8, 0.0, 1.0)
    blue = np.clip(0.35 + 1.2 * scaled - 1.4 * scaled**2, 0.0, 1.0)
    error_cmap = frozen.mpl.colors.ListedColormap(np.stack([red, green, blue], axis=1))
    scalar = frozen.mpl.cm.ScalarMappable(
        norm=frozen.mpl.colors.Normalize(vmin=0.0, vmax=0.40), cmap=error_cmap,
    )
    color_axis = fig.add_axes([0.26, 0.037, 0.53, 0.012])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.2, 0.4], labels=["0.00", "0.20", "0.40"])
    colorbar.ax.tick_params(labelsize=6.8, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(
        "H 域逐 texel 平均 RGB 绝对误差（超过 0.40 时按上限显示）",
        fontsize=6.8, labelpad=1.5, color="#4C5660",
    )
    fig.text(
        0.53, 0.072,
        "上排：共享目标几何下的目标视图渲染；下排：H 区域、统一色标误差图或目标 H 纹理，灰色表示非 H 区域。",
        ha="center", fontsize=6.7, color="#4C5660",
    )
    return ORIGINAL_SAVE_FIGURE(fig, assets, "v15_high_support_real_faces_zh")


def qualitative_proxy(root: Path) -> Path:
    """Create a value-identical manifest proxy from the frozen public selection."""
    source = ROOT / "figure_source_data" / "v14_high_support_selection_manifest.json"
    selection = json.loads(source.read_text(encoding="utf-8"))
    cases = []
    for case in selection["cases"]:
        cases.append(
            {
                "identity": case["identity"],
                "pair_id": case["pair_id"],
                "source_view": case["source_view"],
                "target_view": case["target_view"],
                "support_texels": case["hidden_support_texels"],
                "hidden_uv_crop_box_xyxy": case["hidden_uv_crop_box_xyxy"],
                "seed": case["seed"],
                "output_sha256": case["derived_file_sha256_by_role"],
            }
        )
    proxy = {
        "schema_version": "frugalface3d.v15.high_support_visuals.proxy.v1",
        "selection_rule": selection["selection_rule"],
        "upstream_sha256": {"source_tensors": selection["source_binding_sha256"]},
        "cases": cases,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_text(root / "manifest.json", json.dumps(proxy, ensure_ascii=False, indent=2) + "\n")
    return root


def transform_outputs(temporary_root: Path, figure_data: Path) -> dict[str, object]:
    figure_data.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for old_name, new_name in SOURCE_NAME_MAP.items():
        payload = (temporary_root / old_name).read_bytes()
        output = figure_data / new_name
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
        source_hashes[new_name] = sha256_file(output)

    selection = json.loads((temporary_root / "v14_high_support_selection_manifest.json").read_text(encoding="utf-8"))
    selection["schema_version"] = "frugalface3d.v15.high_support_selection.v1"
    selection["status"] = "PASS_PRESET_OUTPUT_BLIND_SELECTION_V15_DISPLAY"
    selection_path = figure_data / "v15_high_support_selection_manifest.json"
    atomic_text(selection_path, json.dumps(selection, ensure_ascii=False, indent=2) + "\n")

    manifest = json.loads((temporary_root / "v14_figure_manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "frugalface3d.v15.figure_manifest.v1"
    manifest["status"] = "PASS_V15_FIGURES_REBUILT_FROM_FROZEN_V14_DATA"
    manifest["figure_count"] = 6
    manifest["display_scope"] = (
        "model-first V15 publication display; frozen values and qualitative selection unchanged"
    )
    manifest["source_sha256"] = source_hashes
    manifest["qualitative_selection_manifest"] = {
        "filename": selection_path.name,
        "sha256": sha256_file(selection_path),
    }
    manifest_path = figure_data / "v15_figure_manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    raise SystemExit(
        "This dependency is not a public entry point; use figures/scripts/rebuild_public_figures.py."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-archive", type=Path, required=True)
    parser.add_argument("--efficiency-delta", type=Path, required=True)
    parser.add_argument(
        "--qualitative-root",
        type=Path,
        help="optional per-case V14 derivatives; the frozen combined plate is reused when omitted",
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--figure-data", type=Path, required=True)
    args = parser.parse_args()

    # Keep archived method IDs unchanged while exposing only the manuscript's
    # reader-facing name in V15 figures and derived display tables.
    frozen.DISPLAY["b_lite_ft"] = "B-lite 同任务微调"
    frozen.DISPLAY["b_lite"] = "固定权重 B-lite"
    frozen.DISPLAY["freeuv_conserved"] = "FreeUV\n（已观测纹理保持输出）"
    frozen.save_figure = mapped_save_figure
    frozen.figure_protocol_compression = region_figure_with_model
    frozen.figure_forest = figure_multimetric_effects

    previous_argv = sys.argv[:]
    try:
        with tempfile.TemporaryDirectory(prefix="frugalface3d-v15-figures-") as temporary_name:
            temporary_root = Path(temporary_name)
            qualitative_root = args.qualitative_root
            if qualitative_root is None or not (qualitative_root / "manifest.json").is_file():
                qualitative_root = qualitative_proxy(temporary_root / "qualitative_proxy")
                frozen.figure_qualitative = reuse_frozen_qualitative_plate
            sys.argv = [
                str(Path(frozen.__file__).resolve()),
                "--v14-archive",
                str(args.v14_archive),
                "--efficiency-delta",
                str(args.efficiency_delta),
                "--qualitative-root",
                str(qualitative_root),
                "--assets",
                str(args.assets),
                "--figure-data",
                str(temporary_root),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                frozen.main()
            manifest = transform_outputs(temporary_root, args.figure_data)
    finally:
        sys.argv = previous_argv

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "figures": manifest["figure_count"],
                "assets": len(manifest["asset_sha256"]),
                "frozen_values_changed": False,
                "qualitative_cases_changed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
