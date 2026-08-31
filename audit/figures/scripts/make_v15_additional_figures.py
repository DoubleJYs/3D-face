#!/usr/bin/env python3
"""Rebuild V15 Figure 5 and Supplementary Figures S2 and S3.

The portable renderer reads the package's public CSV files and performs no
training, model inference, resampling, or hypothesis testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "figures" / "rebuilt_public_outputs"
SOURCE = ROOT / "figures" / "source_data"
QA = ROOT / "validation"
ANALYSIS = ROOT / "robustness"
SUPPLEMENT = ROOT / "protocol" / "REALY_DIRECTIONAL_EXPLORATORY_ANALYSIS_PLAN.md"
CONTRACT = ROOT / "provenance" / "REALY_DIRECTIONAL_ANALYSIS_BINDING_RECEIPT.json"
EFFICIENCY = SOURCE / "v15_quality_resource_bubbles.csv"

PANORAMA_SOURCE = SOURCE / "v15_descriptive_multimetric_panorama.csv"
BUBBLE_SOURCE = SOURCE / "v15_quality_resource_bubbles.csv"
DIRECTION_SOURCE = SOURCE / "v15_realy_12direction_effects.csv"
MANIFEST = SOURCE / "v15_additional_figure_manifest.json"
CHECKSUMS = SOURCE / "v15_additional_output_sha256.txt"
QA_JSON = QA / "V15_ADDITIONAL_FIGURE_QA.json"
QA_MD = QA / "V15_ADDITIONAL_FIGURE_QA.md"

PANORAMA_STEM = ASSETS / "v15_descriptive_multimetric_panorama_zh"
BUBBLE_STEM = ASSETS / "v15_quality_resource_bubbles_zh"
DIRECTION_STEM = ASSETS / "v15_realy_12direction_effects_zh"

DATASET_ORDER = ("D1", "D2")
DATASET_LABEL = {"D1": "FaceScape", "D2": "REALY"}
METRIC_ORDER = ("h_mae", "a_mae", "lpips", "sface")
METRIC_LABEL = {
    "h_mae": "H-MAE\n↓ 越低越优",
    "a_mae": "A-MAE\n↓ 越低越优",
    "lpips": "LPIPS\n↓ 越低越优",
    "sface": "SFace\n↑ 越高越优",
}
METHOD_ORDER = (
    "full",
    "condition0",
    "b_lite_ft",
    "b_lite",
    "lama",
    "zits",
    "freeuv_conserved",
)
METHOD_DISPLAY = {
    "full": "FrugalFace3D-Lite",
    "condition0": "无显式条件残差（NoCond）",
    "b_lite_ft": "B-lite 同任务微调",
    "b_lite": "固定权重 B-lite",
    "lama": "LaMa-UV",
    "zits": "ZITS-UV",
    "freeuv_conserved": "FreeUV（已观测纹理保持输出）",
}
METHOD_CATEGORY = {
    "full": "本文模型",
    "condition0": "条件消融对照",
    "b_lite_ft": "同任务适配对照",
    "b_lite": "固定权重基线",
    "lama": "通用修复参照",
    "zits": "通用修复参照",
    "freeuv_conserved": "统一 UV 输入下的人脸 UV 参照",
}

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
INK = "#25313B"
MUTED = "#5A6874"

METHOD_COLOR = {
    "full": BLUE,
    "condition0": ORANGE,
    "b_lite_ft": GREEN,
    "b_lite": GRAY,
    "lama": PURPLE,
    "zits": VERMILLION,
    "freeuv_conserved": NAVY,
}
METHOD_MARKER = {
    "full": "o",
    "b_lite": "s",
    "b_lite_ft": "D",
    "lama": "^",
    "zits": "v",
    "freeuv_conserved": "P",
}

S13_METHOD_MAP = {
    "FrugalFace3D-Lite": "full",
    "无显式条件残差（NoCond）": "condition0",
    "B-lite 同任务微调": "b_lite_ft",
    "固定权重 B-lite": "b_lite",
    "LaMa-UV": "lama",
    "ZITS-UV": "zits",
    "FreeUV（已观测纹理保持输出，统一 UV 输入）": "freeuv_conserved",
}

RESOURCE_METHOD_MAP = {
    "full": "full",
    "b_lite": "b_lite",
    "b_lite_ft": "b_lite",
    "lama": "lama_big",
    "zits": "zits",
    "freeuv_conserved": "freeuv_raw",
}
BUBBLE_METHODS = (
    "full",
    "b_lite",
    "b_lite_ft",
    "lama",
    "zits",
    "freeuv_conserved",
)
TASK_UPDATES = {
    "full": 89386,
    "b_lite": 0,
    "b_lite_ft": 122164,
    "lama": 0,
    "zits": 0,
    "freeuv_conserved": 0,
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "PingFang SC",
                "Hiragino Sans GB",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 7.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
            "axes.edgecolor": "#7E8993",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "frugalface3d-v15-additional",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_number(text: str) -> float:
    return float(text.replace("**", "").strip())


def parse_s13() -> list[dict[str, Any]]:
    """Parse the 14 published rows in Chinese Table S13 into long form."""
    lines = SUPPLEMENT.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("**表 S13  ")
    )
    table_rows: list[list[str]] = []
    for line in lines[start + 3 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and not cells[0].startswith("---"):
            table_rows.append(cells)
    if len(table_rows) != 14:
        raise RuntimeError(f"s13_row_count:{len(table_rows)}")

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for dataset_label, method_label, h_mae, a_mae, lpips, sface in table_rows:
        dataset_id = {"FaceScape": "D1", "REALY": "D2"}.get(dataset_label)
        method_id = S13_METHOD_MAP.get(method_label)
        if dataset_id is None or method_id is None:
            raise RuntimeError(f"s13_unmapped:{dataset_label}:{method_label}")
        key = (dataset_id, method_id)
        if key in seen:
            raise RuntimeError(f"s13_duplicate:{key}")
        seen.add(key)
        values = {
            "h_mae": clean_number(h_mae),
            "a_mae": clean_number(a_mae),
            "lpips": clean_number(lpips),
            "sface": clean_number(sface),
        }
        for metric_id in METRIC_ORDER:
            identity_count = 100 if dataset_id == "D2" else (19 if metric_id == "sface" else 20)
            output.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "method_id": method_id,
                    "method_label": METHOD_DISPLAY[method_id],
                    "method_category": METHOD_CATEGORY[method_id],
                    "metric_id": metric_id,
                    "metric_label": metric_id.replace("_", "-").upper(),
                    "direction": "higher_is_better" if metric_id == "sface" else "lower_is_better",
                    "descriptive_median": f"{values[metric_id]:.6f}",
                    "identity_count": identity_count,
                    "output_mode": "observed_texture_preserved",
                    "aggregation": (
                        "pair_median_then_seed_median_then_identity_median"
                        if method_id in {"full", "condition0", "b_lite_ft"}
                        else "pair_median_then_identity_median"
                    ),
                    "source": "table_S13_descriptive_median",
                }
            )
    expected = {(dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER}
    if seen != expected:
        raise RuntimeError(f"s13_grid_mismatch:{sorted(expected - seen)}:{sorted(seen - expected)}")
    return output


def panorama_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["dataset_id"], row["method_id"], row["metric_id"])
        result[key] = float(row["descriptive_median"])
    if len(result) != 56:
        raise RuntimeError(f"panorama_cell_count:{len(result)}")
    return result


def save_figure(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".tiff", ".pdf", ".svg")]
    fig.savefig(outputs[0], dpi=300, facecolor="white")
    fig.savefig(outputs[1], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(
        outputs[2],
        facecolor="white",
        metadata={"Creator": "V15 Python figure pipeline", "CreationDate": None},
    )
    fig.savefig(
        outputs[3],
        facecolor="white",
        metadata={"Creator": "V15 Python figure pipeline", "Date": None},
    )
    plt.close(fig)
    return outputs


def readable_text_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else INK


def figure_panorama(rows: list[dict[str, Any]]) -> list[Path]:
    lookup = panorama_lookup(rows)
    panel_metric_limits = {
        (dataset, metric): (
            min(lookup[(dataset, method, metric)] for method in METHOD_ORDER),
            max(lookup[(dataset, method, metric)] for method in METHOD_ORDER),
        )
        for dataset in DATASET_ORDER
        for metric in METRIC_ORDER
    }
    colormap = LinearSegmentedColormap.from_list(
        "v15_sequential_blue", ["#F5F7F8", "#C9DFEA", SKY, BLUE, NAVY]
    )
    normalizers = {
        key: Normalize(vmin=limits[0], vmax=limits[1])
        for key, limits in panel_metric_limits.items()
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 113 / 25.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.subplots_adjust(left=0.285, right=0.985, top=0.78, bottom=0.17)
    fig.suptitle(
        "统一已观测纹理保持输出下的多指标描述性全景",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=10.2,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.02,
        0.925,
        "单元格为身份级描述性中位数；色阶在各数据集—指标列内独立归一化。",
        ha="left",
        va="top",
        fontsize=6.9,
        color=MUTED,
    )

    for panel_index, (axis, dataset_id) in enumerate(zip(axes, DATASET_ORDER)):
        axis.set_xlim(-0.5, len(METRIC_ORDER) - 0.5)
        axis.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
        axis.set_xticks(range(len(METRIC_ORDER)), [METRIC_LABEL[metric] for metric in METRIC_ORDER])
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", length=0, pad=5)
        axis.tick_params(axis="y", length=0, pad=17)
        axis.set_title(
            f"{DATASET_LABEL[dataset_id]}  "
            + ("n=20 个身份（SFace n=19）" if dataset_id == "D1" else "n=100 个身份"),
            fontsize=8.5,
            weight="bold",
            color=INK,
            pad=34,
        )
        if panel_index == 0:
            labels = [f"{METHOD_DISPLAY[m]}\n{METHOD_CATEGORY[m]}" for m in METHOD_ORDER]
            axis.set_yticks(range(len(METHOD_ORDER)), labels)
            for label in axis.get_yticklabels():
                label.set_fontsize(6.3)
                label.set_linespacing(1.32)
        else:
            axis.tick_params(labelleft=False)

        for row_index, method_id in enumerate(METHOD_ORDER):
            for column_index, metric_id in enumerate(METRIC_ORDER):
                value = lookup[(dataset_id, method_id, metric_id)]
                rgba = colormap(normalizers[(dataset_id, metric_id)](value))
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.49, row_index - 0.47),
                        0.98,
                        0.94,
                        facecolor=rgba,
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.6f}",
                    ha="center",
                    va="center",
                    fontsize=5.55,
                    family="DejaVu Sans Mono",
                    color=readable_text_color(rgba),
                    weight="bold" if method_id == "full" else "normal",
                )

        for boundary in (0.5, 1.5, 2.5, 3.5, 5.5):
            axis.axhline(boundary, color="#AEB8C0", linewidth=0.65, linestyle=(0, (2, 2)), zorder=3)
        for spine in axis.spines.values():
            spine.set_visible(False)

        # Method colors remain a secondary identity cue; exact labels carry category meaning.
        if panel_index == 0:
            for row_index, method_id in enumerate(METHOD_ORDER):
                axis.scatter(
                    -0.57,
                    row_index,
                    s=24,
                    marker="s",
                    color=METHOD_COLOR[method_id],
                    edgecolor="white",
                    linewidth=0.55,
                    clip_on=False,
                    zorder=5,
                )

    fig.text(
        0.02,
        0.07,
        "颜色仅辅助同一数据集和同一指标内的阅读；单元格给出身份级描述性中位数，数值保留六位小数。",
        ha="left",
        va="bottom",
        fontsize=5.85,
        color=MUTED,
    )
    return save_figure(fig, PANORAMA_STEM)


def read_resource_rows() -> dict[str, dict[str, float]]:
    rows = read_csv(EFFICIENCY)
    indexed: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("scope") != "end_to_end":
            continue
        method = row["method_id"]
        if method not in set(RESOURCE_METHOD_MAP.values()):
            continue
        indexed[method] = {
            "p50_ms": float(row["p50_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "inference_parameters": float(row["inference_parameters"]),
            "macs": float(row["macs"]) if row.get("macs") else math.nan,
            "peak_memory_mib": float(row["peak_allocated_bytes"]) / (1024.0**2),
        }
    missing = set(RESOURCE_METHOD_MAP.values()) - set(indexed)
    if missing:
        raise RuntimeError(f"resource_methods_missing:{sorted(missing)}")
    return indexed


def validate_task_updates() -> None:
    for method, expected in {"full": 89386, "b_lite": 0, "b_lite_ft": 122164}.items():
        if TASK_UPDATES.get(method) != expected:
            raise RuntimeError(f"task_update_mismatch:{method}:{TASK_UPDATES.get(method)}:{expected}")


def build_bubble_rows(panorama_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_task_updates()
    lookup = panorama_lookup(panorama_rows)
    resources = read_resource_rows()
    output: list[dict[str, Any]] = []
    for dataset_id in DATASET_ORDER:
        for method_id in BUBBLE_METHODS:
            resource_id = RESOURCE_METHOD_MAP[method_id]
            resource = resources[resource_id]
            output.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": DATASET_LABEL[dataset_id],
                    "method_id": method_id,
                    "method_label": METHOD_DISPLAY[method_id],
                    "method_category": METHOD_CATEGORY[method_id],
                    "h_mae": f"{lookup[(dataset_id, method_id, 'h_mae')]:.6f}",
                    "p50_ms": f"{resource['p50_ms']:.9f}",
                    "p95_ms": f"{resource['p95_ms']:.9f}",
                    "peak_memory_mib": f"{resource['peak_memory_mib']:.9f}",
                    "inference_parameters": int(resource["inference_parameters"]),
                    "macs": "" if math.isnan(resource["macs"]) else int(resource["macs"]),
                    "task_update_parameters": TASK_UPDATES[method_id],
                    "resource_measurement_id": resource_id,
                    "resource_measurement_scope": "RTX4090_FP32_batch1_end_to_end",
                    "resource_binding_note": (
                        "same_inference_graph_as_b_lite; uses frozen b_lite architecture measurement"
                        if method_id == "b_lite_ft"
                        else (
                            "end_to_end timing includes observed-texture-preserving projection; separate 64x64 projection microbenchmark p50 0.0507 ms does not adjust total"
                            if method_id == "freeuv_conserved"
                            else "direct frozen method measurement"
                        )
                    ),
                    "quality_source": "table_S13_descriptive_median",
                }
            )
    if len(output) != 12:
        raise RuntimeError(f"bubble_row_count:{len(output)}")
    return output


def update_label(value: int) -> str:
    if value == 0:
        return "0"
    return f"{value / 1000:.1f}k"


def bubble_area(memory_mib: float, minimum: float, maximum: float) -> float:
    low = math.log10(minimum)
    high = math.log10(maximum)
    if high <= low:
        return 80.0
    fraction = (math.log10(memory_mib) - low) / (high - low)
    return 52.0 + 190.0 * fraction


def figure_bubbles(rows: list[dict[str, Any]]) -> list[Path]:
    memory_values = [float(row["peak_memory_mib"]) for row in rows]
    memory_min, memory_max = min(memory_values), max(memory_values)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 105 / 25.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.70, bottom=0.20)
    fig.suptitle(
        "完整流程时延、隐藏区域误差与显存关系",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=10.2,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.02,
        0.925,
        "统一已观测纹理保持输出；RTX 4090、FP32、batch=1；横轴为对数坐标，气泡面积按峰值显存的对数缩放。",
        ha="left",
        va="top",
        fontsize=6.7,
        color=MUTED,
    )

    legend_handles: list[Line2D] = []
    for method_id in BUBBLE_METHODS:
        label = METHOD_DISPLAY[method_id]
        if method_id == "freeuv_conserved":
            label = "FreeUV"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKER[method_id],
                color="none",
                markerfacecolor=METHOD_COLOR[method_id],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=6.2,
                label=f"{label}（更新 {update_label(TASK_UPDATES[method_id])}）",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.865),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.45,
        borderaxespad=0.0,
        fontsize=5.9,
    )

    short_label = {
        "full": "FrugalFace3D-Lite",
        "b_lite": "固定权重 B-lite",
        "b_lite_ft": "B-lite 同任务微调",
        "lama": "LaMa",
        "zits": "ZITS",
        "freeuv_conserved": "FreeUV",
    }
    offsets = {
        "D1": {
            "full": (110, -12),
            "b_lite": (30, 18),
            "b_lite_ft": (25, 8),
            "lama": (35, 22),
            "zits": (8, 0),
            "freeuv_conserved": (-30, 9),
        },
        "D2": {
            "full": (38, -21),
            "b_lite": (35, 7),
            "b_lite_ft": (35, 28),
            "lama": (38, -38),
            "zits": (8, 0),
            "freeuv_conserved": (-32, -13),
        },
    }
    for axis, dataset_id in zip(axes, DATASET_ORDER):
        selected = [row for row in rows if row["dataset_id"] == dataset_id]
        for row in selected:
            method_id = row["method_id"]
            x = float(row["p50_ms"])
            y = float(row["h_mae"])
            memory = float(row["peak_memory_mib"])
            axis.scatter(
                x,
                y,
                s=bubble_area(memory, memory_min, memory_max),
                marker=METHOD_MARKER[method_id],
                color=METHOD_COLOR[method_id],
                edgecolor="white",
                linewidth=0.85,
                alpha=0.90,
                zorder=4,
            )
            dx, dy = offsets[dataset_id][method_id]
            axis.annotate(
                short_label[method_id],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.45,
                color=INK,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#97A3AD", "lw": 0.55},
                zorder=5,
            )
        axis.set_xscale("log")
        axis.set_xlim(560, 6200)
        axis.set_ylim(0.10, 0.33)
        axis.xaxis.set_major_locator(FixedLocator([600, 1000, 2000, 5000]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.grid(True, which="major", color=GRID, linewidth=0.65, linestyle=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.set_title(DATASET_LABEL[dataset_id], fontsize=8.5, weight="bold", pad=7)
        axis.set_xlabel("完整流程 p50 时延 / ms（对数坐标，越低越优）", labelpad=5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("H-MAE（越低越优）")

    legend_memories = (80, 300, 900, 7700)
    memory_handles = [
        plt.scatter(
            [],
            [],
            s=bubble_area(value, memory_min, memory_max),
            facecolor="#D9E2E8",
            edgecolor="#65737E",
            linewidth=0.65,
        )
        for value in legend_memories
    ]
    axes[1].legend(
        memory_handles,
        [f"{value:,}" for value in legend_memories],
        title="峰值显存 / MiB",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD3D9",
        fontsize=5.6,
        title_fontsize=5.9,
        labelspacing=0.7,
        borderpad=0.6,
    )
    fig.text(
        0.02,
        0.07,
        "标签为当前任务更新参数量；B-lite 与 B-lite 同任务微调共享推理图及资源测量。",
        ha="left",
        va="bottom",
        fontsize=5.75,
        color=MUTED,
    )
    return save_figure(fig, BUBBLE_STEM)


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def first_present(row: dict[str, str], aliases: Iterable[str]) -> str | None:
    normalized = {normalized_name(key): value for key, value in row.items()}
    for alias in aliases:
        value = normalized.get(normalized_name(alias))
        if value not in (None, ""):
            return value
    return None


def normalize_view(value: str) -> str:
    match = re.search(r"(0?[1-4])", value)
    if not match:
        raise RuntimeError(f"invalid_view:{value}")
    return f"V{int(match.group(1)):02d}"


def normalize_comparison(value: str) -> str:
    token = normalized_name(value)
    if "condition0" in token or "nocond" in token:
        return "full_vs_condition0"
    if "bliteft" in token or "blitesametask" in token:
        return "full_vs_b_lite_ft"
    if "freeuv" in token:
        return "full_vs_freeuv_conserved"
    raise RuntimeError(f"invalid_comparison:{value}")


def discover_direction_csv() -> tuple[Path | None, list[dict[str, Any]]]:
    if not ANALYSIS.exists():
        return None, []
    candidates: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(ANALYSIS.rglob("*.csv")):
        try:
            raw_rows = read_csv(path)
        except (UnicodeError, csv.Error):
            continue
        standardized: list[dict[str, Any]] = []
        try:
            for row in raw_rows:
                source_view = first_present(row, ("source_view", "source", "src_view", "src", "source_index"))
                target_view = first_present(row, ("target_view", "target", "tgt_view", "tgt", "target_index"))
                comparison = first_present(row, ("comparison_id", "comparison", "contrast", "comparator_id", "comparator"))
                effect = first_present(
                    row,
                    (
                        "identity_median_effect",
                        "median_identity_effect",
                        "median_effect",
                        "effect_median",
                        "median_h_mae_effect",
                        "median_full_advantage",
                    ),
                )
                if None in (source_view, target_view, comparison, effect):
                    raise KeyError("required_direction_field")
                pair_count = first_present(row, ("pair_count", "n_pairs", "eligible_pair_count", "valid_pair_count"))
                identity_count = first_present(row, ("identity_count", "n_identities", "valid_identity_count"))
                effect_definition = first_present(
                    row,
                    ("effect_definition", "positive_means", "effect_sign_definition", "contrast_definition"),
                )
                positive_favors = first_present(row, ("positive_favors", "positive_effect_favors"))
                median_effect_rgb8 = first_present(row, ("median_identity_effect_rgb8", "median_effect_rgb8"))
                ci_low = first_present(row, ("ci95_identity_bootstrap_low", "ci95_low", "bootstrap_ci95_low"))
                ci_high = first_present(row, ("ci95_identity_bootstrap_high", "ci95_high", "bootstrap_ci95_high"))
                positive_count = first_present(row, ("positive_identity_count", "positive_count"))
                zero_count = first_present(row, ("zero_identity_count", "zero_count"))
                negative_count = first_present(row, ("negative_identity_count", "negative_count"))
                analysis_status = first_present(row, ("analysis_status", "status"))
                standardized.append(
                    {
                        "source_view": normalize_view(source_view),
                        "target_view": normalize_view(target_view),
                        "comparison_id": normalize_comparison(comparison),
                        "median_effect": float(effect),
                        "pair_count": int(float(pair_count)) if pair_count is not None else "",
                        "identity_count": int(float(identity_count)) if identity_count is not None else "",
                        "effect_definition": effect_definition or "",
                        "positive_favors": positive_favors or "",
                        "median_effect_rgb8": float(median_effect_rgb8) if median_effect_rgb8 is not None else "",
                        "ci95_identity_bootstrap_low": float(ci_low) if ci_low is not None else "",
                        "ci95_identity_bootstrap_high": float(ci_high) if ci_high is not None else "",
                        "positive_identity_count": int(float(positive_count)) if positive_count is not None else "",
                        "zero_identity_count": int(float(zero_count)) if zero_count is not None else "",
                        "negative_identity_count": int(float(negative_count)) if negative_count is not None else "",
                        "analysis_status": analysis_status or "",
                    }
                )
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        if len(standardized) == 36:
            candidates.append((path, standardized))
    if not candidates:
        return None, []
    if len(candidates) > 1:
        raise RuntimeError("multiple_direction_csv_candidates:" + ",".join(str(path) for path, _ in candidates))
    path, rows = candidates[0]
    expected_pairs = {(source, target) for source in ("V01", "V02", "V03", "V04") for target in ("V01", "V02", "V03", "V04") if source != target}
    comparisons = {row["comparison_id"] for row in rows}
    expected_comparisons = {
        "full_vs_condition0",
        "full_vs_b_lite_ft",
        "full_vs_freeuv_conserved",
    }
    if comparisons != expected_comparisons:
        raise RuntimeError(f"direction_comparison_grid:{sorted(comparisons)}")
    for comparison in expected_comparisons:
        observed_pairs = {
            (row["source_view"], row["target_view"])
            for row in rows
            if row["comparison_id"] == comparison
        }
        if observed_pairs != expected_pairs:
            raise RuntimeError(f"direction_pair_grid:{comparison}")
    if not all(row["effect_definition"] for row in rows):
        raise RuntimeError("direction_effect_definition_missing")
    if not all(row["positive_favors"] for row in rows):
        raise RuntimeError("direction_positive_favors_missing")
    return path, rows


def figure_directions(rows: list[dict[str, Any]]) -> list[Path]:
    comparison_order = (
        "full_vs_condition0",
        "full_vs_b_lite_ft",
        "full_vs_freeuv_conserved",
    )
    comparison_label = {
        "full_vs_condition0": "与无显式条件残差\n（NoCond）比较",
        "full_vs_b_lite_ft": "与 B-lite 同任务微调比较",
        "full_vs_freeuv_conserved": "与 FreeUV\n（已观测纹理保持输出）比较",
    }
    views = ("V01", "V02", "V03", "V04")
    max_abs = max(abs(float(row["median_effect"])) for row in rows)
    if max_abs == 0:
        max_abs = 1.0
    normalization = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    colormap = LinearSegmentedColormap.from_list(
        "v15_direction_diverge", [ORANGE, "#F4F5F6", BLUE]
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183 / 25.4, 92 / 25.4),
        gridspec_kw={"wspace": 0.12},
    )
    fig.subplots_adjust(left=0.075, right=0.93, top=0.73, bottom=0.19)
    fig.suptitle(
        "REALY 全部有向视图对的探索性 H-MAE 效应",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=10.2,
        weight="bold",
        color=NAVY,
    )
    definitions = sorted({row["effect_definition"] for row in rows})
    positive_favors = sorted({row["positive_favors"] for row in rows})
    if definitions == ["comparator_minus_full"] and positive_favors == ["FrugalFace3D-Lite"]:
        definition_text = (
            "效应 = 比较方法 H-MAE − FrugalFace3D-Lite H-MAE；"
            "正值表示 FrugalFace3D-Lite 误差更低"
        )
    else:
        definition_text = "效应定义：" + "；".join(definitions) + "；正值有利于：" + "、".join(positive_favors)
    fig.text(
        0.02,
        0.915,
        "12 个非对角方向；单元格为身份级中位效应与身份数；" + definition_text,
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    for axis, comparison in zip(axes, comparison_order):
        indexed = {
            (row["source_view"], row["target_view"]): row
            for row in rows
            if row["comparison_id"] == comparison
        }
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(3.5, -0.5)
        axis.set_xticks(range(4), views)
        axis.set_yticks(range(4), views)
        axis.tick_params(length=0)
        axis.set_xlabel("目标视图")
        if axis is axes[0]:
            axis.set_ylabel("源视图")
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(comparison_label[comparison], fontsize=7.8, weight="bold", pad=6)
        for source_index, source in enumerate(views):
            for target_index, target in enumerate(views):
                if source == target:
                    face = "#ECEFF1"
                    axis.add_patch(
                        Rectangle(
                            (target_index - 0.48, source_index - 0.48),
                            0.96,
                            0.96,
                            facecolor=face,
                            edgecolor="white",
                            linewidth=1.1,
                        )
                    )
                    axis.text(target_index, source_index, "—", ha="center", va="center", fontsize=7.0, color="#98A2AA")
                    continue
                row = indexed[(source, target)]
                value = float(row["median_effect"])
                rgba = colormap(normalization(value))
                axis.add_patch(
                    Rectangle(
                        (target_index - 0.48, source_index - 0.48),
                        0.96,
                        0.96,
                        facecolor=rgba,
                        edgecolor="white",
                        linewidth=1.1,
                    )
                )
                coverage = row["identity_count"] or row["pair_count"]
                coverage_label = f"n={coverage}" if coverage != "" else ""
                value_label = f"{value:+.5f}" if 0 < abs(value) < 0.0001 else f"{value:+.4f}"
                axis.text(
                    target_index,
                    source_index - 0.09,
                    value_label,
                    ha="center",
                    va="center",
                    fontsize=5.55,
                    family="DejaVu Sans Mono",
                    color=readable_text_color(rgba),
                )
                axis.text(
                    target_index,
                    source_index + 0.20,
                    coverage_label,
                    ha="center",
                    va="center",
                    fontsize=4.9,
                    color=readable_text_color(rgba),
                )
        for spine in axis.spines.values():
            spine.set_visible(False)

    color_axis = fig.add_axes([0.945, 0.235, 0.014, 0.43])
    boundaries = np.linspace(-max_abs, max_abs, 129)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (lower + upper) / 2.0
        color_axis.add_patch(
            Rectangle(
                (0.0, lower),
                1.0,
                upper - lower,
                facecolor=colormap(normalization(midpoint)),
                edgecolor="none",
            )
        )
    color_axis.set_xlim(0.0, 1.0)
    color_axis.set_ylim(-max_abs, max_abs)
    color_axis.set_xticks([])
    color_axis.yaxis.tick_right()
    color_axis.set_yticks(np.linspace(-max_abs, max_abs, 5))
    color_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.02f}"))
    color_axis.tick_params(axis="y", labelsize=5.4, length=2)
    color_axis.set_title("效应", fontsize=5.8, pad=4)
    for spine in color_axis.spines.values():
        spine.set_color("#7E8993")
        spine.set_linewidth(0.7)
    fig.text(
        0.02,
        0.065,
        "蓝色表示正值，橙色表示负值；三项比较共享以零为中心的色阶。",
        ha="left",
        va="bottom",
        fontsize=5.75,
        color=MUTED,
    )
    return save_figure(fig, DIRECTION_STEM)


def validate_raster(path: Path, expected_dpi: int) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        extrema = image.convert("RGB").getextrema()
        dpi = image.info.get("dpi")
        if isinstance(dpi, tuple):
            dpi_value: Any = [float(item) for item in dpi]
        elif dpi is None:
            dpi_value = None
        else:
            dpi_value = float(dpi)
        blank = all(low >= 250 and high >= 250 for low, high in extrema)
        return {
            "width_px": width,
            "height_px": height,
            "dpi": dpi_value,
            "expected_dpi": expected_dpi,
            "not_blank": not blank,
            "minimum_dimensions_pass": width >= 1800 and height >= 900,
        }


def validate_vector(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return {
        "contains_text_nodes": "<text" in content,
        "contains_embedded_raster": "data:image" in content,
        "size_bytes": path.stat().st_size,
    }


def build_manifest(
    generated_sources: list[Path],
    generated_assets: list[Path],
    direction_input: Path | None,
    direction_status: str,
) -> dict[str, Any]:
    inputs = [SUPPLEMENT, EFFICIENCY, CONTRACT]
    if direction_input is not None:
        inputs.append(direction_input)
    return {
        "schema_version": "frugalface3d.v15.additional_figures.v1",
        "status": direction_status,
        "backend": "python_matplotlib_only",
        "scientific_boundary": {
            "training": False,
            "model_inference": False,
            "new_resampling": False,
            "new_hypothesis_tests": False,
            "composite_score_or_rank": False,
            "direction_analysis_enters_holm_family": False,
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in inputs},
        "source_sha256": {path.name: sha256_file(path) for path in generated_sources},
        "asset_sha256": {path.name: sha256_file(path) for path in generated_assets},
        "figures": {
            "descriptive_multimetric_panorama": {
                "source_rows": 56,
                "datasets": 2,
                "methods": 7,
                "metrics": 4,
                "composite_score": False,
            },
            "quality_resource_bubbles": {
                "source_rows": 12,
                "datasets": 2,
                "methods": 6,
                "x": "end_to_end_p50_ms_log_axis",
                "y": "H_MAE",
                "bubble": "peak_memory_mib_log_area",
                "labels": "task_update_parameters",
                "b_lite_ft_resource_binding": "same inference graph as frozen B-lite measurement",
            },
            "realy_12direction_effects": {
                "status": direction_status,
                "source_rows": 36 if direction_input is not None else 0,
                "new_p_values": False,
            },
        },
    }


def build_qa(
    panorama_rows: list[dict[str, Any]],
    bubble_rows: list[dict[str, Any]],
    assets: list[Path],
    direction_rows: list[dict[str, Any]],
    direction_status: str,
    manual_visual_review: str,
) -> dict[str, Any]:
    raster_checks = {}
    vector_checks = {}
    for path in assets:
        if path.suffix.lower() in {".png", ".tiff"}:
            raster_checks[path.name] = validate_raster(path, 300 if path.suffix.lower() == ".png" else 600)
        elif path.suffix.lower() == ".svg":
            vector_checks[path.name] = validate_vector(path)

    bubble_index = {
        (row["dataset_id"], row["method_id"]): row
        for row in bubble_rows
    }
    full_not_faster_than_b_lite = all(
        float(bubble_index[(dataset, "full")]["p50_ms"])
        > float(bubble_index[(dataset, "b_lite")]["p50_ms"])
        for dataset in DATASET_ORDER
    )
    b_lite_binding = all(
        bubble_index[(dataset, "b_lite_ft")]["p50_ms"]
        == bubble_index[(dataset, "b_lite")]["p50_ms"]
        and bubble_index[(dataset, "b_lite_ft")]["peak_memory_mib"]
        == bubble_index[(dataset, "b_lite")]["peak_memory_mib"]
        for dataset in DATASET_ORDER
    )
    static_pass = (
        len(panorama_rows) == 56
        and len(bubble_rows) == 12
        and full_not_faster_than_b_lite
        and b_lite_binding
        and all(check["not_blank"] and check["minimum_dimensions_pass"] for check in raster_checks.values())
        and all(check["contains_text_nodes"] and not check["contains_embedded_raster"] for check in vector_checks.values())
        and (direction_status != "PASS_DIRECTION_GENERATED" or len(direction_rows) == 36)
    )
    return {
        "schema_version": "frugalface3d.v15.additional_figure_qa.v1",
        "status": "PASS_STATIC_QA" if static_pass else "FAIL_STATIC_QA",
        "manual_visual_review": manual_visual_review,
        "panorama": {
            "row_count": len(panorama_rows),
            "expected_row_count": 56,
            "no_composite_score_or_rank": True,
            "metric_directions_visible": True,
        },
        "quality_resource": {
            "row_count": len(bubble_rows),
            "expected_row_count": 12,
            "full_p50_greater_than_fixed_b_lite": full_not_faster_than_b_lite,
            "b_lite_ft_resource_binding_matches_b_lite": b_lite_binding,
            "freeuv_macs_imputed": False,
        },
        "direction": {
            "status": direction_status,
            "row_count": len(direction_rows),
            "new_p_values": False,
        },
        "raster_checks": raster_checks,
        "vector_checks": vector_checks,
    }


def qa_markdown(qa_result: dict[str, Any]) -> str:
    visual = qa_result["manual_visual_review"]
    direction_status = qa_result["direction"]["status"]
    return f"""# V15 新增图件质量核查

- 静态核查：`{qa_result['status']}`
- 人工视觉核查：`{visual}`
- 12 方向图：`{direction_status}`
- 绘图后端：Python/Matplotlib only
- 新训练、模型推理、重采样或新增显著性检验：均未执行

## 图 A：多指标描述性全景

- 56 行来源数据完整，对应 7 种方法、2 个数据集和 4 项指标。
- H-MAE、A-MAE、LPIPS 的向下箭头和 SFace 的向上箭头可见。
- 单元格显示表 S13 的描述性中位数并保留六位小数；色阶在各数据集—指标列内独立归一化。
- 图中未计算综合分数、平均排名或显著性标记。

## 图 B：完整流程质量—资源关系

- 12 行来源数据完整，对应 6 种方法和 2 个数据集。
- 横轴明确标注为对数坐标；气泡面积明确标注为峰值显存的对数缩放。
- 图例逐方法显示任务更新参数量；固定权重方法在本研究中的更新参数量为 0。
- FrugalFace3D-Lite 位于固定权重 B-lite 的右侧，未形成其更快的视觉暗示。
- B-lite 同任务微调与固定权重 B-lite 的资源坐标来自同一推理图测量，来源 CSV 已明确记录该绑定。

## 图 C：REALY 全部有向视图对

- 36 行来源数据完整，对应 12 个非对角方向和 3 项比较。
- 三个面板使用同一以零为中心的发散色阶；对角线留空。
- 图中明确给出“比较方法 H-MAE − FrugalFace3D-Lite H-MAE”的效应定义，正值表示 FrugalFace3D-Lite 误差更低。
- 每个单元格显示身份级中位效应和身份数；没有 p 值或 Holm 判定。

## 导出检查

- 每张图均已导出 PNG、TIFF、PDF 和 SVG。
- TIFF 为 600 dpi；PNG 为 300 dpi；SVG 保留文字节点且不嵌入栅格图像。
- 视觉核查覆盖标题、图例、坐标、单元格文字、直接标签、色阶、边距和明显遮挡。
"""


def write_checksums(paths: list[Path]) -> None:
    unique = sorted(set(paths), key=lambda path: str(path))
    lines = [f"{sha256_file(path)}  {path.relative_to(ROOT)}" for path in unique]
    atomic_text(CHECKSUMS, "\n".join(lines) + "\n")


def rebuild_from_public_sources(output: Path) -> list[Path]:
    """Render Figure 5 and Supplementary Figures S2 and S3."""
    global PANORAMA_STEM, BUBBLE_STEM, DIRECTION_STEM

    required = (PANORAMA_SOURCE, BUBBLE_SOURCE, DIRECTION_SOURCE)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    panorama_rows = read_csv(PANORAMA_SOURCE)
    bubble_rows = read_csv(BUBBLE_SOURCE)
    direction_rows = read_csv(DIRECTION_SOURCE)
    if len(panorama_rows) != 56:
        raise RuntimeError(f"panorama_row_count:{len(panorama_rows)}")
    if len(bubble_rows) != 12:
        raise RuntimeError(f"bubble_row_count:{len(bubble_rows)}")
    if len(direction_rows) != 36:
        raise RuntimeError(f"direction_row_count:{len(direction_rows)}")
    if any(row.get("analysis_status") != "EXPLORATORY_NO_SIGNIFICANCE_TEST" for row in direction_rows):
        raise RuntimeError("direction_inference_boundary")

    output.mkdir(parents=True, exist_ok=True)
    PANORAMA_STEM = output / "v15_descriptive_multimetric_panorama_zh"
    BUBBLE_STEM = output / "v15_quality_resource_bubbles_zh"
    DIRECTION_STEM = output / "v15_realy_12direction_effects_zh"
    configure_matplotlib()
    return (
        figure_bubbles(bubble_rows)
        + figure_directions(direction_rows)
        + figure_panorama(panorama_rows)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS,
        help="Directory for rebuilt PNG, TIFF, PDF, and SVG files.",
    )
    args = parser.parse_args()
    outputs = rebuild_from_public_sources(args.output)
    print(
        json.dumps(
            {
                "status": "PASS_V15_SUPPLEMENTARY_NONFACE_FIGURES_REBUILT",
                "figure_count": 3,
                "asset_count": len(outputs),
                "training_or_inference_performed": False,
                "new_significance_tests": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
