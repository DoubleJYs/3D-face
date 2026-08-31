#!/usr/bin/env python3
"""Rebuild Supplementary Figure S1 from the packaged REALY sensitivity table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np

import _historical_v14_renderer as frozen


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "robustness" / "REALY_H_SUPPORT_SENSITIVITY.csv"
METHODS = (
    ("condition0", "无显式条件残差\n（NoCond）", "#6F5AA8"),
    ("b_lite_ft", "B-lite 同任务微调", "#E69F00"),
    ("freeuv_conserved", "FreeUV\n（已观测纹理保持输出）", "#0072B2"),
)
THRESHOLDS = (1, 5, 10, 20, 50)


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rebuild_from_public_source(output: Path) -> list[Path]:
    """Render the five-threshold, three-comparator exploratory sensitivity plot."""
    rows = read_rows()
    expected_grid = {
        (comparator, threshold)
        for comparator, _label, _color in METHODS
        for threshold in THRESHOLDS
    }
    observed_grid = {
        (row["comparator_id"], int(row["support_threshold_texels"]))
        for row in rows
    }
    if len(rows) != 15 or observed_grid != expected_grid:
        raise RuntimeError("realy_support_sensitivity_grid")
    if any(
        row["dataset_id"] != "D2"
        or row["metric_id"] != "hidden_uv_mae"
        or row["new_p_value_generated"] != "False"
        or row["holm_recalculated"] != "False"
        for row in rows
    ):
        raise RuntimeError("realy_support_sensitivity_boundary")

    frozen.configure()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=False)
    positions = np.arange(len(THRESHOLDS))
    for axis, (method_id, label, color) in zip(axes, METHODS):
        method_rows = sorted(
            (row for row in rows if row["comparator_id"] == method_id),
            key=lambda row: int(row["support_threshold_texels"]),
        )
        effects = np.array([float(row["median_identity_effect"]) for row in method_rows])
        lows = np.array([float(row["ci95_low"]) for row in method_rows])
        highs = np.array([float(row["ci95_high"]) for row in method_rows])
        axis.axhline(0.0, color="#4C5660", linewidth=0.8, zorder=0)
        axis.errorbar(
            positions,
            effects,
            yerr=np.vstack([effects - lows, highs - effects]),
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=1.15,
            elinewidth=1.1,
            capsize=2.3,
            markersize=4.2,
        )
        axis.set_xticks(
            positions,
            [
                f"{row['support_threshold_texels']}\n{row['pair_count']}/{row['identity_count']}"
                for row in method_rows
            ],
        )
        axis.set_title(label, color="#17365D", weight="bold", fontsize=9.2)
        axis.set_xlabel("最小有效 texel 数\n配对数/身份数", fontsize=7.4)
        axis.grid(axis="y", color="#D6DCE1", linewidth=0.5, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.0)
    axes[0].set_ylabel(
        "目标可见隐藏区域 MAE 身份差异\n（比较方法 − FrugalFace3D-Lite）",
        fontsize=7.8,
    )
    fig.suptitle(
        "REALY 五种子最小有效 texel 数敏感性",
        y=0.985,
        color="#17365D",
        weight="bold",
        fontsize=10.2,
    )
    fig.text(
        0.5,
        0.005,
        "点为身份效应中位数，误差线为未校正 95% 身份自助区间；横轴同时标注最小有效 texel 数、配对数与身份数。",
        ha="center",
        fontsize=7.1,
        color="#4C5660",
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.81, bottom=0.29, wspace=0.29)
    return frozen.save_figure(fig, output, "v15_realy_support_sensitivity_zh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "rebuilt_public_outputs",
    )
    args = parser.parse_args()
    outputs = rebuild_from_public_source(args.output)
    print(
        json.dumps(
            {
                "status": "PASS_V15_REALY_SUPPORT_SENSITIVITY_REBUILT",
                "source": "robustness/REALY_H_SUPPORT_SENSITIVITY.csv",
                "asset_count": len(outputs),
                "new_hypothesis_tests": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
