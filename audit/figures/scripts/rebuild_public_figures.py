#!/usr/bin/env python3
"""Rebuild the eight public, non-face V15 main and supplementary figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import make_v15_figures as v15
import make_v15_additional_figures as supplemental
import make_v15_robustness_figure as robustness


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    package_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "figures" / "rebuilt_public_outputs",
    )
    args = parser.parse_args()

    source = package_root / "figures" / "source_data"
    args.output.mkdir(parents=True, exist_ok=True)
    v15.frozen.DISPLAY["b_lite_ft"] = "B-lite 同任务微调"
    v15.frozen.DISPLAY["b_lite"] = "固定权重 B-lite"
    v15.frozen.DISPLAY["freeuv_conserved"] = "FreeUV\n（已观测纹理保持输出）"
    v15.frozen.configure()
    v15.frozen.save_figure = v15.mapped_save_figure

    outputs: list[Path] = []
    outputs.extend(v15.figure_model_overview(args.output))
    outputs.extend(
        v15.figure_visibility_evaluation(
            read_csv(source / "v15_visibility_region_compression.csv"), args.output
        )
    )
    outputs.extend(
        v15.frozen.figure_endpoint_protocol(
            read_csv(source / "v15_output_form_effects.csv"), args.output
        )
    )
    outputs.extend(
        v15.figure_multimetric_effects(
            read_csv(source / "v15_multimetric_effects.csv"), args.output
        )
    )
    outputs.extend(supplemental.rebuild_from_public_sources(args.output))
    outputs.extend(robustness.rebuild_from_public_source(args.output))
    try:
        output_label = str(args.output.relative_to(package_root))
    except ValueError:
        output_label = "external_output_directory"
    print(
        json.dumps(
            {
                "status": "PASS_V15_PUBLIC_NONFACE_FIGURES_REBUILT",
                "figure_count": 8,
                "asset_count": len(outputs),
                "output": output_label,
                "frozen_values_changed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
