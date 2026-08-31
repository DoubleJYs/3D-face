#!/usr/bin/env python3
"""Recompute the public REALY directional summaries from anonymous effects.

The public identity table contains only anonymous identity tokens, direction
labels, comparison identifiers, five fixed-seed effects, and their identity
median. It deliberately excludes pair identifiers and per-pair support values.
This script verifies that redaction boundary and independently recomputes the
36 exploratory direction summaries, including the prespecified identity
bootstrap intervals. It does not calculate p-values or multiple-comparison
corrections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
DEFAULT_IDENTITY = ROOT / "REALY_DIRECTION_IDENTITY_EFFECTS.csv"
DEFAULT_SUMMARY = ROOT / "REALY_DIRECTION_EFFECT_SUMMARY.csv"

SEEDS = (2026080447, 2026080448, 2026080449, 2026080450, 2026080451)
DIRECTIONS = (
    ("V01", "V02"),
    ("V01", "V03"),
    ("V01", "V04"),
    ("V02", "V01"),
    ("V02", "V03"),
    ("V02", "V04"),
    ("V03", "V01"),
    ("V03", "V02"),
    ("V03", "V04"),
    ("V04", "V01"),
    ("V04", "V02"),
    ("V04", "V03"),
)
COMPARATORS = ("condition0", "b_lite_ft", "freeuv_conserved")
COMPARATOR_SERIAL = {"condition0": 1, "b_lite_ft": 2, "freeuv_conserved": 3}
COMPARATOR_LABEL = {
    "condition0": "NoCond",
    "b_lite_ft": "B-lite 同任务微调",
    "freeuv_conserved": "FreeUV（已观测纹理保持）",
}
BOOTSTRAP_BASE_SEED = 2026082300
BOOTSTRAP_RESAMPLES = 10000

PUBLIC_FIELDS = (
    "direction_index",
    "direction",
    "source_view",
    "target_view",
    "comparison_id",
    "comparator_id",
    "comparator_label",
    "identity_token",
    "identity_effect",
    "identity_effect_rgb8",
    *(f"seed_{seed}_effect" for seed in SEEDS),
)
PRIVATE_ONLY_FIELDS = {
    "pair_id",
    "hidden_support_texels",
    "all_target_visible_support_texels",
    "h_over_a_ratio",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_public_identity(source: Path, destination: Path) -> None:
    """Create the path-free public identity table from the private derivative."""
    rows = read_csv(source)
    if len(rows) != 3600:
        fail(f"private_identity_row_count:{len(rows)}")
    missing = set(PUBLIC_FIELDS) - set(rows[0])
    if missing:
        fail(f"private_identity_missing_fields:{sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in PUBLIC_FIELDS})
    temporary.replace(destination)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        fail("invalid_quantile")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(values: Sequence[float], seed: int) -> tuple[float, float]:
    if len(values) != 100:
        fail(f"bootstrap_identity_count:{len(values)}")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        estimates.append(float(statistics.median(sample)))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def finite(value: str, role: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid_float:{role}") from error
    if not math.isfinite(parsed):
        fail(f"nonfinite:{role}")
    return parsed


def load_public_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if len(rows) != 3600:
        fail(f"public_identity_row_count:{len(rows)}")
    fields = set(rows[0])
    if fields != set(PUBLIC_FIELDS):
        fail(f"public_identity_schema:{sorted(fields)}")
    leaked = fields & PRIVATE_ONLY_FIELDS
    if leaked:
        fail(f"private_fields_present:{sorted(leaked)}")
    return rows


def recompute(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    seen_keys: set[tuple[int, str, str]] = set()
    for row in rows:
        direction_index = int(row["direction_index"])
        if not 0 <= direction_index < len(DIRECTIONS):
            fail(f"direction_index:{direction_index}")
        source_view, target_view = DIRECTIONS[direction_index]
        direction = f"{source_view}→{target_view}"
        comparator = row["comparator_id"]
        if comparator not in COMPARATORS:
            fail(f"comparator:{comparator}")
        expected_comparison = f"full_vs_{comparator}"
        if (
            row["source_view"] != source_view
            or row["target_view"] != target_view
            or row["direction"] != direction
            or row["comparison_id"] != expected_comparison
            or row["comparator_label"] != COMPARATOR_LABEL[comparator]
        ):
            fail(f"direction_or_comparison_binding:{direction_index}:{comparator}")
        identity = row["identity_token"]
        if re.fullmatch(r"D2-\d{3}", identity) is None:
            fail(f"identity_token:{identity}")
        unique = (direction_index, comparator, identity)
        if unique in seen_keys:
            fail(f"duplicate_identity:{unique}")
        seen_keys.add(unique)

        seed_values = [finite(row[f"seed_{seed}_effect"], f"{unique}:{seed}") for seed in SEEDS]
        identity_effect = finite(row["identity_effect"], f"{unique}:identity")
        identity_rgb8 = finite(row["identity_effect_rgb8"], f"{unique}:rgb8")
        if abs(identity_effect - float(statistics.median(seed_values))) > 1e-15:
            fail(f"identity_seed_median:{unique}")
        if abs(identity_rgb8 - identity_effect * 255.0) > 1e-12:
            fail(f"identity_rgb8:{unique}")
        grouped[(direction_index, comparator)].append(row)

    expected_groups = {(index, comparator) for index in range(12) for comparator in COMPARATORS}
    if set(grouped) != expected_groups:
        fail("direction_comparison_grid")

    result: list[dict[str, object]] = []
    for direction_index in range(12):
        source_view, target_view = DIRECTIONS[direction_index]
        for comparator in COMPARATORS:
            group = sorted(grouped[(direction_index, comparator)], key=lambda row: row["identity_token"])
            if len(group) != 100 or len({row["identity_token"] for row in group}) != 100:
                fail(f"identity_coverage:{direction_index}:{comparator}")
            effects = [float(row["identity_effect"]) for row in group]
            bootstrap_seed = BOOTSTRAP_BASE_SEED + 10 * direction_index + COMPARATOR_SERIAL[comparator]
            ci_low, ci_high = bootstrap_interval(effects, bootstrap_seed)
            median_effect = float(statistics.median(effects))
            result.append(
                {
                    "direction_index": direction_index,
                    "direction": f"{source_view}→{target_view}",
                    "source_view": source_view,
                    "target_view": target_view,
                    "comparison_id": f"full_vs_{comparator}",
                    "comparator_id": comparator,
                    "comparator_label": COMPARATOR_LABEL[comparator],
                    "metric_id": "hidden_uv_mae",
                    "effect_definition": "comparator_minus_full",
                    "positive_favors": "FrugalFace3D-Lite",
                    "pair_count": 100,
                    "identity_count": 100,
                    "median_identity_effect": median_effect,
                    "median_identity_effect_rgb8": median_effect * 255.0,
                    "ci95_identity_bootstrap_low": ci_low,
                    "ci95_identity_bootstrap_high": ci_high,
                    "positive_identity_count": sum(value > 0.0 for value in effects),
                    "zero_identity_count": sum(value == 0.0 for value in effects),
                    "negative_identity_count": sum(value < 0.0 for value in effects),
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": bootstrap_seed,
                    "analysis_status": "EXPLORATORY_NO_SIGNIFICANCE_TEST",
                    **{
                        f"seed_{seed}_median_identity_effect": float(
                            statistics.median(float(row[f"seed_{seed}_effect"]) for row in group)
                        )
                        for seed in SEEDS
                    },
                }
            )
    return result


def compare_summary(recomputed: list[dict[str, object]], expected_path: Path) -> None:
    expected = read_csv(expected_path)
    if len(expected) != 36 or len(recomputed) != 36:
        fail("summary_row_count")
    numeric_fields = {
        "pair_count",
        "identity_count",
        "median_identity_effect",
        "median_identity_effect_rgb8",
        "ci95_identity_bootstrap_low",
        "ci95_identity_bootstrap_high",
        "positive_identity_count",
        "zero_identity_count",
        "negative_identity_count",
        "bootstrap_resamples",
        "bootstrap_seed",
        *(f"seed_{seed}_median_identity_effect" for seed in SEEDS),
    }
    for index, (observed, reference) in enumerate(zip(recomputed, expected)):
        if set(observed) != set(reference):
            fail(f"summary_schema:{index}")
        for field, observed_value in observed.items():
            reference_value = reference[field]
            if field in numeric_fields:
                if abs(float(observed_value) - float(reference_value)) > 1e-15:
                    fail(f"summary_value:{index}:{field}")
            elif str(observed_value) != reference_value:
                fail(f"summary_label:{index}:{field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sanitize-source", type=Path)
    parser.add_argument("--sanitize-output", type=Path)
    args = parser.parse_args()

    identity_path = args.identity
    if args.sanitize_source is not None:
        if args.sanitize_output is None:
            fail("sanitize_output_required")
        write_public_identity(args.sanitize_source, args.sanitize_output)
        identity_path = args.sanitize_output
    elif args.sanitize_output is not None:
        fail("sanitize_source_required")

    rows = load_public_rows(identity_path)
    recomputed = recompute(rows)
    compare_summary(recomputed, args.summary)
    print(
        json.dumps(
            {
                "status": "PASS_REALY_DIRECTIONAL_PUBLIC_RECOMPUTATION",
                "identity_rows": len(rows),
                "summary_rows": len(recomputed),
                "directions": 12,
                "comparisons": 3,
                "bootstrap_resamples_each": BOOTSTRAP_RESAMPLES,
                "new_significance_tests": False,
                "new_multiple_comparison_corrections": False,
                "identity_sha256": sha256(identity_path),
                "summary_sha256": sha256(args.summary),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
