#!/usr/bin/env python3
"""Recompute the 18 public identity-level comparisons and verify frozen outputs.

This entry point intentionally starts from the anonymous identity effects.  It
does not read pair-level metrics, images, checkpoints, or input bindings.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = ROOT / "IDENTITY_EFFECTS.csv"
FAMILY_PATH = ROOT / "FAMILY_RESULTS.csv"
RESAMPLES = 10_000
BASE_SEED = 20_260_816
TOLERANCE = 1e-12


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    estimates: list[float] = []
    for _ in range(RESAMPLES):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(statistics.median(sample)))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    effective_n = positive + negative
    if effective_n == 0:
        return 1.0
    smaller = min(positive, negative)
    tail = sum(math.comb(effective_n, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**effective_n))


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    family_size = len(p_values)
    for rank, (original_index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * p_value))
        adjusted[original_index] = running
    return adjusted


def close(actual: float, expected: str, label: str, failures: list[str]) -> None:
    target = float(expected)
    if not math.isclose(actual, target, rel_tol=0.0, abs_tol=TOLERANCE):
        failures.append(f"{label}: expected={target!r}, recomputed={actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-csv",
        type=Path,
        help="optional path for the recomputed comparison table",
    )
    args = parser.parse_args()

    with IDENTITY_PATH.open(newline="", encoding="utf-8") as handle:
        identity_rows = list(csv.DictReader(handle))
    with FAMILY_PATH.open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in identity_rows:
        grouped[row["comparison_id"]].append(float(row["identity_effect"]))

    if len(expected_rows) != 18 or set(grouped) != {row["comparison_id"] for row in expected_rows}:
        raise SystemExit("FAIL: public comparison key space is not the frozen 18-member design")

    recomputed: list[dict[str, object]] = []
    failures: list[str] = []
    for serial, expected in enumerate(expected_rows):
        comparison_id = expected["comparison_id"]
        values = grouped[comparison_id]
        positive = sum(value > 0 for value in values)
        zero = sum(value == 0 for value in values)
        negative = sum(value < 0 for value in values)
        median = float(statistics.median(values))
        ci_low, ci_high = bootstrap_interval(values, BASE_SEED + serial)
        p_raw = exact_two_sided_sign_p(positive, negative)
        recomputed.append(
            {
                "family_id": expected["family_id"],
                "comparison_id": comparison_id,
                "identity_count": len(values),
                "median_identity_effect": median,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_positive": positive,
                "n_zero": zero,
                "n_negative": negative,
                "p_raw_two_sided_exact_sign": p_raw,
            }
        )
        if len(values) != int(expected["identity_count"]):
            failures.append(f"{comparison_id}: identity_count")
        if (positive, zero, negative) != (
            int(expected["n_positive"]),
            int(expected["n_zero"]),
            int(expected["n_negative"]),
        ):
            failures.append(f"{comparison_id}: sign counts")
        close(median, expected["median_identity_effect"], f"{comparison_id}:median", failures)
        close(ci_low, expected["ci95_low"], f"{comparison_id}:ci_low", failures)
        close(ci_high, expected["ci95_high"], f"{comparison_id}:ci_high", failures)
        close(p_raw, expected["p_raw_two_sided_exact_sign"], f"{comparison_id}:p_raw", failures)

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(recomputed):
        by_family[str(row["family_id"])].append(index)
    for indexes in by_family.values():
        adjusted = holm_adjust(
            [float(recomputed[index]["p_raw_two_sided_exact_sign"]) for index in indexes]
        )
        for index, p_holm in zip(indexes, adjusted):
            recomputed[index]["p_holm_within_family"] = p_holm
            expected = expected_rows[index]
            close(
                p_holm,
                expected["p_holm_within_family"],
                f"{expected['comparison_id']}:p_holm",
                failures,
            )
            median = float(recomputed[index]["median_identity_effect"])
            ci_low = float(recomputed[index]["ci95_low"])
            ci_high = float(recomputed[index]["ci95_high"])
            eligible = expected["confirmatory_coverage_eligible"] == "True"
            favorable = eligible and median > 0 and ci_low > 0 and p_holm < 0.05
            unfavorable = eligible and median < 0 and ci_high < 0 and p_holm < 0.05
            indeterminate = not (favorable or unfavorable)
            actual_flags = (favorable, unfavorable, indeterminate)
            expected_flags = tuple(
                expected[name] == "True"
                for name in (
                    "confirmatory_full_favorable",
                    "confirmatory_full_unfavorable",
                    "confirmatory_indeterminate",
                )
            )
            if actual_flags != expected_flags:
                failures.append(f"{expected['comparison_id']}: claim flags")

    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))

    if args.write_csv:
        args.write_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(recomputed[0])
        with args.write_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(recomputed)

    print(
        "PASS: 18 identity-level comparisons, 10,000-resample confidence intervals, "
        "two-sided exact sign tests, four Holm families, and claim flags match the frozen results."
    )


if __name__ == "__main__":
    main()
