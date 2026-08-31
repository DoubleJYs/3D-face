"""Predeclared identity-level inference without distributional assumptions.

Every effect is oriented so that a positive value favours the declared
reference route.  For lower-is-better metrics this is ``comparator -
reference``; for higher-is-better metrics it is ``reference - comparator``.
Samples are first paired and aggregated within identity and replicate;
replicate effects are then collapsed within identity.  Identities, never
samples or seeds, are the inferential unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
TERMINAL_STATES = frozenset(
    {"COMPLETE", "FAILED", "STRUCTURAL_NA", "SKIPPED_BY_CONTRACT"}
)
METRIC_DIRECTIONS = frozenset({"lower_is_better", "higher_is_better"})
FROZEN_METRIC_DIRECTIONS = {
    "hidden_uv_mae": "lower_is_better",
    "hidden_uv_psnr_db": "higher_is_better",
    "lpips": "lower_is_better",
    "sface_identity_cosine": "higher_is_better",
}


class StatisticalContractError(ValueError):
    """An input would change the frozen statistical unit or comparison."""


@dataclass(frozen=True)
class MetricRecord:
    dataset_id: str
    identity_token: str
    sample_id: str
    group_id: str
    replicate_id: str
    metric_id: str
    terminal_state: str
    value: float | None

    def __post_init__(self) -> None:
        for role in (
            "dataset_id",
            "identity_token",
            "sample_id",
            "group_id",
            "replicate_id",
            "metric_id",
        ):
            if SAFE_TOKEN.fullmatch(str(getattr(self, role))) is None:
                raise StatisticalContractError(f"invalid_safe_token:{role}")
        if self.terminal_state not in TERMINAL_STATES:
            raise StatisticalContractError("terminal_state_outside_contract")
        if self.terminal_state == "COMPLETE":
            if self.value is None or not math.isfinite(float(self.value)):
                raise StatisticalContractError("complete_metric_must_be_finite")
        elif self.value is not None:
            raise StatisticalContractError("noncomplete_metric_value_must_be_null")


@dataclass(frozen=True)
class IdentityEffect:
    identity_token: str
    replicate_count: int
    paired_sample_count: int
    effect: float


@dataclass(frozen=True)
class ComparisonResult:
    comparison_id: str
    family_id: str
    inferential: bool
    dataset_id: str
    metric_id: str
    metric_direction: str
    reference_group: str
    comparator_group: str
    effect_definition: str
    identity_count: int
    paired_sample_count: int
    requested_replicate_ids: tuple[str, ...]
    identity_effects: tuple[IdentityEffect, ...]
    median_effect: float
    mean_effect: float
    bootstrap_iterations: int
    bootstrap_seed: int
    bootstrap_ci95_low: float
    bootstrap_ci95_high: float
    favorable_identity_count: int
    neutral_identity_count: int
    unfavorable_identity_count: int
    nonzero_identity_count: int
    exact_sign_test_alternative: str
    exact_sign_test_p_raw: float
    statistical_unit: str = "identity"
    sample_aggregation: str = "within_identity_replicate_median"
    replicate_aggregation: str = "within_identity_median"
    complete_case_policy: str = "predeclared_paired_complete_intersection"
    outlier_removal: bool = False
    missing_value_imputation: bool = False
    distributional_assumption: str = "none_sign_test_and_identity_bootstrap"
    cross_dataset_pooling: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requested_replicate_ids"] = list(self.requested_replicate_ids)
        result["identity_effects"] = [asdict(row) for row in self.identity_effects]
        return result


def exact_one_sided_sign_p(positive: int, nonzero: int) -> float:
    """Return P[X >= positive], X~Binomial(nonzero, 0.5), exactly."""

    if not 0 <= positive <= nonzero:
        raise StatisticalContractError("sign_test_counts_invalid")
    if nonzero == 0:
        return 1.0
    numerator = sum(math.comb(nonzero, value) for value in range(positive, nonzero + 1))
    return float(numerator / (2**nonzero))


def holm_adjust(
    raw_p_values: Mapping[str, float], *, family_size: int | None = None
) -> dict[str, float]:
    """Holm adjustment against a fixed preregistered family size.

    Unavailable preregistered slots are treated as implicit p=1 entries for
    the step-down multiplicity factors.  They are not returned as fabricated
    numerical test results.
    """

    if not raw_p_values:
        raise StatisticalContractError("holm_family_empty")
    for key, value in raw_p_values.items():
        if SAFE_TOKEN.fullmatch(key) is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise StatisticalContractError("holm_input_invalid")
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered) if family_size is None else family_size
    if type(count) is not int or count < len(ordered) or count < 1:
        raise StatisticalContractError("holm_family_size_invalid")
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return {key: adjusted[key] for key in sorted(adjusted)}


def _unique_records(records: Iterable[MetricRecord]) -> tuple[MetricRecord, ...]:
    values = tuple(records)
    if not values:
        raise StatisticalContractError("metric_records_empty")
    keys = [
        (
            row.dataset_id,
            row.identity_token,
            row.sample_id,
            row.group_id,
            row.replicate_id,
            row.metric_id,
        )
        for row in values
    ]
    if len(keys) != len(set(keys)):
        raise StatisticalContractError("duplicate_metric_record")
    return values


def _bootstrap_median_ci(
    effects: np.ndarray, *, iterations: int, seed: int
) -> tuple[float, float]:
    if iterations != 10_000:
        raise StatisticalContractError("bootstrap_iterations_must_equal_10000")
    if effects.ndim != 1 or effects.size < 2 or not bool(np.isfinite(effects).all()):
        raise StatisticalContractError("bootstrap_identity_effects_invalid")
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, effects.size, size=(iterations, effects.size), endpoint=False
    )
    medians = np.median(effects[sample_indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975], method="linear")
    return float(low), float(high)


def run_comparison(
    records: Iterable[MetricRecord],
    *,
    comparison_id: str,
    family_id: str,
    inferential: bool,
    dataset_id: str,
    metric_id: str,
    metric_direction: str,
    reference_group: str,
    comparator_group: str,
    replicate_ids: Sequence[str],
    bootstrap_seed: int,
    expected_identity_count: int,
    bootstrap_iterations: int = 10_000,
) -> ComparisonResult:
    """Run one frozen paired comparison on the common COMPLETE intersection."""

    for token in (
        comparison_id,
        family_id,
        dataset_id,
        metric_id,
        reference_group,
        comparator_group,
    ):
        if SAFE_TOKEN.fullmatch(token) is None:
            raise StatisticalContractError("comparison_token_invalid")
    if type(inferential) is not bool:
        raise StatisticalContractError("inferential_flag_must_be_boolean")
    if metric_direction not in METRIC_DIRECTIONS:
        raise StatisticalContractError("metric_direction_outside_contract")
    expected_direction = FROZEN_METRIC_DIRECTIONS.get(metric_id)
    if expected_direction is None or metric_direction != expected_direction:
        raise StatisticalContractError("metric_direction_changed")
    replicates = tuple(replicate_ids)
    if not replicates or len(replicates) != len(set(replicates)):
        raise StatisticalContractError("replicate_ids_not_unique")
    if any(SAFE_TOKEN.fullmatch(value) is None for value in replicates):
        raise StatisticalContractError("replicate_id_invalid")
    values = _unique_records(records)
    if {row.dataset_id for row in values} != {dataset_id}:
        raise StatisticalContractError("cross_dataset_pooling_forbidden")
    if {row.metric_id for row in values} != {metric_id}:
        raise StatisticalContractError("metric_family_changed")
    allowed_groups = {reference_group, comparator_group}
    if not {row.group_id for row in values}.issubset(allowed_groups):
        raise StatisticalContractError("unrequested_group_in_comparison")
    if not {row.replicate_id for row in values}.issubset(set(replicates)):
        raise StatisticalContractError("unrequested_replicate_in_comparison")

    lookup = {
        (row.identity_token, row.sample_id, row.group_id, row.replicate_id): row
        for row in values
    }
    identities = sorted({row.identity_token for row in values})
    effects: list[IdentityEffect] = []
    for identity in identities:
        replicate_effects: list[float] = []
        paired_samples_total = 0
        identity_samples = sorted(
            {row.sample_id for row in values if row.identity_token == identity}
        )
        for replicate in replicates:
            sample_effects: list[float] = []
            for sample_id in identity_samples:
                reference = lookup.get((identity, sample_id, reference_group, replicate))
                comparator = lookup.get((identity, sample_id, comparator_group, replicate))
                if reference is None or comparator is None:
                    continue
                if reference.terminal_state != "COMPLETE" or comparator.terminal_state != "COMPLETE":
                    continue
                assert reference.value is not None and comparator.value is not None
                if metric_direction == "lower_is_better":
                    effect = float(comparator.value) - float(reference.value)
                else:
                    effect = float(reference.value) - float(comparator.value)
                sample_effects.append(effect)
            if sample_effects:
                replicate_effects.append(float(np.median(np.asarray(sample_effects))))
                paired_samples_total += len(sample_effects)
        if len(replicate_effects) != len(replicates):
            raise StatisticalContractError(
                f"identity_missing_complete_replicate_pair:{identity}"
            )
        effects.append(
            IdentityEffect(
                identity_token=identity,
                replicate_count=len(replicate_effects),
                paired_sample_count=paired_samples_total,
                effect=float(np.median(np.asarray(replicate_effects))),
            )
        )
    if len(effects) != expected_identity_count:
        raise StatisticalContractError("identity_count_changed")
    effect_values = np.asarray([row.effect for row in effects], dtype=np.float64)
    low, high = _bootstrap_median_ci(
        effect_values, iterations=bootstrap_iterations, seed=bootstrap_seed
    )
    positive = int(np.count_nonzero(effect_values > 0.0))
    negative = int(np.count_nonzero(effect_values < 0.0))
    neutral = int(effect_values.size - positive - negative)
    nonzero = positive + negative
    return ComparisonResult(
        comparison_id=comparison_id,
        family_id=family_id,
        inferential=inferential,
        dataset_id=dataset_id,
        metric_id=metric_id,
        metric_direction=metric_direction,
        reference_group=reference_group,
        comparator_group=comparator_group,
        effect_definition=(
            "comparator_minus_reference_positive_favors_reference"
            if metric_direction == "lower_is_better"
            else "reference_minus_comparator_positive_favors_reference"
        ),
        identity_count=len(effects),
        paired_sample_count=sum(row.paired_sample_count for row in effects),
        requested_replicate_ids=replicates,
        identity_effects=tuple(effects),
        median_effect=float(np.median(effect_values)),
        mean_effect=float(np.mean(effect_values, dtype=np.float64)),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        bootstrap_ci95_low=low,
        bootstrap_ci95_high=high,
        favorable_identity_count=positive,
        neutral_identity_count=neutral,
        unfavorable_identity_count=negative,
        nonzero_identity_count=nonzero,
        exact_sign_test_alternative="reference_outperforms_comparator_in_declared_metric_direction",
        exact_sign_test_p_raw=exact_one_sided_sign_p(positive, nonzero),
    )


__all__ = [
    "ComparisonResult",
    "IdentityEffect",
    "MetricRecord",
    "StatisticalContractError",
    "exact_one_sided_sign_p",
    "holm_adjust",
    "run_comparison",
]
