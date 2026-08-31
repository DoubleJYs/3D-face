#!/usr/bin/env python3
"""One-shot post-F2 FaceScape UV metric and invariant closure.

The target for each source frame is the anonymous opposite-view ``partial_uv``.
The common hidden support is exactly

``canonical & ~source_visible & paired_target_visible``.

The runner never exposes original identity identifiers or source paths.  It
retains exactly one of COMPLETE, FAILED, STRUCTURAL_NA, or
SKIPPED_BY_CONTRACT for every raw route/sample key and emits the primary
hidden-UV MAE rows in the exact shape consumed by the identity-statistics
module.  PSNR and the complete invariant/decomposition audit remain in the
descriptive audit ledger; PSNR is intentionally not passed to the current
lower-is-better error-statistics runner.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from frugalface3d.evaluation.masked_uv_metrics import (  # noqa: E402
    masked_rgb_psnr,
)
from reproducibility.w5b49n_mechanism_closure_v1.core.invariants import (  # noqa: E402
    evaluate_native_conserved_mae,
    verify_where_invariants,
)
from reproducibility.w5b49n_mechanism_closure_v1.metrics.context_contract_v1 import (  # noqa: E402
    facescape_context_binding,
    secondary_metric_failure_contract,
)
from reproducibility.w5b49n_mechanism_closure_v1.runtime.eval_cache_io import (  # noqa: E402
    load_eval_cache,
)
from reproducibility.w5b49n_mechanism_closure_v1.statistics.identity_inference import (  # noqa: E402
    MetricRecord,
)
from reproducibility.w5b49n_mechanism_closure_v1.training.cache_io import (  # noqa: E402
    canonical_json_bytes,
    pair_and_donor_maps,
    sha256_file,
    write_json,
)


PROGRAM_ID = "FRUGALFACE3D-MECHANISM-CLOSURE-V1-FACESCAPE-METRIC"
METRIC_SCHEMA_VERSION = "frugalface3d.w5b49n.facescape_metric_audit.v1"
NORMALIZED_SCHEMA_VERSION = "frugalface3d.w5b49n.identity_inference_input.v1"
TERMINAL_SCHEMA_VERSION = "frugalface3d.w5b49n.facescape_metric_terminal.v1"
EXPECTED_DATASET = "D1"
EXPECTED_SAMPLE_COUNT = 160
EXPECTED_IDENTITY_COUNT = 20
PRIMARY_METRIC_ID = "hidden_uv_mae"
TERMINAL_STATES = frozenset(
    {"COMPLETE", "FAILED", "STRUCTURAL_NA", "SKIPPED_BY_CONTRACT"}
)
RAW_SUCCESS_STATES = frozenset({"SUCCESS", "COMPLETE"})
SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SEEDS = (2026080447, 2026080448, 2026080449)
VARIANTS = (
    "Full",
    "XYZ0",
    "Normal0",
    "Expression0",
    "XYZNormal0",
    "GateEqual",
    "CAConv",
    "ETCPlain",
)
COMPONENT_VARIANTS = (
    "XYZ0",
    "Normal0",
    "Expression0",
    "GateEqual",
    "CAConv",
    "ETCPlain",
)
NEGATIVE_CONTROL_METHODS = ("median_fill", "nearest_fill")


class MetricClosureError(ValueError):
    """A post-F2 input or metric invariant changed from the fixed contract."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise MetricClosureError(code)


def _safe_token(value: Any, role: str) -> str:
    result = str(value)
    _require(SAFE_TOKEN.fullmatch(result) is not None, f"unsafe_token:{role}")
    return result


def _mask(value: Any, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (1, 64, 64):
        array = array[0]
    _require(array.shape == (64, 64), f"mask_shape:{role}")
    _require(bool(np.isfinite(array).all()), f"mask_nonfinite:{role}")
    _require(bool(np.all((array == 0) | (array == 1))), f"mask_not_binary:{role}")
    return np.ascontiguousarray(array.astype(bool, copy=False))


def _uv(value: Any, role: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.dtype == np.dtype(np.float32), f"uv_dtype:{role}")
    _require(array.shape == (3, 64, 64), f"uv_shape:{role}")
    _require(bool(np.isfinite(array).all()), f"uv_nonfinite:{role}")
    _require(
        not bool(np.any((array < 0.0) | (array > 1.0))),
        f"uv_outside_unit_interval:{role}",
    )
    return np.ascontiguousarray(array)


def _null_metrics() -> dict[str, Any]:
    return {
        "native": None,
        "conserved": None,
        "native_conserved_hidden_mae_delta": None,
        "native_conserved_hidden_mae_exactly_unchanged": None,
        "native_conserved_hidden_psnr_exactly_unchanged": None,
    }


def evaluate_one_sample(
    *,
    native: np.ndarray,
    conserved: np.ndarray,
    source_partial: np.ndarray,
    source_visible: np.ndarray,
    canonical: np.ndarray,
    paired_target_partial: np.ndarray,
    paired_target_visible: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one already-frozen native/conserved pair.

    Candidate generation is already complete.  The paired target is admitted
    only in this function, at the final metric boundary.
    """

    native_uv = _uv(native, "native")
    conserved_uv = _uv(conserved, "conserved")
    observed_uv = _uv(source_partial, "source_partial")
    reference_uv = _uv(paired_target_partial, "paired_target_partial")
    observed = _mask(source_visible, "source_visible")
    target_visible = _mask(paired_target_visible, "paired_target_visible")
    canonical_mask = _mask(canonical, "canonical")
    _require(
        not bool(np.any(observed & ~canonical_mask)), "source_visible_outside_canonical"
    )
    _require(
        not bool(np.any(target_visible & ~canonical_mask)),
        "paired_target_visible_outside_canonical",
    )
    _require(
        bool(
            np.array_equal(
                observed_uv[:, ~observed], np.zeros_like(observed_uv[:, ~observed])
            )
        ),
        "source_partial_hidden_not_exact_zero",
    )
    _require(
        bool(
            np.array_equal(
                reference_uv[:, ~target_visible],
                np.zeros_like(reference_uv[:, ~target_visible]),
            )
        ),
        "paired_target_partial_hidden_not_exact_zero",
    )

    where = verify_where_invariants(
        native_uv_chw_float32=native_uv,
        observed_uv_chw_float32=observed_uv,
        observed_mask_bool=observed,
        conserved_uv_chw_float32=conserved_uv,
    )
    _require(where.passed, "native_conserved_where_invariant_failed")
    common_hidden = canonical_mask & ~observed & target_visible
    evaluable = canonical_mask & target_visible
    observed_evaluable = evaluable & observed
    support = {
        "formula": "canonical_AND_NOT_source_visible_AND_paired_target_visible",
        "canonical_texels": int(canonical_mask.sum()),
        "target_visible_texels": int(target_visible.sum()),
        "evaluable_texels": int(evaluable.sum()),
        "observed_evaluable_texels": int(observed_evaluable.sum()),
        "common_hidden_texels": int(common_hidden.sum()),
    }
    invariant = {
        "operator_id": where.operator_id,
        "exact_boolean_where": where.exact_boolean_where,
        "observed_region_exact": where.observed_region_exact,
        "hidden_region_exact": where.hidden_region_exact,
        "observed_max_absolute_error": where.observed_max_absolute_error,
        "hidden_max_absolute_change": where.hidden_max_absolute_change,
        "passed": where.passed,
    }
    if not bool(common_hidden.any()):
        return {
            "terminal_state": "STRUCTURAL_NA",
            "failure_code": "EMPTY_COMMON_HIDDEN_SUPPORT",
            "support": support,
            "where_invariant": invariant,
            "metrics": _null_metrics(),
        }
    _require(bool(observed_evaluable.any()), "empty_observed_evaluable_support")

    paired = evaluate_native_conserved_mae(
        native_uv_chw_float32=native_uv,
        conserved_uv_chw_float32=conserved_uv,
        reference_uv_chw_float32=reference_uv,
        observed_mask_bool=observed,
        canonical_mask_bool=evaluable,
    )
    _require(
        paired.hidden_mae_exactly_unchanged and paired.hidden_mae_delta == 0.0,
        "native_conserved_hidden_mae_changed",
    )
    native_psnr = masked_rgb_psnr(native_uv, reference_uv, common_hidden)
    conserved_psnr = masked_rgb_psnr(conserved_uv, reference_uv, common_hidden)
    psnr_equal = bool(
        native_psnr.perfect_match == conserved_psnr.perfect_match
        and native_psnr.value_db == conserved_psnr.value_db
        and native_psnr.support_texels == conserved_psnr.support_texels
    )
    _require(psnr_equal, "native_conserved_hidden_psnr_changed")

    def values(mode: str, decomposition: Any, psnr: Any) -> dict[str, Any]:
        return {
            "mode": mode,
            "hidden_uv_mae": float(decomposition.hidden_mae),
            "hidden_uv_psnr_db": (
                None if psnr.value_db is None else float(psnr.value_db)
            ),
            "hidden_uv_psnr_perfect_match": bool(psnr.perfect_match),
            "support_mae": float(decomposition.full_mae),
            "observed_evaluable_mae": float(decomposition.observed_mae),
            "decomposition": {
                "observed_fraction": float(decomposition.observed_fraction),
                "hidden_fraction": float(decomposition.hidden_fraction),
                "recomposed_support_mae": float(decomposition.recomposed_mae),
                "absolute_residual": float(decomposition.absolute_residual),
                "dtype": "float64",
                "identity_verified": bool(decomposition.absolute_residual <= 1e-7),
            },
        }

    return {
        "terminal_state": "COMPLETE",
        "failure_code": None,
        "support": support,
        "where_invariant": invariant,
        "metrics": {
            "native": values("native", paired.native, native_psnr),
            "conserved": values("conserved", paired.conserved, conserved_psnr),
            "native_conserved_hidden_mae_delta": float(paired.hidden_mae_delta),
            "native_conserved_hidden_mae_exactly_unchanged": True,
            "native_conserved_hidden_psnr_exactly_unchanged": True,
        },
    }


def _sample_id(row: Mapping[str, Any]) -> str:
    return _safe_token(
        f"{row['dataset_token']}:{row['identity_token']}:"
        f"{row['expression_token']}:{row['view_token']}",
        "sample_id",
    )


def _read_json(path: Path, role: str) -> dict[str, Any]:
    _require(not path.is_symlink() and path.is_file(), f"regular_json_required:{role}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                MetricClosureError(f"nonfinite_json:{role}:{item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MetricClosureError(f"invalid_json:{role}") from error
    _require(isinstance(value, dict), f"json_object_required:{role}")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
    temporary.replace(path)
    return sha256_file(path)


def _route_group(receipt: Mapping[str, Any], mode: str) -> tuple[str, str]:
    method = _safe_token(receipt.get("method_id"), "method_id")
    variant_raw = receipt.get("variant")
    base = _safe_token(variant_raw, "variant") if variant_raw is not None else method
    intervention = str(receipt.get("intervention") or "aligned")
    intervention = "spatial_shuffle" if intervention == "shuffle" else intervention
    intervention = _safe_token(intervention, "intervention")
    if method in NEGATIVE_CONTROL_METHODS and intervention == method:
        return _safe_token(f"{method}__{mode}", "group_id"), intervention
    suffix = "" if intervention == "aligned" else f"__{intervention}"
    return _safe_token(f"{base}{suffix}__{mode}", "group_id"), intervention


def _replicate_id(receipt: Mapping[str, Any]) -> str:
    seed = receipt.get("seed")
    if seed is None:
        return "fixed"
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed_not_integer")
    return _safe_token(f"S{seed}", "replicate_id")


def _normalized_replicates(
    receipt: Mapping[str, Any], mode: str
) -> tuple[str, ...]:
    """Align fixed D1 controls to the three preregistered Full replicates."""

    replicate = _replicate_id(receipt)
    method = str(receipt.get("method_id"))
    if (
        replicate == "fixed"
        and mode == "conserved"
        and method in {"b_lite", *NEGATIVE_CONTROL_METHODS}
    ):
        return tuple(f"S{seed}" for seed in SEEDS)
    return (replicate,)


def _normalized_row(
    *,
    dataset_id: str,
    identity_token: str,
    sample_id: str,
    group_id: str,
    replicate_id: str,
    terminal_state: str,
    value: float | None,
) -> dict[str, Any]:
    row = MetricRecord(
        dataset_id=dataset_id,
        identity_token=identity_token,
        sample_id=sample_id,
        group_id=group_id,
        replicate_id=replicate_id,
        metric_id=PRIMARY_METRIC_ID,
        terminal_state=terminal_state,
        value=value,
    )
    return {name: getattr(row, name) for name in MetricRecord.__dataclass_fields__}


def _comparison_plan(groups: set[str]) -> list[dict[str, Any]]:
    reference = "Full__conserved"
    required = {reference}
    comparisons: list[dict[str, Any]] = []
    for comparator_base in COMPONENT_VARIANTS:
        comparator = f"{comparator_base}__conserved"
        required.add(comparator)
        comparisons.append(
            {
                "comparison_id": f"Full_vs_{comparator_base}_conserved",
                "family_id": "component",
                "inferential": True,
                "reference_group": reference,
                "comparator_group": comparator,
                "replicate_ids": [f"S{seed}" for seed in SEEDS],
            }
        )

    mechanism = (
        ("Full_vs_B_lite_conserved", "b_lite__conserved"),
        ("Full_vs_median_fill_conserved", "median_fill__conserved"),
        ("Full_vs_nearest_fill_conserved", "nearest_fill__conserved"),
        (
            "Full_aligned_vs_spatial_shuffle_conserved",
            "Full__spatial_shuffle__conserved",
        ),
    )
    for comparison_id, comparator in mechanism:
        required.add(comparator)
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "family_id": "mechanism",
                "inferential": True,
                "reference_group": reference,
                "comparator_group": comparator,
                "replicate_ids": [f"S{seed}" for seed in SEEDS],
            }
        )

    exploratory = (
        ("Full_vs_XYZNormal0_conserved", "XYZNormal0__conserved"),
        (
            "Full_aligned_vs_cross_identity_conserved",
            "Full__cross_identity__conserved",
        ),
    )
    for comparison_id, comparator in exploratory:
        required.add(comparator)
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "family_id": "exploratory",
                "inferential": False,
                "reference_group": reference,
                "comparator_group": comparator,
                "replicate_ids": [f"S{seed}" for seed in SEEDS],
            }
        )
    _require(required.issubset(groups), "full_family_comparison_groups_missing")
    return comparisons


def _raw_ledger(
    raw_root: Path, terminal: Mapping[str, Any], route_ids: Sequence[str]
) -> dict[tuple[str, int], Mapping[str, Any]]:
    ledger_path = raw_root / "SAMPLE_TERMINAL_LEDGER.jsonl"
    _require(
        not ledger_path.is_symlink() and ledger_path.is_file(), "raw_ledger_missing"
    )
    expected_sha = terminal.get("sample_terminal_ledger_sha256")
    _require(
        isinstance(expected_sha, str)
        and HEX64.fullmatch(expected_sha) is not None
        and sha256_file(ledger_path) == expected_sha,
        "raw_ledger_sha256_mismatch",
    )
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetricClosureError(
                    f"raw_ledger_invalid_json:{line_number}"
                ) from error
            _require(isinstance(row, dict), "raw_ledger_row_not_object")
            route_id = _safe_token(row.get("route_id"), "raw_ledger_route_id")
            index = row.get("sample_index")
            _require(
                isinstance(index, int) and 0 <= index < EXPECTED_SAMPLE_COUNT,
                "raw_ledger_index",
            )
            key = (route_id, index)
            _require(key not in result, "raw_ledger_duplicate_key")
            result[key] = row
    expected = {
        (route_id, index)
        for route_id in route_ids
        for index in range(EXPECTED_SAMPLE_COUNT)
    }
    _require(set(result) == expected, "raw_ledger_terminal_keyspace_changed")
    return result


def _upstream_state(value: Any) -> str:
    state = str(value)
    if state in RAW_SUCCESS_STATES:
        return "COMPLETE"
    if state in TERMINAL_STATES:
        return state
    if state in {"ABORT", "STRUCTURAL_N/A"}:
        return "STRUCTURAL_NA"
    return "FAILED"


def _failed_audit(
    *,
    route_id: str,
    receipt: Mapping[str, Any],
    row: Mapping[str, Any],
    paired_row: Mapping[str, Any],
    state: str,
    failure_code: str,
) -> dict[str, Any]:
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "dataset_id": EXPECTED_DATASET,
        "identity_token": _safe_token(row["identity_token"], "identity_token"),
        "sample_id": _sample_id(row),
        "paired_target_sample_id": _sample_id(paired_row),
        "route_id": route_id,
        "method_id": _safe_token(receipt.get("method_id"), "method_id"),
        "variant": receipt.get("variant"),
        "seed": receipt.get("seed"),
        "intervention": receipt.get("intervention"),
        "replicate_id": _replicate_id(receipt),
        "terminal_state": state,
        "failure_code": _safe_token(failure_code, "failure_code"),
        "candidate_generation_completed_before_target_access": True,
        "target_used_only_at_final_metric_boundary": True,
        "statistical_unit": "identity",
        "support": None,
        "where_invariant": None,
        "metrics": _null_metrics(),
    }


def _route_arrays(raw_root: Path, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch

    route_id = _safe_token(receipt.get("route_id"), "route_id")
    route_dir = raw_root / "routes" / route_id
    _require(
        route_dir.is_dir() and not route_dir.is_symlink(), "route_directory_missing"
    )
    route_terminal = _read_json(route_dir / "ROUTE_TERMINAL.json", "route_terminal")
    _require(route_terminal == receipt, "embedded_route_terminal_mismatch")
    output_name = route_terminal.get("raw_output_file")
    _require(output_name == "RAW_OUTPUTS.pt", "route_output_name_changed")
    payload_path = route_dir / output_name
    expected_sha = route_terminal.get("raw_output_sha256")
    _require(
        isinstance(expected_sha, str)
        and HEX64.fullmatch(expected_sha) is not None
        and sha256_file(payload_path) == expected_sha,
        "route_output_sha256_mismatch",
    )
    arrays = torch.load(payload_path, map_location="cpu", weights_only=True)
    _require(
        isinstance(arrays, dict) and set(arrays) == {"native", "conserved"},
        "route_tensor_keyspace",
    )
    for role in ("native", "conserved"):
        value = arrays[role]
        _require(
            value.dtype == torch.float32
            and tuple(value.shape) == (EXPECTED_SAMPLE_COUNT, 3, 64, 64)
            and bool(torch.isfinite(value).all())
            and float(value.min()) >= 0.0
            and float(value.max()) <= 1.0,
            f"route_tensor_contract:{role}",
        )
    return arrays


def _load_raw_campaign(
    raw_root: Path,
    *,
    expected_status: str,
    expected_context: Mapping[str, Any],
    role: str,
) -> tuple[Path, dict[str, Any], list[Mapping[str, Any]], dict[tuple[str, int], Mapping[str, Any]]]:
    root = raw_root.expanduser().resolve(strict=True)
    terminal_path = root / "RAW_TERMINAL.json"
    terminal = _read_json(terminal_path, f"{role}_terminal")
    _require(terminal.get("status") == expected_status, f"{role}_not_f2_complete")
    _require(
        terminal.get("sample_count_per_route") == EXPECTED_SAMPLE_COUNT
        and terminal.get("paper_metrics_accessed") is False
        and terminal.get("retry_count") == 0
        and terminal.get("immutable_context_binding") == expected_context,
        f"{role}_semantics_changed",
    )
    receipts = terminal.get("routes")
    _require(isinstance(receipts, list) and receipts, f"{role}_receipts_missing")
    route_ids = [_safe_token(row.get("route_id"), "route_id") for row in receipts]
    _require(len(route_ids) == len(set(route_ids)), f"{role}_route_ids_not_unique")
    _require(terminal.get("route_count") == len(route_ids), f"{role}_route_count")
    for receipt in receipts:
        _require(
            receipt.get("immutable_context_binding") == expected_context,
            f"{role}_route_context_binding_changed",
        )
    return root, terminal, receipts, _raw_ledger(root, terminal, route_ids)


def execute(
    eval_cache_root: Path,
    raw_root: Path,
    negative_control_raw_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Consume both F2-frozen D1 raw campaigns exactly once."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("metric_output_root_exists_no_rerun")
    cache = load_eval_cache(eval_cache_root)
    context_binding = facescape_context_binding(cache)
    main_root, main_terminal, main_receipts, main_ledger = _load_raw_campaign(
        raw_root,
        expected_status="PASS_FACESCAPE_FULL_FAMILY_RAW_OUTPUTS_FROZEN_READY",
        expected_context=context_binding,
        role="full_family_raw",
    )
    negative_root, negative_terminal, negative_receipts, negative_ledger = (
        _load_raw_campaign(
            negative_control_raw_root,
            expected_status="PASS_FACESCAPE_NEGATIVE_CONTROLS_F2_READY",
            expected_context=context_binding,
            role="negative_control_raw",
        )
    )
    _require(
        {str(row.get("method_id")) for row in negative_receipts}
        == set(NEGATIVE_CONTROL_METHODS),
        "negative_control_method_set_changed",
    )
    campaigns = (
        (main_root, main_receipts, main_ledger),
        (negative_root, negative_receipts, negative_ledger),
    )
    all_receipts = [*main_receipts, *negative_receipts]
    all_route_ids = [str(row["route_id"]) for row in all_receipts]
    _require(len(all_route_ids) == len(set(all_route_ids)), "cross_campaign_route_duplicate")
    pairs, _donors = pair_and_donor_maps(cache.rows, list(range(EXPECTED_SAMPLE_COUNT)))

    output_root.mkdir(parents=True, mode=0o700)
    write_json(
        output_root / "ATTEMPT.json",
        {
            "status": "STARTED_POST_F2_FINAL_METRIC_ONCE",
            "program_id": PROGRAM_ID,
            "automatic_retry": False,
            "maximum_attempts_per_route_sample": 1,
            "raw_terminal_sha256": sha256_file(main_root / "RAW_TERMINAL.json"),
            "negative_control_raw_terminal_sha256": sha256_file(
                negative_root / "RAW_TERMINAL.json"
            ),
            "eval_cache_tensor_sha256": cache.manifest["tensor_sha256"],
            "immutable_context_binding": context_binding,
        },
    )
    audits: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    try:
        for campaign_root, receipts, raw_ledger in campaigns:
            for receipt in receipts:
                route_id = _safe_token(receipt.get("route_id"), "route_id")
                replicate = _replicate_id(receipt)
                route_arrays: Mapping[str, Any] | None = None
                route_error = False
                try:
                    route_arrays = _route_arrays(campaign_root, receipt)
                except Exception:
                    route_error = True
                for index, row in enumerate(cache.rows):
                    paired_row = cache.rows[pairs[index]]
                    raw_sample_terminal = raw_ledger[(route_id, index)]
                    _require(
                        raw_sample_terminal.get("sample_token") == _sample_id(row)
                        and raw_sample_terminal.get("tensor_index") in {None, index},
                        "raw_ledger_sample_binding_changed",
                    )
                    upstream = _upstream_state(
                        raw_sample_terminal.get("terminal_state")
                    )
                    if upstream == "COMPLETE":
                        _require(
                            raw_sample_terminal.get("tensor_index") == index
                            and raw_sample_terminal.get("output_sha256")
                            == receipt.get("raw_output_sha256"),
                            "raw_ledger_route_artifact_binding_changed",
                        )
                    if route_error:
                        upstream = "FAILED"
                        failure_code = "ROUTE_ARTIFACT_CONTRACT_FAILED"
                    elif upstream != "COMPLETE":
                        failure_code = f"UPSTREAM_{upstream}"
                    else:
                        failure_code = "METRIC_RUNTIME_CONTRACT_FAILED"
                    if upstream != "COMPLETE":
                        audit = _failed_audit(
                            route_id=route_id,
                            receipt=receipt,
                            row=row,
                            paired_row=paired_row,
                            state=upstream,
                            failure_code=failure_code,
                        )
                    else:
                        assert route_arrays is not None
                        try:
                            result = evaluate_one_sample(
                                native=route_arrays["native"][index].numpy(),
                                conserved=route_arrays["conserved"][index].numpy(),
                                source_partial=cache.tensors["partial_uv"][index].numpy(),
                                source_visible=cache.tensors["visibility"][index].numpy(),
                                canonical=cache.tensors["canonical_mask"][index].numpy(),
                                paired_target_partial=cache.tensors["partial_uv"][
                                    pairs[index]
                                ].numpy(),
                                paired_target_visible=cache.tensors["visibility"][
                                    pairs[index]
                                ].numpy(),
                            )
                            audit = {
                                "schema_version": METRIC_SCHEMA_VERSION,
                                "dataset_id": EXPECTED_DATASET,
                                "identity_token": _safe_token(
                                    row["identity_token"], "identity_token"
                                ),
                                "sample_id": _sample_id(row),
                                "paired_target_sample_id": _sample_id(paired_row),
                                "route_id": route_id,
                                "method_id": _safe_token(
                                    receipt.get("method_id"), "method_id"
                                ),
                                "variant": receipt.get("variant"),
                                "seed": receipt.get("seed"),
                                "intervention": receipt.get("intervention"),
                                "replicate_id": replicate,
                                "terminal_state": result["terminal_state"],
                                "failure_code": result["failure_code"],
                                "candidate_generation_completed_before_target_access": True,
                                "target_used_only_at_final_metric_boundary": True,
                                "statistical_unit": "identity",
                                "support": result["support"],
                                "where_invariant": result["where_invariant"],
                                "metrics": result["metrics"],
                            }
                        except Exception:
                            audit = _failed_audit(
                                route_id=route_id,
                                receipt=receipt,
                                row=row,
                                paired_row=paired_row,
                                state="FAILED",
                                failure_code="METRIC_RUNTIME_CONTRACT_FAILED",
                            )
                    audits.append(audit)
                    state = str(audit["terminal_state"])
                    _require(state in TERMINAL_STATES, "audit_terminal_state_invalid")
                    for mode in ("native", "conserved"):
                        group_id, _intervention = _route_group(receipt, mode)
                        metric = audit["metrics"][mode]
                        value = (
                            float(metric["hidden_uv_mae"])
                            if state == "COMPLETE" and metric is not None
                            else None
                        )
                        for normalized_replicate in _normalized_replicates(
                            receipt, mode
                        ):
                            normalized.append(
                                _normalized_row(
                                    dataset_id=EXPECTED_DATASET,
                                    identity_token=str(audit["identity_token"]),
                                    sample_id=str(audit["sample_id"]),
                                    group_id=group_id,
                                    replicate_id=normalized_replicate,
                                    terminal_state=state,
                                    value=value,
                                )
                            )

        expected_audits = len(all_receipts) * EXPECTED_SAMPLE_COUNT
        _require(len(audits) == expected_audits, "metric_audit_count_changed")
        keys = [(row["route_id"], row["sample_id"]) for row in audits]
        _require(len(keys) == len(set(keys)), "metric_audit_duplicate_key")
        normalized_keys = [
            (
                row["dataset_id"],
                row["identity_token"],
                row["sample_id"],
                row["group_id"],
                row["replicate_id"],
                row["metric_id"],
            )
            for row in normalized
        ]
        _require(
            len(normalized_keys) == len(set(normalized_keys)),
            "normalized_metric_duplicate_key",
        )
        groups = {str(row["group_id"]) for row in normalized}
        comparisons = _comparison_plan(groups)
        audit_sha = _write_jsonl(output_root / "METRIC_AUDIT_ROWS.jsonl", audits)
        normalized_sha = _write_jsonl(
            output_root / "NORMALIZED_METRIC_ROWS.jsonl", normalized
        )
        statistics_input = {
            "schema_version": NORMALIZED_SCHEMA_VERSION,
            "dataset_id": EXPECTED_DATASET,
            "metric_id": PRIMARY_METRIC_ID,
            "metric_direction": "lower_is_better",
            "bootstrap_seed": 20260816,
            "bootstrap_iterations": 10_000,
            "expected_identity_count": EXPECTED_IDENTITY_COUNT,
            "rows": normalized,
            "comparisons": comparisons,
        }
        write_json(output_root / "NORMALIZED_METRIC_ROWS.json", statistics_input)
        state_counts = Counter(str(row["terminal_state"]) for row in audits)
        invariant_failures = sum(
            row["terminal_state"] == "COMPLETE"
            and (
                row["where_invariant"] is None
                or row["where_invariant"]["passed"] is not True
            )
            for row in audits
        )
        sample_failures = (
            state_counts.get("FAILED", 0)
            + state_counts.get("SKIPPED_BY_CONTRACT", 0)
            + invariant_failures
        )
        mc0_complete = (
            context_binding.get("mc0_geometry_state_status")
            == "PASS_ALL_FOUR_CONTEXT_ROLES_REFERENCE_EXACT"
            and context_binding.get("input_output_context_reference_exact") is True
            and context_binding.get("geometry_updates_or_outputs") == 0
        )
        if sample_failures or not mc0_complete:
            status = "TERMINAL_POST_F2_FACESCAPE_METRIC_LEDGER_WITH_RETAINED_FAILURES"
        else:
            status = "PASS_POST_F2_FACESCAPE_METRIC_LEDGER_COMPLETE"
        secondary = secondary_metric_failure_contract(
            dataset_id=EXPECTED_DATASET,
            render_context_ready=True,
            reason_suffix="SEPARATE_POST_F2_SECONDARY_RUNNER_NOT_EXECUTED",
        )
        family_counts = Counter(str(row["family_id"]) for row in comparisons)
        terminal = {
            "schema_version": TERMINAL_SCHEMA_VERSION,
            "status": status,
            "program_id": PROGRAM_ID,
            "dataset_id": EXPECTED_DATASET,
            "route_count": len(all_receipts),
            "main_raw_route_count": len(main_receipts),
            "negative_control_route_count": len(negative_receipts),
            "sample_count_per_route": EXPECTED_SAMPLE_COUNT,
            "audit_row_count": len(audits),
            "hidden_uv_mae_audit_row_count": len(audits),
            "hidden_uv_psnr_audit_row_count": len(audits),
            "normalized_primary_metric_row_count": len(normalized),
            "terminal_state_counts": {
                state: int(state_counts.get(state, 0))
                for state in sorted(TERMINAL_STATES)
            },
            "all_four_terminal_states_retained_by_schema": True,
            "failure_rows_dropped": 0,
            "retry_count": 0,
            "imputation_count": 0,
            "identity_is_only_statistical_unit": True,
            "statistics_executed": False,
            "cross_dataset_pooling": False,
            "comparison_family_counts": dict(sorted(family_counts.items())),
            "exploratory_comparisons_in_confirmatory_holm": False,
            "fixed_controls_reused_across_preregistered_full_seeds": True,
            "common_hidden_support_formula": "canonical_AND_NOT_source_visible_AND_paired_target_visible",
            "paired_target_source": "anonymous_opposite_view_partial_uv",
            "target_used_only_at_final_metric_boundary": True,
            "native_conserved_invariant_failure_count": invariant_failures,
            "immutable_context_binding": context_binding,
            "mc0_complete": mc0_complete,
            "mc0_blocking_code": (
                None
                if mc0_complete
                else "METHOD_FAILURE_D1_IMMUTABLE_CONTEXT_BINDING_INCOMPLETE"
            ),
            "metric_audit_rows_sha256": audit_sha,
            "normalized_metric_jsonl_sha256": normalized_sha,
            "normalized_metric_json_sha256": sha256_file(
                output_root / "NORMALIZED_METRIC_ROWS.json"
            ),
            "raw_terminal_sha256": sha256_file(main_root / "RAW_TERMINAL.json"),
            "negative_control_raw_terminal_sha256": sha256_file(
                negative_root / "RAW_TERMINAL.json"
            ),
            "eval_cache_tensor_sha256": cache.manifest["tensor_sha256"],
            "secondary_metric_contract": secondary,
            "lpips_status": secondary[0]["failure_code"],
            "sface_status": secondary[1]["failure_code"],
            "lpips_or_sface_values_fabricated": False,
        }
        write_json(output_root / "METRIC_TERMINAL.json", terminal)
        return terminal
    except Exception:
        write_json(
            output_root / "METRIC_FAILURE.json",
            {
                "status": "FAILED_METRIC_ROOT_RETAINED_NO_RETRY",
                "failure_code": "GLOBAL_METRIC_CONTRACT_FAILED",
                "automatic_retry": False,
                "partial_audit_row_count": len(audits),
                "partial_normalized_row_count": len(normalized),
            },
        )
        raise


def source_check() -> dict[str, Any]:
    """Run one synthetic-only check without opening any experiment artifact."""

    canonical = np.ones((1, 64, 64), dtype=np.float32)
    source_visible = np.zeros((1, 64, 64), dtype=np.float32)
    source_visible[:, :, :32] = 1.0
    target_visible = np.zeros((1, 64, 64), dtype=np.float32)
    target_visible[:, :, 16:] = 1.0
    source = np.zeros((3, 64, 64), dtype=np.float32)
    source[:, :, :32] = 0.25
    target = np.zeros((3, 64, 64), dtype=np.float32)
    target[:, :, 16:] = 0.75
    native = np.full((3, 64, 64), 0.5, dtype=np.float32)
    conserved = np.where(source_visible.astype(bool), source, native).astype(np.float32)
    result = evaluate_one_sample(
        native=native,
        conserved=conserved,
        source_partial=source,
        source_visible=source_visible,
        canonical=canonical,
        paired_target_partial=target,
        paired_target_visible=target_visible,
    )
    _require(
        result["terminal_state"] == "COMPLETE", "synthetic_source_check_not_complete"
    )
    _require(
        result["support"]["common_hidden_texels"] == 32 * 64,
        "synthetic_common_support_changed",
    )
    _require(
        result["metrics"]["native_conserved_hidden_mae_exactly_unchanged"] is True,
        "synthetic_hidden_mae_invariant_failed",
    )
    _require(
        result["metrics"]["native"]["decomposition"]["identity_verified"] is True,
        "synthetic_decomposition_failed",
    )
    return {
        "status": "PASS_SYNTHETIC_SOURCE_CHECK_NO_REAL_METRICS",
        "research_evidence": False,
        "private_artifact_reads": 0,
        "real_metric_rows": 0,
        "statistics_executed": False,
        "lpips_or_sface_values_fabricated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--eval-cache-root", type=Path)
    parser.add_argument("--raw-route-root", type=Path)
    parser.add_argument("--negative-control-raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.source_check:
        _require(
            arguments.eval_cache_root is None
            and arguments.raw_route_root is None
            and arguments.negative_control_raw_root is None
            and arguments.output_root is None,
            "source_check_cannot_open_real_paths",
        )
        result = source_check()
    else:
        _require(
            arguments.eval_cache_root is not None
            and arguments.raw_route_root is not None
            and arguments.negative_control_raw_root is not None
            and arguments.output_root is not None,
            "real_run_requires_four_roots",
        )
        result = execute(
            arguments.eval_cache_root,
            arguments.raw_route_root,
            arguments.negative_control_raw_root,
            arguments.output_root,
        )
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
