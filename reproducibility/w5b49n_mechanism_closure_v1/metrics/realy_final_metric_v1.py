#!/usr/bin/env python3
"""One-shot post-F2 REALY-100 paired UV metric closure.

One frozen source output is evaluated against each of the other three views of
the same anonymous identity.  Targets are selected from the 1,200-row directed
pair roster only after the formal CUDA raw terminal is complete.
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

from reproducibility.w5b49n_mechanism_closure_v1.metrics.facescape_final_metric_v1 import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    NORMALIZED_SCHEMA_VERSION,
    PRIMARY_METRIC_ID,
    SEEDS,
    TERMINAL_STATES,
    MetricClosureError,
    _normalized_row,
    _require,
    _route_group,
    _safe_token,
    _write_jsonl,
    evaluate_one_sample,
)
from reproducibility.w5b49n_mechanism_closure_v1.metrics.context_contract_v1 import (  # noqa: E402
    realy_context_binding,
    secondary_metric_failure_contract,
)
from reproducibility.w5b49n_mechanism_closure_v1.runtime.realy_eval_cache_io import (  # noqa: E402
    IDENTITY_COUNT,
    PAIR_COUNT,
    SAMPLE_COUNT,
    VIEW_COUNT,
    anonymous_source_roster_sha256,
    load_realy_source_cache,
    sha256_file,
)
from reproducibility.w5b49n_mechanism_closure_v1.training.cache_io import (  # noqa: E402
    canonical_json_bytes,
    write_json,
)


PROGRAM_ID = "FRUGALFACE3D-MECHANISM-CLOSURE-V1-REALY-METRIC"
TERMINAL_SCHEMA = "frugalface3d.w5b49n.realy_metric_terminal.v1"
DATASET_ID = "D2"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


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


def _source_sample_id(row: Mapping[str, Any]) -> str:
    return _safe_token(
        f"D2:{row['identity_token']}:{row['view_token']}", "realy_source_sample_id"
    )


def _load_pairs(
    roster_root: Path,
    cache: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Bind and load the directed-pair authority only at the post-F2 boundary."""

    root = roster_root.expanduser().resolve(strict=True)
    manifest_path = root / "cache_manifest.json"
    ready_path = root / "CACHE_READY.json"
    path = root / "directed_pairs.jsonl"
    manifest = _read_json(manifest_path, "realy_post_f2_roster_manifest")
    ready = _read_json(ready_path, "realy_post_f2_pair_ready")
    _require(
        ready.get("status") == "PASS_REALY100_1200_DIRECTED_PAIRS_READY"
        and ready.get("cache_manifest_sha256") == sha256_file(manifest_path)
        and ready.get("identity_count") == IDENTITY_COUNT
        and ready.get("view_count") == VIEW_COUNT
        and ready.get("asset_count") == SAMPLE_COUNT
        and ready.get("directed_pair_count") == PAIR_COUNT
        and manifest.get("status") == "PASS_REALY100_PRIVATE_CACHE_READY"
        and manifest.get("dataset_token") == DATASET_ID
        and manifest.get("identity_count") == IDENTITY_COUNT
        and manifest.get("view_count") == VIEW_COUNT
        and manifest.get("asset_count") == SAMPLE_COUNT
        and manifest.get("directed_pair_count") == PAIR_COUNT,
        "realy_post_f2_pair_authority_contract_changed",
    )
    _require(
        anonymous_source_roster_sha256(manifest)
        == cache.manifest.get("anonymous_source_roster_projection_sha256"),
        "realy_post_f2_source_roster_projection_mismatch",
    )
    _require(not path.is_symlink() and path.is_file(), "realy_pair_roster_not_regular")
    digest = sha256_file(path)
    _require(
        digest == ready.get("directed_pairs_sha256"),
        "realy_pair_roster_sha256_mismatch",
    )
    lookup = {str(row["asset_id"]): int(row["sample_index"]) for row in cache.rows}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetricClosureError(f"realy_pair_json_invalid:{line_number}") from error
            _require(
                isinstance(value, dict)
                and set(value)
                == {
                    "pair_id",
                    "identity_token",
                    "source_view_token",
                    "target_view_token",
                    "source_relative_path",
                    "target_relative_path",
                },
                "realy_pair_keyspace_changed",
            )
            identity = _safe_token(value["identity_token"], "realy_pair_identity")
            source_view = _safe_token(value["source_view_token"], "realy_pair_source_view")
            target_view = _safe_token(value["target_view_token"], "realy_pair_target_view")
            pair_id = _safe_token(value["pair_id"], "realy_pair_id")
            _require(source_view != target_view, "realy_pair_self_pair")
            _require(
                pair_id == f"P-{identity}-{source_view}-{target_view}",
                "realy_pair_id_binding_changed",
            )
            source_asset = f"{identity}-{source_view}"
            target_asset = f"{identity}-{target_view}"
            _require(source_asset in lookup and target_asset in lookup, "realy_pair_asset_missing")
            _require(
                value["source_relative_path"] == f"assets/{identity}/{source_view}.jpg"
                and value["target_relative_path"] == f"assets/{identity}/{target_view}.jpg",
                "realy_pair_anonymous_path_binding_changed",
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "identity_token": identity,
                    "source_index": lookup[source_asset],
                    "target_index": lookup[target_asset],
                }
            )
    _require(len(rows) == PAIR_COUNT, "realy_pair_count_changed")
    _require(len({row["pair_id"] for row in rows}) == PAIR_COUNT, "realy_pair_id_duplicate")
    source_counts = Counter(int(row["source_index"]) for row in rows)
    _require(
        set(source_counts) == set(range(SAMPLE_COUNT)) and set(source_counts.values()) == {3},
        "realy_pair_source_degree_changed",
    )
    return rows, digest


def _load_raw_ledger(
    raw_root: Path, raw_terminal: Mapping[str, Any], route_ids: Sequence[str]
) -> dict[tuple[str, int], Mapping[str, Any]]:
    path = raw_root / "SAMPLE_TERMINAL_LEDGER.jsonl"
    expected_sha = raw_terminal.get("sample_terminal_ledger_sha256")
    _require(
        not path.is_symlink()
        and path.is_file()
        and isinstance(expected_sha, str)
        and HEX64.fullmatch(expected_sha) is not None
        and sha256_file(path) == expected_sha,
        "realy_raw_ledger_sha256_mismatch",
    )
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetricClosureError(f"realy_raw_ledger_json:{line_number}") from error
            route_id = _safe_token(row.get("route_id"), "realy_ledger_route")
            index = row.get("sample_index")
            _require(isinstance(index, int) and 0 <= index < SAMPLE_COUNT, "realy_ledger_index")
            key = (route_id, index)
            _require(key not in result, "realy_ledger_duplicate_key")
            result[key] = row
    expected = {
        (route_id, index) for route_id in route_ids for index in range(SAMPLE_COUNT)
    }
    _require(set(result) == expected, "realy_ledger_keyspace_changed")
    return result


def _load_route(raw_root: Path, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch

    route_id = _safe_token(receipt.get("route_id"), "realy_route_id")
    route_root = raw_root / "routes" / route_id
    terminal = _read_json(route_root / "ROUTE_TERMINAL.json", "realy_route_terminal")
    _require(terminal == receipt, "realy_embedded_route_terminal_mismatch")
    _require(
        terminal.get("status") == "SUCCESS"
        and terminal.get("source_sample_count") == SAMPLE_COUNT
        and terminal.get("native_and_conserved_same_forward") is True,
        "realy_route_terminal_semantics_changed",
    )
    path = route_root / str(terminal.get("raw_output_file"))
    _require(
        path.name == "RAW_OUTPUTS.pt"
        and sha256_file(path) == terminal.get("raw_output_sha256"),
        "realy_route_output_sha256_mismatch",
    )
    arrays = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(arrays, dict) and set(arrays) == {"native", "conserved"}, "realy_route_keys")
    for role in ("native", "conserved"):
        value = arrays[role]
        _require(
            value.dtype == torch.float32
            and tuple(value.shape) == (SAMPLE_COUNT, 3, 64, 64)
            and bool(torch.isfinite(value).all())
            and float(value.min()) >= 0.0
            and float(value.max()) <= 1.0,
            f"realy_route_tensor_contract:{role}",
        )
    return arrays


def _null_metrics() -> dict[str, Any]:
    return {
        "native": None,
        "conserved": None,
        "native_conserved_hidden_mae_delta": None,
        "native_conserved_hidden_mae_exactly_unchanged": None,
        "native_conserved_hidden_psnr_exactly_unchanged": None,
    }


def _failed_audit(
    receipt: Mapping[str, Any], pair: Mapping[str, Any], *, code: str
) -> dict[str, Any]:
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "identity_token": pair["identity_token"],
        "sample_id": pair["pair_id"],
        "source_sample_id": None,
        "paired_target_sample_id": None,
        "route_id": receipt["route_id"],
        "method_id": receipt["method_id"],
        "variant": receipt.get("variant"),
        "seed": receipt.get("seed"),
        "intervention": receipt.get("intervention"),
        "replicate_id": (
            f"S{receipt['seed']}" if receipt.get("seed") is not None else "fixed"
        ),
        "terminal_state": "FAILED",
        "failure_code": _safe_token(code, "realy_failure_code"),
        "candidate_generation_completed_before_target_access": True,
        "target_used_only_at_final_metric_boundary": True,
        "statistical_unit": "identity",
        "support": None,
        "where_invariant": None,
        "metrics": _null_metrics(),
    }


def execute(
    source_cache_root: Path,
    roster_root: Path,
    raw_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("realy_metric_output_root_exists_no_rerun")
    cache = load_realy_source_cache(source_cache_root)
    context_binding = realy_context_binding(cache)
    raw_root = raw_root.expanduser().resolve(strict=True)
    raw_terminal_path = raw_root / "RAW_TERMINAL.json"
    raw_terminal = _read_json(raw_terminal_path, "realy_raw_terminal")
    _require(
        raw_terminal.get("status") == "PASS_REALY_FULL3_BLITE_RAW_OUTPUTS_F2_READY",
        "realy_formal_f2_raw_terminal_required",
    )
    _require(
        raw_terminal.get("source_sample_count_per_route") == SAMPLE_COUNT
        and raw_terminal.get("target_view_image_reads") == 0
        and raw_terminal.get("directed_pair_file_reads") == 0
        and raw_terminal.get("directed_pair_rows_read") == 0
        and raw_terminal.get("pair_relationship_bound_pre_f2") is False
        and raw_terminal.get("directed_pair_binding_phase") == "POST_F2_METRICS_ONLY"
        and raw_terminal.get("paper_metrics_accessed") is False
        and raw_terminal.get("component_ablation_routes") == 0
        and raw_terminal.get("retry_count") == 0
        and raw_terminal.get("immutable_context_binding") == context_binding,
        "realy_f2_raw_semantics_changed",
    )
    receipts = raw_terminal.get("routes")
    _require(isinstance(receipts, list) and len(receipts) == 7, "realy_route_receipt_count")
    _require(
        all(row.get("immutable_context_binding") == context_binding for row in receipts),
        "realy_route_context_binding_changed",
    )
    route_ids = [_safe_token(row.get("route_id"), "realy_route_id") for row in receipts]
    _require(len(set(route_ids)) == len(route_ids), "realy_route_ids_duplicate")
    ledger = _load_raw_ledger(raw_root, raw_terminal, route_ids)
    pairs, pair_sha = _load_pairs(roster_root, cache)

    output_root.mkdir(parents=True, mode=0o700)
    write_json(
        output_root / "ATTEMPT.json",
        {
            "status": "STARTED_POST_F2_REALY_PAIRED_METRIC_ONCE",
            "program_id": PROGRAM_ID,
            "source_sample_count": SAMPLE_COUNT,
            "directed_pair_count": PAIR_COUNT,
            "raw_terminal_sha256": sha256_file(raw_terminal_path),
            "source_cache_tensor_sha256": cache.manifest["tensor_sha256"],
            "directed_pairs_sha256": pair_sha,
            "automatic_retry": False,
            "immutable_context_binding": context_binding,
        },
    )
    audits: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    try:
        for receipt in receipts:
            route_id = str(receipt["route_id"])
            replicate_id = (
                f"S{receipt['seed']}" if receipt.get("seed") is not None else "fixed"
            )
            try:
                arrays = _load_route(raw_root, receipt)
                route_failed = False
            except Exception:
                arrays = None
                route_failed = True
            for pair in pairs:
                source_index = int(pair["source_index"])
                target_index = int(pair["target_index"])
                source_row = cache.rows[source_index]
                target_row = cache.rows[target_index]
                raw_row = ledger[(route_id, source_index)]
                _require(
                    raw_row.get("sample_token") == _source_sample_id(source_row)
                    and raw_row.get("tensor_index") == source_index
                    and raw_row.get("attempt") == 1,
                    "realy_raw_sample_binding_changed",
                )
                if route_failed or raw_row.get("terminal_state") != "COMPLETE":
                    audit = _failed_audit(
                        receipt,
                        pair,
                        code=(
                            "REALY_ROUTE_ARTIFACT_CONTRACT_FAILED"
                            if route_failed
                            else "REALY_UPSTREAM_SOURCE_NOT_COMPLETE"
                        ),
                    )
                    audit["source_sample_id"] = _source_sample_id(source_row)
                    audit["paired_target_sample_id"] = _source_sample_id(target_row)
                else:
                    assert arrays is not None
                    try:
                        result = evaluate_one_sample(
                            native=arrays["native"][source_index].numpy(),
                            conserved=arrays["conserved"][source_index].numpy(),
                            source_partial=cache.tensors["partial_uv"][source_index].numpy(),
                            source_visible=cache.tensors["visibility"][source_index].numpy(),
                            canonical=cache.tensors["canonical_mask"][source_index].numpy(),
                            paired_target_partial=cache.tensors["partial_uv"][target_index].numpy(),
                            paired_target_visible=cache.tensors["visibility"][target_index].numpy(),
                        )
                        audit = {
                            "schema_version": METRIC_SCHEMA_VERSION,
                            "dataset_id": DATASET_ID,
                            "identity_token": pair["identity_token"],
                            "sample_id": pair["pair_id"],
                            "source_sample_id": _source_sample_id(source_row),
                            "paired_target_sample_id": _source_sample_id(target_row),
                            "route_id": route_id,
                            "method_id": receipt["method_id"],
                            "variant": receipt.get("variant"),
                            "seed": receipt.get("seed"),
                            "intervention": receipt.get("intervention"),
                            "replicate_id": replicate_id,
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
                            receipt, pair, code="REALY_METRIC_RUNTIME_CONTRACT_FAILED"
                        )
                        audit["source_sample_id"] = _source_sample_id(source_row)
                        audit["paired_target_sample_id"] = _source_sample_id(target_row)
                audits.append(audit)
                terminal_state = str(audit["terminal_state"])
                _require(terminal_state in TERMINAL_STATES, "realy_terminal_state_invalid")
                for mode in ("native", "conserved"):
                    group_id, _intervention = _route_group(receipt, mode)
                    metric = audit["metrics"][mode]
                    replicate_ids = (
                        tuple(f"S{seed}" for seed in SEEDS)
                        if receipt.get("method_id") == "b_lite"
                        and mode == "conserved"
                        and replicate_id == "fixed"
                        else (replicate_id,)
                    )
                    for normalized_replicate in replicate_ids:
                        normalized.append(
                            _normalized_row(
                                dataset_id=DATASET_ID,
                                identity_token=str(pair["identity_token"]),
                                sample_id=str(pair["pair_id"]),
                                group_id=group_id,
                                replicate_id=normalized_replicate,
                                terminal_state=terminal_state,
                                value=(
                                    float(metric["hidden_uv_mae"])
                                    if terminal_state == "COMPLETE"
                                    and metric is not None
                                    else None
                                ),
                            )
                        )

        _require(len(audits) == len(receipts) * PAIR_COUNT, "realy_audit_count_changed")
        audit_keys = [(row["route_id"], row["sample_id"]) for row in audits]
        _require(len(audit_keys) == len(set(audit_keys)), "realy_audit_duplicate")
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
        _require(len(normalized_keys) == len(set(normalized_keys)), "realy_normalized_duplicate")
        groups = {str(row["group_id"]) for row in normalized}
        _require(
            {"Full__conserved", "Full__spatial_shuffle__conserved"}.issubset(groups),
            "realy_mc1_groups_missing",
        )
        _require("b_lite__conserved" in groups, "realy_b_lite_group_missing")
        comparisons = [
            {
                "comparison_id": "Full_vs_B_lite_conserved",
                "family_id": "external",
                "inferential": True,
                "reference_group": "Full__conserved",
                "comparator_group": "b_lite__conserved",
                "replicate_ids": [f"S{seed}" for seed in SEEDS],
            },
            {
                "comparison_id": "Full_aligned_vs_spatial_shuffle_conserved",
                "family_id": "external",
                "inferential": True,
                "reference_group": "Full__conserved",
                "comparator_group": "Full__spatial_shuffle__conserved",
                "replicate_ids": [f"S{seed}" for seed in SEEDS],
            }
        ]
        audit_sha = _write_jsonl(output_root / "METRIC_AUDIT_ROWS.jsonl", audits)
        normalized_sha = _write_jsonl(
            output_root / "NORMALIZED_METRIC_ROWS.jsonl", normalized
        )
        statistics_input = {
            "schema_version": NORMALIZED_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "metric_id": PRIMARY_METRIC_ID,
            "metric_direction": "lower_is_better",
            "bootstrap_seed": 20260816,
            "bootstrap_iterations": 10_000,
            "expected_identity_count": IDENTITY_COUNT,
            "rows": normalized,
            "comparisons": comparisons,
        }
        write_json(output_root / "NORMALIZED_METRIC_ROWS.json", statistics_input)
        state_counts = Counter(str(row["terminal_state"]) for row in audits)
        status = (
            "PASS_POST_F2_REALY_PAIRED_METRIC_LEDGER_COMPLETE"
            if state_counts.get("FAILED", 0) == 0
            and state_counts.get("SKIPPED_BY_CONTRACT", 0) == 0
            else "TERMINAL_POST_F2_REALY_PAIRED_METRIC_LEDGER_WITH_RETAINED_FAILURES"
        )
        secondary = secondary_metric_failure_contract(
            dataset_id=DATASET_ID,
            render_context_ready=True,
            reason_suffix="SEPARATE_POST_F2_SECONDARY_RUNNER_NOT_EXECUTED",
        )
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": status,
            "program_id": PROGRAM_ID,
            "dataset_id": DATASET_ID,
            "route_count": len(receipts),
            "source_sample_count": SAMPLE_COUNT,
            "directed_pair_count_per_route": PAIR_COUNT,
            "audit_row_count": len(audits),
            "hidden_uv_mae_audit_row_count": len(audits),
            "hidden_uv_psnr_audit_row_count": len(audits),
            "normalized_primary_metric_row_count": len(normalized),
            "terminal_state_counts": {
                state: int(state_counts.get(state, 0)) for state in sorted(TERMINAL_STATES)
            },
            "all_four_terminal_states_retained_by_schema": True,
            "failure_rows_dropped": 0,
            "retry_count": 0,
            "imputation_count": 0,
            "identity_is_only_statistical_unit": True,
            "statistics_executed": False,
            "cross_dataset_pooling": False,
            "component_ablation_routes": 0,
            "fixed_b_lite_reused_across_preregistered_full_seeds": True,
            "external_comparison_count": len(comparisons),
            "common_hidden_support_formula": "canonical_AND_NOT_source_visible_AND_paired_target_visible",
            "target_used_only_at_final_metric_boundary": True,
            "immutable_context_binding": context_binding,
            "mc0_complete": True,
            "metric_audit_rows_sha256": audit_sha,
            "normalized_metric_jsonl_sha256": normalized_sha,
            "normalized_metric_json_sha256": sha256_file(
                output_root / "NORMALIZED_METRIC_ROWS.json"
            ),
            "raw_terminal_sha256": sha256_file(raw_terminal_path),
            "source_cache_tensor_sha256": cache.manifest["tensor_sha256"],
            "directed_pairs_sha256": pair_sha,
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
                "status": "FAILED_REALY_METRIC_ROOT_RETAINED_NO_RETRY",
                "failure_code": "GLOBAL_REALY_METRIC_CONTRACT_FAILED",
                "automatic_retry": False,
                "partial_audit_row_count": len(audits),
                "partial_normalized_row_count": len(normalized),
            },
        )
        raise


def source_check() -> dict[str, Any]:
    _require(PAIR_COUNT == 1200 and SAMPLE_COUNT == 400, "realy_source_counts_changed")
    _require(IDENTITY_COUNT == 100, "realy_identity_count_changed")
    return {
        "status": "PASS_REALY_PAIRED_METRIC_SYNTHETIC_SOURCE_CHECK",
        "research_evidence": False,
        "private_artifact_reads": 0,
        "real_metric_rows": 0,
        "statistics_executed": False,
        "formal_raw_status_required": "PASS_REALY_FULL3_BLITE_RAW_OUTPUTS_F2_READY",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--roster-root", type=Path)
    parser.add_argument("--raw-route-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.source_check:
        _require(
            arguments.source_cache_root is None
            and arguments.roster_root is None
            and arguments.raw_route_root is None
            and arguments.output_root is None,
            "realy_source_check_cannot_open_real_paths",
        )
        result = source_check()
    else:
        _require(
            arguments.source_cache_root is not None
            and arguments.roster_root is not None
            and arguments.raw_route_root is not None
            and arguments.output_root is not None,
            "realy_real_run_requires_four_roots",
        )
        result = execute(
            arguments.source_cache_root,
            arguments.roster_root,
            arguments.raw_route_root,
            arguments.output_root,
        )
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
