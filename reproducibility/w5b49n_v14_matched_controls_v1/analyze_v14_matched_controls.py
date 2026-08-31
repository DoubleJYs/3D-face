#!/usr/bin/env python3
"""Fail-closed identity-level statistics for W5B49N V14 matched controls.

The program never edits an input. It validates every bound input and the full
metric keyspace before creating a new, non-existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CONTRACT_SCHEMA = "frugalface3d.w5b49n.v14.analysis_contract.v1"
BINDING_SCHEMA = "frugalface3d.w5b49n.v14.input_binding.v1"
PAIR_ROSTER_SCHEMA = "frugalface3d.w5b49n.v14.pair_roster.v1"
METRIC_ROW_SCHEMA = "frugalface3d.w5b49n.v14.metric_row.v1"
OUTPUT_SCHEMA = "frugalface3d.w5b49n.v14.analysis_output.v1"
REQUIRED_SEEDS = [2026080447, 2026080448, 2026080449, 2026080450, 2026080451]
SEEDED_METHODS = ["full", "condition0", "b_lite_ft"]
FIXED_METHODS = ["freeuv_conserved"]
METRIC_ORDER = ["hidden_uv_mae", "lpips_alex_v0_1", "sface_source_to_render_cosine"]

EXPECTED_FAMILIES = [
    {
        "family_id": "F1-MAE-ATTRIBUTION",
        "metric_id": "hidden_uv_mae",
        "members": [
            {"dataset_id": "D1", "comparator_id": "condition0"},
            {"dataset_id": "D1", "comparator_id": "b_lite_ft"},
            {"dataset_id": "D2", "comparator_id": "condition0"},
            {"dataset_id": "D2", "comparator_id": "b_lite_ft"},
        ],
    },
    {
        "family_id": "F2-MAE-PUBLIC",
        "metric_id": "hidden_uv_mae",
        "members": [
            {"dataset_id": "D1", "comparator_id": "freeuv_conserved"},
            {"dataset_id": "D2", "comparator_id": "freeuv_conserved"},
        ],
    },
    {
        "family_id": "F3-LPIPS",
        "metric_id": "lpips_alex_v0_1",
        "members": [
            {"dataset_id": "D1", "comparator_id": "condition0"},
            {"dataset_id": "D1", "comparator_id": "b_lite_ft"},
            {"dataset_id": "D1", "comparator_id": "freeuv_conserved"},
            {"dataset_id": "D2", "comparator_id": "condition0"},
            {"dataset_id": "D2", "comparator_id": "b_lite_ft"},
            {"dataset_id": "D2", "comparator_id": "freeuv_conserved"},
        ],
    },
    {
        "family_id": "F4-SFACE",
        "metric_id": "sface_source_to_render_cosine",
        "members": [
            {"dataset_id": "D1", "comparator_id": "condition0"},
            {"dataset_id": "D1", "comparator_id": "b_lite_ft"},
            {"dataset_id": "D1", "comparator_id": "freeuv_conserved"},
            {"dataset_id": "D2", "comparator_id": "condition0"},
            {"dataset_id": "D2", "comparator_id": "b_lite_ft"},
            {"dataset_id": "D2", "comparator_id": "freeuv_conserved"},
        ],
    },
]

EXPECTED_ARTIFACT_ROLES = {
    "statistical_analysis_plan",
    "claim_decision_matrix",
    "analysis_program",
    "b_lite_frozen_terminal",
    *(f"full_seed_{seed}_terminal" for seed in REQUIRED_SEEDS),
    *(f"condition0_seed_{seed}_terminal" for seed in REQUIRED_SEEDS),
    *(f"b_lite_ft_seed_{seed}_terminal" for seed in REQUIRED_SEEDS),
    "freeuv_v1_2_terminal",
    "d1_eval_cache_terminal",
    "d2_eval_cache_terminal",
    "shared_render_terminal",
    "lpips_linux_qualification_terminal",
    "lpips_terminal",
    "sface_linux_qualification_terminal",
    "sface_terminal",
}


class FailClosedError(RuntimeError):
    """Raised whenever the frozen analysis contract is not fully satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailClosedError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser message is platform-specific
        raise FailClosedError(f"无法读取 JSON：{path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON 顶层必须为对象：{path}")
    return value


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validated_regular_file(raw_path: Any, label: str, require_absolute: bool = True) -> Path:
    _require(isinstance(raw_path, str) and raw_path.strip(), f"{label} 缺少路径")
    path = Path(raw_path)
    if require_absolute:
        _require(path.is_absolute(), f"{label} 必须使用绝对路径：{path}")
    _require(path.exists(), f"{label} 不存在：{path}")
    _require(not path.is_symlink(), f"{label} 不得为符号链接：{path}")
    _require(path.is_file(), f"{label} 必须为普通文件：{path}")
    return path.resolve()


def _verify_hash_entry(entry: Any, label: str) -> tuple[Path, str]:
    _require(isinstance(entry, dict), f"{label} 的绑定必须为对象")
    path = _validated_regular_file(entry.get("path"), label)
    expected = entry.get("sha256")
    _require(_is_sha256(expected), f"{label} 的 SHA-256 未完成绑定")
    actual = sha256_file(path)
    _require(actual == expected, f"{label} 哈希不一致：expected={expected}, actual={actual}")
    return path, actual


def _validate_contract(contract: dict[str, Any], allow_synthetic: bool = False) -> None:
    _require(contract.get("schema_version") == CONTRACT_SCHEMA, "合同 schema_version 不匹配")
    _require(contract.get("contract_id") == "W5B49N_V14_MATCHED_CONTROLS_V1", "合同标识不匹配")
    mode = contract.get("contract_mode")
    _require(mode == "production" or (allow_synthetic and mode == "synthetic"), "非生产合同被拒绝")
    _require(contract.get("status") == "FROZEN_BEFORE_NEW_METRIC_REVIEW", "合同未处于冻结状态")
    _require(contract.get("analysis_unit") == "identity", "推断单位必须为身份")
    _require(contract.get("seeds") == REQUIRED_SEEDS, "随机种子必须严格为 2026080447—2026080451")
    seed_notes = contract.get("seed_notes")
    _require(isinstance(seed_notes, dict), "缺少随机种子来源说明")
    _require(seed_notes.get("full_cuda_retrain_required") == REQUIRED_SEEDS, "Full 必须在 CUDA 环境重训五个种子")
    _require(seed_notes.get("condition0_required") == REQUIRED_SEEDS, "Condition0 必须训练五个种子")
    _require(seed_notes.get("b_lite_ft_required") == REQUIRED_SEEDS, "B-lite-FT 必须训练五个种子")
    _require(
        seed_notes.get("historical_mps_full_2026080447_to_2026080449") == "excluded_from_confirmatory_families",
        "历史 MPS Full 必须排除在确认性比较之外",
    )
    _require(contract.get("families") == EXPECTED_FAMILIES, "四个预设比较族或其顺序发生变化")
    _require(
        contract.get("matched_training")
        == {
            "device_backend": "cuda",
            "device_name": "NVIDIA GeForce RTX 4090",
            "training_units": 15,
            "steps_per_unit": 512,
            "total_steps": 7680,
            "cross_terminal_equal_hash_fields": [
                "environment_manifest_sha256",
                "training_split_sha256",
                "training_budget_manifest_sha256",
            ],
            "per_terminal_unique_hash_field": "checkpoint_sha256",
        },
        "确认性训练必须是同一 RTX 4090/CUDA 环境下的 15 个单元和 7,680 步",
    )

    methods = contract.get("methods")
    _require(isinstance(methods, dict) and set(methods) == set(SEEDED_METHODS + FIXED_METHODS), "方法集合不匹配")
    for method_id in SEEDED_METHODS:
        _require(methods[method_id].get("seed_mode") == "five_fixed_seeds", f"{method_id} 未固定五个种子")
    _require(methods["freeuv_conserved"].get("seed_mode") == "single_frozen_output", "FreeUV 必须为一次冻结输出")

    metrics = contract.get("metrics")
    _require(isinstance(metrics, dict) and list(metrics) == METRIC_ORDER, "指标集合或顺序不匹配")
    expected_directions = {
        "hidden_uv_mae": "lower_is_better",
        "lpips_alex_v0_1": "lower_is_better",
        "sface_source_to_render_cosine": "higher_is_better",
    }
    for metric_id, direction in expected_directions.items():
        _require(metrics[metric_id].get("direction") == direction, f"{metric_id} 的方向不匹配")

    aggregation = contract.get("aggregation")
    _require(
        aggregation == {
            "pair_to_identity": "median",
            "seed_to_identity": "median",
            "identity_to_overall": "median",
            "zero_effect_handling": "exclude_from_effective_sign_count",
        },
        "聚合规则发生变化",
    )
    inference = contract.get("inference")
    _require(isinstance(inference, dict), "缺少推断规则")
    _require(inference.get("sign_test") == "two_sided_exact_binomial_p_0_5", "只允许双侧精确符号检验")
    _require(inference.get("multiple_comparison") == "holm_within_each_prespecified_family", "只允许族内 Holm 校正")
    _require(inference.get("familywise_alpha") == 0.05, "家族错误率必须为 0.05")
    bootstrap = inference.get("bootstrap")
    _require(
        bootstrap == {
            "unit": "identity",
            "method": "percentile",
            "resamples": 10000,
            "confidence_level": 0.95,
            "base_seed": 20260816,
        },
        "身份自助法规则发生变化",
    )

    datasets = contract.get("datasets")
    _require(isinstance(datasets, dict) and set(datasets) == {"D1", "D2"}, "数据集集合不匹配")
    if mode == "production":
        expected_counts = {
            "D1": (20, 160, 148, 18),
            "D2": (100, 1200, 1200, 90),
        }
        for dataset_id, counts in expected_counts.items():
            item = datasets[dataset_id]
            observed = (
                item.get("expected_identity_count"),
                item.get("expected_total_pair_count"),
                item.get("expected_analysis_pair_count"),
                item.get("sface_min_identity_count"),
            )
            _require(observed == counts, f"{dataset_id} 的固定样本规模不匹配")
    else:
        for dataset_id, item in datasets.items():
            for key in ("expected_identity_count", "expected_total_pair_count", "expected_analysis_pair_count", "sface_min_identity_count"):
                _require(isinstance(item.get(key), int) and item[key] > 0, f"合成合同 {dataset_id}.{key} 非法")
            _require(item["expected_analysis_pair_count"] <= item["expected_total_pair_count"], f"合成合同 {dataset_id} 配对数非法")
            _require(item["sface_min_identity_count"] <= item["expected_identity_count"], f"合成合同 {dataset_id} SFace 身份门槛非法")

    artifacts = contract.get("required_artifacts")
    _require(isinstance(artifacts, dict), "合同缺少 required_artifacts")
    _require(set(artifacts) == EXPECTED_ARTIFACT_ROLES, "必需输入角色集合不完整或存在额外角色")
    _require(
        artifacts["statistical_analysis_plan"]
        == {
            "kind": "text_file",
            "required_basename": "STATISTICAL_ANALYSIS_PLAN.md",
            "required_markers": ["V14-SAP-1.5", "RTX 4090/CUDA", "7,680 步", "10,000 次"],
        },
        "统计计划绑定规则发生变化",
    )
    _require(
        artifacts["claim_decision_matrix"]
        == {
            "kind": "text_file",
            "required_basename": "V14_CLAIM_DECISION_MATRIX.md",
            "required_markers": ["V14 主张判定矩阵", "五个预定随机种子"],
        },
        "主张矩阵绑定规则发生变化",
    )
    _require(
        artifacts["analysis_program"]
        == {"kind": "analysis_program", "required_basename": "analyze_v14_matched_controls.py"},
        "统计程序绑定规则发生变化",
    )
    expected_training_status = {
        "full": "PASS_V14_FULL_SEED_COMPLETE",
        "condition0": "PASS_V14_CONDITION0_SEED_COMPLETE",
        "b_lite_ft": "PASS_V14_B_LITE_FT_SEED_COMPLETE",
    }
    for method_id, terminal_status in expected_training_status.items():
        for seed in REQUIRED_SEEDS:
            role = f"{method_id}_seed_{seed}_terminal"
            rule = artifacts[role]
            _require(rule.get("kind") == "json_terminal" and rule.get("require_bound_files") is True, f"{role} 绑定规则非法")
            _require(
                rule.get("required_fields")
                == {
                    "schema_version": "frugalface3d.w5b49n.v14.terminal.v1",
                    "status": terminal_status,
                    "method_id": method_id,
                    "seed": seed,
                },
                f"{role} 终态规则发生变化",
            )
    expected_qualification_fields = {
        "lpips_linux_qualification_terminal": {
            "schema_version": "frugalface3d.w5b49n.v14.terminal.v1",
            "status": "PASS_V14_LPIPS_LINUX_QUALIFIED",
            "metric_id": "lpips_alex_v0_1",
            "operating_system": "linux",
            "device_backend": "cpu",
            "cuda_calls": 0,
            "fresh_output_root": True,
            "prior_method_failure_reused": False,
            "probe_status": "PASS",
            "probe_tolerance_abs": 1e-6,
        },
        "sface_linux_qualification_terminal": {
            "schema_version": "frugalface3d.w5b49n.v14.terminal.v1",
            "status": "PASS_V14_SFACE_LINUX_QUALIFIED",
            "metric_id": "sface_source_to_render_cosine",
            "operating_system": "linux",
            "device_backend": "cpu",
            "cuda_calls": 0,
            "fresh_output_root": True,
            "prior_method_failure_reused": False,
            "probe_status": "PASS",
            "probe_tolerance_abs": 1e-6,
        },
    }
    for role, expected_fields in expected_qualification_fields.items():
        rule = artifacts[role]
        _require(rule.get("kind") == "json_terminal" and rule.get("require_bound_files") is True, f"{role} 绑定规则非法")
        _require(rule.get("required_fields") == expected_fields, f"{role} 资格规则发生变化")
    expected_result_terminal_core = {
        "lpips_terminal": {
            "status": "PASS_V14_LPIPS_COMPLETE",
            "metric_id": "lpips_alex_v0_1",
        },
        "sface_terminal": {
            "status": "PASS_V14_SFACE_COMPLETE",
            "metric_id": "sface_source_to_render_cosine",
        },
    }
    for role, expected_core in expected_result_terminal_core.items():
        rule = artifacts[role]
        required_fields = rule.get("required_fields", {})
        _require(rule.get("kind") == "json_terminal" and rule.get("require_bound_files") is True, f"{role} 绑定规则非法")
        _require(required_fields.get("schema_version") == "frugalface3d.w5b49n.v14.terminal.v1", f"{role} schema 规则非法")
        _require(required_fields.get("status") == expected_core["status"], f"{role} 完成状态规则非法")
        _require(required_fields.get("metric_id") == expected_core["metric_id"], f"{role} 指标规则非法")
        _require(required_fields.get("device_backend") == "cpu", f"{role} 必须在 CPU 评估器上运行")
        _require(required_fields.get("cuda_calls") == 0, f"{role} 的 CUDA 调用数必须为 0")
        _require(required_fields.get("qualification_required") is True, f"{role} 必须绑定资格终态")
        if role == "sface_terminal":
            _require(required_fields.get("failure_ledger_complete") is True, "SFace 必须提供完整失败 ledger")
            _require(required_fields.get("silent_row_drop_count") == 0, "SFace 不允许静默丢行")
    _require(contract.get("fail_closed") == {
        "reject_missing_or_extra_metric_rows": True,
        "reject_duplicate_metric_rows": True,
        "reject_nonfinite_or_out_of_range_values": True,
        "reject_incomplete_terminal": True,
        "reject_hash_mismatch": True,
        "reject_existing_output_root": True,
        "create_output_only_after_all_checks": True,
    }, "fail_closed 规则发生变化")
    if mode == "production":
        expected_freeuv_hashes = {
            "25e26864d5cf6429171faf76c3575944a7f860315e3835b32adbe7f5710e418c",
            "3f25312879e395676aeac32c5ee3a1d1b08bb3db3703bb34ac73a28a0ee02ff0",
            "60d6ad02174cbdae1ea466e5a94e1f7b456fc7537aa11c91d10235f05a67e430",
            "ba64333dc39daafdfb45a13363705fb4cf8e716d4cc6901c352e744b78dcbeb2",
            "f701dd3931c995de947e732838ad6acdaf2ae1665aeb633cc746df08b02b4357",
            "63008d05585994d0ec2e7830e8739a61ce4b38897b2bb3c495a19dd9b2e0c616",
        }
        _require(
            set(artifacts["freeuv_v1_2_terminal"].get("required_bound_sha256", [])) == expected_freeuv_hashes,
            "FreeUV V1.2 固定输入哈希发生变化",
        )


def _validate_bound_files(terminal: dict[str, Any], role: str, rule: dict[str, Any]) -> list[dict[str, str]]:
    if not rule.get("require_bound_files"):
        return []
    bound_files = terminal.get("bound_files")
    _require(isinstance(bound_files, list) and bound_files, f"{role} 缺少 bound_files")
    verified: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(bound_files):
        path, actual = _verify_hash_entry(entry, f"{role}.bound_files[{index}]")
        path_text = str(path)
        _require(path_text not in seen_paths, f"{role} 重复绑定同一文件：{path_text}")
        seen_paths.add(path_text)
        verified.append({"path": path_text, "sha256": actual})
    required_hashes = rule.get("required_bound_sha256", [])
    _require(isinstance(required_hashes, list) and all(_is_sha256(item) for item in required_hashes), f"{role} 的固定 bound hash 非法")
    observed_hashes = {item["sha256"] for item in verified}
    missing_hashes = sorted(set(required_hashes) - observed_hashes)
    _require(not missing_hashes, f"{role} 缺少固定输入哈希：{missing_hashes}")
    return verified


def _validate_binding(
    contract_path: Path,
    contract: dict[str, Any],
    binding_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _read_json(binding_path)
    _require(binding.get("schema_version") == BINDING_SCHEMA, "输入绑定 schema_version 不匹配")
    _require(binding.get("status") == "FROZEN_COMPLETE", "输入绑定尚未完成；必须为 FROZEN_COMPLETE")
    contract_hash = sha256_file(contract_path)
    _require(binding.get("contract_sha256") == contract_hash, "合同哈希与绑定不一致")

    pair_roster_path, pair_roster_hash = _verify_hash_entry(binding.get("pair_roster"), "pair_roster")
    metric_rows_path, metric_rows_hash = _verify_hash_entry(binding.get("metric_rows"), "metric_rows")
    _require(pair_roster_path != metric_rows_path, "配对清单与指标文件不得为同一文件")

    bound_artifacts = binding.get("artifacts")
    _require(isinstance(bound_artifacts, dict), "绑定缺少 artifacts")
    rules = contract["required_artifacts"]
    _require(set(bound_artifacts) == set(rules), "绑定的输入角色不完整或存在额外角色")

    verified_artifacts: dict[str, Any] = {}
    artifact_paths: set[str] = set()
    training_terminal_records: list[dict[str, Any]] = []
    terminal_payloads: dict[str, dict[str, Any]] = {}
    for role in rules:
        rule = rules[role]
        _require(isinstance(rule, dict), f"合同中的 {role} 规则非法")
        path, actual_hash = _verify_hash_entry(bound_artifacts[role], role)
        path_text = str(path)
        _require(path_text not in artifact_paths, f"多个输入角色绑定到同一文件：{path_text}")
        artifact_paths.add(path_text)
        kind = rule.get("kind")
        required_basename = rule.get("required_basename")
        if required_basename is not None:
            _require(path.name == required_basename, f"{role} 文件名不匹配：{path.name}")

        record: dict[str, Any] = {"path": path_text, "sha256": actual_hash, "kind": kind}
        if kind == "text_file":
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                raise FailClosedError(f"无法读取文本输入 {role}: {exc}") from exc
            markers = rule.get("required_markers", [])
            _require(isinstance(markers, list) and all(isinstance(item, str) for item in markers), f"{role} 标记规则非法")
            for marker in markers:
                _require(marker in content, f"{role} 缺少冻结标记：{marker}")
        elif kind == "analysis_program":
            _require(path == Path(__file__).resolve(), "analysis_program 必须绑定当前执行脚本")
        elif kind == "json_terminal":
            terminal = _read_json(path)
            terminal_payloads[role] = terminal
            required_fields = rule.get("required_fields")
            _require(isinstance(required_fields, dict), f"{role} 缺少 required_fields")
            for key, expected_value in required_fields.items():
                _require(terminal.get(key) == expected_value, f"{role}.{key} 未达到冻结终态")
            record["terminal_status"] = terminal.get("status")
            record["bound_files"] = _validate_bound_files(terminal, role, rule)
            observed_bound_hashes = {item["sha256"] for item in record["bound_files"]}
            if role == "freeuv_v1_2_terminal":
                source_terminal_hash = terminal.get("source_terminal_sha256")
                _require(_is_sha256(source_terminal_hash), "freeuv_v1_2_terminal.source_terminal_sha256 非法")
                _require(source_terminal_hash in observed_bound_hashes, "FreeUV wrapper 未绑定原始 V1.2 活动终态")
                _require(terminal.get("source_package_sha256") in observed_bound_hashes, "FreeUV wrapper 未绑定原始 V1.2 结果包")
                source_terminal_entries = [
                    item for item in record["bound_files"] if item["sha256"] == source_terminal_hash
                ]
                _require(len(source_terminal_entries) == 1, "FreeUV 原始活动终态绑定不唯一")
                source_terminal = _read_json(Path(source_terminal_entries[0]["path"]))
                _require(source_terminal.get("status") == "PASS_F2_FREEUV_D1D2_RAW_OUTPUTS_FROZEN", "FreeUV 原始 V1.2 活动终态未通过")
                _require(
                    source_terminal.get("successful_total_forward_count") == terminal.get("forward_count"),
                    "FreeUV 原始活动终态的成功前向数不匹配",
                )
                _require(source_terminal.get("automatic_retry") is False, "FreeUV 原始活动包含自动重试")
                record["source_terminal_sha256"] = source_terminal_hash
            qualification_hash_fields = {
                "lpips_linux_qualification_terminal": [
                    "qualification_script_sha256",
                    "runtime_manifest_sha256",
                    "evaluator_export_sha256",
                    "probe_manifest_sha256",
                ],
                "sface_linux_qualification_terminal": [
                    "qualification_script_sha256",
                    "runtime_manifest_sha256",
                    "detector_model_sha256",
                    "recognizer_model_sha256",
                    "probe_manifest_sha256",
                ],
            }
            if role in qualification_hash_fields:
                for field in qualification_hash_fields[role]:
                    value = terminal.get(field)
                    _require(_is_sha256(value), f"{role}.{field} 未绑定有效 SHA-256")
                    _require(value in observed_bound_hashes, f"{role}.{field} 未出现在 bound_files 中")
            if role.startswith("full_seed_") or role.startswith("condition0_seed_") or role.startswith("b_lite_ft_seed_"):
                training_spec = contract["matched_training"]
                _require(terminal.get("device_backend") == training_spec["device_backend"], f"{role} 不是 CUDA 训练")
                _require(terminal.get("device_name") == training_spec["device_name"], f"{role} 不是同一 RTX 4090 设备类别")
                _require(terminal.get("training_steps") == training_spec["steps_per_unit"], f"{role} 训练步数不是 512")
                matched_fields: dict[str, str] = {}
                required_hash_fields = [
                    *training_spec["cross_terminal_equal_hash_fields"],
                    training_spec["per_terminal_unique_hash_field"],
                ]
                for field in required_hash_fields:
                    value = terminal.get(field)
                    _require(_is_sha256(value), f"{role}.{field} 未绑定有效 SHA-256")
                    _require(value in observed_bound_hashes, f"{role}.{field} 未出现在 bound_files 中")
                    matched_fields[field] = value
                record["matched_training"] = {
                    "method_id": terminal.get("method_id"),
                    "seed": terminal.get("seed"),
                    "device_backend": terminal.get("device_backend"),
                    "device_name": terminal.get("device_name"),
                    "training_steps": terminal.get("training_steps"),
                    **matched_fields,
                }
                training_terminal_records.append(record["matched_training"])
        else:
            raise FailClosedError(f"{role} 使用未知输入类型：{kind}")
        verified_artifacts[role] = record

    training_spec = contract["matched_training"]
    _require(
        len(training_terminal_records) == training_spec["training_units"],
        f"确认性训练终态必须恰好为 {training_spec['training_units']} 个",
    )
    for field in training_spec["cross_terminal_equal_hash_fields"]:
        values = {record[field] for record in training_terminal_records}
        _require(len(values) == 1, f"15 个训练终态的 {field} 不一致")
    checkpoint_field = training_spec["per_terminal_unique_hash_field"]
    checkpoint_hashes = [record[checkpoint_field] for record in training_terminal_records]
    _require(len(set(checkpoint_hashes)) == training_spec["training_units"], "15 个训练单元没有绑定 15 个独立 checkpoint")
    observed_method_seed = {(record["method_id"], record["seed"]) for record in training_terminal_records}
    expected_method_seed = {(method_id, seed) for method_id in SEEDED_METHODS for seed in REQUIRED_SEEDS}
    _require(observed_method_seed == expected_method_seed, "训练终态未完整覆盖三种方法的五个固定种子")
    qualification_links = {
        "lpips_terminal": "lpips_linux_qualification_terminal",
        "sface_terminal": "sface_linux_qualification_terminal",
    }
    for result_role, qualification_role in qualification_links.items():
        qualification_hash = verified_artifacts[qualification_role]["sha256"]
        result_terminal = terminal_payloads[result_role]
        _require(
            result_terminal.get("qualification_terminal_sha256") == qualification_hash,
            f"{result_role} 未绑定本次通过的 {qualification_role}",
        )
        result_bound_hashes = {item["sha256"] for item in verified_artifacts[result_role]["bound_files"]}
        _require(qualification_hash in result_bound_hashes, f"{result_role}.bound_files 缺少资格终态")

    raw_output_root = binding.get("output_root")
    _require(isinstance(raw_output_root, str) and raw_output_root.strip(), "缺少 output_root")
    output_root = Path(raw_output_root)
    _require(output_root.is_absolute(), "output_root 必须为绝对路径")
    _require(not output_root.exists(), f"output_root 已存在，程序不会覆盖：{output_root}")
    _require(output_root.parent.exists() and output_root.parent.is_dir(), f"output_root 的父目录不存在：{output_root.parent}")
    _require(not output_root.parent.is_symlink(), f"output_root 的父目录不得为符号链接：{output_root.parent}")

    binding_summary = {
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "pair_roster_path": str(pair_roster_path),
        "pair_roster_sha256": pair_roster_hash,
        "metric_rows_path": str(metric_rows_path),
        "metric_rows_sha256": metric_rows_hash,
        "artifacts": verified_artifacts,
        "output_root": str(output_root),
    }
    paths = {
        "pair_roster": pair_roster_path,
        "metric_rows": metric_rows_path,
        "output_root": output_root,
    }
    return binding_summary, paths


def _load_pair_roster(contract: dict[str, Any], path: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, Any]]:
    roster = _read_json(path)
    _require(roster.get("schema_version") == PAIR_ROSTER_SCHEMA, "配对清单 schema_version 不匹配")
    datasets = roster.get("datasets")
    _require(isinstance(datasets, dict) and set(datasets) == set(contract["datasets"]), "配对清单数据集集合不匹配")

    eligible: dict[str, list[tuple[str, str]]] = {}
    summary: dict[str, Any] = {}
    for dataset_id, spec in contract["datasets"].items():
        block = datasets[dataset_id]
        _require(isinstance(block, dict) and isinstance(block.get("rows"), list), f"{dataset_id} 配对清单缺少 rows")
        rows = block["rows"]
        expected_total = spec["expected_total_pair_count"]
        expected_analysis = spec["expected_analysis_pair_count"]
        expected_identities = spec["expected_identity_count"]
        _require(len(rows) == expected_total, f"{dataset_id} 总配对数不匹配：{len(rows)} != {expected_total}")

        seen_pairs: set[str] = set()
        identities: set[str] = set()
        identity_eligible_counts: dict[str, int] = {}
        eligible_rows: list[tuple[str, str]] = []
        structural_na_count = 0
        for row_index, row in enumerate(rows):
            _require(isinstance(row, dict), f"{dataset_id} rows[{row_index}] 不是对象")
            identity = row.get("identity_token")
            pair_id = row.get("pair_id")
            analysis_eligible = row.get("analysis_eligible")
            structural_state = row.get("structural_state")
            _require(isinstance(identity, str) and identity, f"{dataset_id} rows[{row_index}] identity_token 非法")
            _require(isinstance(pair_id, str) and pair_id, f"{dataset_id} rows[{row_index}] pair_id 非法")
            _require(pair_id not in seen_pairs, f"{dataset_id} 重复 pair_id：{pair_id}")
            _require(type(analysis_eligible) is bool, f"{dataset_id}/{pair_id} analysis_eligible 必须为布尔值")
            expected_state = "EVALUABLE" if analysis_eligible else "STRUCTURAL_NA"
            _require(structural_state == expected_state, f"{dataset_id}/{pair_id} structural_state 与可评价状态不一致")
            seen_pairs.add(pair_id)
            identities.add(identity)
            if analysis_eligible:
                eligible_rows.append((identity, pair_id))
                identity_eligible_counts[identity] = identity_eligible_counts.get(identity, 0) + 1
            else:
                structural_na_count += 1

        _require(len(identities) == expected_identities, f"{dataset_id} 身份数不匹配：{len(identities)} != {expected_identities}")
        _require(len(eligible_rows) == expected_analysis, f"{dataset_id} 可评价配对数不匹配：{len(eligible_rows)} != {expected_analysis}")
        missing_identity_support = sorted(identity for identity in identities if identity_eligible_counts.get(identity, 0) == 0)
        _require(not missing_identity_support, f"{dataset_id} 存在无可评价配对身份：{missing_identity_support}")
        eligible[dataset_id] = sorted(eligible_rows, key=lambda item: (item[0], item[1]))
        summary[dataset_id] = {
            "identity_count": len(identities),
            "total_pair_count": len(rows),
            "analysis_pair_count": len(eligible_rows),
            "structural_na_count": structural_na_count,
            "identity_analysis_pair_counts": dict(sorted(identity_eligible_counts.items())),
        }
    return eligible, summary


MetricKey = tuple[str, str, str, str, str, int | None]
MetricValue = float | None


def _expected_metric_keys(eligible: dict[str, list[tuple[str, str]]]) -> set[MetricKey]:
    keys: set[MetricKey] = set()
    for dataset_id, pairs in eligible.items():
        for identity, pair_id in pairs:
            for metric_id in METRIC_ORDER:
                for method_id in SEEDED_METHODS:
                    for seed in REQUIRED_SEEDS:
                        keys.add((dataset_id, metric_id, method_id, identity, pair_id, seed))
                for method_id in FIXED_METHODS:
                    keys.add((dataset_id, metric_id, method_id, identity, pair_id, None))
    return keys


def _load_metric_rows(
    contract: dict[str, Any],
    eligible: dict[str, list[tuple[str, str]]],
    path: Path,
) -> tuple[dict[MetricKey, MetricValue], dict[str, Any]]:
    pair_to_identity = {
        (dataset_id, pair_id): identity
        for dataset_id, pairs in eligible.items()
        for identity, pair_id in pairs
    }
    index: dict[MetricKey, MetricValue] = {}
    support_by_pair: dict[tuple[str, str], int] = {}
    sface_states_by_pair: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    failure_code_counts: dict[str, int] = {}
    line_count = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except Exception as exc:
        raise FailClosedError(f"无法打开指标 JSONL：{exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                raise FailClosedError(f"指标 JSONL 第 {line_number} 行解析失败：{exc}") from exc
            _require(isinstance(row, dict), f"指标 JSONL 第 {line_number} 行不是对象")
            _require(row.get("schema_version") == METRIC_ROW_SCHEMA, f"指标 JSONL 第 {line_number} 行 schema_version 不匹配")
            terminal_state = row.get("terminal_state")
            dataset_id = row.get("dataset_id")
            metric_id = row.get("metric_id")
            method_id = row.get("method_id")
            identity = row.get("identity_token")
            pair_id = row.get("pair_id")
            seed = row.get("seed")
            value = row.get("value")
            failure_code = row.get("failure_code")
            support = row.get("support_texels")
            _require(dataset_id in contract["datasets"], f"第 {line_number} 行未知 dataset_id：{dataset_id}")
            _require(metric_id in contract["metrics"], f"第 {line_number} 行未知 metric_id：{metric_id}")
            _require(method_id in contract["methods"], f"第 {line_number} 行未知 method_id：{method_id}")
            _require(isinstance(identity, str) and identity, f"第 {line_number} 行 identity_token 非法")
            _require(isinstance(pair_id, str) and pair_id, f"第 {line_number} 行 pair_id 非法")
            _require((dataset_id, pair_id) in pair_to_identity, f"第 {line_number} 行不是预定可评价配对：{dataset_id}/{pair_id}")
            _require(pair_to_identity[(dataset_id, pair_id)] == identity, f"第 {line_number} 行身份与配对清单不一致")
            if method_id in SEEDED_METHODS:
                _require(type(seed) is int and seed in REQUIRED_SEEDS, f"第 {line_number} 行训练方法 seed 非法")
            else:
                _require(seed is None, f"第 {line_number} 行 FreeUV seed 必须为 null")
            metric_spec = contract["metrics"][metric_id]
            minimum = metric_spec.get("minimum")
            maximum = metric_spec.get("maximum")
            if metric_id != "sface_source_to_render_cosine":
                _require(terminal_state == "COMPLETE", f"指标 JSONL 第 {line_number} 行尚未完成")
                _require(failure_code is None, f"第 {line_number} 行非 SFace 指标不得带 failure_code")
                _require(type(value) in (int, float) and math.isfinite(value), f"第 {line_number} 行 value 必须为有限数")
            elif terminal_state == "COMPLETE":
                _require(failure_code is None, f"第 {line_number} 行完成的 SFace 指标不得带 failure_code")
                _require(type(value) in (int, float) and math.isfinite(value), f"第 {line_number} 行完成的 SFace value 必须为有限数")
            elif terminal_state == "EVALUATION_FAILURE":
                allowed_failure_codes = {
                    "SOURCE_DETECTION_FAILURE",
                    "TARGET_DETECTION_FAILURE",
                    "METHOD_EMBEDDING_FAILURE",
                    "NONFINITE_EMBEDDING",
                }
                _require(value is None, f"第 {line_number} 行失败的 SFace value 必须为 null")
                _require(failure_code in allowed_failure_codes, f"第 {line_number} 行 SFace failure_code 非法")
                failure_code_counts[failure_code] = failure_code_counts.get(failure_code, 0) + 1
            else:
                raise FailClosedError(f"第 {line_number} 行 SFace terminal_state 非法或尚未完成")
            if value is not None and minimum is not None:
                _require(value >= minimum, f"第 {line_number} 行 {metric_id} 低于允许范围")
            if value is not None and maximum is not None:
                _require(value <= maximum, f"第 {line_number} 行 {metric_id} 高于允许范围")
            _require(type(support) is int and support > 0, f"第 {line_number} 行 support_texels 必须为正整数")
            pair_key = (dataset_id, pair_id)
            if pair_key in support_by_pair:
                _require(support_by_pair[pair_key] == support, f"{dataset_id}/{pair_id} 的支持数在方法或指标间变化")
            else:
                support_by_pair[pair_key] = support
            key: MetricKey = (dataset_id, metric_id, method_id, identity, pair_id, seed)
            _require(key not in index, f"重复指标行：{key}")
            index[key] = float(value) if value is not None else None
            if metric_id == "sface_source_to_render_cosine":
                sface_states_by_pair.setdefault((dataset_id, pair_id), []).append((terminal_state, failure_code))

    expected = _expected_metric_keys(eligible)
    observed = set(index)
    missing = expected - observed
    extra = observed - expected
    _require(not missing and not extra, f"指标键空间不完整：missing={len(missing)}, extra={len(extra)}")
    _require(line_count == len(expected), f"指标有效行数不匹配：{line_count} != {len(expected)}")
    expected_sface_rows_per_pair = len(SEEDED_METHODS) * len(REQUIRED_SEEDS) + len(FIXED_METHODS)
    shared_failure_codes = {"SOURCE_DETECTION_FAILURE", "TARGET_DETECTION_FAILURE"}
    for pair_key, states in sface_states_by_pair.items():
        _require(len(states) == expected_sface_rows_per_pair, f"{pair_key} 的 SFace ledger 行数不完整")
        observed_shared = {code for state, code in states if state == "EVALUATION_FAILURE" and code in shared_failure_codes}
        if observed_shared:
            _require(len(observed_shared) == 1, f"{pair_key} 同时出现多种共享检测失败")
            shared_code = next(iter(observed_shared))
            _require(
                all(state == "EVALUATION_FAILURE" and code == shared_code for state, code in states),
                f"{pair_key} 的源/目标检测失败没有对所有方法和种子对称记录",
            )
    coverage = {
        "expected_metric_row_count": len(expected),
        "observed_metric_row_count": line_count,
        "dataset_analysis_pairs": {dataset_id: len(pairs) for dataset_id, pairs in eligible.items()},
        "metric_count": len(METRIC_ORDER),
        "seeded_method_count": len(SEEDED_METHODS),
        "seed_count": len(REQUIRED_SEEDS),
        "fixed_method_count": len(FIXED_METHODS),
        "all_expected_keys_present": True,
        "sface_complete_row_count": sum(
            value is not None for key, value in index.items() if key[1] == "sface_source_to_render_cosine"
        ),
        "sface_failure_row_count": sum(failure_code_counts.values()),
        "sface_failure_code_counts": dict(sorted(failure_code_counts.items())),
        "sface_failure_ledger_complete": True,
    }
    return index, coverage


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "无法对空数组计算分位数")
    _require(0.0 <= probability <= 1.0, "分位概率必须位于 [0,1]")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_two_sided_sign_p(n_positive: int, n_negative: int) -> float:
    _require(type(n_positive) is int and n_positive >= 0, "n_positive 非法")
    _require(type(n_negative) is int and n_negative >= 0, "n_negative 非法")
    effective_n = n_positive + n_negative
    if effective_n == 0:
        return 1.0
    smaller = min(n_positive, n_negative)
    tail_numerator = sum(math.comb(effective_n, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail_numerator / (2 ** effective_n))


def holm_adjust(p_values: list[float]) -> list[float]:
    _require(bool(p_values), "Holm 校正需要至少一个 p 值")
    _require(all(type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0 for value in p_values), "Holm 输入 p 值非法")
    ordered = sorted(enumerate(float(value) for value in p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 0.0
    family_size = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_value)
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def identity_bootstrap_interval(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    _require(bool(values), "身份自助法不能使用空数组")
    _require(type(resamples) is int and resamples == 10000, "本合同固定 10,000 次自助抽样")
    rng = random.Random(seed)
    sample_size = len(values)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        estimates.append(float(statistics.median(sample)))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def _comparison_id(family_id: str, dataset_id: str, comparator_id: str) -> str:
    return f"{family_id}::{dataset_id}::full_vs_{comparator_id}"


def _compute_comparison(
    contract: dict[str, Any],
    family_id: str,
    metric_id: str,
    dataset_id: str,
    comparator_id: str,
    serial: int,
    eligible: dict[str, list[tuple[str, str]]],
    index: dict[MetricKey, MetricValue],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs_by_identity: dict[str, list[str]] = {}
    for identity, pair_id in eligible[dataset_id]:
        pairs_by_identity.setdefault(identity, []).append(pair_id)
    direction = contract["metrics"][metric_id]["direction"]
    identity_rows: list[dict[str, Any]] = []
    excluded_identities: list[str] = []
    for identity in sorted(pairs_by_identity):
        seed_effects: dict[str, float] = {}
        seed_relative_effects: dict[str, float | None] = {}
        seed_pair_counts: dict[str, int] = {}
        identity_has_all_seeds = True
        for seed in REQUIRED_SEEDS:
            pair_effects: list[float] = []
            comparator_values: list[float] = []
            for pair_id in sorted(pairs_by_identity[identity]):
                full_value = index[(dataset_id, metric_id, "full", identity, pair_id, seed)]
                comparator_seed = seed if comparator_id in SEEDED_METHODS else None
                comparator_value = index[(dataset_id, metric_id, comparator_id, identity, pair_id, comparator_seed)]
                if full_value is None or comparator_value is None:
                    continue
                if direction == "lower_is_better":
                    effect = comparator_value - full_value
                elif direction == "higher_is_better":
                    effect = full_value - comparator_value
                else:  # guarded by contract validation
                    raise FailClosedError(f"未知指标方向：{direction}")
                pair_effects.append(effect)
                comparator_values.append(comparator_value)
            seed_pair_counts[str(seed)] = len(pair_effects)
            if not pair_effects:
                identity_has_all_seeds = False
                break
            seed_effects[str(seed)] = float(statistics.median(pair_effects))
            comparator_center = float(statistics.median(comparator_values))
            seed_relative_effects[str(seed)] = (
                seed_effects[str(seed)] / comparator_center
                if direction == "lower_is_better" and comparator_center != 0.0
                else None
            )
        if not identity_has_all_seeds:
            excluded_identities.append(identity)
            continue
        identity_effect = float(statistics.median(seed_effects.values()))
        defined_relative = [value for value in seed_relative_effects.values() if value is not None]
        identity_relative_effect = (
            float(statistics.median(defined_relative))
            if direction == "lower_is_better" and len(defined_relative) == len(REQUIRED_SEEDS)
            else None
        )
        identity_rows.append(
            {
                "family_id": family_id,
                "comparison_id": _comparison_id(family_id, dataset_id, comparator_id),
                "metric_id": metric_id,
                "dataset_id": dataset_id,
                "comparator_id": comparator_id,
                "identity_token": identity,
                "analysis_pair_count": min(seed_pair_counts.values()),
                "seed_analysis_pair_counts": seed_pair_counts,
                "identity_effect": identity_effect,
                "identity_relative_effect": identity_relative_effect,
                "seed_effects": seed_effects,
                "seed_relative_effects": seed_relative_effects,
            }
        )

    _require(bool(identity_rows), f"{dataset_id}/{metric_id}/full_vs_{comparator_id} 没有可形成身份效应的完整记录")
    identity_effects = [row["identity_effect"] for row in identity_rows]
    positive = sum(value > 0 for value in identity_effects)
    negative = sum(value < 0 for value in identity_effects)
    zero = sum(value == 0 for value in identity_effects)
    bootstrap_spec = contract["inference"]["bootstrap"]
    bootstrap_seed = bootstrap_spec["base_seed"] + serial
    ci_low, ci_high = identity_bootstrap_interval(identity_effects, bootstrap_spec["resamples"], bootstrap_seed)
    seed_overall = {
        str(seed): float(statistics.median(row["seed_effects"][str(seed)] for row in identity_rows))
        for seed in REQUIRED_SEEDS
    }
    median_effect = float(statistics.median(identity_effects))
    relative_effects = [row["identity_relative_effect"] for row in identity_rows if row["identity_relative_effect"] is not None]
    minimum_identity_count = (
        contract["datasets"][dataset_id]["sface_min_identity_count"]
        if metric_id == "sface_source_to_render_cosine"
        else contract["datasets"][dataset_id]["expected_identity_count"]
    )
    confirmatory_coverage_eligible = len(identity_rows) >= minimum_identity_count
    result = {
        "family_id": family_id,
        "comparison_id": _comparison_id(family_id, dataset_id, comparator_id),
        "metric_id": metric_id,
        "dataset_id": dataset_id,
        "method_id": "full",
        "comparator_id": comparator_id,
        "favorable_direction": "positive",
        "identity_count": len(identity_rows),
        "minimum_confirmatory_identity_count": minimum_identity_count,
        "confirmatory_coverage_eligible": confirmatory_coverage_eligible,
        "excluded_identity_count": len(excluded_identities),
        "excluded_identities_no_complete_pair_all_five_seeds": excluded_identities,
        "median_identity_effect": median_effect,
        "median_identity_relative_effect": (
            float(statistics.median(relative_effects)) if relative_effects else None
        ),
        "relative_effect_defined_identity_count": len(relative_effects),
        "q1_identity_effect": quantile(identity_effects, 0.25),
        "q3_identity_effect": quantile(identity_effects, 0.75),
        "iqr_identity_effect": quantile(identity_effects, 0.75) - quantile(identity_effects, 0.25),
        "ci95_identity_bootstrap_unadjusted": [ci_low, ci_high],
        "bootstrap_resamples": bootstrap_spec["resamples"],
        "bootstrap_seed": bootstrap_seed,
        "n_positive": positive,
        "n_zero": zero,
        "n_negative": negative,
        "effective_sign_n": positive + negative,
        "sign_test": "two_sided_exact_binomial_p_0_5",
        "p_raw_two_sided_exact_sign": (
            exact_two_sided_sign_p(positive, negative) if confirmatory_coverage_eligible else None
        ),
        "p_holm_within_family": None,
        "seed_overall_median_effects": seed_overall,
        "all_five_seed_medians_positive": all(value > 0 for value in seed_overall.values()),
        "all_five_seed_medians_negative": all(value < 0 for value in seed_overall.values()),
    }
    if metric_id == "hidden_uv_mae":
        result["median_identity_effect_rgb8"] = median_effect * 255.0
    return result, identity_rows


def _compute_results(
    contract: dict[str, Any],
    eligible: dict[str, list[tuple[str, str]]],
    index: dict[MetricKey, MetricValue],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    serial = 0
    for family in contract["families"]:
        family_results: list[dict[str, Any]] = []
        for member in family["members"]:
            result, per_identity = _compute_comparison(
                contract=contract,
                family_id=family["family_id"],
                metric_id=family["metric_id"],
                dataset_id=member["dataset_id"],
                comparator_id=member["comparator_id"],
                serial=serial,
                eligible=eligible,
                index=index,
            )
            family_results.append(result)
            identity_rows.extend(per_identity)
            serial += 1
        holm_inputs = [
            item["p_raw_two_sided_exact_sign"]
            if item["p_raw_two_sided_exact_sign"] is not None
            else 1.0
            for item in family_results
        ]
        adjusted = holm_adjust(holm_inputs)
        for item, p_adjusted in zip(family_results, adjusted):
            item["holm_missing_comparison_preserved_as_nonrejection"] = item["p_raw_two_sided_exact_sign"] is None
            item["p_holm_within_family"] = (
                p_adjusted if item["p_raw_two_sided_exact_sign"] is not None else None
            )
            ci_low, ci_high = item["ci95_identity_bootstrap_unadjusted"]
            alpha = contract["inference"]["familywise_alpha"]
            item["confirmatory_full_favorable"] = bool(
                item["confirmatory_coverage_eligible"]
                and item["median_identity_effect"] > 0
                and ci_low > 0
                and item["p_holm_within_family"] is not None
                and item["p_holm_within_family"] < alpha
            )
            item["confirmatory_full_unfavorable"] = bool(
                item["confirmatory_coverage_eligible"]
                and item["median_identity_effect"] < 0
                and ci_high < 0
                and item["p_holm_within_family"] is not None
                and item["p_holm_within_family"] < alpha
            )
            item["confirmatory_indeterminate"] = not (
                item["confirmatory_full_favorable"] or item["confirmatory_full_unfavorable"]
            )
        comparisons.extend(family_results)
    _require(len(comparisons) == 18, f"预设比较总数必须为 18，实际为 {len(comparisons)}")
    return comparisons, identity_rows


def _csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _build_output_files(
    contract: dict[str, Any],
    binding_summary: dict[str, Any],
    roster_summary: dict[str, Any],
    coverage: dict[str, Any],
    comparisons: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
) -> dict[str, str]:
    result_document = {
        "schema_version": OUTPUT_SCHEMA,
        "contract_id": contract["contract_id"],
        "status": "PASS_COMPLETE_FOUR_FAMILIES",
        "analysis_unit": "identity",
        "seeds": REQUIRED_SEEDS,
        "comparison_count": len(comparisons),
        "families": [family["family_id"] for family in contract["families"]],
        "comparisons": comparisons,
    }
    identity_jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in identity_rows)

    seed_fields = [f"seed_{seed}" for seed in REQUIRED_SEEDS]
    identity_csv_rows = []
    for row in identity_rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in ("seed_effects", "seed_relative_effects", "seed_analysis_pair_counts")
        }
        for seed in REQUIRED_SEEDS:
            flat[f"seed_{seed}"] = row["seed_effects"][str(seed)]
        identity_csv_rows.append(flat)
    identity_csv_fields = [
        "family_id",
        "comparison_id",
        "metric_id",
        "dataset_id",
        "comparator_id",
        "identity_token",
        "analysis_pair_count",
        "identity_effect",
        "identity_relative_effect",
        *seed_fields,
    ]

    family_csv_rows = []
    seed_csv_rows = []
    gate_rows = []
    for result in comparisons:
        family_csv_rows.append(
            {
                "family_id": result["family_id"],
                "comparison_id": result["comparison_id"],
                "metric_id": result["metric_id"],
                "dataset_id": result["dataset_id"],
                "comparator_id": result["comparator_id"],
                "identity_count": result["identity_count"],
                "minimum_confirmatory_identity_count": result["minimum_confirmatory_identity_count"],
                "confirmatory_coverage_eligible": result["confirmatory_coverage_eligible"],
                "excluded_identity_count": result["excluded_identity_count"],
                "median_identity_effect": result["median_identity_effect"],
                "median_identity_relative_effect": result["median_identity_relative_effect"],
                "relative_effect_defined_identity_count": result["relative_effect_defined_identity_count"],
                "ci95_low": result["ci95_identity_bootstrap_unadjusted"][0],
                "ci95_high": result["ci95_identity_bootstrap_unadjusted"][1],
                "n_positive": result["n_positive"],
                "n_zero": result["n_zero"],
                "n_negative": result["n_negative"],
                "p_raw_two_sided_exact_sign": result["p_raw_two_sided_exact_sign"],
                "p_holm_within_family": result["p_holm_within_family"],
                "confirmatory_full_favorable": result["confirmatory_full_favorable"],
                "confirmatory_full_unfavorable": result["confirmatory_full_unfavorable"],
                "confirmatory_indeterminate": result["confirmatory_indeterminate"],
                "confirmatory_coverage_eligible": result["confirmatory_coverage_eligible"],
            }
        )
        for seed in REQUIRED_SEEDS:
            seed_csv_rows.append(
                {
                    "family_id": result["family_id"],
                    "comparison_id": result["comparison_id"],
                    "metric_id": result["metric_id"],
                    "dataset_id": result["dataset_id"],
                    "comparator_id": result["comparator_id"],
                    "seed": seed,
                    "median_identity_effect": result["seed_overall_median_effects"][str(seed)],
                }
            )
        gate_rows.append(
            {
                "comparison_id": result["comparison_id"],
                "family_id": result["family_id"],
                "dataset_id": result["dataset_id"],
                "metric_id": result["metric_id"],
                "comparator_id": result["comparator_id"],
                "confirmatory_full_favorable": result["confirmatory_full_favorable"],
                "confirmatory_full_unfavorable": result["confirmatory_full_unfavorable"],
                "confirmatory_indeterminate": result["confirmatory_indeterminate"],
                "all_five_seed_medians_positive": result["all_five_seed_medians_positive"],
                "all_five_seed_medians_negative": result["all_five_seed_medians_negative"],
            }
        )

    family_csv_fields = [
        "family_id",
        "comparison_id",
        "metric_id",
        "dataset_id",
        "comparator_id",
        "identity_count",
        "minimum_confirmatory_identity_count",
        "confirmatory_coverage_eligible",
        "excluded_identity_count",
        "median_identity_effect",
        "median_identity_relative_effect",
        "relative_effect_defined_identity_count",
        "ci95_low",
        "ci95_high",
        "n_positive",
        "n_zero",
        "n_negative",
        "p_raw_two_sided_exact_sign",
        "p_holm_within_family",
        "confirmatory_full_favorable",
        "confirmatory_full_unfavorable",
        "confirmatory_indeterminate",
    ]
    seed_csv_fields = [
        "family_id",
        "comparison_id",
        "metric_id",
        "dataset_id",
        "comparator_id",
        "seed",
        "median_identity_effect",
    ]
    coverage_document = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_EXACT_KEYSPACE_COMPLETE",
        "pair_roster": roster_summary,
        "metric_rows": coverage,
    }
    gate_document = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "DESCRIPTIVE_CLAIM_GATES_ONLY",
        "rule": "median direction, unadjusted 95% identity bootstrap interval, and within-family Holm p must all agree",
        "comparisons": gate_rows,
    }
    input_document = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_ALL_BOUND_INPUTS_VERIFIED",
        **binding_summary,
    }
    return {
        "ANALYSIS_RESULTS.json": _json_text(result_document),
        "IDENTITY_EFFECTS.jsonl": identity_jsonl,
        "IDENTITY_EFFECTS.csv": _csv_text(identity_csv_fields, identity_csv_rows),
        "FAMILY_RESULTS.csv": _csv_text(family_csv_fields, family_csv_rows),
        "SEED_RESULTS.csv": _csv_text(seed_csv_fields, seed_csv_rows),
        "METRIC_COVERAGE.json": _json_text(coverage_document),
        "CLAIM_GATE_RESULTS.json": _json_text(gate_document),
        "INPUT_VALIDATION.json": _json_text(input_document),
    }


def _write_output_atomically(
    output_root: Path,
    files: dict[str, str],
    contract: dict[str, Any],
    binding_summary: dict[str, Any],
) -> None:
    _require(not output_root.exists(), f"output_root 已存在：{output_root}")
    temp_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    _require(not temp_root.exists(), f"临时输出目录已存在：{temp_root}")
    temp_root.mkdir()
    for filename, content in files.items():
        (temp_root / filename).write_text(content, encoding="utf-8")
    output_hashes = {filename: sha256_file(temp_root / filename) for filename in sorted(files)}
    terminal = {
        "schema_version": OUTPUT_SCHEMA,
        "contract_id": contract["contract_id"],
        "status": "PASS_V14_MATCHED_CONTROLS_ANALYSIS_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_program_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": binding_summary["contract_sha256"],
        "binding_sha256": binding_summary["binding_sha256"],
        "metric_rows_sha256": binding_summary["metric_rows_sha256"],
        "pair_roster_sha256": binding_summary["pair_roster_sha256"],
        "seeds": REQUIRED_SEEDS,
        "bootstrap_resamples": 10000,
        "sign_test": "two_sided_exact_binomial_p_0_5",
        "multiple_comparison": "holm_within_each_prespecified_family",
        "family_ids": [family["family_id"] for family in contract["families"]],
        "output_sha256": output_hashes,
    }
    (temp_root / "ANALYSIS_TERMINAL.json").write_text(_json_text(terminal), encoding="utf-8")
    os.replace(temp_root, output_root)


def run_analysis(contract_path: Path, binding_path: Path, allow_synthetic: bool = False) -> Path:
    contract_path = _validated_regular_file(str(contract_path.resolve()), "contract")
    binding_path = _validated_regular_file(str(binding_path.resolve()), "binding")
    contract = _read_json(contract_path)
    _validate_contract(contract, allow_synthetic=allow_synthetic)
    binding_summary, paths = _validate_binding(contract_path, contract, binding_path)
    eligible, roster_summary = _load_pair_roster(contract, paths["pair_roster"])
    metric_index, coverage = _load_metric_rows(contract, eligible, paths["metric_rows"])
    comparisons, identity_rows = _compute_results(contract, eligible, metric_index)
    files = _build_output_files(contract, binding_summary, roster_summary, coverage, comparisons, identity_rows)
    _write_output_atomically(paths["output_root"], files, contract, binding_summary)
    return paths["output_root"]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("contract-check", help="只检查生产合同")
    check_parser.add_argument("--contract", required=True, type=Path)
    analyze_parser = subparsers.add_parser("analyze", help="执行冻结的生产分析")
    analyze_parser.add_argument("--contract", required=True, type=Path)
    analyze_parser.add_argument("--binding", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "contract-check":
            contract_path = _validated_regular_file(str(args.contract.resolve()), "contract")
            contract = _read_json(contract_path)
            _validate_contract(contract, allow_synthetic=False)
            print(
                json.dumps(
                    {
                        "status": "PASS_PRODUCTION_CONTRACT_VALID",
                        "contract_sha256": sha256_file(contract_path),
                        "seeds": REQUIRED_SEEDS,
                        "family_ids": [family["family_id"] for family in contract["families"]],
                        "bootstrap_resamples": 10000,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        output_root = run_analysis(args.contract, args.binding, allow_synthetic=False)
        print(json.dumps({"status": "PASS_ANALYSIS_COMPLETE", "output_root": str(output_root)}, ensure_ascii=False, sort_keys=True))
        return 0
    except FailClosedError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
