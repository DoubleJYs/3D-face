#!/usr/bin/env python3
"""Fail-closed source and contract checks for the V14 CUDA run package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import core  # noqa: E402
import run  # noqa: E402


REQUIRED_UPSTREAM_SOURCES = {
    "frugalface3d/models/structure_aware_b_qualification.py",
    "frugalface3d/models/structure_aware_b_mechanism_v1.py",
    "frugalface3d/models/uv_completion_lite.py",
    "frugalface3d/evaluation/fixed_alignment_sface.py",
    "frugalface3d/evaluation/mps_uv_pipeline.py",
    "frugalface3d/evaluation/paired_render_metrics.py",
    "reproducibility/w5b49n_mechanism_closure_v1/training/cache_io.py",
    "reproducibility/w5b49n_mechanism_closure_v1/runtime/eval_cache_io.py",
    "reproducibility/w5b49n_mechanism_closure_v1/runtime/realy_eval_cache_io.py",
    "reproducibility/w5b49n_mechanism_closure_v1/metrics/realy_final_metric_v1.py",
}
REQUIRED_PACKAGE_SOURCES = {
    "reproducibility/w5b49n_v14_matched_controls_v1/contract.json",
    "reproducibility/w5b49n_v14_matched_controls_v1/contract.v1.json",
    "reproducibility/w5b49n_v14_matched_controls_v1/core.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/run.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/postprocess.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/analyze_v14_matched_controls.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/source_check.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/test_synthetic.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/test_postprocess_synthetic.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/test_statistics_synthetic.py",
    "reproducibility/w5b49n_v14_matched_controls_v1/requirements.txt",
    "reproducibility/w5b49n_v14_matched_controls_v1/requirements-postfreeze.txt",
    "reproducibility/w5b49n_v14_matched_controls_v1/README.md",
    "reproducibility/w5b49n_v14_matched_controls_v1/STATISTICS_README.md",
    "reproducibility/w5b49n_v14_matched_controls_v1/INPUT_BINDING.template.json",
    "reproducibility/w5b49n_v14_matched_controls_v1/POSTPROCESS_INPUT_BINDING.template.json",
}
REQUIRED_PROTECTED_MANUSCRIPTS = {
    "paper_rewriting_output/final_paper_w49n_v13_zh_working/main.zh.md",
    "paper_rewriting_output/final_paper_w49n_v14_zh_working/main.zh.md",
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"source_check:{label}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_object:{path.name}")
    return value


def verify_hash_rows(repository_root: Path, rows: Any, label: str) -> int:
    require(isinstance(rows, list) and rows, f"{label}_list")
    count = 0
    for row in rows:
        require(isinstance(row, dict), f"{label}_row")
        relative = row.get("path")
        expected = row.get("sha256")
        require(isinstance(relative, str), f"{label}_path")
        require(isinstance(expected, str) and len(expected) == 64, f"{label}_sha")
        path = (repository_root / relative).resolve(strict=True)
        path.relative_to(repository_root)
        require(core.sha256_file(path) == expected, f"{label}_hash:{relative}")
        count += 1
    return count


def check_contract() -> dict[str, Any]:
    contract = read_json(PACKAGE_ROOT / "contract.json")
    require(
        contract.get("schema_version")
        == "frugalface3d.w5b49n.v14_matched_controls.v1",
        "contract_schema",
    )
    require(
        contract.get("status") == "SOURCE_PACKAGE_ONLY_NO_SCIENTIFIC_RESULT",
        "contract_status",
    )
    require(tuple(contract["seed_plans"]["cuda_five"]) == core.SEED_PLANS["cuda_five"], "seed_plan")
    matrix = contract["training_matrix"]["cuda_five"]
    require(matrix["historical_mps_full_units_used_for_confirmation"] == 0, "no_mps_full")
    require(matrix["new_full_units"] == 5, "five_new_full")
    require(matrix["new_condition0_units"] == 5, "five_condition0")
    require(matrix["new_b_lite_ft_units"] == 5, "five_b_lite_ft")
    require(matrix["new_training_units"] == 15, "fifteen_units")
    require(matrix["new_optimizer_steps"] == 7680, "optimizer_steps")
    schedule = run.build_schedule(contract, "cuda_five")
    require(len(schedule) == 15, "schedule_count")
    require(all(row["action"] == "TRAIN" for row in schedule), "schedule_all_train")
    require(
        {(row["method"], row["seed"]) for row in schedule}
        == {
            (method, seed)
            for method in core.METHODS
            for seed in core.SEED_PLANS["cuda_five"]
        },
        "schedule_matrix",
    )
    run_source = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    core_source = (PACKAGE_ROOT / "core.py").read_text(encoding="utf-8")
    require("load_structure_checkpoint" not in core_source, "historical_full_loader_absent")
    require("checkpoint_origin\": \"repository_root" not in run_source, "repository_checkpoint_route_absent")
    require('origin == "repository_root"' not in run_source, "repository_checkpoint_loader_absent")
    require("matched_control_formal_device_must_be_cuda" in core_source, "formal_cuda_guard")
    require("torch.zeros_like(geometry)" in core_source, "condition0_geometry_zero")
    require("torch.zeros_like(expression)" in core_source, "condition0_expression_zero")
    return contract


def check_runtime(contract: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    import torch

    device = torch.device("cpu")
    full = core.new_structure_model(device=device, trainable=True)
    require(core.parameter_count(full) == 89_386, "full_parameter_count")
    b_lite_path = repository_root / contract["b_lite_checkpoint"]["path"]
    if b_lite_path.is_file():
        require(
            core.sha256_file(b_lite_path) == contract["b_lite_checkpoint"]["sha256"],
            "b_lite_checkpoint_hash",
        )
        b_lite = core.load_b_lite(
            b_lite_path,
            device=device,
            trainable=False,
            expected_sha256=contract["b_lite_checkpoint"]["sha256"],
        )
        require(core.parameter_count(b_lite, trainable_only=False) == 122_164, "b_lite_parameter_count")
    else:
        from frugalface3d.models.uv_completion_lite import (
            UVCompletionLite,
            count_uv_completion_parameters,
        )

        b_lite = UVCompletionLite(core.exact_b_lite_config())
        require(count_uv_completion_parameters(b_lite) == 122_164, "b_lite_parameter_count")
    formal_cpu_rejected = False
    try:
        core.configure_runtime("cpu", formal=True)
    except RuntimeError as error:
        formal_cpu_rejected = str(error) == "matched_control_formal_device_must_be_cuda"
    require(formal_cpu_rejected, "formal_cpu_rejected")
    return {
        "torch_version": str(torch.__version__),
        "full_parameters": 89_386,
        "b_lite_parameters": 122_164,
        "b_lite_checkpoint_present": b_lite_path.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.expanduser().resolve(strict=True)
    contract = check_contract()
    source_lock = read_json(PACKAGE_ROOT / "source_lock.json")
    require(
        source_lock.get("schema_version")
        == "frugalface3d.w5b49n.v14_matched_controls.source_lock.v1",
        "source_lock_schema",
    )
    source_rows = source_lock.get("source_files")
    package_rows = source_lock.get("package_execution_files")
    protected_rows = source_lock.get("protected_manuscripts")
    require(
        isinstance(source_rows, list)
        and {row.get("path") for row in source_rows} == REQUIRED_UPSTREAM_SOURCES,
        "source_lock_upstream_keyspace",
    )
    require(
        isinstance(package_rows, list)
        and {row.get("path") for row in package_rows} == REQUIRED_PACKAGE_SOURCES,
        "source_lock_package_keyspace",
    )
    require(
        isinstance(protected_rows, list)
        and {row.get("path") for row in protected_rows}
        == REQUIRED_PROTECTED_MANUSCRIPTS,
        "source_lock_protected_keyspace",
    )
    source_count = verify_hash_rows(repository_root, source_rows, "source")
    package_count = verify_hash_rows(repository_root, package_rows, "package")
    protected_count = verify_hash_rows(
        repository_root, protected_rows, "protected"
    )
    runtime = check_runtime(contract, repository_root) if args.runtime else None
    print(
        json.dumps(
            {
                "status": "PASS_V14_MATCHED_CONTROL_SOURCE_CHECK",
                "source_files_verified": source_count,
                "package_execution_files_verified": package_count,
                "protected_manuscripts_verified": protected_count,
                "new_training_units": 15,
                "new_optimizer_steps": 7680,
                "historical_mps_full_units_used_for_confirmation": 0,
                "runtime": runtime,
                "scientific_result_generated": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
