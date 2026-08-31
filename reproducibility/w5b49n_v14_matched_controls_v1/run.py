#!/usr/bin/env python3
"""Plan, train, and run inference for the V14 matched-control matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (  # noqa: E402
    METHOD_BLITE_FT,
    METHOD_CONDITION0,
    METHOD_FULL,
    METHODS,
    SEED_PLANS,
    STEPS,
    atomic_torch_save,
    canonical_json_bytes,
    configure_runtime,
    infer_samples,
    load_trained_control,
    runtime_fingerprint,
    sha256_file,
    train_one,
    write_json,
)


PROGRAM_ID = "FRUGALFACE3D-W5B49N-V14-MATCHED-CONTROLS-V1"
UNIT_TERMINAL_SCHEMA = "frugalface3d.w5b49n.v14.terminal.v1"
EXPECTED_FORMAL_DEVICE_NAME = "NVIDIA GeForce RTX 4090"


def read_contract() -> dict[str, Any]:
    value = json.loads((PACKAGE_ROOT / "contract.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != "frugalface3d.w5b49n.v14_matched_controls.v1":
        raise RuntimeError("matched_control_contract_schema")
    return value


def method_slug(method: str) -> str:
    return {
        METHOD_FULL: "full",
        METHOD_CONDITION0: "condition0",
        METHOD_BLITE_FT: "b_lite_ft",
    }[method]


def unit_status(method: str, *, development: bool) -> str:
    if development:
        return "DEVELOPMENT_ONLY_NOT_FOR_CONFIRMATORY_ANALYSIS"
    return {
        METHOD_FULL: "PASS_V14_FULL_SEED_COMPLETE",
        METHOD_CONDITION0: "PASS_V14_CONDITION0_SEED_COMPLETE",
        METHOD_BLITE_FT: "PASS_V14_B_LITE_FT_SEED_COMPLETE",
    }[method]


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bound_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_cuda_smoke(path: Path, runtime: Mapping[str, Any]) -> tuple[Path, str]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "frugalface3d.w5b49n.v14.cuda_smoke.v1",
        "status": "PASS_V14_MATCHED_CONTROL_SYNTHETIC",
        "device_backend": "cuda",
        "device_name": EXPECTED_FORMAL_DEVICE_NAME,
        "optimizer_steps": 3,
        "condition0_joint_zeroing_verified": True,
        "observed_uv_exactness_verified": True,
        "scientific_result_generated": False,
        "contract_sha256": sha256_file(PACKAGE_ROOT / "contract.json"),
        "source_lock_sha256": sha256_file(PACKAGE_ROOT / "source_lock.json"),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("matched_control_cuda_smoke_contract")
    tested = value.get("tested_methods")
    if not isinstance(tested, list) or [row.get("method") for row in tested] != list(METHODS):
        raise RuntimeError("matched_control_cuda_smoke_methods")
    if value.get("runtime_fingerprint") != runtime:
        raise RuntimeError("matched_control_cuda_smoke_runtime_changed")
    return resolved, sha256_file(resolved)


def build_schedule(contract: Mapping[str, Any], seed_plan: str) -> list[dict[str, Any]]:
    seeds = SEED_PLANS[seed_plan]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in seeds:
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "action": "TRAIN",
                    "checkpoint_origin": "training_root",
                    "checkpoint_path": (
                        f"units/{method_slug(method)}/{seed}/step_{STEPS:04d}.pt"
                    ),
                }
            )
    return rows


def resolve_repository_asset(repository_root: Path, relative: str) -> Path:
    candidate = (repository_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(repository_root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError("matched_control_repository_asset_escape") from error
    return candidate


def command_plan(args: argparse.Namespace) -> int:
    contract = read_contract()
    schedule = build_schedule(contract, args.seed_plan)
    result = {
        "program_id": PROGRAM_ID,
        "seed_plan": args.seed_plan,
        "seeds": list(SEED_PLANS[args.seed_plan]),
        "total_method_seed_routes": len(schedule),
        "reused_units": 0,
        "new_training_units": sum(row["action"] == "TRAIN" for row in schedule),
        "new_optimizer_steps": sum(row["action"] == "TRAIN" for row in schedule)
        * STEPS,
        "schedule": schedule,
        "scientific_result_generated": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from reproducibility.w5b49n_mechanism_closure_v1.training.cache_io import (
        MANIFEST_FILE,
        TENSOR_FILE,
        load_cache,
    )

    contract = read_contract()
    repository_root = args.repository_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("matched_control_training_output_exists_no_automatic_retry")
    fit_cache = args.fit_cache.expanduser().resolve(strict=True)
    cache = load_cache(fit_cache, exact=True)
    b_binding = contract["b_lite_checkpoint"]
    b_lite_checkpoint = (
        args.b_lite_checkpoint.expanduser().resolve(strict=True)
        if args.b_lite_checkpoint is not None
        else resolve_repository_asset(repository_root, str(b_binding["path"]))
    )
    b_lite_sha256 = str(b_binding["sha256"])
    if sha256_file(b_lite_checkpoint) != b_lite_sha256:
        raise RuntimeError("matched_control_b_lite_checkpoint_binding")
    schedule = build_schedule(contract, args.seed_plan)
    device = configure_runtime(args.device, formal=not args.development)
    runtime_start = runtime_fingerprint(args.device)
    if not args.development and runtime_start.get("cuda_device_name") != EXPECTED_FORMAL_DEVICE_NAME:
        raise RuntimeError(
            "matched_control_formal_device_name:"
            f"{runtime_start.get('cuda_device_name')}!={EXPECTED_FORMAL_DEVICE_NAME}"
        )
    cuda_smoke_path: Path | None = None
    cuda_smoke_sha256: str | None = None
    if not args.development:
        if args.cuda_smoke_receipt is None:
            raise RuntimeError("matched_control_cuda_smoke_receipt_required")
        cuda_smoke_path, cuda_smoke_sha256 = validate_cuda_smoke(
            args.cuda_smoke_receipt, runtime_start
        )
    output_root.mkdir(parents=True, mode=0o700)
    contract_path = PACKAGE_ROOT / "contract.json"
    source_lock_path = PACKAGE_ROOT / "source_lock.json"
    environment_manifest_path = output_root / "ENVIRONMENT_MANIFEST.json"
    split_manifest_path = output_root / "TRAINING_SPLIT_MANIFEST.json"
    budget_manifest_path = output_root / "TRAINING_BUDGET_MANIFEST.json"
    fit_train_rows = [row for row in cache.rows if row["partition"] == "fit_train"]
    fit_validation_rows = [
        row for row in cache.rows if row["partition"] == "fit_validation"
    ]
    write_json(
        environment_manifest_path,
        {
            "schema_version": "frugalface3d.w5b49n.v14.environment_manifest.v1",
            "program_id": PROGRAM_ID,
            "runtime_fingerprint": runtime_start,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "contract_sha256": sha256_file(contract_path),
            "source_lock_sha256": sha256_file(source_lock_path),
            "cuda_smoke_receipt_sha256": cuda_smoke_sha256,
            "formal_cuda_campaign": not args.development,
        },
    )
    write_json(
        split_manifest_path,
        {
            "schema_version": "frugalface3d.w5b49n.v14.training_split_manifest.v1",
            "fit_cache_manifest_sha256": sha256_file(fit_cache / MANIFEST_FILE),
            "fit_cache_tensor_sha256": sha256_file(fit_cache / TENSOR_FILE),
            "row_count": len(cache.rows),
            "fit_train_count": len(fit_train_rows),
            "fit_validation_count": len(fit_validation_rows),
            "eligible_train_count": 238,
            "anonymous_rows_sha256": json_sha256(list(cache.rows)),
            "eligibility_rule": "canonical_mask*(1-source_visibility)*paired_visibility has at least one texel",
        },
    )
    write_json(
        budget_manifest_path,
        {
            "schema_version": "frugalface3d.w5b49n.v14.training_budget_manifest.v1",
            "methods": list(METHODS),
            "seeds": list(SEED_PLANS[args.seed_plan]),
            "steps_per_unit": STEPS,
            "training_units": 15,
            "optimizer_steps": 15 * STEPS,
            "training_contract": contract["training"],
            "automatic_retry": False,
            "checkpoint_selection": False,
        },
    )
    manifest_hashes = {
        "environment_manifest_sha256": sha256_file(environment_manifest_path),
        "training_split_sha256": sha256_file(split_manifest_path),
        "training_budget_manifest_sha256": sha256_file(budget_manifest_path),
    }
    b_lite_frozen_terminal_path = output_root / "B_LITE_FROZEN_TERMINAL.json"
    write_json(
        b_lite_frozen_terminal_path,
        {
            "schema_version": UNIT_TERMINAL_SCHEMA,
            "status": "PASS_V14_B_LITE_FROZEN_BOUND",
            "method_id": "b_lite",
            "checkpoint_sha256": b_lite_sha256,
            "bound_files": [bound_file(b_lite_checkpoint)],
        },
    )
    attempt = {
        "schema_version": "frugalface3d.w5b49n.v14_matched_training_attempt.v1",
        "status": "STARTED",
        "program_id": PROGRAM_ID,
        "seed_plan": args.seed_plan,
        "device": args.device,
        "development_only": bool(args.development),
        "automatic_retry": False,
        "checkpoint_selection": False,
        "runtime_fingerprint": runtime_start,
        "cuda_smoke_receipt_path": str(cuda_smoke_path) if cuda_smoke_path else None,
        "cuda_smoke_receipt_sha256": cuda_smoke_sha256,
        **manifest_hashes,
        "fit_cache_manifest_sha256": sha256_file(fit_cache / MANIFEST_FILE),
        "fit_cache_tensor_sha256": sha256_file(fit_cache / TENSOR_FILE),
        "b_lite_checkpoint_sha256": b_lite_sha256,
        "schedule": schedule,
    }
    write_json(output_root / "ATTEMPT.json", attempt)
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for row in schedule:
            method = str(row["method"])
            seed = int(row["seed"])
            unit_dir = output_root / "units" / method_slug(method) / str(seed)
            trained = train_one(
                cache,
                method=method,
                seed=seed,
                device=device,
                device_name=args.device,
                b_lite_checkpoint=b_lite_checkpoint,
                b_lite_sha256=b_lite_sha256,
                output=unit_dir,
                steps=STEPS,
                expected_eligible=238,
            )
            trained["action"] = "TRAIN"
            trained["checkpoint_path"] = (
                unit_dir / str(trained["checkpoint_path"])
            ).relative_to(output_root).as_posix()
            checkpoint_path = output_root / str(trained["checkpoint_path"])
            unit_terminal_path = unit_dir / "UNIT_TERMINAL.json"
            write_json(
                unit_terminal_path,
                {
                    "schema_version": UNIT_TERMINAL_SCHEMA,
                    "status": unit_status(method, development=bool(args.development)),
                    "method_id": method_slug(method),
                    "method_label": method,
                    "seed": seed,
                    "device_backend": str(runtime_start["device_type"]),
                    "device_name": str(
                        runtime_start.get("cuda_device_name", "DEVELOPMENT_ONLY")
                    ),
                    "torch_version": str(runtime_start["torch_version"]),
                    "torch_cuda_version": str(runtime_start["torch_cuda_version"]),
                    "cudnn_version": runtime_start["cudnn_version"],
                    "runtime_fingerprint": runtime_start,
                    "training_steps": STEPS,
                    "optimizer_steps": STEPS,
                    "trainable_parameters": trained["trainable_parameters"],
                    "automatic_retry": False,
                    "checkpoint_selection": False,
                    "formal_training": not args.development,
                    "development_only": bool(args.development),
                    **manifest_hashes,
                    "checkpoint_sha256": trained["checkpoint_sha256"],
                    "bound_files": [
                        bound_file(environment_manifest_path),
                        bound_file(split_manifest_path),
                        bound_file(budget_manifest_path),
                        bound_file(checkpoint_path),
                        *(
                            [bound_file(cuda_smoke_path)]
                            if cuda_smoke_path is not None
                            else []
                        ),
                    ],
                },
            )
            trained["unit_terminal_path"] = unit_terminal_path.relative_to(
                output_root
            ).as_posix()
            trained["unit_terminal_sha256"] = sha256_file(unit_terminal_path)
            completed.append(trained)
            gc.collect()
            if args.device == "cuda":
                import torch

                torch.cuda.empty_cache()
            print(
                f"MATCHED_CONTROL_TRAIN_PROGRESS={len(completed)}/{len(schedule)} "
                f"method={method} seed={seed} action={row['action']}",
                flush=True,
            )
        trained_count = len(completed)
        runtime_end = runtime_fingerprint(args.device)
        if runtime_end != runtime_start:
            raise RuntimeError("matched_control_runtime_fingerprint_changed")
        terminal = {
            "schema_version": "frugalface3d.w5b49n.v14_matched_training_terminal.v1",
            "status": (
                "MATCHED_CONTROL_TRAINING_COMPLETE"
                if not args.development
                else "DEVELOPMENT_ONLY_TRAINING_COMPLETE"
            ),
            "program_id": PROGRAM_ID,
            "seed_plan": args.seed_plan,
            "seeds": list(SEED_PLANS[args.seed_plan]),
            "device": args.device,
            "development_only": bool(args.development),
            "method_seed_route_count": len(completed),
            "historical_mps_full_units_used_for_confirmation": 0,
            "new_training_units": trained_count,
            "optimizer_steps": trained_count * STEPS,
            "automatic_retry": False,
            "retry_count": 0,
            "checkpoint_selection": False,
            "fit_cache_manifest_sha256": attempt["fit_cache_manifest_sha256"],
            "fit_cache_tensor_sha256": attempt["fit_cache_tensor_sha256"],
            "b_lite_checkpoint_sha256": b_lite_sha256,
            "runtime_fingerprint": runtime_end,
            **manifest_hashes,
            "b_lite_frozen_terminal_sha256": sha256_file(
                b_lite_frozen_terminal_path
            ),
            "cuda_smoke_receipt_sha256": cuda_smoke_sha256,
            "elapsed_seconds": time.perf_counter() - started,
            "rows": completed,
        }
        write_json(output_root / "TRAINING_TERMINAL.json", terminal)
        print(
            json.dumps(
                {
                    "status": terminal["status"],
                    "new_training_units": trained_count,
                    "optimizer_steps": terminal["optimizer_steps"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        write_json(
            output_root / "FAILURE.json",
            {
                "status": "FAILED_ROOT_RETAINED_NO_AUTOMATIC_RETRY",
                "program_id": PROGRAM_ID,
                "completed_routes": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
                "checkpoint_selection": False,
            },
        )
        raise


def load_training_terminal(training_root: Path) -> Mapping[str, Any]:
    value = json.loads(
        (training_root / "TRAINING_TERMINAL.json").read_text(encoding="utf-8")
    )
    if (
        value.get("schema_version")
        != "frugalface3d.w5b49n.v14_matched_training_terminal.v1"
        or value.get("status") != "MATCHED_CONTROL_TRAINING_COMPLETE"
        or value.get("device") != "cuda"
        or value.get("development_only") is not False
        or value.get("historical_mps_full_units_used_for_confirmation") != 0
        or value.get("new_training_units") != 15
        or value.get("optimizer_steps") != 15 * STEPS
    ):
        raise RuntimeError("matched_control_training_terminal_status")
    seeds = tuple(int(value_) for value_ in value.get("seeds", []))
    if seeds not in SEED_PLANS.values():
        raise RuntimeError("matched_control_training_terminal_seeds")
    rows = value.get("rows")
    expected = {(method, seed) for method in METHODS for seed in seeds}
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected)
        or {
            (str(row.get("method")), int(row.get("seed", -1))) for row in rows
        }
        != expected
    ):
        raise RuntimeError("matched_control_training_terminal_matrix")
    for row in rows:
        method = str(row.get("method"))
        seed = int(row.get("seed", -1))
        if (
            row.get("action") != "TRAIN"
            or row.get("checkpoint_origin") != "training_root"
            or row.get("optimizer_steps") != STEPS
            or row.get("selection_or_best_of_n") is not False
            or row.get("automatic_retry") is not False
        ):
            raise RuntimeError("matched_control_training_terminal_row")
        unit_relative = str(row.get("unit_terminal_path"))
        unit_path = (training_root / unit_relative).resolve(strict=True)
        try:
            unit_path.relative_to(training_root)
        except ValueError as error:
            raise RuntimeError("matched_control_unit_terminal_escape") from error
        if sha256_file(unit_path) != row.get("unit_terminal_sha256"):
            raise RuntimeError("matched_control_unit_terminal_sha256")
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        if (
            unit.get("schema_version") != UNIT_TERMINAL_SCHEMA
            or unit.get("status") != unit_status(method, development=False)
            or unit.get("method_id") != method_slug(method)
            or unit.get("seed") != seed
            or unit.get("device_backend") != "cuda"
            or unit.get("device_name") != EXPECTED_FORMAL_DEVICE_NAME
            or unit.get("training_steps") != STEPS
            or unit.get("formal_training") is not True
            or unit.get("development_only") is not False
            or unit.get("checkpoint_sha256") != row.get("checkpoint_sha256")
        ):
            raise RuntimeError("matched_control_unit_terminal_contract")
    return value


def resolve_checkpoint_row(row: Mapping[str, Any], *, training_root: Path) -> Path:
    origin = row.get("checkpoint_origin")
    relative = str(row.get("checkpoint_path"))
    if origin == "training_root":
        path = (training_root / relative).resolve(strict=True)
        try:
            path.relative_to(training_root)
        except ValueError as error:
            raise RuntimeError("matched_control_training_checkpoint_escape") from error
        return path
    raise RuntimeError("matched_control_checkpoint_origin")


def command_infer(args: argparse.Namespace) -> int:
    import torch

    if args.dataset == "facescape":
        from reproducibility.w5b49n_mechanism_closure_v1.runtime.eval_cache_io import (
            MANIFEST_FILE,
            TENSOR_FILE,
            load_eval_cache,
        )

        cache_loader = load_eval_cache
        expected_count = 160
    else:
        from reproducibility.w5b49n_mechanism_closure_v1.runtime.realy_eval_cache_io import (
            MANIFEST_FILE,
            TENSOR_FILE,
            load_realy_source_cache,
        )

        cache_loader = load_realy_source_cache
        expected_count = 400
    training_root = args.training_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("matched_control_inference_output_exists_no_automatic_retry")
    terminal = load_training_terminal(training_root)
    cache_root = args.eval_cache.expanduser().resolve(strict=True)
    cache = cache_loader(cache_root)
    if len(cache.rows) != expected_count:
        raise RuntimeError("matched_control_inference_cache_count")
    device = configure_runtime(args.device, formal=not args.development)
    output_root.mkdir(parents=True, mode=0o700)
    attempt = {
        "schema_version": "frugalface3d.w5b49n.v14_matched_inference_attempt.v1",
        "status": "STARTED",
        "program_id": PROGRAM_ID,
        "dataset": args.dataset,
        "device": args.device,
        "development_only": bool(args.development),
        "source_sample_count": len(cache.rows),
        "target_pair_reads": 0,
        "cache_manifest_sha256": sha256_file(cache_root / MANIFEST_FILE),
        "cache_tensor_sha256": sha256_file(cache_root / TENSOR_FILE),
        "training_terminal_sha256": sha256_file(
            training_root / "TRAINING_TERMINAL.json"
        ),
    }
    write_json(output_root / "ATTEMPT.json", attempt)
    routes: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for row in terminal["rows"]:
            method = str(row["method"])
            seed = int(row["seed"])
            checkpoint = resolve_checkpoint_row(
                row, training_root=training_root
            )
            expected_sha = str(row["checkpoint_sha256"])
            model = load_trained_control(
                method, checkpoint, device=device, expected_sha256=expected_sha
            )
            native, conserved = infer_samples(
                cache,
                method=method,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )
            route_dir = output_root / "routes" / method_slug(method) / str(seed)
            route_dir.mkdir(parents=True, exist_ok=False)
            payload_path = route_dir / "RAW_OUTPUTS.pt"
            atomic_torch_save(
                payload_path,
                {
                    "native": native,
                    "conserved": conserved,
                },
            )
            route = {
                "schema_version": "frugalface3d.w5b49n.v14_matched_raw_route.v1",
                "status": "SUCCESS",
                "dataset": args.dataset,
                "method": method,
                "paper_label": "NoStruct" if method == METHOD_CONDITION0 else method,
                "seed": seed,
                "sample_count": len(cache.rows),
                "checkpoint_sha256": expected_sha,
                "raw_output_path": payload_path.relative_to(output_root).as_posix(),
                "raw_output_sha256": sha256_file(payload_path),
                "source_observed_uv_exact": True,
                "hidden_native_equals_conserved": True,
                "target_pair_reads": 0,
            }
            write_json(route_dir / "ROUTE_TERMINAL.json", route)
            routes.append(route)
            del model, native, conserved
            if args.device == "cuda":
                torch.cuda.empty_cache()
            print(
                f"MATCHED_CONTROL_INFER_PROGRESS={len(routes)}/{len(terminal['rows'])} "
                f"dataset={args.dataset} method={method} seed={seed}",
                flush=True,
            )
        result = {
            "schema_version": "frugalface3d.w5b49n.v14_matched_inference_terminal.v1",
            "status": "MATCHED_CONTROL_INFERENCE_COMPLETE",
            "program_id": PROGRAM_ID,
            "dataset": args.dataset,
            "device": args.device,
            "development_only": bool(args.development),
            "source_sample_count": len(cache.rows),
            "method_seed_route_count": len(routes),
            "target_pair_reads": 0,
            "automatic_retry": False,
            "cache_manifest_sha256": attempt["cache_manifest_sha256"],
            "cache_tensor_sha256": attempt["cache_tensor_sha256"],
            "training_terminal_sha256": attempt["training_terminal_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "routes": routes,
        }
        write_json(output_root / "INFERENCE_TERMINAL.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "dataset": args.dataset,
                    "method_seed_route_count": len(routes),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        write_json(
            output_root / "FAILURE.json",
            {
                "status": "FAILED_ROOT_RETAINED_NO_AUTOMATIC_RETRY",
                "dataset": args.dataset,
                "completed_routes": routes,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
                "target_pair_reads": 0,
            },
        )
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--seed-plan", choices=tuple(SEED_PLANS), default="cuda_five")
    plan.set_defaults(function=command_plan)

    train = subparsers.add_parser("train")
    train.add_argument("--seed-plan", choices=tuple(SEED_PLANS), default="cuda_five")
    train.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    train.add_argument("--fit-cache", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    train.add_argument("--b-lite-checkpoint", type=Path)
    train.add_argument("--cuda-smoke-receipt", type=Path)
    train.add_argument("--development", action="store_true")
    train.set_defaults(function=command_train)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    infer.add_argument("--dataset", choices=("facescape", "realy"), required=True)
    infer.add_argument("--eval-cache", type=Path, required=True)
    infer.add_argument("--training-root", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    infer.add_argument("--batch-size", type=int, default=24)
    infer.add_argument("--development", action="store_true")
    infer.set_defaults(function=command_infer)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
