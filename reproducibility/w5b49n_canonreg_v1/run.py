#!/usr/bin/env python3
"""Plan, train, infer, and close the five-seed CanonReg sensitivity run."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from canonreg_loss import compute_training_loss
from integrity import (
    CONTRACT_PATH,
    PACKAGE_ROOT,
    SOURCE_MANIFEST_PATH,
    bound_file,
    discover_repository_root,
    json_sha256,
    load_historical_core,
    package_binding,
    read_contract,
    sha256_file,
    verify_package_manifest,
    write_json,
)


PROGRAM_ID = "FRUGALFACE3D-V15-REVIEW-CANONREG-V1"
METHOD_ID = "canonreg"
METHOD_LABEL = "CanonReg"
SEEDS = (2026080447, 2026080448, 2026080449, 2026080450, 2026080451)
STEPS = 512
BATCH_SIZE = 24
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
TRAINABLE_PARAMETERS = 89_386
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 4090"
TRAINING_SCHEMA = "frugalface3d.review_closure.canonreg.training_terminal.v1"
UNIT_SCHEMA = "frugalface3d.review_closure.canonreg.unit_terminal.v1"
INFERENCE_SCHEMA = "frugalface3d.review_closure.canonreg.inference_terminal.v1"


def _context(repository_root: Path | None) -> tuple[Path, dict[str, Any], Any]:
    root = discover_repository_root(repository_root)
    contract = read_contract()
    verify_package_manifest()
    core = load_historical_core(root, contract)
    _verify_fixed_contract(contract, core)
    return root, contract, core


def _verify_fixed_contract(contract: Mapping[str, Any], core: Any) -> None:
    training = contract["training"]
    variant = contract["variant"]
    if (
        contract.get("program_id") != PROGRAM_ID
        or tuple(contract.get("seed_plan", [])) != SEEDS
        or variant.get("reference_method") != "Full"
        or variant.get("architecture_change") is not False
        or variant.get("trainable_parameters") != TRAINABLE_PARAMETERS
        or training.get("optimizer") != "AdamW"
        or training.get("steps_per_unit") != STEPS
        or training.get("batch_size") != BATCH_SIZE
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("weight_decay") != WEIGHT_DECAY
        or training.get("gradient_clip_l2") != GRADIENT_CLIP
        or training.get("training_units") != 5
        or training.get("optimizer_steps_total") != 5 * STEPS
        or training.get("automatic_retry") is not False
        or training.get("checkpoint_selection") is not False
        or training.get("loss_weights") != core.LOSS_WEIGHTS
    ):
        raise RuntimeError("canonreg_fixed_contract_changed")


def _runtime(core: Any) -> tuple[Any, dict[str, Any]]:
    device = core.configure_runtime("cuda", formal=True)
    fingerprint = core.runtime_fingerprint("cuda")
    expected = read_contract()["formal_environment"]
    observed = {
        "python_version": platform.python_version(),
        "torch_version": fingerprint.get("torch_version"),
        "device_backend": fingerprint.get("device_type"),
        "device_name": fingerprint.get("cuda_device_name"),
        "cuda_version": fingerprint.get("torch_cuda_version"),
        "cudnn_version": fingerprint.get("cudnn_version"),
        "compute_capability": fingerprint.get("cuda_compute_capability"),
        "deterministic_algorithms": fingerprint.get(
            "deterministic_algorithms_enabled"
        ),
        "tf32": False,
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise RuntimeError(
                f"canonreg_formal_environment:{key}:{value}!={expected.get(key)}"
            )
    return device, fingerprint


def _resolve_repository_asset(repository_root: Path, relative: str) -> Path:
    path = (repository_root / relative).resolve(strict=True)
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise RuntimeError("canonreg_repository_asset_escape") from error
    return path


def _schedule() -> list[dict[str, Any]]:
    return [
        {
            "method": METHOD_ID,
            "seed": seed,
            "action": "TRAIN",
            "steps": STEPS,
            "checkpoint_path": f"units/{METHOD_ID}/{seed}/step_{STEPS:04d}.pt",
        }
        for seed in SEEDS
    ]


def _validate_cuda_smoke(
    path: Path, runtime: Mapping[str, Any], binding: Mapping[str, str], contract: Mapping[str, Any]
) -> tuple[Path, str]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    core_hash = contract["upstream_source_lock"][0]["sha256"]
    expected = {
        "schema_version": "frugalface3d.review_closure.canonreg.cuda_smoke.v1",
        "status": "PASS_CANONREG_SYNTHETIC_SMOKE",
        "device_backend": "cuda",
        "device_name": EXPECTED_DEVICE_NAME,
        "runtime_fingerprint": dict(runtime),
        "optimizer_steps": 1,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "contract_sha256": binding["contract_sha256"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "historical_core_sha256": core_hash,
        "scientific_result_generated": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"canonreg_cuda_smoke_binding:{resolved}")
    if value.get("regularizer_contract_verified") is not True:
        raise RuntimeError("canonreg_cuda_smoke_regularizer_contract")
    if value.get("observed_uv_exactness_verified") is not True:
        raise RuntimeError("canonreg_cuda_smoke_observed_uv")
    return resolved, sha256_file(resolved)


def command_plan(args: argparse.Namespace) -> int:
    _root, contract, _core = _context(args.repository_root)
    result = {
        "status": "CANONREG_PLAN_FROZEN",
        "program_id": PROGRAM_ID,
        "method": METHOD_ID,
        "reference_method": contract["variant"]["reference_method"],
        "seeds": list(SEEDS),
        "new_training_units": 5,
        "steps_per_unit": STEPS,
        "new_optimizer_steps": 5 * STEPS,
        "automatic_retry": False,
        "checkpoint_selection": False,
        "changed_terms_only": contract["variant"]["changed_terms_only"],
        "datasets_for_raw_inference": ["facescape", "realy"],
        "private_cache_bindings": contract["private_inputs"],
        "schedule": _schedule(),
        "scientific_result_generated": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _train_one(
    core: Any,
    cache: Any,
    *,
    seed: int,
    device: Any,
    output: Path,
) -> dict[str, Any]:
    import torch

    train_positions = [
        index for index, row in enumerate(cache.rows) if row["partition"] == "fit_train"
    ]
    pair_map, donor_map = core.pair_and_donor_maps(cache.rows, train_positions)
    support = {
        index: int(
            (
                cache.tensors["canonical_mask"][index]
                * (1.0 - cache.tensors["visibility"][index])
                * cache.tensors["visibility"][pair_map[index]]
            ).sum()
        )
        for index in train_positions
    }
    eligible = [index for index in train_positions if support[index] > 0]
    if len(eligible) != 238:
        raise RuntimeError(f"canonreg_eligible_fit_rows:{len(eligible)}")
    core.seed_all(seed, "cuda")
    model = core.new_structure_model(device=device, trainable=True)
    if core.parameter_count(model) != TRAINABLE_PARAMETERS:
        raise RuntimeError("canonreg_trainable_parameter_count")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for zero_step in range(STEPS):
        local = core.deterministic_epoch_batch_indices(
            sample_count=len(eligible),
            zero_based_step=zero_step,
            seed=seed,
            batch_size=BATCH_SIZE,
        )
        anchors = [eligible[int(index)] for index in local]
        pairs = [pair_map[index] for index in anchors]
        donors = [donor_map[index] for index in anchors]
        batch = core.model_batch(cache, [*anchors, *pairs, *donors], device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms, _output, diagnostics = compute_training_loss(
            core, model, batch, anchor_count=len(anchors)
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("canonreg_loss_nonfinite")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not math.isfinite(float(gradient.detach().cpu())):
            raise RuntimeError("canonreg_gradient_nonfinite")
        optimizer.step()
        core.synchronize("cuda")
        trace.append(
            {
                "step": zero_step + 1,
                "loss": float(loss.detach().cpu()),
                "hidden_uv_l1": float(terms["paired_hidden_UV_L1"].detach().cpu()),
                "bounded_residual_l2": float(
                    terms["bounded_residual_L2"].detach().cpu()
                ),
                "total_variation": float(terms["total_variation"].detach().cpu()),
                "residual_support_texels_min": float(
                    diagnostics["residual_support_texels"].min().detach().cpu()
                ),
                "canonical_edges_x_min": float(
                    diagnostics["canonical_edges_x"].min().detach().cpu()
                ),
                "canonical_edges_y_min": float(
                    diagnostics["canonical_edges_y"].min().detach().cpu()
                ),
                "gradient_norm_before_clip": float(gradient.detach().cpu()),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / f"step_{STEPS:04d}.pt"
    core.atomic_torch_save(
        checkpoint,
        {
            "schema_version": "frugalface3d.review_closure.canonreg.checkpoint.v1",
            "method": core.METHOD_FULL,
            "variant": METHOD_ID,
            "seed": seed,
            "step": STEPS,
            "optimizer_steps": STEPS,
            "selection_or_best_of_n": False,
            "automatic_retry": False,
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "changed_terms_only": ["bounded_residual_L2", "total_variation"],
            "model_config": asdict(model.config),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
        },
    )
    trace_path = output / "TRAIN_TRACE.json"
    write_json(trace_path, trace)
    return {
        "method": METHOD_ID,
        "reference_architecture": "Full",
        "seed": seed,
        "action": "TRAIN",
        "optimizer_steps": STEPS,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "checkpoint_origin": "training_root",
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "trace_path": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_or_best_of_n": False,
        "automatic_retry": False,
    }


def command_train(args: argparse.Namespace) -> int:
    repository_root, contract, core = _context(args.repository_root)
    from reproducibility.w5b49n_mechanism_closure_v1.training.cache_io import (
        MANIFEST_FILE,
        TENSOR_FILE,
        load_cache,
    )

    binding = package_binding()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("canonreg_training_output_exists_no_automatic_retry")
    fit_cache = args.fit_cache.expanduser().resolve(strict=True)
    cache = load_cache(fit_cache, exact=True)
    fit_binding = contract["private_inputs"]["fit_cache"]
    if (
        sha256_file(fit_cache / MANIFEST_FILE) != fit_binding["manifest_sha256"]
        or sha256_file(fit_cache / TENSOR_FILE) != fit_binding["tensor_sha256"]
    ):
        raise RuntimeError("canonreg_fit_cache_frozen_hash_binding")
    checkpoint_binding = contract["b_lite_checkpoint"]
    b_lite_checkpoint = (
        args.b_lite_checkpoint.expanduser().resolve(strict=True)
        if args.b_lite_checkpoint is not None
        else _resolve_repository_asset(
            repository_root, checkpoint_binding["repository_relative_path"]
        )
    )
    if sha256_file(b_lite_checkpoint) != checkpoint_binding["sha256"]:
        raise RuntimeError("canonreg_b_lite_checkpoint_binding")
    device, runtime_start = _runtime(core)
    smoke_path, smoke_sha256 = _validate_cuda_smoke(
        args.cuda_smoke_receipt, runtime_start, binding, contract
    )
    fit_train_rows = [row for row in cache.rows if row["partition"] == "fit_train"]
    fit_validation_rows = [
        row for row in cache.rows if row["partition"] == "fit_validation"
    ]
    if len(fit_train_rows) != 240 or len(fit_validation_rows) != 48:
        raise RuntimeError("canonreg_fit_partition_counts")
    output_root.mkdir(parents=True, mode=0o700)
    environment_path = output_root / "ENVIRONMENT_MANIFEST.json"
    split_path = output_root / "TRAINING_SPLIT_MANIFEST.json"
    budget_path = output_root / "TRAINING_BUDGET_MANIFEST.json"
    write_json(
        environment_path,
        {
            "schema_version": "frugalface3d.review_closure.canonreg.environment.v1",
            "program_id": PROGRAM_ID,
            "runtime_fingerprint": runtime_start,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "formal_cuda_campaign": True,
            "cuda_smoke_receipt_sha256": smoke_sha256,
            **binding,
        },
    )
    write_json(
        split_path,
        {
            "schema_version": "frugalface3d.review_closure.canonreg.split.v1",
            "fit_cache_manifest_sha256": sha256_file(fit_cache / MANIFEST_FILE),
            "fit_cache_tensor_sha256": sha256_file(fit_cache / TENSOR_FILE),
            "row_count": len(cache.rows),
            "fit_train_count": len(fit_train_rows),
            "fit_validation_count": len(fit_validation_rows),
            "eligible_fit_train_count": 238,
            "anonymous_rows_sha256": json_sha256(list(cache.rows)),
            "eligibility_rule": "Mcanon*(1-Vs)*Vpaired contains at least one texel",
        },
    )
    write_json(
        budget_path,
        {
            "schema_version": "frugalface3d.review_closure.canonreg.budget.v1",
            "method": METHOD_ID,
            "seeds": list(SEEDS),
            "steps_per_unit": STEPS,
            "training_units": 5,
            "optimizer_steps": 5 * STEPS,
            "training_contract": contract["training"],
            "regularization_definition": contract["regularization_definition"],
            "automatic_retry": False,
            "checkpoint_selection": False,
        },
    )
    manifest_hashes = {
        "environment_manifest_sha256": sha256_file(environment_path),
        "training_split_manifest_sha256": sha256_file(split_path),
        "training_budget_manifest_sha256": sha256_file(budget_path),
    }
    attempt = {
        "schema_version": "frugalface3d.review_closure.canonreg.training_attempt.v1",
        "status": "STARTED",
        "program_id": PROGRAM_ID,
        "method": METHOD_ID,
        "device": "cuda",
        "automatic_retry": False,
        "checkpoint_selection": False,
        "runtime_fingerprint": runtime_start,
        "cuda_smoke_receipt_path": str(smoke_path),
        "cuda_smoke_receipt_sha256": smoke_sha256,
        "fit_cache_manifest_sha256": sha256_file(fit_cache / MANIFEST_FILE),
        "fit_cache_tensor_sha256": sha256_file(fit_cache / TENSOR_FILE),
        "b_lite_checkpoint_sha256": checkpoint_binding["sha256"],
        "schedule": _schedule(),
        **binding,
        **manifest_hashes,
    }
    write_json(output_root / "ATTEMPT.json", attempt)
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            unit_dir = output_root / "units" / METHOD_ID / str(seed)
            trained = _train_one(
                core, cache, seed=seed, device=device, output=unit_dir
            )
            checkpoint_path = unit_dir / trained["checkpoint_path"]
            trace_path = unit_dir / trained["trace_path"]
            trained["checkpoint_path"] = checkpoint_path.relative_to(
                output_root
            ).as_posix()
            trained["trace_path"] = trace_path.relative_to(output_root).as_posix()
            unit_terminal_path = unit_dir / "UNIT_TERMINAL.json"
            write_json(
                unit_terminal_path,
                {
                    "schema_version": UNIT_SCHEMA,
                    "status": "PASS_CANONREG_SEED_COMPLETE",
                    "program_id": PROGRAM_ID,
                    "method_id": METHOD_ID,
                    "reference_architecture": "Full",
                    "seed": seed,
                    "device_backend": "cuda",
                    "device_name": runtime_start["cuda_device_name"],
                    "runtime_fingerprint": runtime_start,
                    "training_steps": STEPS,
                    "optimizer_steps": STEPS,
                    "trainable_parameters": TRAINABLE_PARAMETERS,
                    "automatic_retry": False,
                    "checkpoint_selection": False,
                    "checkpoint_sha256": trained["checkpoint_sha256"],
                    "trace_sha256": trained["trace_sha256"],
                    "changed_terms_only": [
                        "bounded_residual_L2",
                        "total_variation",
                    ],
                    **binding,
                    **manifest_hashes,
                    "bound_files": [
                        bound_file(environment_path),
                        bound_file(split_path),
                        bound_file(budget_path),
                        bound_file(checkpoint_path),
                        bound_file(trace_path),
                        bound_file(smoke_path),
                    ],
                },
            )
            trained["unit_terminal_path"] = unit_terminal_path.relative_to(
                output_root
            ).as_posix()
            trained["unit_terminal_sha256"] = sha256_file(unit_terminal_path)
            completed.append(trained)
            gc.collect()
            import torch

            torch.cuda.empty_cache()
            print(
                f"CANONREG_TRAIN_PROGRESS={len(completed)}/5 seed={seed}",
                flush=True,
            )
        runtime_end = core.runtime_fingerprint("cuda")
        if runtime_end != runtime_start:
            raise RuntimeError("canonreg_runtime_fingerprint_changed")
        terminal = {
            "schema_version": TRAINING_SCHEMA,
            "status": "CANONREG_FIVE_SEED_TRAINING_COMPLETE",
            "program_id": PROGRAM_ID,
            "method": METHOD_ID,
            "reference_architecture": "Full",
            "seeds": list(SEEDS),
            "device": "cuda",
            "new_training_units": len(completed),
            "optimizer_steps": len(completed) * STEPS,
            "trainable_parameters_per_unit": TRAINABLE_PARAMETERS,
            "automatic_retry": False,
            "retry_count": 0,
            "checkpoint_selection": False,
            "fit_cache_manifest_sha256": attempt["fit_cache_manifest_sha256"],
            "fit_cache_tensor_sha256": attempt["fit_cache_tensor_sha256"],
            "b_lite_checkpoint_sha256": checkpoint_binding["sha256"],
            "runtime_fingerprint": runtime_end,
            "cuda_smoke_receipt_sha256": smoke_sha256,
            "elapsed_seconds": time.perf_counter() - started,
            "rows": completed,
            **binding,
            **manifest_hashes,
        }
        write_json(output_root / "TRAINING_TERMINAL.json", terminal)
        print(
            json.dumps(
                {
                    "status": terminal["status"],
                    "new_training_units": len(completed),
                    "optimizer_steps": len(completed) * STEPS,
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


def _validate_training_terminal(
    training_root: Path, contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    terminal_path = training_root / "TRAINING_TERMINAL.json"
    value = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != TRAINING_SCHEMA
        or value.get("status") != "CANONREG_FIVE_SEED_TRAINING_COMPLETE"
        or value.get("method") != METHOD_ID
        or value.get("reference_architecture") != "Full"
        or tuple(value.get("seeds", [])) != SEEDS
        or value.get("device") != "cuda"
        or value.get("new_training_units") != 5
        or value.get("optimizer_steps") != 5 * STEPS
        or value.get("trainable_parameters_per_unit") != TRAINABLE_PARAMETERS
        or value.get("automatic_retry") is not False
        or value.get("retry_count") != 0
        or value.get("checkpoint_selection") is not False
        or value.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or value.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH)
    ):
        raise RuntimeError("canonreg_training_terminal_contract")
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 5
        or {int(row.get("seed", -1)) for row in rows} != set(SEEDS)
    ):
        raise RuntimeError("canonreg_training_terminal_rows")
    for row in rows:
        if (
            row.get("method") != METHOD_ID
            or row.get("reference_architecture") != "Full"
            or row.get("action") != "TRAIN"
            or row.get("optimizer_steps") != STEPS
            or row.get("trainable_parameters") != TRAINABLE_PARAMETERS
            or row.get("checkpoint_origin") != "training_root"
            or row.get("selection_or_best_of_n") is not False
            or row.get("automatic_retry") is not False
        ):
            raise RuntimeError("canonreg_training_terminal_row_contract")
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("trace_path", "trace_sha256"),
            ("unit_terminal_path", "unit_terminal_sha256"),
        ):
            path = (training_root / str(row[path_key])).resolve(strict=True)
            try:
                path.relative_to(training_root)
            except ValueError as error:
                raise RuntimeError("canonreg_training_artifact_escape") from error
            if sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"canonreg_training_artifact_hash:{path_key}")
        unit_path = training_root / str(row["unit_terminal_path"])
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        if (
            unit.get("schema_version") != UNIT_SCHEMA
            or unit.get("status") != "PASS_CANONREG_SEED_COMPLETE"
            or unit.get("seed") != row.get("seed")
            or unit.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or unit.get("trace_sha256") != row.get("trace_sha256")
        ):
            raise RuntimeError("canonreg_unit_terminal_contract")
    if value.get("b_lite_checkpoint_sha256") != contract["b_lite_checkpoint"]["sha256"]:
        raise RuntimeError("canonreg_training_b_lite_binding")
    return value


def command_infer(args: argparse.Namespace) -> int:
    import torch

    repository_root, contract, core = _context(args.repository_root)
    if str(repository_root) not in os.sys.path:
        os.sys.path.insert(0, str(repository_root))
    if args.dataset == "facescape":
        from reproducibility.w5b49n_mechanism_closure_v1.runtime.eval_cache_io import (
            MANIFEST_FILE,
            TENSOR_FILE,
            load_eval_cache,
        )

        loader = load_eval_cache
        expected_count = 160
        dataset_token = "D1"
    else:
        from reproducibility.w5b49n_mechanism_closure_v1.runtime.realy_eval_cache_io import (
            MANIFEST_FILE,
            TENSOR_FILE,
            load_realy_source_cache,
        )

        loader = load_realy_source_cache
        expected_count = 400
        dataset_token = "D2"
    training_root = args.training_root.expanduser().resolve(strict=True)
    training_terminal = _validate_training_terminal(training_root, contract)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("canonreg_inference_output_exists_no_automatic_retry")
    cache_root = args.eval_cache.expanduser().resolve(strict=True)
    cache = loader(cache_root)
    if len(cache.rows) != expected_count:
        raise RuntimeError("canonreg_inference_cache_count")
    cache_binding = contract["private_inputs"][
        "facescape_eval_cache" if args.dataset == "facescape" else "realy_eval_cache"
    ]
    if (
        sha256_file(cache_root / MANIFEST_FILE) != cache_binding["manifest_sha256"]
        or sha256_file(cache_root / TENSOR_FILE) != cache_binding["tensor_sha256"]
    ):
        raise RuntimeError(f"canonreg_{args.dataset}_cache_frozen_hash_binding")
    device, runtime_start = _runtime(core)
    if runtime_start != training_terminal["runtime_fingerprint"]:
        raise RuntimeError("canonreg_inference_runtime_differs_from_training")
    output_root.mkdir(parents=True, mode=0o700)
    attempt = {
        "schema_version": "frugalface3d.review_closure.canonreg.inference_attempt.v1",
        "status": "STARTED",
        "program_id": PROGRAM_ID,
        "dataset": args.dataset,
        "dataset_token": dataset_token,
        "device": "cuda",
        "source_sample_count": len(cache.rows),
        "target_pair_reads": 0,
        "automatic_retry": False,
        "cache_manifest_sha256": sha256_file(cache_root / MANIFEST_FILE),
        "cache_tensor_sha256": sha256_file(cache_root / TENSOR_FILE),
        "training_terminal_sha256": sha256_file(
            training_root / "TRAINING_TERMINAL.json"
        ),
        **package_binding(),
    }
    write_json(output_root / "ATTEMPT.json", attempt)
    routes: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for row in sorted(training_terminal["rows"], key=lambda item: int(item["seed"])):
            seed = int(row["seed"])
            checkpoint = (training_root / str(row["checkpoint_path"])).resolve(
                strict=True
            )
            model = core.load_trained_control(
                core.METHOD_FULL,
                checkpoint,
                device=device,
                expected_sha256=str(row["checkpoint_sha256"]),
            )
            native, conserved = core.infer_samples(
                cache,
                method=core.METHOD_FULL,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )
            route_dir = output_root / "routes" / METHOD_ID / str(seed)
            route_dir.mkdir(parents=True, exist_ok=False)
            payload_path = route_dir / "RAW_OUTPUTS.pt"
            core.atomic_torch_save(
                payload_path,
                {"native": native, "conserved": conserved},
            )
            route_receipt_path = route_dir / "ROUTE_TERMINAL.json"
            route_receipt = {
                "schema_version": "frugalface3d.review_closure.canonreg.raw_route.v1",
                "status": "SUCCESS",
                "program_id": PROGRAM_ID,
                "dataset": args.dataset,
                "dataset_token": dataset_token,
                "method": METHOD_ID,
                "reference_architecture": "Full",
                "seed": seed,
                "sample_count": len(cache.rows),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "raw_output_path": payload_path.relative_to(output_root).as_posix(),
                "raw_output_sha256": sha256_file(payload_path),
                "source_observed_uv_exact": True,
                "hidden_native_equals_conserved": True,
                "target_pair_reads": 0,
            }
            write_json(route_receipt_path, route_receipt)
            routes.append(
                {
                    **route_receipt,
                    "route_terminal_path": route_receipt_path.relative_to(
                        output_root
                    ).as_posix(),
                    "route_terminal_sha256": sha256_file(route_receipt_path),
                }
            )
            del model, native, conserved
            torch.cuda.empty_cache()
            print(
                f"CANONREG_INFER_PROGRESS={len(routes)}/5 dataset={args.dataset} seed={seed}",
                flush=True,
            )
        runtime_end = core.runtime_fingerprint("cuda")
        if runtime_end != runtime_start:
            raise RuntimeError("canonreg_inference_runtime_changed")
        result = {
            "schema_version": INFERENCE_SCHEMA,
            "status": "CANONREG_INFERENCE_COMPLETE",
            "program_id": PROGRAM_ID,
            "dataset": args.dataset,
            "dataset_token": dataset_token,
            "device": "cuda",
            "runtime_fingerprint": runtime_end,
            "source_sample_count": len(cache.rows),
            "method_seed_route_count": len(routes),
            "target_pair_reads": 0,
            "automatic_retry": False,
            "cache_manifest_sha256": attempt["cache_manifest_sha256"],
            "cache_tensor_sha256": attempt["cache_tensor_sha256"],
            "training_terminal_sha256": attempt["training_terminal_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "routes": routes,
            **package_binding(),
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


def _validate_inference_terminal(
    root: Path,
    *,
    dataset: str,
    expected_count: int,
    training_terminal_sha256: str,
) -> Mapping[str, Any]:
    path = root / "INFERENCE_TERMINAL.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != INFERENCE_SCHEMA
        or value.get("status") != "CANONREG_INFERENCE_COMPLETE"
        or value.get("dataset") != dataset
        or value.get("device") != "cuda"
        or value.get("source_sample_count") != expected_count
        or value.get("method_seed_route_count") != 5
        or value.get("target_pair_reads") != 0
        or value.get("automatic_retry") is not False
        or value.get("training_terminal_sha256") != training_terminal_sha256
        or value.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or value.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH)
    ):
        raise RuntimeError(f"canonreg_inference_terminal_contract:{dataset}")
    routes = value.get("routes")
    if (
        not isinstance(routes, list)
        or len(routes) != 5
        or {int(route.get("seed", -1)) for route in routes} != set(SEEDS)
    ):
        raise RuntimeError(f"canonreg_inference_routes:{dataset}")
    for route in routes:
        for path_key, hash_key in (
            ("raw_output_path", "raw_output_sha256"),
            ("route_terminal_path", "route_terminal_sha256"),
        ):
            artifact = (root / str(route[path_key])).resolve(strict=True)
            try:
                artifact.relative_to(root)
            except ValueError as error:
                raise RuntimeError("canonreg_inference_artifact_escape") from error
            if sha256_file(artifact) != route[hash_key]:
                raise RuntimeError(f"canonreg_inference_artifact_hash:{dataset}:{path_key}")
        if (
            route.get("method") != METHOD_ID
            or route.get("reference_architecture") != "Full"
            or route.get("sample_count") != expected_count
            or route.get("source_observed_uv_exact") is not True
            or route.get("hidden_native_equals_conserved") is not True
            or route.get("target_pair_reads") != 0
        ):
            raise RuntimeError(f"canonreg_inference_route_contract:{dataset}")
    return value


def command_close(args: argparse.Namespace) -> int:
    _repository_root, contract, _core = _context(args.repository_root)
    training_root = args.training_root.expanduser().resolve(strict=True)
    d1_root = args.d1_root.expanduser().resolve(strict=True)
    d2_root = args.d2_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("canonreg_closure_output_exists_no_overwrite")
    _validate_training_terminal(training_root, contract)
    training_terminal_path = training_root / "TRAINING_TERMINAL.json"
    training_sha = sha256_file(training_terminal_path)
    d1 = _validate_inference_terminal(
        d1_root,
        dataset="facescape",
        expected_count=160,
        training_terminal_sha256=training_sha,
    )
    d2 = _validate_inference_terminal(
        d2_root,
        dataset="realy",
        expected_count=400,
        training_terminal_sha256=training_sha,
    )
    output_root.mkdir(parents=True, mode=0o700)
    artifacts = [
        bound_file(training_terminal_path),
        bound_file(d1_root / "INFERENCE_TERMINAL.json"),
        bound_file(d2_root / "INFERENCE_TERMINAL.json"),
        *[
            bound_file(d1_root / route["raw_output_path"])
            for route in d1["routes"]
        ],
        *[
            bound_file(d2_root / route["raw_output_path"])
            for route in d2["routes"]
        ],
    ]
    artifact_manifest_path = output_root / "ARTIFACT_MANIFEST.json"
    write_json(
        artifact_manifest_path,
        {
            "schema_version": "frugalface3d.review_closure.canonreg.artifacts.v1",
            "status": "BOUND",
            "program_id": PROGRAM_ID,
            "files": artifacts,
            "ordered_file_rows_sha256": json_sha256(artifacts),
        },
    )
    terminal = {
        "schema_version": "frugalface3d.review_closure.canonreg.closure_terminal.v1",
        "status": "PASS_CANONREG_TRAIN_D1_D2_RAW_CLOSURE",
        "program_id": PROGRAM_ID,
        "method": METHOD_ID,
        "reference_architecture": "Full",
        "seeds": list(SEEDS),
        "training_units": 5,
        "optimizer_steps": 5 * STEPS,
        "d1_raw_routes": 5,
        "d1_source_samples_per_route": 160,
        "d2_raw_routes": 5,
        "d2_source_samples_per_route": 400,
        "pair_metrics_generated": False,
        "automatic_retry": False,
        "checkpoint_selection": False,
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        **package_binding(),
    }
    write_json(output_root / "CLOSURE_TERMINAL.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--repository-root", type=Path)
    commands = value.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.set_defaults(function=command_plan)

    train = commands.add_parser("train")
    train.add_argument("--fit-cache", type=Path, required=True)
    train.add_argument("--b-lite-checkpoint", type=Path)
    train.add_argument("--cuda-smoke-receipt", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.set_defaults(function=command_train)

    infer = commands.add_parser("infer")
    infer.add_argument("--dataset", choices=("facescape", "realy"), required=True)
    infer.add_argument("--eval-cache", type=Path, required=True)
    infer.add_argument("--training-root", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    infer.set_defaults(function=command_infer)

    close = commands.add_parser("close")
    close.add_argument("--training-root", type=Path, required=True)
    close.add_argument("--d1-root", type=Path, required=True)
    close.add_argument("--d2-root", type=Path, required=True)
    close.add_argument("--output-root", type=Path, required=True)
    close.set_defaults(function=command_close)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
