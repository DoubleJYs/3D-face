#!/usr/bin/env python3
"""Small CPU-only synthetic test; it produces no scientific result."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import core  # noqa: E402
import run  # noqa: E402


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"synthetic_test:{label}")


def make_cache() -> SimpleNamespace:
    import torch

    generator = torch.Generator().manual_seed(8675309)
    rows = []
    for identity in ("ID_A", "ID_B"):
        for view in ("VIEW_L", "VIEW_R"):
            rows.append(
                {
                    "sample_index": len(rows),
                    "partition": "fit_train",
                    "identity_token": identity,
                    "expression_token": "EXP_0",
                    "view_token": view,
                }
            )
    count = len(rows)
    canonical = torch.ones(count, 1, 64, 64)
    visibility = torch.zeros(count, 1, 64, 64)
    for index, row in enumerate(rows):
        if row["view_token"] == "VIEW_L":
            visibility[index, :, :, :32] = 1.0
        else:
            visibility[index, :, :, 32:] = 1.0
    full_texture = torch.rand(count, 3, 64, 64, generator=generator)
    partial = full_texture * visibility
    tensors = {
        "partial_uv": partial,
        "visibility": visibility,
        "geometry_map": torch.rand(count, 6, 64, 64, generator=generator) * 2.0 - 1.0,
        "canonical_mask": canonical,
        "base_completion": torch.rand(count, 3, 64, 64, generator=generator),
        "texture_feature": torch.rand(count, 160, 16, 16, generator=generator),
        "expression_token": torch.rand(count, 128, generator=generator),
    }
    return SimpleNamespace(rows=tuple(rows), tensors=tensors, manifest={"synthetic": True})


def main() -> int:
    import torch
    from frugalface3d.models.uv_completion_lite import UVCompletionLite

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = core.configure_runtime(args.device, formal=args.device == "cuda")
    runtime = core.runtime_fingerprint(args.device)
    if args.device == "cuda":
        require(
            runtime.get("cuda_device_name") == run.EXPECTED_FORMAL_DEVICE_NAME,
            "cuda_device_name",
        )
    cache = make_cache()
    contract = run.read_contract()
    schedule = run.build_schedule(contract, "cuda_five")
    require(len(schedule) == 15, "confirmatory_schedule_count")
    require(all(row["action"] == "TRAIN" for row in schedule), "confirmatory_schedule_all_new")

    batch = core.model_batch(cache, [0, 1], device)
    altered = {name: value.clone() for name, value in batch.items()}
    altered["geometry_map"] = altered["geometry_map"] + 7.0
    altered["expression_token"] = altered["expression_token"] - 5.0
    probe = core.new_structure_model(device=device, trainable=False)
    with torch.inference_mode():
        left = core.forward_matched(probe, core.METHOD_CONDITION0, batch)
        right = core.forward_matched(probe, core.METHOD_CONDITION0, altered)
    require(torch.equal(left.completed_uv, right.completed_uv), "condition0_output_invariant")
    require(torch.equal(left.identity_embedding, right.identity_embedding), "condition0_embedding_invariant")

    completed = []
    with tempfile.TemporaryDirectory(prefix="v14-matched-synthetic-") as temporary:
        root = Path(temporary)
        b_lite_checkpoint = root / "B_LITE_SYNTHETIC.pt"
        b_lite = UVCompletionLite(core.exact_b_lite_config())
        core.atomic_torch_save(b_lite_checkpoint, {"state_dict": b_lite.state_dict()})
        b_lite_sha256 = core.sha256_file(b_lite_checkpoint)
        for offset, method in enumerate(core.METHODS):
            unit = root / run.method_slug(method)
            row = core.train_one(
                cache,
                method=method,
                seed=7001 + offset,
                device=device,
                device_name=args.device,
                b_lite_checkpoint=b_lite_checkpoint,
                b_lite_sha256=b_lite_sha256,
                output=unit,
                steps=1,
                expected_eligible=4,
            )
            checkpoint = unit / row["checkpoint_path"]
            model = core.load_trained_control(
                method,
                checkpoint,
                device=device,
                expected_sha256=row["checkpoint_sha256"],
            )
            native, conserved = core.infer_samples(
                cache, method=method, model=model, device=device, batch_size=2
            )
            visible = cache.tensors["visibility"].bool().expand_as(conserved)
            require(torch.equal(conserved[visible], cache.tensors["partial_uv"][visible]), f"visible_exact:{method}")
            require(tuple(native.shape) == (4, 3, 64, 64), f"shape:{method}")
            completed.append(
                {
                    "method": method,
                    "optimizer_steps": row["optimizer_steps"],
                    "checkpoint_sha256_length": len(row["checkpoint_sha256"]),
                }
            )
    result = {
        "schema_version": "frugalface3d.w5b49n.v14.cuda_smoke.v1",
        "status": "PASS_V14_MATCHED_CONTROL_SYNTHETIC",
        "device_backend": args.device,
        "device_name": runtime.get("cuda_device_name", "CPU_DEVELOPMENT_ONLY"),
        "runtime_fingerprint": runtime,
        "contract_sha256": core.sha256_file(PACKAGE_ROOT / "contract.json"),
        "source_lock_sha256": core.sha256_file(PACKAGE_ROOT / "source_lock.json"),
        "tested_methods": completed,
        "optimizer_steps": len(completed),
        "condition0_joint_zeroing_verified": True,
        "observed_uv_exactness_verified": True,
        "scientific_result_generated": False,
    }
    if args.receipt is not None:
        receipt = args.receipt.expanduser().resolve()
        if receipt.exists():
            raise FileExistsError("synthetic_receipt_exists_no_overwrite")
        core.write_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
