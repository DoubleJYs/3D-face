#!/usr/bin/env python3
"""One-step synthetic CPU/CUDA smoke for CanonReg; no scientific data are read."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from canonreg_loss import compute_training_loss
from integrity import (
    PACKAGE_ROOT,
    discover_repository_root,
    load_historical_core,
    package_binding,
    read_contract,
    sha256_file,
    write_json,
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"canonreg_smoke:{label}")


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
    canonical = torch.zeros(count, 1, 64, 64)
    canonical[:, :, 3:61, 4:60] = 1.0
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


def run_smoke(
    repository_root: Path | None,
    *,
    device_name: str,
    receipt: Path | None = None,
) -> dict[str, object]:
    import torch

    root = discover_repository_root(repository_root)
    contract = read_contract()
    core = load_historical_core(root, contract)
    if device_name == "cpu":
        torch.set_num_threads(1)
    device = core.configure_runtime(device_name, formal=device_name == "cuda")
    runtime = core.runtime_fingerprint(device_name)
    if device_name == "cuda":
        require(runtime.get("cuda_device_name") == "NVIDIA GeForce RTX 4090", "cuda_device_name")
    cache = make_cache()
    core.seed_all(7001, device_name)
    model = core.new_structure_model(device=device, trainable=True)
    require(core.parameter_count(model) == 89_386, "parameter_count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    positions = list(range(len(cache.rows)))
    pair_map, donor_map = core.pair_and_donor_maps(cache.rows, positions)
    anchors = positions
    pairs = [pair_map[index] for index in anchors]
    donors = [donor_map[index] for index in anchors]
    batch = core.model_batch(cache, [*anchors, *pairs, *donors], device)
    optimizer.zero_grad(set_to_none=True)
    loss, terms, _output, diagnostics = compute_training_loss(
        core, model, batch, anchor_count=len(anchors)
    )
    require(bool(torch.isfinite(loss)), "loss_finite")
    require(float(diagnostics["residual_support_texels"].min()) > 0, "residual_support")
    require(float(diagnostics["canonical_edges_x"].min()) > 0, "tv_x_support")
    require(float(diagnostics["canonical_edges_y"].min()) > 0, "tv_y_support")
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    require(bool(torch.isfinite(gradient)), "gradient_finite")
    optimizer.step()
    core.synchronize(device_name)
    with tempfile.TemporaryDirectory(prefix="canonreg-smoke-") as temporary:
        checkpoint = Path(temporary) / "step_0001.pt"
        core.atomic_torch_save(
            checkpoint,
            {
                "schema_version": "frugalface3d.review_closure.canonreg.checkpoint.v1",
                "method": core.METHOD_FULL,
                "variant": "canonreg",
                "seed": 7001,
                "step": 1,
                "optimizer_steps": 1,
                "selection_or_best_of_n": False,
                "automatic_retry": False,
                "trainable_parameters": 89_386,
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            },
        )
        loaded = core.load_trained_control(
            core.METHOD_FULL,
            checkpoint,
            device=device,
            expected_sha256=sha256_file(checkpoint),
        )
        native, conserved = core.infer_samples(
            cache,
            method=core.METHOD_FULL,
            model=loaded,
            device=device,
            batch_size=2,
        )
    visible = cache.tensors["visibility"].bool().expand_as(conserved)
    require(torch.equal(conserved[visible], cache.tensors["partial_uv"][visible]), "observed_uv_exact")
    require(tuple(native.shape) == (4, 3, 64, 64), "inference_shape")
    binding = package_binding()
    result: dict[str, object] = {
        "schema_version": "frugalface3d.review_closure.canonreg.cuda_smoke.v1",
        "status": "PASS_CANONREG_SYNTHETIC_SMOKE",
        "device_backend": device_name,
        "device_name": runtime.get("cuda_device_name", "CPU_DEVELOPMENT_ONLY"),
        "runtime_fingerprint": runtime,
        "optimizer_steps": 1,
        "trainable_parameters": 89_386,
        "loss_finite": True,
        "gradient_finite": True,
        "regularizer_contract_verified": True,
        "observed_uv_exactness_verified": True,
        "contract_sha256": binding["contract_sha256"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "historical_core_sha256": contract["upstream_source_lock"][0]["sha256"],
        "scientific_result_generated": False,
    }
    if receipt is not None:
        output = receipt.expanduser().resolve()
        if output.exists():
            raise FileExistsError("canonreg_smoke_receipt_exists_no_overwrite")
        write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = run_smoke(
        args.repository_root,
        device_name=args.device,
        receipt=args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
