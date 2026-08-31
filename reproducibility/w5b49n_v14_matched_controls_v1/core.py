"""Core models and deterministic training/inference for V14 matched controls.

Inputs are validated anonymous tensor caches created by the existing W49N
mechanism-closure package. This module never reads raw images or manuscript
files and never forms source-target pairs during inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_FULL = "Full"
METHOD_CONDITION0 = "Condition0"
METHOD_BLITE_FT = "B-lite-FT"
METHODS = (METHOD_FULL, METHOD_CONDITION0, METHOD_BLITE_FT)

SEED_PLANS: dict[str, tuple[int, ...]] = {
    "cuda_five": (2026080447, 2026080448, 2026080449, 2026080450, 2026080451),
}

STEPS = 512
BATCH_SIZE = 24
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
IDENTITY_TRIPLET_MARGIN = 0.1
LOSS_WEIGHTS = {
    "paired_hidden_UV_L1": 8.0,
    "paired_hidden_uncertainty_NLL": 0.25,
    "same_identity_view_consistency": 0.5,
    "identity_triplet": 0.25,
    "structure_gate_noncollapse": 0.05,
    "bounded_residual_L2": 0.01,
    "total_variation": 0.01,
}


@dataclass
class MatchedOutputs:
    completed_uv: Any
    log_variance: Any
    identity_embedding: Any
    structure_gate: Any | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility for the local synthetic test.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"matched_control_checkpoint_mapping_required:{path.name}")
    return payload


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def exact_b_lite_config() -> Any:
    from frugalface3d.models.uv_completion_lite import UVCompletionLiteConfig

    return UVCompletionLiteConfig(
        mode="no_geometry",
        input_size=64,
        base_channels=40,
        geometry_channels=6,
        normalization="group",
        uncertainty_parameterization="softplus_variance",
        max_parameters=250_000,
    )


def load_b_lite(
    checkpoint: Path,
    *,
    device: Any,
    trainable: bool,
    expected_sha256: str | None = None,
) -> Any:
    from frugalface3d.models.uv_completion_lite import (
        UVCompletionLite,
        count_uv_completion_parameters,
    )

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if expected_sha256 is not None and sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError("matched_control_b_lite_checkpoint_sha256")
    payload = _torch_load(checkpoint)
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("matched_control_b_lite_state_dict_missing")
    model = UVCompletionLite(exact_b_lite_config())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("matched_control_b_lite_state_dict_incompatible")
    model.to(device)
    model.train(trainable)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    if count_uv_completion_parameters(model, trainable_only=False) != 122_164:
        raise RuntimeError("matched_control_b_lite_parameter_count")
    return model


def new_structure_model(*, device: Any, trainable: bool = True) -> Any:
    from frugalface3d.models.structure_aware_b_qualification import (
        StructureAwareBQualification,
        StructureAwareBQualificationConfig,
    )

    model = StructureAwareBQualification(StructureAwareBQualificationConfig()).to(device)
    model.train(trainable)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    if int(sum(value.numel() for value in model.parameters())) != 89_386:
        raise RuntimeError("matched_control_structure_parameter_count")
    return model


def parameter_count(model: Any, *, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))


def seed_all(seed: int, device_name: str) -> None:
    import torch

    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device_name == "mps":
        torch.mps.manual_seed(seed)


def configure_runtime(device_name: str, *, formal: bool) -> Any:
    import torch

    if formal and device_name != "cuda":
        raise RuntimeError("matched_control_formal_device_must_be_cuda")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("matched_control_cuda_unavailable")
        if formal and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("matched_control_cublas_workspace_not_frozen")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("matched_control_mps_unavailable")
    elif device_name != "cpu" or formal:
        raise RuntimeError("matched_control_development_device")
    torch.use_deterministic_algorithms(True)
    return torch.device(device_name)


def runtime_fingerprint(device_name: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "device_type": device_name,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device_name == "cuda":
        index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": str(properties.name),
                "cuda_compute_capability": [int(properties.major), int(properties.minor)],
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return result


def synchronize(device_name: str) -> None:
    import torch

    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def deterministic_epoch_batch_indices(
    *, sample_count: int, zero_based_step: int, seed: int, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    if sample_count <= 0 or zero_based_step < 0 or batch_size <= 0:
        raise ValueError("matched_control_batch_arguments")
    batches_per_epoch = math.ceil(sample_count / batch_size)
    epoch = zero_based_step // batches_per_epoch
    batch = zero_based_step % batches_per_epoch
    permutation = np.random.default_rng(seed + epoch).permutation(sample_count)
    start = batch * batch_size
    return permutation[start : min(start + batch_size, sample_count)].astype(
        np.int64, copy=False
    )


def model_batch(cache: Any, positions: Sequence[int], device: Any) -> dict[str, Any]:
    import torch

    indices = torch.tensor(positions, dtype=torch.long)
    names = (
        "partial_uv",
        "visibility",
        "geometry_map",
        "canonical_mask",
        "base_completion",
        "texture_feature",
        "expression_token",
    )
    return {
        name: cache.tensors[name].index_select(0, indices).to(device)
        for name in names
    }


def forward_matched(model: Any, method: str, batch: Mapping[str, Any]) -> MatchedOutputs:
    import torch
    import torch.nn.functional as F

    if method in {METHOD_FULL, METHOD_CONDITION0}:
        geometry = batch["geometry_map"]
        expression = batch["expression_token"]
        if method == METHOD_CONDITION0:
            # Only explicit coordinate-normal and expression content is removed.
            # Visibility, canonical support, B-lite completion, and texture
            # bottleneck remain unchanged and spatially aligned.
            geometry = torch.zeros_like(geometry)
            expression = torch.zeros_like(expression)
        output = model(
            batch["partial_uv"],
            batch["visibility"],
            geometry,
            batch["canonical_mask"],
            batch["base_completion"],
            batch["texture_feature"],
            expression,
        )
        return MatchedOutputs(
            completed_uv=output.completed_uv,
            log_variance=output.log_variance,
            identity_embedding=output.identity_embedding,
            structure_gate=output.structure_gate,
        )
    if method != METHOD_BLITE_FT:
        raise ValueError(f"matched_control_unknown_method:{method}")
    features: list[Any] = []
    handle = model.bottleneck.register_forward_hook(
        lambda _module, _inputs, value: features.append(value)
    )
    try:
        output = model(batch["partial_uv"], batch["visibility"], None)
    finally:
        handle.remove()
    if len(features) != 1:
        raise RuntimeError("matched_control_b_lite_bottleneck_hook_count")
    embedding = F.normalize(
        F.adaptive_avg_pool2d(features[0], 1).flatten(1), dim=1
    )
    return MatchedOutputs(
        completed_uv=output.completed_uv,
        log_variance=output.log_variance,
        identity_embedding=embedding,
        structure_gate=None,
    )


def hidden_mask(anchor: Mapping[str, Any], paired: Mapping[str, Any]) -> Any:
    return anchor["canonical_mask"] * (1.0 - anchor["visibility"]) * paired["visibility"]


def masked_l1(prediction: Any, target: Any, mask: Any) -> Any:
    denominator = (mask.sum(dim=(1, 2, 3)) * prediction.shape[1]).clamp_min(1.0)
    return ((prediction - target).abs() * mask).sum(dim=(1, 2, 3)) / denominator


def compute_training_loss(
    model: Any,
    method: str,
    batch: Mapping[str, Any],
    *,
    anchor_count: int,
) -> tuple[Any, dict[str, Any], MatchedOutputs]:
    import torch
    import torch.nn.functional as F

    output = forward_matched(model, method, batch)
    anchor = {name: value[:anchor_count] for name, value in batch.items()}
    paired = {
        name: value[anchor_count : 2 * anchor_count]
        for name, value in batch.items()
    }
    mask = hidden_mask(anchor, paired)
    if bool((mask.sum(dim=(1, 2, 3)) < 1.0).any()):
        raise RuntimeError("matched_control_hidden_support_changed")
    factual = output.completed_uv[:anchor_count]
    target = paired["partial_uv"]
    hidden_l1 = masked_l1(factual, target, mask).mean()
    rgb_error = ((factual - target).abs() * mask).sum(dim=1, keepdim=True) / 3.0
    nll = (
        (rgb_error * torch.exp(-output.log_variance[:anchor_count])
         + output.log_variance[:anchor_count])
        * mask
    ).sum() / mask.sum().clamp_min(1.0)
    canonical = anchor["canonical_mask"]
    view_consistency = (
        (output.completed_uv[:anchor_count]
         - output.completed_uv[anchor_count : 2 * anchor_count]).abs()
        * canonical
    ).sum() / (canonical.sum().clamp_min(1.0) * 3.0)
    positive = F.cosine_similarity(
        output.identity_embedding[:anchor_count],
        output.identity_embedding[anchor_count : 2 * anchor_count],
    )
    negative = F.cosine_similarity(
        output.identity_embedding[:anchor_count],
        output.identity_embedding[2 * anchor_count : 3 * anchor_count],
    )
    triplet = F.relu(IDENTITY_TRIPLET_MARGIN - positive + negative).mean()
    if output.structure_gate is None:
        gate_noncollapse = factual.sum() * 0.0
    else:
        structure_mean = output.structure_gate[:anchor_count, 1].mean()
        gate_noncollapse = F.relu(0.10 - structure_mean) + F.relu(structure_mean - 0.90)
    residual = (factual - anchor["base_completion"]) * (1.0 - anchor["visibility"])
    residual_l2 = residual.square().mean()
    tv = (residual[:, :, 1:] - residual[:, :, :-1]).abs().mean()
    tv = tv + (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
    terms = {
        "paired_hidden_UV_L1": hidden_l1,
        "paired_hidden_uncertainty_NLL": nll,
        "same_identity_view_consistency": view_consistency,
        "identity_triplet": triplet,
        "structure_gate_noncollapse": gate_noncollapse,
        "bounded_residual_L2": residual_l2,
        "total_variation": tv,
    }
    loss = sum(LOSS_WEIGHTS[name] * value for name, value in terms.items())
    return loss, terms, output


def pair_and_donor_maps(
    rows: Sequence[Mapping[str, Any]], positions: Sequence[int]
) -> tuple[dict[int, int], dict[int, int]]:
    lookup = {
        (
            str(rows[position]["identity_token"]),
            str(rows[position]["expression_token"]),
            str(rows[position]["view_token"]),
        ): position
        for position in positions
    }
    identities = sorted({key[0] for key in lookup})
    views = sorted({key[2] for key in lookup})
    if len(views) != 2:
        raise ValueError("matched_control_fit_cache_two_views_required")
    pairs: dict[int, int] = {}
    donors: dict[int, int] = {}
    for (identity, expression, view), position in lookup.items():
        paired_view = views[1] if view == views[0] else views[0]
        pairs[position] = lookup[(identity, expression, paired_view)]
        donor_identity = identities[(identities.index(identity) + 1) % len(identities)]
        donors[position] = lookup[(donor_identity, expression, view)]
    return pairs, donors


def make_trainable_model(
    method: str,
    *,
    seed: int,
    device: Any,
    device_name: str,
    b_lite_checkpoint: Path,
    b_lite_sha256: str,
) -> Any:
    seed_all(seed, device_name)
    if method in {METHOD_FULL, METHOD_CONDITION0}:
        return new_structure_model(device=device, trainable=True)
    if method == METHOD_BLITE_FT:
        return load_b_lite(
            b_lite_checkpoint,
            device=device,
            trainable=True,
            expected_sha256=b_lite_sha256,
        )
    raise ValueError(f"matched_control_train_method:{method}")


def train_one(
    cache: Any,
    *,
    method: str,
    seed: int,
    device: Any,
    device_name: str,
    b_lite_checkpoint: Path,
    b_lite_sha256: str,
    output: Path,
    steps: int = STEPS,
    expected_eligible: int | None = 238,
) -> dict[str, Any]:
    import torch

    train_positions = [
        index for index, row in enumerate(cache.rows) if row["partition"] == "fit_train"
    ]
    pair_map, donor_map = pair_and_donor_maps(cache.rows, train_positions)
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
    if expected_eligible is not None and len(eligible) != expected_eligible:
        raise RuntimeError(f"matched_control_eligible_count:{len(eligible)}")
    model = make_trainable_model(
        method,
        seed=seed,
        device=device,
        device_name=device_name,
        b_lite_checkpoint=b_lite_checkpoint,
        b_lite_sha256=b_lite_sha256,
    )
    expected_parameters = 122_164 if method == METHOD_BLITE_FT else 89_386
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("matched_control_trainable_parameter_count")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for zero_step in range(steps):
        local = deterministic_epoch_batch_indices(
            sample_count=len(eligible), zero_based_step=zero_step, seed=seed
        )
        anchors = [eligible[int(index)] for index in local]
        pairs = [pair_map[index] for index in anchors]
        donors = [donor_map[index] for index in anchors]
        batch = model_batch(cache, [*anchors, *pairs, *donors], device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms, _output = compute_training_loss(
            model, method, batch, anchor_count=len(anchors)
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("matched_control_loss_nonfinite")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not math.isfinite(float(gradient.detach().cpu())):
            raise RuntimeError("matched_control_gradient_nonfinite")
        optimizer.step()
        synchronize(device_name)
        trace.append(
            {
                "step": zero_step + 1,
                "loss": float(loss.detach().cpu()),
                "hidden_uv_l1": float(terms["paired_hidden_UV_L1"].detach().cpu()),
                "gradient_norm_before_clip": float(gradient.detach().cpu()),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / f"step_{steps:04d}.pt"
    checkpoint_payload = {
        "schema_version": "frugalface3d.w5b49n.v14_matched_control_checkpoint.v1",
        "method": method,
        "paper_label": "NoStruct" if method == METHOD_CONDITION0 else method,
        "seed": seed,
        "step": steps,
        "optimizer_steps": steps,
        "selection_or_best_of_n": False,
        "automatic_retry": False,
        "trainable_parameters": expected_parameters,
        "b_lite_initialization_sha256": (
            b_lite_sha256 if method == METHOD_BLITE_FT else None
        ),
        "condition0_zeroed_inputs": (
            ["geometry_map", "expression_token"]
            if method == METHOD_CONDITION0
            else []
        ),
        "model_config": asdict(model.config),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
    }
    atomic_torch_save(checkpoint, checkpoint_payload)
    write_json(output / "TRAIN_TRACE.json", trace)
    return {
        "method": method,
        "seed": seed,
        "unit_status": "TRAINED",
        "optimizer_steps": steps,
        "trainable_parameters": expected_parameters,
        "checkpoint_origin": "training_root",
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_or_best_of_n": False,
        "automatic_retry": False,
    }


def load_trained_control(
    method: str,
    checkpoint: Path,
    *,
    device: Any,
    expected_sha256: str,
) -> Any:
    from frugalface3d.models.uv_completion_lite import UVCompletionLite

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError("matched_control_trained_checkpoint_sha256")
    payload = _torch_load(checkpoint)
    if payload.get("method") != method:
        raise RuntimeError("matched_control_trained_checkpoint_method")
    if method in {METHOD_FULL, METHOD_CONDITION0}:
        model = new_structure_model(device=device, trainable=False)
    elif method == METHOD_BLITE_FT:
        model = UVCompletionLite(exact_b_lite_config()).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(f"matched_control_load_method:{method}")
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("matched_control_trained_checkpoint_incompatible")
    return model


def infer_samples(
    cache: Any,
    *,
    method: str,
    model: Any,
    device: Any,
    batch_size: int = BATCH_SIZE,
) -> tuple[Any, Any]:
    import torch

    positions = list(range(len(cache.rows)))
    chunks = []
    for start in range(0, len(positions), batch_size):
        current = positions[start : start + batch_size]
        batch = model_batch(cache, current, device)
        with torch.inference_mode():
            output = forward_matched(model, method, batch).completed_uv.detach().cpu()
        chunks.append(output)
    native = torch.cat(chunks, dim=0).contiguous()
    if tuple(native.shape) != (len(cache.rows), 3, 64, 64):
        raise RuntimeError("matched_control_inference_shape")
    if not bool(torch.isfinite(native).all()) or float(native.min()) < 0.0 or float(native.max()) > 1.0:
        raise RuntimeError("matched_control_inference_range_or_finite")
    visible = cache.tensors["visibility"].to(dtype=torch.bool)
    observed = cache.tensors["partial_uv"]
    conserved = torch.where(visible, observed, native).contiguous()
    expanded = visible.expand_as(observed)
    if not torch.equal(conserved[expanded], observed[expanded]):
        raise RuntimeError("matched_control_observed_uv_not_exact")
    hidden = ~expanded
    if not torch.equal(conserved[hidden], native[hidden]):
        raise RuntimeError("matched_control_hidden_projection_changed")
    return native, conserved


__all__ = [
    "BATCH_SIZE",
    "LOSS_WEIGHTS",
    "METHOD_BLITE_FT",
    "METHOD_CONDITION0",
    "METHOD_FULL",
    "METHODS",
    "SEED_PLANS",
    "STEPS",
    "atomic_torch_save",
    "canonical_json_bytes",
    "compute_training_loss",
    "configure_runtime",
    "exact_b_lite_config",
    "forward_matched",
    "infer_samples",
    "load_b_lite",
    "load_trained_control",
    "make_trainable_model",
    "model_batch",
    "new_structure_model",
    "parameter_count",
    "runtime_fingerprint",
    "seed_all",
    "sha256_file",
    "train_one",
    "write_json",
]
