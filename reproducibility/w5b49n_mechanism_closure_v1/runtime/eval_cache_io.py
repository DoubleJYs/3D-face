"""Private FaceScape evaluation-cache contract.

The cache is repository-external derived biometric material.  Its public
manifest uses anonymous tokens only; the tensor payload is never eligible for
redistribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from reproducibility.w5b49n_mechanism_closure_v1.training.cache_io import sha256_file


SCHEMA_VERSION = "frugalface3d.w5b49n.mechanism_facescape_eval_cache.v1"
STATUS = "PASS_PRIVATE_FACESCAPE_160_EVAL_CACHE"
TENSOR_FILE = "EVAL_TENSORS.pt"
MANIFEST_FILE = "EVAL_CACHE_MANIFEST.json"

MODEL_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "partial_uv": (3, 64, 64),
    "visibility": (1, 64, 64),
    "geometry_map": (6, 64, 64),
    "canonical_mask": (1, 64, 64),
    "base_completion": (3, 64, 64),
    "texture_feature": (160, 16, 16),
    "expression_token": (128,),
}
GEOMETRY_SHAPES: dict[str, tuple[int, ...]] = {
    "camera_state": (3,),
    "vertices": (5023, 3),
    "projected_vertices": (5023, 3),
}
RENDER_CONTEXT_SHAPES: dict[str, tuple[int, ...]] = {
    "screen_rgb_uint8": (3, 224, 224),
    "screen_uv_float32": (224, 224, 2),
    "screen_visibility_bool": (1, 224, 224),
}
TENSOR_SHAPES = {
    **MODEL_INPUT_SHAPES,
    **GEOMETRY_SHAPES,
    **RENDER_CONTEXT_SHAPES,
}


@dataclass(frozen=True)
class EvalCache:
    rows: tuple[Mapping[str, Any], ...]
    tensors: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "sample_index",
        "dataset_token",
        "identity_token",
        "expression_token",
        "view_token",
    }
    if len(rows) != 160 or any(set(row) != required for row in rows):
        raise ValueError("facescape_eval_cache_row_contract")
    if [int(row["sample_index"]) for row in rows] != list(range(160)):
        raise ValueError("facescape_eval_cache_sample_order")
    if {str(row["dataset_token"]) for row in rows} != {"D1"}:
        raise ValueError("facescape_eval_cache_dataset_token")
    if len({str(row["identity_token"]) for row in rows}) != 20:
        raise ValueError("facescape_eval_cache_identity_count")
    if len({str(row["expression_token"]) for row in rows}) != 4:
        raise ValueError("facescape_eval_cache_expression_count")
    if len({str(row["view_token"]) for row in rows}) != 2:
        raise ValueError("facescape_eval_cache_view_count")


def load_eval_cache(root: Path) -> EvalCache:
    """Load and validate the private cache once before raw inference."""

    import torch

    root = root.resolve(strict=True)
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != STATUS:
        raise ValueError("facescape_eval_cache_schema_or_status")
    if manifest.get("contains_original_identity_ids") is not False:
        raise ValueError("facescape_eval_cache_identity_boundary")
    if manifest.get("contains_source_paths") is not False:
        raise ValueError("facescape_eval_cache_path_boundary")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("facescape_eval_cache_rows_list")
    _validate_rows(rows)
    tensor_path = root / TENSOR_FILE
    if sha256_file(tensor_path) != manifest.get("tensor_sha256"):
        raise ValueError("facescape_eval_cache_tensor_sha256")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if set(tensors) != set(TENSOR_SHAPES):
        raise ValueError("facescape_eval_cache_tensor_keyspace")
    for name, shape in TENSOR_SHAPES.items():
        value = tensors[name]
        if tuple(value.shape) != (160, *shape):
            raise ValueError(f"facescape_eval_cache_tensor_shape:{name}:{tuple(value.shape)}")
        if name != "screen_uv_float32" and not bool(torch.isfinite(value).all()):
            raise ValueError(f"facescape_eval_cache_tensor_nonfinite:{name}")
    finite_screen_uv = torch.isfinite(tensors["screen_uv_float32"]).all(dim=3)
    if not torch.equal(finite_screen_uv, tensors["screen_visibility_bool"][:, 0]):
        raise ValueError("facescape_eval_cache_screen_uv_visibility_binding")
    return EvalCache(rows=tuple(rows), tensors=tensors, manifest=manifest)


__all__ = [
    "EvalCache",
    "GEOMETRY_SHAPES",
    "MANIFEST_FILE",
    "MODEL_INPUT_SHAPES",
    "RENDER_CONTEXT_SHAPES",
    "SCHEMA_VERSION",
    "STATUS",
    "TENSOR_FILE",
    "TENSOR_SHAPES",
    "load_eval_cache",
]
