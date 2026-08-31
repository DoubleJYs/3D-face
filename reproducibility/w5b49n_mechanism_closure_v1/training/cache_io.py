"""Private tensor-cache contract for W49N mechanism training.

The cache contains no source paths or original identity numbers.  Tensor bytes
remain private derived biometric data and must stay outside the repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CACHE_SCHEMA = "frugalface3d.w49n.mechanism_training_cache.v1"
CACHE_STATUS = "PASS_PRIVATE_MECHANISM_TRAINING_CACHE_288"
TENSOR_FILE = "TRAINING_TENSORS.pt"
MANIFEST_FILE = "CACHE_MANIFEST.json"

TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "partial_uv": (3, 64, 64),
    "visibility": (1, 64, 64),
    "geometry_map": (6, 64, 64),
    "canonical_mask": (1, 64, 64),
    "base_completion": (3, 64, 64),
    "texture_feature": (160, 16, 16),
    "expression_token": (128,),
}


@dataclass(frozen=True)
class CachedSamples:
    rows: tuple[Mapping[str, Any], ...]
    tensors: Mapping[str, Any]
    manifest: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, exact: bool) -> None:
    if exact and len(rows) != 288:
        raise ValueError(f"mechanism_cache_row_count:{len(rows)}")
    required = {"sample_index", "partition", "identity_token", "expression_token", "view_token"}
    if any(set(row) != required for row in rows):
        raise ValueError("mechanism_cache_row_keyspace")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("mechanism_cache_sample_order")
    partitions = [str(row["partition"]) for row in rows]
    if exact and (partitions.count("fit_train") != 240 or partitions.count("fit_validation") != 48):
        raise ValueError("mechanism_cache_partition_counts")
    if any(value not in {"fit_train", "fit_validation"} for value in partitions):
        raise ValueError("mechanism_cache_partition_value")


def load_cache(root: Path, *, exact: bool = True) -> CachedSamples:
    """Load and fully validate the cache once before the training campaign."""

    import torch

    root = root.resolve(strict=True)
    manifest_path = root / MANIFEST_FILE
    tensor_path = root / TENSOR_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != CACHE_STATUS:
        raise ValueError("mechanism_cache_schema_or_status")
    if manifest.get("contains_original_identity_ids") is not False:
        raise ValueError("mechanism_cache_identity_boundary")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("mechanism_cache_rows_list")
    _validate_rows(rows, exact=exact)
    expected_sha = str(manifest.get("tensor_sha256", ""))
    if len(expected_sha) != 64 or sha256_file(tensor_path) != expected_sha:
        raise ValueError("mechanism_cache_tensor_sha256")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if set(tensors) != set(TENSOR_SHAPES):
        raise ValueError("mechanism_cache_tensor_keyspace")
    for name, shape in TENSOR_SHAPES.items():
        value = tensors[name]
        if tuple(value.shape) != (len(rows), *shape):
            raise ValueError(f"mechanism_cache_tensor_shape:{name}:{tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"mechanism_cache_tensor_nonfinite:{name}")
    return CachedSamples(rows=tuple(rows), tensors=tensors, manifest=manifest)


def pair_and_donor_maps(rows: Sequence[Mapping[str, Any]], positions: Sequence[int]) -> tuple[dict[int, int], dict[int, int]]:
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
        raise ValueError("mechanism_cache_two_views_required")
    pairs: dict[int, int] = {}
    donors: dict[int, int] = {}
    for (identity, expression, view), position in lookup.items():
        paired_view = views[1] if view == views[0] else views[0]
        pairs[position] = lookup[(identity, expression, paired_view)]
        donor_identity = identities[(identities.index(identity) + 1) % len(identities)]
        donors[position] = lookup[(donor_identity, expression, view)]
    return pairs, donors


__all__ = [
    "CACHE_SCHEMA",
    "CACHE_STATUS",
    "CachedSamples",
    "MANIFEST_FILE",
    "TENSOR_FILE",
    "TENSOR_SHAPES",
    "canonical_json_bytes",
    "load_cache",
    "pair_and_donor_maps",
    "sha256_file",
    "write_json",
]
