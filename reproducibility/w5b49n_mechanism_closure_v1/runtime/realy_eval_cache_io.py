"""Anonymous private REALY-100 source-cache contract.

The cache stores one frozen SMIRK-to-UV source derivation for each of the
100 identities x 4 views.  It deliberately contains no target-view tensor:
the existing anonymous 1,200-pair roster remains the only source/target
relationship record and is consumed later by the evaluator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "frugalface3d.w5b49n.mechanism_realy_source_cache.v3"
STATUS = "PASS_PRIVATE_REALY100_400_SOURCE_UV_AND_RENDER_CONTEXT_CACHE"
TENSOR_FILE = "REALY_SOURCE_TENSORS.pt"
MANIFEST_FILE = "REALY_SOURCE_CACHE_MANIFEST.json"
SAMPLE_COUNT = 400
IDENTITY_COUNT = 100
VIEW_COUNT = 4
PAIR_COUNT = 1200

MODEL_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "partial_uv": (3, 64, 64),
    "visibility": (1, 64, 64),
    "geometry_map": (6, 64, 64),
    "canonical_mask": (1, 64, 64),
    "base_completion": (3, 64, 64),
    "texture_feature": (160, 16, 16),
    "expression_token": (128,),
}

# These arrays are captured while each of the 400 images is processed as a
# source.  They contain no pair relationship and no target-selected value.
# Pairing remains forbidden until the post-F2 evaluator opens the anonymous
# directed-pair roster.  Keeping the exact render state here avoids decoding a
# target image or rerunning geometry after candidate generation.
RENDER_CONTEXT_SHAPES: dict[str, tuple[int, ...]] = {
    "render_vertices_float32": (5023, 3),
    "render_camera_state_float32": (3,),
    "render_projected_vertices_float32": (5023, 3),
    "screen_rgb_uint8": (3, 224, 224),
    "screen_uv_float32": (224, 224, 2),
    "screen_visibility_bool": (1, 224, 224),
}
RENDER_CONTEXT_DTYPES: dict[str, str] = {
    "render_vertices_float32": "float32",
    "render_camera_state_float32": "float32",
    "render_projected_vertices_float32": "float32",
    "screen_rgb_uint8": "uint8",
    "screen_uv_float32": "float32",
    "screen_visibility_bool": "bool",
}
TENSOR_SHAPES = {**MODEL_INPUT_SHAPES, **RENDER_CONTEXT_SHAPES}

_IDENTITY_TOKEN = re.compile(r"D2-[0-9]{3}\Z")
_VIEW_TOKEN = re.compile(r"V[0-9]{2}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_BASELINE_METHOD_IDS = (
    "freeuv_raw",
    "lama_big",
    "zits",
    "mat_ffhq_512",
)

SOURCE_ROSTER_PROJECTION_FIELDS = (
    "schema_version",
    "status",
    "dataset_token",
    "rights_projection_sha256",
    "source_root_opaque_sha256",
    "ordered_source_aggregate_sha256",
    "identity_count",
    "view_count",
    "asset_count",
    "total_bytes",
    "assets",
    "privacy",
)


class RealyCacheError(ValueError):
    """A private REALY cache or public-baseline interface is invalid."""


@dataclass(frozen=True)
class RealySourceCache:
    rows: tuple[Mapping[str, Any], ...]
    tensors: Mapping[str, Any]
    manifest: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anonymous_source_roster_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash only the 400-source roster projection, never pair metadata."""

    _require(
        all(field in manifest for field in SOURCE_ROSTER_PROJECTION_FIELDS),
        "realy_source_roster_projection_fields",
    )
    projection = {field: manifest[field] for field in SOURCE_ROSTER_PROJECTION_FIELDS}
    payload = (
        json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_array_sha256(value: Any) -> str:
    """Hash dtype, shape, and C-order bytes without exposing array values."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.dtype("<u8")).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RealyCacheError(code)


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "sample_index",
        "dataset_token",
        "asset_id",
        "identity_token",
        "view_token",
        "support",
        "render_context",
    }
    _require(len(rows) == SAMPLE_COUNT, "realy_source_cache_sample_count")
    _require(
        [int(row.get("sample_index", -1)) for row in rows] == list(range(SAMPLE_COUNT)),
        "realy_source_cache_sample_order",
    )
    _require(all(set(row) == required for row in rows), "realy_source_cache_row_keyspace")
    _require({str(row["dataset_token"]) for row in rows} == {"D2"}, "realy_dataset_token")
    identities = {str(row["identity_token"]) for row in rows}
    views = {str(row["view_token"]) for row in rows}
    _require(len(identities) == IDENTITY_COUNT, "realy_source_cache_identity_count")
    _require(len(views) == VIEW_COUNT, "realy_source_cache_view_count")
    _require(all(_IDENTITY_TOKEN.fullmatch(value) for value in identities), "realy_identity_token")
    _require(all(_VIEW_TOKEN.fullmatch(value) for value in views), "realy_view_token")
    expected_assets = {
        f"{identity}-{view}" for identity in identities for view in views
    }
    _require(
        {str(row["asset_id"]) for row in rows} == expected_assets,
        "realy_source_cache_asset_cartesian_product",
    )
    for row in rows:
        support = row["support"]
        _require(isinstance(support, Mapping), "realy_source_cache_support_object")
        _require(
            set(support)
            == {"visible_triangles", "geometry_support", "observed_UV", "rendered_pixels"},
            "realy_source_cache_support_keyspace",
        )
        _require(
            all(isinstance(value, int) and value > 0 for value in support.values()),
            "realy_source_cache_support_nonpositive",
        )
        context = row["render_context"]
        _require(isinstance(context, Mapping), "realy_render_context_object")
        _require(
            set(context)
            == {
                "vertices_sha256",
                "camera_sha256",
                "projected_vertices_sha256",
                "screen_rgb_sha256",
                "screen_uv_sha256",
                "screen_visibility_sha256",
            },
            "realy_render_context_keyspace",
        )
        _require(
            all(isinstance(value, str) and _HEX64.fullmatch(value) for value in context.values()),
            "realy_render_context_hash_invalid",
        )


def load_realy_source_cache(root: Path) -> RealySourceCache:
    """Load the complete 400-source cache after its formal terminal exists."""

    import torch

    root = root.expanduser().resolve(strict=True)
    manifest_path = root / MANIFEST_FILE
    tensor_path = root / TENSOR_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "realy_cache_schema")
    _require(manifest.get("status") == STATUS, "realy_cache_status")
    _require(manifest.get("contains_original_identity_ids") is False, "realy_identity_boundary")
    _require(manifest.get("contains_source_paths") is False, "realy_path_boundary")
    _require(manifest.get("target_view_image_reads") == 0, "realy_target_read_boundary")
    _require(manifest.get("paper_metrics_generated") == 0, "realy_metric_boundary")
    _require(
        manifest.get("render_context_materialized_source_only") is True,
        "realy_render_context_source_only_boundary",
    )
    _require(manifest.get("source_sample_count") == SAMPLE_COUNT, "realy_manifest_sample_count")
    source_roster_sha = manifest.get("anonymous_source_roster_projection_sha256")
    _require(
        isinstance(source_roster_sha, str)
        and _HEX64.fullmatch(source_roster_sha) is not None,
        "realy_source_roster_projection_sha",
    )
    _require(
        manifest.get("directed_pair_file_reads") == 0
        and manifest.get("directed_pair_rows_read") == 0
        and manifest.get("pair_relationship_bound_pre_f2") is False
        and manifest.get("directed_pair_binding_phase") == "POST_F2_METRICS_ONLY",
        "realy_pre_f2_pair_isolation",
    )
    _require(
        "directed_pair_count" not in manifest
        and "directed_pairs_sha256" not in manifest,
        "realy_source_cache_pair_binding_forbidden",
    )
    rows = manifest.get("rows")
    _require(isinstance(rows, list), "realy_source_cache_rows_list")
    validate_rows(rows)
    _require(tensor_path.is_file() and not tensor_path.is_symlink(), "realy_tensor_plain_file")
    _require(sha256_file(tensor_path) == manifest.get("tensor_sha256"), "realy_tensor_sha")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    _require(set(tensors) == set(TENSOR_SHAPES), "realy_tensor_keyspace")
    for name, shape in MODEL_INPUT_SHAPES.items():
        value = tensors[name]
        _require(tuple(value.shape) == (SAMPLE_COUNT, *shape), f"realy_tensor_shape:{name}")
        _require(bool(torch.isfinite(value).all()), f"realy_tensor_nonfinite:{name}")
        _require(value.dtype == torch.float32, f"realy_tensor_dtype:{name}")
    for name, shape in RENDER_CONTEXT_SHAPES.items():
        value = tensors[name]
        _require(tuple(value.shape) == (SAMPLE_COUNT, *shape), f"realy_tensor_shape:{name}")
        expected_dtype = {
            "float32": torch.float32,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }[RENDER_CONTEXT_DTYPES[name]]
        _require(value.dtype == expected_dtype, f"realy_tensor_dtype:{name}")
        if name != "screen_uv_float32":
            _require(bool(torch.isfinite(value).all()), f"realy_tensor_nonfinite:{name}")
    _require(
        bool(((tensors["visibility"] == 0) | (tensors["visibility"] == 1)).all()),
        "realy_visibility_not_binary",
    )
    _require(
        bool(((tensors["canonical_mask"] == 0) | (tensors["canonical_mask"] == 1)).all()),
        "realy_canonical_not_binary",
    )
    for name in ("partial_uv", "base_completion"):
        _require(
            float(tensors[name].min()) >= 0.0 and float(tensors[name].max()) <= 1.0,
            f"realy_rgb_range:{name}",
        )
    screen_uv = tensors["screen_uv_float32"]
    screen_visibility = tensors["screen_visibility_bool"][:, 0]
    finite_uv = torch.isfinite(screen_uv).all(dim=3)
    _require(
        bool(torch.equal(finite_uv, screen_visibility)),
        "realy_screen_uv_visibility_equivalence",
    )
    _require(bool(screen_visibility.any()), "realy_screen_visibility_empty")
    visible_uv = screen_uv[screen_visibility]
    _require(
        float(visible_uv.min()) >= 0.0 and float(visible_uv.max()) <= 1.0,
        "realy_screen_uv_visible_range",
    )
    for index, row in enumerate(rows):
        context = row["render_context"]
        observed = {
            "vertices_sha256": canonical_array_sha256(
                tensors["render_vertices_float32"][index]
            ),
            "camera_sha256": canonical_array_sha256(
                tensors["render_camera_state_float32"][index]
            ),
            "projected_vertices_sha256": canonical_array_sha256(
                tensors["render_projected_vertices_float32"][index]
            ),
            "screen_rgb_sha256": canonical_array_sha256(
                tensors["screen_rgb_uint8"][index]
            ),
            "screen_uv_sha256": canonical_array_sha256(
                tensors["screen_uv_float32"][index]
            ),
            "screen_visibility_sha256": canonical_array_sha256(
                tensors["screen_visibility_bool"][index]
            ),
        }
        _require(observed == context, f"realy_render_context_row_hash:{index}")
    return RealySourceCache(rows=tuple(rows), tensors=tensors, manifest=manifest)


def validate_public_baseline_native(method_id: str, native: Any) -> Any:
    """Validate, but never synthesize, an external baseline's native output.

    Public baseline adapters may call this boundary after their official model
    has produced a 400-source tensor.  The function intentionally has no
    fallback or filling behavior.
    """

    import torch

    _require(method_id in PUBLIC_BASELINE_METHOD_IDS, "realy_public_method_unknown")
    _require(isinstance(native, torch.Tensor), "realy_public_native_tensor_required")
    _require(
        tuple(native.shape) == (SAMPLE_COUNT, 3, 64, 64),
        "realy_public_native_shape",
    )
    _require(native.dtype == torch.float32, "realy_public_native_float32")
    _require(bool(torch.isfinite(native).all()), "realy_public_native_nonfinite")
    _require(
        float(native.min()) >= 0.0 and float(native.max()) <= 1.0,
        "realy_public_native_range",
    )
    return native.contiguous()


__all__ = [
    "IDENTITY_COUNT",
    "MANIFEST_FILE",
    "MODEL_INPUT_SHAPES",
    "RENDER_CONTEXT_DTYPES",
    "RENDER_CONTEXT_SHAPES",
    "PAIR_COUNT",
    "PUBLIC_BASELINE_METHOD_IDS",
    "RealyCacheError",
    "RealySourceCache",
    "SAMPLE_COUNT",
    "SCHEMA_VERSION",
    "STATUS",
    "TENSOR_FILE",
    "TENSOR_SHAPES",
    "VIEW_COUNT",
    "anonymous_source_roster_sha256",
    "load_realy_source_cache",
    "canonical_array_sha256",
    "sha256_file",
    "validate_public_baseline_native",
    "validate_rows",
]
