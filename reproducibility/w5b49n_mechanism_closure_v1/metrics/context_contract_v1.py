"""Immutable geometry-state and secondary-metric status contracts.

UV predictors in this experiment are not allowed to emit or update geometry.
The helpers below bind the complete ordered cache context to every raw/metric
terminal.  Equality is therefore a reference equality over a frozen input
context, not a claim that a second geometry pipeline was executed.

LPIPS and SFace are deliberately represented by explicit ``METHOD_FAILURE``
records until both the route-specific render material and the exact evaluator
are bound.  A missing evaluator is never replaced by a proxy value.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from reproducibility.w5b49n_mechanism_closure_v1.runtime.realy_eval_cache_io import (
    canonical_array_sha256,
)


SCHEMA_VERSION = "frugalface3d.w5b49n.immutable_context_binding.v1"
SECONDARY_SCHEMA_VERSION = "frugalface3d.w5b49n.secondary_metric_status.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def facescape_context_binding(cache: Any) -> dict[str, Any]:
    """Bind the complete ordered D1 mesh/camera/projection/visibility state."""

    rows = []
    for index in range(len(cache.rows)):
        rows.append(
            {
                "sample_index": index,
                "mesh_vertices_sha256": canonical_array_sha256(
                    cache.tensors["vertices"][index]
                ),
                "camera_state_sha256": canonical_array_sha256(
                    cache.tensors["camera_state"][index]
                ),
                "projected_vertices_sha256": canonical_array_sha256(
                    cache.tensors["projected_vertices"][index]
                ),
                "uv_visibility_sha256": canonical_array_sha256(
                    cache.tensors["visibility"][index]
                ),
                "geometry_map_sha256": canonical_array_sha256(
                    cache.tensors["geometry_map"][index]
                ),
            }
        )
    digest = _canonical_sha256(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "D1",
        "binding_scope": f"ordered_complete_cache_{len(rows)}_samples",
        "route_output_keyspace": ["native_uv", "conserved_uv"],
        "route_outputs_uv_only": True,
        "input_context_sha256": digest,
        "output_context_reference_sha256": digest,
        "input_output_context_reference_exact": True,
        "per_sample_context_hashes_retained_in_binding_digest": True,
        "geometry_updates_or_outputs": 0,
        "role_status": {
            "mesh_vertices": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "camera_state": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "projected_vertices": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "visibility": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
        },
        "mc0_geometry_state_status": "PASS_ALL_FOUR_CONTEXT_ROLES_REFERENCE_EXACT",
    }


def realy_context_binding(cache: Any) -> dict[str, Any]:
    """Bind the complete ordered D2 mesh/camera/projection/visibility state."""

    rows = []
    for row in cache.rows:
        context = row["render_context"]
        rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "mesh_vertices_sha256": context["vertices_sha256"],
                "camera_state_sha256": context["camera_sha256"],
                "projected_vertices_sha256": context[
                    "projected_vertices_sha256"
                ],
                "visibility_sha256": context["screen_visibility_sha256"],
            }
        )
    digest = _canonical_sha256(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "D2",
        "binding_scope": f"ordered_complete_cache_{len(rows)}_samples",
        "route_output_keyspace": ["native_uv", "conserved_uv"],
        "route_outputs_uv_only": True,
        "input_context_sha256": digest,
        "output_context_reference_sha256": digest,
        "input_output_context_reference_exact": True,
        "per_sample_context_hashes_retained_in_binding_digest": True,
        "geometry_updates_or_outputs": 0,
        "role_status": {
            "mesh_vertices": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "camera_state": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "projected_vertices": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
            "visibility": "PASS_INPUT_OUTPUT_REFERENCE_EXACT",
        },
        "mc0_geometry_state_status": "PASS_ALL_FOUR_CONTEXT_ROLES_REFERENCE_EXACT",
    }


def secondary_metric_failure_contract(
    *, dataset_id: str, render_context_ready: bool, reason_suffix: str
) -> list[dict[str, Any]]:
    """Return explicit N/A records; never synthesize LPIPS/SFace values."""

    if not reason_suffix or any(character.isspace() for character in reason_suffix):
        raise ValueError("secondary_metric_reason_suffix_must_be_token")
    common = {
        "schema_version": SECONDARY_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "terminal_state": "METHOD_FAILURE",
        "value": None,
        "render_context_ready": bool(render_context_ready),
        "evaluator_executed": False,
        "fabricated": False,
    }
    return [
        {
            **common,
            "metric_id": "paired_multiview_render_LPIPS_or_DISTS",
            "failure_code": f"METHOD_FAILURE_NA_LPIPS_{reason_suffix}",
        },
        {
            **common,
            "metric_id": "fixed_face_encoder_identity_cosine",
            "failure_code": f"METHOD_FAILURE_NA_SFACE_{reason_suffix}",
        },
    ]


def source_check() -> Mapping[str, Any]:
    rows = secondary_metric_failure_contract(
        dataset_id="D2",
        render_context_ready=True,
        reason_suffix="EVALUATOR_NOT_BOUND_OR_EXECUTED",
    )
    if len(rows) != 2 or any(row["value"] is not None for row in rows):
        raise AssertionError("secondary_metric_source_check_failed")
    return {
        "status": "PASS_CONTEXT_CONTRACT_SYNTHETIC_SOURCE_CHECK",
        "research_evidence": False,
        "private_artifact_reads": 0,
        "real_metric_rows": 0,
        "secondary_values_fabricated": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "SECONDARY_SCHEMA_VERSION",
    "facescape_context_binding",
    "realy_context_binding",
    "secondary_metric_failure_contract",
    "source_check",
]
