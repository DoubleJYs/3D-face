"""MPS-resident UV geometry, screen-to-UV, and UV-to-screen helpers.

The functions in this module mirror the frozen NumPy contracts used by the
W5-B-47 full graph, but keep the large intermediate arrays on the Torch device.
They deliberately do not own model selection, thresholds, or any learnable
parameters.  Numerical equivalence is evaluated by the W5-B-48E runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchStaticUVRasterPlan:
    """Device tensors for an already frozen static UV raster plan."""

    uv_size: int
    faces: Any
    pixel_indices: Any
    vertex_indices: Any
    barycentric_weights: Any
    counts: Any


def torch_static_uv_plan(plan: Any, faces: np.ndarray, device: Any) -> TorchStaticUVRasterPlan:
    """Copy a frozen :class:`StaticUVRasterPlan` to ``device`` once."""

    import torch

    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mps_uv_triangular_faces_required")
    return TorchStaticUVRasterPlan(
        uv_size=int(plan.uv_size),
        faces=torch.from_numpy(triangles).to(device=device, dtype=torch.long),
        pixel_indices=torch.from_numpy(
            np.asarray(plan.pixel_indices, dtype=np.int64)
        ).to(device=device, dtype=torch.long),
        vertex_indices=torch.from_numpy(
            np.asarray(plan.vertex_indices, dtype=np.int64)
        ).to(device=device, dtype=torch.long),
        barycentric_weights=torch.from_numpy(
            np.asarray(plan.barycentric_weights, dtype=np.float32)
        ).to(device=device, dtype=torch.float32),
        counts=torch.from_numpy(np.asarray(plan.counts, dtype=np.float32)).to(
            device=device, dtype=torch.float32
        ),
    )


def rasterize_uv_geometry_torch(vertices: Any, plan: TorchStaticUVRasterPlan) -> tuple[Any, Any]:
    """Rasterize normalized positions and vertex normals without leaving Torch."""

    import torch
    import torch.nn.functional as functional

    if vertices.ndim == 3:
        if vertices.shape[0] != 1:
            raise ValueError("mps_uv_batch_one_required")
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mps_uv_vertices_vx3_required")
    value = vertices.to(dtype=torch.float32)
    centered = value - value.mean(dim=0, keepdim=True)
    scale = torch.linalg.vector_norm(centered, dim=1).amax().clamp_min(1e-8)
    normalized_positions = centered / scale

    triangles = value[plan.faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    face_normals = functional.normalize(face_normals, dim=1, eps=1e-8)
    normals = torch.zeros_like(value)
    for corner in range(3):
        normals.index_add_(0, plan.faces[:, corner], face_normals)
    normals = functional.normalize(normals, dim=1, eps=1e-8)
    attributes = torch.cat((normalized_positions, normals), dim=1)

    weights = plan.barycentric_weights
    indices = plan.vertex_indices
    contributions = (
        weights[:, 0, None] * attributes[indices[:, 0]]
        + weights[:, 1, None] * attributes[indices[:, 1]]
        + weights[:, 2, None] * attributes[indices[:, 2]]
    )
    sums = torch.zeros(
        plan.uv_size * plan.uv_size,
        attributes.shape[1],
        device=value.device,
        dtype=torch.float32,
    )
    sums.index_add_(0, plan.pixel_indices, contributions)
    support = plan.counts > 0
    denominator = plan.counts.clamp_min(1.0)[:, None]
    rasterized = torch.where(support[:, None], sums / denominator, torch.zeros_like(sums))
    return (
        rasterized.reshape(plan.uv_size, plan.uv_size, attributes.shape[1])
        .permute(2, 0, 1)[None]
        .contiguous(),
        support.reshape(plan.uv_size, plan.uv_size),
    )


def splat_screen_to_uv_torch(
    source_rgb_uint8: Any,
    screen_uv: Any,
    visibility: Any,
    canonical_mask: Any,
    uv_size: int,
    device: Any,
    *,
    channels_last: bool = False,
) -> tuple[Any, Any, Any]:
    """Nearest-texel mean splat matching the frozen NumPy screen-to-UV rule."""

    import torch

    source_tensor = (
        source_rgb_uint8.to(device=device)
        if torch.is_tensor(source_rgb_uint8)
        else torch.from_numpy(np.asarray(source_rgb_uint8, dtype=np.uint8)).to(device)
    )
    uv_tensor = (
        screen_uv.to(device=device, dtype=torch.float32)
        if torch.is_tensor(screen_uv)
        else torch.from_numpy(np.asarray(screen_uv, dtype=np.float32)).to(
            device=device, dtype=torch.float32
        )
    )
    visible_tensor = (
        visibility.to(device=device) > 0
        if torch.is_tensor(visibility)
        else torch.from_numpy(np.asarray(visibility) > 0).to(device=device)
    )
    if (
        source_tensor.shape[:2] != uv_tensor.shape[:2]
        or uv_tensor.shape[2:] != (2,)
        or visible_tensor.shape != uv_tensor.shape[:2]
    ):
        raise ValueError("mps_screen_uv_shape_contract_changed")
    source_tensor = source_tensor.to(dtype=torch.float32)
    visible_tensor = visible_tensor.to(dtype=torch.bool)
    finite = torch.isfinite(uv_tensor).all(dim=2)
    active = finite & visible_tensor
    safe_uv = torch.where(active[:, :, None], uv_tensor.clamp(0.0, 1.0), torch.zeros_like(uv_tensor))
    x = torch.round(safe_uv[..., 0] * float(uv_size - 1)).to(torch.long)
    y = torch.round((1.0 - safe_uv[..., 1]) * float(uv_size - 1)).to(torch.long)
    flat_index = (y * uv_size + x)[active]
    colors = source_tensor[active]
    sums = torch.zeros(uv_size * uv_size, 3, device=device, dtype=torch.float32)
    counts = torch.zeros(uv_size * uv_size, device=device, dtype=torch.float32)
    sums.index_add_(0, flat_index, colors)
    counts.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.float32))
    occupied = counts > 0
    averaged = torch.where(
        occupied[:, None], sums / counts.clamp_min(1.0)[:, None], torch.zeros_like(sums)
    )
    # The legacy contract rounds the mean in uint8 space before normalization.
    partial = (torch.round(averaged).clamp(0.0, 255.0) / 255.0).reshape(
        uv_size, uv_size, 3
    ).permute(2, 0, 1)[None]
    canonical = canonical_mask.to(device=device, dtype=torch.bool).reshape(uv_size, uv_size)
    valid = occupied.reshape(uv_size, uv_size) & canonical
    visibility_tensor = valid.to(torch.float32)[None, None]
    partial = partial * visibility_tensor
    if channels_last:
        partial = partial.contiguous(memory_format=torch.channels_last)
        visibility_tensor = visibility_tensor.contiguous(memory_format=torch.channels_last)
    else:
        partial = partial.contiguous()
        visibility_tensor = visibility_tensor.contiguous()
    return partial, visibility_tensor, valid


def render_composite_torch(
    completed_uv: Any,
    screen_uv: Any,
    visibility: Any,
    source_rgb_uint8: Any,
    device: Any,
    *,
    return_mask_numpy: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Bilinearly sample UV and composite on device, returning one final frame."""

    import torch
    import torch.nn.functional as functional

    if completed_uv.ndim != 4 or completed_uv.shape[0] != 1 or completed_uv.shape[1] != 3:
        raise ValueError("mps_render_completed_uv_shape_changed")
    uv_tensor = (
        screen_uv.to(device=device, dtype=torch.float32)
        if torch.is_tensor(screen_uv)
        else torch.from_numpy(np.asarray(screen_uv, dtype=np.float32)).to(
            device=device, dtype=torch.float32
        )
    )
    visible_tensor = (
        visibility.to(device=device) > 0
        if torch.is_tensor(visibility)
        else torch.from_numpy(np.asarray(visibility) > 0).to(device=device)
    )
    source_tensor = (
        source_rgb_uint8.to(device=device)
        if torch.is_tensor(source_rgb_uint8)
        else torch.from_numpy(np.asarray(source_rgb_uint8, dtype=np.uint8)).to(device)
    )
    if (
        uv_tensor.shape[:2] != source_tensor.shape[:2]
        or uv_tensor.shape[2:] != (2,)
        or visible_tensor.shape != uv_tensor.shape[:2]
    ):
        raise ValueError("mps_render_screen_contract_changed")
    finite = torch.isfinite(uv_tensor).all(dim=2)
    active = finite & visible_tensor.to(dtype=torch.bool)
    safe = torch.where(finite[:, :, None], uv_tensor.clamp(0.0, 1.0), torch.zeros_like(uv_tensor))
    grid = torch.empty_like(safe)
    grid[..., 0] = 2.0 * safe[..., 0] - 1.0
    grid[..., 1] = 2.0 * (1.0 - safe[..., 1]) - 1.0
    sampled = functional.grid_sample(
        completed_uv.to(dtype=torch.float32),
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].permute(1, 2, 0)
    source_tensor = source_tensor.to(dtype=torch.float32)
    rendered_uint8_space = torch.round(sampled.clamp(0.0, 1.0) * 255.0)
    final = torch.where(active[:, :, None], rendered_uint8_space, source_tensor)
    result = final.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    mask = active.cpu().numpy() if return_mask_numpy else None
    return result, mask


__all__ = [
    "TorchStaticUVRasterPlan",
    "rasterize_uv_geometry_torch",
    "render_composite_torch",
    "splat_screen_to_uv_torch",
    "torch_static_uv_plan",
]
