"""Frozen paired-render preprocessing and injected LPIPS evaluation.

This module deliberately does not import ``lpips`` or resolve model weights.
The caller must provide an already constructed evaluator whose source, version,
weights, and licence have been frozen by the enclosing experiment contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


TARGET_HEIGHT = 128
TARGET_WIDTH = 128
NEUTRAL_BACKGROUND = 0.5
BILINEAR_ALIGN_CORNERS = False


@dataclass(frozen=True)
class PairedRenderLPIPSInput:
    """A single, symmetrically masked render pair ready for LPIPS."""

    prediction: np.ndarray
    reference: np.ndarray
    common_face_texels: int
    common_face_mask_sha256: str
    target_frame_manifest_id: str
    interpolation: str = "bilinear"
    align_corners: bool = BILINEAR_ALIGN_CORNERS
    neutral_background: float = NEUTRAL_BACKGROUND


@dataclass(frozen=True)
class InjectedLPIPSResult:
    """Finite scalar returned by an externally supplied LPIPS evaluator."""

    value: float
    device: str
    evaluator_manifest_id: str
    common_face_mask_sha256: str
    target_frame_manifest_id: str


def _nonempty_manifest_id(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role}_must_be_a_nonempty_string")
    return value


def common_face_mask_sha256(mask: np.ndarray) -> str:
    """Hash a mask with its exact shape and canonical C-order uint8 bytes."""

    binary = np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    digest = hashlib.sha256()
    digest.update(np.asarray(binary.shape, dtype=np.dtype("<u8")).tobytes())
    digest.update(binary.tobytes(order="C"))
    return digest.hexdigest()


def _validate_rgb_pair(
    prediction: Any,
    reference: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(prediction, np.ndarray) or not isinstance(reference, np.ndarray):
        raise TypeError("prediction_and_reference_must_be_numpy_arrays")
    if prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError("prediction_must_be_CxHxW_RGB")
    if reference.shape != prediction.shape:
        raise ValueError("prediction_and_reference_must_be_matching_CxHxW_RGB")
    if prediction.dtype.kind != "f" or reference.dtype.kind != "f":
        raise TypeError("prediction_and_reference_must_have_floating_dtype")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("prediction_or_reference_contains_nonfinite_values")
    if (
        float(prediction.min()) < 0.0
        or float(prediction.max()) > 1.0
        or float(reference.min()) < 0.0
        or float(reference.max()) > 1.0
    ):
        raise ValueError("prediction_and_reference_must_be_in_closed_unit_interval")
    return prediction, reference


def _validate_common_face_mask(mask: Any, height: int, width: int) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("common_face_mask_must_be_a_numpy_array")
    if mask.shape != (height, width):
        raise ValueError("common_face_mask_must_match_render_HxW")
    if mask.dtype.kind not in "buif":
        raise TypeError("common_face_mask_must_have_numeric_or_boolean_dtype")
    if not np.isfinite(mask).all():
        raise ValueError("common_face_mask_contains_nonfinite_values")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("common_face_mask_must_be_exact_binary")
    binary = mask.astype(bool, copy=False)
    if not bool(binary.any()):
        raise ValueError("common_face_mask_must_not_be_empty")
    return binary


def prepare_paired_render_for_lpips(
    prediction: Any,
    reference: Any,
    common_face_mask: Any,
    *,
    target_frame_manifest_id: str,
) -> PairedRenderLPIPSInput:
    """Prepare one method-independent, common-mask render pair.

    Inputs are strictly matching ``C x H x W`` RGB NumPy arrays with floating
    dtype and values in ``[0, 1]``. ``common_face_mask`` is one exact-binary
    ``H x W`` array shared by both images; method-specific masks are not
    accepted by this API. Pixels outside that mask are replaced with 0.5 in
    both images before deterministic 128 x 128 bilinear resizing. The returned
    arrays have shape ``1 x 3 x 128 x 128`` and range ``[-1, 1]``.
    """

    frame_id = _nonempty_manifest_id(
        target_frame_manifest_id, "target_frame_manifest_id"
    )
    pred, ref = _validate_rgb_pair(prediction, reference)
    mask = _validate_common_face_mask(common_face_mask, pred.shape[1], pred.shape[2])
    mask_sha256 = common_face_mask_sha256(mask)

    pred_masked = np.where(mask[None, :, :], pred, NEUTRAL_BACKGROUND).astype(
        np.float32, copy=False
    )
    ref_masked = np.where(mask[None, :, :], ref, NEUTRAL_BACKGROUND).astype(
        np.float32, copy=False
    )

    import torch
    import torch.nn.functional as functional

    pair = torch.from_numpy(np.stack((pred_masked, ref_masked), axis=0))
    resized = functional.interpolate(
        pair,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=BILINEAR_ALIGN_CORNERS,
        antialias=False,
    )
    normalized = resized.mul(2.0).sub(1.0).cpu().numpy()
    if normalized.shape != (2, 3, TARGET_HEIGHT, TARGET_WIDTH):
        raise RuntimeError("unexpected_paired_render_preprocessing_shape")
    if not np.isfinite(normalized).all():
        raise RuntimeError("paired_render_preprocessing_produced_nonfinite_values")
    if float(normalized.min()) < -1.0 or float(normalized.max()) > 1.0:
        raise RuntimeError("paired_render_preprocessing_exceeded_lpips_range")

    return PairedRenderLPIPSInput(
        prediction=np.ascontiguousarray(normalized[0:1]),
        reference=np.ascontiguousarray(normalized[1:2]),
        common_face_texels=int(mask.sum()),
        common_face_mask_sha256=mask_sha256,
        target_frame_manifest_id=frame_id,
    )


def _validate_prepared_input(prepared: PairedRenderLPIPSInput) -> None:
    if not isinstance(prepared, PairedRenderLPIPSInput):
        raise TypeError("prepared_must_be_PairedRenderLPIPSInput")
    if prepared.interpolation != "bilinear":
        raise ValueError("prepared_interpolation_must_be_frozen_bilinear")
    if prepared.align_corners is not BILINEAR_ALIGN_CORNERS:
        raise ValueError("prepared_align_corners_must_be_frozen_false")
    if prepared.neutral_background != NEUTRAL_BACKGROUND:
        raise ValueError("prepared_neutral_background_must_be_frozen_0_5")
    if (
        isinstance(prepared.common_face_texels, (bool, np.bool_))
        or not isinstance(prepared.common_face_texels, (int, np.integer))
        or int(prepared.common_face_texels) <= 0
    ):
        raise ValueError("prepared_common_face_texels_must_be_a_positive_integer")
    if (
        not isinstance(prepared.common_face_mask_sha256, str)
        or len(prepared.common_face_mask_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in prepared.common_face_mask_sha256
        )
    ):
        raise ValueError("prepared_common_face_mask_sha256_must_be_lowercase_sha256")
    _nonempty_manifest_id(
        prepared.target_frame_manifest_id, "prepared_target_frame_manifest_id"
    )
    expected = (1, 3, TARGET_HEIGHT, TARGET_WIDTH)
    for name, array in (
        ("prediction", prepared.prediction),
        ("reference", prepared.reference),
    ):
        if not isinstance(array, np.ndarray) or array.shape != expected:
            raise ValueError(f"prepared_{name}_must_be_NCHW_1x3x128x128")
        if array.dtype.kind != "f":
            raise TypeError(f"prepared_{name}_must_have_floating_dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"prepared_{name}_contains_nonfinite_values")
        if float(array.min()) < -1.0 or float(array.max()) > 1.0:
            raise ValueError(f"prepared_{name}_must_be_in_closed_minus_one_one_interval")


def evaluate_injected_lpips(
    prepared: PairedRenderLPIPSInput,
    evaluator: Any,
    *,
    device: str,
    evaluator_manifest_id: str,
) -> InjectedLPIPSResult:
    """Evaluate a prepared pair with an injected, already-frozen evaluator.

    The evaluator must implement the standard ``torch.nn.Module`` surface:
    ``to(device)``, ``eval()``, and a two-tensor call. Evaluation is forced into
    eval mode and ``torch.no_grad()``. All evaluator parameters/buffers, both
    inputs, and the scalar output must reside on the requested device. Missing
    MPS/CUDA support fails closed; there is no device fallback.
    """

    _validate_prepared_input(prepared)
    manifest_id = _nonempty_manifest_id(
        evaluator_manifest_id, "evaluator_manifest_id"
    )
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device_must_be_a_nonempty_explicit_string")
    if not hasattr(evaluator, "to") or not callable(evaluator.to):
        raise TypeError("injected_lpips_evaluator_must_implement_to")
    if not hasattr(evaluator, "eval") or not callable(evaluator.eval):
        raise TypeError("injected_lpips_evaluator_must_implement_eval")
    if not callable(evaluator):
        raise TypeError("injected_lpips_evaluator_must_be_callable")

    import torch

    requested = torch.device(device)
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested_mps_device_is_unavailable_no_fallback")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested_cuda_device_is_unavailable_no_fallback")

    moved = evaluator.to(requested)
    if moved is None:
        raise RuntimeError("injected_lpips_evaluator_to_returned_none")
    evaluator = moved
    evaluated = evaluator.eval()
    if evaluated is not None:
        evaluator = evaluated
    if bool(getattr(evaluator, "training", False)):
        raise RuntimeError("injected_lpips_evaluator_remained_in_training_mode")

    for collection_name in ("parameters", "buffers"):
        collection = getattr(evaluator, collection_name, None)
        if collection is None:
            continue
        if not callable(collection):
            raise TypeError(f"injected_lpips_evaluator_{collection_name}_must_be_callable")
        for tensor in collection():
            if tensor.device != requested:
                raise RuntimeError("injected_lpips_evaluator_device_mismatch")

    pred = torch.from_numpy(np.ascontiguousarray(prepared.prediction)).to(requested)
    ref = torch.from_numpy(np.ascontiguousarray(prepared.reference)).to(requested)
    if pred.device != requested or ref.device != requested:
        raise RuntimeError("paired_render_input_device_mismatch")

    with torch.no_grad():
        output = evaluator(pred, ref)
    if not isinstance(output, torch.Tensor):
        raise TypeError("injected_lpips_evaluator_must_return_a_tensor")
    if output.device != requested:
        raise RuntimeError("injected_lpips_output_device_mismatch")
    if output.numel() != 1:
        raise ValueError("injected_lpips_evaluator_must_return_one_scalar_per_pair")
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("injected_lpips_evaluator_returned_nonfinite_output")
    return InjectedLPIPSResult(
        value=float(output.item()),
        device=str(requested),
        evaluator_manifest_id=manifest_id,
        common_face_mask_sha256=prepared.common_face_mask_sha256,
        target_frame_manifest_id=prepared.target_frame_manifest_id,
    )


__all__ = [
    "BILINEAR_ALIGN_CORNERS",
    "InjectedLPIPSResult",
    "NEUTRAL_BACKGROUND",
    "PairedRenderLPIPSInput",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "common_face_mask_sha256",
    "evaluate_injected_lpips",
    "prepare_paired_render_for_lpips",
]
