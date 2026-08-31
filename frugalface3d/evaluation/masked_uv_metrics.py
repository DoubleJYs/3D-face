"""Frozen-support RGB metrics for canonical UV completion evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskedPSNR:
    value_db: float | None
    perfect_match: bool
    support_texels: int


@dataclass(frozen=True)
class MaskedSSIM:
    value: float
    valid_center_count: int
    support_texels: int


def _validated_pair(
    prediction: Any,
    reference: Any,
    support: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(prediction, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(support)
    if first.ndim != 3 or first.shape[0] != 3 or second.shape != first.shape:
        raise ValueError("prediction_and_reference_must_be_matching_CxHxW_RGB")
    if mask.shape == (1, *first.shape[1:]):
        mask = mask[0]
    if mask.shape != first.shape[1:]:
        raise ValueError("support_shape_mismatch")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("prediction_or_reference_contains_nonfinite_values")
    if min(float(first.min()), float(second.min())) < 0.0 or max(
        float(first.max()), float(second.max())
    ) > 1.0:
        raise ValueError("prediction_and_reference_must_be_in_closed_unit_interval")
    if not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("support_must_be_exact_binary")
    mask = mask.astype(bool, copy=False)
    if not bool(mask.any()):
        raise ValueError("empty_evaluation_support")
    return first, second, mask


def masked_rgb_mae(prediction: Any, reference: Any, support: Any) -> float:
    first, second, mask = _validated_pair(prediction, reference, support)
    return float(np.abs(first - second)[:, mask].mean(dtype=np.float64))


def masked_rgb_psnr(
    prediction: Any,
    reference: Any,
    support: Any,
) -> MaskedPSNR:
    first, second, mask = _validated_pair(prediction, reference, support)
    mse = float(np.square(first - second)[:, mask].mean(dtype=np.float64))
    if mse == 0.0:
        return MaskedPSNR(
            value_db=None,
            perfect_match=True,
            support_texels=int(mask.sum()),
        )
    return MaskedPSNR(
        value_db=float(10.0 * np.log10(1.0 / mse)),
        perfect_match=False,
        support_texels=int(mask.sum()),
    )


def masked_local_ssim(
    prediction: Any,
    reference: Any,
    support: Any,
) -> MaskedSSIM:
    """Average the SSIM map only at centers whose full 11x11 window is valid."""

    first, second, mask = _validated_pair(prediction, reference, support)
    from scipy.ndimage import binary_erosion
    from skimage.metrics import structural_similarity

    valid_centers = binary_erosion(
        mask,
        structure=np.ones((11, 11), dtype=bool),
        border_value=0,
    )
    count = int(valid_centers.sum())
    if count == 0:
        raise ValueError("no_complete_11x11_hidden_support_window")
    _global_value, score_map = structural_similarity(
        np.moveaxis(first, 0, -1),
        np.moveaxis(second, 0, -1),
        data_range=1.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        win_size=11,
        use_sample_covariance=False,
        full=True,
    )
    values = np.asarray(score_map, dtype=np.float64)[valid_centers]
    if not np.isfinite(values).all():
        raise ValueError("masked_ssim_map_contains_nonfinite_values")
    return MaskedSSIM(
        value=float(values.mean(dtype=np.float64)),
        valid_center_count=count,
        support_texels=int(mask.sum()),
    )


__all__ = [
    "MaskedPSNR",
    "MaskedSSIM",
    "masked_local_ssim",
    "masked_rgb_mae",
    "masked_rgb_psnr",
]
