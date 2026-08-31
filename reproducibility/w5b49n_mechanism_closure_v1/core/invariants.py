"""Deterministic MC0 invariants and exact MAE decomposition.

The conserved output is constructed with a boolean ``numpy.where``.  No
blending, thresholding, interpolation, or learned operation is permitted.
MAE calls reuse the repository evaluator rather than reimplementing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from frugalface3d.evaluation.masked_uv_metrics import masked_rgb_mae


class InvariantError(ValueError):
    """An array violates the fixed MC0 tensor or conservation contract."""


@dataclass(frozen=True)
class WhereInvariantReceipt:
    operator_id: str
    observed_texels: int
    hidden_texels: int
    exact_boolean_where: bool
    observed_region_exact: bool
    hidden_region_exact: bool
    observed_max_absolute_error: float
    hidden_max_absolute_change: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaeDecomposition:
    full_mae: float
    observed_mae: float
    hidden_mae: float
    observed_fraction: float
    hidden_fraction: float
    recomposed_mae: float
    absolute_residual: float
    canonical_texels: int
    observed_texels: int
    hidden_texels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeConservedMae:
    native: MaeDecomposition
    conserved: MaeDecomposition
    hidden_mae_delta: float
    hidden_mae_exactly_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "native": self.native.to_dict(),
            "conserved": self.conserved.to_dict(),
            "hidden_mae_delta": self.hidden_mae_delta,
            "hidden_mae_exactly_unchanged": self.hidden_mae_exactly_unchanged,
        }


def _rgb_float32(value: Any, role: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise InvariantError(f"{role}_must_be_numpy_array")
    if value.dtype != np.dtype(np.float32):
        raise InvariantError(f"{role}_must_be_float32")
    if value.ndim != 3 or value.shape[0] != 3:
        raise InvariantError(f"{role}_must_be_chw_rgb")
    if not bool(np.isfinite(value).all()):
        raise InvariantError(f"{role}_contains_nonfinite")
    if bool(np.any((value < 0.0) | (value > 1.0))):
        raise InvariantError(f"{role}_outside_closed_unit_interval")
    return value


def _same_shape(first: np.ndarray, second: np.ndarray, role: str) -> None:
    if first.shape != second.shape:
        raise InvariantError(f"{role}_shape_mismatch")


def _mask_2d(mask: Any, shape: tuple[int, int], role: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise InvariantError(f"{role}_must_be_numpy_array")
    if mask.dtype != np.dtype(bool):
        raise InvariantError(f"{role}_must_be_boolean")
    value = mask
    if value.shape == (1, *shape):
        value = value[0]
    if value.shape != shape:
        raise InvariantError(f"{role}_shape_mismatch")
    return value


def _mask_3d(mask: np.ndarray) -> np.ndarray:
    return np.broadcast_to(mask[None, ...], (3, *mask.shape))


def conserve_observed_where(
    native_uv_chw_float32: np.ndarray,
    observed_uv_chw_float32: np.ndarray,
    observed_mask_bool: np.ndarray,
) -> np.ndarray:
    """Apply ``where(mask, observed, native)`` with exact branch selection."""

    native = _rgb_float32(native_uv_chw_float32, "native_uv")
    observed = _rgb_float32(observed_uv_chw_float32, "observed_uv")
    _same_shape(native, observed, "native_observed")
    mask = _mask_2d(observed_mask_bool, native.shape[1:], "observed_mask")
    result = np.ascontiguousarray(np.where(_mask_3d(mask), observed, native))
    if result.dtype != np.dtype(np.float32):
        raise InvariantError("where_changed_float32_dtype")
    result.setflags(write=False)
    return result


def _maximum_error(first: np.ndarray, second: np.ndarray, support: np.ndarray) -> float:
    if not bool(support.any()):
        raise InvariantError("invariant_support_empty")
    return float(np.max(np.abs(first[support] - second[support])))


def verify_where_invariants(
    *,
    native_uv_chw_float32: np.ndarray,
    observed_uv_chw_float32: np.ndarray,
    observed_mask_bool: np.ndarray,
    conserved_uv_chw_float32: np.ndarray,
) -> WhereInvariantReceipt:
    """Verify exact observed copying and exact hidden-output preservation."""

    native = _rgb_float32(native_uv_chw_float32, "native_uv")
    observed = _rgb_float32(observed_uv_chw_float32, "observed_uv")
    conserved = _rgb_float32(conserved_uv_chw_float32, "conserved_uv")
    _same_shape(native, observed, "native_observed")
    _same_shape(native, conserved, "native_conserved")
    mask_2d = _mask_2d(observed_mask_bool, native.shape[1:], "observed_mask")
    if not bool(mask_2d.any()) or bool(mask_2d.all()):
        raise InvariantError("observed_and_hidden_support_must_both_be_nonempty")
    mask = _mask_3d(mask_2d)
    hidden = np.logical_not(mask)
    expected = conserve_observed_where(native, observed, mask_2d)
    exact_where = bool(np.array_equal(conserved, expected))
    observed_exact = bool(np.array_equal(conserved[mask], observed[mask]))
    hidden_exact = bool(np.array_equal(conserved[hidden], native[hidden]))
    observed_error = _maximum_error(conserved, observed, mask)
    hidden_change = _maximum_error(conserved, native, hidden)
    passed = bool(
        exact_where
        and observed_exact
        and hidden_exact
        and observed_error == 0.0
        and hidden_change == 0.0
    )
    return WhereInvariantReceipt(
        operator_id="numpy.where(boolean_mask, observed, native)",
        observed_texels=int(mask_2d.sum()),
        hidden_texels=int(mask_2d.size - mask_2d.sum()),
        exact_boolean_where=exact_where,
        observed_region_exact=observed_exact,
        hidden_region_exact=hidden_exact,
        observed_max_absolute_error=observed_error,
        hidden_max_absolute_change=hidden_change,
        passed=passed,
    )


def decompose_mae(
    prediction_uv_chw_float32: np.ndarray,
    reference_uv_chw_float32: np.ndarray,
    observed_mask_bool: np.ndarray,
    canonical_mask_bool: np.ndarray | None = None,
) -> MaeDecomposition:
    """Decompose full-support MAE into observed and hidden contributions.

    The returned residual is numerical audit information.  The three MAEs are
    all computed by ``masked_rgb_mae`` from the existing repository evaluator.
    """

    prediction = _rgb_float32(prediction_uv_chw_float32, "prediction_uv")
    reference = _rgb_float32(reference_uv_chw_float32, "reference_uv")
    _same_shape(prediction, reference, "prediction_reference")
    observed_mask = _mask_2d(observed_mask_bool, prediction.shape[1:], "observed_mask")
    if canonical_mask_bool is None:
        canonical = np.ones(prediction.shape[1:], dtype=bool)
    else:
        canonical = _mask_2d(
            canonical_mask_bool, prediction.shape[1:], "canonical_mask"
        )
    observed = np.logical_and(canonical, observed_mask)
    hidden = np.logical_and(canonical, np.logical_not(observed_mask))
    canonical_count = int(canonical.sum())
    observed_count = int(observed.sum())
    hidden_count = int(hidden.sum())
    if canonical_count == 0 or observed_count == 0 or hidden_count == 0:
        raise InvariantError("mae_decomposition_requires_nonempty_full_observed_hidden")
    if observed_count + hidden_count != canonical_count:
        raise InvariantError("mae_support_partition_invalid")

    full_mae = float(masked_rgb_mae(prediction, reference, canonical))
    observed_mae = float(masked_rgb_mae(prediction, reference, observed))
    hidden_mae = float(masked_rgb_mae(prediction, reference, hidden))
    observed_fraction = observed_count / canonical_count
    hidden_fraction = hidden_count / canonical_count
    recomposed = observed_fraction * observed_mae + hidden_fraction * hidden_mae
    residual = abs(full_mae - recomposed)
    if not math.isfinite(residual) or residual > 5e-15:
        raise InvariantError("mae_decomposition_identity_failed")
    return MaeDecomposition(
        full_mae=full_mae,
        observed_mae=observed_mae,
        hidden_mae=hidden_mae,
        observed_fraction=observed_fraction,
        hidden_fraction=hidden_fraction,
        recomposed_mae=recomposed,
        absolute_residual=residual,
        canonical_texels=canonical_count,
        observed_texels=observed_count,
        hidden_texels=hidden_count,
    )


def evaluate_native_conserved_mae(
    *,
    native_uv_chw_float32: np.ndarray,
    conserved_uv_chw_float32: np.ndarray,
    reference_uv_chw_float32: np.ndarray,
    observed_mask_bool: np.ndarray,
    canonical_mask_bool: np.ndarray | None = None,
) -> NativeConservedMae:
    """Show that conservation may change full/observed MAE, never hidden MAE."""

    native = decompose_mae(
        native_uv_chw_float32,
        reference_uv_chw_float32,
        observed_mask_bool,
        canonical_mask_bool,
    )
    conserved = decompose_mae(
        conserved_uv_chw_float32,
        reference_uv_chw_float32,
        observed_mask_bool,
        canonical_mask_bool,
    )
    delta = conserved.hidden_mae - native.hidden_mae
    return NativeConservedMae(
        native=native,
        conserved=conserved,
        hidden_mae_delta=delta,
        hidden_mae_exactly_unchanged=bool(delta == 0.0),
    )


__all__ = [
    "InvariantError",
    "MaeDecomposition",
    "NativeConservedMae",
    "WhereInvariantReceipt",
    "conserve_observed_where",
    "decompose_mae",
    "evaluate_native_conserved_mae",
    "verify_where_invariants",
]
