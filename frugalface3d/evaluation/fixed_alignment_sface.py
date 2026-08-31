"""In-memory SFace identity audit with one frozen target alignment per tuple.

The source image and paired target reference are detected exactly once.  A
method render is never detected: every method is aligned with the canonical
face row detected on the paired target reference.  The wrapper never returns
an embedding; persistence inside an injected recognizer is outside its proof
scope and must be covered by the enclosing runtime manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


FEATURE_DIMENSION = 128


@dataclass(frozen=True)
class SFaceMethodResult:
    """One method's source-to-render identity-proxy result."""

    method_id: str
    status: str
    cosine: float | None
    failure_reason: str | None
    target_alignment_shared: bool
    target_frame_manifest_id: str
    runtime_manifest_id: str
    method_detection_calls: int = 0
    wrapper_returned_embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedAlignmentSFaceResult:
    """Tuple-level audit without biometric feature vectors."""

    source_detection_calls: int
    target_reference_detection_calls: int
    method_detection_calls: int
    retry_count: int
    source_face_count: int | None
    target_reference_face_count: int | None
    source_failure_reason: str | None
    target_reference_failure_reason: str | None
    target_frame_manifest_id: str
    runtime_manifest_id: str
    all_method_renders_share_target_alignment: bool
    runtime_device_attested_by_wrapper: bool
    methods: tuple[SFaceMethodResult, ...]
    wrapper_returned_embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "methods": [item.to_dict() for item in self.methods],
        }


@dataclass(frozen=True)
class _Detection:
    calls: int
    face_count: int | None
    face_row: np.ndarray | None
    failure_reason: str | None


def _load_cv2() -> Any:
    """Import OpenCV only when evaluation is actually requested."""

    import cv2

    return cv2


def _image_validation_reason(image: Any, role: str) -> str | None:
    if not isinstance(image, np.ndarray):
        return f"{role}_must_be_numpy_array"
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        return f"{role}_must_be_HxWx3_RGB"
    if np.issubdtype(image.dtype, np.number):
        if np.iscomplexobj(image):
            return f"{role}_must_be_real_uint8"
        try:
            finite = bool(np.isfinite(image).all())
        except TypeError:
            finite = False
        if not finite:
            return f"{role}_contains_nonfinite_values"
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if minimum < 0.0 or maximum > 255.0:
            return f"{role}_values_outside_closed_0_255"
    if image.dtype != np.dtype(np.uint8):
        return f"{role}_must_have_exact_uint8_dtype"
    return None


def _prepare_bgr(
    image: Any,
    *,
    role: str,
    input_size: int,
    cv2_module: Any,
) -> tuple[np.ndarray | None, str | None]:
    reason = _image_validation_reason(image, role)
    if reason is not None:
        return None, reason
    bgr = cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)
    resized = cv2_module.resize(
        bgr,
        (input_size, input_size),
        interpolation=cv2_module.INTER_AREA,
    )
    resized = np.asarray(resized)
    if resized.dtype != np.dtype(np.uint8) or resized.shape != (
        input_size,
        input_size,
        3,
    ):
        return None, f"{role}_prepared_image_contract_failed"
    return np.ascontiguousarray(resized), None


def _detect_single_face(
    prepared_bgr: np.ndarray | None,
    preparation_reason: str | None,
    *,
    role: str,
    detector: Any,
    input_size: int,
) -> _Detection:
    if preparation_reason is not None or prepared_bgr is None:
        return _Detection(0, None, None, preparation_reason)
    detector.setInputSize((input_size, input_size))
    output = detector.detect(prepared_bgr)
    faces = output[1] if isinstance(output, tuple) and len(output) >= 2 else output
    if faces is None:
        count = 0
    else:
        faces_array = np.asarray(faces)
        if faces_array.ndim != 2:
            return _Detection(1, None, None, f"{role}_detector_output_invalid")
        count = int(faces_array.shape[0])
    if count != 1:
        return _Detection(
            1,
            count,
            None,
            f"{role}_face_count_{count}_expected_1",
        )
    try:
        row = np.ascontiguousarray(np.asarray(faces)[0], dtype=np.dtype("<f4"))
        if row.ndim != 1 or row.size == 0 or not bool(np.isfinite(row).all()):
            raise ValueError("invalid_face_row")
    except (TypeError, ValueError, OverflowError):
        return _Detection(1, 1, None, f"{role}_face_row_invalid")
    return _Detection(1, 1, row, None)


def _normalized_feature(
    prepared_bgr: np.ndarray,
    face_row: np.ndarray,
    *,
    role: str,
    recognizer: Any,
) -> tuple[np.ndarray | None, str | None]:
    aligned = recognizer.alignCrop(prepared_bgr, face_row)
    raw = recognizer.feature(aligned)
    value = np.asarray(raw)
    if not np.isrealobj(value):
        return None, f"{role}_feature_must_be_real"
    try:
        feature = np.ascontiguousarray(value, dtype=np.dtype("<f4")).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None, f"{role}_feature_invalid"
    if feature.size != FEATURE_DIMENSION:
        return None, f"{role}_feature_dimension_{feature.size}_expected_{FEATURE_DIMENSION}"
    if not bool(np.isfinite(feature).all()):
        return None, f"{role}_feature_nonfinite"
    feature64 = feature.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(feature64))
    if not math.isfinite(norm) or norm <= 0.0:
        return None, f"{role}_feature_zero_or_invalid_norm"
    normalized = np.ascontiguousarray(feature64 / norm)
    if not bool(np.isfinite(normalized).all()):
        return None, f"{role}_normalized_feature_nonfinite"
    return normalized, None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.dot(left, right))
    if not math.isfinite(value) or value < -1.000001 or value > 1.000001:
        raise ValueError("normalized_cosine_out_of_range")
    return float(np.clip(value, -1.0, 1.0))


def evaluate_fixed_alignment_sface_tuple(
    source_rgb_uint8: Any,
    target_reference_rgb_uint8: Any,
    method_renders_rgb_uint8: Mapping[str, Any],
    *,
    detector: Any,
    recognizer: Any,
    detector_input_size: int = 320,
    cv2_module: Any | None = None,
    target_frame_manifest_id: str,
    runtime_manifest_id: str,
) -> FixedAlignmentSFaceResult:
    """Evaluate source-to-render SFace cosine with one target alignment.

    Runtime or data failures produce explicit ``FAILED`` method rows with a
    ``None`` cosine.  No value is imputed, no detection/recognition is retried,
    and no feature vector is returned or persisted.
    """

    if not isinstance(method_renders_rgb_uint8, Mapping) or not method_renders_rgb_uint8:
        raise ValueError("method_renders_must_be_nonempty_mapping")
    method_items = list(method_renders_rgb_uint8.items())
    if any(not isinstance(name, str) or not name for name, _ in method_items):
        raise ValueError("method_id_must_be_nonempty_string")
    if isinstance(detector_input_size, bool) or not isinstance(detector_input_size, int):
        raise ValueError("detector_input_size_must_be_positive_integer")
    if detector_input_size <= 0:
        raise ValueError("detector_input_size_must_be_positive_integer")
    if not isinstance(target_frame_manifest_id, str) or not target_frame_manifest_id.strip():
        raise ValueError("target_frame_manifest_id_must_be_nonempty_string")
    if not isinstance(runtime_manifest_id, str) or not runtime_manifest_id.strip():
        raise ValueError("runtime_manifest_id_must_be_nonempty_string")
    cv2_backend = _load_cv2() if cv2_module is None else cv2_module

    source_bgr, source_preparation_reason = _prepare_bgr(
        source_rgb_uint8,
        role="source_rgb",
        input_size=detector_input_size,
        cv2_module=cv2_backend,
    )
    target_bgr, target_preparation_reason = _prepare_bgr(
        target_reference_rgb_uint8,
        role="target_reference_rgb",
        input_size=detector_input_size,
        cv2_module=cv2_backend,
    )
    source_detection = _detect_single_face(
        source_bgr,
        source_preparation_reason,
        role="source",
        detector=detector,
        input_size=detector_input_size,
    )
    target_detection = _detect_single_face(
        target_bgr,
        target_preparation_reason,
        role="target_reference",
        detector=detector,
        input_size=detector_input_size,
    )

    source_feature: np.ndarray | None = None
    source_failure = source_detection.failure_reason
    if source_failure is None:
        assert source_bgr is not None and source_detection.face_row is not None
        source_feature, source_failure = _normalized_feature(
            source_bgr,
            source_detection.face_row,
            role="source",
            recognizer=recognizer,
        )

    target_failure = target_detection.failure_reason
    results: list[SFaceMethodResult] = []
    for method_id, method_image in method_items:
        failure = source_failure or target_failure
        cosine: float | None = None
        if failure is None:
            failure = _image_validation_reason(
                method_image, f"method_{method_id}_rgb"
            )
            if failure is None and method_image.shape != target_reference_rgb_uint8.shape:
                failure = f"method_{method_id}_rgb_must_match_target_reference_HxWx3"
            method_bgr: np.ndarray | None = None
            if failure is None:
                method_bgr, failure = _prepare_bgr(
                    method_image,
                    role=f"method_{method_id}_rgb",
                    input_size=detector_input_size,
                    cv2_module=cv2_backend,
                )
            if failure is None:
                assert method_bgr is not None
                assert target_detection.face_row is not None
                method_feature, failure = _normalized_feature(
                    method_bgr,
                    target_detection.face_row,
                    role=f"method_{method_id}",
                    recognizer=recognizer,
                )
                if failure is None:
                    assert source_feature is not None and method_feature is not None
                    try:
                        cosine = _cosine(source_feature, method_feature)
                    except ValueError as error:
                        failure = str(error)
        results.append(
            SFaceMethodResult(
                method_id=method_id,
                status="SUCCESS" if failure is None else "FAILED",
                cosine=cosine if failure is None else None,
                failure_reason=failure,
                target_alignment_shared=True,
                target_frame_manifest_id=target_frame_manifest_id,
                runtime_manifest_id=runtime_manifest_id,
            )
        )

    return FixedAlignmentSFaceResult(
        source_detection_calls=source_detection.calls,
        target_reference_detection_calls=target_detection.calls,
        method_detection_calls=0,
        retry_count=0,
        source_face_count=source_detection.face_count,
        target_reference_face_count=target_detection.face_count,
        source_failure_reason=source_failure,
        target_reference_failure_reason=target_failure,
        target_frame_manifest_id=target_frame_manifest_id,
        runtime_manifest_id=runtime_manifest_id,
        all_method_renders_share_target_alignment=True,
        runtime_device_attested_by_wrapper=False,
        methods=tuple(results),
    )


__all__ = [
    "FEATURE_DIMENSION",
    "FixedAlignmentSFaceResult",
    "SFaceMethodResult",
    "evaluate_fixed_alignment_sface_tuple",
]
