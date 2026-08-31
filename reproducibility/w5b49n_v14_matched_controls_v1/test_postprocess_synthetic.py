#!/usr/bin/env python3
"""CPU-only synthetic contract tests for the V14 post-processing stage.

The tests do not load a face image, checkpoint, evaluator model, or result
archive.  They validate support definitions, metric-ledger states, SFace
failure symmetry helpers, qualification fail-closed behavior, and binary-path
resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reproducibility.w5b49n_v14_matched_controls_v1 import postprocess as pp


@dataclass
class _Cache:
    tensors: dict[str, np.ndarray]


def _pair(identity: str = "D1-I001", pair_id: str = "D1-P0000") -> pp.PairRow:
    return pp.PairRow(
        dataset_id="D1",
        pair_index=0,
        pair_id=pair_id,
        identity_token=identity,
        source_index=0,
        target_index=1,
        source_runtime_id="D1-S000",
        target_runtime_id="D1-S001",
        target_frame_manifest_id="f" * 64,
    )


def test_supports_and_mae() -> None:
    canonical = np.ones((2, 1, 64, 64), dtype=np.uint8)
    source_visible = np.zeros((64, 64), dtype=np.uint8)
    target_visible = np.zeros((64, 64), dtype=np.uint8)
    source_visible[0, 0] = 1
    source_visible[1, 1] = 1
    target_visible[0, 0] = 1
    target_visible[2, 2] = 1
    visibility = np.stack((source_visible, target_visible), axis=0)[:, None]
    cache = _Cache({"visibility": visibility, "canonical_mask": canonical})
    hidden, all_target = pp.uv_supports(cache, _pair())
    assert int(hidden.sum()) == 1
    assert bool(hidden[2, 2])
    assert int(all_target.sum()) == 2
    reference = np.zeros((3, 64, 64), dtype=np.float32)
    prediction = reference.copy()
    prediction[:, 2, 2] = np.float32(0.75)
    assert pp.masked_rgb_mae(prediction, reference, hidden) == 0.75
    assert pp.masked_rgb_mae(prediction, reference, np.zeros_like(hidden)) is None


def test_metric_ledger_states() -> None:
    pair = _pair()
    route = pp.EndpointRoute(
        route_id="full__2026080447",
        method_id="full",
        seed=2026080447,
        output_mode="conserved",
        values=np.zeros((2, 3, 64, 64), dtype=np.float32),
        origin="synthetic",
        bound_sha256="0" * 64,
    )
    complete = pp._analysis_metric_row(
        pair, route, "hidden_uv_mae", 0.125, 17, None
    )
    assert complete["terminal_state"] == "COMPLETE"
    assert complete["failure_code"] is None
    failed = pp._analysis_metric_row(
        pair,
        route,
        "sface_source_to_render_cosine",
        None,
        17,
        "SOURCE_DETECTION_FAILURE",
    )
    assert failed["terminal_state"] == "EVALUATION_FAILURE"
    assert failed["value"] is None
    try:
        pp._analysis_metric_row(
            pair, route, "lpips_alex_v0_1", None, 17, "UNDECLARED"
        )
    except pp.PostprocessError as error:
        assert str(error).startswith("non_sface_metric_must_be_complete")
    else:
        raise AssertionError("non-SFace missing metric did not fail closed")


def test_sface_failure_mapping_and_coverage() -> None:
    assert (
        pp._sface_failure_code(
            "source_face_count_0_expected_1",
            source_failure="source_face_count_0_expected_1",
            target_failure=None,
        )
        == "SOURCE_DETECTION_FAILURE"
    )
    assert (
        pp._sface_failure_code(
            "target_reference_face_count_0_expected_1",
            source_failure=None,
            target_failure="target_reference_face_count_0_expected_1",
        )
        == "TARGET_DETECTION_FAILURE"
    )
    assert (
        pp._sface_failure_code(
            "method_x_feature_nonfinite",
            source_failure=None,
            target_failure=None,
        )
        == "NONFINITE_EMBEDDING"
    )
    assert (
        pp._sface_failure_code(
            "method_x_feature_invalid",
            source_failure=None,
            target_failure=None,
        )
        == "METHOD_EMBEDDING_FAILURE"
    )

    rows: list[dict[str, object]] = []
    for identity, pair_id, failed in (
        ("D1-I001", "D1-P0000", False),
        ("D1-I002", "D1-P0001", True),
    ):
        for index in range(16):
            rows.append(
                {
                    "dataset_id": "D1",
                    "metric_id": "sface_source_to_render_cosine",
                    "identity_token": identity,
                    "pair_id": pair_id,
                    "terminal_state": (
                        "EVALUATION_FAILURE" if failed and index == 0 else "COMPLETE"
                    ),
                }
            )
    coverage = pp._identity_coverage(
        rows, "D1", "sface_source_to_render_cosine"
    )
    assert coverage["identity_count"] == 2
    assert coverage["covered_identity_count"] == 1
    assert coverage["covered_identities"] == ["D1-I001"]


def test_cv2_binary_resolution() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "cv2"
        root.mkdir()
        module = root / "__init__.py"
        binary = root / "cv2.abi3.so"
        module.write_text("", encoding="utf-8")
        binary.write_bytes(b"synthetic-not-loadable")
        fake = SimpleNamespace(__file__=str(module))
        assert pp._cv2_binary_path(fake) == binary.resolve()


def test_qualification_failure_retained_without_data_reads() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assets = root / "empty-assets"
        output = root / "qualification"
        assets.mkdir()
        overall = pp.qualify(assets, output, formal=False)
        assert overall["status"] == "TERMINAL_V14_EVALUATOR_QUALIFICATION_WITH_RETAINED_FAILURES"
        assert overall["real_image_reads"] == 0
        assert overall["metric_rows"] == 0
        for name, metric in (
            ("LPIPS_LINUX_QUALIFICATION_TERMINAL.json", "lpips_alex_v0_1"),
            (
                "SFACE_LINUX_QUALIFICATION_TERMINAL.json",
                "sface_source_to_render_cosine",
            ),
        ):
            terminal = json.loads((output / name).read_text(encoding="utf-8"))
            assert terminal["status"] == "METHOD_FAILURE_V14_EVALUATOR_QUALIFICATION"
            assert terminal["metric_id"] == metric
            assert terminal["device_backend"] == "cpu"
            assert terminal["cuda_calls"] == 0
            assert terminal["real_image_reads"] == 0
            assert terminal["metric_rows"] == 0
        try:
            pp.qualify(assets, output, formal=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("qualification root was reused")


def main() -> int:
    pp.source_check()
    test_supports_and_mae()
    test_metric_ledger_states()
    test_sface_failure_mapping_and_coverage()
    test_cv2_binary_resolution()
    test_qualification_failure_retained_without_data_reads()
    print(
        json.dumps(
            {
                "status": "PASS_V14_POSTPROCESS_SYNTHETIC",
                "real_image_reads": 0,
                "scientific_result_generated": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
