#!/usr/bin/env python3
"""Fail-closed V14 post-processing for matched UV-completion controls.

The runner has two deliberately separate phases:

``qualify``
    Requalifies the frozen LPIPS TorchScript asset and the YuNet/SFace assets
    in the current Linux CPU post-processing environment.  It reads no
    dataset image or metric input.  The historical cloud METHOD_FAILURE is
    retained as provenance and is never renamed or reused as a PASS.

``execute``
    Consumes already-frozen V14 raw routes, frozen B-lite/LaMa/ZITS endpoints,
    the frozen FreeUV V1.2 package, paired-view authorities, and the existing
    render caches.  It computes hidden-support (H) and target-visible-support
    (A) MAE, materializes method renders against the exact shared target
    frames, and evaluates LPIPS/SFace only when their current-host
    qualification receipts pass.

No model training, geometry estimation, baseline inference, FreeUV inference,
retry, imputation, or result-driven sample selection is implemented here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402


PROGRAM_ID = "FRUGALFACE3D-W5B49N-V14-POSTPROCESS-V1_1"
TERMINAL_SCHEMA = "frugalface3d.w5b49n.v14.terminal.v1"
PAIR_ROSTER_SCHEMA = "frugalface3d.w5b49n.v14.pair_roster.v1"
METRIC_ROW_SCHEMA = "frugalface3d.w5b49n.v14.metric_row.v1"
PAIR_METRIC_SCHEMA = "frugalface3d.w5b49n.v14.pair_metric.v1"
RENDER_ROW_SCHEMA = "frugalface3d.w5b49n.v14.render_row.v1"

SEEDS = (2026080447, 2026080448, 2026080449, 2026080450, 2026080451)
SEEDED_METHODS = ("full", "condition0", "b_lite_ft")
FIXED_ANALYSIS_METHOD = "freeuv_conserved"
FIXED_CONTEXT_METHODS = (
    "b_lite",
    "lama",
    "zits",
    "freeuv_native",
    "freeuv_conserved",
)
EXPECTED_ROUTE_COUNT = 20
QUALIFICATION_TOLERANCE_ABS = 1.0e-6
SFACE_FORMAL_OPENCV_VERSION = "4.10.0"
NEUTRAL_BACKGROUND_RGB_UINT8 = 128
SAFE_FAILURE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")

FREEUV_ARCHIVE_SHA256 = (
    "25e26864d5cf6429171faf76c3575944a7f860315e3835b32adbe7f5710e418c"
)
FREEUV_D1_SHA256 = (
    "3f25312879e395676aeac32c5ee3a1d1b08bb3db3703bb34ac73a28a0ee02ff0"
)
FREEUV_D2_SHA256 = (
    "60d6ad02174cbdae1ea466e5a94e1f7b456fc7537aa11c91d10235f05a67e430"
)
FREEUV_TARGET_MANIFEST_SHA256 = (
    "63008d05585994d0ec2e7830e8739a61ce4b38897b2bb3c495a19dd9b2e0c616"
)
FREEUV_RENDER_MANIFEST_SHA256 = (
    "f701dd3931c995de947e732838ad6acdaf2ae1665aeb633cc746df08b02b4357"
)
FREEUV_SAFE_SAMPLE_MAP_SHA256 = (
    "ba64333dc39daafdfb45a13363705fb4cf8e716d4cc6901c352e744b78dcbeb2"
)

LPIPS_FILES = {
    "artifact": (
        "lpips_alex_v0_1_cpu.ts",
        "270f23e5609e49e1bb3289d0964e381a81380d6b2889f4e8767de29bc17e4fea",
    ),
    "manifest": (
        "lpips_alex_v0_1_cpu.manifest.json",
        "6dd95f62e305af568d829e5e5cdecd5e3d31dedc0b4539ebe672ce1106136fab",
    ),
    "export": (
        "lpips_alex_v0_1_cpu.export_receipt.json",
        "90e5aa39e1a558fdebad34c5307ef0179dad7b13f95ef448eb680e3697ef8348",
    ),
    "source_receipt": (
        "SOURCE_DIRECT_LPIPS_EXPORT_RECEIPT.json",
        "bb5e221011c9d323c5b6f1babe90e7b745f91beb0c09fccce5edcb0009f0674b",
    ),
}
LPIPS_EXPECTED_OPERATORS = (
    "aten::_convolution",
    "aten::add.Tensor",
    "aten::div.Tensor",
    "aten::max_pool2d",
    "aten::mean.dim",
    "aten::pow.Tensor_Scalar",
    "aten::relu_",
    "aten::sqrt",
    "aten::sub.Tensor",
    "aten::sum.dim_IntList",
)

SFACE_FILES = {
    "yunet_model": (
        "sface/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "sface_model": (
        "sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "yunet_license": (
        "sface/LICENSE.yunet",
        "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958",
    ),
    "sface_license": (
        "sface/LICENSE.sface",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
}


class PostprocessError(RuntimeError):
    """A contract or runtime condition failed closed."""


@dataclass(frozen=True)
class EndpointRoute:
    route_id: str
    method_id: str
    seed: int | None
    output_mode: str
    values: np.ndarray
    origin: str
    bound_sha256: str


@dataclass(frozen=True)
class PairRow:
    dataset_id: str
    pair_index: int
    pair_id: str
    identity_token: str
    source_index: int
    target_index: int
    source_runtime_id: str
    target_runtime_id: str
    target_frame_manifest_id: str


@dataclass
class QualifiedEvaluators:
    lpips: Any | None
    lpips_terminal: Mapping[str, Any]
    sface_detector: Any | None
    sface_recognizer: Any | None
    sface_cv2: Any | None
    sface_runtime_id: str | None
    sface_terminal: Mapping[str, Any]


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PostprocessError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_array_sha256(value: Any) -> str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.dtype("<u8")).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)
    return sha256_file(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path), count


def read_json(path: Path, role: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"plain_json_required:{role}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                PostprocessError(f"nonfinite_json:{role}:{item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PostprocessError(f"invalid_json:{role}") from error
    require(isinstance(value, dict), f"json_object_required:{role}")
    return value


def read_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"plain_jsonl_required:{role}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PostprocessError(f"invalid_jsonl:{role}:{line_number}") from error
            require(isinstance(value, dict), f"jsonl_object_required:{role}:{line_number}")
            rows.append(value)
    return rows


def safe_failure(error: BaseException) -> str:
    value = str(error).strip()
    if SAFE_FAILURE.fullmatch(value):
        return value
    name = type(error).__name__
    return name if SAFE_FAILURE.fullmatch(name) else "UNSAFE_FAILURE_REDACTED"


def bound_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


@contextmanager
def offline_guard() -> Iterator[dict[str, int]]:
    """Reject socket construction during evaluator load/forward."""

    counters = {"network_attempts": 0}
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        counters["network_attempts"] += 1
        raise RuntimeError("network_access_forbidden")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield counters
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def _environment_manifest(*, include_cv2: bool) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "schema_version": "frugalface3d.w5b49n.v14.postfreeze_environment.v1",
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve(strict=True)),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "runtime_device": "cpu",
        "cuda_calls": 0,
        "network_allowed": False,
        "training_allowed": False,
        "fallback_allowed": False,
    }
    try:
        from PIL import __version__ as pillow_version

        result["pillow_version"] = str(pillow_version)
    except Exception:
        result["pillow_version"] = None
    if include_cv2:
        import cv2

        cv2_path = _cv2_binary_path(cv2)
        result.update(
            {
                "opencv_version": str(cv2.__version__),
                "cv2_binary_basename": cv2_path.name,
                "cv2_binary_sha256": sha256_file(cv2_path),
            }
        )
    return result


def _cv2_binary_path(cv2: Any) -> Path:
    """Resolve the loaded OpenCV extension, never the package ``__init__``."""

    module_path = Path(str(cv2.__file__)).resolve(strict=True)
    if module_path.suffix in {".so", ".dylib", ".pyd"}:
        return module_path
    candidates = sorted(
        {
            *module_path.parent.glob("cv2*.so"),
            *module_path.parent.glob("cv2*.dylib"),
            *module_path.parent.glob("cv2*.pyd"),
        }
    )
    require(len(candidates) == 1, "sface_cv2_binary_not_unique")
    return candidates[0].resolve(strict=True)


def _formal_linux_environment(environment: Mapping[str, Any]) -> None:
    require(str(environment.get("system")).lower() == "linux", "postfreeze_linux_required")
    require(environment.get("machine") == "x86_64", "postfreeze_x86_64_required")
    require(environment.get("python_version") == "3.10.20", "postfreeze_python_version")
    require(environment.get("torch_version") == "2.4.0+cu121", "postfreeze_torch_version")
    require(environment.get("numpy_version") == "1.23.1", "postfreeze_numpy_version")
    if "opencv_version" in environment:
        require(
            environment.get("opencv_version") == SFACE_FORMAL_OPENCV_VERSION,
            "postfreeze_opencv_version",
        )
    require(environment.get("pillow_version") == "10.4.0", "postfreeze_pillow_version")


def _resolve_assets(root: Path, specification: Mapping[str, tuple[str, str]]) -> dict[str, Path]:
    base = root.expanduser().resolve(strict=True)
    require(base.is_dir() and not base.is_symlink(), "asset_root_plain_directory_required")
    result: dict[str, Path] = {}
    for role, (relative_text, expected_sha) in specification.items():
        relative = Path(relative_text)
        require(not relative.is_absolute() and ".." not in relative.parts, f"asset_relative:{role}")
        candidate = base
        for part in relative.parts:
            candidate = candidate / part
            require(not candidate.is_symlink(), f"asset_symlink:{role}")
        path = candidate.resolve(strict=True)
        path.relative_to(base)
        require(path.is_file() and sha256_file(path) == expected_sha, f"asset_sha256:{role}")
        result[role] = path
    return result


def _lpips_probe_results(model: Any) -> list[dict[str, Any]]:
    import torch

    zeros = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device="cpu")
    sequence = torch.arange(3 * 128 * 128, dtype=torch.float32, device="cpu").reshape(
        1, 3, 128, 128
    )
    pattern = sequence.remainder(256.0).div(127.5).sub(1.0)
    probes = (
        ("IDENTICAL_ZERO", zeros, zeros.clone()),
        ("DETERMINISTIC_HORIZONTAL_FLIP", pattern, torch.flip(pattern, dims=(3,))),
        ("CONSTANT_NEGATIVE_ONE_TO_POSITIVE_ONE", zeros.sub(1.0), zeros.add(1.0)),
    )
    rows: list[dict[str, Any]] = []
    for probe_id, left, right in probes:
        values: list[float] = []
        with torch.inference_mode():
            for _ in range(3):
                output = model(left, right)
                require(isinstance(output, torch.Tensor) and output.numel() == 1, f"lpips_probe_scalar:{probe_id}")
                value = float(output.detach().cpu().item())
                require(math.isfinite(value) and value >= 0.0, f"lpips_probe_value:{probe_id}")
                values.append(value)
        require(len({value.hex() for value in values}) == 1, f"lpips_probe_repeat:{probe_id}")
        require((probe_id == "IDENTICAL_ZERO" and values[0] == 0.0) or (probe_id != "IDENTICAL_ZERO" and values[0] > 0.0), f"lpips_probe_relation:{probe_id}")
        rows.append(
            {
                "probe_id": probe_id,
                "value_float64_hex": values[0].hex(),
                "repeat_count": 3,
                "repeat_bit_exact": True,
            }
        )
    return rows


def _load_lpips_and_probe(asset_root: Path, *, formal: bool) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    assets = _resolve_assets(asset_root, LPIPS_FILES)
    environment = _environment_manifest(include_cv2=False)
    if formal:
        _formal_linux_environment(environment)
    source_receipt = read_json(assets["source_receipt"], "lpips_source_receipt")
    manifest = read_json(assets["manifest"], "lpips_manifest")
    export = read_json(assets["export"], "lpips_export")
    require(source_receipt.get("status") == "PASS_DIRECT_OFFLINE_LPIPS_CPU_EXPORT_AND_RELOAD_PROBES", "lpips_source_receipt_status")
    require(source_receipt.get("artifact_sha256") == LPIPS_FILES["artifact"][1], "lpips_source_artifact_binding")
    require(manifest.get("artifact", {}).get("sha256") == LPIPS_FILES["artifact"][1], "lpips_manifest_artifact_binding")
    require(export.get("artifact_sha256") == LPIPS_FILES["artifact"][1], "lpips_export_artifact_binding")
    require(export.get("model_export_performed") is False and export.get("artifact_reused_exact_bytes") is True, "lpips_no_reexport_boundary")
    with offline_guard() as network:
        model = torch.jit.load(str(assets["artifact"]), map_location="cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            require(parameter.device.type == "cpu", "lpips_parameter_cpu")
        require(all(buffer.device.type == "cpu" for buffer in model.buffers()), "lpips_buffer_cpu")
        operators = tuple(sorted(torch.jit.export_opnames(model)))
        require(operators == LPIPS_EXPECTED_OPERATORS, "lpips_operator_graph")
        observed = _lpips_probe_results(model)
    require(network["network_attempts"] == 0, "lpips_network_attempt")
    source = source_receipt.get("probe_results")
    require(isinstance(source, list) and len(source) == 3, "lpips_source_probe_count")
    deltas: list[dict[str, Any]] = []
    for observed_row, source_row in zip(observed, source, strict=True):
        require(observed_row["probe_id"] == source_row.get("probe_id"), "lpips_probe_id_binding")
        source_value = float.fromhex(str(source_row.get("value_float64_hex")))
        observed_value = float.fromhex(str(observed_row["value_float64_hex"]))
        delta = abs(observed_value - source_value)
        require(delta <= QUALIFICATION_TOLERANCE_ABS, f"lpips_probe_tolerance:{observed_row['probe_id']}")
        deltas.append(
            {
                "probe_id": observed_row["probe_id"],
                "source_float64_hex": source_value.hex(),
                "observed_float64_hex": observed_value.hex(),
                "absolute_delta": delta,
                "within_predeclared_tolerance": True,
            }
        )
    probe_manifest = {
        "schema_version": "frugalface3d.w5b49n.v14.lpips_probe_manifest.v1",
        "probe_status": "PASS",
        "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
        "repeat_policy": "three_calls_bit_exact_on_current_host",
        "source_expectation_rewritten": False,
        "operators": list(operators),
        "observed_probes": observed,
        "source_comparison": deltas,
        "maximum_absolute_delta": max(row["absolute_delta"] for row in deltas),
        "real_image_reads": 0,
        "metric_rows": 0,
    }
    return model, environment, probe_manifest


def _sface_probe(detector: Any, recognizer: Any, cv2: Any) -> dict[str, Any]:
    height = width = 320
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        (
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=2,
    )
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    detector.setInputSize((width, height))

    def detection_digest() -> tuple[int, str]:
        output = detector.detect(bgr)
        faces = output[1] if isinstance(output, tuple) and len(output) >= 2 else output
        if faces is None:
            return 0, hashlib.sha256(b"NONE").hexdigest()
        array = np.ascontiguousarray(faces, dtype=np.float32)
        require(bool(np.isfinite(array).all()), "sface_probe_detector_nonfinite")
        return int(array.shape[0]), canonical_array_sha256(array)

    detection_rows = [detection_digest(), detection_digest()]
    require(detection_rows[0] == detection_rows[1], "sface_probe_detector_repeat")
    face_row = np.asarray(
        [
            60.0, 40.0, 200.0, 240.0,
            110.0, 120.0, 210.0, 120.0, 160.0, 170.0,
            120.0, 220.0, 200.0, 220.0, 0.99,
        ],
        dtype=np.float32,
    )

    def feature_digest() -> tuple[str, int]:
        aligned = recognizer.alignCrop(bgr, face_row)
        feature = np.ascontiguousarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        require(feature.size == 128 and bool(np.isfinite(feature).all()), "sface_probe_feature_contract")
        return canonical_array_sha256(feature), int(feature.size)

    feature_rows = [feature_digest(), feature_digest()]
    require(feature_rows[0] == feature_rows[1], "sface_probe_feature_repeat")
    return {
        "schema_version": "frugalface3d.w5b49n.v14.sface_probe_manifest.v1",
        "probe_status": "PASS",
        "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
        "synthetic_detector_face_count": detection_rows[0][0],
        "synthetic_detector_output_sha256": detection_rows[0][1],
        "synthetic_recognizer_feature_dimension": feature_rows[0][1],
        "synthetic_recognizer_feature_sha256": feature_rows[0][0],
        "repeat_bit_exact": True,
        "embedding_persisted": False,
        "real_image_reads": 0,
        "metric_rows": 0,
    }


def _load_sface_and_probe(asset_root: Path, *, formal: bool) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], str]:
    import cv2

    assets = _resolve_assets(asset_root, SFACE_FILES)
    environment = _environment_manifest(include_cv2=True)
    if formal:
        _formal_linux_environment(environment)
    require(int(cv2.dnn.DNN_BACKEND_OPENCV) == 3 and int(cv2.dnn.DNN_TARGET_CPU) == 0, "sface_dnn_backend")
    cv2.setNumThreads(1)
    require(int(cv2.getNumThreads()) == 1, "sface_single_thread")
    cv2.ocl.setUseOpenCL(False)
    require(cv2.ocl.useOpenCL() is False, "sface_opencl_disabled")
    cv2.setUseOptimized(True)
    require(cv2.useOptimized() is True, "sface_optimized")
    cv2.setRNGSeed(2026080649)
    with offline_guard() as network:
        detector = cv2.FaceDetectorYN_create(
            str(assets["yunet_model"]), "", (320, 320), 0.9, 0.3, 5000, 3, 0
        )
        recognizer = cv2.FaceRecognizerSF_create(str(assets["sface_model"]), "", 3, 0)
        require(detector is not None and recognizer is not None, "sface_factory")
        probe_manifest = _sface_probe(detector, recognizer, cv2)
    require(network["network_attempts"] == 0, "sface_network_attempt")
    runtime_id = json_sha256(
        {
            "environment": environment,
            "assets": {role: expected for role, (_relative, expected) in SFACE_FILES.items()},
            "probe": probe_manifest,
        }
    )
    return detector, recognizer, cv2, environment, probe_manifest, runtime_id


def _qualification_failure(metric_id: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_SCHEMA,
        "status": "METHOD_FAILURE_V14_EVALUATOR_QUALIFICATION",
        "metric_id": metric_id,
        "operating_system": platform.system().lower(),
        "device_backend": "cpu",
        "cuda_calls": 0,
        "fresh_output_root": True,
        "prior_method_failure_reused": False,
        "probe_status": "FAILED",
        "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
        "failure_code": safe_failure(error),
        "real_image_reads": 0,
        "metric_rows": 0,
        "bound_files": [],
    }


def qualify(asset_root: Path, output_root: Path, *, formal: bool = True) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError("qualification_output_exists_no_rerun")
    output.mkdir(parents=True, mode=0o700)
    script_path = Path(__file__).resolve(strict=True)
    results: dict[str, dict[str, Any]] = {}
    try:
        try:
            _model, environment, probes = _load_lpips_and_probe(asset_root, formal=formal)
            environment_path = output / "LPIPS_RUNTIME_MANIFEST.json"
            probe_path = output / "LPIPS_PROBE_MANIFEST.json"
            write_json(environment_path, environment)
            write_json(probe_path, probes)
            assets = _resolve_assets(asset_root, LPIPS_FILES)
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": "PASS_V14_LPIPS_LINUX_QUALIFIED",
                "metric_id": "lpips_alex_v0_1",
                "operating_system": "linux" if formal else platform.system().lower(),
                "device_backend": "cpu",
                "cuda_calls": 0,
                "fresh_output_root": True,
                "prior_method_failure_reused": False,
                "probe_status": "PASS",
                "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
                "qualification_script_sha256": sha256_file(script_path),
                "runtime_manifest_sha256": sha256_file(environment_path),
                "evaluator_export_sha256": sha256_file(assets["export"]),
                "probe_manifest_sha256": sha256_file(probe_path),
                "real_image_reads": 0,
                "metric_rows": 0,
                "network_calls": 0,
                "bound_files": [
                    bound_file(script_path),
                    bound_file(environment_path),
                    bound_file(probe_path),
                    *[bound_file(path) for path in assets.values()],
                ],
            }
        except Exception as error:
            terminal = _qualification_failure("lpips_alex_v0_1", error)
        write_json(output / "LPIPS_LINUX_QUALIFICATION_TERMINAL.json", terminal)
        results["lpips"] = terminal

        try:
            _detector, _recognizer, cv2, environment, probes, runtime_id = _load_sface_and_probe(
                asset_root, formal=formal
            )
            environment_path = output / "SFACE_RUNTIME_MANIFEST.json"
            probe_path = output / "SFACE_PROBE_MANIFEST.json"
            write_json(environment_path, {**environment, "runtime_manifest_id": runtime_id})
            write_json(probe_path, probes)
            assets = _resolve_assets(asset_root, SFACE_FILES)
            cv2_path = _cv2_binary_path(cv2)
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": "PASS_V14_SFACE_LINUX_QUALIFIED",
                "metric_id": "sface_source_to_render_cosine",
                "operating_system": "linux" if formal else platform.system().lower(),
                "device_backend": "cpu",
                "cuda_calls": 0,
                "fresh_output_root": True,
                "prior_method_failure_reused": False,
                "probe_status": "PASS",
                "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
                "qualification_script_sha256": sha256_file(script_path),
                "runtime_manifest_sha256": sha256_file(environment_path),
                "detector_model_sha256": sha256_file(assets["yunet_model"]),
                "recognizer_model_sha256": sha256_file(assets["sface_model"]),
                "probe_manifest_sha256": sha256_file(probe_path),
                "runtime_manifest_id": runtime_id,
                "real_image_reads": 0,
                "metric_rows": 0,
                "network_calls": 0,
                "embedding_persisted": False,
                "bound_files": [
                    bound_file(script_path),
                    bound_file(environment_path),
                    bound_file(probe_path),
                    bound_file(cv2_path),
                    *[bound_file(path) for path in assets.values()],
                ],
            }
        except Exception as error:
            terminal = _qualification_failure("sface_source_to_render_cosine", error)
        write_json(output / "SFACE_LINUX_QUALIFICATION_TERMINAL.json", terminal)
        results["sface"] = terminal

        overall = {
            "schema_version": "frugalface3d.w5b49n.v14.qualification_terminal.v1",
            "status": (
                "PASS_V14_ALL_EVALUATORS_QUALIFIED"
                if all(row["probe_status"] == "PASS" for row in results.values())
                else "TERMINAL_V14_EVALUATOR_QUALIFICATION_WITH_RETAINED_FAILURES"
            ),
            "lpips_terminal_sha256": sha256_file(output / "LPIPS_LINUX_QUALIFICATION_TERMINAL.json"),
            "sface_terminal_sha256": sha256_file(output / "SFACE_LINUX_QUALIFICATION_TERMINAL.json"),
            "real_image_reads": 0,
            "metric_rows": 0,
            "prior_method_failure_reused": False,
        }
        write_json(output / "QUALIFICATION_TERMINAL.json", overall)
        return overall
    except Exception as error:
        write_json(
            output / "QUALIFICATION_FAILURE.json",
            {
                "status": "FAILED_QUALIFICATION_ROOT_RETAINED_NO_RETRY",
                "failure_code": safe_failure(error),
                "real_image_reads": 0,
                "metric_rows": 0,
            },
        )
        raise


def _load_qualified_evaluators(
    qualification_root: Path, asset_root: Path, *, formal: bool
) -> QualifiedEvaluators:
    root = qualification_root.expanduser().resolve(strict=True)
    lpips_terminal_path = root / "LPIPS_LINUX_QUALIFICATION_TERMINAL.json"
    sface_terminal_path = root / "SFACE_LINUX_QUALIFICATION_TERMINAL.json"
    lpips_terminal = read_json(lpips_terminal_path, "lpips_qualification_terminal")
    sface_terminal = read_json(sface_terminal_path, "sface_qualification_terminal")
    qualification_core = {
        "schema_version": TERMINAL_SCHEMA,
        "operating_system": "linux" if formal else platform.system().lower(),
        "device_backend": "cpu",
        "cuda_calls": 0,
        "fresh_output_root": True,
        "prior_method_failure_reused": False,
        "probe_status": "PASS",
        "probe_tolerance_abs": QUALIFICATION_TOLERANCE_ABS,
    }
    for key, expected in qualification_core.items():
        require(lpips_terminal.get(key) == expected, f"lpips_qualification_field:{key}")
        require(sface_terminal.get(key) == expected, f"sface_qualification_field:{key}")
    require(
        lpips_terminal.get("status") == "PASS_V14_LPIPS_LINUX_QUALIFIED"
        and lpips_terminal.get("metric_id") == "lpips_alex_v0_1",
        "lpips_current_host_qualification_required",
    )
    require(
        sface_terminal.get("status") == "PASS_V14_SFACE_LINUX_QUALIFIED"
        and sface_terminal.get("metric_id") == "sface_source_to_render_cosine",
        "sface_current_host_qualification_required",
    )
    lpips = None
    detector = recognizer = cv2 = None
    runtime_id = None
    if lpips_terminal.get("status") == "PASS_V14_LPIPS_LINUX_QUALIFIED":
        lpips, environment, probes = _load_lpips_and_probe(asset_root, formal=formal)
        require(sha256_file(root / "LPIPS_RUNTIME_MANIFEST.json") == lpips_terminal.get("runtime_manifest_sha256"), "lpips_qualification_runtime_hash")
        require(sha256_file(root / "LPIPS_PROBE_MANIFEST.json") == lpips_terminal.get("probe_manifest_sha256"), "lpips_qualification_probe_hash")
        require(read_json(root / "LPIPS_RUNTIME_MANIFEST.json", "lpips_runtime_manifest") == environment, "lpips_runtime_replay")
        require(read_json(root / "LPIPS_PROBE_MANIFEST.json", "lpips_probe_manifest") == probes, "lpips_probe_replay")
    if sface_terminal.get("status") == "PASS_V14_SFACE_LINUX_QUALIFIED":
        detector, recognizer, cv2, environment, probes, runtime_id = _load_sface_and_probe(
            asset_root, formal=formal
        )
        expected_environment = read_json(root / "SFACE_RUNTIME_MANIFEST.json", "sface_runtime_manifest")
        require(expected_environment == {**environment, "runtime_manifest_id": runtime_id}, "sface_runtime_replay")
        require(read_json(root / "SFACE_PROBE_MANIFEST.json", "sface_probe_manifest") == probes, "sface_probe_replay")
    return QualifiedEvaluators(
        lpips=lpips,
        lpips_terminal={**lpips_terminal, "_terminal_path": str(lpips_terminal_path)},
        sface_detector=detector,
        sface_recognizer=recognizer,
        sface_cv2=cv2,
        sface_runtime_id=runtime_id,
        sface_terminal={**sface_terminal, "_terminal_path": str(sface_terminal_path)},
    )


def _unit_uv(value: Any, role: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    require(array.dtype == np.dtype(np.float32), f"uv_float32:{role}")
    require(array.shape == (3, 64, 64), f"uv_shape:{role}")
    require(bool(np.isfinite(array).all()), f"uv_finite:{role}")
    require(float(array.min()) >= 0.0 and float(array.max()) <= 1.0, f"uv_range:{role}")
    return array


def _uv_batch(value: Any, expected: int, role: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    require(array.dtype == np.dtype(np.float32), f"uv_batch_float32:{role}")
    require(array.shape == (expected, 3, 64, 64), f"uv_batch_shape:{role}")
    require(bool(np.isfinite(array).all()), f"uv_batch_finite:{role}")
    require(float(array.min()) >= 0.0 and float(array.max()) <= 1.0, f"uv_batch_range:{role}")
    return array


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, Mapping), f"torch_mapping:{path.name}")
    return value


def _dataset_sample_id(row: Mapping[str, Any], dataset_id: str) -> str:
    if dataset_id == "D1":
        return (
            f"D1:{row['identity_token']}:{row['expression_token']}:{row['view_token']}"
        )
    return f"D2:{row['identity_token']}:{row['view_token']}"


def _load_training_authority(path: Path) -> tuple[dict[tuple[str, int], str], Path]:
    terminal_path = path.expanduser().resolve(strict=True)
    terminal = read_json(terminal_path, "v14_training_terminal")
    require(
        terminal.get("schema_version")
        == "frugalface3d.w5b49n.v14_matched_training_terminal.v1"
        and terminal.get("status") == "MATCHED_CONTROL_TRAINING_COMPLETE"
        and terminal.get("device") == "cuda"
        and terminal.get("development_only") is False
        and terminal.get("method_seed_route_count") == 15
        and terminal.get("new_training_units") == 15
        and terminal.get("optimizer_steps") == 7680
        and terminal.get("historical_mps_full_units_used_for_confirmation") == 0
        and terminal.get("automatic_retry") is False,
        "v14_training_terminal_contract",
    )
    rows = terminal.get("rows")
    require(isinstance(rows, list) and len(rows) == 15, "v14_training_rows")
    method_map = {"Full": "full", "Condition0": "condition0", "B-lite-FT": "b_lite_ft"}
    authority: dict[tuple[str, int], str] = {}
    for row in rows:
        method = method_map.get(str(row.get("method")))
        seed = row.get("seed")
        checkpoint_sha = row.get("checkpoint_sha256")
        require(
            method in SEEDED_METHODS
            and type(seed) is int
            and seed in SEEDS
            and isinstance(checkpoint_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is not None,
            "v14_training_row_contract",
        )
        key = (method, seed)
        require(key not in authority, "v14_training_row_duplicate")
        authority[key] = checkpoint_sha
    require(
        set(authority)
        == {(method, seed) for method in SEEDED_METHODS for seed in SEEDS},
        "v14_training_matrix",
    )
    return authority, terminal_path


def _load_v14_routes(
    raw_root: Path,
    dataset_id: str,
    expected: int,
    training_authority: Mapping[tuple[str, int], str],
    training_terminal_sha256: str,
) -> list[EndpointRoute]:
    root = raw_root.expanduser().resolve(strict=True)
    terminal_path = root / "INFERENCE_TERMINAL.json"
    terminal = read_json(terminal_path, f"{dataset_id}_v14_inference_terminal")
    expected_dataset_name = {"D1": "facescape", "D2": "realy"}[dataset_id]
    require(
        terminal.get("schema_version")
        == "frugalface3d.w5b49n.v14_matched_inference_terminal.v1"
        and terminal.get("status") == "MATCHED_CONTROL_INFERENCE_COMPLETE"
        and terminal.get("dataset") == expected_dataset_name
        and terminal.get("device") == "cuda"
        and terminal.get("development_only") is False
        and terminal.get("training_terminal_sha256") == training_terminal_sha256,
        "v14_inference_status",
    )
    require(terminal.get("source_sample_count") == expected and terminal.get("method_seed_route_count") == 15, "v14_inference_counts")
    require(terminal.get("target_pair_reads") == 0 and terminal.get("automatic_retry") is False, "v14_inference_isolation")
    rows = terminal.get("routes")
    require(isinstance(rows, list) and len(rows) == 15, "v14_route_rows")
    method_map = {"Full": "full", "Condition0": "condition0", "B-lite-FT": "b_lite_ft"}
    routes: list[EndpointRoute] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        method = method_map.get(str(row.get("method")))
        seed = row.get("seed")
        require(method in SEEDED_METHODS and type(seed) is int and seed in SEEDS, "v14_route_method_seed")
        require((method, seed) not in seen, "v14_route_duplicate")
        require(
            row.get("schema_version")
            == "frugalface3d.w5b49n.v14_matched_raw_route.v1"
            and row.get("status") == "SUCCESS"
            and row.get("dataset") == expected_dataset_name
            and row.get("sample_count") == expected
            and row.get("source_observed_uv_exact") is True
            and row.get("hidden_native_equals_conserved") is True
            and row.get("target_pair_reads") == 0
            and row.get("checkpoint_sha256")
            == training_authority[(method, seed)],
            "v14_route_contract",
        )
        seen.add((method, seed))
        relative = Path(str(row.get("raw_output_path")))
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        require(sha256_file(path) == row.get("raw_output_sha256"), "v14_raw_sha256")
        route_terminal = read_json(path.parent / "ROUTE_TERMINAL.json", "v14_route_terminal")
        require(route_terminal == row, "v14_route_terminal_replay")
        payload = _torch_load(path)
        require(set(payload) == {"native", "conserved"}, "v14_raw_keyspace")
        values = _uv_batch(payload["conserved"], expected, f"{method}:{seed}")
        routes.append(
            EndpointRoute(
                route_id=f"{method}__{seed}",
                method_id=method,
                seed=seed,
                output_mode="conserved",
                values=values,
                origin="v14_cuda_matched_control",
                bound_sha256=sha256_file(path),
            )
        )
    require(seen == {(method, seed) for method in SEEDED_METHODS for seed in SEEDS}, "v14_route_matrix")
    return routes


def _load_frozen_b_lite(route_root: Path, expected: int) -> EndpointRoute:
    root = route_root.expanduser().resolve(strict=True)
    terminal = read_json(root / "ROUTE_TERMINAL.json", "frozen_b_lite_terminal")
    require(terminal.get("status") == "SUCCESS" and terminal.get("method_id") == "b_lite", "b_lite_terminal")
    count = terminal.get("sample_count", terminal.get("source_sample_count"))
    require(count == expected and terminal.get("native_and_conserved_same_forward") is True, "b_lite_count_semantics")
    path = root / str(terminal.get("raw_output_file"))
    require(path.name == "RAW_OUTPUTS.pt" and sha256_file(path) == terminal.get("raw_output_sha256"), "b_lite_raw_sha")
    payload = _torch_load(path)
    require(set(payload) == {"native", "conserved"}, "b_lite_raw_keyspace")
    values = _uv_batch(payload["conserved"], expected, "b_lite_conserved")
    return EndpointRoute(
        route_id="b_lite",
        method_id="b_lite",
        seed=None,
        output_mode="conserved",
        values=values,
        origin="frozen_historical_cuda_inference",
        bound_sha256=sha256_file(path),
    )


def _public_sample_root(batch_root: Path, index: int, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return batch_root / "samples" / f"sample-{index:06d}-{digest}"


def _load_public_batch(
    batch_root: Path, cache: Any, dataset_id: str, expected: int
) -> EndpointRoute:
    root = batch_root.expanduser().resolve(strict=True)
    batch_terminal_path = root / "BATCH_TERMINAL.json"
    batch = read_json(batch_terminal_path, f"{dataset_id}_public_batch")
    method_id = str(batch.get("method_id"))
    method_map = {"lama_big": "lama", "zits": "zits"}
    require(method_id in method_map, "public_method_not_lama_or_zits")
    require(
        batch.get("schema_version")
        == "frugalface3d.w5b49n.public_baseline_method_batch_terminal.v1"
        and batch.get("status") == "TERMINAL_COMPLETE"
        and batch.get("expected_sample_count") == expected
        and batch.get("terminal_sample_count") == expected
        and batch.get("sample_attempt") == 1
        and batch.get("automatic_retry") is False
        and batch.get("target_texture_consumed") is False,
        "public_batch_contract",
    )
    values: list[np.ndarray] = []
    artifact_hashes: list[str] = []
    for offset, row in enumerate(cache.rows):
        sequence = offset + 1
        sample_id = _dataset_sample_id(row, dataset_id)
        sample_root = _public_sample_root(root, sequence, sample_id)
        terminal = read_json(sample_root / "TERMINAL.json", "public_sample_terminal")
        require(
            terminal.get("schema_version")
            == "frugalface3d.w5b49n.public_baseline_sample_terminal.v1"
            and terminal.get("method_id") == method_id
            and terminal.get("sample_id") == sample_id
            and terminal.get("state") == "SUCCESS"
            and terminal.get("attempt") == 1
            and terminal.get("automatic_retry") is False
            and terminal.get("target_texture_consumed") is False,
            "public_sample_contract",
        )
        artifact = sample_root / "NATIVE_AND_CONSERVED.npz"
        require(sha256_file(artifact) == terminal.get("artifact_sha256"), "public_sample_artifact_sha")
        with np.load(artifact, allow_pickle=False) as payload:
            require(set(payload.files) == {"native_rgb", "conserved_rgb"}, "public_npz_keyspace")
            value = np.ascontiguousarray(payload["conserved_rgb"])
        require(value.shape == (1, 3, 64, 64), "public_sample_shape")
        values.append(_unit_uv(value[0], f"{method_id}:{offset}"))
        artifact_hashes.append(sha256_file(artifact))
    batch_values = np.ascontiguousarray(np.stack(values, axis=0), dtype=np.float32)
    return EndpointRoute(
        route_id=method_map[method_id],
        method_id=method_map[method_id],
        seed=None,
        output_mode="conserved",
        values=batch_values,
        origin="frozen_official_public_baseline_batch",
        bound_sha256=json_sha256(
            {
                "batch_terminal": sha256_file(batch_terminal_path),
                "artifacts": artifact_hashes,
            }
        ),
    )


def _freeuv_base(root: Path) -> Path:
    resolved = root.expanduser().resolve(strict=True)
    candidates = (
        resolved,
        resolved / "W5B49N_FREEUV_D1D2_20260820V12",
    )
    for candidate in candidates:
        if (candidate / "activity/ACTIVITY_TERMINAL.json").is_file() and (
            candidate / "shared_render/SHARED_RENDER_TERMINAL.json"
        ).is_file():
            return candidate.resolve(strict=True)
    raise PostprocessError("freeuv_v1_2_root_not_found")


def _load_freeuv(
    root: Path,
    archive: Path,
    safe_sample_map: Path,
    expected_counts: Mapping[str, int],
) -> tuple[
    dict[str, list[EndpointRoute]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Path],
]:
    archive_path = archive.expanduser().resolve(strict=True)
    require(sha256_file(archive_path) == FREEUV_ARCHIVE_SHA256, "freeuv_archive_sha256")
    safe_sample_map_path = safe_sample_map.expanduser().resolve(strict=True)
    require(
        safe_sample_map_path.is_file()
        and not safe_sample_map_path.is_symlink()
        and sha256_file(safe_sample_map_path) == FREEUV_SAFE_SAMPLE_MAP_SHA256,
        "freeuv_safe_sample_map_sha256",
    )
    base = _freeuv_base(root)
    activity_path = base / "activity/ACTIVITY_TERMINAL.json"
    shared_path = base / "shared_render/SHARED_RENDER_TERMINAL.json"
    activity = read_json(activity_path, "freeuv_activity_terminal")
    shared = read_json(shared_path, "freeuv_shared_render_terminal")
    require(
        activity.get("status") == "PASS_F2_FREEUV_D1D2_RAW_OUTPUTS_FROZEN"
        and activity.get("successful_total_forward_count") == 560
        and activity.get("automatic_retry") is False
        and activity.get("pair_roster_reads") == 0
        and activity.get("target_reads") == 0
        and activity.get("fallback_calls") == 0
        and activity.get("sample_map_sha256") == FREEUV_SAFE_SAMPLE_MAP_SHA256,
        "freeuv_activity_contract",
    )
    require(
        shared.get("status") == "PASS_SHARED_TARGET_AND_FREEUV_ENDPOINT_RENDERS_MATERIALIZED_ONCE"
        and shared.get("pair_count") == 1360
        and shared.get("render_call_count") == 2720
        and shared.get("endpoint_count") == 2
        and shared.get("geometry_recomputed") is False
        and shared.get("automatic_retry") is False
        and shared.get("target_frame_materialization_count") == 560,
        "freeuv_shared_render_contract",
    )
    routes: dict[str, list[EndpointRoute]] = {}
    aggregates = {
        "D1": (base / "activity/D1/FREEUV_D1_COMMON64_OUTPUTS.npz", FREEUV_D1_SHA256),
        "D2": (base / "activity/D2/FREEUV_D2_COMMON64_OUTPUTS.npz", FREEUV_D2_SHA256),
    }
    for dataset_id, (path, expected_sha) in aggregates.items():
        require(sha256_file(path) == expected_sha, f"freeuv_{dataset_id}_aggregate_sha")
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == {"native", "conserved"}, "freeuv_npz_keyspace")
            native = _uv_batch(payload["native"], expected_counts[dataset_id], f"freeuv_{dataset_id}_native")
            conserved = _uv_batch(payload["conserved"], expected_counts[dataset_id], f"freeuv_{dataset_id}_conserved")
        routes[dataset_id] = [
            EndpointRoute("freeuv_native", "freeuv_native", None, "native", native, "freeuv_v1_2_frozen", expected_sha),
            EndpointRoute("freeuv_conserved", "freeuv_conserved", None, "conserved", conserved, "freeuv_v1_2_frozen", expected_sha),
        ]
    target_manifest_path = base / "shared_render/TARGET_FRAME_MANIFEST.jsonl"
    render_manifest_path = base / "shared_render/RENDER_MANIFEST.jsonl"
    require(sha256_file(target_manifest_path) == FREEUV_TARGET_MANIFEST_SHA256, "freeuv_target_manifest_sha")
    require(sha256_file(render_manifest_path) == FREEUV_RENDER_MANIFEST_SHA256, "freeuv_render_manifest_sha")
    target_rows = read_jsonl(target_manifest_path, "freeuv_target_manifest")
    render_rows = read_jsonl(render_manifest_path, "freeuv_render_manifest")
    require(len(target_rows) == 560 and len(render_rows) == 2720, "freeuv_manifest_counts")
    target_index: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        runtime_id = str(row.get("target_runtime_id"))
        require(runtime_id not in target_index and row.get("materialization_count") == 1, "freeuv_target_duplicate")
        target_index[runtime_id] = row
    render_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in render_rows:
        key = (str(row.get("dataset_id")), str(row.get("pair_runtime_id")), str(row.get("route_id")))
        require(key not in render_index and row.get("render_materialization_count") == 1, "freeuv_render_duplicate")
        require(row.get("geometry_recomputed") is False and row.get("retry_count") == 0, "freeuv_render_semantics")
        render_index[key] = row
    paths = {
        "archive": archive_path,
        "safe_sample_map": safe_sample_map_path,
        "activity_terminal": activity_path,
        "shared_terminal": shared_path,
        "target_manifest": target_manifest_path,
        "render_manifest": render_manifest_path,
        "base": base,
    }
    return routes, render_rows, target_index, render_index, paths


def _pairs_from_freeuv(
    render_rows: Sequence[Mapping[str, Any]], caches: Mapping[str, Any], d2_roster_root: Path
) -> dict[str, list[PairRow]]:
    conserved = [row for row in render_rows if row.get("route_id") == "freeuv__conserved"]
    require(len(conserved) == 1360, "freeuv_conserved_pair_count")
    by_dataset: dict[str, list[PairRow]] = {"D1": [], "D2": []}
    for row in conserved:
        dataset_id = str(row.get("dataset_id"))
        require(dataset_id in by_dataset, "freeuv_pair_dataset")
        source_runtime = str(row.get("source_runtime_id"))
        target_runtime = str(row.get("target_runtime_id"))
        prefix = dataset_id + "-S"
        require(source_runtime.startswith(prefix) and target_runtime.startswith(prefix), "freeuv_pair_runtime_id")
        source_index = int(source_runtime[len(prefix) :])
        target_index = int(target_runtime[len(prefix) :])
        cache = caches[dataset_id]
        require(0 <= source_index < len(cache.rows) and 0 <= target_index < len(cache.rows), "freeuv_pair_index")
        source_row = cache.rows[source_index]
        target_row = cache.rows[target_index]
        require(source_row["identity_token"] == target_row["identity_token"], "pair_cross_identity")
        manifest_identity = row.get("anonymous_identity_token")
        require(
            isinstance(manifest_identity, str)
            and manifest_identity == source_row["identity_token"],
            "pair_manifest_identity_mismatch",
        )
        if dataset_id == "D1":
            require(source_row["expression_token"] == target_row["expression_token"], "d1_pair_expression")
            require(source_row["view_token"] != target_row["view_token"], "d1_pair_view")
        else:
            require(source_row["view_token"] != target_row["view_token"], "d2_pair_view")
        by_dataset[dataset_id].append(
            PairRow(
                dataset_id=dataset_id,
                pair_index=int(row.get("pair_index")),
                pair_id=str(row.get("pair_runtime_id")),
                identity_token=manifest_identity,
                source_index=source_index,
                target_index=target_index,
                source_runtime_id=source_runtime,
                target_runtime_id=target_runtime,
                target_frame_manifest_id=str(row.get("target_frame_manifest_id")),
            )
        )
    for dataset_id, expected in (("D1", 160), ("D2", 1200)):
        rows = sorted(by_dataset[dataset_id], key=lambda item: item.pair_index)
        require(len(rows) == expected and len({row.pair_id for row in rows}) == expected, f"{dataset_id}_pair_count")
        require([row.pair_index for row in rows] == list(range(expected)), f"{dataset_id}_pair_order")
        by_dataset[dataset_id] = rows

    from reproducibility.w5b49n_mechanism_closure_v1.metrics.realy_final_metric_v1 import (
        _load_pairs as load_realy_pairs,
    )

    authority, _authority_sha = load_realy_pairs(d2_roster_root, caches["D2"])
    authority_keys = {
        (
            str(row["identity_token"]),
            int(row["source_index"]),
            int(row["target_index"]),
        )
        for row in authority
    }
    observed_keys = {
        (row.identity_token, row.source_index, row.target_index)
        for row in by_dataset["D2"]
    }
    require(observed_keys == authority_keys, "d2_freeuv_pair_authority_mismatch")
    return by_dataset


def _mask(value: Any, role: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape == (1, 64, 64):
        array = array[0]
    require(array.shape == (64, 64) and bool(np.isfinite(array).all()), f"mask_shape_finite:{role}")
    require(bool(np.all((array == 0) | (array == 1))), f"mask_binary:{role}")
    return np.ascontiguousarray(array.astype(bool, copy=False))


def uv_supports(cache: Any, pair: PairRow) -> tuple[np.ndarray, np.ndarray]:
    source_visible = _mask(cache.tensors["visibility"][pair.source_index], "source_visible")
    target_visible = _mask(cache.tensors["visibility"][pair.target_index], "target_visible")
    canonical = _mask(cache.tensors["canonical_mask"][pair.source_index], "canonical")
    target_canonical = _mask(cache.tensors["canonical_mask"][pair.target_index], "target_canonical")
    require(np.array_equal(canonical, target_canonical), "canonical_mask_pair_changed")
    hidden = canonical & ~source_visible & target_visible
    all_target = canonical & target_visible
    require(bool(all_target.any()), "target_visible_support_empty")
    return np.ascontiguousarray(hidden), np.ascontiguousarray(all_target)


def masked_rgb_mae(prediction: np.ndarray, reference: np.ndarray, support: np.ndarray) -> float | None:
    pred = _unit_uv(prediction, "mae_prediction")
    ref = _unit_uv(reference, "mae_reference")
    mask = np.ascontiguousarray(support, dtype=bool)
    require(mask.shape == (64, 64), "mae_support_shape")
    if not bool(mask.any()):
        return None
    value = float(np.abs(pred.astype(np.float64) - ref.astype(np.float64))[:, mask].mean())
    require(math.isfinite(value) and 0.0 <= value <= 1.0, "mae_value")
    return value


def _canonical_screen_support(
    screen_uv: np.ndarray, screen_visibility: np.ndarray, canonical_mask: np.ndarray
) -> np.ndarray:
    uv = np.ascontiguousarray(screen_uv, dtype=np.float32)
    visible = np.ascontiguousarray(screen_visibility, dtype=bool)
    canonical = np.ascontiguousarray(canonical_mask, dtype=bool)
    require(uv.shape == (224, 224, 2) and visible.shape == (224, 224), "screen_context_shape")
    require(np.array_equal(np.isfinite(uv).all(axis=2), visible), "screen_uv_visibility")
    visible_uv = uv[visible]
    require(visible_uv.size > 0 and float(visible_uv.min()) >= 0.0 and float(visible_uv.max()) <= 1.0, "screen_uv_range")
    x = visible_uv[:, 0] * np.float32(63.0)
    y = (np.float32(1.0) - visible_uv[:, 1]) * np.float32(63.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, 63)
    y1 = np.minimum(y0 + 1, 63)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    values = canonical.astype(np.float32)
    score = (
        (1.0 - wx) * (1.0 - wy) * values[y0, x0]
        + wx * (1.0 - wy) * values[y0, x1]
        + (1.0 - wx) * wy * values[y1, x0]
        + wx * wy * values[y1, x1]
    )
    supported = np.zeros(visible.shape, dtype=bool)
    supported[visible] = np.isclose(score, np.float32(1.0), rtol=0.0, atol=1.0e-6)
    result = np.ascontiguousarray(visible & supported)
    require(bool(result.any()), "screen_common_mask_empty")
    return result


def _screen_context(cache: Any, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = cache.tensors["screen_rgb_uint8"][index]
    uv = cache.tensors["screen_uv_float32"][index]
    visibility = cache.tensors["screen_visibility_bool"][index, 0]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
        uv = uv.detach().cpu().numpy()
        visibility = visibility.detach().cpu().numpy()
    rgb_hwc = np.ascontiguousarray(np.asarray(rgb).transpose(1, 2, 0))
    uv_hwc = np.ascontiguousarray(uv, dtype=np.float32)
    visibility_hw = np.ascontiguousarray(visibility, dtype=bool)
    canonical = _mask(cache.tensors["canonical_mask"][index], "screen_canonical")
    common = _canonical_screen_support(uv_hwc, visibility_hw, canonical)
    require(rgb_hwc.shape == (224, 224, 3) and rgb_hwc.dtype == np.uint8, "screen_rgb_contract")
    return rgb_hwc, uv_hwc, common, visibility_hw


def _load_png(path: Path, expected_sha: str, role: str) -> np.ndarray:
    from PIL import Image

    require(path.is_file() and not path.is_symlink(), f"png_plain:{role}")
    require(sha256_file(path) == expected_sha, f"png_sha256:{role}")
    with Image.open(path) as image:
        image.load()
        require(image.mode == "RGB", f"png_rgb:{role}")
        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    require(array.shape == (224, 224, 3), f"png_shape:{role}")
    return array


def _write_png(path: Path, value: np.ndarray) -> str:
    from PIL import Image

    array = np.ascontiguousarray(value)
    require(array.dtype == np.uint8 and array.shape == (224, 224, 3), "write_png_contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array, mode="RGB").save(
        temporary, format="PNG", optimize=False, compress_level=9
    )
    temporary.replace(path)
    return sha256_file(path)


def _render_candidate(candidate: np.ndarray, screen_uv: np.ndarray, common_mask: np.ndarray) -> np.ndarray:
    import torch
    from frugalface3d.evaluation.mps_uv_pipeline import render_composite_torch

    completed = torch.from_numpy(_unit_uv(candidate, "render_candidate"))[None]
    neutral = np.full((224, 224, 3), NEUTRAL_BACKGROUND_RGB_UINT8, dtype=np.uint8)
    with torch.inference_mode():
        render, actual = render_composite_torch(
            completed, screen_uv, common_mask, neutral, torch.device("cpu")
        )
    require(actual is not None and np.array_equal(actual, common_mask), "render_mask_changed")
    result = np.ascontiguousarray(render)
    require(result.dtype == np.uint8 and result.shape == (224, 224, 3), "render_contract")
    require(bool(np.all(result[~common_mask] == NEUTRAL_BACKGROUND_RGB_UINT8)), "render_background")
    return result


def _analysis_method(route: EndpointRoute) -> bool:
    return route.method_id in SEEDED_METHODS or route.method_id == FIXED_ANALYSIS_METHOD


def _analysis_metric_row(
    pair: PairRow,
    route: EndpointRoute,
    metric_id: str,
    value: float | None,
    support: int,
    failure_code: str | None,
) -> dict[str, Any]:
    complete = value is not None and failure_code is None
    if not complete:
        require(
            metric_id == "sface_source_to_render_cosine",
            f"non_sface_metric_must_be_complete:{metric_id}",
        )
        require(
            failure_code
            in {
                "SOURCE_DETECTION_FAILURE",
                "TARGET_DETECTION_FAILURE",
                "METHOD_EMBEDDING_FAILURE",
                "NONFINITE_EMBEDDING",
            },
            "sface_failure_code_not_predeclared",
        )
    return {
        "schema_version": METRIC_ROW_SCHEMA,
        "dataset_id": pair.dataset_id,
        "metric_id": metric_id,
        "method_id": route.method_id,
        "identity_token": pair.identity_token,
        "pair_id": pair.pair_id,
        "seed": route.seed,
        "value": value,
        "support_texels": support,
        "terminal_state": "COMPLETE" if complete else "EVALUATION_FAILURE",
        "failure_code": None if complete else failure_code,
    }


def _pair_metric_row(
    pair: PairRow,
    route: EndpointRoute,
    *,
    hidden_support: int,
    all_target_support: int,
    hidden_mae: float | None,
    all_target_mae: float | None,
    lpips: float | None,
    lpips_failure: str | None,
    sface: float | None,
    sface_failure: str | None,
    render_sha256: str | None,
    render_file: str | None,
    render_origin: str,
) -> dict[str, Any]:
    return {
        "schema_version": PAIR_METRIC_SCHEMA,
        "dataset_id": pair.dataset_id,
        "identity_token": pair.identity_token,
        "pair_id": pair.pair_id,
        "source_index": pair.source_index,
        "target_index": pair.target_index,
        "source_runtime_id": pair.source_runtime_id,
        "target_runtime_id": pair.target_runtime_id,
        "target_frame_manifest_id": pair.target_frame_manifest_id,
        "method_id": route.method_id,
        "route_id": route.route_id,
        "seed": route.seed,
        "output_mode": route.output_mode,
        "origin": route.origin,
        "hidden_support_texels": hidden_support,
        "all_target_visible_support_texels": all_target_support,
        "hidden_uv_mae": hidden_mae,
        "all_target_visible_uv_mae": all_target_mae,
        "lpips_alex_v0_1": lpips,
        "lpips_terminal_state": "COMPLETE" if lpips is not None else "EVALUATION_FAILURE",
        "lpips_failure_code": lpips_failure,
        "sface_source_to_render_cosine": sface,
        "sface_terminal_state": "COMPLETE" if sface is not None else "EVALUATION_FAILURE",
        "sface_failure_code": sface_failure,
        "method_render_sha256": render_sha256,
        "method_render_file": render_file,
        "render_origin": render_origin,
        "geometry_recomputed": False,
        "method_render_detection_calls": 0,
        "retry_count": 0,
        "imputed": False,
    }


def _target_assets(
    freeuv_base: Path,
    target_index: Mapping[str, Mapping[str, Any]],
    pair: PairRow,
    cache: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_row = target_index[pair.source_runtime_id]
    target_row = target_index[pair.target_runtime_id]
    shared_root = freeuv_base / "shared_render"
    source_rgb = _load_png(
        shared_root / str(source_row["target_rgb_file"]),
        str(source_row["target_rgb_sha256"]),
        "source_target_frame",
    )
    target_rgb = _load_png(
        shared_root / str(target_row["target_rgb_file"]),
        str(target_row["target_rgb_sha256"]),
        "target_frame",
    )
    mask_path = shared_root / str(target_row["common_mask_file"])
    require(mask_path.is_file() and sha256_file(mask_path) == target_row["common_mask_sha256"], "target_mask_sha")
    common_mask = np.ascontiguousarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    cached_source_rgb, _source_uv, _source_common, _ = _screen_context(cache, pair.source_index)
    cached_target_rgb, target_uv, cached_common, _ = _screen_context(cache, pair.target_index)
    require(np.array_equal(source_rgb, cached_source_rgb), "source_frame_cache_mismatch")
    require(np.array_equal(target_rgb, cached_target_rgb), "target_frame_cache_mismatch")
    require(np.array_equal(common_mask, cached_common), "target_mask_cache_mismatch")
    require(canonical_array_sha256(target_uv) == target_row["screen_uv_sha256"], "target_screen_uv_hash")
    require(str(target_row["target_frame_manifest_id"]) == pair.target_frame_manifest_id, "target_frame_id_mismatch")
    return source_rgb, target_rgb, target_uv, common_mask


def _existing_freeuv_render(
    freeuv_base: Path,
    render_index: Mapping[tuple[str, str, str], Mapping[str, Any]],
    pair: PairRow,
    route: EndpointRoute,
) -> tuple[np.ndarray, str, str]:
    manifest_route = {
        "freeuv_native": "freeuv__native",
        "freeuv_conserved": "freeuv__conserved",
    }[route.method_id]
    row = render_index[(pair.dataset_id, pair.pair_id, manifest_route)]
    require(row.get("target_frame_manifest_id") == pair.target_frame_manifest_id, "freeuv_render_target_frame")
    path = freeuv_base / "shared_render" / str(row["method_render_file"])
    render = _load_png(path, str(row["method_render_sha256"]), "freeuv_method_render")
    return render, str(row["method_render_sha256"]), str(path)


def _lpips_value(model: Any, render: np.ndarray, target: np.ndarray, mask: np.ndarray, frame_id: str) -> float:
    from frugalface3d.evaluation.paired_render_metrics import (
        evaluate_injected_lpips,
        prepare_paired_render_for_lpips,
    )

    render_float = np.ascontiguousarray(render.transpose(2, 0, 1).astype(np.float32) / 255.0)
    target_float = np.ascontiguousarray(target.transpose(2, 0, 1).astype(np.float32) / 255.0)
    prepared = prepare_paired_render_for_lpips(
        render_float, target_float, mask, target_frame_manifest_id=frame_id
    )
    with offline_guard() as network:
        outcome = evaluate_injected_lpips(
            prepared,
            model,
            device="cpu",
            evaluator_manifest_id=f"lpips:{LPIPS_FILES['artifact'][1]}",
        )
    require(network["network_attempts"] == 0, "lpips_metric_network")
    value = float(outcome.value)
    require(math.isfinite(value) and value >= 0.0, "lpips_metric_value")
    return value


def _sface_failure_code(
    reason: str | None,
    *,
    source_failure: str | None,
    target_failure: str | None,
) -> str:
    """Map the wrapper's detailed reason to the predeclared public ledger."""

    require(isinstance(reason, str) and bool(reason), "sface_failure_reason_missing")
    if source_failure is not None:
        return "SOURCE_DETECTION_FAILURE"
    if target_failure is not None:
        return "TARGET_DETECTION_FAILURE"
    lowered = reason.lower()
    if "nonfinite" in lowered or "zero_or_invalid_norm" in lowered or "out_of_range" in lowered:
        return "NONFINITE_EMBEDDING"
    return "METHOD_EMBEDDING_FAILURE"


def _execute_dataset(
    *,
    dataset_id: str,
    cache: Any,
    pairs: Sequence[PairRow],
    routes: Sequence[EndpointRoute],
    evaluator: QualifiedEvaluators,
    freeuv_base: Path,
    target_index: Mapping[str, Mapping[str, Any]],
    render_index: Mapping[tuple[str, str, str], Mapping[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    from frugalface3d.evaluation.fixed_alignment_sface import (
        evaluate_fixed_alignment_sface_tuple,
    )

    pair_metrics: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    counters = {
        "analysis_pairs": 0,
        "structural_na_pairs": 0,
        "new_render_count": 0,
        "freeuv_render_reuse_count": 0,
        "lpips_complete": 0,
        "lpips_failed": 0,
        "sface_complete": 0,
        "sface_failed": 0,
        "source_detection_calls": 0,
        "target_detection_calls": 0,
        "method_detection_calls": 0,
    }
    require(len(routes) == EXPECTED_ROUTE_COUNT and len({route.route_id for route in routes}) == EXPECTED_ROUTE_COUNT, "route_count_or_duplicate")
    for pair in pairs:
        hidden_support, all_target_support = uv_supports(cache, pair)
        hidden_count = int(hidden_support.sum())
        all_target_count = int(all_target_support.sum())
        eligible = hidden_count > 0
        roster_rows.append(
            {
                "identity_token": pair.identity_token,
                "pair_id": pair.pair_id,
                "analysis_eligible": eligible,
                "structural_state": "EVALUABLE" if eligible else "STRUCTURAL_NA",
            }
        )
        if not eligible:
            counters["structural_na_pairs"] += 1
            continue
        counters["analysis_pairs"] += 1
        source_rgb, target_rgb, target_uv, common_mask = _target_assets(
            freeuv_base, target_index, pair, cache
        )
        reference = _unit_uv(cache.tensors["partial_uv"][pair.target_index], "paired_target")
        source_partial = _unit_uv(cache.tensors["partial_uv"][pair.source_index], "source_partial")
        source_visible = _mask(cache.tensors["visibility"][pair.source_index], "source_visible")
        renders: dict[str, np.ndarray] = {}
        interim: dict[str, dict[str, Any]] = {}
        for route in routes:
            candidate = _unit_uv(route.values[pair.source_index], route.route_id)
            if route.output_mode == "conserved":
                require(
                    np.array_equal(
                        candidate[:, source_visible], source_partial[:, source_visible]
                    ),
                    f"observed_conservation:{route.route_id}",
                )
            hidden_mae = masked_rgb_mae(candidate, reference, hidden_support)
            all_target_mae = masked_rgb_mae(candidate, reference, all_target_support)
            render_file: str | None = None
            render_hash: str | None = None
            render_origin = "new_shared_target_render"
            if route.method_id in {"freeuv_native", "freeuv_conserved"}:
                render, render_hash, render_file = _existing_freeuv_render(
                    freeuv_base, render_index, pair, route
                )
                render_origin = "freeuv_v1_2_frozen_render_reused"
                counters["freeuv_render_reuse_count"] += 1
            else:
                render = _render_candidate(candidate, target_uv, common_mask)
                relative = Path("renders") / dataset_id / pair.pair_id / f"{route.route_id}.png"
                render_path = output_root / relative
                render_hash = _write_png(render_path, render)
                render_file = relative.as_posix()
                counters["new_render_count"] += 1
            renders[route.route_id] = render
            require(evaluator.lpips is not None, "lpips_qualified_model_missing")
            lpips = _lpips_value(
                evaluator.lpips,
                render,
                target_rgb,
                common_mask,
                pair.target_frame_manifest_id,
            )
            lpips_failure = None
            counters["lpips_complete"] += 1
            interim[route.route_id] = {
                "route": route,
                "hidden_mae": hidden_mae,
                "all_target_mae": all_target_mae,
                "lpips": lpips,
                "lpips_failure": lpips_failure,
                "render_hash": render_hash,
                "render_file": render_file,
                "render_origin": render_origin,
            }
            render_rows.append(
                {
                    "schema_version": RENDER_ROW_SCHEMA,
                    "dataset_id": dataset_id,
                    "pair_id": pair.pair_id,
                    "route_id": route.route_id,
                    "method_id": route.method_id,
                    "seed": route.seed,
                    "target_frame_manifest_id": pair.target_frame_manifest_id,
                    "method_render_sha256": render_hash,
                    "method_render_file": render_file,
                    "render_origin": render_origin,
                    "terminal_state": "COMPLETE",
                    "failure_code": None,
                    "geometry_recomputed": False,
                    "method_render_detection_calls": 0,
                    "retry_count": 0,
                }
            )

        sface_results: dict[str, tuple[float | None, str | None]] = {}
        require(
            evaluator.sface_detector is not None
            and evaluator.sface_recognizer is not None
            and evaluator.sface_runtime_id is not None
            and evaluator.sface_cv2 is not None,
            "sface_qualified_runtime_missing",
        )
        result = evaluate_fixed_alignment_sface_tuple(
            source_rgb,
            target_rgb,
            renders,
            detector=evaluator.sface_detector,
            recognizer=evaluator.sface_recognizer,
            cv2_module=evaluator.sface_cv2,
            target_frame_manifest_id=pair.target_frame_manifest_id,
            runtime_manifest_id=evaluator.sface_runtime_id,
        )
        counters["source_detection_calls"] += int(result.source_detection_calls)
        counters["target_detection_calls"] += int(result.target_reference_detection_calls)
        counters["method_detection_calls"] += int(result.method_detection_calls)
        require(result.method_detection_calls == 0, "sface_method_detection_nonzero")
        observed = {row.method_id: row for row in result.methods}
        require(set(observed) == set(renders), "sface_result_route_set")
        for route_id, row in observed.items():
            if row.status == "SUCCESS":
                require(row.cosine is not None, "sface_success_value_missing")
                sface_results[route_id] = (float(row.cosine), None)
            else:
                sface_results[route_id] = (
                    None,
                    _sface_failure_code(
                        row.failure_reason,
                        source_failure=result.source_failure_reason,
                        target_failure=result.target_reference_failure_reason,
                    ),
                )
        for route in routes:
            require(route.route_id in sface_results, "sface_route_result_missing")
            sface, sface_failure = sface_results[route.route_id]
            if sface is None:
                counters["sface_failed"] += 1
            else:
                counters["sface_complete"] += 1
            row = interim[route.route_id]
            pair_metrics.append(
                _pair_metric_row(
                    pair,
                    route,
                    hidden_support=hidden_count,
                    all_target_support=all_target_count,
                    hidden_mae=row["hidden_mae"],
                    all_target_mae=row["all_target_mae"],
                    lpips=row["lpips"],
                    lpips_failure=row["lpips_failure"],
                    sface=sface,
                    sface_failure=sface_failure,
                    render_sha256=row["render_hash"],
                    render_file=row["render_file"],
                    render_origin=row["render_origin"],
                )
            )
            if _analysis_method(route):
                analysis_rows.extend(
                    (
                        _analysis_metric_row(pair, route, "hidden_uv_mae", row["hidden_mae"], hidden_count, None),
                        _analysis_metric_row(pair, route, "lpips_alex_v0_1", row["lpips"], hidden_count, row["lpips_failure"]),
                        _analysis_metric_row(pair, route, "sface_source_to_render_cosine", sface, hidden_count, sface_failure),
                    )
                )
    return pair_metrics, analysis_rows, render_rows, roster_rows, counters


def _identity_coverage(
    analysis_rows: Sequence[Mapping[str, Any]], dataset_id: str, metric_id: str
) -> dict[str, Any]:
    relevant = [
        row
        for row in analysis_rows
        if row["dataset_id"] == dataset_id and row["metric_id"] == metric_id
    ]
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in relevant:
        by_pair.setdefault((str(row["identity_token"]), str(row["pair_id"])), []).append(row)
    complete_pairs: dict[str, int] = {}
    for (identity, _pair_id), rows in by_pair.items():
        if len(rows) == 16 and all(row["terminal_state"] == "COMPLETE" for row in rows):
            complete_pairs[identity] = complete_pairs.get(identity, 0) + 1
    identities = sorted({str(row["identity_token"]) for row in relevant})
    covered = sorted(identity for identity in identities if complete_pairs.get(identity, 0) >= 1)
    return {
        "identity_count": len(identities),
        "covered_identity_count": len(covered),
        "covered_identities": covered,
        "complete_pair_count": sum(complete_pairs.values()),
        "complete_pairs_per_identity": dict(sorted(complete_pairs.items())),
        "symmetric_complete_case_rule": "all_16_analysis_routes_complete_for_pair",
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("postprocess_output_exists_no_rerun")

    # Evaluator qualification is verified before any cache, target frame, or
    # metric image is opened.
    evaluator = _load_qualified_evaluators(
        args.qualification_root, args.evaluator_assets, formal=not args.development
    )
    training_authority, training_terminal_path = _load_training_authority(
        args.training_terminal
    )
    training_terminal_sha = sha256_file(training_terminal_path)

    from reproducibility.w5b49n_mechanism_closure_v1.runtime.eval_cache_io import (
        MANIFEST_FILE as D1_MANIFEST,
        TENSOR_FILE as D1_TENSOR,
        load_eval_cache,
    )
    from reproducibility.w5b49n_mechanism_closure_v1.runtime.realy_eval_cache_io import (
        MANIFEST_FILE as D2_MANIFEST,
        TENSOR_FILE as D2_TENSOR,
        load_realy_source_cache,
    )

    d1_root = args.d1_eval_cache.expanduser().resolve(strict=True)
    d2_root = args.d2_eval_cache.expanduser().resolve(strict=True)
    d1_cache = load_eval_cache(d1_root)
    d2_cache = load_realy_source_cache(d2_root)
    caches = {"D1": d1_cache, "D2": d2_cache}
    freeuv_routes, freeuv_render_rows, target_index, freeuv_render_index, freeuv_paths = _load_freeuv(
        args.freeuv_root,
        args.freeuv_archive,
        args.freeuv_safe_sample_map,
        {"D1": 160, "D2": 400},
    )
    pairs = _pairs_from_freeuv(freeuv_render_rows, caches, args.d2_roster_root)

    d1_routes = [
        *_load_v14_routes(
            args.d1_v14_raw_root,
            "D1",
            160,
            training_authority,
            training_terminal_sha,
        ),
        _load_frozen_b_lite(args.d1_b_lite_route_root, 160),
        _load_public_batch(args.d1_lama_root, d1_cache, "D1", 160),
        _load_public_batch(args.d1_zits_root, d1_cache, "D1", 160),
        *freeuv_routes["D1"],
    ]
    d2_routes = [
        *_load_v14_routes(
            args.d2_v14_raw_root,
            "D2",
            400,
            training_authority,
            training_terminal_sha,
        ),
        _load_frozen_b_lite(args.d2_b_lite_route_root, 400),
        _load_public_batch(args.d2_lama_root, d2_cache, "D2", 400),
        _load_public_batch(args.d2_zits_root, d2_cache, "D2", 400),
        *freeuv_routes["D2"],
    ]
    output_root.mkdir(parents=True, mode=0o700)
    attempt = {
        "schema_version": "frugalface3d.w5b49n.v14.postprocess_attempt.v1",
        "status": "STARTED_AFTER_CURRENT_HOST_QUALIFICATION_REPLAY",
        "program_id": PROGRAM_ID,
        "automatic_retry": False,
        "training_performed": False,
        "geometry_recomputed": False,
        "freeuv_inference_performed": False,
        "baseline_inference_performed": False,
        "lpips_qualification_terminal_sha256": sha256_file(Path(str(evaluator.lpips_terminal["_terminal_path"]))),
        "sface_qualification_terminal_sha256": sha256_file(Path(str(evaluator.sface_terminal["_terminal_path"]))),
        "bound_inputs": {
            "d1_cache_manifest": bound_file(d1_root / D1_MANIFEST),
            "d1_cache_tensor": bound_file(d1_root / D1_TENSOR),
            "d2_cache_manifest": bound_file(d2_root / D2_MANIFEST),
            "d2_cache_tensor": bound_file(d2_root / D2_TENSOR),
            "v14_training_terminal": bound_file(training_terminal_path),
            "d1_v14_inference_terminal": bound_file(
                args.d1_v14_raw_root.expanduser().resolve(strict=True)
                / "INFERENCE_TERMINAL.json"
            ),
            "d2_v14_inference_terminal": bound_file(
                args.d2_v14_raw_root.expanduser().resolve(strict=True)
                / "INFERENCE_TERMINAL.json"
            ),
            "freeuv_archive": bound_file(freeuv_paths["archive"]),
            "freeuv_safe_sample_map": bound_file(freeuv_paths["safe_sample_map"]),
            "freeuv_activity_terminal": bound_file(freeuv_paths["activity_terminal"]),
            "freeuv_shared_terminal": bound_file(freeuv_paths["shared_terminal"]),
            "freeuv_target_manifest": bound_file(freeuv_paths["target_manifest"]),
            "freeuv_render_manifest": bound_file(freeuv_paths["render_manifest"]),
        },
    }
    write_json(output_root / "ATTEMPT.json", attempt)
    try:
        all_pair_metrics: list[dict[str, Any]] = []
        all_analysis_rows: list[dict[str, Any]] = []
        all_render_rows: list[dict[str, Any]] = []
        roster_blocks: dict[str, Any] = {}
        counters_by_dataset: dict[str, Any] = {}
        for dataset_id, cache, dataset_pairs, routes in (
            ("D1", d1_cache, pairs["D1"], d1_routes),
            ("D2", d2_cache, pairs["D2"], d2_routes),
        ):
            pair_metrics, analysis_rows, render_rows, roster_rows, counters = _execute_dataset(
                dataset_id=dataset_id,
                cache=cache,
                pairs=dataset_pairs,
                routes=routes,
                evaluator=evaluator,
                freeuv_base=freeuv_paths["base"],
                target_index=target_index,
                render_index=freeuv_render_index,
                output_root=output_root,
            )
            all_pair_metrics.extend(pair_metrics)
            all_analysis_rows.extend(analysis_rows)
            all_render_rows.extend(render_rows)
            roster_blocks[dataset_id] = {"rows": roster_rows}
            counters_by_dataset[dataset_id] = counters

        require(counters_by_dataset["D1"]["analysis_pairs"] == 148, "d1_analysis_pair_count")
        require(counters_by_dataset["D1"]["structural_na_pairs"] == 12, "d1_structural_na_count")
        require(counters_by_dataset["D2"]["analysis_pairs"] == 1200, "d2_analysis_pair_count")
        require(counters_by_dataset["D2"]["structural_na_pairs"] == 0, "d2_structural_na_count")
        pair_roster_path = output_root / "PAIR_ROSTER.json"
        write_json(
            pair_roster_path,
            {"schema_version": PAIR_ROSTER_SCHEMA, "datasets": roster_blocks},
        )
        pair_metric_sha, pair_metric_count = write_jsonl(
            output_root / "PAIR_METRICS.jsonl", all_pair_metrics
        )
        analysis_sha, analysis_count = write_jsonl(
            output_root / "ANALYSIS_METRIC_ROWS.jsonl", all_analysis_rows
        )
        render_sha, render_count = write_jsonl(
            output_root / "RENDER_MANIFEST.jsonl", all_render_rows
        )
        require(pair_metric_count == 1348 * EXPECTED_ROUTE_COUNT, "pair_metric_count")
        require(analysis_count == 1348 * 16 * 3, "analysis_metric_count")
        require(render_count == 1348 * EXPECTED_ROUTE_COUNT, "render_manifest_count")

        freeuv_wrapper = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "PASS_V14_FREEUV_V1_2_COMPLETE",
            "method_id": "freeuv_conserved",
            "forward_count": 560,
            "source_package_sha256": FREEUV_ARCHIVE_SHA256,
            "source_terminal_sha256": sha256_file(freeuv_paths["activity_terminal"]),
            "no_new_inference": True,
            "new_forward_count": 0,
            "prior_outputs_consumed_only": True,
            "bound_files": [
                bound_file(freeuv_paths["archive"]),
                bound_file(freeuv_paths["safe_sample_map"]),
                bound_file(freeuv_paths["activity_terminal"]),
                bound_file(freeuv_paths["shared_terminal"]),
                bound_file(freeuv_paths["target_manifest"]),
                bound_file(freeuv_paths["render_manifest"]),
                bound_file(freeuv_paths["base"] / "activity/D1/FREEUV_D1_COMMON64_OUTPUTS.npz"),
                bound_file(freeuv_paths["base"] / "activity/D2/FREEUV_D2_COMMON64_OUTPUTS.npz"),
            ],
        }
        write_json(output_root / "FREEUV_V1_2_TERMINAL.json", freeuv_wrapper)
        for dataset_id, root, manifest_name, tensor_name, expected_pairs in (
            ("D1", d1_root, D1_MANIFEST, D1_TENSOR, 148),
            ("D2", d2_root, D2_MANIFEST, D2_TENSOR, 1200),
        ):
            write_json(
                output_root / f"{dataset_id}_EVAL_CACHE_TERMINAL.json",
                {
                    "schema_version": TERMINAL_SCHEMA,
                    "status": "PASS_V14_EVAL_CACHE_COMPLETE",
                    "dataset_id": dataset_id,
                    "analysis_pair_count": expected_pairs,
                    "bound_files": [
                        bound_file(root / manifest_name),
                        bound_file(root / tensor_name),
                        bound_file(pair_roster_path),
                    ],
                },
            )

        total_new_renders = sum(row["new_render_count"] for row in counters_by_dataset.values())
        total_reused_freeuv = sum(row["freeuv_render_reuse_count"] for row in counters_by_dataset.values())
        require(total_new_renders == 1348 * 18, "new_render_count")
        require(total_reused_freeuv == 1348 * 2, "freeuv_render_reuse_count")
        shared_terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "PASS_V14_SHARED_RENDER_COMPLETE",
            "analysis_pair_count": 1348,
            "new_method_render_count": total_new_renders,
            "frozen_freeuv_render_reuse_count": total_reused_freeuv,
            "target_frame_materialization_count": 0,
            "target_frames_reused_from_freeuv_v1_2": 560,
            "geometry_recomputed": False,
            "render_manifest_sha256": render_sha,
            "bound_files": [
                bound_file(output_root / "RENDER_MANIFEST.jsonl"),
                bound_file(freeuv_paths["target_manifest"]),
                bound_file(freeuv_paths["render_manifest"]),
                bound_file(pair_roster_path),
            ],
        }
        write_json(output_root / "SHARED_RENDER_TERMINAL.json", shared_terminal)

        lpips_qualification_path = Path(str(evaluator.lpips_terminal["_terminal_path"]))
        sface_qualification_path = Path(str(evaluator.sface_terminal["_terminal_path"]))
        lpips_expected = 1348 * 16
        lpips_complete = sum(
            row["metric_id"] == "lpips_alex_v0_1" and row["terminal_state"] == "COMPLETE"
            for row in all_analysis_rows
        )
        lpips_terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": (
                "PASS_V14_LPIPS_COMPLETE"
                if evaluator.lpips is not None and lpips_complete == lpips_expected
                else "TERMINAL_V14_LPIPS_WITH_RETAINED_FAILURES"
            ),
            "metric_id": "lpips_alex_v0_1",
            "analysis_pair_count": 1348,
            "device_backend": "cpu",
            "cuda_calls": 0,
            "qualification_required": True,
            "qualification_terminal_sha256": sha256_file(lpips_qualification_path),
            "expected_metric_row_count": lpips_expected,
            "complete_metric_row_count": lpips_complete,
            "failure_metric_row_count": lpips_expected - lpips_complete,
            "failure_rows_retained": True,
            "bound_files": [
                bound_file(lpips_qualification_path),
                bound_file(output_root / "ANALYSIS_METRIC_ROWS.jsonl"),
                bound_file(output_root / "PAIR_METRICS.jsonl"),
            ],
        }
        write_json(output_root / "LPIPS_TERMINAL.json", lpips_terminal)

        sface_expected = 1348 * 16
        sface_complete = sum(
            row["metric_id"] == "sface_source_to_render_cosine"
            and row["terminal_state"] == "COMPLETE"
            for row in all_analysis_rows
        )
        coverage = {
            dataset_id: _identity_coverage(
                all_analysis_rows, dataset_id, "sface_source_to_render_cosine"
            )
            for dataset_id in ("D1", "D2")
        }
        coverage_gate = (
            coverage["D1"]["covered_identity_count"] >= 18
            and coverage["D2"]["covered_identity_count"] >= 90
        )
        sface_terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": (
                "PASS_V14_SFACE_COMPLETE"
                if evaluator.sface_detector is not None
                and len(
                    [
                        row
                        for row in all_analysis_rows
                        if row["metric_id"] == "sface_source_to_render_cosine"
                    ]
                )
                == sface_expected
                else "TERMINAL_V14_SFACE_WITH_RETAINED_FAILURES"
            ),
            "metric_id": "sface_source_to_render_cosine",
            "analysis_pair_count": 1348,
            "device_backend": "cpu",
            "cuda_calls": 0,
            "qualification_required": True,
            "qualification_terminal_sha256": sha256_file(sface_qualification_path),
            "expected_metric_row_count": sface_expected,
            "complete_metric_row_count": sface_complete,
            "failure_metric_row_count": sface_expected - sface_complete,
            "failure_rows_retained": True,
            "failure_ledger_complete": True,
            "silent_row_drop_count": 0,
            "symmetric_complete_case": True,
            "identity_coverage": coverage,
            "identity_coverage_gate_pass": coverage_gate,
            "confirmation_gate": {"D1_min_identities": 18, "D2_min_identities": 90},
            "embedding_persisted": False,
            "bound_files": [
                bound_file(sface_qualification_path),
                bound_file(output_root / "ANALYSIS_METRIC_ROWS.jsonl"),
                bound_file(output_root / "PAIR_METRICS.jsonl"),
            ],
        }
        write_json(output_root / "SFACE_TERMINAL.json", sface_terminal)

        terminal = {
            "schema_version": "frugalface3d.w5b49n.v14.postprocess_terminal.v1",
            "status": (
                "PASS_V14_POSTPROCESS_COMPLETE"
                if lpips_terminal["status"] == "PASS_V14_LPIPS_COMPLETE"
                and sface_terminal["status"] == "PASS_V14_SFACE_COMPLETE"
                else "TERMINAL_V14_POSTPROCESS_WITH_RETAINED_METRIC_FAILURES"
            ),
            "analysis_pair_count": 1348,
            "route_count_per_pair": EXPECTED_ROUTE_COUNT,
            "pair_metric_row_count": pair_metric_count,
            "analysis_metric_row_count": analysis_count,
            "render_manifest_row_count": render_count,
            "pair_roster_sha256": sha256_file(pair_roster_path),
            "pair_metrics_sha256": pair_metric_sha,
            "analysis_metric_rows_sha256": analysis_sha,
            "render_manifest_sha256": render_sha,
            "freeuv_inference_performed": False,
            "new_freeuv_forward_count": 0,
            "training_performed": False,
            "geometry_recomputed": False,
            "automatic_retry": False,
            "imputation_count": 0,
            "counters": counters_by_dataset,
            "downstream_statistics_authorized": (
                lpips_terminal["status"] == "PASS_V14_LPIPS_COMPLETE"
                and sface_terminal["status"] == "PASS_V14_SFACE_COMPLETE"
            ),
            "v14_training_terminal_sha256": training_terminal_sha,
            "d1_v14_inference_terminal_sha256": sha256_file(
                args.d1_v14_raw_root.expanduser().resolve(strict=True)
                / "INFERENCE_TERMINAL.json"
            ),
            "d2_v14_inference_terminal_sha256": sha256_file(
                args.d2_v14_raw_root.expanduser().resolve(strict=True)
                / "INFERENCE_TERMINAL.json"
            ),
            "bound_files": [
                bound_file(Path(str(evaluator.lpips_terminal["_terminal_path"]))),
                bound_file(Path(str(evaluator.sface_terminal["_terminal_path"]))),
                bound_file(training_terminal_path),
                bound_file(
                    args.d1_v14_raw_root.expanduser().resolve(strict=True)
                    / "INFERENCE_TERMINAL.json"
                ),
                bound_file(
                    args.d2_v14_raw_root.expanduser().resolve(strict=True)
                    / "INFERENCE_TERMINAL.json"
                ),
                bound_file(pair_roster_path),
                bound_file(output_root / "PAIR_METRICS.jsonl"),
                bound_file(output_root / "ANALYSIS_METRIC_ROWS.jsonl"),
                bound_file(output_root / "RENDER_MANIFEST.jsonl"),
                bound_file(output_root / "FREEUV_V1_2_TERMINAL.json"),
                bound_file(output_root / "D1_EVAL_CACHE_TERMINAL.json"),
                bound_file(output_root / "D2_EVAL_CACHE_TERMINAL.json"),
                bound_file(output_root / "SHARED_RENDER_TERMINAL.json"),
                bound_file(output_root / "LPIPS_TERMINAL.json"),
                bound_file(output_root / "SFACE_TERMINAL.json"),
            ],
        }
        write_json(output_root / "POSTPROCESS_TERMINAL.json", terminal)
        return terminal
    except Exception as error:
        write_json(
            output_root / "POSTPROCESS_FAILURE.json",
            {
                "status": "FAILED_POSTPROCESS_ROOT_RETAINED_NO_RETRY",
                "failure_code": safe_failure(error),
                "automatic_retry": False,
                "freeuv_inference_performed": False,
                "geometry_recomputed": False,
                "imputation_count": 0,
            },
        )
        raise


def source_check() -> dict[str, Any]:
    require(len(SEEDS) == 5 and len(SEEDED_METHODS) == 3, "source_seed_matrix")
    require(EXPECTED_ROUTE_COUNT == 20, "source_route_count")
    require(QUALIFICATION_TOLERANCE_ABS == 1.0e-6, "source_probe_tolerance")
    require(SFACE_FORMAL_OPENCV_VERSION == "4.10.0", "source_sface_opencv_version")
    require(FREEUV_ARCHIVE_SHA256.startswith("25e26864"), "source_freeuv_hash")
    return {
        "status": "PASS_V14_POSTPROCESS_SOURCE_CHECK",
        "new_training_units": 0,
        "freeuv_inference_routes": 0,
        "baseline_inference_routes": 0,
        "qualification_precedes_real_image_reads": True,
        "qualification_failure_rows_retained": True,
        "real_image_reads": 0,
        "metric_rows": 0,
        "scientific_result_generated": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("source-check")
    check.set_defaults(function=lambda _args: source_check())

    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--evaluator-assets", type=Path, required=True)
    qualify_parser.add_argument("--output-root", type=Path, required=True)
    qualify_parser.add_argument("--development", action="store_true")
    qualify_parser.set_defaults(
        function=lambda args: qualify(
            args.evaluator_assets, args.output_root, formal=not args.development
        )
    )

    execute_parser = subparsers.add_parser("execute")
    for name in (
        "d1-eval-cache",
        "d2-eval-cache",
        "d2-roster-root",
        "training-terminal",
        "d1-v14-raw-root",
        "d2-v14-raw-root",
        "d1-b-lite-route-root",
        "d2-b-lite-route-root",
        "d1-lama-root",
        "d2-lama-root",
        "d1-zits-root",
        "d2-zits-root",
        "freeuv-root",
        "freeuv-archive",
        "freeuv-safe-sample-map",
        "qualification-root",
        "evaluator-assets",
        "output-root",
    ):
        execute_parser.add_argument("--" + name, type=Path, required=True)
    execute_parser.add_argument("--development", action="store_true")
    execute_parser.set_defaults(function=execute)
    return value


def main() -> int:
    args = parser().parse_args()
    result = args.function(args)
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
