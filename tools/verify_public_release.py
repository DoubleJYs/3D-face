#!/usr/bin/env python3
"""Verify a FaceUV-Eval public release without network access."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any


STATUS = "PASS_FACEUV_EVAL_PUBLIC_RELEASE"
MANIFEST_NAME = "PUBLIC_FILE_MANIFEST.json"
MANIFEST_SCHEMA = "faceuv-eval.public-manifest.v1"
BUILD_STATUS = "PASS_FACEUV_EVAL_RELEASE_TREE_BUILD"
MAX_FILE_BYTES = 50 * 1024 * 1024
CC_BY_4_0_BYTES = 18656
CC_BY_4_0_SHA256 = "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"
ALLOWED_RIGHTS = frozenset({"Apache-2.0", "CC-BY-4.0", "NOASSERTION"})
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {"paper", "manuscript", "submission", "private", "restricted_assets"}
)
PUBLIC_FORBIDDEN_PATH_TOKENS = frozenset(
    {
        "internal",
        "internal_records",
        "internal_record",
        "work_records",
        "work_record",
        "work_logs",
        "work_log",
        "experiment_logs",
        "experiment_log",
        "cloud_logs",
        "cloud_log",
        "logs",
        "permission",
        "permissions",
        "permission_records",
        "permission_record",
        "authorization_emails",
        "authorization_email",
        "emails",
        "email",
        "correspondence",
        "private_correspondence",
        "data",
        "dataset",
        "datasets",
        "raw_data",
        "private_data",
        "faces",
        "face",
        "face_images",
        "portraits",
        "portrait",
        "portrait_images",
        "biometrics",
        "biometric",
        "biometric_data",
        "biometric_tensors",
        "identity_mappings",
        "identity_mapping",
        "identity_map",
        "identity_maps",
        "identity_roster",
        "case_rosters",
        "case_roster",
        "weights",
        "weight",
        "model_weights",
        "model_weight",
        "checkpoint",
        "checkpoints",
        "archive",
        "archives",
        "experiment_archives",
        "experiment_archive",
    }
)
PUBLIC_FORBIDDEN_COMPACT_TOKENS = frozenset(
    token.replace("_", "") for token in PUBLIC_FORBIDDEN_PATH_TOKENS
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".docx",
        ".tex",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".h5",
        ".hdf5",
        ".npy",
        ".npz",
        ".pkl",
        ".pickle",
        ".sqlite",
        ".db",
        ".env",
        ".pem",
        ".key",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".zst",
        ".tgz",
        ".rar",
        ".parquet",
        ".arrow",
        ".feather",
        ".mat",
        ".bin",
        ".joblib",
        ".jpg",
        ".jpeg",
    }
)
AUDIT_FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".tif", ".tiff", ".svg"})
DATA_SUFFIXES = frozenset({".csv", ".json", ".jsonl"})
DATA_TABLE_SUFFIXES = frozenset({".csv", ".jsonl"})
APPROVED_AUDIT_TABLE_PREFIXES = (
    "audit/mappings/",
    "audit/statistics/",
    "audit/robustness/",
    "audit/figures/source_data/",
)
HEX64 = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """Raised when the public release violates its exact contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _safe_relative(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"empty {label} path")
    pure = PurePosixPath(value)
    _require(
        value == pure.as_posix()
        and not pure.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"unsafe {label} path: {value}",
    )
    _require(
        not FORBIDDEN_DIRECTORY_NAMES.intersection(part.lower() for part in pure.parts),
        f"forbidden directory in {label} path: {value}",
    )
    tokens: set[str] = set()
    for part in pure.parts:
        tokens.add(re.sub(r"[- .]+", "_", part.casefold()))
        tokens.add(re.sub(r"[- .]+", "_", PurePosixPath(part).stem.casefold()))
    _require(
        not PUBLIC_FORBIDDEN_PATH_TOKENS.intersection(tokens)
        and not PUBLIC_FORBIDDEN_COMPACT_TOKENS.intersection(
            token.replace("_", "") for token in tokens
        ),
        f"forbidden public path component: {value}",
    )
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"regular file required: {label}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse JSON {label}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {label}")
    return value


def _expected_rights(relative: str) -> str:
    suffix = PurePosixPath(relative).suffix.lower()
    if relative == "LICENSE" or relative.startswith("LICENSES/"):
        return "NOASSERTION"
    if relative.startswith("audit/mappings/") and suffix == ".csv":
        return "CC-BY-4.0"
    if relative.startswith(("audit/statistics/", "audit/robustness/")) and suffix in DATA_SUFFIXES:
        return "CC-BY-4.0"
    if relative.startswith(("audit/figures/source_data/", "audit/figures/reference_outputs/")):
        return "CC-BY-4.0"
    return "Apache-2.0"


def _scan_png(relative: str, payload: bytes) -> None:
    _require(payload.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG signature: {relative}")
    _require(len(payload) >= 24 and payload[12:16] == b"IHDR", f"invalid PNG header: {relative}")


def _scan_pdf(relative: str, payload: bytes) -> None:
    _require(payload.startswith(b"%PDF-"), f"invalid PDF signature: {relative}")
    _require(b"%%EOF" in payload[-2048:], f"invalid PDF trailer: {relative}")


def _scan_tiff(relative: str, payload: bytes) -> None:
    valid = (
        payload.startswith(b"II*\x00")
        or payload.startswith(b"MM\x00*")
        or payload.startswith(b"II+\x00\x08\x00\x00\x00")
        or payload.startswith(b"MM\x00+\x00\x00\x00\x08")
    )
    _require(valid, f"invalid TIFF signature: {relative}")


def _scan_svg(relative: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"non-UTF8 SVG: {relative}") from exc
    lowered = text.lower()
    _require("<!doctype" not in lowered and "<script" not in lowered, f"unsafe SVG: {relative}")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise VerificationError(f"invalid SVG XML: {relative}") from exc
    _require(root.tag.rsplit("}", 1)[-1].lower() == "svg", f"invalid SVG root: {relative}")
    for element in root.iter():
        _require(element.tag.rsplit("}", 1)[-1].casefold() != "script", f"unsafe SVG: {relative}")
        for attribute, value in element.attrib.items():
            if attribute.rsplit("}", 1)[-1].casefold() == "href":
                _require(not _unsafe_svg_reference(value), f"unsafe SVG reference: {relative}")
            for css_reference in re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", value, flags=re.IGNORECASE):
                _require(not _unsafe_svg_reference(css_reference), f"unsafe SVG reference: {relative}")
    for css_reference in re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", text, flags=re.IGNORECASE):
        _require(not _unsafe_svg_reference(css_reference), f"unsafe SVG reference: {relative}")
    _scan_text(relative, payload)


def _unsafe_svg_reference(value: str) -> bool:
    reference = value.strip()
    if not reference or reference.startswith("#"):
        return False
    return True


def _candidate_markers() -> tuple[str, ...]:
    return (
        "FINAL_SOURCE_" + "CANDIDATE",
        "EXACT_COMMIT_UNBOUND__" + "NOT_PUBLISHED",
        "PAYLOAD_RIGHTS_CLASSIFIED__" + "PUBLICATION_NOT_AUTHORIZED",
        "publication_" + "authorized=false",
        '"publication_' + 'authorized": false',
        "V15 source " + "candidate",
        "source candidate is not yet " + "bound",
    )


def _scan_text(relative: str, payload: bytes) -> None:
    _require(b"\x00" not in payload, f"binary payload outside audited figures: {relative}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"non-UTF8 text payload: {relative}") from exc

    host_markers = (
        "/" + "Users/",
        "/" + "home/",
        "/private/" + "var/folders/",
        "C:" + "\\Users\\",
        "/" + "root/",
        "/" + "autodl-tmp/",
        "/" + "autodl-fs/",
        "xwechat_" + "files",
        ".codex/" + "attachments/",
    )
    for marker in host_markers:
        _require(marker not in text, f"host or cloud path forbidden: {relative}")
    _require(
        re.search(r"(?i)(?:s3|gs|oss|cos|obs)://", text) is None,
        f"cloud URI forbidden: {relative}",
    )
    _require(
        ("-----BEGIN " + "PRIVATE KEY-----") not in text
        and ("-----BEGIN RSA " + "PRIVATE KEY-----") not in text
        and ("-----BEGIN EC " + "PRIVATE KEY-----") not in text
        and ("-----BEGIN OPENSSH " + "PRIVATE KEY-----") not in text,
        f"private key forbidden: {relative}",
    )
    _require(
        re.search(r"(?i)gh[pousr]_[A-Za-z0-9_]{20,}", text) is None,
        f"GitHub token form forbidden: {relative}",
    )
    standalone_patterns = (
        r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])",
        r"(?<![A-Z0-9])(?:AK" + r"IA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])",
        r"(?<![A-Za-z0-9_-])AI" + r"za[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{7,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_-])xo" + r"x[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_])np" + r"m_[A-Za-z0-9]{36}(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_-])s" + r"k-proj-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_-])s" + r"k-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])",
        r"(?<![A-Za-z0-9_])(?:s" + r"k|rk)_(?:live|test)_[A-Za-z0-9]{16,}(?![A-Za-z0-9_])",
    )
    _require(
        all(re.search(pattern, text) is None for pattern in standalone_patterns),
        f"credential-like text forbidden: {relative}",
    )
    _require(
        re.search(
            r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|password)"
            r"\s*[:=]\s*['\"][^'\"<>$\{\}\s]{8,}['\"]",
            text,
        )
        is None,
        f"credential assignment forbidden: {relative}",
    )
    _require(
        re.search(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:]+:[^\s/@]+@", text) is None,
        f"URL credential forbidden: {relative}",
    )
    for marker in _candidate_markers():
        _require(marker not in text, f"candidate state forbidden: {relative}")
    latex_document = "\\" + "documentclass"
    latex_abstract = "\\begin{" + "abstract}"
    _require(
        latex_document not in text and latex_abstract not in text,
        f"manuscript source forbidden: {relative}",
    )


def _scan_payload(relative: str, payload: bytes, rights_id: str) -> None:
    suffix = PurePosixPath(relative).suffix.lower()
    _require(
        not payload.startswith(b"version https://git-lfs.github.com/spec/v1"),
        f"Git LFS pointer forbidden: {relative}",
    )
    if suffix in AUDIT_FIGURE_SUFFIXES:
        _require(
            relative.startswith("audit/figures/reference_outputs/")
            and rights_id == "CC-BY-4.0",
            f"figure binary outside checksum-bound audit outputs: {relative}",
        )
        if suffix == ".png":
            _scan_png(relative, payload)
        elif suffix == ".pdf":
            _scan_pdf(relative, payload)
        elif suffix in {".tif", ".tiff"}:
            _scan_tiff(relative, payload)
        else:
            _scan_svg(relative, payload)
        return
    _require(suffix not in FORBIDDEN_SUFFIXES, f"forbidden suffix: {relative}")
    if suffix in DATA_TABLE_SUFFIXES:
        known_source_table = not relative.startswith("audit/") and rights_id == "Apache-2.0"
        _require(
            known_source_table or relative.startswith(APPROVED_AUDIT_TABLE_PREFIXES),
            f"data table outside approved public subtree: {relative}",
        )
    _scan_text(relative, payload)


def _walk_release(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    symlinks: list[str] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(root)
        if relative_dir == Path(".") and ".git" in names:
            git_path = current / ".git"
            if git_path.is_symlink():
                symlinks.append(".git")
            names.remove(".git")
        for name in list(names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative, "directory")
            if path.is_symlink():
                symlinks.append(relative)
                names.remove(name)
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative, "file")
            if path.is_symlink():
                symlinks.append(relative)
            else:
                files.add(relative)
    return files, symlinks


def _verify_audit_checksums(root: Path) -> None:
    audit = root / "audit"
    checksum_path = audit / "SHA256SUMS.txt"
    _require(checksum_path.is_file() and not checksum_path.is_symlink(), "audit checksum missing")
    listed: set[str] = set()
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        _require("  " in line, f"malformed audit checksum row {number}")
        expected, relative = line.split("  ", 1)
        _require(HEX64.fullmatch(expected) is not None, f"bad audit checksum row {number}")
        relative = _safe_relative(relative, "audit checksum")
        _require(relative not in listed, f"duplicate audit checksum path: {relative}")
        listed.add(relative)
        path = audit.joinpath(*PurePosixPath(relative).parts)
        _require(path.is_file() and not path.is_symlink(), f"audit checksum target missing: {relative}")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"audit checksum mismatch: {relative}")
    actual = {
        path.relative_to(audit).as_posix()
        for path in audit.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != checksum_path
        and "rebuilt_public_outputs" not in path.parts
        and "__pycache__" not in path.parts
    }
    _require(listed == actual, "audit checksum membership mismatch")


def _verify_metadata(root: Path) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    _require(readme.startswith("# FaceUV-Eval\n"), "README project identity mismatch")
    for command in (
        "python3 -B tools/verify_public_release.py",
        "python3 -B START_HERE.py smoke",
        "python3 -B START_HERE.py test-source",
        "python3 -B audit/statistics/recompute_public_statistics.py",
    ):
        _require(command in readme, f"README command missing: {command}")
    _require("separately licensed assets" in readme, "README asset boundary missing")

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    for required in (
        "type: software",
        'version: "1.0.0"',
        "date-released: 2026-08-31",
        "https://github.com/DoubleJYs/FaceUV-Eval",
    ):
        _require(required in citation, f"CITATION field missing: {required}")
    _require("abstract:" not in citation.lower() and "email:" not in citation.lower(), "CITATION private field")

    security = (root / "SECURITY.md").read_text(encoding="utf-8").lower()
    _require("private vulnerability reporting" in security and "@" not in security, "SECURITY route mismatch")
    cc_payload = (root / "LICENSES/CC-BY-4.0.txt").read_bytes()
    _require(
        len(cc_payload) == CC_BY_4_0_BYTES
        and hashlib.sha256(cc_payload).hexdigest() == CC_BY_4_0_SHA256
        and cc_payload.startswith(b"Attribution 4.0 International\n"),
        "CC-BY-4.0 reviewed payload mismatch",
    )
    apache = (root / "LICENSE").read_text(encoding="utf-8")
    _require("Apache License" in apache and "Version 2.0" in apache, "Apache-2.0 license missing")

    start = (root / "START_HERE.py").read_text(encoding="utf-8")
    _require("tools/verify_public_release.py" in start, "START_HERE check target mismatch")
    _require("build_package.py" not in start and "verify_package.py" not in start, "obsolete START_HERE target")


def verify(root: Path) -> dict[str, object]:
    """Verify exact manifest membership, hashes, sizes, rights IDs, forbidden paths and suffixes, credential patterns, symlinks, and release metadata."""
    root = root.resolve()
    _require(root.is_dir(), "release root is not a directory")
    manifest_path = root / MANIFEST_NAME
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "public manifest regular file required",
    )
    _require(manifest_path.stat().st_size <= MAX_FILE_BYTES, "public manifest exceeds 50 MiB")
    manifest = _read_json(manifest_path, MANIFEST_NAME)
    _require(
        set(manifest) == {"schema_version", "status", "source_file_count", "audit_file_count", "files"},
        "unexpected public manifest keys",
    )
    _require(manifest["schema_version"] == MANIFEST_SCHEMA, "public manifest schema mismatch")
    _require(manifest["status"] == BUILD_STATUS, "public manifest status mismatch")
    _require(
        isinstance(manifest["source_file_count"], int) and manifest["source_file_count"] > 0,
        "invalid source file count",
    )
    _require(
        isinstance(manifest["audit_file_count"], int) and manifest["audit_file_count"] > 0,
        "invalid audit file count",
    )
    rows = manifest["files"]
    _require(isinstance(rows, list) and rows, "public manifest files missing")

    expected: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for index, row in enumerate(rows):
        _require(
            isinstance(row, dict) and set(row) == {"path", "sha256", "byte_size", "rights_id"},
            f"invalid public manifest row {index}",
        )
        relative = _safe_relative(row["path"], "manifest")
        _require(relative != MANIFEST_NAME, "manifest cannot list itself")
        _require(relative not in expected, f"duplicate manifest path: {relative}")
        _require(HEX64.fullmatch(row["sha256"]) is not None, f"invalid manifest hash: {relative}")
        _require(isinstance(row["byte_size"], int) and row["byte_size"] >= 0, f"invalid byte size: {relative}")
        _require(row["byte_size"] <= MAX_FILE_BYTES, f"file exceeds 50 MiB: {relative}")
        _require(row["rights_id"] in ALLOWED_RIGHTS, f"unknown rights id: {relative}")
        _require(row["rights_id"] == _expected_rights(relative), f"rights class mismatch: {relative}")
        expected[relative] = row
        ordered_paths.append(relative)
    _require(ordered_paths == sorted(ordered_paths), "manifest paths are not sorted")
    audit_paths = [relative for relative in expected if relative.startswith("audit/")]
    non_audit_paths = [relative for relative in expected if not relative.startswith("audit/")]
    _require(
        len(audit_paths) == manifest["audit_file_count"] + 1,
        "audit file count mismatch",
    )
    _require(
        len(non_audit_paths) == manifest["source_file_count"] + 5,
        "source file count mismatch",
    )

    actual, symlinks = _walk_release(root)
    _require(not symlinks, f"symlink forbidden: {symlinks[0] if symlinks else ''}")
    _require(actual == set(expected) | {MANIFEST_NAME}, "public manifest membership mismatch")

    total_bytes = 0
    for relative, row in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        _require(path.is_file() and not path.is_symlink(), f"regular file required: {relative}")
        size = path.stat().st_size
        _require(size <= MAX_FILE_BYTES, f"file exceeds 50 MiB: {relative}")
        _require(size == row["byte_size"], f"byte size mismatch: {relative}")
        payload = path.read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == row["sha256"], f"hash mismatch: {relative}")
        _scan_payload(relative, payload, row["rights_id"])
        total_bytes += size

    _scan_text(MANIFEST_NAME, (root / MANIFEST_NAME).read_bytes())
    _verify_audit_checksums(root)
    _verify_metadata(root)
    return {
        "status": STATUS,
        "verified_file_count": len(expected) + 1,
        "verified_payload_bytes": total_bytes,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        result = verify(root)
    except VerificationError as exc:
        print(json.dumps({"status": "FAIL_FACEUV_EVAL_PUBLIC_RELEASE", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
