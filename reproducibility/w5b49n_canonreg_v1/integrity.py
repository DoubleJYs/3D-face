"""Source and runtime binding helpers for the CanonReg sensitivity package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE_ROOT / "contract.json"
SOURCE_MANIFEST_PATH = PACKAGE_ROOT / "SOURCE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def discover_repository_root(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit is not None else [PACKAGE_ROOT, *PACKAGE_ROOT.parents]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve(strict=True)
        if (
            (resolved / "frugalface3d").is_dir()
            and (resolved / "reproducibility/w5b49n_v14_matched_controls_v1/core.py").is_file()
        ):
            return resolved
    raise RuntimeError("canonreg_repository_root_not_found_use_repository_root_argument")


def read_contract() -> dict[str, Any]:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "frugalface3d.review_closure.canonreg.v1"
        or value.get("status") != "SOURCE_ONLY_NO_SCIENTIFIC_RESULT"
    ):
        raise RuntimeError("canonreg_contract_schema_or_status")
    return value


def read_source_manifest() -> dict[str, Any]:
    value = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "frugalface3d.review_closure.canonreg.source_manifest.v1"
        or value.get("status") != "FROZEN_SOURCE_ONLY_NO_PRIVATE_ASSETS"
    ):
        raise RuntimeError("canonreg_source_manifest_schema_or_status")
    return value


def verify_package_manifest() -> dict[str, Any]:
    manifest = read_source_manifest()
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("canonreg_source_manifest_rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("canonreg_source_manifest_row")
        relative = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        if relative in seen or len(expected) != 64:
            raise RuntimeError("canonreg_source_manifest_duplicate_or_hash")
        seen.add(relative)
        path = (PACKAGE_ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(PACKAGE_ROOT)
        except ValueError as error:
            raise RuntimeError("canonreg_source_manifest_path_escape") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"canonreg_source_manifest_hash:{relative}")
    aggregate = json_sha256(rows)
    if manifest.get("ordered_file_rows_sha256") != aggregate:
        raise RuntimeError("canonreg_source_manifest_aggregate")
    return manifest


def verify_upstream_sources(repository_root: Path, contract: Mapping[str, Any]) -> int:
    rows = contract.get("upstream_source_lock")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("canonreg_upstream_source_lock")
    count = 0
    for row in rows:
        relative = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        path = (repository_root / relative).resolve(strict=True)
        try:
            path.relative_to(repository_root)
        except ValueError as error:
            raise RuntimeError("canonreg_upstream_source_escape") from error
        if len(expected) != 64 or sha256_file(path) != expected:
            raise RuntimeError(f"canonreg_upstream_source_hash:{relative}")
        count += 1
    return count


def load_historical_core(repository_root: Path, contract: Mapping[str, Any]) -> ModuleType:
    verify_upstream_sources(repository_root, contract)
    module_name = "_frugalface3d_canonreg_locked_v14_core"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    path = repository_root / "reproducibility/w5b49n_v14_matched_controls_v1/core.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("canonreg_historical_core_import_spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bound_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def package_binding() -> dict[str, str]:
    verify_package_manifest()
    return {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "ordered_source_rows_sha256": read_source_manifest()["ordered_file_rows_sha256"],
    }


__all__ = [
    "CONTRACT_PATH",
    "PACKAGE_ROOT",
    "SOURCE_MANIFEST_PATH",
    "bound_file",
    "canonical_json_bytes",
    "discover_repository_root",
    "json_sha256",
    "load_historical_core",
    "package_binding",
    "read_contract",
    "read_source_manifest",
    "sha256_file",
    "verify_package_manifest",
    "verify_upstream_sources",
    "write_json",
]
