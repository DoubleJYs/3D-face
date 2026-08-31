#!/usr/bin/env python3
"""Regenerate the deterministic V15 package SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and "rebuilt_public_outputs" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    )
    payload = "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files)
    temporary = MANIFEST.with_suffix(".txt.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(MANIFEST)
    print(f"PASS: wrote {len(files)} V15 package hashes")


if __name__ == "__main__":
    main()
