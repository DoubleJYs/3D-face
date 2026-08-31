#!/usr/bin/env python3
"""Build and verify a deterministic, metadata-free V15 delivery archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "anonymous_evaluation_package_v15"
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if (
            "rebuilt_public_outputs" in relative.parts
            or "__pycache__" in relative.parts
            or path.suffix.lower() == ".pyc"
            or path.name in EXCLUDED_NAMES
            or path.name.startswith("._")
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def tar_info(name: str, *, directory: bool, size: int = 0, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mode = 0o755 if directory or executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def build(output: Path) -> dict[str, object]:
    files = included_files()
    expected_file_names = {
        f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}" for path in files
    }
    directories = {ARCHIVE_ROOT}
    for path in files:
        relative = PurePosixPath(path.relative_to(ROOT).as_posix())
        for parent in relative.parents:
            if str(parent) != ".":
                directories.add(f"{ARCHIVE_ROOT}/{parent.as_posix()}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v15-anonymous-archive-") as temporary_name:
        raw_tar = Path(temporary_name) / "package.tar"
        with tarfile.open(raw_tar, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(directories, key=lambda value: (value.count("/"), value)):
                archive.addfile(tar_info(name, directory=True))
            for path in files:
                name = f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}"
                executable = bool(path.stat().st_mode & 0o111)
                with path.open("rb") as handle:
                    archive.addfile(
                        tar_info(name, directory=False, size=path.stat().st_size, executable=executable),
                        fileobj=handle,
                    )
        temporary_output = Path(temporary_name) / output.name
        with raw_tar.open("rb") as source, temporary_output.open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(chunk)
        os.replace(temporary_output, output)

    with tarfile.open(output, mode="r:gz") as archive:
        members = archive.getmembers()
    regular_names = {member.name for member in members if member.isfile()}
    failures = []
    if regular_names != expected_file_names:
        failures.append("archive_file_set")
    for member in members:
        parts = PurePosixPath(member.name).parts
        if (
            not parts
            or parts[0] != ARCHIVE_ROOT
            or member.name.startswith("/")
            or ".." in parts
        ):
            failures.append(f"unsafe_member:{member.name}")
        if any(part.startswith("._") or part == "__MACOSX" for part in parts):
            failures.append(f"system_metadata:{member.name}")
        if member.uid != 0 or member.gid != 0 or member.uname or member.gname:
            failures.append(f"owner_metadata:{member.name}")
        if member.mtime != 0:
            failures.append(f"mtime:{member.name}")
        if any("xattr" in key.lower() or "provenance" in key.lower() for key in member.pax_headers):
            failures.append(f"pax_metadata:{member.name}")
    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    return {
        "status": "PASS_ANONYMOUS_ARCHIVE_CONTAINER",
        "archive": output.name,
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "regular_file_count": len(regular_names),
        "member_count": len(members),
        "uid_gid": "0/0",
        "uname_gname": "empty",
        "mtime": 0,
        "appledouble_members": 0,
        "xattr_or_provenance_headers": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
