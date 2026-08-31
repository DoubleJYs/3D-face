#!/usr/bin/env python3
"""Verify a provider-acquired Mcanon mask without redistributing the asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FILE_SHA256 = "a5069d8ffaf020008ae92d5062c4e98600f723aac4eb869731f190e4630467b5"
EXPECTED_ARRAY_SHA256 = "7a3f9bd59eebcaf3471892c4569f3e4aa5a0510d9d5c003a1ddce3977fd2fd69"
EXPECTED_TRUE_TEXELS = 1515


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype_bytes = array.dtype.str.encode("ascii")
    digest.update(len(dtype_bytes).to_bytes(2, "little"))
    digest.update(dtype_bytes)
    digest.update(array.ndim.to_bytes(2, "little"))
    digest.update(np.asarray(array.shape, dtype=np.dtype("<u8")).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_mask(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(
            handle.convert("L").resize((64, 64), Image.Resampling.NEAREST),
            dtype=np.uint8,
        ) > 127


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.expanduser().resolve(strict=True)
    array = parse_mask(path)
    result = {
        "schema_version": "frugalface3d.v15.mcanon_verification.v1",
        "file_sha256": sha256_file(path),
        "parsed_array_sha256": canonical_array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "true_texels": int(array.sum()),
    }
    result["status"] = (
        "PASS_MCANON_PROVIDER_ASSET_BINDING"
        if result["file_sha256"] == EXPECTED_FILE_SHA256
        and result["parsed_array_sha256"] == EXPECTED_ARRAY_SHA256
        and result["shape"] == [64, 64]
        and result["dtype"] == "bool"
        and result["true_texels"] == EXPECTED_TRUE_TEXELS
        else "FAIL_MCANON_PROVIDER_ASSET_BINDING"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
