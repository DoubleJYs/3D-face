#!/usr/bin/env python3
"""Verify P/O/H/A arithmetic using a tiny deterministic synthetic fixture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def mean_error(prediction: list[float], target: list[float], mask: list[bool]) -> float:
    values = [abs(a - b) for a, b, keep in zip(prediction, target, mask) if keep]
    if not values:
        raise ValueError("smoke mask is empty")
    return sum(values) / len(values)


def rounded(value: float) -> float:
    return round(value, 12)


def main() -> int:
    fixture = json.loads((ROOT / "protocol_fixture.json").read_text(encoding="utf-8"))
    canonical_support = [bool(value) for value in fixture["canonical_support"]]
    source_visible = [bool(value) for value in fixture["source_visibility"]]
    target_visible = [bool(value) for value in fixture["target_visibility"]]
    target = fixture["target_texture"]
    model = fixture["model_texture"]
    lengths = {len(canonical_support), len(source_visible), len(target_visible), len(target), len(model)}
    if lengths != {len(canonical_support)}:
        raise ValueError("fixture arrays have different lengths")

    observed = [m and source and target_v for m, source, target_v in zip(canonical_support, source_visible, target_visible)]
    hidden = [m and not source and target_v for m, source, target_v in zip(canonical_support, source_visible, target_visible)]
    all_visible = [m and target_v for m, target_v in zip(canonical_support, target_visible)]
    preserved = [truth if source else estimate for truth, estimate, source in zip(target, model, source_visible)]

    def metrics(prediction: list[float]) -> tuple[float, float, float]:
        return (
            mean_error(prediction, target, observed),
            mean_error(prediction, target, hidden),
            mean_error(prediction, target, all_visible),
        )

    observed_model, hidden_model, all_model = metrics(model)
    observed_preserved, hidden_preserved, all_preserved = metrics(preserved)
    observed_count, hidden_count, all_count = sum(observed), sum(hidden), sum(all_visible)

    def identity_holds(observed_mae: float, hidden_mae: float, all_mae: float) -> bool:
        reconstructed = (observed_count * observed_mae + hidden_count * hidden_mae) / all_count
        return math.isclose(reconstructed, all_mae, rel_tol=0.0, abs_tol=1e-12)

    output = {
        "all_visible_mae_model": rounded(all_model),
        "all_visible_mae_preserved": rounded(all_preserved),
        "all_visible_texels": all_count,
        "dilution_identity_holds_model": identity_holds(observed_model, hidden_model, all_model),
        "dilution_identity_holds_preserved": identity_holds(observed_preserved, hidden_preserved, all_preserved),
        "hidden_mae_model": rounded(hidden_model),
        "hidden_mae_preserved": rounded(hidden_preserved),
        "hidden_texels": hidden_count,
        "observed_mae_model": rounded(observed_model),
        "observed_mae_preserved": rounded(observed_preserved),
        "observed_texels": observed_count,
        "schema_version": "frugalface3d.visibility_protocol_smoke.output.v1",
        "status": "PASS_VISIBILITY_PROTOCOL_SMOKE",
    }
    payload = canonical(output)
    expected = (ROOT / "expected_output.json").read_bytes()
    expected_digest = (ROOT / "expected_output.sha256").read_text(encoding="utf-8").strip()
    observed_digest = hashlib.sha256(payload).hexdigest()
    if payload != expected or observed_digest != expected_digest:
        print(json.dumps({"status": "VISIBILITY_PROTOCOL_SMOKE_FAILED", "observed_sha256": observed_digest}, sort_keys=True))
        return 1
    print(json.dumps({"status": output["status"], "output_sha256": observed_digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
