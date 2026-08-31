#!/usr/bin/env python3
"""Static and optional runtime audit for the CanonReg upload source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from integrity import (
    PACKAGE_ROOT,
    discover_repository_root,
    load_historical_core,
    read_contract,
    read_source_manifest,
    verify_package_manifest,
    verify_upstream_sources,
)


REQUIRED_PACKAGE_FILES = {
    "README.md",
    "contract.json",
    "requirements.txt",
    "integrity.py",
    "canonreg_loss.py",
    "run.py",
    "source_check.py",
    "test_loss.py",
    "test_smoke.py",
    "test_local.py",
}
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".txt"}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".zip",
    ".tar",
    ".gz",
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"canonreg_source_check:{label}")


def run_check(repository_root: Path | None, *, runtime: bool) -> dict[str, Any]:
    root = discover_repository_root(repository_root)
    contract = read_contract()
    manifest = verify_package_manifest()
    manifest_files = {str(row["path"]) for row in manifest["files"]}
    require(manifest_files == REQUIRED_PACKAGE_FILES, "manifest_file_keyspace")
    upstream_count = verify_upstream_sources(root, contract)
    actual_files = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "SOURCE_MANIFEST.json"
    }
    require(actual_files == REQUIRED_PACKAGE_FILES, "unexpected_or_missing_files")
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        require(path.suffix.lower() in ALLOWED_SUFFIXES, f"allowed_suffix:{path.name}")
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"private_suffix:{path.name}")
        require(path.stat().st_size < 2_000_000, f"oversized_source:{path.name}")
    training = contract["training"]
    variant = contract["variant"]
    require(tuple(contract["seed_plan"]) == tuple(range(2026080447, 2026080452)), "seeds")
    require(variant["architecture_change"] is False, "architecture_unchanged")
    require(variant["trainable_parameters"] == 89_386, "parameters")
    require(variant["changed_terms_only"] == ["bounded_residual_L2", "total_variation"], "changed_terms")
    require(training["optimizer"] == "AdamW", "optimizer")
    require(training["steps_per_unit"] == 512, "steps")
    require(training["learning_rate"] == 5e-4, "learning_rate")
    require(training["weight_decay"] == 1e-4, "weight_decay")
    require(training["training_units"] == 5, "training_units")
    require(training["automatic_retry"] is False, "no_retry")
    require(training["checkpoint_selection"] is False, "no_selection")
    formal_environment = contract["formal_environment"]
    require(formal_environment["python_version"] == "3.10.20", "python_version")
    require(formal_environment["torch_version"] == "2.4.0+cu121", "torch_version")
    require(formal_environment["cuda_version"] == "12.1", "cuda_version")
    require(formal_environment["cudnn_version"] == 90100, "cudnn_version")
    require(formal_environment["compute_capability"] == [8, 9], "compute_capability")
    private_inputs = contract["private_inputs"]
    expected_private_hashes = {
        "fit_cache": (
            "c407b911230ef749afeca7c0b1a571a67869a92200e5ef8698be3927068610ec",
            "98db0b5962f8505c84f15df74a3302e30210df8eb4d5493c5894a971a40917fc",
        ),
        "facescape_eval_cache": (
            "b8ac441086a519295b1deda0afbe397ea7c420d8c827b3e9bc5882fb8dfcd278",
            "5fa2e9b576561e0751058bd72758a90d6db583d5fd08aad30642302860fbbeb4",
        ),
        "realy_eval_cache": (
            "f61f4ca6868267ca5d3fa5f45bde90c9412554c5bbbff5393f46e048e0e957a7",
            "7981a40b74a8e4828a001aef22c37dc0404944594943bd8b27bd79ffb27d1562",
        ),
    }
    require(
        all(
            (
                private_inputs[name]["manifest_sha256"],
                private_inputs[name]["tensor_sha256"],
            )
            == hashes
            for name, hashes in expected_private_hashes.items()
        ),
        "private_cache_hash_bindings",
    )
    loss_source = (PACKAGE_ROOT / "canonreg_loss.py").read_text(encoding="utf-8")
    run_source = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    require("canonical_mask * (1.0 - visibility)" in loss_source, "lres_mask")
    require("canonical_mask[:, :, :, 1:] * canonical_mask[:, :, :, :-1]" in loss_source, "tv_x_edges")
    require("canonical_mask[:, :, 1:, :] * canonical_mask[:, :, :-1, :]" in loss_source, "tv_y_edges")
    require("--steps" not in run_source, "no_steps_override")
    require("retry" not in {action.dest for action in _parser_actions()}, "no_retry_cli")
    runtime_result = None
    if runtime:
        core = load_historical_core(root, contract)
        device = __import__("torch").device("cpu")
        model = core.new_structure_model(device=device, trainable=True)
        require(core.parameter_count(model) == 89_386, "runtime_parameter_count")
        runtime_result = {
            "torch_version": str(__import__("torch").__version__),
            "trainable_parameters": core.parameter_count(model),
        }
    return {
        "status": "PASS_CANONREG_SOURCE_CHECK",
        "package_files_verified": len(manifest_files),
        "upstream_files_verified": upstream_count,
        "private_data_or_weight_files": 0,
        "private_cache_hash_bindings": 3,
        "training_units": 5,
        "optimizer_steps": 2560,
        "runtime": runtime_result,
        "scientific_result_generated": False,
    }


def _parser_actions() -> list[Any]:
    import run

    actions: list[Any] = list(run.parser()._actions)
    for action in run.parser()._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                actions.extend(subparser._actions)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    result = run_check(args.repository_root, runtime=bool(args.runtime))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
