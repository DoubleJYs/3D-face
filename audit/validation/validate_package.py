#!/usr/bin/env python3
"""Validate V15 audit-package integrity, anonymity, scope, and statistics."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".jsonl", ".py", ".sha256"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".npz", ".npy", ".pyc"}
FORBIDDEN_SYSTEM_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
FACE_IMAGE_MARKERS = {"high_support", "real_face", "identity_face", "display_derivative"}
PATTERNS = {
    "local_or_cloud_absolute_path": re.compile(r"/(?:Users|root|autodl-fs|autodl-tmp)/", re.IGNORECASE),
    "local_user_identifier": re.compile(r"zhangjiyan|张继岩|彦彦", re.IGNORECASE),
    "credential_assignment": re.compile(
        r"(?:api[_-]?key|password|private[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[^\s,'\"]+",
        re.IGNORECASE,
    ),
    "input_binding_path_or_file": re.compile(r"input_binding(?:\.v\d+)?\.json|binding_path", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def equal_float(left: str, right: str, tolerance: float = 1e-12) -> bool:
    if left == "" or right == "":
        return left == right
    return abs(float(left) - float(right)) <= tolerance


def png_dimensions(path: Path) -> tuple[int, int] | None:
    """Return PNG dimensions without adding a Pillow dependency to validation."""
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])


def dimensions_within_tolerance(
    observed: tuple[int, int] | None,
    reference: tuple[int, int] | None,
    relative_tolerance: float = 0.08,
) -> bool:
    if observed is None or reference is None:
        return False
    return all(
        expected > 0 and abs(actual - expected) / expected <= relative_tolerance
        for actual, expected in zip(observed, reference)
    )


def main() -> None:
    failures: list[str] = []
    manifest = ROOT / "SHA256SUMS.txt"
    listed: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        listed.add(path)
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected:
            failures.append(f"hash:{relative}")

    all_files = {path for path in ROOT.rglob("*") if path.is_file()}
    for path in sorted(all_files):
        relative = path.relative_to(ROOT)
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            failures.append(f"python_cache:{relative}")

    actual = {
        path
        for path in all_files
        if path.is_file()
        and path != manifest
        and "rebuilt_public_outputs" not in path.parts
    }
    if listed != actual:
        for path in sorted(actual - listed):
            failures.append(f"unlisted:{path.relative_to(ROOT)}")
        for path in sorted(listed - actual):
            failures.append(f"listed_but_absent:{path.relative_to(ROOT)}")

    for path in sorted(actual):
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_SYSTEM_NAMES or path.name.startswith("._") or "__MACOSX" in path.parts:
            failures.append(f"system_metadata:{relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"model_or_array:{relative}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if "reference_outputs" not in path.parts:
                failures.append(f"image_outside_reference_outputs:{relative}")
            if any(marker in path.name.lower() for marker in FACE_IMAGE_MARKERS):
                failures.append(f"face_image:{relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == "validate_package.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}:{relative}")

    sap = ROOT / "protocol" / "STATISTICAL_ANALYSIS_PLAN.md"
    sap_digest = ROOT / "provenance" / "SAP.sha256"
    if sap.is_file() and sap_digest.is_file():
        expected, relative = sap_digest.read_text(encoding="utf-8").strip().split("  ", 1)
        if relative != "protocol/STATISTICAL_ANALYSIS_PLAN.md" or sha256(sap) != expected:
            failures.append("statistical_analysis_plan_digest")
    else:
        failures.append("missing_statistical_analysis_plan_or_digest")

    frozen_core_path = ROOT / "provenance" / "FROZEN_V14_CORE_SHA256.json"
    if not frozen_core_path.is_file():
        failures.append("missing_frozen_v14_core_manifest")
    else:
        frozen_core = json.loads(frozen_core_path.read_text(encoding="utf-8"))
        frozen_files = frozen_core.get("files", {})
        if frozen_core.get("core_file_count") != 16 or len(frozen_files) != 16:
            failures.append("frozen_v14_core_count")
        for relative, expected in frozen_files.items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                failures.append(f"frozen_v14_core_hash:{relative}")

    family_path = ROOT / "statistics" / "FAMILY_RESULTS.csv"
    family_rows = csv_rows(family_path)
    if len(family_rows) != 18:
        failures.append("formal_comparison_count")

    complete_path = ROOT / "statistics" / "CONFIRMATORY_COMPARISONS_COMPLETE.csv"
    complete_rows = csv_rows(complete_path)
    required_complete_columns = {
        "family_id", "comparison_id", "metric_id", "dataset_id", "comparator_id",
        "identity_count", "median_identity_effect", "q1_identity_effect",
        "q3_identity_effect", "iqr_identity_effect", "ci95_low", "ci95_high",
        "n_positive", "n_zero", "n_negative", "p_raw_two_sided_exact_sign",
        "p_holm_within_family", "median_identity_relative_effect",
        "median_identity_relative_effect_percent", "median_identity_effect_8bit_rgb",
        "bootstrap_seed", "bootstrap_resamples", "inference_conditioning",
    }
    if len(complete_rows) != 18:
        failures.append("complete_confirmatory_comparison_count")
    if complete_rows and not required_complete_columns.issubset(complete_rows[0]):
        failures.append("complete_confirmatory_reporting_fields")
    formal_by_id = {row["comparison_id"]: row for row in family_rows}
    complete_by_id = {row["comparison_id"]: row for row in complete_rows}
    if formal_by_id.keys() != complete_by_id.keys():
        failures.append("complete_confirmatory_comparison_ids")
    for comparison_id in sorted(formal_by_id.keys() & complete_by_id.keys()):
        formal = formal_by_id[comparison_id]
        complete = complete_by_id[comparison_id]
        for field in (
            "family_id", "metric_id", "dataset_id", "comparator_id", "identity_count",
            "n_positive", "n_zero", "n_negative", "relative_effect_defined_identity_count",
        ):
            if formal[field] != complete[field]:
                failures.append(f"complete_confirmatory_field:{comparison_id}:{field}")
        for field in (
            "median_identity_effect", "median_identity_relative_effect", "ci95_low", "ci95_high",
            "p_raw_two_sided_exact_sign", "p_holm_within_family",
        ):
            if not equal_float(formal[field], complete[field]):
                failures.append(f"complete_confirmatory_value:{comparison_id}:{field}")

    complete_identity_rows = csv_rows(ROOT / "statistics" / "CONFIRMATORY_IDENTITY_EFFECTS.csv")
    if len(complete_identity_rows) != 1077:
        failures.append("complete_confirmatory_identity_effect_count")
    if any(not re.fullmatch(r"(?:D1-I\d{3}|D2-\d{3})", row["identity_token"]) for row in complete_identity_rows):
        failures.append("complete_confirmatory_identity_token_scope")

    support_path = ROOT / "robustness" / "FACESCAPE_H_SUPPORT_SENSITIVITY.csv"
    support_rows = csv_rows(support_path)
    required_thresholds = {1, 5, 10, 20, 50}
    required_comparators = {"condition0", "b_lite_ft", "freeuv_conserved"}
    observed_pairs = {
        (int(row["support_threshold_texels"]), row["comparator_id"])
        for row in support_rows
    }
    expected_pairs = {(threshold, comparator) for threshold in required_thresholds for comparator in required_comparators}
    if len(support_rows) != 15 or observed_pairs != expected_pairs:
        failures.append("facescape_support_threshold_design")
    if any(
        row["dataset_id"] != "D1"
        or row["metric_id"] != "hidden_uv_mae"
        or row["new_p_value_generated"] != "False"
        or row["holm_recalculated"] != "False"
        for row in support_rows
    ):
        failures.append("facescape_support_confirmatory_boundary")
    for row in support_rows:
        if int(row["support_threshold_texels"]) != 1:
            continue
        formal = complete_by_id.get(row["comparison_id"])
        if formal is None or not equal_float(row["median_identity_effect"], formal["median_identity_effect"]):
            failures.append(f"facescape_support_threshold1_binding:{row['comparison_id']}")

    direction_plan = ROOT / "protocol" / "REALY_DIRECTIONAL_EXPLORATORY_ANALYSIS_PLAN.md"
    direction_binding_path = ROOT / "provenance" / "REALY_DIRECTIONAL_ANALYSIS_BINDING_RECEIPT.json"
    direction_support_path = ROOT / "robustness" / "REALY_DIRECTION_SUPPORT_SUMMARY.csv"
    direction_distribution_path = ROOT / "robustness" / "REALY_HA_DISTRIBUTION_SUMMARY.json"
    direction_effect_path = ROOT / "robustness" / "REALY_DIRECTION_EFFECT_SUMMARY.csv"
    direction_identity_path = ROOT / "robustness" / "REALY_DIRECTION_IDENTITY_EFFECTS.csv"
    direction_recompute_path = ROOT / "robustness" / "recompute_realy_directional_summary.py"
    direction_validation_path = ROOT / "validation" / "REALY_DIRECTIONAL_VALIDATION_RECEIPT.json"
    direction_independent_path = ROOT / "validation" / "REALY_DIRECTIONAL_INDEPENDENT_VALIDATION_RECEIPT.json"

    expected_directions = {
        ("V01", "V02"), ("V01", "V03"), ("V01", "V04"),
        ("V02", "V01"), ("V02", "V03"), ("V02", "V04"),
        ("V03", "V01"), ("V03", "V02"), ("V03", "V04"),
        ("V04", "V01"), ("V04", "V02"), ("V04", "V03"),
    }
    expected_direction_comparators = {"condition0", "b_lite_ft", "freeuv_conserved"}
    direction_support_rows = csv_rows(direction_support_path)
    observed_support_directions = {
        (row["source_view"], row["target_view"]) for row in direction_support_rows
    }
    if len(direction_support_rows) != 12 or observed_support_directions != expected_directions:
        failures.append("realy_direction_support_grid")
    if any(
        int(row["pair_count"]) != 100
        or int(row["identity_count"]) != 100
        or int(row["h_count"]) != 100
        or int(row["a_count"]) != 100
        or int(row["h_over_a_count"]) != 100
        for row in direction_support_rows
    ):
        failures.append("realy_direction_support_coverage")

    direction_effect_rows = csv_rows(direction_effect_path)
    observed_effect_grid = {
        (row["source_view"], row["target_view"], row["comparator_id"])
        for row in direction_effect_rows
    }
    expected_effect_grid = {
        (source, target, comparator)
        for source, target in expected_directions
        for comparator in expected_direction_comparators
    }
    if len(direction_effect_rows) != 36 or observed_effect_grid != expected_effect_grid:
        failures.append("realy_direction_effect_grid")
    prohibited_direction_fields = {
        field
        for field in (direction_effect_rows[0].keys() if direction_effect_rows else [])
        if re.search(r"(?:^p(?:$|_)|holm|significance)", field, re.IGNORECASE)
    }
    if prohibited_direction_fields:
        failures.append("realy_direction_prohibited_inference_fields")
    if any(
        row["metric_id"] != "hidden_uv_mae"
        or row["effect_definition"] != "comparator_minus_full"
        or row["positive_favors"] != "FrugalFace3D-Lite"
        or int(row["pair_count"]) != 100
        or int(row["identity_count"]) != 100
        or int(row["bootstrap_resamples"]) != 10000
        or row["analysis_status"] != "EXPLORATORY_NO_SIGNIFICANCE_TEST"
        for row in direction_effect_rows
    ):
        failures.append("realy_direction_effect_boundary")

    direction_identity_rows = csv_rows(direction_identity_path)
    expected_identity_fields = {
        "direction_index", "direction", "source_view", "target_view",
        "comparison_id", "comparator_id", "comparator_label", "identity_token",
        "identity_effect", "identity_effect_rgb8",
        "seed_2026080447_effect", "seed_2026080448_effect", "seed_2026080449_effect",
        "seed_2026080450_effect", "seed_2026080451_effect",
    }
    observed_identity_fields = set(direction_identity_rows[0]) if direction_identity_rows else set()
    if observed_identity_fields != expected_identity_fields:
        failures.append("realy_direction_public_identity_schema")
    if observed_identity_fields & {
        "pair_id", "hidden_support_texels", "all_target_visible_support_texels", "h_over_a_ratio"
    }:
        failures.append("realy_direction_private_fields_present")
    if any(re.search(r"path|file|uri", field, re.IGNORECASE) for field in observed_identity_fields):
        failures.append("realy_direction_path_fields_present")
    if len(direction_identity_rows) != 3600:
        failures.append("realy_direction_identity_row_count")
    if any(re.fullmatch(r"D2-\d{3}", row["identity_token"]) is None for row in direction_identity_rows):
        failures.append("realy_direction_identity_token_scope")
    if any(
        re.search(r"/(?:Users|root|autodl-fs|autodl-tmp)/|\\\\|\.\./", value, re.IGNORECASE)
        for row in direction_identity_rows
        for value in row.values()
    ):
        failures.append("realy_direction_identity_path_value")
    direction_identity_groups: dict[tuple[str, str, str], set[str]] = {}
    for row in direction_identity_rows:
        key = (row["source_view"], row["target_view"], row["comparator_id"])
        direction_identity_groups.setdefault(key, set()).add(row["identity_token"])
    if set(direction_identity_groups) != expected_effect_grid or any(
        len(identities) != 100 for identities in direction_identity_groups.values()
    ):
        failures.append("realy_direction_identity_coverage")

    direction_binding = json.loads(direction_binding_path.read_text(encoding="utf-8"))
    if (
        direction_binding.get("status") != "PASS_PATH_FREE_BINDING_TO_FROZEN_RESULTS"
        or direction_binding.get("analysis_plan_sha256") != sha256(direction_plan)
        or direction_binding.get("formal_archive", {}).get("sha256")
        != "41d62c40ec3c7959c91eeb896da3a9895128734479413ab42377b2425312e665"
        or direction_binding.get("public_redaction", {}).get("pair_identifiers_included") is not False
        or direction_binding.get("public_redaction", {}).get("per_pair_support_included") is not False
        or direction_binding.get("public_redaction", {}).get("absolute_paths_included") is not False
        or direction_binding.get("public_redaction", {}).get("public_identity_effects_sha256")
        != sha256(direction_identity_path)
    ):
        failures.append("realy_direction_path_free_binding")
    bound_outputs = direction_binding.get("public_outputs_sha256", {})
    for label, path in {
        "direction_support_summary": direction_support_path,
        "ha_distribution_summary": direction_distribution_path,
        "direction_effect_summary": direction_effect_path,
        "direction_identity_effects": direction_identity_path,
        "public_recompute_program": direction_recompute_path,
        "validation_receipt": direction_validation_path,
        "independent_validation_receipt": direction_independent_path,
    }.items():
        if bound_outputs.get(label) != sha256(path):
            failures.append(f"realy_direction_bound_output:{label}")

    direction_validation = json.loads(direction_validation_path.read_text(encoding="utf-8"))
    direction_independent = json.loads(direction_independent_path.read_text(encoding="utf-8"))
    if (
        direction_validation.get("status")
        != "PASS_ALL_12_REALY_DIRECTIONS_AND_EXPLORATORY_EFFECTS_VALIDATED"
        or direction_validation.get("identity_effect_row_count") != 3600
        or direction_validation.get("effect_summary_row_count") != 36
        or direction_validation.get("new_significance_tests_computed") is not False
        or direction_validation.get("new_multiple_comparison_corrections_computed") is not False
    ):
        failures.append("realy_direction_validation_receipt")
    if (
        direction_independent.get("status")
        != "PASS_INDEPENDENT_RECOMPUTATION_FROM_FROZEN_PAIR_METRICS"
        or direction_independent.get("identity_effect_rows_recomputed") != 3600
        or direction_independent.get("direction_effect_rows_recomputed") != 36
        or direction_independent.get("new_significance_tests_computed") is not False
        or direction_independent.get("new_multiple_comparison_corrections_computed") is not False
    ):
        failures.append("realy_direction_independent_receipt")

    derivation_receipt = json.loads(
        (ROOT / "validation" / "P0_STATS_VALIDATION_RECEIPT.json").read_text(encoding="utf-8")
    )
    if derivation_receipt.get("status") != "PASS_ALL_P0_STATISTICAL_CHECKS":
        failures.append("p0_derivation_receipt_status")
    checks = derivation_receipt.get("checks", {})
    if len(checks) != 215 or not all(checks.values()):
        failures.append("p0_derivation_receipt_checks")
    expected_counts = {
        "confirmatory_comparisons": 18,
        "confirmatory_identity_effects": 1077,
        "facescape_h_support_sensitivity": 15,
    }
    for label, expected in expected_counts.items():
        if derivation_receipt.get("row_counts", {}).get(label) != expected:
            failures.append(f"p0_derivation_receipt_count:{label}")

    registry = json.loads((ROOT / "robustness" / "DERIVED_OUTPUT_REGISTRY.json").read_text(encoding="utf-8"))
    support_registry = next(
        (entry for entry in registry.get("entries", []) if entry.get("file") == "FACESCAPE_H_SUPPORT_SENSITIVITY.csv"),
        None,
    )
    if (
        support_registry is None
        or support_registry.get("status") != "included_and_independently_validated"
        or support_registry.get("sha256") != sha256(support_path)
        or support_registry.get("new_p_values") is not False
        or support_registry.get("holm_recalculated") is not False
    ):
        failures.append("facescape_support_registry")
    realy_support_registry = next(
        (
            entry
            for entry in registry.get("entries", [])
            if entry.get("file") == "REALY_H_SUPPORT_SENSITIVITY.csv"
        ),
        None,
    )
    if (
        realy_support_registry is None
        or realy_support_registry.get("sha256")
        != sha256(ROOT / "robustness" / "REALY_H_SUPPORT_SENSITIVITY.csv")
        or realy_support_registry.get("figure_binding") != "Supplementary Figure S1"
        or realy_support_registry.get("confirmatory") is not False
    ):
        failures.append("realy_support_registry")
    direction_registry = next(
        (entry for entry in registry.get("entries", []) if entry.get("file") == "REALY_DIRECTION_EFFECT_SUMMARY.csv"),
        None,
    )
    if (
        direction_registry is None
        or direction_registry.get("status") != "included_and_independently_recomputable"
        or direction_registry.get("sha256") != sha256(direction_effect_path)
        or direction_registry.get("identity_effect_sha256") != sha256(direction_identity_path)
        or direction_registry.get("identity_effect_row_count") != 3600
        or direction_registry.get("pair_identifiers_included") is not False
        or direction_registry.get("per_pair_support_included") is not False
        or direction_registry.get("new_p_values") is not False
        or direction_registry.get("holm_recalculated") is not False
    ):
        failures.append("realy_direction_registry")
    reporting_registry = registry.get("complete_confirmatory_reporting", {})
    if (
        reporting_registry.get("comparison_count") != 18
        or reporting_registry.get("sha256") != sha256(complete_path)
        or reporting_registry.get("confirmatory_design_changed") is not False
    ):
        failures.append("complete_confirmatory_registry")

    figure_manifest = json.loads((ROOT / "figures" / "source_data" / "v15_figure_manifest.json").read_text(encoding="utf-8"))
    for name, expected in figure_manifest["source_sha256"].items():
        path = ROOT / "figures" / "source_data" / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"figure_source_hash:{name}")
    for name, expected in figure_manifest["asset_sha256"].items():
        if not name.endswith(".png") or "high_support_real_faces" in name:
            continue
        path = ROOT / "figures" / "reference_outputs" / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"reference_figure_hash:{name}")

    four_format_stems = {
        "v15_visibility_region_evaluation_zh",
        "v15_multimetric_effects_zh",
    }
    four_format_suffixes = {".png", ".tiff", ".pdf", ".svg"}
    expected_four_format_files = {
        f"{stem}{suffix}"
        for stem in four_format_stems
        for suffix in four_format_suffixes
    }
    for name in expected_four_format_files:
        path = ROOT / "figures" / "reference_outputs" / name
        expected = figure_manifest["asset_sha256"].get(name)
        if expected is None or not path.is_file() or sha256(path) != expected:
            failures.append(f"four_format_reference_asset:{name}")
    actual_auxiliary_reference_files = {
        path.name
        for path in (ROOT / "figures" / "reference_outputs").iterdir()
        if path.is_file() and path.suffix.lower() != ".png"
    }
    expected_auxiliary_reference_files = {
        name for name in expected_four_format_files if not name.endswith(".png")
    }
    if actual_auxiliary_reference_files != expected_auxiliary_reference_files:
        failures.append("four_format_reference_asset_set")
    active_renderer = ROOT / "figures" / "scripts" / "make_v15_figures.py"
    expected_renderers = {
        "main": ROOT / "figures" / "scripts" / "make_v15_figures.py",
        "additional": ROOT / "figures" / "scripts" / "make_v15_additional_figures.py",
        "robustness": ROOT / "figures" / "scripts" / "make_v15_robustness_figure.py",
        "entry_point": ROOT / "figures" / "scripts" / "rebuild_public_figures.py",
    }
    observed_renderers = figure_manifest.get("active_renderers", {})
    for label, path in expected_renderers.items():
        if observed_renderers.get(label) != {
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }:
            failures.append(f"active_renderer_hash:{label}")

    expected_active_figure_mapping = {
        "Figure 1": "v15_model_architecture_zh.png",
        "Figure 2": "v15_visibility_region_evaluation_zh.png",
        "Figure 3": "v15_output_form_comparison_zh.png",
        "Figure 4": "v15_multimetric_effects_zh.png",
        "Figure 5": "v15_quality_resource_bubbles_zh.png",
        "Supplementary Figure S1": "v15_realy_support_sensitivity_zh.png",
        "Supplementary Figure S2": "v15_realy_12direction_effects_zh.png",
        "Supplementary Figure S3": "v15_descriptive_multimetric_panorama_zh.png",
    }
    if figure_manifest.get("active_manuscript_mapping") != expected_active_figure_mapping:
        failures.append("active_manuscript_figure_mapping")
    expected_registry_figure_bindings = {
        item: f"../figures/reference_outputs/{asset}"
        for item, asset in expected_active_figure_mapping.items()
    }
    if registry.get("active_figure_bindings") != expected_registry_figure_bindings:
        failures.append("active_registry_figure_mapping")
    p0_display_sync = registry.get("reader_layer_sync", {})
    if (
        p0_display_sync.get("active_renderer_sha256") != sha256(active_renderer)
        or p0_display_sync.get("licensed_face_items_redistributed") is not False
        or p0_display_sync.get("Figure 2", {}).get("source_sha256")
        != sha256(ROOT / "figures" / "source_data" / "v15_visibility_region_compression.csv")
        or p0_display_sync.get("Figure 4", {}).get("source_sha256")
        != sha256(ROOT / "figures" / "source_data" / "v15_multimetric_effects.csv")
    ):
        failures.append("reader_layer_registry")
    for item, stem in (
        ("Figure 2", "v15_visibility_region_evaluation_zh"),
        ("Figure 4", "v15_multimetric_effects_zh"),
    ):
        expected_assets = {
            suffix.lstrip("."): figure_manifest["asset_sha256"][f"{stem}{suffix}"]
            for suffix in (".png", ".tiff", ".pdf", ".svg")
        }
        if p0_display_sync.get(item, {}).get("asset_sha256") != expected_assets:
            failures.append(f"reader_layer_registry_assets:{item}")
    if figure_manifest.get("licensed_face_items_excluded") != {
        "Figure 6": "not redistributed",
        "Supplementary Figure S4": "not redistributed",
    }:
        failures.append("licensed_face_items_excluded")
    public_scope = json.loads(
        (ROOT / "provenance" / "PUBLIC_SCOPE.json").read_text(encoding="utf-8")
    )
    if (
        public_scope.get("public_nonface_figure_count") != 8
        or public_scope.get("public_nonface_manuscript_items")
        != list(expected_active_figure_mapping)
        or public_scope.get("four_format_reference_asset_items") != ["Figure 2", "Figure 4"]
        or public_scope.get("licensed_face_items_excluded")
        != ["Figure 6", "Supplementary Figure S4"]
        or public_scope.get("figure_6_face_derivatives_redistributed") is not False
        or public_scope.get("supplementary_figure_s4_face_derivatives_redistributed") is not False
        or public_scope.get("frozen_v14_core_file_count") != 16
    ):
        failures.append("public_scope_figure_mapping")
    qa_record = json.loads(
        (ROOT / "validation" / "V15_PACKAGE_QA.json").read_text(encoding="utf-8")
    )
    qa_checks = qa_record.get("checks", {})
    if (
        qa_checks.get("active_main_figure_mapping")
        != {key: value for key, value in expected_active_figure_mapping.items() if key.startswith("Figure")}
        or qa_checks.get("active_supplementary_figure_mapping")
        != {
            key: value
            for key, value in expected_active_figure_mapping.items()
            if key.startswith("Supplementary")
        }
        or qa_checks.get("retired_adaptation_tradeoff_active_mapping_absent") is not True
        or qa_checks.get("licensed_figure_6_and_s4_excluded") is not True
        or qa_checks.get("reader_layer_sync_verified") is not True
        or qa_checks.get("reader_layer_condition0_name") != "无显式条件残差（NoCond）"
        or qa_checks.get("retired_reader_layer_condition0_name_absent") is not True
        or qa_checks.get("four_format_reference_asset_items") != ["Figure 2", "Figure 4"]
        or qa_checks.get("reference_nonface_file_count") != 14
        or qa_checks.get("four_format_reference_file_count") != 8
        or qa_checks.get("figure_2_h_over_a_notation_verified") is not True
        or qa_checks.get("figure_2_fixed_baseline_name_verified") is not True
        or qa_checks.get("figure_2_zits_marker_margin_verified") is not True
        or qa_checks.get("figure_2_and_4_internal_b_lite_ft_abbreviation_absent") is not True
        or qa_checks.get("figure_4_duplicated_procedural_footer_absent") is not True
        or qa_checks.get("figure_6_face_derivatives_redistributed") is not False
        or qa_checks.get("supplementary_figure_s4_face_derivatives_redistributed") is not False
        or qa_checks.get("frozen_v14_core_hashes_verified") != 16
        or qa_checks.get("realy_direction_paths_included") is not False
        or qa_checks.get("direction_identity_schema_path_free") is not True
        or qa_checks.get("nonpublic_renderer_cli_disabled") is not True
    ):
        failures.append("qa_figure_mapping")
    robustness_source_hashes = figure_manifest.get("robustness_source_sha256", {})
    support_source_key = "robustness/REALY_H_SUPPORT_SENSITIVITY.csv"
    if robustness_source_hashes.get(support_source_key) != sha256(
        ROOT / support_source_key
    ):
        failures.append("realy_support_figure_source_hash")
    expected_reference_pngs = set(expected_active_figure_mapping.values())
    actual_reference_pngs = {
        path.name for path in (ROOT / "figures" / "reference_outputs").glob("*.png")
    }
    if actual_reference_pngs != expected_reference_pngs:
        failures.append("active_reference_png_set")

    additional_figure_design = {
        "v15_descriptive_multimetric_panorama.csv": 56,
        "v15_quality_resource_bubbles.csv": 12,
        "v15_realy_12direction_effects.csv": 36,
    }
    for name, expected_rows in additional_figure_design.items():
        path = ROOT / "figures" / "source_data" / name
        rows = csv_rows(path)
        if len(rows) != expected_rows:
            failures.append(f"additional_figure_row_count:{name}")
    direction_figure_rows = csv_rows(
        ROOT / "figures" / "source_data" / "v15_realy_12direction_effects.csv"
    )
    if any(
        row["analysis_status"] != "EXPLORATORY_NO_SIGNIFICANCE_TEST"
        or row["effect_definition"] != "comparator_minus_full"
        or row["positive_favors"] != "FrugalFace3D-Lite"
        or row["input_csv"] != "robustness/REALY_DIRECTION_EFFECT_SUMMARY.csv"
        for row in direction_figure_rows
    ):
        failures.append("direction_figure_source_boundary")
    if figure_manifest.get("public_nonface_rebuildable_figure_count") != 8:
        failures.append("public_nonface_figure_count")

    output_map_path = ROOT / "mappings" / "OUTPUT_MAP.csv"
    output_rows = {row["manuscript_item"]: row for row in csv_rows(output_map_path)}
    for item, asset_name in expected_active_figure_mapping.items():
        row = output_rows.get(item)
        if row is None or row["reference_output"] != f"figures/reference_outputs/{asset_name}":
            failures.append(f"manuscript_output_map:{item}")
    for item in ("Supplementary Table S9", "Supplementary Table S10"):
        if item not in output_rows:
            failures.append(f"manuscript_output_map:{item}")
    figure6_output = output_rows.get("Figure 6", {})
    figure_s4_output = output_rows.get("Supplementary Figure S4", {})
    if (
        figure6_output.get("public_status") != "not redistributed"
        or figure6_output.get("reference_output") not in (None, "")
        or figure_s4_output.get("public_status") != "not redistributed"
        or figure_s4_output.get("reference_output") not in (None, "")
    ):
        failures.append("licensed_face_output_map_scope")
    expected_source_data_files = {
        "v15_descriptive_multimetric_panorama.csv",
        "v15_figure_manifest.json",
        "v15_multimetric_effects.csv",
        "v15_output_form_effects.csv",
        "v15_quality_resource_bubbles.csv",
        "v15_realy_12direction_effects.csv",
        "v15_visibility_region_compression.csv",
    }
    actual_source_data_files = {
        path.name for path in (ROOT / "figures" / "source_data").iterdir() if path.is_file()
    }
    if actual_source_data_files != expected_source_data_files:
        failures.append("active_figure_source_data_set")

    method_name_rows = {
        row["internal_identifier"]: row["paper_display_name"]
        for row in csv_rows(ROOT / "mappings" / "METHOD_NAME_MAP.csv")
    }
    if method_name_rows.get("condition0") != "无显式条件残差（NoCond）":
        failures.append("reader_layer_condition0_name")
    reader_layer_files = (
        ROOT / "README.md",
        ROOT / "mappings" / "METHOD_NAME_MAP.csv",
        ROOT / "figures" / "source_data" / "v15_descriptive_multimetric_panorama.csv",
        ROOT / "figures" / "scripts" / "make_v15_additional_figures.py",
        ROOT / "figures" / "scripts" / "make_v15_robustness_figure.py",
    )
    for path in reader_layer_files:
        if "无几何—表情条件对照" in path.read_text(encoding="utf-8"):
            failures.append(f"retired_reader_layer_condition0_name:{path.relative_to(ROOT)}")

    for script in ROOT.rglob("*.py"):
        try:
            ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except SyntaxError as exc:
            failures.append(f"python_syntax:{script.relative_to(ROOT)}:{exc.lineno}")

    if not (ROOT / "validation" / "build_anonymous_archive.py").is_file():
        failures.append("anonymous_archive_builder_missing")

    active_v14 = ROOT / "figures" / "scripts" / "make_v14_figures.py"
    historical_renderer = ROOT / "figures" / "scripts" / "_historical_v14_renderer.py"
    if active_v14.exists():
        failures.append("active_v14_figure_builder_present")
    if not historical_renderer.is_file() or "not an active V15 entry point" not in historical_renderer.read_text(encoding="utf-8"):
        failures.append("historical_renderer_not_explicitly_scoped")
    rebuild_text = (ROOT / "figures" / "scripts" / "rebuild_public_figures.py").read_text(encoding="utf-8")
    if "import make_v15_figures as v15" not in rebuild_text or "make_v14_figures" in rebuild_text:
        failures.append("active_figure_entry_version_mapping")
    if (
        "import make_v15_additional_figures as supplemental" not in rebuild_text
        or "supplemental.rebuild_from_public_sources" not in rebuild_text
    ):
        failures.append("supplementary_figure_entry_version_mapping")
    if (
        "import make_v15_robustness_figure as robustness" not in rebuild_text
        or "robustness.rebuild_from_public_source" not in rebuild_text
    ):
        failures.append("support_sensitivity_figure_entry_mapping")
    if (
        "v15.figure_visibility_evaluation" not in rebuild_text
        or "v15.figure_multimetric_effects" not in rebuild_text
    ):
        failures.append("p0_display_rebuild_entry_mapping")
    if "not a public entry point" not in active_renderer.read_text(encoding="utf-8"):
        failures.append("nonpublic_renderer_cli_not_disabled")

    figure2_svg = (
        ROOT / "figures" / "reference_outputs" / "v15_visibility_region_evaluation_zh.svg"
    ).read_text(encoding="utf-8")
    figure4_svg = (
        ROOT / "figures" / "reference_outputs" / "v15_multimetric_effects_zh.svg"
    ).read_text(encoding="utf-8")
    if "B-lite-FT" in figure2_svg or "B-lite-FT" in figure4_svg:
        failures.append("internal_b_lite_ft_abbreviation_in_reference")
    if "|H|/|A|" not in figure2_svg:
        failures.append("figure2_h_over_a_notation")
    if "固定权重 B-lite" not in figure2_svg:
        failures.append("figure2_fixed_baseline_name")
    if "axis.set_xlim(-0.15, 0.25)" not in active_renderer.read_text(encoding="utf-8"):
        failures.append("figure2_zits_marker_margin")
    if "置信区间未作多重校正" in figure4_svg:
        failures.append("figure4_duplicated_procedural_footer")

    if not failures:
        verification = subprocess.run(
            [sys.executable, str(ROOT / "statistics" / "recompute_public_statistics.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if verification.returncode != 0:
            failures.append("public_statistics_recompute:" + verification.stdout + verification.stderr)

    if not failures:
        direction_verification = subprocess.run(
            [sys.executable, str(direction_recompute_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if direction_verification.returncode != 0:
            failures.append(
                "public_directional_recompute:"
                + direction_verification.stdout
                + direction_verification.stderr
            )
        elif "PASS_REALY_DIRECTIONAL_PUBLIC_RECOMPUTATION" not in direction_verification.stdout:
            failures.append("public_directional_recompute_status")

    if not failures:
        with tempfile.TemporaryDirectory(prefix="v15-public-figures-") as temporary_directory:
            rebuilt_root = Path(temporary_directory)
            figure_verification = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "figures" / "scripts" / "rebuild_public_figures.py"),
                    "--output",
                    str(rebuilt_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if figure_verification.returncode != 0:
                failures.append(
                    "public_figure_rebuild:"
                    + figure_verification.stdout
                    + figure_verification.stderr
                )
            else:
                expected_public_pngs = {
                    name
                    for name in figure_manifest["asset_sha256"]
                    if name.endswith(".png") and "high_support_real_faces" not in name
                }
                if len(expected_public_pngs) != 8:
                    failures.append("rebuilt_public_png_count")
                for name in expected_public_pngs:
                    rebuilt = rebuilt_root / name
                    reference = ROOT / "figures" / "reference_outputs" / name
                    if not rebuilt.is_file() or rebuilt.stat().st_size < 1024:
                        failures.append(f"rebuilt_public_png_missing_or_empty:{name}")
                    elif not dimensions_within_tolerance(
                        png_dimensions(rebuilt), png_dimensions(reference)
                    ):
                        failures.append(f"rebuilt_public_png_dimensions:{name}")
                rebuilt_figure2_svg = (
                    rebuilt_root / "v15_visibility_region_evaluation_zh.svg"
                ).read_text(encoding="utf-8")
                rebuilt_figure4_svg = (
                    rebuilt_root / "v15_multimetric_effects_zh.svg"
                ).read_text(encoding="utf-8")
                if "B-lite-FT" in rebuilt_figure2_svg or "B-lite-FT" in rebuilt_figure4_svg:
                    failures.append("internal_b_lite_ft_abbreviation_in_rebuild")
                if "|H|/|A|" not in rebuilt_figure2_svg:
                    failures.append("rebuilt_figure2_h_over_a_notation")
                if "固定权重 B-lite" not in rebuilt_figure2_svg:
                    failures.append("rebuilt_figure2_fixed_baseline_name")
                if "置信区间未作多重校正" in rebuilt_figure4_svg:
                    failures.append("rebuilt_figure4_duplicated_procedural_footer")

    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    print(
        "PASS: exact V15 Figure 1-5/S1/S2/S3 mapping, all 16 immutable V14 core hashes, package "
        "integrity, anonymity and scope checks, all 18 confirmatory comparisons, the 3,600/36 "
        "REALY directional recomputation, and a fresh rebuild of all eight public non-face figures "
        "were verified."
    )


if __name__ == "__main__":
    main()
