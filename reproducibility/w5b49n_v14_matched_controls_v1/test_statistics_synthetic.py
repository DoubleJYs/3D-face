#!/usr/bin/env python3
"""Synthetic tests for the V14 fail-closed statistical analysis.

No synthetic number is a manuscript result. The test uses two identities per
dataset only to exercise the frozen computation and rejection paths quickly.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import analyze_v14_matched_controls as analysis


HERE = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _hash_entry(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": analysis.sha256_file(path)}


def _synthetic_contract(temp_root: Path) -> tuple[dict, Path]:
    contract = copy.deepcopy(json.loads((HERE / "contract.v1.json").read_text(encoding="utf-8")))
    contract["contract_mode"] = "synthetic"
    for dataset_id in ("D1", "D2"):
        contract["datasets"][dataset_id]["expected_identity_count"] = 2
        contract["datasets"][dataset_id]["expected_total_pair_count"] = 4
        contract["datasets"][dataset_id]["expected_analysis_pair_count"] = 4
        contract["datasets"][dataset_id]["sface_min_identity_count"] = 2
    contract["required_artifacts"]["freeuv_v1_2_terminal"].pop("required_bound_sha256", None)
    contract["required_artifacts"]["freeuv_v1_2_terminal"]["required_fields"]["forward_count"] = 8
    source_package = temp_root / "synthetic_freeuv_source_package.bin"
    source_package.write_bytes(b"synthetic-freeuv-v1.2-source-package")
    contract["required_artifacts"]["freeuv_v1_2_terminal"]["required_fields"]["source_package_sha256"] = analysis.sha256_file(source_package)
    _write_json(
        temp_root / "synthetic_freeuv_activity_terminal.json",
        {
            "status": "PASS_F2_FREEUV_D1D2_RAW_OUTPUTS_FROZEN",
            "successful_total_forward_count": 8,
            "automatic_retry": False,
        },
    )
    contract["required_artifacts"]["d1_eval_cache_terminal"]["required_fields"]["analysis_pair_count"] = 4
    contract["required_artifacts"]["d2_eval_cache_terminal"]["required_fields"]["analysis_pair_count"] = 4
    contract["required_artifacts"]["shared_render_terminal"]["required_fields"]["analysis_pair_count"] = 8
    contract["required_artifacts"]["lpips_terminal"]["required_fields"]["analysis_pair_count"] = 8
    contract["required_artifacts"]["sface_terminal"]["required_fields"]["analysis_pair_count"] = 8
    contract_path = temp_root / "contract.synthetic.json"
    _write_json(contract_path, contract)
    return contract, contract_path


def _build_roster(path: Path) -> dict[str, list[tuple[str, str]]]:
    datasets = {}
    eligible = {}
    for dataset_id in ("D1", "D2"):
        rows = []
        pairs = []
        for identity_index in range(2):
            identity = f"{dataset_id}-identity-{identity_index}"
            for pair_index in range(2):
                pair_id = f"{dataset_id}-pair-{identity_index}-{pair_index}"
                rows.append(
                    {
                        "identity_token": identity,
                        "pair_id": pair_id,
                        "analysis_eligible": True,
                        "structural_state": "EVALUABLE",
                    }
                )
                pairs.append((identity, pair_id))
        datasets[dataset_id] = {"rows": rows}
        eligible[dataset_id] = pairs
    _write_json(path, {"schema_version": analysis.PAIR_ROSTER_SCHEMA, "datasets": datasets})
    return eligible


def _metric_value(metric_id: str, method_id: str, pair_serial: int, seed: int | None) -> float:
    seed_offset = 0.0001 * ((seed - analysis.REQUIRED_SEEDS[0]) if seed is not None else 0)
    pair_offset = 0.001 * pair_serial
    if metric_id == "sface_source_to_render_cosine":
        full = 0.85 - pair_offset - seed_offset
        offsets = {"full": 0.0, "condition0": -0.03, "b_lite_ft": -0.02, "freeuv_conserved": -0.05}
        return full + offsets[method_id]
    full = 0.10 + pair_offset + seed_offset
    offsets = {"full": 0.0, "condition0": 0.03, "b_lite_ft": 0.02, "freeuv_conserved": 0.05}
    return full + offsets[method_id]


def _build_metric_rows(path: Path, eligible: dict[str, list[tuple[str, str]]]) -> None:
    lines = []
    for dataset_id, pairs in eligible.items():
        for pair_serial, (identity, pair_id) in enumerate(pairs):
            support = 10 + pair_serial
            for metric_id in analysis.METRIC_ORDER:
                for method_id in analysis.SEEDED_METHODS:
                    for seed in analysis.REQUIRED_SEEDS:
                        lines.append(
                            json.dumps(
                                {
                                    "schema_version": analysis.METRIC_ROW_SCHEMA,
                                    "dataset_id": dataset_id,
                                    "metric_id": metric_id,
                                    "method_id": method_id,
                                    "identity_token": identity,
                                    "pair_id": pair_id,
                                    "seed": seed,
                                    "value": _metric_value(metric_id, method_id, pair_serial, seed),
                                    "support_texels": support,
                                    "terminal_state": "COMPLETE",
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                lines.append(
                    json.dumps(
                        {
                            "schema_version": analysis.METRIC_ROW_SCHEMA,
                            "dataset_id": dataset_id,
                            "metric_id": metric_id,
                            "method_id": "freeuv_conserved",
                            "identity_token": identity,
                            "pair_id": pair_id,
                            "seed": None,
                            "value": _metric_value(metric_id, "freeuv_conserved", pair_serial, None),
                            "support_texels": support,
                            "terminal_state": "COMPLETE",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_binding(temp_root: Path, contract: dict, contract_path: Path) -> tuple[dict, Path, Path]:
    pair_roster_path = temp_root / "pair_roster.json"
    eligible = _build_roster(pair_roster_path)
    metric_rows_path = temp_root / "metric_rows.jsonl"
    _build_metric_rows(metric_rows_path, eligible)

    synthetic_binding_specs = {
        "statistical_analysis_plan": (
            "STATISTICAL_ANALYSIS_PLAN.md",
            "V14-SAP-1.5\nRTX 4090/CUDA\n7,680 步\n10,000 次\n",
        ),
        "claim_decision_matrix": (
            "V14_CLAIM_DECISION_MATRIX.md",
            "V14 主张判定矩阵\n五个预定随机种子\n",
        ),
    }
    synthetic_binding_paths = {}
    for role, (basename, marker_text) in synthetic_binding_specs.items():
        path = temp_root / basename
        path.write_text(marker_text, encoding="utf-8")
        synthetic_binding_paths[role] = path

    shared_files = {}
    for label in ("environment_manifest", "training_split", "training_budget_manifest"):
        path = temp_root / f"{label}.bin"
        path.write_bytes(f"synthetic-{label}".encode("utf-8"))
        shared_files[label] = path

    artifacts = {}
    for role, rule in contract["required_artifacts"].items():
        kind = rule["kind"]
        if role in synthetic_binding_paths:
            artifacts[role] = _hash_entry(synthetic_binding_paths[role])
            continue
        if role == "analysis_program":
            artifacts[role] = _hash_entry(Path(analysis.__file__).resolve())
            continue
        if kind != "json_terminal":
            raise AssertionError(f"unexpected synthetic artifact kind: {kind}")

        payload = dict(rule["required_fields"])
        bound_files = []
        is_training = role.startswith("full_seed_") or role.startswith("condition0_seed_") or role.startswith("b_lite_ft_seed_")
        if is_training:
            checkpoint_path = temp_root / f"{role}.checkpoint"
            checkpoint_path.write_bytes(f"synthetic-{role}-checkpoint".encode("utf-8"))
            payload.update(
                {
                    "device_backend": "cuda",
                    "device_name": "NVIDIA GeForce RTX 4090",
                    "training_steps": 512,
                    "environment_manifest_sha256": analysis.sha256_file(shared_files["environment_manifest"]),
                    "training_split_sha256": analysis.sha256_file(shared_files["training_split"]),
                    "training_budget_manifest_sha256": analysis.sha256_file(shared_files["training_budget_manifest"]),
                    "checkpoint_sha256": analysis.sha256_file(checkpoint_path),
                }
            )
            bound_files = [
                _hash_entry(shared_files["environment_manifest"]),
                _hash_entry(shared_files["training_split"]),
                _hash_entry(shared_files["training_budget_manifest"]),
                _hash_entry(checkpoint_path),
            ]
        elif role == "freeuv_v1_2_terminal":
            source_package = temp_root / "synthetic_freeuv_source_package.bin"
            source_terminal = temp_root / "synthetic_freeuv_activity_terminal.json"
            payload["source_terminal_sha256"] = analysis.sha256_file(source_terminal)
            bound_files = [_hash_entry(source_package), _hash_entry(source_terminal)]
        elif role in ("lpips_linux_qualification_terminal", "sface_linux_qualification_terminal"):
            qualification_fields = {
                "lpips_linux_qualification_terminal": [
                    "qualification_script_sha256",
                    "runtime_manifest_sha256",
                    "evaluator_export_sha256",
                    "probe_manifest_sha256",
                ],
                "sface_linux_qualification_terminal": [
                    "qualification_script_sha256",
                    "runtime_manifest_sha256",
                    "detector_model_sha256",
                    "recognizer_model_sha256",
                    "probe_manifest_sha256",
                ],
            }[role]
            for field in qualification_fields:
                field_path = temp_root / f"{role}.{field}.bin"
                field_path.write_bytes(f"synthetic-{role}-{field}".encode("utf-8"))
                payload[field] = analysis.sha256_file(field_path)
                bound_files.append(_hash_entry(field_path))
        elif role in ("lpips_terminal", "sface_terminal"):
            qualification_role = (
                "lpips_linux_qualification_terminal" if role == "lpips_terminal" else "sface_linux_qualification_terminal"
            )
            qualification_path = Path(artifacts[qualification_role]["path"])
            payload["qualification_terminal_sha256"] = analysis.sha256_file(qualification_path)
            result_path = temp_root / f"{role}.result_manifest"
            result_path.write_bytes(f"synthetic-{role}-results".encode("utf-8"))
            bound_files = [_hash_entry(qualification_path), _hash_entry(result_path)]
        else:
            bound_path = temp_root / f"{role}.bound"
            bound_path.write_bytes(f"synthetic-{role}-bound".encode("utf-8"))
            bound_files = [_hash_entry(bound_path)]
        payload["bound_files"] = bound_files
        terminal_path = temp_root / f"{role}.json"
        _write_json(terminal_path, payload)
        artifacts[role] = _hash_entry(terminal_path)

    output_root = temp_root / "analysis_output"
    binding = {
        "schema_version": analysis.BINDING_SCHEMA,
        "status": "FROZEN_COMPLETE",
        "contract_sha256": analysis.sha256_file(contract_path),
        "pair_roster": _hash_entry(pair_roster_path),
        "metric_rows": _hash_entry(metric_rows_path),
        "artifacts": artifacts,
        "output_root": str(output_root.resolve()),
    }
    binding_path = temp_root / "binding.json"
    _write_json(binding_path, binding)
    return binding, binding_path, output_root


class StatisticalFunctionsTest(unittest.TestCase):
    def test_exact_two_sided_sign(self) -> None:
        self.assertEqual(analysis.exact_two_sided_sign_p(0, 0), 1.0)
        self.assertAlmostEqual(analysis.exact_two_sided_sign_p(5, 0), 0.0625)
        self.assertAlmostEqual(analysis.exact_two_sided_sign_p(20, 0), 2.0 / (2**20))
        self.assertEqual(analysis.exact_two_sided_sign_p(3, 3), 1.0)

    def test_holm_step_down(self) -> None:
        adjusted = analysis.holm_adjust([0.01, 0.03, 0.04])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_bootstrap_is_identity_level_and_deterministic(self) -> None:
        first = analysis.identity_bootstrap_interval([0.01, 0.02, 0.03], 10000, 20260816)
        second = analysis.identity_bootstrap_interval([0.01, 0.02, 0.03], 10000, 20260816)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.02)
        self.assertGreaterEqual(first[1], 0.02)


class FailClosedIntegrationTest(unittest.TestCase):
    def test_complete_keyspace_produces_four_families(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            _, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            result = analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertEqual(result.resolve(), output_root.resolve())
            terminal = json.loads((output_root / "ANALYSIS_TERMINAL.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["status"], "PASS_V14_MATCHED_CONTROLS_ANALYSIS_COMPLETE")
            self.assertEqual(terminal["family_ids"], [item["family_id"] for item in analysis.EXPECTED_FAMILIES])
            results = json.loads((output_root / "ANALYSIS_RESULTS.json").read_text(encoding="utf-8"))
            self.assertEqual(results["comparison_count"], 18)
            self.assertTrue(all(item["median_identity_effect"] > 0 for item in results["comparisons"]))
            coverage = json.loads((output_root / "METRIC_COVERAGE.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage["metric_rows"]["expected_metric_row_count"], 384)

    def test_missing_metric_row_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            metric_path = Path(binding["metric_rows"]["path"])
            lines = metric_path.read_text(encoding="utf-8").splitlines()
            metric_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            binding["metric_rows"]["sha256"] = analysis.sha256_file(metric_path)
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(analysis.FailClosedError, "指标键空间不完整"):
                analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertFalse(output_root.exists())

    def test_hash_mismatch_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            metric_path = Path(binding["metric_rows"]["path"])
            metric_path.write_text(metric_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(analysis.FailClosedError, "哈希不一致"):
                analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertFalse(output_root.exists())

    def test_incomplete_terminal_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            role = "lpips_terminal"
            terminal_path = Path(binding["artifacts"][role]["path"])
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["status"] = "METHOD_FAILURE"
            _write_json(terminal_path, terminal)
            binding["artifacts"][role]["sha256"] = analysis.sha256_file(terminal_path)
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(analysis.FailClosedError, "未达到冻结终态"):
                analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertFalse(output_root.exists())

    def test_historical_lpips_method_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            role = "lpips_linux_qualification_terminal"
            terminal_path = Path(binding["artifacts"][role]["path"])
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["status"] = "METHOD_FAILURE"
            _write_json(terminal_path, terminal)
            binding["artifacts"][role]["sha256"] = analysis.sha256_file(terminal_path)
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(analysis.FailClosedError, "未达到冻结终态"):
                analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertFalse(output_root.exists())

    def test_mps_full_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            role = "full_seed_2026080447_terminal"
            terminal_path = Path(binding["artifacts"][role]["path"])
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["device_backend"] = "mps"
            terminal["device_name"] = "Apple M5 Pro"
            _write_json(terminal_path, terminal)
            binding["artifacts"][role]["sha256"] = analysis.sha256_file(terminal_path)
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(analysis.FailClosedError, "不是 CUDA"):
                analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            self.assertFalse(output_root.exists())

    def test_complete_sface_failure_ledger_closes_confirmatory_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_root = Path(raw_temp)
            contract, contract_path = _synthetic_contract(temp_root)
            binding, binding_path, output_root = _build_binding(temp_root, contract, contract_path)
            metric_path = Path(binding["metric_rows"]["path"])
            updated_lines = []
            for line in metric_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if (
                    row["dataset_id"] == "D1"
                    and row["metric_id"] == "sface_source_to_render_cosine"
                    and row["identity_token"] == "D1-identity-0"
                ):
                    row["terminal_state"] = "EVALUATION_FAILURE"
                    row["failure_code"] = "SOURCE_DETECTION_FAILURE"
                    row["value"] = None
                updated_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            metric_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            binding["metric_rows"]["sha256"] = analysis.sha256_file(metric_path)
            _write_json(binding_path, binding)
            analysis.run_analysis(contract_path, binding_path, allow_synthetic=True)
            results = json.loads((output_root / "ANALYSIS_RESULTS.json").read_text(encoding="utf-8"))
            d1_sface = [
                item
                for item in results["comparisons"]
                if item["family_id"] == "F4-SFACE" and item["dataset_id"] == "D1"
            ]
            self.assertEqual(len(d1_sface), 3)
            self.assertTrue(all(item["identity_count"] == 1 for item in d1_sface))
            self.assertTrue(all(item["confirmatory_coverage_eligible"] is False for item in d1_sface))
            self.assertTrue(all(item["p_raw_two_sided_exact_sign"] is None for item in d1_sface))
            self.assertTrue(all(item["confirmatory_indeterminate"] is True for item in d1_sface))


if __name__ == "__main__":
    unittest.main(verbosity=2)
