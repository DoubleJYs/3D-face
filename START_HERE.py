#!/usr/bin/env python3
"""Single offline entry point for the FaceUV-Eval public release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cwd: Path, relative: str, *arguments: str) -> tuple[int, str]:
    command = [sys.executable, "-B", str(cwd / relative), *arguments]
    process = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return process.returncode, process.stdout.strip()


def source_layout() -> tuple[Path, str]:
    if (ROOT / "frugalface3d").is_dir():
        return ROOT, "reproducibility/w5b49n_canonreg_v1"
    repository = ROOT.parents[1]
    canonreg = (
        "paper_rewriting_output/final_paper_w49n_v15_zh_working/"
        "review_closure_20260826/private_cloud_run_source/canonreg_v1"
    )
    return repository, canonreg


def test_source() -> int:
    base, canonreg = source_layout()
    steps = [
        ("matched_model_synthetic", "reproducibility/w5b49n_v14_matched_controls_v1/test_synthetic.py", ()),
        ("matched_postprocess_synthetic", "reproducibility/w5b49n_v14_matched_controls_v1/test_postprocess_synthetic.py", ()),
        ("matched_statistics_synthetic", "reproducibility/w5b49n_v14_matched_controls_v1/test_statistics_synthetic.py", ()),
        ("canonreg_combined_local", f"{canonreg}/test_local.py", ("--repository-root", str(base))),
    ]
    results = []
    for name, relative, arguments in steps:
        code, output = run(base, relative, *arguments)
        results.append({"name": name, "exit_code": code, "output": output})
        if code:
            print(json.dumps({"status": "V15_SOURCE_SYNTHETIC_SUITE_FAILED", "steps": results}, ensure_ascii=False, sort_keys=True))
            return code
    print(json.dumps({"status": "PASS_V15_SOURCE_SYNTHETIC_SUITE", "steps": results}, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("check", "smoke", "test-source"), default="check")
    args = parser.parse_args()
    if args.action == "smoke":
        code, output = run(ROOT, "smoke/run_protocol_smoke.py")
        if output:
            print(output)
        return code
    if args.action == "test-source":
        return test_source()
    code, output = run(ROOT, "tools/verify_public_release.py")
    if output:
        print(output)
    return code
if __name__ == "__main__":
    raise SystemExit(main())
