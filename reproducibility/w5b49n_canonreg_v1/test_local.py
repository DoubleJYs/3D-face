#!/usr/bin/env python3
"""Run the complete local preflight once: source, loss, then CPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_check import run_check
from test_loss import run_tests
from test_smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    result = {
        "status": "PASS_CANONREG_LOCAL_PREFLIGHT",
        "source_check": run_check(args.repository_root, runtime=True),
        "loss_tests": run_tests(),
        "cpu_smoke": run_smoke(args.repository_root, device_name="cpu"),
        "scientific_result_generated": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
