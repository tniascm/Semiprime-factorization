#!/usr/bin/env python3
"""
Regression guardrail for full c30 completion on the normal adaptive path.

Purpose:
- keep the recent c30 completion milestone from regressing
- use the normal release binary path (sqrt enabled)
- assert that a nontrivial factor is produced for the fixed c30 case
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


C30 = "684217602914977371691118975023"


def parse_result_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    pos = text.find("{")
    if pos < 0:
        return {}
    try:
        return json.loads(text[pos:])
    except json.JSONDecodeError:
        return {}


def run_case(rust_bin: Path, cwd: Path, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    env.update(
        {
            "POTAPOV_NFS_MAX_VARIANTS": "1",
        }
    )
    proc = subprocess.run(
        [str(rust_bin), "--factor", C30],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    result = parse_result_json(proc.stdout)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "result": result,
        "viability": result.get("viability") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/potapov-nfs"
    rust_bin = rust_dir / "target/release/potapov-nfs"

    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            check=True,
        )

    run = run_case(rust_bin, rust_dir, args.timeout)
    result = run["result"]
    viability = run["viability"]
    factor_text = result.get("factor")
    factor = int(factor_text) if factor_text else 0
    n = int(C30)
    ok = (
        run["returncode"] == 0
        and factor > 1
        and factor < n
        and n % factor == 0
        and int(result.get("dependencies_found", 0) or 0) > 0
    )

    print(
        f"[c30] factor={factor_text or 'none'} "
        f"deps={int(result.get('dependencies_found', 0) or 0)} "
        f"rows={int(result.get('matrix_rows', 0) or 0)} "
        f"cols={int(result.get('matrix_cols', 0) or 0)} "
        f"remap={int(viability.get('remap_valid_relations', 0) or 0)} "
        f"hd_res={int(viability.get('remap_invalid_hd_residual', 0) or 0)} "
        f"total_ms={float(result.get('total_ms', 0.0) or 0.0):.1f}"
    )

    if ok:
        print("\nPASS")
        return 0

    print("\nFAIL")
    if run["returncode"] != 0:
        print(f"- process returned {run['returncode']}")
    if not factor_text:
        print("- no factor returned")
    elif factor <= 1 or factor >= n or n % factor != 0:
        print(f"- invalid factor returned: {factor_text}")
    if int(result.get("dependencies_found", 0) or 0) <= 0:
        print("- no dependencies found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
