#!/usr/bin/env python3
"""
Regression guardrail for front-half remap viability on fixed c30/c45 cases.

This is intentionally narrow:
- release binary only
- fixed seeds/cases
- front-half only (sqrt disabled)
- fails if the recent remap fixes regress below known-good floors
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CASES: dict[str, str] = {
    "c30": "684217602914977371691118975023",
    "c45": "210073637581919767032888794008555542395100797",
}

FLOORS: dict[str, dict[str, int]] = {
    "c30": {
        "filtered_relations_min": 1900,
        "remap_valid_relations_min": 1900,
        "set_rows_recomputed_min": 1500,
        "deps_found_min": 1,
        "hd_residual_max": 0,
    },
    "c45": {
        "filtered_relations_min": 700,
        "remap_valid_relations_min": 700,
        "set_rows_recomputed_min": 600,
        "deps_found_min": 0,
        "hd_residual_max": 0,
    },
}


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


def run_case(rust_bin: Path, cwd: Path, n: str, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("RUST_NFS_MAX_RAW_RELS", None)
    env.update(
        {
            "RUST_NFS_SKIP_SQRT": "1",
            "RUST_NFS_SINGLETON_PRUNE": "0",
            "RUST_NFS_MAX_VARIANTS": "1",
            "RUST_NFS_MAX_Q_WINDOWS": "5",
            "RUST_NFS_HD_RESIDUAL_SAMPLE_LIMIT": "4",
        }
    )
    proc = subprocess.run(
        [str(rust_bin), "--factor", n],
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
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/rust-nfs"
    rust_bin = rust_dir / "target/release/rust-nfs"

    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            check=True,
        )

    failures: list[str] = []
    for tier, n in CASES.items():
        run = run_case(rust_bin, rust_dir, n, args.timeout)
        v = run["viability"]
        print(
            f"[{tier}] filtered={int(v.get('filtered_relations', 0) or 0)} "
            f"remap={int(v.get('remap_valid_relations', 0) or 0)} "
            f"sets={int(v.get('set_rows_recomputed', 0) or 0)} "
            f"deps={int(v.get('deps_found', 0) or 0)} "
            f"hd_res={int(v.get('remap_invalid_hd_residual', 0) or 0)}"
        )

        if run["returncode"] != 0:
            failures.append(f"{tier}: process returned {run['returncode']}")
            continue

        floor = FLOORS[tier]
        for key, min_value in floor.items():
            actual_key = key
            comparator = "min"
            if key.endswith("_max"):
                actual_key = key[:-4]
                comparator = "max"
            elif key.endswith("_min"):
                actual_key = key[:-4]

            actual = int(v.get(actual_key, 0) or 0)
            if comparator == "min" and actual < min_value:
                failures.append(f"{tier}: {actual_key}={actual} < {min_value}")
            if comparator == "max" and actual > min_value:
                failures.append(f"{tier}: {actual_key}={actual} > {min_value}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
