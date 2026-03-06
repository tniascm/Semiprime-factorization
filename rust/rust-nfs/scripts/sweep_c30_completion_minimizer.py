#!/usr/bin/env python3
"""
Sweep c30 completion profiles and rank by minimum viable successful completion.

Purpose:
- preserve actual factor completion as the first criterion
- search for smaller collection / LA / dependency settings
- use the new matrix-based adaptive stop and basis-dependency cap controls
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


C30 = "684217602914977371691118975023"

PROFILE_ENVS: dict[str, dict[str, str]] = {
    "baseline": {},
    "matrix_1p35_basis512": {
        "RUST_NFS_ADAPTIVE_USE_MATRIX": "1",
        "RUST_NFS_ADAPTIVE_MATRIX_ROWS_RATIO": "1.35",
        "RUST_NFS_DEP_BASIS_LIMIT": "512",
        "RUST_NFS_DEP_RANDOM_COUNT": "128",
        "RUST_NFS_MAX_DEPS_TRY": "64",
    },
    "matrix_1p25_basis256": {
        "RUST_NFS_ADAPTIVE_USE_MATRIX": "1",
        "RUST_NFS_ADAPTIVE_MATRIX_ROWS_RATIO": "1.25",
        "RUST_NFS_DEP_BASIS_LIMIT": "256",
        "RUST_NFS_DEP_RANDOM_COUNT": "96",
        "RUST_NFS_MAX_DEPS_TRY": "48",
    },
    "matrix_1p15_basis128": {
        "RUST_NFS_ADAPTIVE_USE_MATRIX": "1",
        "RUST_NFS_ADAPTIVE_MATRIX_ROWS_RATIO": "1.15",
        "RUST_NFS_DEP_BASIS_LIMIT": "128",
        "RUST_NFS_DEP_RANDOM_COUNT": "64",
        "RUST_NFS_MAX_DEPS_TRY": "32",
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


def is_valid_factor(n: int, factor_text: str | None) -> bool:
    if not factor_text:
        return False
    try:
        factor = int(factor_text)
    except ValueError:
        return False
    return factor > 1 and factor < n and n % factor == 0


def run_profile(
    rust_bin: Path,
    cwd: Path,
    profile: str,
    env_overrides: dict[str, str],
    timeout_s: int,
    raw_log: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    env.update(
        {
            "RUST_NFS_MAX_VARIANTS": "1",
        }
    )
    env.update(env_overrides)

    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [str(rust_bin), "--factor", C30],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\n[timeout]"
    wall_s = time.perf_counter() - t0

    result = parse_result_json(stdout)
    viability = result.get("viability") or {}
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(
        json.dumps(
            {
                "profile": profile,
                "returncode": rc,
                "timed_out": timed_out,
                "wall_s": wall_s,
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
                "env": {
                    k: env[k]
                    for k in sorted(env)
                    if k.startswith("RUST_NFS_") or k.startswith("GNFS_")
                },
            },
            indent=2,
        )
    )
    return {
        "profile": profile,
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": wall_s,
        "result": result,
        "viability": viability,
    }


def score(run: dict[str, Any]) -> tuple[float, ...]:
    result = run["result"]
    viability = run["viability"]
    success = is_valid_factor(int(C30), result.get("factor"))
    return (
        1.0 if success else 0.0,
        -float(result.get("total_ms", 1e18) or 1e18),
        -float(int(viability.get("rows_minus_cols", -10**9) or -10**9)),
        -float(int(viability.get("remap_valid_relations", 0) or 0)),
        -float(int(result.get("dependencies_found", 0) or 0)),
    )


def format_row(run: dict[str, Any]) -> str:
    result = run["result"]
    viability = run["viability"]
    success = is_valid_factor(int(C30), result.get("factor"))
    return (
        f"{run['profile']:<22} "
        f"success={'yes' if success else 'no ':<3} "
        f"total_ms={float(result.get('total_ms', 0.0) or 0.0):>8.1f} "
        f"remap={int(viability.get('remap_valid_relations', 0) or 0):>6} "
        f"final={int(result.get('matrix_rows', 0) or 0):>5}x{int(result.get('matrix_cols', 0) or 0):<5} "
        f"rows-cols={int(viability.get('rows_minus_cols', 0) or 0):>6} "
        f"deps={int(result.get('dependencies_found', 0) or 0):>6} "
        f"factor={result.get('factor') or 'none'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--profiles", default="baseline,matrix_1p35_basis512,matrix_1p25_basis256")
    ap.add_argument("--output", default="")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/rust-nfs"
    rust_bin = rust_dir / "target/release/rust-nfs"
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in PROFILE_ENVS]
    if unknown:
        raise SystemExit(f"unknown profiles: {', '.join(unknown)}")

    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/c30_completion_minimizer_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)

    runs: list[dict[str, Any]] = []
    for profile in profiles:
        print(f"[sweep] {profile}", flush=True)
        run = run_profile(
            rust_bin=rust_bin,
            cwd=rust_dir,
            profile=profile,
            env_overrides=PROFILE_ENVS[profile],
            timeout_s=args.timeout,
            raw_log=raw_dir / f"{profile}.json",
        )
        runs.append(run)
        print(format_row(run), flush=True)

    ranked = sorted(runs, key=score, reverse=True)
    artifact = {
        "timestamp_unix": ts,
        "repo": str(repo),
        "case": C30,
        "profiles": profiles,
        "ranking_keys": [
            "factor_found",
            "-total_ms",
            "-rows_minus_cols",
            "-remap_valid_relations",
            "-dependencies_found",
        ],
        "ranked_profiles": [
            {
                "profile": run["profile"],
                "returncode": run["returncode"],
                "timed_out": run["timed_out"],
                "wall_s": run["wall_s"],
                "factor_found": is_valid_factor(int(C30), run["result"].get("factor")),
                "result": run["result"],
                "viability": run["viability"],
            }
            for run in ranked
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    if ranked:
        print("\nBest")
        print(format_row(ranked[0]))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
