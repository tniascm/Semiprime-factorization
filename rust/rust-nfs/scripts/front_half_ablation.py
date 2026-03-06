#!/usr/bin/env python3
"""
Front-half viability ablation runner for rust-nfs.

Purpose:
- keep sqrt out of the loop
- run a fixed case set through a small profile matrix
- rank profiles by matrix viability rather than wall time
- preserve raw stdout/stderr for debugging
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


FIXED_CASES: dict[str, list[str]] = {
    "c30": ["684217602914977371691118975023"],
    "c45": ["210073637581919767032888794008555542395100797"],
}

PROFILE_ENVS: dict[str, dict[str, str]] = {
    "baseline": {},
    "singleton_off": {
        "RUST_NFS_SINGLETON_PRUNE": "0",
    },
    "ignore_special_q": {
        "RUST_NFS_IGNORE_SPECIAL_Q_COLUMN": "1",
    },
    "qc0": {
        "RUST_NFS_QC_COUNT": "0",
    },
    "higher_rel_target": {
        "RUST_NFS_REL_TARGET_MULT": "0.35",
        "RUST_NFS_REL_TARGET_MIN": "3000",
    },
    "direct_matrix": {
        "RUST_NFS_PARTIAL_MERGE_2LP": "0",
    },
    "ignore_special_q_singleton_off": {
        "RUST_NFS_IGNORE_SPECIAL_Q_COLUMN": "1",
        "RUST_NFS_SINGLETON_PRUNE": "0",
    },
    "qc0_singleton_off": {
        "RUST_NFS_QC_COUNT": "0",
        "RUST_NFS_SINGLETON_PRUNE": "0",
    },
    "higher_rel_target_singleton_off": {
        "RUST_NFS_REL_TARGET_MULT": "0.35",
        "RUST_NFS_REL_TARGET_MIN": "3000",
        "RUST_NFS_SINGLETON_PRUNE": "0",
    },
    "direct_matrix_singleton_off": {
        "RUST_NFS_PARTIAL_MERGE_2LP": "0",
        "RUST_NFS_SINGLETON_PRUNE": "0",
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


def score_run(run: dict[str, Any]) -> tuple[float, ...]:
    viability = run.get("viability") or {}
    filtered = int(viability.get("filtered_relations", 0) or 0)
    remap_valid = int(viability.get("remap_valid_relations", 0) or 0)
    set_rows = int(viability.get("set_rows_recomputed", 0) or 0)
    deps = int(viability.get("deps_found", 0) or 0)
    rows_minus_cols = int(viability.get("rows_minus_cols", -10**9) or 0)
    remap_keep = remap_valid / filtered if filtered else 0.0
    set_row_yield = set_rows / remap_valid if remap_valid else 0.0
    return (float(deps), float(rows_minus_cols), remap_keep, set_row_yield)


def derive_metrics(viability: dict[str, Any]) -> dict[str, float]:
    filtered = int(viability.get("filtered_relations", 0) or 0)
    remap_valid = int(viability.get("remap_valid_relations", 0) or 0)
    set_rows = int(viability.get("set_rows_recomputed", 0) or 0)
    return {
        "set_rows_per_filtered": set_rows / filtered if filtered else 0.0,
        "set_rows_per_remap_valid": set_rows / remap_valid if remap_valid else 0.0,
    }


def run_case(
    rust_bin: Path,
    cwd: Path,
    n: str,
    env: dict[str, str],
    timeout_s: int,
    raw_log: Path,
) -> dict[str, Any]:
    cmd = [str(rust_bin), "--factor", n]
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
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
                "cmd": cmd,
                "cwd": str(cwd),
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
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": wall_s,
        "result": result,
        "viability": viability,
        "derived": derive_metrics(viability),
        "score": score_run({"viability": viability}),
    }


def format_row(run: dict[str, Any]) -> str:
    viability = run.get("viability") or {}
    filtered = int(viability.get("filtered_relations", 0) or 0)
    remap_valid = int(viability.get("remap_valid_relations", 0) or 0)
    set_rows = int(viability.get("set_rows_recomputed", 0) or 0)
    sets_per_filtered = set_rows / filtered if filtered else 0.0
    sets_per_remap = set_rows / remap_valid if remap_valid else 0.0
    return (
        f"{run['profile']:<16} "
        f"filtered={filtered:>5} "
        f"remap={remap_valid:>5} "
        f"hd_res={int(viability.get('remap_invalid_hd_residual', 0) or 0):>5} "
        f"sets={set_rows:>5} "
        f"set_yield=({sets_per_filtered:>4.2f},{sets_per_remap:>4.2f}) "
        f"dense=({int(viability.get('active_dense_rat_cols', 0) or 0):>4},"
        f"{int(viability.get('active_dense_alg_cols', 0) or 0):>4},"
        f"{int(viability.get('active_special_q_cols', 0) or 0):>3}) "
        f"final={int(viability.get('final_rows', 0) or 0):>5}x{int(viability.get('final_cols', 0) or 0):<5} "
        f"rows-cols={int(viability.get('rows_minus_cols', 0) or 0):>6} "
        f"deps={int(viability.get('deps_found', 0) or 0):>5}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--profiles",
        default="baseline,singleton_off,ignore_special_q,qc0,higher_rel_target,direct_matrix",
    )
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-q-windows", type=int, default=1)
    ap.add_argument("--max-variants", type=int, default=1)
    ap.add_argument("--hd-residual-samples", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/rust-nfs"
    rust_bin = rust_dir / "target/release/rust-nfs"

    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    if not profile_names:
        raise SystemExit("no profiles selected")
    unknown = [name for name in profile_names if name not in PROFILE_ENVS]
    if unknown:
        raise SystemExit(f"unknown profiles: {', '.join(sorted(unknown))}")

    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/front_half_ablation_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    if args.threads > 0:
        env_base["RAYON_NUM_THREADS"] = str(args.threads)

    if not args.skip_build:
        build = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            env=env_base,
            text=True,
            capture_output=True,
        )
        (raw_dir / "build.json").write_text(
            json.dumps(
                {
                    "returncode": build.returncode,
                    "stdout": build.stdout,
                    "stderr": build.stderr,
                },
                indent=2,
            )
        )
        if build.returncode != 0:
            raise SystemExit("build failed")

    base_env = {
        "RUST_NFS_SKIP_SQRT": "1",
        "RUST_NFS_MAX_Q_WINDOWS": str(args.max_q_windows),
        "RUST_NFS_MAX_VARIANTS": str(args.max_variants),
        "RUST_NFS_HD_RESIDUAL_SAMPLE_LIMIT": str(args.hd_residual_samples),
    }

    artifact: dict[str, Any] = {
        "timestamp_unix": ts,
        "repo": str(repo),
        "cases": FIXED_CASES,
        "profiles": {name: PROFILE_ENVS[name] for name in profile_names},
        "settings": {
            "threads": args.threads,
            "max_q_windows": args.max_q_windows,
            "max_variants": args.max_variants,
            "hd_residual_samples": args.hd_residual_samples,
            "timeout_s": args.timeout,
        },
        "runs": [],
        "best_by_case": {},
    }

    for tier, numbers in FIXED_CASES.items():
        for case_idx, n in enumerate(numbers, start=1):
            case_key = f"{tier}_{case_idx}"
            case_runs: list[dict[str, Any]] = []
            print(f"[ablation] {case_key} {n}")
            for profile in profile_names:
                env = env_base.copy()
                env.update(base_env)
                env.update(PROFILE_ENVS[profile])
                rec = run_case(
                    rust_bin=rust_bin,
                    cwd=rust_dir,
                    n=n,
                    env=env,
                    timeout_s=args.timeout,
                    raw_log=raw_dir / f"{case_key}_{profile}.json",
                )
                rec.update(
                    {
                        "tier": tier,
                        "case_key": case_key,
                        "n": n,
                        "profile": profile,
                    }
                )
                artifact["runs"].append(rec)
                case_runs.append(rec)
                print("  " + format_row(rec))

            ranked = sorted(case_runs, key=score_run, reverse=True)
            artifact["best_by_case"][case_key] = [
                {
                    "profile": run["profile"],
                    "score": list(run["score"]),
                    "viability": run["viability"],
                    "derived": run["derived"],
                }
                for run in ranked
            ]
            if ranked:
                best = ranked[0]
                print(f"  best={best['profile']} score={list(best['score'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
