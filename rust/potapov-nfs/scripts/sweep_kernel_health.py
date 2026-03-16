#!/usr/bin/env python3
"""
Reproducible kernel-health sweep for potapov-nfs.

Purpose:
- quickly scan sieve/LA parameter combinations without entering expensive sqrt
- capture matrix rows/cols/dependency counts and stage timings
- write one JSON artifact plus per-run raw logs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("empty integer list")
    return out


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


def run_case(
    rust_bin: Path,
    cwd: Path,
    n: str,
    env: dict[str, str],
    raw_log: Path,
) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run([str(rust_bin), "--factor", n], cwd=cwd, env=env, text=True, capture_output=True)
    wall_s = time.perf_counter() - t0

    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(
        json.dumps(
            {
                "returncode": proc.returncode,
                "wall_s": wall_s,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "env": {
                    k: env.get(k)
                    for k in sorted(env)
                    if k.startswith("POTAPOV_NFS_") or k.startswith("GNFS_")
                },
            },
            indent=2,
        )
    )
    return proc.returncode, wall_s, proc.stdout, proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--n", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--log-i", default="9,10,11")
    ap.add_argument("--q-windows", default="10,20")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dep-seed", type=int, default=42)
    ap.add_argument("--dep-random-count", type=int, default=200)
    ap.add_argument("--dep-xor-k", type=int, default=4)
    ap.add_argument("--qc-count", type=int, default=30)
    ap.add_argument("--sparse-premerge", type=int, default=1)
    ap.add_argument("--singleton-prune", type=int, default=0)
    ap.add_argument("--singleton-min-weight", type=int, default=2)
    ap.add_argument("--require-coprime", action="store_true")
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--ignore-special-q", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/potapov-nfs"
    rust_bin = rust_dir / "target/release/potapov-nfs"

    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/potapov_nfs_kernel_sweep_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    run_logs = run_root / "rust_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_logs.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    if args.threads > 0:
        env_base["RAYON_NUM_THREADS"] = str(args.threads)

    if not args.skip_build:
        build = subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, env=env_base)
        if build.returncode != 0:
            raise SystemExit("build failed")

    log_i_values = parse_csv_ints(args.log_i)
    q_window_values = parse_csv_ints(args.q_windows)

    sweep: dict[str, Any] = {
        "timestamp_unix": ts,
        "n": args.n,
        "settings": {
            "variants": args.variants,
            "threads": args.threads,
            "dep_seed": args.dep_seed,
            "dep_random_count": args.dep_random_count,
            "dep_xor_k": args.dep_xor_k,
            "qc_count": args.qc_count,
            "sparse_premerge": args.sparse_premerge,
            "singleton_prune": args.singleton_prune,
            "singleton_min_weight": args.singleton_min_weight,
            "require_coprime": args.require_coprime,
            "full_only": args.full_only,
            "ignore_special_q": args.ignore_special_q,
            "log_i_values": log_i_values,
            "q_window_values": q_window_values,
            "skip_sqrt": True,
        },
        "runs": [],
    }

    run_id = 0
    for log_i in log_i_values:
        for q_windows in q_window_values:
            run_id += 1
            env = env_base.copy()
            env.update(
                {
                    "POTAPOV_NFS_MAX_VARIANTS": str(args.variants),
                    "POTAPOV_NFS_MAX_Q_WINDOWS": str(q_windows),
                    "POTAPOV_NFS_OVR_LOG_I": str(log_i),
                    "POTAPOV_NFS_QC_COUNT": str(args.qc_count),
                    "POTAPOV_NFS_DEP_SEED": str(args.dep_seed),
                    "POTAPOV_NFS_DEP_RANDOM_COUNT": str(args.dep_random_count),
                    "POTAPOV_NFS_DEP_XOR_K": str(args.dep_xor_k),
                    "POTAPOV_NFS_SKIP_SQRT": "1",
                    "POTAPOV_NFS_LOG_DIR": str(run_logs),
                }
            )
            env["POTAPOV_NFS_SPARSE_PREMERGE"] = "1" if args.sparse_premerge else "0"
            env["POTAPOV_NFS_SINGLETON_PRUNE"] = "1" if args.singleton_prune else "0"
            env["POTAPOV_NFS_SINGLETON_PRUNE_MIN_WEIGHT"] = str(args.singleton_min_weight)
            if args.require_coprime:
                env["POTAPOV_NFS_REQUIRE_COPRIME_AB"] = "1"
            if args.full_only:
                env["POTAPOV_NFS_FULL_ONLY"] = "1"
            if args.ignore_special_q:
                env["POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN"] = "1"

            tag = f"logi{log_i}_qw{q_windows}"
            rc, wall_s, stdout, stderr = run_case(
                rust_bin=rust_bin,
                cwd=rust_dir,
                n=args.n,
                env=env,
                raw_log=raw_dir / f"{run_id:02d}_{tag}.json",
            )
            result = parse_result_json(stdout)
            rows = int(result.get("matrix_rows", 0) or 0)
            cols = int(result.get("matrix_cols", 0) or 0)
            deps = int(result.get("dependencies_found", 0) or 0)
            sweep["runs"].append(
                {
                    "id": run_id,
                    "tag": tag,
                    "log_i": log_i,
                    "q_windows": q_windows,
                    "returncode": rc,
                    "wall_s": wall_s,
                    "rows": rows,
                    "cols": cols,
                    "kernel_margin_rows_minus_cols": rows - cols,
                    "dependencies_found": deps,
                    "sieve_ms": result.get("sieve_ms"),
                    "filter_ms": result.get("filter_ms"),
                    "la_ms": result.get("la_ms"),
                    "result": result,
                    "stderr_tail": stderr.strip().splitlines()[-8:],
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
