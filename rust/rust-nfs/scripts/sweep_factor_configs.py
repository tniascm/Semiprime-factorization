#!/usr/bin/env python3
"""
Reproducible rust-nfs config sweep with raw logs and parsed metrics.

Designed for fast tuning loops:
- keeps max_variants fixed (default 1) for apples-to-apples comparisons
- stores full stdout/stderr and effective env per run
- writes one JSON summary artifact with best successful config
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEP_LENS_RE = re.compile(
    r"dep lens expanded min/p50/p90/p99/max=([0-9]+)/([0-9]+)/([0-9]+)/([0-9]+)/([0-9]+)"
)


def parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise ValueError(f"expected positive integer, got {v}")
        out.append(v)
    if not out:
        raise ValueError("empty integer list")
    return sorted(set(out))


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


def parse_dep_lens(stderr: str) -> dict[str, int] | None:
    for line in stderr.splitlines():
        m = DEP_LENS_RE.search(line)
        if not m:
            continue
        vals = [int(m.group(i)) for i in range(1, 6)]
        return {
            "min": vals[0],
            "p50": vals[1],
            "p90": vals[2],
            "p99": vals[3],
            "max": vals[4],
        }
    return None


def run_case(
    rust_bin: Path,
    cwd: Path,
    n: str,
    env: dict[str, str],
    timeout_s: int,
    raw_path: Path,
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

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "cwd": str(cwd),
                "returncode": rc,
                "timed_out": timed_out,
                "wall_s": wall_s,
                "stdout": stdout,
                "stderr": stderr,
                "env": {
                    k: env[k]
                    for k in sorted(env)
                    if k.startswith("RUST_NFS_") or k.startswith("GNFS_")
                },
            },
            indent=2,
        )
    )

    result = parse_result_json(stdout)
    dep_lens = parse_dep_lens(stderr)
    return {
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": wall_s,
        "result": result,
        "dep_lens": dep_lens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--n", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--log-i", default="11")
    ap.add_argument("--q-windows", default="10,12,14,16,20")
    ap.add_argument("--full-only", default="0,1", help="comma list of 0/1")
    ap.add_argument("--require-coprime", type=int, default=1)
    ap.add_argument("--sparse-premerge", type=int, default=1)
    ap.add_argument("--singleton-prune", type=int, default=0)
    ap.add_argument("--singleton-min-weight", type=int, default=2)
    ap.add_argument("--max-variants", type=int, default=1)
    ap.add_argument("--max-dep-len", type=int, default=400)
    ap.add_argument("--dep-len-tiers", default="")
    ap.add_argument("--dep-require-coprime-rel", type=int, default=0)
    ap.add_argument("--max-deps-try", type=int, default=30)
    ap.add_argument("--trivial-bail", type=int, default=30)
    ap.add_argument("--nf-mode", default="")
    ap.add_argument("--try-neg-m", type=int, default=-1)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/rust-nfs"
    rust_bin = rust_dir / "target/release/rust-nfs"

    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/rust_nfs_factor_sweep_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    rust_logs = run_root / "rust_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rust_logs.mkdir(parents=True, exist_ok=True)

    log_i_vals = parse_csv_ints(args.log_i)
    qw_vals = parse_csv_ints(args.q_windows)
    full_vals = sorted(
        set(int(v.strip()) for v in args.full_only.split(",") if v.strip() != "")
    )
    if not full_vals or any(v not in (0, 1) for v in full_vals):
        raise SystemExit("--full-only must be comma list of 0/1")

    env_base = os.environ.copy()
    if not args.skip_build:
        print("[sweep] building rust-nfs release", flush=True)
        proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            env=env_base,
            text=True,
            capture_output=True,
        )
        (raw_dir / "build.json").write_text(
            json.dumps(
                {
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
                indent=2,
            )
        )
        if proc.returncode != 0:
            raise SystemExit("build failed (see raw/build.json)")

    runs: list[dict[str, Any]] = []
    idx = 0
    for log_i in log_i_vals:
        for qw in qw_vals:
            for full_only in full_vals:
                idx += 1
                cfg_name = f"logi{log_i}_qw{qw}_full{full_only}"
                print(f"[sweep] #{idx} {cfg_name}", flush=True)
                env = env_base.copy()
                env.update(
                    {
                        "RUST_NFS_LOG_DIR": str(rust_logs),
                        "RUST_NFS_MAX_VARIANTS": str(args.max_variants),
                        "RUST_NFS_OVR_LOG_I": str(log_i),
                        "RUST_NFS_MAX_Q_WINDOWS": str(qw),
                        "RUST_NFS_SPARSE_PREMERGE": str(args.sparse_premerge),
                        "RUST_NFS_SINGLETON_PRUNE": str(args.singleton_prune),
                        "RUST_NFS_SINGLETON_PRUNE_MIN_WEIGHT": str(
                            args.singleton_min_weight
                        ),
                        "RUST_NFS_REQUIRE_COPRIME_AB": str(args.require_coprime),
                        "RUST_NFS_FULL_ONLY": str(full_only),
                        "RUST_NFS_MAX_DEP_LEN": str(args.max_dep_len),
                        "RUST_NFS_DEP_REQUIRE_COPRIME_REL": str(
                            args.dep_require_coprime_rel
                        ),
                        "RUST_NFS_MAX_DEPS_TRY": str(args.max_deps_try),
                        "RUST_NFS_TRIVIAL_BAIL": str(args.trivial_bail),
                    }
                )
                if args.dep_len_tiers:
                    env["RUST_NFS_DEP_LEN_TIERS"] = args.dep_len_tiers
                if args.nf_mode:
                    env["GNFS_NF_ELEMENT_MODE"] = args.nf_mode
                if args.try_neg_m in (0, 1):
                    env["GNFS_TRY_NEG_M"] = str(args.try_neg_m)

                rec = run_case(
                    rust_bin=rust_bin,
                    cwd=rust_dir,
                    n=args.n,
                    env=env,
                    timeout_s=args.timeout,
                    raw_path=raw_dir / f"{idx:02d}_{cfg_name}.json",
                )
                rec.update(
                    {
                        "config": {
                            "log_i": log_i,
                            "q_windows": qw,
                            "full_only": full_only,
                            "require_coprime": args.require_coprime,
                            "sparse_premerge": args.sparse_premerge,
                            "singleton_prune": args.singleton_prune,
                            "singleton_min_weight": args.singleton_min_weight,
                            "max_dep_len": args.max_dep_len,
                            "dep_len_tiers": args.dep_len_tiers or None,
                            "dep_require_coprime_rel": args.dep_require_coprime_rel,
                            "max_variants": args.max_variants,
                            "max_deps_try": args.max_deps_try,
                            "trivial_bail": args.trivial_bail,
                            "nf_mode": args.nf_mode or None,
                            "try_neg_m": args.try_neg_m if args.try_neg_m in (0, 1) else None,
                        }
                    }
                )
                runs.append(rec)

    successful = [
        r
        for r in runs
        if r["returncode"] == 0 and r.get("result", {}).get("factor") is not None
    ]
    best_success = None
    if successful:
        best_success = min(
            successful,
            key=lambda r: r.get("result", {}).get("total_ms", float("inf")),
        )

    out = {
        "timestamp_unix": ts,
        "n": args.n,
        "settings": {
            "log_i": log_i_vals,
            "q_windows": qw_vals,
            "full_only": full_vals,
            "require_coprime": args.require_coprime,
            "sparse_premerge": args.sparse_premerge,
            "singleton_prune": args.singleton_prune,
            "singleton_min_weight": args.singleton_min_weight,
            "max_variants": args.max_variants,
            "max_dep_len": args.max_dep_len,
            "dep_len_tiers": args.dep_len_tiers or None,
            "dep_require_coprime_rel": args.dep_require_coprime_rel,
            "max_deps_try": args.max_deps_try,
            "trivial_bail": args.trivial_bail,
            "nf_mode": args.nf_mode or None,
            "try_neg_m": args.try_neg_m if args.try_neg_m in (0, 1) else None,
            "timeout_s": args.timeout,
        },
        "runs": runs,
        "successful_count": len(successful),
        "best_success": best_success,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[sweep] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
