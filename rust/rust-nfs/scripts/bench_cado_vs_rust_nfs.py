#!/usr/bin/env python3
"""
Reproducible CADO vs rust-nfs benchmark harness.

Writes:
- parsed benchmark JSON
- raw stdout/stderr logs for each case and engine
- rust-nfs structured run logs (if RUST_NFS_LOG_DIR is configured)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any


CASES: dict[str, list[str]] = {
    "c30": [
        "684217602914977371691118975023",
        "291695886709214217732173542261",
        "457206828091130153032152360761",
    ],
    "c45": [
        "327714360917956624476008484661697514621409657",
        "257924597305621056220013631148395117044787913",
        "960564208229425980896064150966530613470478361",
    ],
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def is_probable_prime(n: int, rounds: int = 16, rng: random.Random | None = None) -> bool:
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    # n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    rr = rng or random.Random(0)
    for _ in range(rounds):
        a = rr.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def random_prime(bits: int, rng: random.Random) -> int:
    while True:
        cand = rng.getrandbits(bits)
        cand |= (1 << (bits - 1))  # force top bit
        cand |= 1  # odd
        if is_probable_prime(cand, rounds=20, rng=rng):
            return cand


def generate_semiprime_cases(cases_per_tier: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {"c30": [], "c45": []}
    bits_by_tier = {"c30": 100, "c45": 148}

    for tier, n_bits in bits_by_tier.items():
        seen: set[int] = set()
        p_bits = n_bits // 2
        q_bits = n_bits - p_bits
        while len(out[tier]) < cases_per_tier:
            p = random_prime(p_bits, rng)
            q = random_prime(q_bits, rng)
            if p == q:
                continue
            n = p * q
            if n in seen:
                continue
            seen.add(n)
            out[tier].append(str(n))
    return out


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    wall = time.perf_counter() - start

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "cwd": str(cwd),
                "returncode": proc.returncode,
                "wall_s": wall,
                "stdout": stdout,
                "stderr": stderr,
            },
            indent=2,
        )
    )
    return proc.returncode, stdout, stderr, wall


def parse_cado(stdout: str, stderr: str) -> dict[str, Any]:
    clean = ANSI_RE.sub("", stdout + "\n" + stderr)
    total = re.search(
        r"Total cpu/elapsed time for entire Complete Factorization\s+[0-9.]+/([0-9.]+)",
        clean,
    )
    sieve = re.findall(r"Lattice Sieving: Total time: ([0-9.]+)s", clean)
    factors = re.search(r"Factors:\s+([0-9]+)\s+([0-9]+)", clean)
    return {
        "elapsed_s": float(total.group(1)) if total else None,
        "sieve_s": float(sieve[-1]) if sieve else None,
        "success": bool(factors),
        "factors": [factors.group(1), factors.group(2)] if factors else [],
    }


def parse_rust(stdout: str, stderr: str) -> dict[str, Any]:
    # rust-nfs writes final JSON to stdout; stderr contains progress logs.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--cado-dir", default="/Users/andriipotapov/cado-nfs")
    ap.add_argument("--output", default="")
    ap.add_argument("--cases-per-tier", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument(
        "--q-windows",
        type=int,
        default=0,
        help="Cap special-q windows for rust-nfs (0 = uncapped, default).",
    )
    ap.add_argument("--dep-seed", type=int, default=42)
    ap.add_argument("--dep-xor-k", type=int, default=0)
    ap.add_argument("--dep-random-count", type=int, default=0)
    ap.add_argument("--max-deps-try", type=int, default=0)
    ap.add_argument("--trivial-bail", type=int, default=0)
    ap.add_argument("--norm-block", type=int, default=0)
    ap.add_argument("--rel-target-mult", type=float, default=0.0)
    ap.add_argument("--rel-target-min", type=int, default=0)
    ap.add_argument("--max-lp-keys", type=int, default=0)
    ap.add_argument("--partial-merge-maxsets", type=int, default=0)
    ap.add_argument("--skip-build-rust", action="store_true")
    ap.add_argument("--random-semiprimes", action="store_true")
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo)
    cado_dir = Path(args.cado_dir)
    rust_bin = repo / "rust/rust-nfs/target/release/rust-nfs"
    cado_py = cado_dir / "cado-nfs.py"
    cado_python = cado_dir / "cado-nfs.venv/bin/python"

    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/cado_rust_c30_c45_benchmark_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    rust_log_root = run_root / "rust_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rust_log_root.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")

    if not args.skip_build_rust:
        print("[bench] building rust-nfs release binary", flush=True)
        build_rc, _, _, _ = run_cmd(
            ["cargo", "build", "--release"],
            cwd=repo / "rust/rust-nfs",
            env=env_base,
            log_path=raw_dir / "build_rust.json",
        )
        if build_rc != 0:
            raise SystemExit("rust-nfs build failed; see raw/build_rust.json")

    benchmark: dict[str, Any] = {
        "timestamp_unix": ts,
        "repo": str(repo),
        "cado_dir": str(cado_dir),
        "settings": {
            "threads": args.threads,
            "cases_per_tier": args.cases_per_tier,
            "rust_variants": args.variants,
            "rust_q_windows": args.q_windows if args.q_windows > 0 else None,
            "rust_dep_seed": args.dep_seed,
            "rust_dep_xor_k": args.dep_xor_k,
            "rust_dep_random_count": args.dep_random_count,
            "rust_max_deps_try": args.max_deps_try,
            "rust_trivial_bail": args.trivial_bail,
            "rust_norm_block": args.norm_block if args.norm_block > 0 else None,
            "rust_rel_target_mult": args.rel_target_mult if args.rel_target_mult > 0 else None,
            "rust_rel_target_min": args.rel_target_min if args.rel_target_min > 0 else None,
            "rust_max_lp_keys": args.max_lp_keys if args.max_lp_keys > 0 else None,
            "rust_partial_merge_maxsets": args.partial_merge_maxsets
            if args.partial_merge_maxsets > 0
            else None,
            "random_semiprimes": args.random_semiprimes,
            "random_seed": args.random_seed,
        },
        "runs": [],
    }

    cases_by_tier = (
        generate_semiprime_cases(args.cases_per_tier, args.random_seed)
        if args.random_semiprimes
        else CASES
    )
    benchmark["cases"] = cases_by_tier

    for tier in ("c30", "c45"):
        for i, n in enumerate(cases_by_tier[tier][: args.cases_per_tier]):
            case_id = f"{tier}_{i+1}"
            print(f"[bench] {case_id} n={n[:24]}...", flush=True)

            cado_cmd = [
                str(cado_python),
                str(cado_py),
                "--parameters",
                str(cado_dir / f"parameters/factor/params.{tier}"),
                "-t",
                str(args.threads),
                n,
                "server.whitelist=0.0.0.0/0",
                "slaves.hostnames=localhost",
                "slaves.nrclients=2",
                "server.address=localhost",
            ]
            cado_rc, cado_out, cado_err, cado_wall = run_cmd(
                cado_cmd,
                cwd=cado_dir,
                env=env_base,
                log_path=raw_dir / f"{case_id}_cado.json",
            )
            cado_metrics = parse_cado(cado_out, cado_err)
            cado_metrics["returncode"] = cado_rc
            cado_metrics["wall_s"] = cado_wall

            rust_env = env_base.copy()
            rust_env["RUST_NFS_MAX_VARIANTS"] = str(args.variants)
            if args.q_windows > 0:
                rust_env["RUST_NFS_MAX_Q_WINDOWS"] = str(args.q_windows)
            rust_env["RUST_NFS_DEP_SEED"] = str(args.dep_seed)
            rust_env["RUST_NFS_LOG_DIR"] = str(rust_log_root)
            if args.dep_xor_k > 0:
                rust_env["RUST_NFS_DEP_XOR_K"] = str(args.dep_xor_k)
            if args.dep_random_count > 0:
                rust_env["RUST_NFS_DEP_RANDOM_COUNT"] = str(args.dep_random_count)
            if args.max_deps_try > 0:
                rust_env["RUST_NFS_MAX_DEPS_TRY"] = str(args.max_deps_try)
            if args.trivial_bail > 0:
                rust_env["RUST_NFS_TRIVIAL_BAIL"] = str(args.trivial_bail)
            if args.norm_block > 0:
                rust_env["RUST_NFS_NORM_BLOCK"] = str(args.norm_block)
            if args.rel_target_mult > 0:
                rust_env["RUST_NFS_REL_TARGET_MULT"] = str(args.rel_target_mult)
            if args.rel_target_min > 0:
                rust_env["RUST_NFS_REL_TARGET_MIN"] = str(args.rel_target_min)
            if args.max_lp_keys > 0:
                rust_env["RUST_NFS_MAX_LP_KEYS"] = str(args.max_lp_keys)
            if args.partial_merge_maxsets > 0:
                rust_env["RUST_NFS_PARTIAL_MERGE_MAXSETS"] = str(args.partial_merge_maxsets)
            rust_cmd = [str(rust_bin), "--factor", n]
            rust_rc, rust_out, rust_err, rust_wall = run_cmd(
                rust_cmd,
                cwd=repo / "rust/rust-nfs",
                env=rust_env,
                log_path=raw_dir / f"{case_id}_rust.json",
            )
            rust_metrics = parse_rust(rust_out, rust_err)
            rust_metrics["returncode"] = rust_rc
            rust_metrics["wall_s"] = rust_wall

            benchmark["runs"].append(
                {
                    "case_id": case_id,
                    "tier": tier,
                    "n": n,
                    "cado": cado_metrics,
                    "rust": rust_metrics,
                }
            )

    benchmark["rust_log_run_count"] = len(list(rust_log_root.glob("*/run_config.json")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark, indent=2))
    print(f"[bench] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
