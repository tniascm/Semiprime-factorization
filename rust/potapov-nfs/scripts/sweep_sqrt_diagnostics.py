#!/usr/bin/env python3
"""
Sweep sqrt/debug modes for potapov-nfs with reproducible settings.

Produces one JSON artifact with per-config:
- runtime metrics (from potapov-nfs stdout JSON)
- parsed gcd diagnostic signatures (from stderr)
- raw logs per config under <output_dir>/raw/
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

GCD_RE = re.compile(r"gcd\(x-y\)=([0-9]+), gcd\(x\+y\)=([0-9a-zA-Z]+)")


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


def parse_diag(stderr: str) -> dict[str, Any]:
    y_lines = [ln for ln in stderr.splitlines() if "[diag] y#" in ln and "gcd(x-y)=" in ln]
    gcd_n = 0
    gcd_one_one = 0
    samples: list[str] = []
    for ln in y_lines:
        m = GCD_RE.search(ln)
        if not m:
            continue
        gxy, gxp = m.group(1), m.group(2)
        if len(samples) < 6:
            samples.append(ln)
        if gxy != "1" and gxp in ("1", "pending"):
            gcd_n += 1
        if gxy == "1" and gxp == "1":
            gcd_one_one += 1
    return {
        "diag_lines": len(y_lines),
        "diag_gcd_n_like": gcd_n,
        "diag_gcd_one_one": gcd_one_one,
        "diag_samples": samples,
    }


def run_case(
    rust_bin: Path,
    cwd: Path,
    n: str,
    env: dict[str, str],
    raw_log: Path,
) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run([str(rust_bin), "--factor", n], cwd=cwd, env=env, text=True, capture_output=True)
    wall = time.perf_counter() - t0

    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(
        json.dumps(
            {
                "returncode": proc.returncode,
                "wall_s": wall,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "env": {k: env.get(k) for k in sorted(env) if k.startswith("POTAPOV_NFS_") or k.startswith("GNFS_")},
            },
            indent=2,
        )
    )
    return proc.returncode, wall, proc.stdout, proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--n", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--max-deps", type=int, default=200)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--q-windows", type=int, default=10)
    ap.add_argument("--dep-seed", type=int, default=42)
    ap.add_argument("--verbose-deps", type=int, default=1)
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/potapov-nfs"
    rust_bin = rust_dir / "target/release/potapov-nfs"

    ts = int(time.time())
    out_path = Path(args.output) if args.output else repo / f"rust/data/potapov_nfs_sqrt_sweep_{ts}.json"
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    run_logs = run_root / "rust_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_logs.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    build = subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, env=env_base)
    if build.returncode != 0:
        raise SystemExit("build failed")

    configs = [
        {
            "name": "baseline",
            "env": {
                "GNFS_NF_ELEMENT_MODE": "a_minus_ba",
                "GNFS_TRY_NEG_M": "0",
                "POTAPOV_NFS_QC_COUNT": "30",
            },
        },
        {
            "name": "nf_plus",
            "env": {
                "GNFS_NF_ELEMENT_MODE": "a_plus_ba",
                "GNFS_TRY_NEG_M": "0",
                "POTAPOV_NFS_QC_COUNT": "30",
            },
        },
        {
            "name": "nf_swapped",
            "env": {
                "GNFS_NF_ELEMENT_MODE": "b_alpha_minus_a",
                "GNFS_TRY_NEG_M": "0",
                "POTAPOV_NFS_QC_COUNT": "30",
            },
        },
        {
            "name": "qc0",
            "env": {
                "GNFS_NF_ELEMENT_MODE": "a_minus_ba",
                "GNFS_TRY_NEG_M": "0",
                "POTAPOV_NFS_QC_COUNT": "0",
            },
        },
        {
            "name": "neg_m",
            "env": {
                "GNFS_NF_ELEMENT_MODE": "a_minus_ba",
                "GNFS_TRY_NEG_M": "1",
                "POTAPOV_NFS_QC_COUNT": "30",
            },
        },
    ]

    sweep: dict[str, Any] = {
        "timestamp_unix": ts,
        "n": args.n,
        "settings": {
            "max_deps": args.max_deps,
            "variants": args.variants,
            "q_windows": args.q_windows,
            "dep_seed": args.dep_seed,
            "verbose_deps": args.verbose_deps,
        },
        "configs": [],
    }

    for cfg in configs:
        env = env_base.copy()
        env.update(cfg["env"])
        env.update(
            {
                "POTAPOV_NFS_MAX_VARIANTS": str(args.variants),
                "POTAPOV_NFS_MAX_Q_WINDOWS": str(args.q_windows),
                "POTAPOV_NFS_DEP_SEED": str(args.dep_seed),
                "POTAPOV_NFS_MAX_DEPS_TRY": str(args.max_deps),
                "POTAPOV_NFS_TRIVIAL_BAIL": str(args.max_deps),
                "POTAPOV_NFS_SQRT_VERBOSE_DEPS": str(args.verbose_deps),
                "POTAPOV_NFS_LOG_DIR": str(run_logs),
            }
        )

        rc, wall_s, stdout, stderr = run_case(
            rust_bin=rust_bin,
            cwd=rust_dir,
            n=args.n,
            env=env,
            raw_log=raw_dir / f"{cfg['name']}.json",
        )

        result = parse_result_json(stdout)
        diag = parse_diag(stderr)
        sweep["configs"].append(
            {
                "name": cfg["name"],
                "env": cfg["env"],
                "returncode": rc,
                "wall_s": wall_s,
                "result": result,
                "diag": diag,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
