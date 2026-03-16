#!/usr/bin/env python3
"""
Sweep exact c45 polynomial variants and rank them by matrix viability.

This uses the fixed c45 quality-probe settings:
- sqrt disabled
- fixed q-window cap
- exact variant selection via POTAPOV_NFS_VARIANT_START + MAX_VARIANTS=1

Primary ranking:
1. deps_found
2. dense_gap = set_rows - dense_cols
3. -dense_singletons
4. set_rows / remap_valid
5. remap_valid / filtered
6. -wall_s
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


C45 = "210073637581919767032888794008555542395100797"


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
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


def derive_metrics(viability: dict[str, Any]) -> dict[str, float | int]:
    filtered = int(viability.get("filtered_relations", 0) or 0)
    remap_valid = int(viability.get("remap_valid_relations", 0) or 0)
    set_rows = int(viability.get("set_rows_recomputed", 0) or 0)
    deps = int(viability.get("deps_found", 0) or 0)
    active_rat = int(viability.get("active_dense_rat_cols", 0) or 0)
    active_alg = int(viability.get("active_dense_alg_cols", 0) or 0)
    active_qc = int(viability.get("active_qc_cols", 0) or 0)
    active_sq = int(viability.get("active_special_q_cols", 0) or 0)
    dense_cols = active_rat + active_alg + active_qc + 2
    dense_gap = set_rows - dense_cols
    dense_singletons = (
        int(viability.get("pre_prune_singleton_rat_fb_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_alg_dense_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_sign_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_qc_cols", 0) or 0)
    )
    sq_singletons = int(viability.get("pre_prune_singleton_special_q_cols", 0) or 0)
    set_ratio = set_rows / remap_valid if remap_valid else 0.0
    remap_keep = remap_valid / filtered if filtered else 0.0
    return {
        "deps_found": deps,
        "dense_cols": dense_cols,
        "dense_gap": dense_gap,
        "dense_singletons": dense_singletons,
        "sq_singletons": sq_singletons,
        "set_ratio": set_ratio,
        "remap_keep": remap_keep,
        "active_sq_cols": active_sq,
    }


def run_variant(
    rust_bin: Path,
    cwd: Path,
    variant: int,
    max_q_windows: int,
    timeout_s: int,
    raw_log: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    env.update(
        {
            "POTAPOV_NFS_SKIP_SQRT": "1",
            "POTAPOV_NFS_FALLBACK_RHO": "0",
            "POTAPOV_NFS_MAX_VARIANTS": "1",
            "POTAPOV_NFS_VARIANT_START": str(variant),
            "POTAPOV_NFS_MAX_Q_WINDOWS": str(max_q_windows),
        }
    )

    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [str(rust_bin), "--factor", C45],
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
    metrics = derive_metrics(viability)
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(
        json.dumps(
            {
                "variant": variant,
                "returncode": rc,
                "timed_out": timed_out,
                "wall_s": wall_s,
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
                "metrics": metrics,
                "env": {
                    k: env[k]
                    for k in sorted(env)
                    if k.startswith("POTAPOV_NFS_") or k.startswith("GNFS_")
                },
            },
            indent=2,
        )
    )
    return {
        "variant": variant,
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": wall_s,
        "result": result,
        "viability": viability,
        "metrics": metrics,
    }


def score(run: dict[str, Any]) -> tuple[float, ...]:
    m = run["metrics"]
    return (
        float(m["deps_found"]),
        float(m["dense_gap"]),
        -float(m["dense_singletons"]),
        float(m["set_ratio"]),
        float(m["remap_keep"]),
        -float(run["wall_s"]),
    )


def format_row(run: dict[str, Any]) -> str:
    v = run["viability"]
    m = run["metrics"]
    return (
        f"variant={run['variant']:>2} "
        f"filtered={int(v.get('filtered_relations', 0) or 0):>5} "
        f"remap={int(v.get('remap_valid_relations', 0) or 0):>5} "
        f"sets={int(v.get('set_rows_recomputed', 0) or 0):>5} "
        f"deps={int(m['deps_found']):>3} "
        f"dense_gap={int(m['dense_gap']):>6} "
        f"dense_singletons={int(m['dense_singletons']):>4} "
        f"set_ratio={float(m['set_ratio']):>5.3f} "
        f"remap_keep={float(m['remap_keep']):>5.3f} "
        f"sq_cols={int(m['active_sq_cols']):>4} "
        f"sq_singletons={int(m['sq_singletons']):>3} "
        f"wall_s={run['wall_s']:.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--output", default="")
    ap.add_argument("--variants", default="0,1,2,3")
    ap.add_argument("--max-q-windows", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/potapov-nfs"
    rust_bin = rust_dir / "target/release/potapov-nfs"
    variants = parse_csv_ints(args.variants)
    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/c45_poly_variant_sweep_{ts}.json"
    )
    run_root = out_path.with_suffix("")
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)

    runs: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[sweep] variant={variant}", flush=True)
        run = run_variant(
            rust_bin=rust_bin,
            cwd=rust_dir,
            variant=variant,
            max_q_windows=args.max_q_windows,
            timeout_s=args.timeout,
            raw_log=raw_dir / f"variant_{variant:02d}.json",
        )
        runs.append(run)
        print(format_row(run), flush=True)

    ranked = sorted(runs, key=score, reverse=True)
    artifact = {
        "timestamp_unix": ts,
        "repo": str(repo),
        "case": C45,
        "variants": variants,
        "max_q_windows": args.max_q_windows,
        "ranking_keys": [
            "deps_found",
            "dense_gap",
            "-dense_singletons",
            "set_ratio",
            "remap_keep",
            "-wall_s",
        ],
        "ranked_variants": [
            {
                "variant": run["variant"],
                "returncode": run["returncode"],
                "timed_out": run["timed_out"],
                "wall_s": run["wall_s"],
                "metrics": run["metrics"],
                "viability": run["viability"],
            }
            for run in ranked
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    best = ranked[0] if ranked else None
    if best is not None:
        print("\nBest")
        print(format_row(best))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
