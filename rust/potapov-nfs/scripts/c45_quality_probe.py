#!/usr/bin/env python3
"""
Targeted c45 relation-quality probe.

Purpose:
- stop before sqrt
- use a fixed c45 case and a fixed collection frontier
- summarize whether the matrix is limited by dense-column burden,
  special-q/QC burden, or remap-valid yield
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


def probe_run(
    rust_bin: Path,
    cwd: Path,
    max_q_windows: int,
    timeout: int,
    raw_log: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    env.update(
        {
            "RUST_NFS_SKIP_SQRT": "1",
            "RUST_NFS_MAX_VARIANTS": "1",
            "RUST_NFS_MAX_Q_WINDOWS": str(max_q_windows),
        }
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(rust_bin), "--factor", C45],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    wall_s = time.perf_counter() - t0

    result = parse_result_json(proc.stdout)
    viability = result.get("viability") or {}
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(
        json.dumps(
            {
                "returncode": proc.returncode,
                "wall_s": wall_s,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "result": result,
            },
            indent=2,
        )
    )

    return {
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "result": result,
        "viability": viability,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/andriipotapov/Semiprime")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-q-windows", type=int, default=32)
    ap.add_argument("--output", default="")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    rust_dir = repo / "rust/rust-nfs"
    rust_bin = rust_dir / "target/release/rust-nfs"
    ts = int(time.time())
    out_path = (
        Path(args.output)
        if args.output
        else repo / f"rust/data/c45_quality_probe_{ts}.json"
    )

    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            check=True,
        )

    run = probe_run(rust_bin, rust_dir, args.max_q_windows, args.timeout, out_path)
    result = run["result"]
    viability = run["viability"]

    active_rat = int(viability.get("active_dense_rat_cols", 0) or 0)
    active_alg = int(viability.get("active_dense_alg_cols", 0) or 0)
    active_sq = int(viability.get("active_special_q_cols", 0) or 0)
    active_qc = int(viability.get("active_qc_cols", 0) or 0)
    dense_cols = active_rat + active_alg + active_qc + 2
    dense_rows_gap = int(viability.get("set_rows_matrix", 0) or 0) - dense_cols
    remap_valid = int(viability.get("remap_valid_relations", 0) or 0)
    filtered = int(viability.get("filtered_relations", 0) or 0)
    set_rows = int(viability.get("set_rows_recomputed", 0) or 0)
    keep_ratio = remap_valid / filtered if filtered else 0.0
    set_ratio = set_rows / remap_valid if remap_valid else 0.0
    dense_singletons = (
        int(viability.get("pre_prune_singleton_rat_fb_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_alg_dense_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_sign_cols", 0) or 0)
        + int(viability.get("pre_prune_singleton_qc_cols", 0) or 0)
    )

    print(
        f"[c45] raw={int(result.get('relations_found', 0) or 0)} "
        f"filtered={filtered} remap={remap_valid} keep={keep_ratio:.3f} "
        f"sets={set_rows} set_ratio={set_ratio:.3f} "
        f"dense_cols={dense_cols} dense_gap={dense_rows_gap} "
        f"sq_cols={active_sq} qc_cols={active_qc} "
        f"dense_singletons={dense_singletons} "
        f"sq_singletons={int(viability.get('pre_prune_singleton_special_q_cols', 0) or 0)} "
        f"deps={int(viability.get('deps_found', 0) or 0)}"
    )

    if dense_rows_gap < 0:
        print("diagnosis=dense_column_burden")
    elif dense_singletons > 0:
        print("diagnosis=collision_quality")
    elif keep_ratio < 0.9:
        print("diagnosis=remap_or_filter_loss")
    else:
        print("diagnosis=needs_followup")

    return 0 if run["returncode"] == 0 else run["returncode"]


if __name__ == "__main__":
    sys.exit(main())
