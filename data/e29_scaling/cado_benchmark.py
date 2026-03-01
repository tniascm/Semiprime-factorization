#!/usr/bin/env python3
"""
CADO-NFS vs MCMC head-to-head benchmark.

Runs CADO-NFS on the same semiprimes used in E29 scaling experiment,
extracts sieving-phase metrics, and compares with MCMC results.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CADO_NFS_PY = "/Users/andriipotapov/cado-nfs/cado-nfs.py"
SCRIPT_DIR = Path(__file__).parent
SUMMARY_FILE = SCRIPT_DIR / "summary.json"
OUTPUT_FILE = SCRIPT_DIR / "cado_comparison.json"
WORKDIR_BASE = Path("/tmp/cado_bench")

# Bit sizes where CADO-NFS has parameter files (c30+ = 30+ digits = ~100+ bits)
MIN_BITS = 96
SEMIPRIMES_PER_SIZE = 3  # subset of E29's 10 per size
THREADS = 4
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)


def parse_cado_output(output: str) -> dict:
    """Extract sieving metrics from CADO-NFS stdout."""
    clean = strip_ansi(output)
    result = {}

    # Lattice Sieving: Total number of relations: 63610
    m = re.search(r'Lattice Sieving: Total number of relations:\s*(\d+)', clean)
    if m:
        result['sieve_relations'] = int(m.group(1))

    # Lattice Sieving: Total time: 1.81s
    m = re.search(r'Lattice Sieving: Total time:\s*([\d.]+)s', clean)
    if m:
        result['sieve_time_s'] = float(m.group(1))

    # Total cpu/elapsed time for entire Complete Factorization 4.36/7.53952
    m = re.search(r'Total cpu/elapsed time for entire Complete Factorization\s+([\d.]+)/([\d.]+)', clean)
    if m:
        result['total_cpu_s'] = float(m.group(1))
        result['total_wall_s'] = float(m.group(2))

    # Polynomial Selection (size optimized): Total time: 0.11
    m = re.search(r'Polynomial Selection \(size optimized\): Total time:\s*([\d.]+)', clean)
    if m:
        result['polyselect_time_s'] = float(m.group(1))

    # Lattice Sieving: Average J: 256 for 316 special-q
    m = re.search(r'Average J:\s*(\d+)\s+for\s+(\d+)\s+special-q', clean)
    if m:
        result['avg_J'] = int(m.group(1))
        result['num_special_q'] = int(m.group(2))

    # Extract factors from last line
    lines = clean.strip().split('\n')
    last_line = lines[-1].strip()
    parts = last_line.split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        result['factor1'] = int(parts[0])
        result['factor2'] = int(parts[1])

    if 'sieve_relations' in result and 'sieve_time_s' in result and result['sieve_time_s'] > 0:
        result['sieve_rels_per_sec'] = result['sieve_relations'] / result['sieve_time_s']

    return result


def run_cado(n_decimal: int, n_hex: str, bits: int) -> dict:
    """Run CADO-NFS on a single semiprime and return metrics."""
    workdir = WORKDIR_BASE / f"{bits}bit_{n_hex}"
    if workdir.exists():
        shutil.rmtree(workdir)

    cmd = [
        sys.executable, CADO_NFS_PY,
        str(n_decimal),
        "--workdir", str(workdir),
        "-t", str(THREADS),
    ]

    print(f"  Running CADO-NFS on {bits}-bit N={n_hex}...", flush=True)
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        elapsed = time.time() - start
        output = proc.stdout + proc.stderr
        metrics = parse_cado_output(output)
        metrics['script_wall_s'] = elapsed
        metrics['success'] = proc.returncode == 0

        # Validate factors
        if 'factor1' in metrics and 'factor2' in metrics:
            product = metrics['factor1'] * metrics['factor2']
            metrics['factors_valid'] = (product == n_decimal)
        else:
            metrics['factors_valid'] = False

        if not metrics['success']:
            print(f"    FAILED (exit code {proc.returncode})", flush=True)
        else:
            rels = metrics.get('sieve_relations', '?')
            st = metrics.get('sieve_time_s', '?')
            rps = metrics.get('sieve_rels_per_sec', '?')
            if isinstance(rps, float):
                rps = f"{rps:.0f}"
            print(f"    OK: {rels} relations in {st}s ({rps} rels/sec), total {elapsed:.1f}s", flush=True)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        metrics = {'success': False, 'error': 'timeout', 'script_wall_s': elapsed}
        print(f"    TIMEOUT after {elapsed:.0f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - start
        metrics = {'success': False, 'error': str(e), 'script_wall_s': elapsed}
        print(f"    ERROR: {e}", flush=True)
    finally:
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)

    return metrics


def main():
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    # Build MCMC lookup: bits -> per_bit_size entry
    mcmc_by_bits = {entry['bits']: entry for entry in summary['per_bit_size']}

    # Collect semiprimes grouped by bits
    semiprimes_by_bits = {}
    for sp in summary['per_semiprime']:
        b = sp['bits']
        if b < MIN_BITS:
            continue
        if b not in semiprimes_by_bits:
            semiprimes_by_bits[b] = []
        semiprimes_by_bits[b].append(sp)

    results = []
    per_bit_results = []

    for bits in sorted(semiprimes_by_bits.keys()):
        sps = semiprimes_by_bits[bits][:SEMIPRIMES_PER_SIZE]
        print(f"\n--- {bits} bits ({len(sps)} semiprimes) ---", flush=True)

        bit_cado_results = []
        for sp in sps:
            n_hex = sp['n_hex']
            n_dec = int(n_hex, 16)

            cado_metrics = run_cado(n_dec, n_hex, bits)

            entry = {
                'n_hex': n_hex,
                'bits': bits,
                'n_decimal': str(n_dec),
                'mcmc_unique_smooth_rate': sp['mcmc']['unique_both_smooth'] / max(sp['mcmc']['unique_valid'], 1) if sp['mcmc']['unique_valid'] > 0 else 0,
                'mcmc_unique_both_smooth': sp['mcmc']['unique_both_smooth'],
                'mcmc_time_ms': sp['mcmc']['time_ms'],
                'mcmc_rels_per_sec': sp['mcmc']['unique_both_smooth'] / (sp['mcmc']['time_ms'] / 1000.0) if sp['mcmc']['time_ms'] > 0 and sp['mcmc']['unique_both_smooth'] > 0 else 0,
                'cado': cado_metrics,
            }
            results.append(entry)
            if cado_metrics.get('success'):
                bit_cado_results.append(cado_metrics)

        # Aggregate per bit size
        mcmc_entry = mcmc_by_bits.get(bits, {})
        mcmc_rps = mcmc_entry.get('mcmc_mean_rels_per_sec', 0)

        if bit_cado_results:
            cado_mean_rels = sum(r.get('sieve_relations', 0) for r in bit_cado_results) / len(bit_cado_results)
            cado_mean_time = sum(r.get('sieve_time_s', 0) for r in bit_cado_results) / len(bit_cado_results)
            cado_mean_rps = sum(r.get('sieve_rels_per_sec', 0) for r in bit_cado_results) / len(bit_cado_results)
            cado_mean_total = sum(r.get('total_wall_s', 0) for r in bit_cado_results) / len(bit_cado_results)
            mcmc_vs_cado = mcmc_rps / cado_mean_rps if cado_mean_rps > 0 else None
        else:
            cado_mean_rels = None
            cado_mean_time = None
            cado_mean_rps = None
            cado_mean_total = None
            mcmc_vs_cado = None

        bit_summary = {
            'bits': bits,
            'cado_semiprimes_tested': len(bit_cado_results),
            'mcmc_mean_unique_smooth_rate': mcmc_entry.get('mcmc_mean_unique_smooth_rate', 0),
            'mcmc_mean_rels_per_sec': mcmc_rps,
            'mcmc_mean_unique_both_smooth': mcmc_entry.get('mcmc_mean_unique_both_smooth', 0),
            'cado_mean_sieve_relations': cado_mean_rels,
            'cado_mean_sieve_time_s': cado_mean_time,
            'cado_mean_sieve_rels_per_sec': cado_mean_rps,
            'cado_mean_total_wall_s': cado_mean_total,
            'mcmc_vs_cado_ratio': mcmc_vs_cado,
        }
        per_bit_results.append(bit_summary)

        # Print summary
        if cado_mean_rps:
            ratio_str = f"{mcmc_vs_cado:.4f}" if mcmc_vs_cado else "N/A"
            print(f"  Summary: CADO {cado_mean_rps:.0f} rels/sec, MCMC {mcmc_rps:.0f} rels/sec, ratio {ratio_str}", flush=True)

    output = {
        'experiment': 'E29 CADO-NFS vs MCMC head-to-head',
        'cado_binary': CADO_NFS_PY,
        'threads': THREADS,
        'semiprimes_per_size': SEMIPRIMES_PER_SIZE,
        'per_bit_size': per_bit_results,
        'per_semiprime': results,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
