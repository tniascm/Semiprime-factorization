"""
E10: Integer-carry signals breaking CRT separability
=====================================================

Motivation
----------
All previous experiments (E7-E7e) tested functions on Z/NZ that are
CRT-separable: f(t) = g(t mod p) * h(t mod q). Their DFTs factor as
f^(xi) = g^(xi mod p) * h^(xi mod q), yielding flat spectra with no
factor-localized peaks (peak ~ N^{-0.25}).

However, the INTEGER representation t in [0, N) introduces non-separable
coupling via "carry bits." The CRT reconstruction
  t = a*q*q_inv + b*p*p_inv  (mod N)
involves a carry c = floor((a*q*q_inv + b*p*p_inv)/N) in {0,1}.
Functions that depend on this carry (e.g., floor(t^2/N)) are NOT rank-1
CRT-separable, even though they are computable in poly(log N) time.

Signals tested
--------------
1. f_jacobi(t) = J(t^2-4, N)             -- CONTROL (known flat, E7c)
2. f_carry_jacobi(t) = J(floor(t^2/N), N) -- Jacobi of "high word" of t^2
3. f_carry_parity(t) = (-1)^floor(t^2/N)  -- parity of integer quotient
4. f_mixed(t) = J(t,N) * (-1)^floor(2t/sqrt(N))
5. f_carry_sum(t) = sum_{k=1}^{10} (-1)^floor(k*t^2/N)
6. f_lattice(t) = J(t^2 mod floor(sqrt(N)), N)

All signals are poly(log N)-computable. Signals 2-6 involve integer-carry
operations that break CRT separability.

Metrics
-------
- DFT peak-to-bulk ratio
- Factor-localized energy fraction (gcd(xi,N) > 1)
- Scaling exponents across N sizes
- CRT rank estimate via SVD
"""

import sys
import os
import json
import time

import numpy as np
from scipy import stats

# ── semiprime generation ──────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import generate_semiprimes

# ── Sage type coercion ────────────────────────────────────────────────

def _py(v):
    """Convert Sage types to native Python for JSON serialization."""
    if isinstance(v, (bool, type(None), str)):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    # Sage RealDoubleElement, Integer, Rational, etc.
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def _py_dict(d):
    """Recursively convert all values in a dict for JSON."""
    return {k: _py(v) for k, v in d.items()}


class SageEncoder(json.JSONEncoder):
    """JSON encoder that handles Sage types."""
    def default(self, obj):
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        try:
            return int(obj)
        except (TypeError, ValueError):
            return str(obj)

# ── Signal definitions ────────────────────────────────────────────────

def compute_signals(N):
    """
    Compute all 6 signals for semiprime N. Returns dict: name -> np.array of length N.
    """
    N_int = int(N)
    sqrtN = int(isqrt(N_int))

    # Precompute t^2 and related quantities for all t in [0, N)
    signals = {}

    # Signal 1: CONTROL — pure Jacobi (known flat from E7c)
    sig1 = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        disc = t * t - 4
        if disc == 0:
            sig1[t] = 0.0
        else:
            sig1[t] = float(kronecker_symbol(disc, N_int))
    signals['jacobi_control'] = sig1

    # Signal 2: Jacobi of floor(t^2 / N) — carry-based
    sig2 = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        high_word = (t * t) // N_int
        if high_word == 0:
            sig2[t] = 0.0
        else:
            g = gcd(high_word, N_int)
            if g > 1:
                sig2[t] = 0.0  # Jacobi undefined when gcd > 1
            else:
                sig2[t] = float(kronecker_symbol(high_word, N_int))
    signals['carry_jacobi'] = sig2

    # Signal 3: (-1)^floor(t^2 / N) — carry parity
    sig3 = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        high_word = (t * t) // N_int
        sig3[t] = 1.0 if (high_word % 2 == 0) else -1.0
    signals['carry_parity'] = sig3

    # Signal 4: J(t, N) * (-1)^floor(2t / sqrt(N)) — Jacobi x approx-root sign
    sig4 = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        j_val = kronecker_symbol(t, N_int)
        if j_val == 0:
            sig4[t] = 0.0
        else:
            sign_bit = (2 * t) // sqrtN
            sig4[t] = float(j_val) * (1.0 if (sign_bit % 2 == 0) else -1.0)
    signals['mixed_jacobi_carry'] = sig4

    # Signal 5: sum_{k=1}^{10} (-1)^floor(k*t^2/N) — accumulated carry oscillation
    sig5 = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        t2 = t * t
        acc = 0.0
        for k in range(1, 11):
            high_word = (k * t2) // N_int
            acc += 1.0 if (high_word % 2 == 0) else -1.0
        sig5[t] = acc
    signals['carry_sum'] = sig5

    # Signal 6: J(t^2 mod floor(sqrt(N)), N) — lattice-reduced Jacobi
    sig6 = np.zeros(N_int, dtype=np.float64)
    if sqrtN > 1:
        for t in range(N_int):
            reduced = (t * t) % sqrtN
            if reduced == 0:
                sig6[t] = 0.0
            else:
                g = gcd(reduced, N_int)
                if g > 1:
                    sig6[t] = 0.0
                else:
                    sig6[t] = float(kronecker_symbol(reduced, N_int))
    signals['lattice_jacobi'] = sig6

    return signals


# ── DFT and spectral analysis ────────────────────────────────────────

def analyze_signal(sig, N, p, q):
    """
    Compute DFT and extract spectral metrics for a signal on Z/NZ.
    Returns dict of metrics.
    """
    N_int = int(N)
    X = np.fft.fft(sig) / N_int  # Normalized DFT
    mags = np.abs(X)
    mags2 = mags ** 2

    # Skip DC component for spectral analysis
    mags_ac = mags[1:]
    mags2_ac = mags2[1:]

    # Peak-to-bulk ratio
    peak = np.max(mags_ac)
    median_val = np.median(mags_ac)
    peak_to_bulk = float(peak / median_val) if median_val > 0 else float('inf')

    # Factor-localized energy: energy at xi with gcd(xi, N) > 1
    energy_total = float(np.sum(mags2_ac))
    energy_factor = 0.0
    energy_p = 0.0
    energy_q = 0.0
    for xi in range(1, N_int):
        g = gcd(xi, N_int)
        if g > 1:
            energy_factor += mags2[xi]
        if xi % p == 0:
            energy_p += mags2[xi]
        if xi % q == 0:
            energy_q += mags2[xi]

    factor_frac = float(energy_factor / energy_total) if energy_total > 0 else 0.0
    p_frac = float(energy_p / energy_total) if energy_total > 0 else 0.0
    q_frac = float(energy_q / energy_total) if energy_total > 0 else 0.0

    # Expected factor energy under flat spectrum: (q-1+p-1+1-1)/(N-1) = (p+q-2)/(N-1)
    n_factor_modes = 0
    for xi in range(1, N_int):
        if gcd(xi, N_int) > 1:
            n_factor_modes += 1
    expected_factor_frac = float(n_factor_modes) / (N_int - 1)

    # Top-mode factor extraction: do the K largest modes have gcd > 1?
    sorted_indices = np.argsort(mags_ac)[::-1] + 1  # +1 to skip DC
    top_10_factor = sum(1 for idx in sorted_indices[:10] if gcd(int(idx), N_int) > 1)
    top_5_factor = sum(1 for idx in sorted_indices[:5] if gcd(int(idx), N_int) > 1)

    # CRT rank estimate: reshape signal as p x q matrix and compute SVD
    # t = CRT(a, b) maps Z/NZ -> Z/pZ x Z/qZ
    M = np.zeros((int(p), int(q)), dtype=np.float64)
    for t in range(N_int):
        a = t % p
        b = t % q
        M[a, b] = sig[t]
    sv = np.linalg.svd(M, compute_uv=False)
    sv_total = float(np.sum(sv))
    # Effective rank: number of singular values for 90% of total
    if sv_total > 0:
        sv_norm = sv / sv_total
        cumsum = np.cumsum(sv_norm)
        eff_rank_90 = int(np.searchsorted(cumsum, 0.90)) + 1
    else:
        eff_rank_90 = 0
    # Nuclear norm ratio: sum(sv) / max(sv) — rank-1 has ratio 1
    nuclear_ratio = float(sv_total / sv[0]) if sv[0] > 0 else 0.0

    return {
        'peak': float(peak),
        'median': float(median_val),
        'peak_to_bulk': peak_to_bulk,
        'energy_total': energy_total,
        'factor_energy_frac': factor_frac,
        'p_energy_frac': p_frac,
        'q_energy_frac': q_frac,
        'expected_factor_frac': expected_factor_frac,
        'factor_energy_excess': factor_frac / expected_factor_frac if expected_factor_frac > 0 else 0.0,
        'top_5_factor_count': top_5_factor,
        'top_10_factor_count': top_10_factor,
        'crt_eff_rank_90': eff_rank_90,
        'crt_nuclear_ratio': nuclear_ratio,
        'crt_top_sv': float(sv[0]),
        'crt_sv_2': float(sv[1]) if len(sv) > 1 else 0.0,
    }

# ── Main experiment ───────────────────────────────────────────────────

def main():
    print("=" * 72, flush=True)
    print("E10: Integer-carry signals — CRT separability probe", flush=True)
    print("=" * 72, flush=True)

    # Quick probe: N ~ 10^3 to 10^4, ~30 semiprimes per size class
    size_classes = [
        (500, 2000, 30),     # small
        (2000, 5000, 30),    # medium
        (5000, 12000, 30),   # large
    ]

    signal_names = [
        'jacobi_control',
        'carry_jacobi',
        'carry_parity',
        'mixed_jacobi_carry',
        'carry_sum',
        'lattice_jacobi',
    ]

    all_results = []

    for min_N, max_N, n_samples in size_classes:
        print(f"\n--- Size class: N in [{min_N}, {max_N}], {n_samples} samples ---",
              flush=True)
        semiprimes = generate_semiprimes(max_N, num_samples=n_samples, min_N=min_N)
        print(f"Generated {len(semiprimes)} semiprimes", flush=True)

        # Header
        hdr = f"{'N':>7} {'p':>5} {'q':>5} {'signal':>20} {'peak/bulk':>10} {'fac_E':>7} {'excess':>7} {'rank90':>7} {'time':>7}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

        for N, p, q in semiprimes:
            t0 = time.time()
            sigs = compute_signals(N)
            dt_signals = time.time() - t0

            for sname in signal_names:
                t1 = time.time()
                metrics = analyze_signal(sigs[sname], N, p, q)
                dt_analysis = time.time() - t1

                row = {
                    'N': int(N),
                    'p': int(p),
                    'q': int(q),
                    'ratio_pq': float(min(p, q)) / float(max(p, q)),
                    'signal': sname,
                }
                row.update(_py_dict(metrics))
                row['time_signals'] = round(float(dt_signals), 4)
                row['time_analysis'] = round(float(dt_analysis), 4)
                all_results.append(row)

                print(f"{N:>7} {p:>5} {q:>5} {sname:>20} "
                      f"{metrics['peak_to_bulk']:>10.2f} "
                      f"{metrics['factor_energy_frac']:>7.4f} "
                      f"{metrics['factor_energy_excess']:>7.3f} "
                      f"{metrics['crt_eff_rank_90']:>7} "
                      f"{dt_analysis:>7.3f}s", flush=True)

    # ── Save JSON results ─────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E10_carry_signals_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=int(2), cls=SageEncoder)
    print(f"\nResults saved to {out_path}", flush=True)

    # ── Statistical analysis: scaling exponents per signal ────────────
    print("\n" + "=" * 72, flush=True)
    print("SCALING ANALYSIS", flush=True)
    print("=" * 72, flush=True)

    for sname in signal_names:
        rows = [r for r in all_results if r['signal'] == sname]
        Ns = np.array([r['N'] for r in rows], dtype=np.float64)
        peaks = np.array([r['peak'] for r in rows], dtype=np.float64)
        fac_excess = np.array([r['factor_energy_excess'] for r in rows], dtype=np.float64)
        ranks = np.array([r['crt_eff_rank_90'] for r in rows], dtype=np.float64)

        print(f"\n--- {sname} ---", flush=True)

        # Peak scaling: peak ~ N^alpha
        mask = peaks > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(peaks[mask]))
            print(f"  Peak scaling:    peak ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)
        else:
            print(f"  Peak scaling:    insufficient data", flush=True)

        # Factor energy excess scaling
        mask = fac_excess > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(fac_excess[mask]))
            print(f"  Excess scaling:  excess ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)
        else:
            print(f"  Excess scaling:  insufficient data", flush=True)

        # CRT rank scaling
        mask = ranks > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(ranks[mask]))
            print(f"  Rank scaling:    rank90 ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)
        else:
            print(f"  Rank scaling:    insufficient data", flush=True)

        # Mean metrics
        mean_ptb = np.mean([r['peak_to_bulk'] for r in rows])
        mean_excess = np.mean(fac_excess)
        mean_rank = np.mean(ranks)
        print(f"  Mean peak/bulk:  {mean_ptb:.2f}", flush=True)
        print(f"  Mean excess:     {mean_excess:.3f}", flush=True)
        print(f"  Mean CRT rank90: {mean_rank:.1f}", flush=True)

    # ── Comparative summary ───────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("COMPARATIVE SUMMARY", flush=True)
    print("=" * 72, flush=True)

    print(f"\n{'signal':>20} {'mean_peak/bulk':>14} {'mean_fac_excess':>15} "
          f"{'mean_rank90':>12} {'peak_alpha':>11}", flush=True)
    print("-" * 75, flush=True)

    for sname in signal_names:
        rows = [r for r in all_results if r['signal'] == sname]
        Ns = np.array([r['N'] for r in rows], dtype=np.float64)
        peaks = np.array([r['peak'] for r in rows], dtype=np.float64)
        fac_excess = np.array([r['factor_energy_excess'] for r in rows], dtype=np.float64)
        ranks = np.array([r['crt_eff_rank_90'] for r in rows], dtype=np.float64)

        mean_ptb = np.mean([r['peak_to_bulk'] for r in rows])
        mean_excess = np.mean(fac_excess)
        mean_rank = np.mean(ranks)

        mask = peaks > 0
        if np.sum(mask) >= 3:
            sl, _, _, _, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(peaks[mask]))
            alpha_str = f"{sl:.3f}+/-{se:.3f}"
        else:
            alpha_str = "N/A"

        print(f"{sname:>20} {mean_ptb:>14.2f} {mean_excess:>15.3f} "
              f"{mean_rank:>12.1f} {alpha_str:>11}", flush=True)

    # ── Plotting ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('E10: Integer-carry signals — DFT peak scaling', fontsize=14)

        for idx, sname in enumerate(signal_names):
            ax = axes[idx // 3, idx % 3]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            peaks = np.array([r['peak'] for r in rows])

            ax.scatter(Ns, peaks, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('DFT peak')
            ax.set_title(sname)

            mask = peaks > 0
            if np.sum(mask) >= 3:
                sl, intc, rv, _, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(peaks[mask]))
                x_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                y_fit = 10**intc * x_fit**sl
                ax.plot(x_fit, y_fit, 'r-', lw=1.5,
                        label=f'N^{{{sl:.3f}}} (R2={rv**2:.3f})')
                ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E10_carry_signals_peaks.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}", flush=True)

        # Second plot: factor energy excess
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        fig2.suptitle('E10: Factor-localized energy excess by signal', fontsize=14)

        for idx, sname in enumerate(signal_names):
            ax = axes2[idx // 3, idx % 3]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            excess = np.array([r['factor_energy_excess'] for r in rows])

            ax.scatter(Ns, excess, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.axhline(y=1.0, color='gray', ls='--', lw=0.8, label='flat (excess=1)')
            ax.set_xlabel('N')
            ax.set_ylabel('Factor energy excess')
            ax.set_title(sname)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path2 = os.path.join(data_dir, 'E10_carry_signals_excess.png')
        plt.savefig(plot_path2, dpi=150)
        print(f"Plot saved to {plot_path2}", flush=True)

        # Third plot: CRT rank
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
        fig3.suptitle('E10: CRT effective rank (90% energy) by signal', fontsize=14)

        for idx, sname in enumerate(signal_names):
            ax = axes3[idx // 3, idx % 3]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            ranks = np.array([r['crt_eff_rank_90'] for r in rows])

            ax.scatter(Ns, ranks, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('CRT rank (90%)')
            ax.set_title(sname)

            mask = ranks > 0
            if np.sum(mask) >= 3:
                sl, intc, rv, _, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(ranks[mask]))
                x_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                y_fit = 10**intc * x_fit**sl
                ax.plot(x_fit, y_fit, 'r-', lw=1.5,
                        label=f'N^{{{sl:.3f}}} (R2={rv**2:.3f})')
                ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path3 = os.path.join(data_dir, 'E10_carry_signals_rank.png')
        plt.savefig(plot_path3, dpi=150)
        print(f"Plot saved to {plot_path3}", flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("VERDICT", flush=True)
    print("=" * 72, flush=True)

    # Compare carry signals to control
    control_rows = [r for r in all_results if r['signal'] == 'jacobi_control']
    control_mean_excess = np.mean([r['factor_energy_excess'] for r in control_rows])
    control_mean_ptb = np.mean([r['peak_to_bulk'] for r in control_rows])

    any_promising = False
    for sname in signal_names[1:]:  # skip control
        rows = [r for r in all_results if r['signal'] == sname]
        mean_excess = np.mean([r['factor_energy_excess'] for r in rows])
        mean_ptb = np.mean([r['peak_to_bulk'] for r in rows])

        if mean_excess > 1.5 * control_mean_excess or mean_ptb > 2 * control_mean_ptb:
            print(f"  PROMISING: {sname} shows elevated signal", flush=True)
            print(f"    excess={mean_excess:.3f} vs control={control_mean_excess:.3f}", flush=True)
            print(f"    peak/bulk={mean_ptb:.2f} vs control={control_mean_ptb:.2f}", flush=True)
            any_promising = True

    if not any_promising:
        print("  No carry signal shows elevated factor energy or peaks", flush=True)
        print("  compared to the pure-Jacobi control.", flush=True)
        print("  → Integer-carry operations do NOT break spectral flatness", flush=True)
        print("  → The CRT barrier extends to the integer-arithmetic oracle model", flush=True)
    else:
        print("  Some carry signals show elevated metrics.", flush=True)
        print("  → Extend to larger N (10^6) to confirm scaling.", flush=True)

    print("\nDone.", flush=True)

if __name__ == '__main__':
    main()
