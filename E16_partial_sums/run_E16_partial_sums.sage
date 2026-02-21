"""
E16: Partial-Range Exponential Sums
=====================================

Full Gauss sums decompose via CRT: G(chi, N) = G(chi_p, p) * G(chi_q, q).
But partial sums over [0, M) for M not a multiple of p or q are NOT
CRT-decomposable. The CRT obstruction proof assumes the sum runs over
all of Z/NZ.

This experiment tests whether partial ("truncated") exponential sums
with Jacobi weight show factor-correlated peaks.

Signals tested
--------------
For each semiprime N = pq and cutoff M:

1. Untwisted partial sum: S(xi, M) = sum_{t=0}^{M-1} e^{2*pi*i*xi*t/N}
   (Geometric series — analytically known, serves as baseline)

2. Jacobi-twisted partial sum: T(xi, M) = sum_{t=0}^{M-1} J(t,N) * e^{2*pi*i*xi*t/N}
   (This is the interesting case — NOT CRT-decomposable)

3. Carry-twisted partial sum: C(xi, M) = sum_{t=0}^{M-1} floor(t^2/N) * e^{2*pi*i*xi*t/N}
   (Carry function weight, also non-CRT-separable)

Cutoffs: M in {N//3, N//2, 2*N//3, isqrt(N), N - isqrt(N)}

For each (signal, cutoff), we compute |S(xi)| for all xi in [0,N) and
analyze factor-localized energy, peak location, and scaling with N.
"""
import sys
import os
import time

import numpy as np
from scipy import stats

set_random_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import generate_semiprimes
from sage_encoding import safe_json_dump
from spectral import precompute_gcd_classes

# ============================================================
# Partial exponential sum computation
# ============================================================

def partial_exp_sum(weights, N, M, xi):
    """
    Compute a single partial exponential sum:
        S(xi) = sum_{t=0}^{M-1} w(t) * e^{-2*pi*i*xi*t/N}

    Parameters
    ----------
    weights : array of length N (only [0, M) used)
    N : int
    M : int (cutoff, M <= N)
    xi : int (frequency)

    Returns complex value.
    """
    N_int = int(N)
    M_int = int(M)
    xi_int = int(xi)
    t = np.arange(M_int)
    phases = np.exp(-2.0j * np.pi * xi_int * t / N_int)
    return np.sum(weights[:M_int] * phases)


def partial_exp_sum_all_freqs(weights, N, M):
    """
    Compute partial exponential sums for ALL frequencies xi in [0, N).

    S(xi) = sum_{t=0}^{M-1} w(t) * e^{-2*pi*i*xi*t/N}

    Uses vectorized computation via outer product.
    For N up to ~15000 this is feasible (N x M matrix).
    """
    N_int = int(N)
    M_int = int(M)
    t = np.arange(M_int)
    xi = np.arange(N_int)
    # phase_matrix[xi, t] = e^{-2*pi*i*xi*t/N}, shape (N, M)
    phase_matrix = np.exp(-2.0j * np.pi * np.outer(xi, t) / N_int)
    # S[xi] = sum_t w(t) * phase[xi, t]
    S = phase_matrix @ weights[:M_int]
    return S


def compute_weights(N):
    """Compute weight signals for partial sums.

    Returns dict: name -> array of length N.
    """
    N_int = int(N)
    weights = {}

    # Untwisted (constant weight = 1)
    weights['untwisted'] = np.ones(N_int, dtype=np.float64)

    # Jacobi-twisted: J(t, N)
    jac = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        if t == 0:
            jac[t] = 0.0
        else:
            jac[t] = float(kronecker_symbol(t, N_int))
    weights['jacobi'] = jac

    # Carry-twisted: floor(t^2 / N)
    carry = np.zeros(N_int, dtype=np.float64)
    for t in range(N_int):
        carry[t] = float((t * t) // N_int)
    weights['carry'] = carry

    return weights


# ============================================================
# Spectral analysis of partial sums
# ============================================================

def analyze_partial_sum(S, N, p, q, gcd_class, n_factor_modes):
    """Analyze a partial exponential sum spectrum.

    Parameters
    ----------
    S : complex array of length N (partial sum at each frequency)
    N, p, q : int
    gcd_class, n_factor_modes : precomputed from spectral.py

    Returns dict of metrics.
    """
    N_int = int(N)
    mags = np.abs(S)
    mags2 = mags ** 2

    # Skip DC (xi=0)
    mags_ac = mags[1:]
    mags2_ac = mags2[1:]

    peak = float(np.max(mags_ac))
    peak_idx = int(np.argmax(mags_ac)) + 1  # +1 because we skipped DC
    median_val = float(np.median(mags_ac))
    peak_to_bulk = float(peak / median_val) if median_val > 0 else float('inf')

    # Energy analysis
    energy_total = float(np.sum(mags2_ac))
    factor_mask = gcd_class[1:] > 0
    energy_factor = float(np.sum(mags2_ac[factor_mask]))
    factor_frac = float(energy_factor / energy_total) if energy_total > 0 else 0.0
    expected_frac = float(n_factor_modes) / (N_int - 1) if N_int > 1 else 0.0
    excess = float(factor_frac / expected_frac) if expected_frac > 0 else 0.0

    # Peak at factor frequency?
    peak_at_factor = 1 if gcd_class[peak_idx] > 0 else 0

    # Top-5 modes: how many at factor frequencies?
    sorted_idx = np.argsort(mags_ac)[::-1]
    top5_factor = sum(1 for i in sorted_idx[:5] if gcd_class[i + 1] > 0)
    top10_factor = sum(1 for i in sorted_idx[:10] if gcd_class[i + 1] > 0)

    return {
        'peak': peak,
        'peak_idx': peak_idx,
        'peak_at_factor': peak_at_factor,
        'median': median_val,
        'peak_to_bulk': peak_to_bulk,
        'energy_total': energy_total,
        'factor_energy_frac': factor_frac,
        'expected_factor_frac': expected_frac,
        'factor_energy_excess': excess,
        'top5_factor': top5_factor,
        'top10_factor': top10_factor,
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 76, flush=True)
    print("E16: Partial-Range Exponential Sums", flush=True)
    print("=" * 76, flush=True)

    # Size classes — keep N modest since we build N x M matrices
    size_classes = [
        (500, 2000, 20),
        (2000, 5000, 20),
        (5000, 12000, 15),
    ]

    weight_names = ['untwisted', 'jacobi', 'carry']

    all_results = []

    for min_N, max_N, n_samples in size_classes:
        print("\n--- Size class: N in [%d, %d], %d samples ---" % (
            min_N, max_N, n_samples), flush=True)
        semiprimes = generate_semiprimes(max_N, num_samples=n_samples, min_N=min_N)
        print("Generated %d semiprimes" % len(semiprimes), flush=True)

        for N, p, q in semiprimes:
            N_int = int(N)
            sqrtN = int(isqrt(N_int))

            # Define cutoffs
            cutoffs = OrderedDict([
                ('N//3', N_int // 3),
                ('N//2', N_int // 2),
                ('2N//3', 2 * N_int // 3),
                ('sqrt(N)', sqrtN),
                ('N-sqrt(N)', N_int - sqrtN),
            ])

            # Precompute
            gcd_class, n_factor_modes = precompute_gcd_classes(N_int, int(p), int(q))
            all_weights = compute_weights(N_int)

            t0 = time.time()

            for wname in weight_names:
                w = all_weights[wname]
                for cname, M in cutoffs.items():
                    if M < 2 or M > N_int:
                        continue

                    # Compute partial sum for all frequencies
                    S = partial_exp_sum_all_freqs(w, N_int, M)

                    # Analyze
                    metrics = analyze_partial_sum(S, N_int, p, q, gcd_class, n_factor_modes)

                    # Also compute full sum for comparison (M=N)
                    S_full = partial_exp_sum_all_freqs(w, N_int, N_int)
                    metrics_full = analyze_partial_sum(S_full, N_int, p, q, gcd_class, n_factor_modes)

                    row = {
                        'N': int(N),
                        'p': int(p),
                        'q': int(q),
                        'ratio_pq': float(min(p, q)) / float(max(p, q)),
                        'weight': wname,
                        'cutoff_name': cname,
                        'cutoff_M': int(M),
                        'cutoff_frac': float(M) / float(N_int),
                    }
                    # Partial sum metrics
                    for k, v in metrics.items():
                        row['partial_%s' % k] = float(v)
                    # Full sum metrics for comparison
                    for k, v in metrics_full.items():
                        row['full_%s' % k] = float(v)

                    all_results.append(row)

            dt = time.time() - t0
            print("N=%d (p=%d, q=%d): %.2fs, %d tests" % (
                N_int, p, q, dt, len(cutoffs) * len(weight_names)), flush=True)

    # --- Save JSON ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E16_partial_sums_results.json')
    safe_json_dump(all_results, out_path)

    # --- Scaling analysis ---
    print("\n" + "=" * 76, flush=True)
    print("SCALING ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    scaling_results = {}

    for wname in weight_names:
        print("\n=== Weight: %s ===" % wname, flush=True)
        cutoff_names = list(OrderedDict([
            ('N//3', None), ('N//2', None), ('2N//3', None),
            ('sqrt(N)', None), ('N-sqrt(N)', None),
        ]).keys())

        for cname in cutoff_names:
            rows = [r for r in all_results
                    if r['weight'] == wname and r['cutoff_name'] == cname]
            if len(rows) < 5:
                continue

            Ns = np.array([r['N'] for r in rows], dtype=np.float64)
            peaks = np.array([r['partial_peak'] for r in rows], dtype=np.float64)
            excess = np.array([r['partial_factor_energy_excess'] for r in rows], dtype=np.float64)
            full_excess = np.array([r['full_factor_energy_excess'] for r in rows], dtype=np.float64)

            # Peak scaling
            mask = peaks > 0
            alpha, alpha_se, alpha_r2 = 0, 0, 0
            if np.sum(mask) >= 3:
                sl, intc, rv, pv, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(peaks[mask]))
                alpha, alpha_se, alpha_r2 = sl, se, rv**2

            # Excess statistics
            mean_excess = float(np.mean(excess))
            mean_full_excess = float(np.mean(full_excess))
            excess_improvement = mean_excess - mean_full_excess

            key = '%s_%s' % (wname, cname)
            scaling_results[key] = {
                'weight': wname,
                'cutoff': cname,
                'alpha': float(alpha),
                'alpha_se': float(alpha_se),
                'alpha_r2': float(alpha_r2),
                'mean_partial_excess': float(mean_excess),
                'mean_full_excess': float(mean_full_excess),
                'excess_improvement': float(excess_improvement),
                'n_samples': len(rows),
            }

            print("  %-12s: peak ~ N^%.3f +/- %.3f  excess=%.3f (full=%.3f, diff=%+.3f)" % (
                cname, alpha, alpha_se, mean_excess, mean_full_excess, excess_improvement),
                flush=True)

    # --- Comparative summary ---
    print("\n" + "=" * 76, flush=True)
    print("COMPARATIVE SUMMARY", flush=True)
    print("=" * 76, flush=True)

    print("\n%-10s %-12s %10s %10s %10s %12s" % (
        'weight', 'cutoff', 'peak_alpha', 'partial_ex', 'full_ex', 'improvement'), flush=True)
    print("-" * 68, flush=True)

    for key in sorted(scaling_results.keys()):
        sr = scaling_results[key]
        print("%-10s %-12s %+10.3f %10.3f %10.3f %+12.3f" % (
            sr['weight'], sr['cutoff'], sr['alpha'],
            sr['mean_partial_excess'], sr['mean_full_excess'],
            sr['excess_improvement']), flush=True)

    # --- Verdict ---
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    any_promising = False
    for key, sr in scaling_results.items():
        if sr['weight'] == 'untwisted':
            continue  # untwisted is analytically known
        if sr['mean_partial_excess'] > 1.5:
            any_promising = True
            print("  PROMISING: %s has partial excess %.3f (full: %.3f)" % (
                key, sr['mean_partial_excess'], sr['mean_full_excess']), flush=True)
        if sr['alpha'] > -0.1 and sr['weight'] != 'untwisted':
            any_promising = True
            print("  PROMISING: %s has peak alpha = %.3f" % (key, sr['alpha']), flush=True)

    if not any_promising:
        print("  No partial-range sum shows elevated factor energy over full sum.", flush=True)
        print("  Truncation does NOT break the spectral flatness barrier.", flush=True)
        print("  The CRT obstruction extends beyond full-range sums.", flush=True)
    else:
        print("  Some partial sums show improvement — extend to larger N.", flush=True)

    results_summary = {
        'scaling': scaling_results,
        'verdict': 'PROMISING' if any_promising else 'BARRIER CONFIRMED',
        'n_total_tests': len(all_results),
    }
    summary_path = os.path.join(data_dir, 'E16_partial_sums_summary.json')
    safe_json_dump(results_summary, summary_path)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('E16: Partial-Range Exponential Sums', fontsize=14)

        # Plot 1-3: Factor energy excess by cutoff for each weight
        for idx, wname in enumerate(weight_names):
            ax = axes[0, idx]
            cutoff_names = ['N//3', 'N//2', '2N//3', 'sqrt(N)', 'N-sqrt(N)']
            for cname in cutoff_names:
                rows = [r for r in all_results
                        if r['weight'] == wname and r['cutoff_name'] == cname]
                if not rows:
                    continue
                Ns = [r['N'] for r in rows]
                ex = [r['partial_factor_energy_excess'] for r in rows]
                ax.scatter(Ns, ex, s=12, alpha=0.5, label=cname)

            # Full sum baseline
            rows_full = [r for r in all_results
                         if r['weight'] == wname and r['cutoff_name'] == 'N//2']
            if rows_full:
                Ns_f = [r['N'] for r in rows_full]
                ex_f = [r['full_factor_energy_excess'] for r in rows_full]
                ax.scatter(Ns_f, ex_f, s=12, alpha=0.3, marker='x', color='black',
                           label='full (M=N)')

            ax.axhline(y=1.0, color='gray', ls='--', lw=0.8)
            ax.set_xscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('Factor Energy Excess')
            ax.set_title('Weight: %s' % wname)
            ax.legend(fontsize=6, ncol=2)

        # Plot 4-5: Peak scaling for jacobi and carry
        for idx, wname in enumerate(['jacobi', 'carry']):
            ax = axes[1, idx]
            cutoff_names = ['N//3', 'N//2', '2N//3']
            for cname in cutoff_names:
                rows = [r for r in all_results
                        if r['weight'] == wname and r['cutoff_name'] == cname]
                if not rows:
                    continue
                Ns = np.array([r['N'] for r in rows], dtype=np.float64)
                peaks = np.array([r['partial_peak'] for r in rows], dtype=np.float64)
                ax.scatter(Ns, peaks, s=12, alpha=0.5, label=cname)

                mask = peaks > 0
                if np.sum(mask) >= 3:
                    sl, intc, rv, _, se = stats.linregress(
                        np.log10(Ns[mask]), np.log10(peaks[mask]))
                    xf = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                    ax.plot(xf, 10**intc * xf**sl, '--', lw=1,
                            label='%s: N^{%.2f}' % (cname, sl))

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('Peak |S(xi)|')
            ax.set_title('Peak Scaling: %s-weighted' % wname)
            ax.legend(fontsize=6)

        # Plot 6: Partial vs full excess comparison
        ax = axes[1, 2]
        for wname in ['jacobi', 'carry']:
            rows = [r for r in all_results
                    if r['weight'] == wname and r['cutoff_name'] == 'N//2']
            if not rows:
                continue
            partial_ex = [r['partial_factor_energy_excess'] for r in rows]
            full_ex = [r['full_factor_energy_excess'] for r in rows]
            ax.scatter(full_ex, partial_ex, s=12, alpha=0.5, label=wname)

        lims = [0, max(3, ax.get_xlim()[1])]
        ax.plot(lims, lims, 'k--', lw=0.8, label='y=x')
        ax.set_xlabel('Full Sum Factor Excess')
        ax.set_ylabel('Partial Sum Factor Excess (M=N//2)')
        ax.set_title('Partial vs Full: Factor Energy')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E16_partial_sums.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("\nPlot saved to %s" % plot_path, flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    from collections import OrderedDict
    main()
