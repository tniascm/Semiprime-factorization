"""
E15: Class Number Sum Anomaly Search
======================================

The Eichler-Selberg trace formula for Tr(T_n | S_k(Gamma_0(N))) contains a
class-number sum over trace values t with t^2 - 4n < 0. The Hurwitz class
number H(|t^2 - 4n|) appears as a weight in this sum.

For semiprimes N = pq, the "special" trace value t* = |p - q| satisfies:
    t*^2 - 4N = (p - q)^2 - 4pq = (p + q)^2 - 8pq - (4pq - (p-q)^2)
             = p^2 - 6pq + q^2

More directly: t* = |p - q| is in the summation range [1, 2*sqrt(N)]
because |p - q| < p + q and for balanced semiprimes (p ~ q ~ sqrt(N)),
|p - q| << 2*sqrt(N).

Key question: is H(4N - t*^2) anomalous compared to H(4N - t^2) for
generic t in the range?

NOTE: t = p + q is OUTSIDE the range since (p+q)^2 = (p-q)^2 + 4pq > 4N.
"""
import sys
import os
import time

import numpy as np
from scipy import stats

set_random_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump


# ============================================================
# Hurwitz class number computation
# ============================================================

def hurwitz_class_number(D):
    """
    Compute the Hurwitz class number H(D) for positive D.

    H(D) = h(D) for D > 4 (fundamental or not), with corrections:
        H(0) = -1/12, H(3) = 1/3, H(4) = 1/2
    For general D > 0:
        H(D) = sum over fundamental discriminants d | D of h(d) * ...

    We use Sage's class number computation for the quadratic order of
    discriminant -D (or the equivalent negative discriminant).

    For the trace formula, we need H(D) where D = 4N - t^2 > 0.
    """
    D = int(D)
    if D < 0:
        return 0
    if D == 0:
        return -1.0 / 12.0
    if D == 3:
        return 1.0 / 3.0
    if D == 4:
        return 0.5

    # H(D) = sum_{f^2 | D, D/f^2 ≡ 0,3 mod 4} h(-D/f^2) / w(-D/f^2)
    # where w is the number of units (= 1 for |disc| > 4)
    # For simplicity, use the weighted sum formula:
    #   H(D) = (2/w_D) * h(-D)  for fundamental discriminants
    # But for general D, we need the full sum over orders.

    # Compute via sum over conductors f with f^2 | D
    total = 0.0
    for f in range(1, int(isqrt(D)) + 1):
        if D % (f * f) != 0:
            continue
        d = D // (f * f)
        # d must give a valid negative discriminant
        # -d must be a discriminant: d ≡ 0 or 3 mod 4
        if d % 4 not in (0, 3):
            continue
        if d == 0:
            continue

        # Compute class number of imaginary quadratic order of discriminant -d
        try:
            h = int(QuadraticField(-d).class_number())
        except Exception:
            try:
                # Fallback: use number_of_classes for binary quadratic forms
                h = int(BinaryQF([1, 0, d]).class_number()) if d > 0 else 0
            except Exception:
                h = 0

        # Weight factor w: w=6 for d=3, w=4 for d=4, w=2 otherwise
        if d == 3:
            w = 6
        elif d == 4:
            w = 4
        else:
            w = 2

        total += float(h) * 2.0 / float(w)

    return total


def hurwitz_class_number_fast(D):
    """
    Fast Hurwitz class number using Sage's built-in.

    Uses the fact that H(D) for D > 0 can be computed via
    the number of reduced binary quadratic forms of discriminant -D.
    """
    D = int(D)
    if D <= 0:
        return 0.0
    if D == 3:
        return 1.0 / 3.0
    if D == 4:
        return 0.5

    # For the trace formula, the "Hurwitz class number" H(n) is:
    #   H(n) = 2 * h(-n) / w(-n) summed over orders of discriminant dividing -n
    # where h is class number and w is the number of roots of unity.

    # Simpler approach: count representations as sum of 3 squares (Gauss)
    # H(n) = (1/12) * (sum of terms from class number formula)

    # Most efficient for our use: compute h(-D) directly via Sage
    try:
        # Number of classes of primitive positive definite binary quadratic
        # forms of discriminant -D
        h = 0
        # Sum over all conductors f with f^2 | D
        for f in range(1, int(isqrt(D)) + 1):
            if D % (f * f) != 0:
                continue
            disc = -(D // (f * f))
            # Check valid discriminant
            neg_disc = -disc  # positive
            if neg_disc % 4 not in (0, 3):
                continue

            # Use Sage's hilbert_class_polynomial degree = class number
            try:
                K = QuadraticField(disc)
                cn = int(K.class_number())
            except Exception:
                cn = 0

            if neg_disc == 3:
                w = 6
            elif neg_disc == 4:
                w = 4
            else:
                w = 2
            h += cn * 2.0 / w

        return float(h)
    except Exception:
        return 0.0


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 76, flush=True)
    print("E15: Class Number Sum Anomaly Search", flush=True)
    print("=" * 76, flush=True)

    # Use balanced semiprimes at modest sizes (class number computation is costly)
    bit_sizes = [14, 16, 18, 20]
    count_per_size = 15

    print("Generating balanced semiprimes...", flush=True)
    semiprimes = balanced_semiprimes(bit_sizes, count_per_size=count_per_size,
                                     min_ratio=0.3, seed=42)
    print("Generated %d semiprimes across bit sizes %s\n" % (len(semiprimes), bit_sizes),
          flush=True)

    # Number of random t values to sample for comparison
    N_RANDOM_T = 50

    all_results = []

    for idx, (N, p, q) in enumerate(semiprimes):
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)
        sqrtN = int(isqrt(N_int))
        t_star = abs(p_int - q_int)
        t_max = 2 * sqrtN  # trace formula range: t in [1, 2*sqrt(N)]

        print("N=%d (p=%d, q=%d), t*=|p-q|=%d, t_max=%d" % (
            N_int, p_int, q_int, t_star, t_max), flush=True)

        if t_star < 1 or t_star > t_max:
            print("  WARNING: t*=%d outside range [1, %d], skipping" % (t_star, t_max),
                  flush=True)
            continue

        t0 = time.time()

        # Compute H(4N - t*^2) at the special trace value
        D_star = 4 * N_int - t_star * t_star
        if D_star <= 0:
            print("  WARNING: D*=%d <= 0, skipping" % D_star, flush=True)
            continue

        H_star = hurwitz_class_number_fast(D_star)

        # Compute H at nearby t values (local anomaly check)
        local_H = {}
        for dt in [-2, -1, 0, 1, 2]:
            t_local = t_star + dt
            if t_local < 1 or t_local > t_max:
                continue
            D_local = 4 * N_int - t_local * t_local
            if D_local > 0:
                local_H[dt] = hurwitz_class_number_fast(D_local)

        # Sample random t values for comparison distribution
        rng = np.random.RandomState(N_int % (2**31))
        random_ts = set()
        attempts = 0
        while len(random_ts) < N_RANDOM_T and attempts < N_RANDOM_T * 10:
            attempts += 1
            rt = int(rng.randint(1, t_max + 1))
            if rt != t_star and abs(rt - t_star) > 5:
                D_rt = 4 * N_int - rt * rt
                if D_rt > 0:
                    random_ts.add(rt)

        random_H = []
        for rt in sorted(random_ts):
            D_rt = 4 * N_int - rt * rt
            if D_rt > 0:
                H_rt = hurwitz_class_number_fast(D_rt)
                random_H.append(H_rt)

        dt = time.time() - t0

        if not random_H:
            print("  No valid random t values, skipping", flush=True)
            continue

        random_H = np.array(random_H)
        median_H = float(np.median(random_H))
        mean_H = float(np.mean(random_H))
        std_H = float(np.std(random_H))

        # Metrics
        ratio_median = float(H_star / median_H) if median_H > 0 else float('inf')
        ratio_mean = float(H_star / mean_H) if mean_H > 0 else float('inf')
        z_score = float((H_star - mean_H) / std_H) if std_H > 0 else 0.0

        # Rank: fraction of random H values below H_star
        rank_frac = float(np.mean(random_H <= H_star))

        # Local anomaly: H at t* vs neighbors
        local_ratio = 0.0
        if len(local_H) >= 3:
            neighbor_H = [v for k, v in local_H.items() if k != 0]
            if neighbor_H and np.mean(neighbor_H) > 0:
                local_ratio = float(H_star / np.mean(neighbor_H))

        row = {
            'N': N_int,
            'p': p_int,
            'q': q_int,
            'bits': int(N_int).bit_length(),
            'ratio_pq': float(min(p_int, q_int)) / float(max(p_int, q_int)),
            't_star': t_star,
            't_max': t_max,
            'D_star': int(D_star),
            'H_star': float(H_star),
            'median_H_random': float(median_H),
            'mean_H_random': float(mean_H),
            'std_H_random': float(std_H),
            'ratio_to_median': float(ratio_median),
            'ratio_to_mean': float(ratio_mean),
            'z_score': float(z_score),
            'rank_fraction': float(rank_frac),
            'local_ratio': float(local_ratio),
            'local_H': {int(k): float(v) for k, v in local_H.items()},
            'n_random_samples': len(random_H),
            'time_s': float(dt),
        }
        all_results.append(row)

        print("  H(t*)=%.2f, median(H_rand)=%.2f, ratio=%.3f, z=%.2f, rank=%.3f  (%.1fs)" % (
            H_star, median_H, ratio_median, z_score, rank_frac, dt), flush=True)

    # --- Save JSON ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E15_class_number_results.json')
    safe_json_dump(all_results, out_path)

    if not all_results:
        print("\nNo valid results. Exiting.", flush=True)
        return

    # --- Scaling analysis ---
    print("\n" + "=" * 76, flush=True)
    print("SCALING ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    Ns = np.array([r['N'] for r in all_results], dtype=np.float64)
    ratios = np.array([r['ratio_to_median'] for r in all_results])
    z_scores = np.array([r['z_score'] for r in all_results])
    ranks = np.array([r['rank_fraction'] for r in all_results])

    # Is the ratio systematically > 1 or < 1?
    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    mean_z = float(np.mean(z_scores))
    mean_rank = float(np.mean(ranks))

    print("\nOverall statistics:", flush=True)
    print("  Mean H(t*)/median(H_rand): %.4f +/- %.4f" % (mean_ratio, std_ratio), flush=True)
    print("  Mean z-score:              %.4f" % mean_z, flush=True)
    print("  Mean rank fraction:        %.4f (0.5 = no anomaly)" % mean_rank, flush=True)

    # One-sample t-test: is ratio significantly different from 1?
    if len(ratios) > 3:
        t_stat, t_pval = stats.ttest_1samp(ratios, 1.0)
        print("  t-test (ratio vs 1.0):     t=%.3f, p=%.4f" % (t_stat, t_pval), flush=True)
    else:
        t_pval = 1.0

    # Scaling: does the anomaly grow with N?
    alpha, alpha_se, alpha_r2 = 0, 0, 0
    if len(Ns) >= 5:
        mask = ratios > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(np.abs(ratios[mask] - 1.0) + 1e-10))
            alpha, alpha_se, alpha_r2 = sl, se, rv**2
            print("\n  Anomaly scaling (|ratio-1| vs N):", flush=True)
            print("    alpha = %.3f +/- %.3f  (R2=%.4f)" % (alpha, alpha_se, alpha_r2),
                  flush=True)

    # Per-bit-size breakdown
    print("\n  Per bit-size:", flush=True)
    for bits in sorted(set(r['bits'] for r in all_results)):
        rows_b = [r for r in all_results if r['bits'] == bits]
        r_b = np.array([r['ratio_to_median'] for r in rows_b])
        z_b = np.array([r['z_score'] for r in rows_b])
        print("    %d-bit (n=%d): ratio=%.3f +/- %.3f, z=%.3f" % (
            bits, len(rows_b), np.mean(r_b), np.std(r_b), np.mean(z_b)), flush=True)

    # --- Verdict ---
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    if mean_ratio > 1.5 and t_pval < 0.01:
        verdict = "ANOMALY DETECTED — H(t*) systematically elevated, investigate further"
    elif mean_ratio < 0.67 and t_pval < 0.01:
        verdict = "ANOMALY DETECTED — H(t*) systematically depressed, investigate further"
    elif abs(mean_ratio - 1.0) > 0.2 and t_pval < 0.05:
        verdict = "WEAK ANOMALY — marginal deviation from null, extend to larger N"
    elif mean_rank > 0.6 or mean_rank < 0.4:
        verdict = "WEAK RANK ANOMALY — t* not uniformly distributed, may reflect structure"
    else:
        verdict = "NO ANOMALY — H(t*) indistinguishable from generic H(t)"

    print("  %s" % verdict, flush=True)
    print("  Mean ratio to median: %.4f" % mean_ratio, flush=True)
    print("  t-test p-value:       %.4f" % t_pval, flush=True)
    print("  Mean rank:            %.4f" % mean_rank, flush=True)

    summary = {
        'mean_ratio': float(mean_ratio),
        'std_ratio': float(std_ratio),
        'mean_z_score': float(mean_z),
        'mean_rank': float(mean_rank),
        't_test_pvalue': float(t_pval),
        'anomaly_scaling_alpha': float(alpha),
        'anomaly_scaling_se': float(alpha_se),
        'n_semiprimes': len(all_results),
        'verdict': verdict,
    }
    summary_path = os.path.join(data_dir, 'E15_class_number_summary.json')
    safe_json_dump(summary, summary_path)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('E15: Class Number Sum Anomaly Search', fontsize=14)

        # Plot 1: H(t*)/median(H_rand) vs N
        ax = axes[0, 0]
        ax.scatter(Ns, ratios, s=20, alpha=0.6, c='blue')
        ax.axhline(y=1.0, color='gray', ls='--', lw=1, label='no anomaly')
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('H(t*) / median(H_random)')
        ax.set_title('Class Number Ratio at t* = |p-q|')
        ax.legend()

        # Plot 2: z-score vs N
        ax = axes[0, 1]
        ax.scatter(Ns, z_scores, s=20, alpha=0.6, c='red')
        ax.axhline(y=0, color='gray', ls='--', lw=1)
        ax.axhline(y=2, color='orange', ls=':', lw=1, label='z=2')
        ax.axhline(y=-2, color='orange', ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('z-score')
        ax.set_title('z-score of H(t*) vs random distribution')
        ax.legend()

        # Plot 3: Rank fraction histogram
        ax = axes[1, 0]
        ax.hist(ranks, bins=int(15), color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='gray', ls='--', lw=1, label='uniform')
        ax.axvline(x=mean_rank, color='red', ls='-', lw=2, label='mean=%.3f' % mean_rank)
        ax.set_xlabel('Rank Fraction of H(t*)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Rank Fractions')
        ax.legend()

        # Plot 4: Local anomaly — H at t* vs neighbors
        ax = axes[1, 1]
        for r in all_results[:20]:  # plot first 20 for readability
            local = r['local_H']
            offsets = sorted(local.keys())
            vals = [local[k] for k in offsets]
            ax.plot(offsets, vals, 'o-', alpha=0.3, markersize=3)
        ax.set_xlabel('Offset from t*')
        ax.set_ylabel('H(4N - (t*+offset)^2)')
        ax.set_title('Local Class Number Profile (first 20 semiprimes)')

        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E15_class_number.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("\nPlot saved to %s" % plot_path, flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
