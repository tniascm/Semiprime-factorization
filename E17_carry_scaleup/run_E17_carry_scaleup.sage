"""
E17: Scale-Up Carry Depth to 24-28 Bits
=========================================

E12 tested deep carry compositions for N up to ~12,000 (13-14 bits).
Power-law fits over <2 decades are unreliable (brainstorm point #4).

This experiment extends to 24-28 bit semiprimes (N up to 2^28 ~ 268M),
providing 4+ decades of N for definitive scaling exponents.

Key design choice: full DFT is O(N log N) which is impractical for
N ~ 10^8. Instead, we compute INDIVIDUAL DFT coefficients at
factor-related frequencies:

    f_hat(xi) = (1/N) * sum_{t=0}^{N-1} f(t) * e^{-2*pi*i*xi*t/N}

This costs O(N) per coefficient. We compute ~12 coefficients per
semiprime (factor freqs + random baselines), so total cost per
semiprime is O(N * 12) = O(N).

Signals tested (from E12 results, most interesting):
    1. carry_xor: XOR of all quotient parities
    2. carry_product_parity: product of (-1)^{q_i mod 2}

Metrics:
    - |f_hat(p)|, |f_hat(q)| (factor frequencies)
    - |f_hat(xi_rand)| for 5 random non-factor xi (baseline)
    - Factor-to-random ratio
    - Scaling of ratio vs N over 4+ decades
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
# Core computation: carry trace (from E12)
# ============================================================

def carry_trace(t, N, depth):
    """
    Compute the carry trace of iterated squaring: t -> t^2 mod N -> ...

    Returns quotients: list of floor(x_i^2 / N) for i=0..depth-1
    """
    N_int = int(N)
    x = int(t) % N_int
    quotients = []
    for _ in range(depth):
        x2 = x * x
        q = x2 // N_int
        r = x2 % N_int
        quotients.append(q)
        x = r
    return quotients


# ============================================================
# Signal computation at a single t (no arrays)
# ============================================================

def carry_signals_at_t(t, N, depth):
    """Compute carry_xor and carry_product_parity for a single t.

    Returns (xor_val, prod_parity_val) as floats in {-1, +1}.
    """
    quots = carry_trace(t, N, depth)
    iquots = [int(qq) for qq in quots]

    # carry_xor: XOR of quotient parities, mapped to +/-1
    xor_val = 0
    for qq in iquots:
        xor_val = xor_val ^ (qq & 1)
    xor_sig = 1.0 if xor_val == 0 else -1.0

    # carry_product_parity: product of (-1)^(q_i mod 2)
    prod = 1
    for qq in iquots:
        if qq & 1 == 1:
            prod = -prod
    prod_sig = float(prod)

    return xor_sig, prod_sig


# ============================================================
# Targeted DFT: compute f_hat(xi) for specific frequencies
# ============================================================

def targeted_dft_coefficient(signal_func, N, depth, xi):
    """
    Compute a single DFT coefficient:
        f_hat(xi) = (1/N) * sum_{t=0}^{N-1} f(t) * e^{-2*pi*i*xi*t/N}

    signal_func(t, N, depth) -> (xor_val, prod_val)

    Returns (|f_hat_xor(xi)|, |f_hat_prod(xi)|).
    """
    N_int = int(N)
    xi_int = int(xi)

    sum_xor_real = 0.0
    sum_xor_imag = 0.0
    sum_prod_real = 0.0
    sum_prod_imag = 0.0

    two_pi_xi_over_N = 2.0 * np.pi * xi_int / N_int

    for t in range(N_int):
        xor_val, prod_val = signal_func(t, N_int, depth)
        phase = two_pi_xi_over_N * t
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        sum_xor_real += xor_val * cos_phase
        sum_xor_imag -= xor_val * sin_phase
        sum_prod_real += prod_val * cos_phase
        sum_prod_imag -= prod_val * sin_phase

    mag_xor = np.sqrt(sum_xor_real**2 + sum_xor_imag**2) / N_int
    mag_prod = np.sqrt(sum_prod_real**2 + sum_prod_imag**2) / N_int

    return mag_xor, mag_prod


def compute_targeted_dft(N, p, q, depth, n_random=5, rng_seed=None):
    """
    Compute DFT magnitudes at factor and random frequencies.

    Returns dict with magnitudes for carry_xor and carry_product_parity.
    """
    N_int = int(N)
    p_int = int(p)
    q_int = int(q)

    # Factor frequencies
    factor_freqs = [p_int, q_int, 2 * p_int, 2 * q_int]
    factor_freqs = [f % N_int for f in factor_freqs if f > 0 and f < N_int]

    # Random non-factor frequencies
    rng = np.random.RandomState(rng_seed if rng_seed is not None else N_int % (2**31))
    random_freqs = []
    attempts = 0
    while len(random_freqs) < n_random and attempts < n_random * 20:
        attempts += 1
        xi = int(rng.randint(1, N_int))
        if xi % p_int != 0 and xi % q_int != 0 and xi not in random_freqs:
            random_freqs.append(xi)

    # Compute all targeted coefficients
    results = {
        'factor_freqs': {},
        'random_freqs': {},
    }

    for xi in factor_freqs:
        mag_xor, mag_prod = targeted_dft_coefficient(
            carry_signals_at_t, N_int, depth, xi)
        results['factor_freqs'][int(xi)] = {
            'xi': int(xi),
            'is_p_multiple': (xi % p_int == 0),
            'is_q_multiple': (xi % q_int == 0),
            'mag_xor': float(mag_xor),
            'mag_prod': float(mag_prod),
        }

    for xi in random_freqs:
        mag_xor, mag_prod = targeted_dft_coefficient(
            carry_signals_at_t, N_int, depth, xi)
        results['random_freqs'][int(xi)] = {
            'xi': int(xi),
            'mag_xor': float(mag_xor),
            'mag_prod': float(mag_prod),
        }

    return results


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 76, flush=True)
    print("E17: Scale-Up Carry Depth to 24-28 Bits", flush=True)
    print("=" * 76, flush=True)

    # Bit sizes spanning 4+ decades of N
    # Fewer samples at larger sizes due to O(N) cost per DFT coefficient
    bit_configs = [
        (14, 8),   # N ~ 2^14 = 16K       — fast
        (16, 8),   # N ~ 2^16 = 65K       — fast
        (18, 6),   # N ~ 2^18 = 262K      — moderate
        (20, 5),   # N ~ 2^20 = 1M        — moderate
        (22, 4),   # N ~ 2^22 = 4M        — slow
        (24, 3),   # N ~ 2^24 = 16M       — slow
        (26, 2),   # N ~ 2^26 = 67M       — very slow
    ]

    all_results = []
    total_semiprimes = sum(c for _, c in bit_configs)
    print("Total semiprimes to test: %d" % total_semiprimes, flush=True)
    print("Estimated time: significant for 24-26 bit sizes\n", flush=True)

    processed = 0
    for bits, count in bit_configs:
        print("--- %d-bit semiprimes (count=%d) ---" % (bits, count), flush=True)

        sps = balanced_semiprimes([bits], count_per_size=count, min_ratio=0.3, seed=42)
        if not sps:
            print("  No semiprimes generated, skipping", flush=True)
            continue

        for N, p, q in sps:
            N_int = int(N)
            depth = N_int.bit_length()
            processed += 1

            print("  [%d/%d] N=%d (%d bits), p=%d, q=%d, depth=%d" % (
                processed, total_semiprimes, N_int, bits, p, q, depth),
                end='', flush=True)

            t0 = time.time()
            dft_results = compute_targeted_dft(N_int, p, q, depth, n_random=5)
            dt = time.time() - t0

            # Extract magnitudes
            factor_mags_xor = [v['mag_xor'] for v in dft_results['factor_freqs'].values()]
            factor_mags_prod = [v['mag_prod'] for v in dft_results['factor_freqs'].values()]
            random_mags_xor = [v['mag_xor'] for v in dft_results['random_freqs'].values()]
            random_mags_prod = [v['mag_prod'] for v in dft_results['random_freqs'].values()]

            # Compute ratios
            median_rand_xor = float(np.median(random_mags_xor)) if random_mags_xor else 1e-15
            median_rand_prod = float(np.median(random_mags_prod)) if random_mags_prod else 1e-15
            max_factor_xor = float(max(factor_mags_xor)) if factor_mags_xor else 0.0
            max_factor_prod = float(max(factor_mags_prod)) if factor_mags_prod else 0.0

            ratio_xor = max_factor_xor / max(median_rand_xor, 1e-15)
            ratio_prod = max_factor_prod / max(median_rand_prod, 1e-15)

            row = {
                'N': N_int,
                'p': int(p),
                'q': int(q),
                'bits': bits,
                'depth': depth,
                'ratio_pq': float(min(p, q)) / float(max(p, q)),
                'max_factor_mag_xor': float(max_factor_xor),
                'max_factor_mag_prod': float(max_factor_prod),
                'median_random_mag_xor': float(median_rand_xor),
                'median_random_mag_prod': float(median_rand_prod),
                'mean_random_mag_xor': float(np.mean(random_mags_xor)) if random_mags_xor else 0.0,
                'mean_random_mag_prod': float(np.mean(random_mags_prod)) if random_mags_prod else 0.0,
                'ratio_xor': float(ratio_xor),
                'ratio_prod': float(ratio_prod),
                'all_factor_mags_xor': [float(v) for v in factor_mags_xor],
                'all_factor_mags_prod': [float(v) for v in factor_mags_prod],
                'all_random_mags_xor': [float(v) for v in random_mags_xor],
                'all_random_mags_prod': [float(v) for v in random_mags_prod],
                'time_s': float(dt),
            }
            all_results.append(row)

            print(" — xor_ratio=%.3f, prod_ratio=%.3f  (%.1fs)" % (
                ratio_xor, ratio_prod, dt), flush=True)

    # --- Save JSON ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E17_carry_scaleup_results.json')
    safe_json_dump(all_results, out_path)

    if not all_results:
        print("\nNo results. Exiting.", flush=True)
        return

    # --- Scaling analysis ---
    print("\n" + "=" * 76, flush=True)
    print("SCALING ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    Ns = np.array([r['N'] for r in all_results], dtype=np.float64)
    ratios_xor = np.array([r['ratio_xor'] for r in all_results])
    ratios_prod = np.array([r['ratio_prod'] for r in all_results])
    factor_xor = np.array([r['max_factor_mag_xor'] for r in all_results])
    factor_prod = np.array([r['max_factor_mag_prod'] for r in all_results])
    random_xor = np.array([r['median_random_mag_xor'] for r in all_results])
    random_prod = np.array([r['median_random_mag_prod'] for r in all_results])

    scaling = {}

    for name, mags, ratios_arr in [
        ('carry_xor_factor', factor_xor, ratios_xor),
        ('carry_xor_random', random_xor, None),
        ('carry_prod_factor', factor_prod, ratios_prod),
        ('carry_prod_random', random_prod, None),
    ]:
        mask = mags > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(mags[mask]))
            scaling[name] = {
                'alpha': float(sl),
                'alpha_se': float(se),
                'r2': float(rv**2),
                'p_value': float(pv),
            }
            print("\n  %s: mag ~ N^%.4f +/- %.4f  (R2=%.4f, p=%.2e)" % (
                name, sl, se, rv**2, pv), flush=True)

    # Ratio scaling (does the factor/random ratio stay constant, grow, or decay?)
    for name, ratios_arr in [('xor_ratio', ratios_xor), ('prod_ratio', ratios_prod)]:
        mask = ratios_arr > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(ratios_arr[mask]))
            scaling[name] = {
                'alpha': float(sl),
                'alpha_se': float(se),
                'r2': float(rv**2),
                'p_value': float(pv),
            }
            print("\n  %s: ratio ~ N^%.4f +/- %.4f  (R2=%.4f, p=%.2e)" % (
                name, sl, se, rv**2, pv), flush=True)

    # Per-bit-size summary
    print("\n  Per bit-size:", flush=True)
    print("  %5s %5s %10s %10s %10s %10s" % (
        'bits', 'n', 'xor_ratio', 'prod_ratio', 'fac_xor', 'rand_xor'), flush=True)
    print("  " + "-" * 55, flush=True)
    for bits in sorted(set(r['bits'] for r in all_results)):
        rows_b = [r for r in all_results if r['bits'] == bits]
        rx = np.mean([r['ratio_xor'] for r in rows_b])
        rp = np.mean([r['ratio_prod'] for r in rows_b])
        fx = np.mean([r['max_factor_mag_xor'] for r in rows_b])
        rdx = np.mean([r['median_random_mag_xor'] for r in rows_b])
        print("  %5d %5d %10.4f %10.4f %10.6f %10.6f" % (
            bits, len(rows_b), rx, rp, fx, rdx), flush=True)

    # --- Verdict ---
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    mean_xor_ratio = float(np.mean(ratios_xor))
    mean_prod_ratio = float(np.mean(ratios_prod))

    xor_ratio_trend = scaling.get('xor_ratio', {}).get('alpha', 0)
    prod_ratio_trend = scaling.get('prod_ratio', {}).get('alpha', 0)

    print("  Mean factor/random ratio (xor):  %.4f" % mean_xor_ratio, flush=True)
    print("  Mean factor/random ratio (prod): %.4f" % mean_prod_ratio, flush=True)
    print("  Ratio trend (xor):  N^%.4f" % xor_ratio_trend, flush=True)
    print("  Ratio trend (prod): N^%.4f" % prod_ratio_trend, flush=True)

    if mean_xor_ratio > 1.5 or mean_prod_ratio > 1.5:
        if xor_ratio_trend > -0.05 or prod_ratio_trend > -0.05:
            verdict = "SIGNAL — factor frequencies persistently elevated at scale"
        else:
            verdict = "WEAK SIGNAL — elevated ratio but decaying with N"
    elif xor_ratio_trend > 0.05 or prod_ratio_trend > 0.05:
        verdict = "INTERESTING — ratio GROWING with N, needs further investigation"
    else:
        verdict = "BARRIER CONFIRMED — factor frequencies indistinguishable from random at scale"

    print("\n  VERDICT: %s" % verdict, flush=True)

    summary = {
        'scaling': scaling,
        'mean_xor_ratio': float(mean_xor_ratio),
        'mean_prod_ratio': float(mean_prod_ratio),
        'n_semiprimes': len(all_results),
        'bit_range': [min(r['bits'] for r in all_results), max(r['bits'] for r in all_results)],
        'verdict': verdict,
    }
    summary_path = os.path.join(data_dir, 'E17_carry_scaleup_summary.json')
    safe_json_dump(summary, summary_path)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('E17: Carry Depth Scale-Up (14-26 bits)', fontsize=14)

        # Plot 1: Factor vs random DFT magnitudes (carry_xor)
        ax = axes[0, 0]
        ax.scatter(Ns, factor_xor, s=20, alpha=0.6, color='blue', label='Factor freq')
        ax.scatter(Ns, random_xor, s=15, alpha=0.4, color='gray', label='Random freq')
        for name, color in [('carry_xor_factor', 'blue'), ('carry_xor_random', 'gray')]:
            if name in scaling:
                s = scaling[name]
                xf = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                yf = 10**(s['alpha'] * np.log10(xf) + np.log10(np.median(
                    factor_xor if 'factor' in name else random_xor)))
                ax.plot(xf, yf, '--', color=color, lw=1,
                        label='N^{%.3f}' % s['alpha'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('|f_hat(xi)|')
        ax.set_title('carry_xor: DFT Magnitudes')
        ax.legend(fontsize=7)

        # Plot 2: Factor vs random DFT magnitudes (carry_prod_parity)
        ax = axes[0, 1]
        ax.scatter(Ns, factor_prod, s=20, alpha=0.6, color='red', label='Factor freq')
        ax.scatter(Ns, random_prod, s=15, alpha=0.4, color='gray', label='Random freq')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('|f_hat(xi)|')
        ax.set_title('carry_prod_parity: DFT Magnitudes')
        ax.legend(fontsize=7)

        # Plot 3: Factor/random ratio vs N
        ax = axes[1, 0]
        ax.scatter(Ns, ratios_xor, s=20, alpha=0.6, color='blue', label='carry_xor')
        ax.scatter(Ns, ratios_prod, s=20, alpha=0.6, color='red', label='carry_prod')
        ax.axhline(y=1.0, color='gray', ls='--', lw=1, label='no signal')
        if 'xor_ratio' in scaling:
            s = scaling['xor_ratio']
            xf = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
            # Fit line in log-log
            if np.sum(ratios_xor > 0) >= 3:
                sl, intc, _, _, _ = stats.linregress(
                    np.log10(Ns[ratios_xor > 0]), np.log10(ratios_xor[ratios_xor > 0]))
                ax.plot(xf, 10**intc * xf**sl, 'b--', lw=1,
                        label='xor: N^{%.3f}' % sl)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('Factor / Random ratio')
        ax.set_title('Factor Frequency Elevation')
        ax.legend(fontsize=7)

        # Plot 4: Per-bit-size box plot of ratios
        ax = axes[1, 1]
        bits_list = sorted(set(r['bits'] for r in all_results))
        box_data = []
        box_labels = []
        for bits in bits_list:
            rows_b = [r['ratio_xor'] for r in all_results if r['bits'] == bits]
            box_data.append(rows_b)
            box_labels.append('%d' % bits)
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.axhline(y=1.0, color='gray', ls='--', lw=1)
            ax.set_xlabel('Bit size')
            ax.set_ylabel('carry_xor factor/random ratio')
            ax.set_title('Ratio Distribution by Bit Size')

        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E17_carry_scaleup.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("\nPlot saved to %s" % plot_path, flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
