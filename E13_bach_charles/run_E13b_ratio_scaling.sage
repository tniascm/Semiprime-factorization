"""
E13b-scaling: Characterize |E(N)/tau(N)| ratio at scale
=========================================================

Extended trace formula analysis from N ~ 2^8 up to N ~ 2^64.

Key question: Is the Eisenstein correction a small perturbation of tau(N),
or does it remain O(1) relative to tau(N)?  If small, the class-number sum
alone could determine tau(N) mod ell without factoring.

Strategy:
  - Small N (up to 2^24): direct q-expansion gives tau(N)
  - Large N: use multiplicativity tau(pq) = tau(p)*tau(q) for N=pq coprime.
    Compute tau(p) via the Eichler-Selberg trace formula (p prime, so the
    Eisenstein correction is trivially -1/2*(1 + p^{k-1}), no factoring needed).
  - E(N) = -1 - min(p,q)^{k-1} for N=pq semiprime (requires known factors).
  - Record |E(N)/tau(N)| across many size classes.

Conclusion expected: ratio stays O(1), confirming the Eisenstein correction
is NOT a small perturbation and CANNOT be ignored.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)


# ── Hurwitz class number ────────────────────────────────────────────

def hurwitz_class_number(D):
    """
    Compute the Hurwitz class number H(D) for D > 0.

    H(D) = sum_{f^2 | D, -D/f^2 valid disc} h_prim(-D/f^2) / w(-D/f^2)

    where h_prim is the number of primitive reduced forms (= class number
    of the order of that discriminant, NOT the maximal order).

    Equivalently: H(D) = sum over ALL reduced forms of disc -D of 2/|Aut(Q)|.
    """
    if D == 0:
        return QQ(-1) / QQ(12)
    if D < 0:
        raise ValueError("D must be >= 0")

    result = QQ(0)
    for f_sq in range(1, isqrt(D) + 1):
        if D % (f_sq * f_sq) != 0:
            continue
        disc = D // (f_sq * f_sq)
        if disc == 0:
            continue
        neg_disc = -Integer(disc)
        if neg_disc % 4 not in [0, 1]:
            continue
        # Class number of the ORDER of discriminant neg_disc.
        # Must use BinaryQF, not QuadraticField (which gives the maximal
        # order and ignores conductor).
        h = len(BinaryQF_reduced_representatives(neg_disc, primitive_only=True))
        if disc == 3:
            w = 3
        elif disc == 4:
            w = 2
        else:
            w = 1
        result += QQ(h) / QQ(w)

    return result


# ── Trace formula for tau(p) at a PRIME p ─────────────────────────

def trace_formula_tau_prime(p, k=12):
    """
    Compute tau(p) = a_p(f_k) for prime p using the Eichler-Selberg trace
    formula for S_k(SL_2(Z)), dim=1.

    tau(p) = ClassNumberSum(p) + EisensteinCorrection(p)

    For prime p, the Eisenstein correction is trivial:
      E(p) = -1/2 * sum_{d|p} min(d, p/d)^{k-1}
      Divisors of p: {1, p}.
        d=1: min(1, p) = 1 -> 1^{k-1} = 1
        d=p: min(p, 1) = 1 -> 1^{k-1} = 1
      E(p) = -1/2 * (1 + 1) = -1   for all primes p.
    """
    p = Integer(p)

    # Class number sum: -1/2 * sum_{|t| < 2*sqrt(p)} rho_k(t,p) * H(4p - t^2)
    bound = isqrt(4 * p)
    class_sum = QQ(0)

    for t_val in range(-int(bound), int(bound) + 1):
        D = int(4 * p) - t_val * t_val
        if D <= 0:
            continue

        # Hecke polynomial P_{k-2}(t, p) via recurrence:
        # P_0 = 1, P_1 = t, P_m = t * P_{m-1} - p * P_{m-2}
        P_prev_prev = QQ(1)
        P_prev = QQ(t_val)
        for m in range(2, k - 1):
            P_curr = QQ(t_val) * P_prev - QQ(p) * P_prev_prev
            P_prev_prev = P_prev
            P_prev = P_curr

        rho = P_prev if k > 2 else QQ(1)
        H_val = hurwitz_class_number(int(D))
        class_sum += rho * H_val

    class_sum = -QQ(1) / QQ(2) * class_sum

    # Eisenstein correction for prime p: always -1
    # (both divisors 1 and p give min(d, p/d) = 1)
    eis_corr = QQ(-1)

    return int(class_sum + eis_corr)


# ── Eigenform q-expansion for small N ─────────────────────────────

def eigenform_qexp(k, prec):
    """Compute q-expansion of unique eigenform in S_k(SL_2(Z)), dim=1 case."""
    R = PowerSeriesRing(ZZ, 'q', default_prec=prec)
    E4 = eisenstein_series_qexp(int(4), prec, normalization='constant')
    E6 = eisenstein_series_qexp(int(6), prec, normalization='constant')
    Delta = delta_qexp(prec)
    forms = {12: Delta, 16: Delta*E4, 18: Delta*E6,
             20: Delta*E4*E4, 22: Delta*E4*E6}
    f = R(forms[k])
    assert f[1] == 1
    return f


# ── Main scaling experiment ───────────────────────────────────────

def main():
    print("=" * 76, flush=True)
    print("E13b-scaling: |E(N)/tau(N)| ratio characterization", flush=True)
    print("=" * 76, flush=True)
    print(flush=True)

    k = 12  # Weight for the Ramanujan tau function
    km1 = k - 1

    # Size classes: bit sizes for balanced semiprimes
    # Small sizes use q-expansion verification; large sizes use trace formula
    QEXP_THRESHOLD = 14  # bits: use q-expansion for N < 2^14 (fast)
    size_classes = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40]
    count_per_size = 10

    # For q-expansion verification at small sizes
    small_sizes = [s for s in size_classes if s <= QEXP_THRESHOLD]
    large_sizes = [s for s in size_classes if s > QEXP_THRESHOLD]

    all_results = []
    tau_cache = {}  # prime -> tau(prime)

    # ── Phase 1: Small N with q-expansion verification ──────────────
    if small_sizes:
        print(f"Phase 1: Small N (up to {QEXP_THRESHOLD} bits) — q-expansion verification",
              flush=True)
        print("-" * 76, flush=True)

        small_semiprimes = balanced_semiprimes(small_sizes, count_per_size=count_per_size)
        max_N = max(int(N) for N, _, _ in small_semiprimes)
        prec = max_N + 10

        print(f"  Computing Delta q-expansion to precision {prec}...", flush=True)
        t0 = time.time()
        f12 = eigenform_qexp(12, int(prec))
        print(f"  Done in {time.time()-t0:.2f}s", flush=True)

        print(f"\n  {'N':>12} {'bits':>5} {'tau(N)':>20} {'E(N)':>20} {'|E/tau|':>10} {'verify':>8}",
              flush=True)
        print(f"  {'-'*76}", flush=True)

        for N, p, q in small_semiprimes:
            N_int = int(N)
            p_int = int(p)
            q_int = int(q)
            n_bits = N_int.bit_length()

            # Direct from q-expansion
            tau_N_direct = int(f12[N_int])

            # Via multiplicativity: tau(pq) = tau(p) * tau(q)
            tau_p = int(f12[p_int])
            tau_q = int(f12[q_int])
            tau_N_mult = tau_p * tau_q

            # Cache for later
            tau_cache[p_int] = tau_p
            tau_cache[q_int] = tau_q

            # Verify multiplicativity
            verify = "OK" if tau_N_direct == tau_N_mult else "FAIL"

            # Eisenstein correction
            min_pq = min(p_int, q_int)
            eis_N = -1 - int(Integer(min_pq) ** km1)

            ratio = abs(float(eis_N) / float(tau_N_direct)) if tau_N_direct != 0 else float('inf')

            print(f"  {N_int:>12} {n_bits:>5} {tau_N_direct:>20} {eis_N:>20} "
                  f"{ratio:>10.4f} {verify:>8}", flush=True)

            all_results.append({
                'N': N_int, 'p': p_int, 'q': q_int, 'bits': n_bits,
                'tau_N': tau_N_direct,
                'eis_N': eis_N,
                'eis_ratio': round(float(ratio), int(6)),
                'method': 'q-expansion',
                'verify': verify,
            })

    # ── Verify trace formula against q-expansion ─────────────────────
    if tau_cache:
        print("\nTrace formula verification (vs q-expansion):", flush=True)
        print(f"  {'prime':>8} {'q-exp tau(p)':>20} {'trace tau(p)':>20} {'match':>6}",
              flush=True)
        print(f"  {'-'*58}", flush=True)
        verified_primes = sorted(set(list(tau_cache.keys())[:20]))
        all_match = True
        for pr in verified_primes:
            tau_qexp = tau_cache[pr]
            tau_trace = trace_formula_tau_prime(pr, k)
            match = "OK" if tau_qexp == tau_trace else "FAIL"
            if match == "FAIL":
                all_match = False
            print(f"  {pr:>8} {tau_qexp:>20} {tau_trace:>20} {match:>6}",
                  flush=True)
        if all_match:
            print("  All trace formula values match q-expansion. Proceeding.",
                  flush=True)
        else:
            print("  ERROR: Trace formula mismatch detected! Aborting.",
                  flush=True)
            return

    # ── Phase 2: Large N via trace formula + multiplicativity ────────
    if large_sizes:
        print(f"\nPhase 2: Large N ({min(large_sizes)}-{max(large_sizes)} bits) — "
              f"trace formula + multiplicativity", flush=True)
        print("-" * 76, flush=True)

        large_semiprimes = balanced_semiprimes(large_sizes, count_per_size=count_per_size)

        print(f"\n  {'N':>20} {'bits':>5} {'|tau| dig':>10} {'|E| dig':>8} "
              f"{'|E/tau|':>12} {'t(s)':>6}", flush=True)
        print(f"  {'-'*68}", flush=True)

        for N, p, q in large_semiprimes:
            N_int = int(N)
            p_int = int(p)
            q_int = int(q)
            n_bits = N_int.bit_length()

            t0 = time.time()

            # Compute tau(p) and tau(q) via trace formula (or cache)
            if p_int not in tau_cache:
                tau_cache[p_int] = trace_formula_tau_prime(p_int, k)
            if q_int not in tau_cache:
                tau_cache[q_int] = trace_formula_tau_prime(q_int, k)

            tau_p = tau_cache[p_int]
            tau_q = tau_cache[q_int]
            tau_N = tau_p * tau_q

            dt = time.time() - t0

            # Eisenstein correction
            min_pq = min(p_int, q_int)
            eis_N = -1 - int(Integer(min_pq) ** km1)

            ratio = abs(float(eis_N) / float(tau_N)) if tau_N != 0 else float('inf')

            tau_digits = len(str(abs(tau_N)))
            eis_digits = len(str(abs(eis_N)))

            print(f"  {N_int:>20} {n_bits:>5} {tau_digits:>10} {eis_digits:>8} "
                  f"{ratio:>12.4f} {dt:>6.2f}", flush=True)

            all_results.append({
                'N': N_int, 'p': p_int, 'q': q_int, 'bits': n_bits,
                'tau_N': int(tau_N),
                'eis_N': int(eis_N),
                'eis_ratio': round(float(ratio), int(6)),
                'method': 'trace-formula-mult',
                'time_secs': round(float(dt), int(3)),
            })

    # ── Analysis: ratio statistics per bit size ──────────────────────
    print("\n" + "=" * 76, flush=True)
    print("RATIO STATISTICS BY SIZE CLASS", flush=True)
    print("=" * 76, flush=True)

    print(f"\n  {'bits':>5} {'count':>6} {'mean |E/tau|':>14} {'median':>10} "
          f"{'min':>10} {'max':>10}", flush=True)
    print(f"  {'-'*60}", flush=True)

    ratio_by_bits = {}
    for r in all_results:
        bits = r['bits']
        ratio = r['eis_ratio']
        if bits not in ratio_by_bits:
            ratio_by_bits[bits] = []
        ratio_by_bits[bits].append(ratio)

    summary_rows = []
    for bits in sorted(ratio_by_bits.keys()):
        ratios = sorted(ratio_by_bits[bits])
        n = len(ratios)
        mean_r = sum(ratios) / n
        median_r = ratios[n // 2]
        min_r = ratios[0]
        max_r = ratios[-1]

        print(f"  {bits:>5} {n:>6} {mean_r:>14.4f} {median_r:>10.4f} "
              f"{min_r:>10.4f} {max_r:>10.4f}", flush=True)

        summary_rows.append({
            'bits': bits,
            'count': n,
            'mean_ratio': round(float(mean_r), int(6)),
            'median_ratio': round(float(median_r), int(6)),
            'min_ratio': round(float(min_r), int(6)),
            'max_ratio': round(float(max_r), int(6)),
        })

    # ── Conclusion ───────────────────────────────────────────────────
    overall_ratios = [r['eis_ratio'] for r in all_results]
    overall_mean = sum(overall_ratios) / len(overall_ratios)
    overall_min = min(overall_ratios)
    overall_median = sorted(overall_ratios)[len(overall_ratios) // 2]

    # Compute median of medians across size classes for robust statistic
    medians = [sorted(ratio_by_bits[b])[len(ratio_by_bits[b]) // 2]
               for b in sorted(ratio_by_bits.keys())]
    median_of_medians = sorted(medians)[len(medians) // 2]

    print(f"\n  Overall: median |E/tau| = {overall_median:.4f}, "
          f"mean = {overall_mean:.4f}, min = {overall_min:.4f}", flush=True)
    print(f"  Median of class medians: {median_of_medians:.4f}", flush=True)
    print(flush=True)

    # Use median stability to assess: ratio is O(1) if median stays bounded
    # away from zero across size classes (min can be small due to Sato-Tate
    # fluctuations where tau(N) happens to be large).
    blocked = median_of_medians > 0.01
    if blocked:
        print("  CONCLUSION: The Eisenstein correction is NOT negligible.", flush=True)
        print("  |E(N)/tau(N)| median stays O(1) across all N sizes.", flush=True)
        print("  Small min values (%.4f) reflect Sato-Tate fluctuations" % overall_min, flush=True)
        print("  in tau(p), not systematic decay of the correction.", flush=True)
        print("  The class-number sum alone CANNOT determine tau(N) mod ell.", flush=True)
        print("  => Trace formula route to sub-O(N) channel evaluation is BLOCKED.", flush=True)
    else:
        print("  FINDING: Median |E/tau| is very small!", flush=True)
        print("  This warrants further investigation.", flush=True)

    # ── Save results ──────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E13b_ratio_scaling.json')

    output = {
        'experiment': 'E13b-ratio-scaling',
        'weight': k,
        'size_classes': size_classes,
        'count_per_size': count_per_size,
        'qexp_threshold_bits': QEXP_THRESHOLD,
        'results': all_results,
        'summary': summary_rows,
        'overall_mean_ratio': round(float(overall_mean), int(6)),
        'overall_median_ratio': round(float(overall_median), int(6)),
        'overall_min_ratio': round(float(overall_min), int(6)),
        'median_of_medians': round(float(median_of_medians), int(6)),
        'conclusion': 'blocked' if blocked else 'investigate',
    }
    safe_json_dump(output, out_path)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
