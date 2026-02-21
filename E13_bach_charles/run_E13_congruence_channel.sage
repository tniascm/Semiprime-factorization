"""
E13: Bach-Charles congruence factoring channel
===============================================

Motivation
----------
Bach-Charles (2007) proves: computing a_N(f) for ANY fixed eigenform f
at composite N in poly(log N) implies polynomial-time factoring.

The proof uses multiplicativity: a_{pq} = a_p * a_q.

LOOPHOLE: Bach-Charles blocks a SINGLE eigenform oracle. But what about
STRUCTURAL RELATIONS between eigenforms? Specifically:

The Eisenstein congruence: for weight k with dim S_k(Gamma_0(1)) = 1,
and prime ell dividing the numerator of B_k, the unique eigenform f_k
satisfies:
    a_p(f_k) ≡ 1 + p^{k-1}  (mod ell)  for all primes p.

For N = pq:
    a_N(f_k) = a_p(f_k) * a_q(f_k) ≡ (1 + p^{k-1})(1 + q^{k-1})  (mod ell)

Expanding:
    a_N(f_k) ≡ 1 + s_{k-1} + N^{k-1}  (mod ell)
where s_{k-1} = p^{k-1} + q^{k-1}.

Newton's identity: s_m = (p+q)*s_{m-1} - pq*s_{m-2} with s_0=2, s_1=p+q.
So s_{k-1} is a degree-(k-1) polynomial in e1 = p+q (with pq = N known).

Given a_N(f_k) mod ell, we can solve for e1 = p+q mod ell.
At most k-1 roots => ~log2(ell/(k-1)) bits of factor information.

Channels tested
---------------
| Weight k | Congruence prime ell | Bernoulli source | Bits (approx) |
|----------|---------------------|------------------|---------------|
| 12       | 691                 | B_12             | ~6            |
| 16       | 3617                | B_16             | ~8            |
| 18       | 43867               | B_18             | ~11           |
| 20       | 283*617=174611      | B_20             | ~13           |
| 22       | 131*6523=854513     | B_22             | ~15           |

Key questions:
1. Does the congruence channel WORK? (Verify on small semiprimes)
2. How many bits per channel in practice?
3. Are the bits independent across channels?
4. Do accumulated bits approach the Coppersmith n/4 threshold?
5. Can this be made poly(log N)? (Spoiler: probably not)
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)
np.random.seed(42)


# ── Eigenform q-expansion computation ─────────────────────────────────

def eigenform_qexp(k, prec):
    """
    Compute the q-expansion of the unique normalized eigenform in
    S_k(Gamma_0(1)) for k in {12, 16, 18, 20, 22} (dim = 1 cases).

    Returns a power series in ZZ[[q]] up to O(q^prec).
    """
    R = PowerSeriesRing(ZZ, 'q', default_prec=prec)
    q = R.gen()

    # Eisenstein series (weight 4, 6) and Delta (weight 12)
    E4 = eisenstein_series_qexp(int(4), prec, normalization='constant')
    E6 = eisenstein_series_qexp(int(6), prec, normalization='constant')
    Delta = delta_qexp(prec)

    if k == 12:
        f = Delta
    elif k == 16:
        f = Delta * E4
    elif k == 18:
        f = Delta * E6
    elif k == 20:
        f = Delta * E4 * E4
    elif k == 22:
        f = Delta * E4 * E6
    else:
        raise ValueError(f"Weight {k} not supported (dim > 1 or < 12)")

    # Normalize: make coefficient of q equal to 1
    f = R(f)
    c1 = f[1]
    if c1 != 1:
        # For these products, c1 should be 1 already (since Delta starts with q)
        # but let's be safe
        f = f / c1

    return f


# ── Newton's identity: compute s_m = p^m + q^m mod ell ───────────────

def power_sum_poly(m, N_mod, ell):
    """
    Compute s_m(e1) = p^m + q^m as a polynomial in e1 = p+q over F_ell,
    where pq = N_mod (known, reduced mod ell).

    Uses Newton's identity: s_i = e1*s_{i-1} - N*s_{i-2}
    with s_0 = 2, s_1 = e1.

    Returns a list of coefficients [c_0, c_1, ..., c_m] such that
    s_m(e1) = c_0 + c_1*e1 + c_2*e1^2 + ... + c_m*e1^m  (mod ell).
    """
    F = GF(ell)
    N_mod = F(N_mod)

    # Represent polynomials as lists of coefficients in F_ell
    # s_0 = 2 (constant polynomial)
    s_prev_prev = [F(2)]  # s_0
    # s_1 = e1 (linear polynomial)
    s_prev = [F(0), F(1)]  # s_1

    if m == 0:
        return s_prev_prev
    if m == 1:
        return s_prev

    for i in range(2, m + 1):
        # s_i = e1 * s_{i-1} - N * s_{i-2}

        # e1 * s_{i-1}: shift coefficients (multiply by x)
        term1 = [F(0)] + list(s_prev)  # len = len(s_prev) + 1

        # N * s_{i-2}
        term2 = [N_mod * c for c in s_prev_prev]

        # Subtract: s_i = term1 - term2
        max_len = max(len(term1), len(term2))
        s_curr = []
        for j in range(max_len):
            v1 = term1[j] if j < len(term1) else F(0)
            v2 = term2[j] if j < len(term2) else F(0)
            s_curr.append(v1 - v2)

        s_prev_prev = s_prev
        s_prev = s_curr

    return s_prev


def eval_poly_mod(coeffs, x, F):
    """Evaluate polynomial with coefficients in F at x using Horner's method."""
    result = F(0)
    for c in reversed(coeffs):
        result = result * F(x) + F(c)
    return result


def solve_congruence_channel(a_N_mod_ell, N, k, ell):
    """
    Given a_N(f_k) mod ell, N, weight k, and congruence prime ell,
    solve for p+q mod ell.

    Returns list of candidate (p+q) mod ell values.
    """
    F = GF(ell)

    # a_N ≡ 1 + s_{k-1} + N^{k-1} mod ell
    # => s_{k-1} ≡ a_N - 1 - N^{k-1} mod ell
    target = F(a_N_mod_ell) - F(1) - F(N) ** (k - 1)

    # s_{k-1}(e1) is a polynomial in e1 = p+q
    coeffs = power_sum_poly(k - 1, int(N) % ell, ell)

    # Find all e1 in F_ell such that s_{k-1}(e1) = target
    candidates = []
    for e1 in range(ell):
        val = eval_poly_mod(coeffs, e1, F)
        if val == target:
            candidates.append(int(e1))

    return candidates


def solve_for_p_mod_ell(e1_candidates, N, ell):
    """
    Given candidates for p+q mod ell and p*q = N, solve for p mod ell.

    For each e1 = p+q mod ell:
      p^2 - e1*p + N ≡ 0 mod ell
      discriminant = e1^2 - 4N mod ell

    Returns list of (e1, p_candidates) tuples.
    """
    F = GF(ell)
    results = []
    for e1 in e1_candidates:
        disc = F(e1) ** 2 - F(4) * F(N)
        if disc == F(0):
            # Double root
            p_val = F(e1) / F(2)
            results.append((int(e1), [int(p_val)]))
        elif disc.is_square():
            sqrt_disc = disc.sqrt()
            p1 = (F(e1) + sqrt_disc) / F(2)
            p2 = (F(e1) - sqrt_disc) / F(2)
            results.append((int(e1), [int(p1), int(p2)]))
        else:
            # No solution mod ell (discriminant is not a QR)
            results.append((int(e1), []))

    return results


# ── Bernoulli number congruence primes ────────────────────────────────

def get_congruence_primes():
    """
    Return the Eisenstein congruence data:
    (weight, congruence_prime, B_k numerator).

    For each weight k with dim S_k(Gamma_0(1)) = 1, the congruence prime ell
    divides the numerator of B_k (the k-th Bernoulli number).
    """
    # Compute Bernoulli numerator primes for weights 12, 16, 18, 20, 22
    channels = []
    for k in [12, 16, 18, 20, 22]:
        Bk = bernoulli(k)
        numer = abs(Bk.numerator())
        # Factor the numerator to find prime divisors
        for p, e in factor(numer):
            p_int = int(p)
            # Eisenstein congruence requires ell > k-1 (so mod-ell Galois
            # representation is irreducible and congruence to sigma_{k-1} holds)
            if p_int > k - 1:
                channels.append({
                    'weight': int(k),
                    'ell': p_int,
                    'bernoulli_numer': int(numer),
                })
    return channels


# ── Main experiment ───────────────────────────────────────────────────

def main():
    print("=" * 76, flush=True)
    print("E13: Bach-Charles congruence factoring channel", flush=True)
    print("=" * 76, flush=True)

    # Get congruence channels
    channels = get_congruence_primes()
    print(f"\nCongruence channels found:", flush=True)
    print(f"  {'Weight':>6} {'ell':>8} {'B_k numer':>12} {'max_roots':>10} {'bits':>6}",
          flush=True)
    print(f"  {'-'*46}", flush=True)
    for ch in channels:
        max_roots = ch['weight'] - 1
        bits = float(log(ch['ell'] / max_roots, 2))
        ch['max_roots'] = max_roots
        ch['approx_bits'] = round(float(bits), 1)
        print(f"  {ch['weight']:>6} {ch['ell']:>8} {ch['bernoulli_numer']:>12} "
              f"{max_roots:>10} {bits:>6.1f}", flush=True)
    total_approx_bits = sum(ch['approx_bits'] for ch in channels)
    print(f"\n  Total approx bits: {total_approx_bits:.1f}", flush=True)

    # Generate balanced semiprimes
    bit_sizes = [10, 12, 14, 16]
    count_per = 6
    semiprimes = balanced_semiprimes(bit_sizes, count_per_size=count_per)
    print(f"\nGenerated {len(semiprimes)} balanced semiprimes across bit sizes {bit_sizes}",
          flush=True)

    # Compute eigenform q-expansions
    max_N = max(N for N, p, q in semiprimes)
    prec = int(max_N) + 10
    print(f"\nComputing eigenform q-expansions to precision {prec}...", flush=True)

    eigenforms = {}
    for k in [12, 16, 18, 20, 22]:
        t0 = time.time()
        eigenforms[k] = eigenform_qexp(k, prec)
        dt = time.time() - t0
        print(f"  Weight {k}: computed in {dt:.2f}s", flush=True)

    # ── Main analysis loop ────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("CHANNEL ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    all_results = []

    for N, p, q in semiprimes:
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)
        e1_true = p_int + q_int  # true p+q

        result = {
            'N': N_int, 'p': p_int, 'q': q_int,
            'bits_N': int(N_int).bit_length(),
            'e1_true': e1_true,
            'channels': [],
        }

        total_bits = 0.0
        all_p_candidates = {}  # ell -> set of p mod ell candidates

        for ch in channels:
            k = ch['weight']
            ell = ch['ell']

            # Get a_N(f_k) from q-expansion
            f = eigenforms[k]
            a_N = int(f[N_int])

            # Also get a_p, a_q for verification
            a_p = int(f[p_int])
            a_q = int(f[q_int])

            # Verify multiplicativity
            assert a_N == a_p * a_q, f"Multiplicativity failed: {a_N} != {a_p}*{a_q}"

            # Verify Eisenstein congruence at primes
            cong_p = (1 + pow(p_int, k - 1, ell)) % ell
            cong_q = (1 + pow(q_int, k - 1, ell)) % ell
            assert a_p % ell == cong_p, \
                f"Congruence failed at p={p_int}: a_p mod {ell} = {a_p % ell}, expected {cong_p}"
            assert a_q % ell == cong_q, \
                f"Congruence failed at q={q_int}: a_q mod {ell} = {a_q % ell}, expected {cong_q}"

            # Solve congruence channel
            a_N_mod = int(a_N) % ell
            e1_candidates = solve_congruence_channel(a_N_mod, N_int, k, ell)
            e1_true_mod = e1_true % ell

            # Verify true e1 is among candidates
            found_true = e1_true_mod in e1_candidates

            # Solve for p mod ell from e1 candidates
            p_solutions = solve_for_p_mod_ell(e1_candidates, N_int, ell)
            p_true_mod = p_int % ell
            q_true_mod = q_int % ell

            # Collect all valid p candidates
            p_cands = set()
            for e1_val, p_vals in p_solutions:
                for pv in p_vals:
                    p_cands.add(int(pv))

            # Check true p is in candidates
            p_found = p_true_mod in p_cands or q_true_mod in p_cands

            # Information: bits = log2(ell / #candidates)
            n_cands = max(len(p_cands), 1)
            bits = float(log(ell, 2)) - float(log(n_cands, 2)) if n_cands < ell else 0.0
            total_bits += bits

            all_p_candidates[ell] = p_cands

            ch_result = {
                'weight': int(k),
                'ell': int(ell),
                'a_N_mod_ell': int(a_N_mod),
                'n_e1_candidates': len(e1_candidates),
                'n_p_candidates': len(p_cands),
                'e1_true_found': found_true,
                'p_true_found': p_found,
                'bits': round(float(bits), 2),
            }
            result['channels'].append(ch_result)

        result['total_bits'] = round(float(total_bits), 2)
        result['bits_needed'] = int(N_int).bit_length() // 4  # Coppersmith n/4

        # CRT combination: how many candidates survive all channels?
        # (This is exponential to compute exactly for large ell, so estimate)
        total_cand_product = 1
        for ch_res in result['channels']:
            total_cand_product *= ch_res['n_p_candidates']
        result['crt_candidate_product'] = int(total_cand_product)

        # The "effective" number after CRT: product / lcm of ells
        total_ell_product = 1
        for ch in channels:
            total_ell_product *= ch['ell']
        result['total_ell_product'] = int(total_ell_product)
        result['crt_reduction_ratio'] = float(total_cand_product) / float(total_ell_product) \
            if total_ell_product > 0 else 0.0

        all_results.append(result)

        # Print summary
        ch_str = " | ".join(f"k{ch_r['weight']}:{ch_r['n_p_candidates']}/{ch_r['ell']}"
                            f"({ch_r['bits']:.1f}b)"
                            for ch_r in result['channels'])
        print(f"N={N_int:>7} p={p_int:>5} q={q_int:>5} | {ch_str} | "
              f"total={total_bits:.1f}b need={result['bits_needed']}b", flush=True)

    # ── Save results ──────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E13_congruence_channel_results.json')
    safe_json_dump(all_results, out_path)

    # ── Statistical summary ───────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("STATISTICAL SUMMARY", flush=True)
    print("=" * 76, flush=True)

    print(f"\n{'Channel':>12} {'Mean #cands':>12} {'Mean bits':>10} "
          f"{'p_found%':>10} {'e1_found%':>10}", flush=True)
    print(f"{'-'*56}", flush=True)

    for ch in channels:
        k = ch['weight']
        ell = ch['ell']
        ch_data = [r for res in all_results
                   for r in res['channels']
                   if r['weight'] == k and r['ell'] == ell]
        mean_cands = np.mean([d['n_p_candidates'] for d in ch_data])
        mean_bits = np.mean([d['bits'] for d in ch_data])
        p_found_pct = 100.0 * np.mean([d['p_true_found'] for d in ch_data])
        e1_found_pct = 100.0 * np.mean([d['e1_true_found'] for d in ch_data])

        print(f"  k={k:>2} l={ell:>6} {mean_cands:>12.1f} {mean_bits:>10.1f} "
              f"{p_found_pct:>10.1f} {e1_found_pct:>10.1f}", flush=True)

    total_bits_all = [r['total_bits'] for r in all_results]
    bits_needed_all = [r['bits_needed'] for r in all_results]
    print(f"\n  Mean total bits per semiprime: {np.mean(total_bits_all):.1f}", flush=True)
    print(f"  Mean bits needed (n/4):        {np.mean(bits_needed_all):.1f}", flush=True)
    print(f"  Ratio (bits obtained / needed): {np.mean(total_bits_all)/np.mean(bits_needed_all):.3f}",
          flush=True)

    # ── Bit independence test ─────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("BIT INDEPENDENCE TEST", flush=True)
    print("=" * 76, flush=True)

    # For each pair of channels, check if bits are independent
    # (i.e., CRT candidates are approximately product of individual candidates)
    for i, ch_i in enumerate(channels):
        for j, ch_j in enumerate(channels):
            if j <= i:
                continue
            ki, li = ch_i['weight'], ch_i['ell']
            kj, lj = ch_j['weight'], ch_j['ell']

            # For each semiprime, check if candidates from channel i
            # are independent of candidates from channel j
            # (We can't easily check CRT combination at this scale,
            # but we can check if the candidate counts are as expected)
            cands_i = [r for res in all_results for r in res['channels']
                       if r['weight'] == ki and r['ell'] == li]
            cands_j = [r for res in all_results for r in res['channels']
                       if r['weight'] == kj and r['ell'] == lj]

            mean_i = np.mean([d['n_p_candidates'] for d in cands_i])
            mean_j = np.mean([d['n_p_candidates'] for d in cands_j])

            print(f"  k={ki},l={li} x k={kj},l={lj}: "
                  f"mean_cands = {mean_i:.1f} x {mean_j:.1f} = {mean_i*mean_j:.0f} "
                  f"(product bound: {li*lj})", flush=True)

    # ── Cost analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("COST ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    print(f"\n  Computing a_N(f_k) requires q-expansion to N terms:", flush=True)
    print(f"  Cost: O(N * k) per eigenform (multiply power series)", flush=True)
    print(f"  This is O(N) = O(2^n) -- EXPONENTIAL in input size n = log2(N).", flush=True)
    print(f"", flush=True)
    print(f"  The congruence channel WORKS (information is real) but", flush=True)
    print(f"  requires O(N) computation per channel -- same as trial division.", flush=True)
    print(f"", flush=True)
    print(f"  For poly(log N): would need to compute a_N(f_k) mod ell", flush=True)
    print(f"  WITHOUT computing the full q-expansion. EC algorithm gives", flush=True)
    print(f"  a_p(f_k) at PRIMES in poly(log p). For composite N = pq,", flush=True)
    print(f"  a_N = a_p * a_q by multiplicativity -- requires FACTORING.", flush=True)
    print(f"", flush=True)
    print(f"  This is exactly the Bach-Charles barrier in action.", flush=True)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    mean_total = np.mean(total_bits_all)
    mean_needed = np.mean(bits_needed_all)

    print(f"\n  The Eisenstein congruence channel WORKS:", flush=True)
    print(f"  - Each channel produces ~{mean_total/len(channels):.1f} bits of factor info", flush=True)
    print(f"  - {len(channels)} channels yield ~{mean_total:.1f} bits total", flush=True)
    print(f"  - Coppersmith threshold: {mean_needed:.0f} bits", flush=True)
    print(f"  - Ratio: {mean_total/mean_needed:.3f} (need ratio >= 1.0)", flush=True)
    print(f"", flush=True)
    print(f"  BUT the computation costs O(N) per channel:", flush=True)
    print(f"  - Computing a_N(f_k) requires expanding q-series to N terms", flush=True)
    print(f"  - This is equivalent to trial division in cost", flush=True)
    print(f"  - No poly(log N) shortcut exists (Bach-Charles)", flush=True)
    print(f"", flush=True)

    if mean_total >= mean_needed:
        print(f"  INFORMATION SUFFICIENT: The congruence channels contain enough", flush=True)
        print(f"  bits to factor (≥ n/4). The barrier is COMPUTATIONAL, not", flush=True)
        print(f"  INFORMATION-THEORETIC. The information IS there in eigenform", flush=True)
        print(f"  products -- it just costs O(N) to extract.", flush=True)
    else:
        shortfall = mean_needed - mean_total
        channels_needed = int(np.ceil(shortfall / (mean_total / len(channels))))
        print(f"  INFORMATION INSUFFICIENT with {len(channels)} channels.", flush=True)
        print(f"  Shortfall: {shortfall:.0f} bits. Need ~{channels_needed} more channels.", flush=True)
        print(f"  Additional channels from higher-weight Bernoulli primes could", flush=True)
        print(f"  close the gap, but computation remains O(N) per channel.", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
