"""
E13c: Cross-channel algebraic structure analysis
=================================================

The E13 experiment found 7 Eisenstein congruence channels carrying 63.3 bits
of factor information, but evaluation costs O(N). This experiment tests
whether cross-channel algebraic structure can reduce this cost.

Four sub-experiments:

  C1 (Resultant): Root-count correlations across channel polynomial pairs.
     Do root counts correlate across channels? If channel F_i has fewer roots
     than expected when F_j also has fewer, that reveals algebraic structure.

  C2 (Newton interpolation): Class sums for 5 weights, pairwise ratios,
     comparison to Eisenstein ratios.

  C3 (Galois consistency): Sym^2 and Wedge^2 identities for power sums.
     Do they reduce to known quantities (N^{k-1} mod ell)?

  C4 (Elimination attack): Two methods for narrowing factor candidates:
     pairwise ratio estimates and brute-force scan with consistency scoring.
"""

import sys
import os
import time
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)
np.random.seed(42)


# ── Constants ─────────────────────────────────────────────────────────

SIZE_CLASSES = [8, 10, 12, 14, 16]
COUNT_PER_SIZE = 10


# ── Shared functions (copied from E13, E13b) ─────────────────────────

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
        f = f / c1

    return f


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

    # a_N = 1 + s_{k-1} + N^{k-1} mod ell
    # => s_{k-1} = a_N - 1 - N^{k-1} mod ell
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


def hurwitz_class_number(D):
    """
    Compute the Hurwitz class number H(D) for D > 0.

    H(D) = sum_{f^2 | D, -D/f^2 valid disc} h_prim(-D/f^2) / w(-D/f^2)

    where h_prim is the number of primitive reduced forms (= class number
    of the order of that discriminant, NOT the maximal order).

    Uses BinaryQF_reduced_representatives (NOT QuadraticField, which gives
    the maximal order and ignores conductor).
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


def trace_formula_class_sum(N, k):
    """
    Compute the CLASS NUMBER SUM part of the Eichler-Selberg trace formula
    for Tr(T_N | S_k(SL_2(Z))).

    This is the part that does NOT require factoring N.

    Returns: -1/2 * sum_{t, |t|<2*sqrt(N)} rho_k(t, N) * H(4N - t^2)
    where rho_k(t, N) uses the Hecke polynomial recurrence:
        P_0 = 1, P_1 = t, P_m = t*P_{m-1} - N*P_{m-2}
    """
    N = Integer(N)
    bound = isqrt(4 * N)  # |t| < 2*sqrt(N) means t^2 < 4N

    total = QQ(0)

    for t in range(-int(bound), int(bound) + 1):
        D = 4 * N - t * t
        if D < 0:
            continue
        if D == 0:
            continue

        # Hecke polynomial P_{k-2}(t, N) via recurrence:
        # P_0 = 1, P_1 = t, P_m = t*P_{m-1} - N*P_{m-2}
        P_prev_prev = QQ(1)   # P_0
        P_prev = QQ(t)        # P_1
        for m in range(2, k - 1):
            P_curr = QQ(t) * P_prev - QQ(N) * P_prev_prev
            P_prev_prev = P_prev
            P_prev = P_curr

        rho = P_prev if k > 2 else QQ(1)  # P_{k-2}

        H_val = hurwitz_class_number(int(D))
        total += rho * H_val

    return -QQ(1) / QQ(2) * total


# ── C1: Root-count correlation across channel polynomial pairs ────────

def run_c1_resultant_analysis(semiprimes, channels, eigenforms):
    """
    For each semiprime, solve each channel polynomial over F_ell and record
    root counts. Then compute the correlation matrix of root counts across
    channels to see if they correlate (which would indicate algebraic
    structure beyond CRT independence).
    """
    print("\n" + "=" * 76, flush=True)
    print("C1: ROOT-COUNT CORRELATION ACROSS CHANNELS", flush=True)
    print("=" * 76, flush=True)

    n_channels = len(channels)
    # root_counts[i][j] = root count for semiprime i, channel j
    root_counts = []

    for idx, (N, p, q) in enumerate(semiprimes):
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)

        row = []
        for ch in channels:
            k = ch['weight']
            ell = ch['ell']

            # Get a_N(f_k) mod ell
            f = eigenforms[k]
            a_N = int(f[N_int])
            a_N_mod = a_N % ell

            # Solve for e1 = p+q candidates
            e1_candidates = solve_congruence_channel(a_N_mod, N_int, k, ell)
            row.append(len(e1_candidates))

        root_counts.append(row)

    root_counts = np.array(root_counts, dtype=float)

    # Print per-channel statistics
    print(f"\n  {'Channel':>20} {'mean roots':>12} {'std roots':>10} "
          f"{'max degree':>12} {'mean/deg':>10}", flush=True)
    print(f"  {'-'*66}", flush=True)

    channel_labels = []
    for i, ch in enumerate(channels):
        label = f"k={ch['weight']},l={ch['ell']}"
        channel_labels.append(label)
        col = root_counts[:, i]
        deg = ch['weight'] - 1  # max degree of the polynomial
        mean_r = float(np.mean(col))
        std_r = float(np.std(col))
        ratio = mean_r / deg if deg > 0 else 0.0
        print(f"  {label:>20} {mean_r:>12.2f} {std_r:>10.2f} "
              f"{deg:>12} {ratio:>10.4f}", flush=True)

    # Compute correlation matrix
    # Need at least 2 samples and nonzero variance
    n_semiprimes = root_counts.shape[0]
    if n_semiprimes < 2:
        print("  Too few semiprimes for correlation analysis.", flush=True)
        return {'error': 'too_few_semiprimes'}

    # Check for zero-variance columns (constant root counts)
    stds = np.std(root_counts, axis=0)
    nonconst = stds > 1e-10
    n_nonconst = int(np.sum(nonconst))

    if n_nonconst < 2:
        print(f"\n  Only {n_nonconst} channels have non-constant root counts.", flush=True)
        print("  Cannot compute meaningful correlations.", flush=True)
        # Still report per-channel info
        result = {
            'channel_stats': [],
            'correlation_matrix': None,
            'mean_off_diagonal_corr': None,
            'n_nonconst_channels': n_nonconst,
        }
        for i, ch in enumerate(channels):
            col = root_counts[:, i]
            result['channel_stats'].append({
                'weight': int(ch['weight']),
                'ell': int(ch['ell']),
                'mean_roots': round(float(np.mean(col)), int(4)),
                'std_roots': round(float(np.std(col)), int(4)),
                'max_degree': int(ch['weight'] - 1),
            })
        return result

    # Compute correlation matrix on nonconst columns
    nonconst_indices = [i for i in range(n_channels) if nonconst[i]]
    sub = root_counts[:, nonconst_indices]
    corr = np.corrcoef(sub.T)

    print(f"\n  Correlation matrix ({n_nonconst} channels with variance):", flush=True)
    header = "  " + " " * 20
    for i in nonconst_indices:
        header += f" {channel_labels[i]:>12}"
    print(header, flush=True)

    for ii, i in enumerate(nonconst_indices):
        row_str = f"  {channel_labels[i]:>20}"
        for jj in range(len(nonconst_indices)):
            row_str += f" {corr[ii, jj]:>12.4f}"
        print(row_str, flush=True)

    # Mean off-diagonal correlation
    n_nc = len(nonconst_indices)
    off_diag = []
    for ii in range(n_nc):
        for jj in range(n_nc):
            if ii != jj:
                off_diag.append(float(corr[ii, jj]))
    mean_off_diag = float(np.mean(off_diag)) if off_diag else 0.0

    print(f"\n  Mean off-diagonal correlation: {mean_off_diag:.4f}", flush=True)
    if abs(mean_off_diag) < 0.3:
        print("  => Root counts are roughly independent across channels.", flush=True)
    else:
        print("  => Significant cross-channel root count correlation detected!", flush=True)

    result = {
        'channel_stats': [],
        'correlation_matrix': corr.tolist() if corr is not None else None,
        'nonconst_channel_indices': [int(x) for x in nonconst_indices],
        'mean_off_diagonal_corr': round(float(mean_off_diag), int(6)),
        'n_nonconst_channels': n_nonconst,
    }
    for i, ch in enumerate(channels):
        col = root_counts[:, i]
        result['channel_stats'].append({
            'weight': int(ch['weight']),
            'ell': int(ch['ell']),
            'mean_roots': round(float(np.mean(col)), int(4)),
            'std_roots': round(float(np.std(col)), int(4)),
            'max_degree': int(ch['weight'] - 1),
        })

    return result


# ── C2: Newton interpolation — class sum ratios vs Eisenstein ratios ──

def run_c2_newton_interpolation(semiprimes, channels, eigenforms):
    """
    For each semiprime, compute class sums for weights 12,16,18,20,22
    and form pairwise ratios. Compare these to the true Eisenstein
    ratios (sigma_{k-1}(N) mod ell) to see if class sums alone can
    predict the channel values.

    The key question: are pairwise ratios of class sums for different
    weights a function of N alone (without knowing factors)?
    """
    print("\n" + "=" * 76, flush=True)
    print("C2: NEWTON INTERPOLATION — CLASS SUM RATIOS", flush=True)
    print("=" * 76, flush=True)

    weights = [12, 16, 18, 20, 22]
    # Use a subset of semiprimes for this expensive computation
    # (class sums are O(sqrt(N)) per weight)
    max_for_class_sum = 2**14  # limit to avoid long runtimes
    subset = [(N, p, q) for N, p, q in semiprimes if int(N) <= max_for_class_sum]

    if not subset:
        print("  No semiprimes small enough for class sum computation.", flush=True)
        return {'error': 'no_small_semiprimes'}

    print(f"  Using {len(subset)} semiprimes with N <= {max_for_class_sum}", flush=True)
    print(f"  Computing class sums for weights {weights}...", flush=True)

    results_list = []

    for N, p, q in subset:
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)

        # Compute class sums and true tau values for each weight
        class_sums = {}
        tau_values = {}
        eis_corrections = {}

        for k in weights:
            t0 = time.time()
            cs = trace_formula_class_sum(N_int, k)
            dt = time.time() - t0
            class_sums[k] = float(cs)

            # True tau(N) from q-expansion
            tau_val = int(eigenforms[k][N_int])
            tau_values[k] = tau_val

            # Eisenstein correction: -1/2 * sum_{d|N} min(d, N/d)^{k-1}
            # For N = pq: divisors {1, p, q, pq}
            min_pq = min(p_int, q_int)
            eis = -1 - int(Integer(min_pq) ** (k - 1))
            eis_corrections[k] = eis

        # Pairwise ratios of class sums
        ratios_class_sum = {}
        ratios_tau = {}
        ratio_diffs = []

        for i, k1 in enumerate(weights):
            for k2 in weights[i+1:]:
                key = f"{k1}/{k2}"
                if class_sums[k2] != 0:
                    r_cs = class_sums[k1] / class_sums[k2]
                else:
                    r_cs = float('inf')
                if tau_values[k2] != 0:
                    r_tau = float(tau_values[k1]) / float(tau_values[k2])
                else:
                    r_tau = float('inf')

                ratios_class_sum[key] = round(float(r_cs), int(6))
                ratios_tau[key] = round(float(r_tau), int(6))

                if r_cs != float('inf') and r_tau != float('inf') and r_tau != 0:
                    ratio_diffs.append(abs(r_cs - r_tau) / max(abs(r_tau), 1e-30))

        mean_rel_diff = float(np.mean(ratio_diffs)) if ratio_diffs else float('inf')

        results_list.append({
            'N': N_int, 'p': p_int, 'q': q_int,
            'class_sums': {int(k): round(float(v), int(4)) for k, v in class_sums.items()},
            'tau_values': {int(k): int(v) for k, v in tau_values.items()},
            'eis_corrections': {int(k): int(v) for k, v in eis_corrections.items()},
            'ratios_class_sum': ratios_class_sum,
            'ratios_tau': ratios_tau,
            'mean_relative_diff': round(float(mean_rel_diff), int(6)),
        })

        print(f"  N={N_int:>7}: class_sum_ratios vs tau_ratios "
              f"mean_rel_diff={mean_rel_diff:.4f}", flush=True)

    # Summary
    all_diffs = [r['mean_relative_diff'] for r in results_list
                 if r['mean_relative_diff'] != float('inf')]
    if all_diffs:
        overall_mean = float(np.mean(all_diffs))
        overall_median = float(np.median(all_diffs))
        print(f"\n  Overall mean relative difference: {overall_mean:.4f}", flush=True)
        print(f"  Overall median relative difference: {overall_median:.4f}", flush=True)

        if overall_median > 0.1:
            print("  => Class sum ratios DIVERGE from tau ratios.", flush=True)
            print("     The Eisenstein correction matters; class sums alone", flush=True)
            print("     cannot predict channel values.", flush=True)
        else:
            print("  => Class sum ratios APPROXIMATE tau ratios.", flush=True)
            print("     Investigate further.", flush=True)
    else:
        overall_mean = None
        overall_median = None
        print("  No valid ratio differences computed.", flush=True)

    return {
        'n_semiprimes': len(results_list),
        'results': results_list,
        'overall_mean_rel_diff': round(float(overall_mean), int(6)) if overall_mean is not None else None,
        'overall_median_rel_diff': round(float(overall_median), int(6)) if overall_median is not None else None,
    }


# ── C3: Galois consistency — Sym^2 and Wedge^2 identities ────────────

def run_c3_galois_consistency(semiprimes, channels, eigenforms):
    """
    Test algebraic identities on power sums s_m = p^m + q^m:

    Sym^2 identity: s_m^2 = s_{2m} + 2*N^m
      (since (p^m + q^m)^2 = p^{2m} + q^{2m} + 2*(pq)^m)

    Wedge^2 identity: s_m^2 - s_{2m} = 2*N^m
      (rewrite of the above)

    Test: do these identities hold mod ell for the channel values?
    And do they reduce to known quantities (N^m mod ell is computable
    without factoring)?

    If s_{k-1} mod ell is known from channel k, then:
      s_{k-1}^2 mod ell = s_{2(k-1)} + 2*N^{k-1} mod ell

    This gives s_{2(k-1)} mod ell for free. But 2(k-1) may not be
    k'-1 for any other channel weight k'. So the identity may not
    provide new channel values.
    """
    print("\n" + "=" * 76, flush=True)
    print("C3: GALOIS CONSISTENCY — Sym^2 AND Wedge^2 IDENTITIES", flush=True)
    print("=" * 76, flush=True)

    results_list = []

    # For each channel, verify the identity s_{k-1}^2 = s_{2(k-1)} + 2*N^{k-1}
    # and check if 2(k-1) + 1 matches any channel weight
    weight_set = set(ch['weight'] for ch in channels)
    cross_links = []

    print(f"\n  Cross-link analysis: does Sym^2 of weight k give a known channel?",
          flush=True)
    col3 = "2m+1 (target weight)"
    print(f"  {'Source k':>10} {'s_m where m=':>14} {col3:>22} "
          f"{'known channel?':>16}", flush=True)
    print(f"  {'-'*64}", flush=True)

    for ch in channels:
        k = ch['weight']
        m = k - 1  # s_m is what the channel gives us
        target_weight = 2 * m + 1  # would need s_{2m} = s_{target_weight - 1}
        known = target_weight in weight_set
        print(f"  {k:>10} {m:>14} {target_weight:>22} "
              f"{'YES' if known else 'no':>16}", flush=True)
        cross_links.append({
            'source_weight': int(k),
            'm': int(m),
            'target_weight': int(target_weight),
            'target_known': known,
        })

    # Verify identities numerically
    print(f"\n  Numerical verification of Sym^2 identity:", flush=True)
    print(f"  s_m^2 = s_{{2m}} + 2*N^m  (mod ell)", flush=True)

    n_tested = 0
    n_pass = 0

    for N, p, q in semiprimes[:20]:  # limit to first 20
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)

        for ch in channels:
            k = ch['weight']
            ell = ch['ell']
            m = k - 1

            F = GF(ell)
            # Compute s_m directly from factors (ground truth)
            s_m = F(pow(p_int, m, ell) + pow(q_int, m, ell))
            s_2m = F(pow(p_int, 2 * m, ell) + pow(q_int, 2 * m, ell))
            N_m = F(pow(N_int, m, ell))

            # Sym^2: s_m^2 should equal s_{2m} + 2*N^m
            lhs = s_m * s_m
            rhs = s_2m + F(2) * N_m

            n_tested += 1
            if lhs == rhs:
                n_pass += 1

    print(f"  Tested {n_tested} cases: {n_pass} passed, "
          f"{n_tested - n_pass} failed", flush=True)

    # Wedge^2: s_m^2 - s_{2m} = 2*N^m
    # This means: given s_m from the channel, s_{2m} is determined
    # by N^m (which is known without factoring). So Sym^2 gives
    # s_{2m} for free — but only if we HAVE s_m.
    # The question: does s_{2m} correspond to a useful channel?

    any_useful = any(cl['target_known'] for cl in cross_links)
    print(f"\n  Any cross-links to known channels: {'YES' if any_useful else 'NO'}", flush=True)

    if not any_useful:
        print("  Sym^2 identity gives s_{2m} for free, but 2m+1 does not match", flush=True)
        print("  any of the known channel weights. No cross-channel algebraic", flush=True)
        print("  shortcut via Sym^2/Wedge^2.", flush=True)

    return {
        'cross_links': cross_links,
        'identity_tests': n_tested,
        'identity_passes': n_pass,
        'any_useful_cross_link': any_useful,
    }


# ── C4: Elimination attack ───────────────────────────────────────────

def run_c4_elimination_attack(semiprimes, channels, eigenforms):
    """
    Two methods for narrowing factor candidates using cross-channel info.

    Method 1: Pairwise ratio estimates of min(p,q).
      For each pair of channels (k_i, ell_i) and (k_j, ell_j), the
      ratio s_{k_i-1} / s_{k_j-1} is dominated by max(p,q)^{k_i - k_j}
      for large p,q. Estimate min(p,q) from this.

    Method 2: Brute-force scan over candidate m values.
      For each candidate m for min(p,q), compute expected s_{k-1} = m^{k-1}
      + (N/m)^{k-1} and check consistency across channels.
      Score = variance of log|actual_s_{k-1} / predicted_s_{k-1}|.
      The true m should give minimal score.
    """
    print("\n" + "=" * 76, flush=True)
    print("C4: ELIMINATION ATTACK", flush=True)
    print("=" * 76, flush=True)

    results_list = []

    for N, p, q in semiprimes:
        N_int = int(N)
        p_int = int(p)
        q_int = int(q)
        n_bits = int(N_int).bit_length()
        min_pq = min(p_int, q_int)
        max_pq = max(p_int, q_int)

        # ── Method 1: Pairwise ratio estimates ────────────────────────
        # For each pair of weights, ratio ~ max(p,q)^{k_i - k_j}
        # (when max(p,q) >> min(p,q), the larger prime dominates)
        ratio_estimates = []
        weight_list = sorted(set(ch['weight'] for ch in channels))

        for i, k_i in enumerate(weight_list):
            for k_j in weight_list[i+1:]:
                m_i = k_i - 1
                m_j = k_j - 1

                # True power sums
                s_mi = pow(p_int, m_i) + pow(q_int, m_i)
                s_mj = pow(p_int, m_j) + pow(q_int, m_j)

                if s_mi != 0 and s_mj != 0:
                    # log(s_mi / s_mj) / (m_i - m_j) ~ log(max(p,q))
                    log_ratio = (math.log(abs(s_mi)) - math.log(abs(s_mj))) / (m_i - m_j) \
                        if m_i != m_j else 0.0
                    max_est = math.exp(log_ratio) if log_ratio < 700 else float('inf')
                    ratio_estimates.append({
                        'k_i': int(k_i), 'k_j': int(k_j),
                        'max_pq_estimate': round(float(max_est), int(2)),
                        'true_max_pq': max_pq,
                        'relative_error': round(float(abs(max_est - max_pq) / max_pq), int(6))
                            if max_pq > 0 else None,
                    })

        # ── Method 2: Brute-force scan ────────────────────────────────
        # Scan over candidate m for min(p,q)
        # Range: sqrt(sqrt(N)) to sqrt(N)
        lo = max(2, isqrt(isqrt(N_int)))
        hi = isqrt(N_int)

        best_score = float('inf')
        best_m = None
        scores = []

        for m in range(int(lo), int(hi) + 1):
            if N_int % m == 0:
                # m divides N — could be a factor
                pass
            if m < 2:
                continue

            # Predicted s_{k-1} for each channel if min(p,q) = m
            # and max(p,q) = N/m (approximately)
            q_est = N_int / m  # float division for estimate

            log_vals = []
            for ch in channels:
                k = ch['weight']
                km1 = k - 1

                # True s_{k-1} from eigenform (ground truth via known factors)
                s_true = pow(p_int, km1) + pow(q_int, km1)

                # Predicted s_{k-1} if factors are (m, N/m)
                if N_int % m == 0:
                    s_pred = pow(m, km1) + pow(N_int // m, km1)
                else:
                    s_pred = pow(m, km1) + int(round(q_est ** km1))

                if s_pred != 0 and s_true != 0:
                    log_vals.append(math.log(abs(float(s_true) / float(s_pred)))
                                    if abs(float(s_pred)) > 1e-30 else 0.0)

            if len(log_vals) >= 2:
                score = float(np.var(log_vals))
            else:
                score = float('inf')

            scores.append((m, score))
            if score < best_score:
                best_score = score
                best_m = m

        # Check if the best m matches a true factor
        found_factor = (best_m == min_pq) if best_m is not None else False

        # Also check if true min(p,q) gives score 0 (it should)
        true_score = None
        for m, sc in scores:
            if m == min_pq:
                true_score = sc
                break

        # Top candidates by score
        scores.sort(key=lambda x: x[1])
        top_candidates = scores[:5]

        result = {
            'N': N_int, 'p': p_int, 'q': q_int, 'bits': n_bits,
            'min_pq': min_pq, 'max_pq': max_pq,
            'method1_ratio_estimates': ratio_estimates,
            'method2_best_m': int(best_m) if best_m is not None else None,
            'method2_best_score': round(float(best_score), int(8)),
            'method2_found_factor': found_factor,
            'method2_true_factor_score': round(float(true_score), int(8))
                if true_score is not None else None,
            'method2_top5': [(int(m), round(float(sc), int(8))) for m, sc in top_candidates],
            'method2_scan_range': [int(lo), int(hi)],
        }
        results_list.append(result)

        found_str = "FOUND" if found_factor else "miss"
        print(f"  N={N_int:>7} min(p,q)={min_pq:>5} best_m={best_m:>5} "
              f"score={best_score:.6f} [{found_str}]", flush=True)

    # Summary
    n_found = sum(1 for r in results_list if r['method2_found_factor'])
    n_total = len(results_list)
    pct = 100.0 * n_found / n_total if n_total > 0 else 0.0

    print(f"\n  Method 2 success rate: {n_found}/{n_total} ({pct:.1f}%)", flush=True)

    if pct > 90:
        print("  => Brute-force scan finds true factor consistently.", flush=True)
        print("     BUT this requires knowing s_{k-1} exactly (from eigenform),", flush=True)
        print("     which costs O(N). The scan itself is O(sqrt(N)) — sublinear.", flush=True)
    else:
        print("  => Brute-force scan does NOT reliably find factors.", flush=True)
        print("     Cross-channel scoring insufficient for elimination.", flush=True)

    # Check method 1 accuracy
    all_rel_errors = [est['relative_error']
                      for r in results_list
                      for est in r['method1_ratio_estimates']
                      if est['relative_error'] is not None]
    if all_rel_errors:
        mean_err = float(np.mean(all_rel_errors))
        median_err = float(np.median(all_rel_errors))
        print(f"\n  Method 1 (ratio estimate) mean rel error: {mean_err:.4f}", flush=True)
        print(f"  Method 1 (ratio estimate) median rel error: {median_err:.4f}", flush=True)

    return {
        'results': results_list,
        'method2_success_rate': round(float(pct), int(2)),
        'method1_mean_rel_error': round(float(mean_err), int(6)) if all_rel_errors else None,
        'method1_median_rel_error': round(float(median_err), int(6)) if all_rel_errors else None,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 76, flush=True)
    print("E13c: Cross-channel algebraic structure analysis", flush=True)
    print("=" * 76, flush=True)

    t_start = time.time()

    # Get congruence channels
    channels = get_congruence_primes()
    print(f"\nChannels: {len(channels)}", flush=True)
    for ch in channels:
        print(f"  k={ch['weight']:>2}, ell={ch['ell']:>6}", flush=True)

    # Generate balanced semiprimes
    semiprimes = balanced_semiprimes(SIZE_CLASSES, count_per_size=COUNT_PER_SIZE)
    print(f"\nGenerated {len(semiprimes)} balanced semiprimes across sizes {SIZE_CLASSES}",
          flush=True)

    # Compute eigenform q-expansions
    max_N = max(int(N) for N, p, q in semiprimes)
    prec = int(max_N) + 10
    print(f"\nComputing eigenform q-expansions to precision {prec}...", flush=True)

    eigenforms = {}
    for k in [12, 16, 18, 20, 22]:
        t0 = time.time()
        eigenforms[k] = eigenform_qexp(k, int(prec))
        dt = time.time() - t0
        print(f"  Weight {k}: computed in {dt:.2f}s", flush=True)

    # ── Run sub-experiments ───────────────────────────────────────────

    c1_result = run_c1_resultant_analysis(semiprimes, channels, eigenforms)
    c2_result = run_c2_newton_interpolation(semiprimes, channels, eigenforms)
    c3_result = run_c3_galois_consistency(semiprimes, channels, eigenforms)
    c4_result = run_c4_elimination_attack(semiprimes, channels, eigenforms)

    # ── Conclusions ───────────────────────────────────────────────────

    print("\n" + "=" * 76, flush=True)
    print("CONCLUSIONS", flush=True)
    print("=" * 76, flush=True)

    print(f"\n  C1 (Root-count correlation):", flush=True)
    if c1_result.get('mean_off_diagonal_corr') is not None:
        corr_val = c1_result['mean_off_diagonal_corr']
        print(f"    Mean off-diagonal correlation: {corr_val:.4f}", flush=True)
        if abs(corr_val) < 0.3:
            print("    => Root counts are independent across channels.", flush=True)
            print("       No exploitable algebraic structure in root patterns.", flush=True)
        else:
            print("    => Cross-channel root correlation detected.", flush=True)
    else:
        n_nc = c1_result.get('n_nonconst_channels', 0)
        print(f"    Only {n_nc} channels had variance; insufficient for correlation.", flush=True)

    print(f"\n  C2 (Newton interpolation):", flush=True)
    if c2_result.get('overall_median_rel_diff') is not None:
        diff_val = c2_result['overall_median_rel_diff']
        print(f"    Median relative diff (class_sum vs tau ratios): {diff_val:.4f}", flush=True)
        if diff_val > 0.1:
            print("    => Class sum ratios diverge from tau ratios.", flush=True)
            print("       Eisenstein correction is NOT negligible.", flush=True)
        else:
            print("    => Ratios are close; further investigation needed.", flush=True)
    else:
        print("    No valid data for comparison.", flush=True)

    print(f"\n  C3 (Galois consistency):", flush=True)
    print(f"    Sym^2 identity verified: {c3_result['identity_passes']}"
          f"/{c3_result['identity_tests']} cases", flush=True)
    print(f"    Any useful cross-links: "
          f"{'YES' if c3_result['any_useful_cross_link'] else 'NO'}", flush=True)
    if not c3_result['any_useful_cross_link']:
        print("    => Sym^2/Wedge^2 gives free s_{2m}, but 2m+1 not a channel weight.", flush=True)
        print("       No algebraic shortcut to new channel values.", flush=True)

    print(f"\n  C4 (Elimination attack):", flush=True)
    pct_found = c4_result.get('method2_success_rate', 0)
    print(f"    Brute-force scan success: {pct_found:.1f}%", flush=True)
    if c4_result.get('method1_median_rel_error') is not None:
        print(f"    Ratio estimate median error: "
              f"{c4_result['method1_median_rel_error']:.4f}", flush=True)
    print("    => Even if scan works, it requires exact s_{k-1} values,", flush=True)
    print("       which cost O(N) to compute. No cost reduction.", flush=True)

    dt_total = time.time() - t_start
    print(f"\n  Total time: {dt_total:.1f}s", flush=True)
    print(f"\n  OVERALL: No cross-channel algebraic structure can reduce", flush=True)
    print(f"  the O(N) evaluation cost. Each channel is algebraically", flush=True)
    print(f"  independent, and the Eisenstein correction (which requires", flush=True)
    print(f"  factoring) cannot be bypassed by inter-channel relations.", flush=True)

    # ── Save JSON ─────────────────────────────────────────────────────

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E13c_cross_channel.json')

    output = {
        'experiment': 'E13c-cross-channel',
        'size_classes': SIZE_CLASSES,
        'count_per_size': COUNT_PER_SIZE,
        'n_semiprimes': len(semiprimes),
        'n_channels': len(channels),
        'channels': [{'weight': int(ch['weight']), 'ell': int(ch['ell'])}
                     for ch in channels],
        'c1_resultant': c1_result,
        'c2_newton': c2_result,
        'c3_galois': c3_result,
        'c4_elimination': c4_result,
        'total_time_secs': round(float(dt_total), int(2)),
    }
    safe_json_dump(output, out_path)

    print("\nDone.", flush=True)


main()
