#!/usr/bin/env sage
"""
E11: Comprehensive Poly(log N) Feature Extraction
==================================================

Tests whether ANY combination of ~111 poly(log N)-computable features
can predict the hinge scalar S_D(N) = chi_D(p) + chi_D(q) for semiprimes N=pq.

This is the "universal test" of the RJI(N) barrier: if no feature combination
achieves R^2_CV > 0, then the barrier extends to ALL poly(log N) computations
we can enumerate.

Feature groups (~111 total):
  1. Jacobi symbols J(k, N) for 20 small primes               (20 features)
  2. Modular exponentiations g^M mod N                          (25 features)
  3. Euler residuals g^{N - 2*isqrt(N) + 1} mod N              (5 features)
  4. Integer carry features floor(t^2/N)                        (15 features)
  5. CF convergents of sqrt(N)                                  (20 features)
  6. Pollard p-1 remnants g^{20!} mod N                         (6 features)
  7. Mixed cross-group interactions                             (10 features)
  8. N mod k controls                                           (5 features)
  9. Random controls (null hypothesis)                          (5 features)

Targets:
  - S_D(N) = chi_D(p) + chi_D(q) for D in {-3, -4, 5, -7, 8}
  - p / sqrt(N) (normalized smaller factor)

Analysis:
  - Per-feature Pearson correlation with each target
  - Per-feature ANOVA F-statistic for categorical targets
  - Ridge regression R^2_CV (closed-form LOOCV via hat matrix)
  - Permutation test (200 shuffles) for statistical significance
  - Feature group comparison
"""
import sys
import os
import json
import time
import numpy as np
from numpy.linalg import svd
from collections import OrderedDict

from sage.all import (
    kronecker_symbol, next_prime, isqrt, ZZ, gcd,
    power_mod, is_prime, set_random_seed
)

# Reproducibility
set_random_seed(42)
np.random.seed(42)

# ============================================================
# Constants
# ============================================================

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
DISCRIMINANTS = [-3, -4, 5, -7, 8]
MODEXP_BASES = [2, 3, 5, 7, 11]


class SageEncoder(json.JSONEncoder):
    """JSON encoder that handles Sage numeric types."""
    def default(self, obj):
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        try:
            return int(obj)
        except (TypeError, ValueError):
            return str(obj)


# ============================================================
# Semiprime generation
# ============================================================

def generate_semiprimes_fixed_bits(bit_size, count, min_ratio=0.3):
    """Generate balanced semiprimes N=pq with exactly `bit_size` bits.

    Only produces semiprimes with p/q >= min_ratio (balanced).
    This ensures trial-division-equivalent features cannot succeed,
    which is the regime relevant to the barrier question.
    """
    results = []
    seen = set()
    half = bit_size // 2

    # p ranges from ~2^{half-2} to ~2^{half} to get varied but balanced pairs
    p_lo = max(int(3), int(2**(half - 2)))
    p_hi = int(2**(half + 1))

    p = next_prime(p_lo)
    while len(results) < count and int(p) < p_hi:
        q_lo = max(int(p) + 2, (2**(bit_size - 1) + int(p) - 1) // int(p))
        q_hi = (2**bit_size - 1) // int(p)
        if q_lo > q_hi:
            p = next_prime(p + 1)
            continue

        q = next_prime(q_lo)
        while int(q) <= q_hi and len(results) < count:
            if q != p:
                pmin, pmax = sorted([int(p), int(q)])
                ratio = float(pmin) / float(pmax)
                if ratio >= min_ratio:
                    N = pmin * pmax
                    if N not in seen and N.bit_length() == bit_size:
                        results.append((N, pmin, pmax))
                        seen.add(N)
            q = next_prime(q + 1)
        p = next_prime(p + 1)

    return results[:count]


# ============================================================
# Continued fraction of sqrt(N)
# ============================================================

def cf_sqrt(N, max_terms=15):
    """Compute CF expansion of sqrt(N) and convergent remainders.

    Returns:
        partial_quotients: list of CF coefficients [a0, a1, ...]
        remainders: list of p_k^2 - N*q_k^2 for k=1,2,...
    """
    N = int(N)
    a0 = int(isqrt(N))
    if a0 * a0 == N:
        return [a0], []

    partial_quotients = [a0]
    remainders = []

    m, d, a = 0, 1, a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    for _ in range(max_terms):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d

        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        partial_quotients.append(int(a))
        remainders.append(int(p_curr * p_curr - N * q_curr * q_curr))

        if a == 2 * a0:
            break

    return partial_quotients, remainders


# ============================================================
# Feature computation
# ============================================================

def compute_features(N):
    """Compute ~111 poly(log N) features for semiprime N.

    All features are computable without knowing the factorization.
    Returns OrderedDict mapping feature_name -> float.
    """
    N = int(N)
    features = OrderedDict()
    sqrtN = int(isqrt(N))

    # --- Group 1: Jacobi symbols (20 features) ---
    for k in SMALL_PRIMES:
        features['J_%d' % k] = int(kronecker_symbol(k, N))

    # --- Group 2: Modular exponentiations (25 features) ---
    for g in MODEXP_BASES:
        g_int = int(g)
        d = int(gcd(g_int, N))
        if d > 1:
            features['modexp_%d_N' % g_int] = -1.0
            features['modexp_%d_half' % g_int] = -1.0
            features['modexp_%d_N2' % g_int] = -1.0
            features['J_modexp_%d' % g_int] = 0
            features['modexp_%d_mod4' % g_int] = -1
        else:
            gN = int(power_mod(g_int, N, N))
            g_half = int(power_mod(g_int, (N - 1) // 2, N))
            gN2 = int(power_mod(g_int, N * N, N))
            features['modexp_%d_N' % g_int] = gN / float(N)
            features['modexp_%d_half' % g_int] = g_half / float(N)
            features['modexp_%d_N2' % g_int] = gN2 / float(N)
            features['J_modexp_%d' % g_int] = int(kronecker_symbol(gN, N))
            features['modexp_%d_mod4' % g_int] = gN % 4

    # --- Group 3: Euler residuals (5 features) ---
    # g^{N - 2*isqrt(N) + 1} mod N = g^{(p-1)(q-1) + (p+q) - 2*isqrt(N)} mod N
    # = g^{p+q - 2*isqrt(N)} mod N  (since g^{phi(N)} = 1)
    # This encodes the error in estimating p+q by 2*sqrt(N).
    euler_exp = N - 2 * sqrtN + 1
    for g in MODEXP_BASES:
        g_int = int(g)
        if int(gcd(g_int, N)) > 1:
            features['euler_resid_%d' % g_int] = -1.0
        else:
            val = int(power_mod(g_int, euler_exp, N))
            features['euler_resid_%d' % g_int] = val / float(N)

    # --- Group 4: Carry/integer features (15 features) ---
    for denom in [2, 3, 4, 5]:
        t = N // denom
        high_word = (t * t) // N
        hw_mod = int(high_word) % N
        features['carry_hw_%d' % denom] = hw_mod / float(N)

        if hw_mod > 0 and int(gcd(hw_mod, N)) == 1:
            features['J_carry_%d' % denom] = int(kronecker_symbol(hw_mod, N))
        else:
            features['J_carry_%d' % denom] = 0

    half_sq = (N // 2) * (N // 2)
    for k in range(2, 9):
        carry_val = int((k * half_sq) // N)
        features['carry_par_%d' % k] = (-1) ** (carry_val % 2)

    # --- Group 5: CF convergent features (20 features) ---
    pqs, remainders = cf_sqrt(N, max_terms=12)

    for i in range(10):
        features['cf_pq_%d' % i] = int(pqs[i]) if i < len(pqs) else 0

    for i in range(5):
        if i < len(remainders) and remainders[i] != 0:
            r = remainders[i]
            features['cf_rem_%d' % i] = abs(r) / float(N)
            r_mod = abs(r) % N
            if r_mod > 0 and int(gcd(r_mod, N)) == 1:
                features['J_cf_rem_%d' % i] = int(kronecker_symbol(r_mod, N))
            else:
                features['J_cf_rem_%d' % i] = 0
        else:
            features['cf_rem_%d' % i] = 0.0
            features['J_cf_rem_%d' % i] = 0

    # --- Group 6: Pollard p-1 remnants (6 features) ---
    # g^{20!} mod N: if ord_p(g) | 20!, then gcd(result-1, N) reveals p
    for g in [2, 3, 5]:
        g_int = int(g)
        if int(gcd(g_int, N)) > 1:
            features['pollard_%d_found' % g_int] = 1
            features['pollard_%d_val' % g_int] = -1.0
        else:
            val = g_int
            for k in range(2, 21):
                val = int(power_mod(val, k, N))
            d = int(gcd(val - 1, N))
            features['pollard_%d_found' % g_int] = 1 if (1 < d < N) else 0
            features['pollard_%d_val' % g_int] = val / float(N) if d == 1 else -1.0

    # --- Group 7: Mixed/interaction features (10 features) ---
    g2N = features.get('modexp_2_N', 0.0)
    g3N = features.get('modexp_3_N', 0.0)

    for k in [2, 3, 5, 7, 11]:
        jk = features.get('J_%d' % k, 0)
        features['mix_J%d_g2N' % k] = jk * g2N

    if g2N >= 0 and g3N >= 0:
        features['mix_g2g3_diff'] = abs(g2N - g3N)
        # Use int.__xor__ because Sage preparser converts ^ to **
        features['mix_g2g3_xor'] = int.__xor__(int(g2N * N), int(g3N * N)) / float(max(N, 1))
    else:
        features['mix_g2g3_diff'] = 0.0
        features['mix_g2g3_xor'] = 0.0

    er2 = features.get('euler_resid_2', 0.0)
    for k in [2, 3, 5]:
        jk = features.get('J_%d' % k, 0)
        features['mix_er2_J%d' % k] = er2 * jk

    # --- Group 8: N-arithmetic controls (5 features) ---
    for k in [3, 5, 7, 8, 11]:
        features['N_mod_%d' % k] = N % k

    # --- Group 9: Random controls (5 features) ---
    rng = np.random.RandomState(N % (2**31))
    for i in range(5):
        features['random_%d' % i] = rng.random()

    return features


def compute_targets(N, p, q):
    """Compute target variables (requires knowledge of factors).

    Returns OrderedDict mapping target_name -> float.
    """
    targets = OrderedDict()

    for D in DISCRIMINANTS:
        chi_p = int(kronecker_symbol(D, int(p)))
        chi_q = int(kronecker_symbol(D, int(q)))
        targets['S_%d' % D] = chi_p + chi_q

    targets['p_normalized'] = float(p) / float(isqrt(N))
    targets['log_ratio'] = float(np.log(float(q) / float(p)))
    targets['sum_normalized'] = float(p + q) / (2.0 * float(isqrt(N)))

    return targets


# ============================================================
# Statistical analysis
# ============================================================

def ridge_loocv(X, y, lambdas=None):
    """Ridge regression with LOOCV using hat matrix formula.

    Returns (best_r2, best_lambda, coefficients_in_original_scale).
    Uses the identity: LOOCV residual_i = (y_i - yhat_i) / (1 - h_ii).
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 50)

    n, p = X.shape

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean

    X_std = Xc.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = Xc / X_std

    U, s, Vt = svd(Xs, full_matrices=False)

    ss_tot = np.sum(yc**2)
    if ss_tot < 1e-15:
        return 0.0, 0.0, np.zeros(p)

    best_r2 = -np.inf
    best_lambda = lambdas[0]
    best_coef = None

    for lam in lambdas:
        d = s**2 / (s**2 + lam)
        H_diag = np.sum(U**2 * d[np.newaxis, :], axis=1)

        y_hat_c = U @ (d * (U.T @ yc))
        resid = yc - y_hat_c
        denom = 1.0 - H_diag
        denom[np.abs(denom) < 1e-10] = 1e-10
        loocv_resid = resid / denom
        loocv_mse = np.mean(loocv_resid**2)

        r2 = 1.0 - (loocv_mse * n) / ss_tot

        if r2 > best_r2:
            best_r2 = r2
            best_lambda = lam
            beta_s = Vt.T @ ((d / (s + 1e-15)) * (U.T @ yc))
            best_coef = beta_s / X_std

    return float(best_r2), float(best_lambda), best_coef


def permutation_test(X, y, n_perm=200, lam=None):
    """Permutation test for ridge regression R^2_CV.

    Shuffles y `n_perm` times, computes R^2_CV each time with fixed lambda.
    Returns (p_value, null_distribution).
    """
    if lam is None:
        _, lam, _ = ridge_loocv(X, y)

    real_r2, _, _ = ridge_loocv(X, y, lambdas=[lam])

    n, p = X.shape
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    X_std = Xc.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = Xc / X_std

    U, s, _ = svd(Xs, full_matrices=False)
    d = s**2 / (s**2 + lam)
    H_diag = np.sum(U**2 * d[np.newaxis, :], axis=1)
    denom = 1.0 - H_diag
    denom[np.abs(denom) < 1e-10] = 1e-10

    null_r2s = np.zeros(n_perm)
    rng = np.random.RandomState(123)

    for i in range(n_perm):
        y_perm = rng.permutation(y)
        yc_perm = y_perm - y_perm.mean()
        ss_tot_perm = np.sum(yc_perm**2)
        if ss_tot_perm < 1e-15:
            null_r2s[i] = 0.0
            continue

        y_hat_c = U @ (d * (U.T @ yc_perm))
        resid = yc_perm - y_hat_c
        loocv_resid = resid / denom
        loocv_mse = np.mean(loocv_resid**2)
        null_r2s[i] = 1.0 - (loocv_mse * n) / ss_tot_perm

    p_value = float(np.mean(null_r2s >= real_r2))
    return p_value, null_r2s


def feature_correlations(X, y, feature_names):
    """Pearson correlation of each feature with target."""
    correlations = {}
    y_std = np.std(y)
    if y_std < 1e-15:
        return {name: 0.0 for name in feature_names}

    for i, name in enumerate(feature_names):
        x = X[:, i]
        if np.std(x) < 1e-15:
            correlations[name] = 0.0
        else:
            correlations[name] = float(np.corrcoef(x, y)[0, 1])

    return correlations


def anova_f_stat(X, y_cat, feature_names):
    """One-way ANOVA F-statistic for each feature vs categorical target."""
    categories = np.unique(y_cat)
    n = len(y_cat)
    k = len(categories)

    f_stats = {}
    for i, name in enumerate(feature_names):
        x = X[:, i]
        grand_mean = np.mean(x)

        ss_between = 0.0
        ss_within = 0.0
        for cat in categories:
            mask = y_cat == cat
            n_cat = np.sum(mask)
            if n_cat == 0:
                continue
            cat_mean = np.mean(x[mask])
            ss_between += n_cat * (cat_mean - grand_mean)**2
            ss_within += np.sum((x[mask] - cat_mean)**2)

        if ss_within < 1e-15 or k <= 1 or n <= k:
            f_stats[name] = 0.0
        else:
            f_stat = (ss_between / (k - 1)) / (ss_within / (n - k))
            f_stats[name] = float(f_stat)

    return f_stats


# ============================================================
# Main experiment
# ============================================================

def run_experiment(bit_sizes, count_per_size, output_dir='../data'):
    """Run the full E11 experiment."""
    print("=" * 70)
    print("E11: Comprehensive Poly(log N) Feature Extraction")
    print("=" * 70)
    print("Bit sizes: %s" % bit_sizes)
    print("Semiprimes per size: %d" % count_per_size)
    print()

    # --- Generate semiprimes ---
    print("Generating semiprimes...")
    all_semiprimes = []
    for bits in bit_sizes:
        sp = generate_semiprimes_fixed_bits(bits, count_per_size)
        print("  %d-bit: %d semiprimes (p/q ratio: %.3f to %.3f)" % (
            bits, len(sp),
            min(s[1]/float(s[2]) for s in sp) if sp else 0,
            max(s[1]/float(s[2]) for s in sp) if sp else 0))
        all_semiprimes.extend(sp)

    n_total = len(all_semiprimes)
    print("Total: %d semiprimes" % n_total)
    print()

    # --- Compute features and targets ---
    print("Computing features and targets...")
    t0 = time.time()

    feature_rows = []
    target_rows = []
    metadata_rows = []
    pollard_found_count = 0

    for idx, (N, p, q) in enumerate(all_semiprimes):
        if idx % 100 == 0 and idx > 0:
            print("  Processing %d/%d..." % (idx, n_total))

        feats = compute_features(N)
        targs = compute_targets(N, p, q)

        feature_rows.append(feats)
        target_rows.append(targs)
        metadata_rows.append({
            'N': N, 'p': p, 'q': q,
            'bits': N.bit_length(),
            'ratio': float(p) / float(q)
        })

        for g in [2, 3, 5]:
            if feats.get('pollard_%d_found' % g, 0) == 1:
                pollard_found_count += 1

    elapsed = time.time() - t0
    print("Feature computation: %.1fs" % elapsed)
    print("Pollard p-1 found factors: %d times (= trial division equivalent)" % pollard_found_count)
    print()

    # --- Build matrices ---
    feature_names = list(feature_rows[0].keys())
    target_names = list(target_rows[0].keys())

    n = len(feature_rows)
    d = len(feature_names)

    X = np.zeros((n, d))
    for i, row in enumerate(feature_rows):
        for j, name in enumerate(feature_names):
            X[i, j] = float(row[name])

    Y = {}
    for tname in target_names:
        Y[tname] = np.array([float(row[tname]) for row in target_rows])

    print("Feature matrix: %d x %d" % (n, d))
    print("Condition number: %.1f" % np.linalg.cond(X - X.mean(axis=0)))
    print("Targets: %s" % target_names)
    print()

    # --- Analysis ---
    results = {
        'config': {
            'bit_sizes': bit_sizes,
            'count_per_size': count_per_size,
            'n_semiprimes': n,
            'n_features': d,
            'feature_names': feature_names,
            'target_names': target_names,
            'pollard_found': pollard_found_count,
            'compute_time_s': float(elapsed),
        },
        'targets': {}
    }

    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    for tname in target_names:
        y = Y[tname]
        print()
        print("--- Target: %s ---" % tname)
        print("  Range: [%.4f, %.4f], Mean: %.4f, Std: %.4f" % (
            y.min(), y.max(), y.mean(), y.std()))

        # Ridge regression LOOCV
        r2, lam, coef = ridge_loocv(X, y)
        print("  Ridge R^2_CV = %.6f (lambda = %.4f)" % (r2, lam))

        # Permutation test
        p_val, null_dist = permutation_test(X, y, n_perm=int(200), lam=lam)
        null_mean = float(np.mean(null_dist))
        null_std = float(np.std(null_dist))
        print("  Permutation test: p-value = %.4f (null: %.4f +/- %.4f)" % (
            p_val, null_mean, null_std))

        # Per-feature correlations
        corrs = feature_correlations(X, y, feature_names)
        top_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        print("  Top 10 correlated features:")
        for fname, corr in top_corrs:
            print("    %-25s: r = %+.4f" % (fname, corr))

        # ANOVA for categorical hinge scalars
        anova = {}
        if tname.startswith('S_'):
            anova = anova_f_stat(X, y.astype(int), feature_names)
            top_anova = sorted(anova.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top ANOVA F-statistics:")
            for fname, f_val in top_anova:
                print("    %-25s: F = %.4f" % (fname, f_val))

        # Feature group mean |correlation|
        group_corrs = {}
        for fname, corr in corrs.items():
            parts = fname.split('_')
            if parts[0] in ('J', 'mix', 'N', 'carry', 'cf', 'modexp', 'euler', 'pollard', 'random'):
                group = parts[0]
            else:
                group = parts[0]
            if group not in group_corrs:
                group_corrs[group] = []
            group_corrs[group].append(abs(corr))

        group_means = {g: float(np.mean(v)) for g, v in group_corrs.items()}
        print("  Mean |r| by feature group:")
        for g in sorted(group_means, key=group_means.get, reverse=True):
            print("    %-15s: %.4f (n=%d)" % (g, group_means[g], len(group_corrs[g])))

        # Save results
        target_result = {
            'r2_cv': r2,
            'best_lambda': lam,
            'perm_p_value': p_val,
            'perm_null_mean': null_mean,
            'perm_null_std': null_std,
            'mean': float(y.mean()),
            'std': float(y.std()),
            'top_correlations': [(f, float(c)) for f, c in top_corrs],
            'all_correlations': {f: float(c) for f, c in corrs.items()},
            'group_mean_abs_corr': group_means,
        }
        if anova:
            top_anova_list = sorted(anova.items(), key=lambda x: x[1], reverse=True)[:10]
            target_result['top_anova'] = [(f, float(v)) for f, v in top_anova_list]

        results['targets'][tname] = target_result

    # --- Per-bit-size analysis ---
    print()
    print("=" * 70)
    print("PER-BIT-SIZE RIDGE R^2_CV")
    print("=" * 70)

    bit_results = {}
    for bits in bit_sizes:
        mask = np.array([m['bits'] == bits for m in metadata_rows])
        n_sub = int(mask.sum())
        if n_sub < 20:
            print("  %d-bit: skipped (n=%d < 20)" % (bits, n_sub))
            continue
        X_sub = X[mask]

        bit_result = {'n': n_sub}
        for tname in ['S_-3', 'S_-4', 'S_5', 'p_normalized']:
            y_sub = Y[tname][mask]
            r2_sub, _, _ = ridge_loocv(X_sub, y_sub)
            bit_result['r2_cv_%s' % tname] = float(r2_sub)

        print("  %d-bit (n=%d): S_{-3} R^2=%.4f, S_{-4} R^2=%.4f, "
              "S_5 R^2=%.4f, p_norm R^2=%.4f" % (
                  bits, n_sub,
                  bit_result['r2_cv_S_-3'], bit_result['r2_cv_S_-4'],
                  bit_result['r2_cv_S_5'], bit_result['r2_cv_p_normalized']))
        bit_results[int(bits)] = bit_result

    results['per_bit_size'] = bit_results

    # --- Exclude Pollard-found semiprimes and re-test ---
    pollard_mask = np.array([
        all(fr.get('pollard_%d_found' % g, 0) == 0 for g in [2, 3, 5])
        for fr in feature_rows
    ])
    n_clean = int(pollard_mask.sum())
    print()
    print("--- Excluding Pollard-found semiprimes: %d -> %d ---" % (n, n_clean))

    if n_clean >= 50:
        X_clean = X[pollard_mask]
        clean_results = {}
        for tname in ['S_-3', 'S_-4', 'S_5']:
            y_clean = Y[tname][pollard_mask]
            r2_clean, lam_clean, _ = ridge_loocv(X_clean, y_clean)
            p_val_clean, _ = permutation_test(
                X_clean, y_clean, n_perm=int(200), lam=lam_clean)
            print("  %s: R^2_CV = %.6f, p-value = %.4f" % (
                tname, r2_clean, p_val_clean))
            clean_results[tname] = {
                'r2_cv': float(r2_clean),
                'perm_p_value': float(p_val_clean),
            }
        results['clean_analysis'] = {
            'n_semiprimes': n_clean,
            'targets': clean_results,
        }

    # --- Overall verdict ---
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    hinge_r2s = []
    hinge_pvals = []
    for D in DISCRIMINANTS:
        tname = 'S_%d' % D
        r2 = results['targets'][tname]['r2_cv']
        pv = results['targets'][tname]['perm_p_value']
        hinge_r2s.append(r2)
        hinge_pvals.append(pv)
        status = "SIGNAL" if (r2 > 0.05 and pv < 0.05) else "FLAT"
        print("  S_{%3d}(N): R^2_CV = %+.6f, p-value = %.3f  [%s]" % (D, r2, pv, status))

    max_r2 = max(hinge_r2s)
    min_pval = min(hinge_pvals)

    if max_r2 > 0.05 and min_pval < 0.01:
        verdict = "SIGNAL DETECTED — extend to larger N"
    elif max_r2 > 0.01 and min_pval < 0.05:
        verdict = "WEAK SIGNAL — needs investigation"
    else:
        verdict = "BARRIER CONFIRMED — no feature combination predicts hinge scalars"

    print()
    print("  Max hinge R^2_CV: %.6f" % max_r2)
    print("  Min hinge p-value: %.4f" % min_pval)
    print("  VERDICT: %s" % verdict)

    # Random control sanity check
    random_corrs = [
        abs(results['targets']['S_-3']['all_correlations'].get('random_%d' % i, 0))
        for i in range(5)]
    real_corrs = [
        abs(c) for f, c in results['targets']['S_-3']['all_correlations'].items()
        if not f.startswith('random')
    ]
    print()
    print("  Sanity check:")
    print("    Random control mean |r|: %.4f" % np.mean(random_corrs))
    print("    Real feature mean |r|:   %.4f" % np.mean(real_corrs))
    print("    (Should be comparable if barrier holds)")

    results['verdict'] = verdict
    results['max_hinge_r2'] = float(max_r2)
    results['min_hinge_pval'] = float(min_pval)

    # --- Save JSON ---
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'E11_feature_extraction_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=int(2), cls=SageEncoder)
    print()
    print("Results saved to %s" % json_path)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: Top feature correlations for each hinge scalar
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('E11: Feature Correlations with Hinge Scalars S_D(N)', fontsize=14)

        for idx, D in enumerate(DISCRIMINANTS):
            ax = axes[idx // 3, idx % 3]
            tname = 'S_%d' % D
            corrs = results['targets'][tname]['all_correlations']

            top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            names = [t[0] for t in top]
            vals = [t[1] for t in top]
            colors = ['red' if abs(v) > 0.1 else 'steelblue' for v in vals]

            ax.barh(range(len(names)), vals, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel('Pearson r')
            r2 = results['targets'][tname]['r2_cv']
            pv = results['targets'][tname]['perm_p_value']
            ax.set_title('S_{%d}(N)  R^2=%.4f  p=%.3f' % (D, r2, pv))
            ax.axvline(0, color='black', linewidth=0.5)
            ax.invert_yaxis()

        # Bottom-right: R^2 by bit size
        ax = axes[1, 2]
        if bit_results:
            bits_list = sorted(bit_results.keys())
            for tname, marker in [('S_-3', 'ko-'), ('S_5', 'bs-'), ('p_normalized', 'r^-')]:
                r2_key = 'r2_cv_%s' % tname
                r2_list = [bit_results[b].get(r2_key, 0) for b in bits_list]
                ax.plot(bits_list, r2_list, marker, markersize=6, label=tname)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Bit size')
            ax.set_ylabel('R^2_CV')
            ax.set_title('Ridge R^2_CV vs N bit size')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'E11_feature_correlations.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("Plot saved to %s" % plot_path)

        # Plot 2: Feature group comparison (box plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        groups_all = {}
        for tname in ['S_-3', 'S_-4', 'S_5']:
            corrs = results['targets'][tname]['all_correlations']
            for fname, c in corrs.items():
                parts = fname.split('_')
                group = parts[0]
                if group not in groups_all:
                    groups_all[group] = []
                groups_all[group].append(abs(c))

        group_names = sorted(groups_all.keys(),
                             key=lambda g: np.median(groups_all[g]), reverse=True)
        group_vals = [groups_all[g] for g in group_names]

        bp = ax.boxplot(group_vals, labels=group_names, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('|Pearson r| with hinge scalars (3 targets pooled)')
        ax.set_title('E11: Feature Group Discriminative Power')
        ax.set_xticklabels(group_names, rotation=45, ha='right')

        plt.tight_layout()
        plot_path2 = os.path.join(output_dir, 'E11_feature_groups.png')
        plt.savefig(plot_path2, dpi=int(150))
        plt.close()
        print("Plot saved to %s" % plot_path2)

        # Plot 3: Permutation test null distribution for S_{-3}
        fig, ax = plt.subplots(figsize=(8, 5))
        tname = 'S_-3'
        real_r2 = results['targets'][tname]['r2_cv']
        _, null_dist = permutation_test(X, Y[tname], n_perm=int(200),
                                        lam=results['targets'][tname]['best_lambda'])
        ax.hist(null_dist, bins=int(30), color='steelblue', alpha=0.7,
                edgecolor='black', label='Null (permuted)')
        ax.axvline(real_r2, color='red', linewidth=2, linestyle='--',
                   label='Observed R^2=%.4f' % real_r2)
        ax.set_xlabel('R^2_CV')
        ax.set_ylabel('Count')
        ax.set_title('E11: Permutation Test for S_{-3}(N)')
        ax.legend()

        plt.tight_layout()
        plot_path3 = os.path.join(output_dir, 'E11_permutation_test.png')
        plt.savefig(plot_path3, dpi=int(150))
        plt.close()
        print("Plot saved to %s" % plot_path3)

    except ImportError:
        print("matplotlib not available, skipping plots")

    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    # Balanced semiprimes only (p/q >= 0.3) to exclude trial-division regime.
    # Bit sizes 16-24 ensure n_samples >> n_features (111) at each size.
    results = run_experiment(
        bit_sizes=[16, 18, 20, 22],
        count_per_size=int(150),
        output_dir='../data'
    )
