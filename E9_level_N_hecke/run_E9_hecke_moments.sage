#!/usr/bin/env sage
"""
E9: Factor Leakage from Good-Prime Hecke Statistics at Level N

Tests whether the factorization of N=pq leaks into good-prime Hecke
eigenvalue statistics of S_2(Gamma_0(N)), using ONLY Hecke operators
T_ell for ell coprime to N.

Access model:
  - N enters through the level Gamma_0(N), NOT as a character twist
  - Bad primes shape the space itself, not by "omitting" Euler factors
  - No Atkin-Lehner operators, no divisor enumeration, no chi_N

Why this is different from E8:
  - E8a/E8b used chi_N twist => chi_N(p)=0 at p|N => local factor omitted
  - E9 uses level N => bad primes affect the space structure (old/new
    decomposition, ramification type) in ways not reducible to gcd

Theory (Eichler-Selberg decomposition):
  Tr(T_ell | S_2(Gamma_0(pq))) = f(ell) + g(ell,p) + g(ell,q) + h(ell,N) + C(N)
  where g(ell,p) = -1/2 Sum_t h(D_t) chi_{D_t}(p) are FACTOR-SPECIFIC
  class-number-weighted character sums. These are SUMS not products --
  structurally different from the E7c CRT product obstruction.

Tests:
  A. Raw Hecke moments -> predict log(min(p,q))
  B. Residualized (remove dim, log N) -> predict log(min(p,q))
  C. Baseline: dim + log(N) alone
  D. Null distribution (permutation)
  E. Coarse classification: is min(p,q) <= median?
  F. Per-prime marginal R^2
  G. R^2 vs number of good primes (amplification test -- KEY metric)
  H. Per-prime-power Hecke products T_{ell^2} for additional features
"""

import numpy as np
import json
import time
import os
from collections import OrderedDict

# ============================================================
# Configuration
# ============================================================
MAX_N = 5000
MIN_N = 200
WEIGHT = 2
GOOD_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
MAX_MOMENT = 3   # Tr(T_ell), Tr(T_ell^2), Tr(T_ell^3)
MIN_BALANCE = 0.15  # p/q >= 0.15

# ============================================================
# Step 1: Generate semiprimes
# ============================================================
def generate_semiprimes(min_N, max_N, min_balance):
    """Generate semiprimes N=pq with p <= q, within bounds."""
    semiprimes = []
    max_p = int(max_N**0.5) + 1
    p = ZZ(2)
    while p <= max_p:
        q = p
        while True:
            q = next_prime(q)
            N_val = p * q
            if N_val > max_N:
                break
            if N_val >= min_N:
                ratio = float(p) / float(q)
                if ratio >= min_balance:
                    semiprimes.append((int(N_val), int(p), int(q)))
        p = next_prime(p)
    semiprimes.sort()
    return semiprimes


# ============================================================
# Step 2: Compute Hecke moment sketch
# ============================================================
def compute_hecke_sketch(N_val, k, primes_list, max_mom):
    """
    Compute {Tr(T_ell^j) : ell in primes, j=1..max_mom}
    for S_k(Gamma_0(N)).

    Uses modular symbols -- does NOT factor N or use Atkin-Lehner.
    """
    S = CuspForms(N_val, k)
    d = S.dimension()
    if d == 0:
        return None

    sketch = OrderedDict()
    for ell in primes_list:
        entry = {}
        if int(N_val) % ell == 0:
            # Bad prime: mark as None
            for j in range(1, max_mom + 1):
                entry[j] = None
        else:
            M = S.hecke_matrix(ell)
            Mpow = M
            for j in range(1, max_mom + 1):
                if j > 1:
                    Mpow = Mpow * M
                entry[j] = int(Mpow.trace())
        sketch[ell] = entry

    return {'dim': int(d), 'sketch': sketch}


# ============================================================
# Step 3: Build feature matrix and targets
# ============================================================
def build_features(results, primes_list, max_mom):
    """
    Build feature matrix X and target vector y.

    Features: Tr(T_ell^j) / dim for each (ell, j).
    Normalization by dim removes the trivial "more eigenvalues = larger trace"
    scaling, giving the j-th moment of the eigenvalue distribution.
    """
    Ns = []
    ps = []
    qs = []
    X_rows = []
    y = []
    dims = []

    for N_val, p, q, sketch_data in results:
        if sketch_data is None:
            continue
        d = sketch_data['dim']
        if d == 0:
            continue
        sk = sketch_data['sketch']

        row = []
        for ell in primes_list:
            entry = sk.get(ell, {})
            for j in range(1, max_mom + 1):
                val = entry.get(j, None)
                if val is None:
                    # NaN for bad-prime or missing data; avoids silently
                    # treating absent eigenvalues as zero signal
                    row.append(float('nan'))
                else:
                    # j-th moment of eigenvalue distribution
                    row.append(float(val) / float(d))
                    # Note: dividing by dim (not dim^j) gives the mean of a_i^j
        X_rows.append(row)
        y.append(np.log(float(min(p, q))))
        Ns.append(N_val)
        ps.append(p)
        qs.append(q)
        dims.append(d)

    X = np.array(X_rows)
    y = np.array(y)
    return Ns, ps, qs, X, y, np.array(dims, dtype=float)


# ============================================================
# Step 4: Ridge LOOCV
# ============================================================
def ridge_loocv_r2(X, y, lambdas=None):
    """Ridge regression with LOOCV using closed-form hat matrix."""
    n, p = X.shape
    if p == 0 or n < 5:
        return -999.0, -999.0

    # z-score features
    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0)
    sd_x[sd_x < 1e-12] = 1.0
    Xz = (X - mu_x) / sd_x

    mu_y = y.mean()
    yc = y - mu_y

    if lambdas is None:
        lambdas = np.logspace(-2, 4, 30)

    best_r2 = -1e9
    best_lam = None

    for lam in lambdas:
        A = Xz.T @ Xz + lam * np.eye(p)
        try:
            H = Xz @ np.linalg.solve(A, Xz.T)
        except np.linalg.LinAlgError:
            continue
        y_hat = H @ yc
        resid = yc - y_hat
        h_ii = np.diag(H)
        denom = 1.0 - h_ii
        denom[np.abs(denom) < 1e-10] = 1e-10
        cv_resid = resid / denom
        ss_cv = np.sum(cv_resid ** 2)
        ss_tot = np.sum(yc ** 2)
        r2 = 1.0 - ss_cv / ss_tot if ss_tot > 0 else -999.0
        if r2 > best_r2:
            best_r2 = r2
            best_lam = lam

    return best_r2, best_lam


# ============================================================
# Step 5: Null distribution
# ============================================================
def null_distribution(X, y, n_perm=500, seed=42):
    """Permutation null for LOOCV R^2."""
    rng = np.random.default_rng(seed)
    r2_null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r2, _ = ridge_loocv_r2(X, y_perm)
        r2_null.append(r2)
    return np.array(r2_null)


# ============================================================
# Step 6: Residualize features
# ============================================================
def residualize(X, y, nuisance):
    """
    Project out nuisance regressors from both X and y.
    Returns X_resid, y_resid.
    """
    Q, _ = np.linalg.qr(nuisance)
    X_resid = X - Q @ (Q.T @ X)
    y_resid = y - Q @ (Q.T @ y)
    return X_resid, y_resid


# ============================================================
# Main
# ============================================================
def main():
    print("E9 -- Factor Leakage from Good-Prime Hecke Statistics at Level N")
    print("=" * 70)
    print("Access model: N in Gamma_0(N), no Atkin-Lehner, no chi_N twist")
    print("Features: Tr(T_ell^j)/dim for good primes ell, moments j=1..%d" % MAX_MOMENT)
    print()

    # ----------------------------------------------------------
    # Step 1: Generate semiprimes
    # ----------------------------------------------------------
    semiprimes = generate_semiprimes(MIN_N, MAX_N, MIN_BALANCE)
    print("Semiprimes: %d  [%d..%d]" % (len(semiprimes), semiprimes[0][0], semiprimes[-1][0]))

    # Show balance distribution
    ratios = [float(p)/float(q) for _, p, q in semiprimes]
    print("  balance p/q: min=%.3f  median=%.3f  max=%.3f" % (
        min(ratios), sorted(ratios)[len(ratios)//2], max(ratios)))
    log_ps = [np.log(float(min(p,q))) for _, p, q in semiprimes]
    print("  log(min factor): min=%.2f  max=%.2f  range=%.2f" % (
        min(log_ps), max(log_ps), max(log_ps)-min(log_ps)))

    # ----------------------------------------------------------
    # Step 2: Compute Hecke moment sketches
    # ----------------------------------------------------------
    print("\nComputing Hecke moment sketches (k=%d, %d primes, %d moments):" % (
        WEIGHT, len(GOOD_PRIMES), MAX_MOMENT))
    results = []
    t_total = time.time()
    for i, (N_val, p, q) in enumerate(semiprimes):
        t0 = time.time()
        sketch = compute_hecke_sketch(N_val, WEIGHT, GOOD_PRIMES, MAX_MOMENT)
        dt = time.time() - t0
        results.append((N_val, p, q, sketch))
        d = sketch['dim'] if sketch else 0
        if i < 5 or i % 50 == 0 or i == len(semiprimes) - 1:
            print("  [%3d/%3d] N=%5d (%d x %d): dim=%4d, time=%.3fs" % (
                i+1, len(semiprimes), N_val, p, q, d, dt))
    print("  Total computation time: %.1fs" % (time.time() - t_total))

    # ----------------------------------------------------------
    # Step 3: Build features
    # ----------------------------------------------------------
    Ns, ps_list, qs_list, X, y, dims = build_features(
        results, GOOD_PRIMES, MAX_MOMENT)
    n_samples, n_features = X.shape
    print("\nFeature matrix: %d samples x %d features" % (n_samples, n_features))
    print("Target: log(min(p,q)), range [%.2f, %.2f]" % (y.min(), y.max()))

    # Feature labels
    feature_labels = []
    for ell in GOOD_PRIMES:
        for j in range(1, MAX_MOMENT + 1):
            feature_labels.append("m%d(T_%d)" % (j, ell))

    # ----------------------------------------------------------
    # Build nuisance matrix: intercept, log(N), dim
    # ----------------------------------------------------------
    log_N = np.log(np.array(Ns, dtype=float))
    Z = np.column_stack([np.ones(n_samples), log_N, dims])

    # ----------------------------------------------------------
    # Test A: Raw features -> log(min(p,q))
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test A: Raw Hecke moments -> factor prediction")
    print("=" * 70)
    r2_raw, lam_raw = ridge_loocv_r2(X, y)
    print("  LOOCV R^2 = %.6f  (lambda=%.4f)" % (r2_raw, lam_raw))

    # ----------------------------------------------------------
    # Test B: Residualized features
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test B: Residualized (remove intercept, log N, dim)")
    print("=" * 70)
    X_resid, y_resid = residualize(X, y, Z)
    r2_resid, lam_resid = ridge_loocv_r2(X_resid, y_resid)
    print("  LOOCV R^2 = %.6f  (lambda=%.4f)" % (r2_resid, lam_resid))

    # ----------------------------------------------------------
    # Test C: Baseline (dim + log N only)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test C: dim + log(N) only (baseline)")
    print("=" * 70)
    r2_baseline, lam_baseline = ridge_loocv_r2(Z[:, 1:], y)
    print("  LOOCV R^2 = %.6f  (lambda=%.4f)" % (r2_baseline, lam_baseline))

    # Also test dim alone
    r2_dim_only, _ = ridge_loocv_r2(dims.reshape(-1, 1), y)
    print("  dim only:   R^2 = %.6f" % r2_dim_only)
    r2_logN_only, _ = ridge_loocv_r2(log_N.reshape(-1, 1), y)
    print("  log(N) only: R^2 = %.6f" % r2_logN_only)

    # ----------------------------------------------------------
    # Test D: Null distribution for residualized test
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test D: Null distribution (500 permutations, residualized)")
    print("=" * 70)
    t0 = time.time()
    null_r2 = null_distribution(X_resid, y_resid, n_perm=500)
    print("  Computation time: %.1fs" % (time.time() - t0))
    p50 = float(np.median(null_r2))
    p95 = float(np.percentile(null_r2, 95))
    p99 = float(np.percentile(null_r2, 99))
    print("  Null 50th: %.6f" % p50)
    print("  Null 95th: %.6f" % p95)
    print("  Null 99th: %.6f" % p99)
    print("  Observed:  %.6f" % r2_resid)
    print("  Significant at 95%%? %s" % ("YES" if r2_resid > p95 else "NO"))
    print("  Significant at 99%%? %s" % ("YES" if r2_resid > p99 else "NO"))

    # ----------------------------------------------------------
    # Test E: Coarse classification
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test E: Coarse classification (is min(p,q) <= median?)")
    print("=" * 70)
    median_p = float(np.median(np.exp(y)))
    y_binary = (np.exp(y) <= median_p).astype(float)
    y_binary_resid = y_binary - Z @ np.linalg.lstsq(Z, y_binary, rcond=None)[0]
    r2_bin_raw, _ = ridge_loocv_r2(X, y_binary)
    r2_bin_resid, _ = ridge_loocv_r2(X_resid, y_binary_resid)
    print("  median(min factor) = %.1f" % median_p)
    print("  Raw R^2      = %.6f" % r2_bin_raw)
    print("  Residualized = %.6f" % r2_bin_resid)

    # ----------------------------------------------------------
    # Test F: Per-prime marginal R^2
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test F: Per-prime marginal R^2 (residualized)")
    print("=" * 70)
    per_prime_r2 = {}
    for i, label in enumerate(feature_labels):
        xi = X_resid[:, i:i+1]
        r2_i, _ = ridge_loocv_r2(xi, y_resid)
        per_prime_r2[label] = float(r2_i)
        print("  %15s: R^2 = %+.6f" % (label, r2_i))

    # ----------------------------------------------------------
    # Test G: R^2 vs K (amplification test — KEY)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test G: R^2 vs number of good primes (amplification test)")
    print("=" * 70)
    r2_vs_K = []
    # For each K, use all moments from the first K available primes
    available = [ell for ell in GOOD_PRIMES]
    for K in range(1, len(available) + 1):
        cols = []
        for j_ell in range(K):
            for j in range(MAX_MOMENT):
                cols.append(j_ell * MAX_MOMENT + j)
        X_k = X_resid[:, cols]
        r2_k, lam_k = ridge_loocv_r2(X_k, y_resid)
        r2_vs_K.append(float(r2_k))
        print("  K=%2d primes (up to %2d): R^2 = %+.6f" % (
            K, available[K-1], r2_k))

    # ----------------------------------------------------------
    # Test H: Moments only (m1 vs m2 vs m3)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test H: Which moment order carries most signal?")
    print("=" * 70)
    for j in range(1, MAX_MOMENT + 1):
        # Select columns for moment j across all primes
        cols_j = [ell_idx * MAX_MOMENT + (j - 1) for ell_idx in range(len(GOOD_PRIMES))]
        X_j = X_resid[:, cols_j]
        r2_j, _ = ridge_loocv_r2(X_j, y_resid)
        print("  Moment m_%d only (all primes): R^2 = %+.6f" % (j, r2_j))

    # ----------------------------------------------------------
    # Test I: Correlations between traces and factor properties
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test I: Top individual Pearson correlations with log(min(p,q))")
    print("=" * 70)
    correlations = []
    for i, label in enumerate(feature_labels):
        xi = X_resid[:, i]
        if np.std(xi) < 1e-12:
            continue
        r = float(np.corrcoef(xi, y_resid)[0, 1])
        correlations.append((label, r, abs(r)))
    correlations.sort(key=lambda x: -x[2])
    for label, r, _ in correlations[:15]:
        print("  %15s: r = %+.4f" % (label, r))

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  Semiprimes:      %d  (N in [%d, %d])" % (n_samples, min(Ns), max(Ns)))
    print("  Features:        %d  (%d primes x %d moments)" % (
        n_features, len(GOOD_PRIMES), MAX_MOMENT))
    print("  Baseline R^2:    %.6f  (dim + log N)" % r2_baseline)
    print("  Raw R^2:         %.6f" % r2_raw)
    print("  Residualized R^2:%.6f" % r2_resid)
    print("  Null 95th:       %.6f" % p95)
    print("  Significant:     %s" % ("YES" if r2_resid > p95 else "NO"))
    print()

    # Amplification direction
    first_half = np.mean(r2_vs_K[:len(r2_vs_K)//2])
    second_half = np.mean(r2_vs_K[len(r2_vs_K)//2:])
    print("  Amplification:")
    print("    First half  (K=1..%d):  mean R^2 = %.6f" % (
        len(r2_vs_K)//2, first_half))
    print("    Second half (K=%d..%d): mean R^2 = %.6f" % (
        len(r2_vs_K)//2 + 1, len(r2_vs_K), second_half))
    if second_half > first_half + 0.01:
        print("    Direction: INCREASING (possible signal)")
    elif second_half < first_half - 0.01:
        print("    Direction: DECREASING (overfitting)")
    else:
        print("    Direction: FLAT (no amplification)")

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    output = {
        'n_semiprimes': n_samples,
        'N_range': [int(min(Ns)), int(max(Ns))],
        'weight': WEIGHT,
        'good_primes': GOOD_PRIMES,
        'max_moment': MAX_MOMENT,
        'r2_raw': float(r2_raw),
        'r2_residualized': float(r2_resid),
        'r2_baseline_dim_logN': float(r2_baseline),
        'r2_dim_only': float(r2_dim_only),
        'r2_logN_only': float(r2_logN_only),
        'null_50th': float(p50),
        'null_95th': float(p95),
        'null_99th': float(p99),
        'significant_95': bool(r2_resid > p95),
        'significant_99': bool(r2_resid > p99),
        'r2_binary_raw': float(r2_bin_raw),
        'r2_binary_resid': float(r2_bin_resid),
        'r2_vs_K': r2_vs_K,
        'per_prime_r2': per_prime_r2,
        'r2_by_moment': {},
        'top_correlations': [
            {'feature': label, 'r': float(r)}
            for label, r, _ in correlations[:20]
        ],
        'semiprimes': [
            {'N': int(N_val), 'p': int(p), 'q': int(q)}
            for N_val, p, q in zip(Ns, ps_list, qs_list)
        ],
    }

    # Moment-specific R^2
    for j in range(1, MAX_MOMENT + 1):
        cols_j = [ell_idx * MAX_MOMENT + (j - 1) for ell_idx in range(len(GOOD_PRIMES))]
        X_j = X_resid[:, cols_j]
        r2_j, _ = ridge_loocv_r2(X_j, y_resid)
        output['r2_by_moment']['m_%d' % j] = float(r2_j)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'E9_hecke_moments_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to %s" % output_path)


if __name__ == '__main__':
    main()
