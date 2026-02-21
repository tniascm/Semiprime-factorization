"""
E8b — Multi-Form L-function Amplification

GOAL: Test whether a FAMILY of twisted L-functions L(s, f_i x chi_N),
evaluated at multiple s-values, carries factor information that scales
with the number of forms K.

KEY QUESTION: Does mutual information about (p,q) increase with K?
  If yes -> amplification corridor exists (noisy-channel decoding).
  If flat -> same CRT collapse appears in every form.

DESIGN:
  Forms: All Hecke eigenforms in S_k(SL_2(Z)) for k = 12,16,...,36
         (~19 forms from level 1, increasing weight)
  Twist: chi_N = kronecker(., N) (computable without factoring)
  s-grid: critical line (s = k/2 + it) + slightly right (s = k/2 + 1 + it)
  Features: log|L|, z-scored per form across semiprimes
  Target: log(min(p,q))
  Primary metric: R^2 vs K (number of forms)

GUARDRAILS:
  - No hidden gcd: no feature tests chi_N(m) = 0 for specific m
  - No oracle: chi_N computed from N alone
  - Root number: epsilon = (-1)^{k/2} for ALL level-1 twists by quadratic
    chi_N (independent of N — carries zero factor information)
"""
import time
import sys
import json
import os

import numpy as np
from scipy import stats, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

set_random_seed(42)
np.random.seed(42)

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from sage_encoding import _py, _py_dict, safe_json_dump

# ─── Configuration ─────────────────────────────────────────────────────

WEIGHTS = [12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
T_CRIT = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
T_RIGHT = [0.0, 1.0, 3.0]
MIN_P, MAX_P = 50, 200
N_SEMIPRIMES = 60

# ─── Semiprime generation ──────────────────────────────────────────────

def gen_semiprimes(count, min_p, max_p):
    """Generate semiprimes with diverse factor ratios."""
    result = []
    seen = set()

    targets = np.logspace(np.log10(min_p), np.log10(max_p), count)
    for tgt in targets:
        p = next_prime(int(tgt))
        if int(p) > max_p:
            continue
        q = next_prime(p)
        N = int(p * q)
        if N not in seen:
            seen.add(N)
            result.append((N, int(p), int(q)))

    for tgt in targets[::2]:
        p = next_prime(int(tgt))
        if int(p) > max_p:
            continue
        q = next_prime(next_prime(p))
        if int(q) <= max_p + 50:
            N = int(p * q)
            if N not in seen:
                seen.add(N)
                result.append((N, int(p), int(q)))

    for tgt in targets[::4]:
        p = next_prime(int(tgt))
        if int(p) > max_p:
            continue
        q = next_prime(next_prime(next_prime(p)))
        if int(q) <= max_p + 80:
            N = int(p * q)
            if N not in seen:
                seen.add(N)
                result.append((N, int(p), int(q)))

    result.sort()
    return result[:count]

# ─── Modular form computation via Sage exact arithmetic ───────────────

def sage_series_to_numpy(ps, prec):
    """Convert a Sage power series to a numpy float array."""
    coeffs = np.zeros(prec, dtype=np.float64)
    cl = ps.padded_list(prec)
    for i in range(min(len(cl), prec)):
        coeffs[i] = float(cl[i])
    return coeffs


def compute_forms(weights, prec):
    """Compute all level-1 Hecke eigenforms using Sage exact arithmetic.

    Strategy:
    1. Compute E4, E6, Delta as exact Sage power series (over ZZ)
    2. Build basis of S_k = Delta * M_{k-12} using exact multiplication
    3. Convert to numpy for Hecke diagonalization and L-value evaluation
    """
    forms = []
    print("Computing modular forms (prec=%d):" % prec, flush=True)

    # Compute E4, E6, Delta over ZZ using Sage built-ins
    t0 = time.perf_counter()
    print("  E4, E6, Delta via Sage...", end="", flush=True)
    e4_sage = eisenstein_series_qexp(4, prec)
    e6_sage = eisenstein_series_qexp(6, prec)
    d_sage = delta_qexp(prec)
    dt = time.perf_counter() - t0
    print(" (%.1fs)" % dt, flush=True)

    # Verify Delta
    print("  Delta[1]=%s, Delta[2]=%s, Delta[3]=%s" %
          (d_sage[1], d_sage[2], d_sage[3]), flush=True)

    print("Computing eigenforms:", flush=True)

    for k in weights:
        sys.stdout.write("  Weight %d: " % k)
        sys.stdout.flush()
        t0 = time.perf_counter()

        j = k - 12
        # Monomials E4^a * E6^b with 4a + 6b = j
        monomials = []
        for b in range(j // 6 + 1):
            rem = j - 6 * b
            if rem >= 0 and rem % 4 == 0:
                a = rem // 4
                monomials.append((a, b))

        d = len(monomials)
        if d == 0:
            print("dim 0, skip", flush=True)
            continue

        # Build basis using exact Sage arithmetic, then convert to numpy
        B = np.zeros((d, prec), dtype=np.float64)
        for i, (a, b) in enumerate(monomials):
            monomial = d_sage * e4_sage**a * e6_sage**b
            B[i, :] = sage_series_to_numpy(monomial, prec)

        if d == 1:
            # Single eigenform
            coeffs = B[0, :].copy()
            norm = coeffs[1]
            if abs(norm) > 1e-10:
                coeffs /= norm
            forms.append({
                'weight': k, 'index': 0,
                'label': 'k%d_0' % k,
                'coeffs': coeffs,
                'epsilon': int((-1) ** (k // 2)),
            })
            dt = time.perf_counter() - t0
            print("1 form (%.2fs)" % dt, flush=True)
        else:
            # Multiple eigenforms: diagonalize T_2
            # T_2: (T_2 f)[n] = f[2n] + 2^{k-1} f[n/2]
            T2B = np.zeros((d, prec), dtype=np.float64)
            two_k1 = float(2) ** (k - 1)
            for i in range(d):
                for n in range(1, prec):
                    val = 0.0
                    if 2 * n < prec:
                        val += B[i, 2 * n]
                    if n % 2 == 0:
                        val += two_k1 * B[i, n // 2]
                    T2B[i, n] = val

            # Find T_2 matrix: T2B = M @ B (solve for M)
            n_use = min(6 * d, prec - 1)
            B_sub = B[:, 1:n_use + 1].T      # (n_use, d)
            T2B_sub = T2B[:, 1:n_use + 1].T  # (n_use, d)
            M_T2, _, _, _ = np.linalg.lstsq(B_sub, T2B_sub, rcond=None)
            M_T2 = M_T2.T  # (d, d)

            evals, evecs = np.linalg.eig(M_T2)

            for idx in range(d):
                v = evecs[:, idx].real
                coeffs = v @ B  # linear combination of basis rows
                norm = coeffs[1]
                if abs(norm) > 1e-10:
                    coeffs /= norm

                # Verify it's an eigenform: check a(2)*a(3) == a(6)
                if abs(coeffs[6] - coeffs[2] * coeffs[3]) > 1e-3 * max(abs(coeffs[6]), 1):
                    sys.stdout.write("[WARN: multiplicativity check failed] ")

                forms.append({
                    'weight': k, 'index': idx,
                    'label': 'k%d_%d' % (k, idx),
                    'coeffs': coeffs,
                    'epsilon': int((-1) ** (k // 2)),
                })

            dt = time.perf_counter() - t0
            print("%d forms (%.2fs)" % (d, dt), flush=True)

    return forms

# ─── chi_N computation (no oracle) ────────────────────────────────────

def compute_chi_N(N, max_n):
    """Compute chi_N(n) = kronecker(n, N) WITHOUT using factors."""
    chi = np.zeros(max_n + 1, dtype=np.float64)
    N_int = int(N)
    for n in range(1, max_n + 1):
        chi[n] = float(kronecker_symbol(n, N_int))
    return chi

# ─── L-value evaluation (generalized AFE) ─────────────────────────────

def eval_L_afe(s, coeffs, chi, N_val, k, epsilon, max_terms):
    """Evaluate L(s, f_k x chi_N) using approximate functional equation.

    Lambda(s) = (N/(2pi))^s * Gamma(s + (k-1)/2) * L(s)
    Lambda(s) = epsilon * Lambda(k - s)
    """
    s_c = complex(s)
    s_dual = complex(k) - s_c
    half_k1 = (k - 1.0) / 2.0

    mt = min(max_terms, len(coeffs) - 1, len(chi) - 1)
    n_arr = np.arange(1, mt + 1, dtype=np.float64)
    an_chi = coeffs[1:mt + 1] * chi[1:mt + 1]

    alpha = 1.0
    x = 2.0 * np.pi * n_arr / (float(N_val) * alpha)
    V = np.exp(-x * x)

    fwd = np.sum(an_chi * np.exp(-s_c * np.log(n_arr)) * V)

    log_gamma_ratio = (complex(special.loggamma(s_dual + half_k1)) -
                       complex(special.loggamma(s_c + half_k1)))
    log_N_factor = (float(k) - 2.0 * s_c) * np.log(float(N_val) / (2.0 * np.pi))
    eps_s = epsilon * np.exp(log_gamma_ratio + log_N_factor)

    dual = np.sum(an_chi * np.exp(-s_dual * np.log(n_arr)) * V)

    return complex(fwd + eps_s * dual)

# ─── Ridge regression with LOOCV ──────────────────────────────────────

def ridge_loocv_r2(X, y, lambdas=None):
    """Ridge regression with leave-one-out CV. Returns best R^2."""
    if lambdas is None:
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    n, p = X.shape
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot < 1e-30:
        return 0.0

    best_r2 = -np.inf
    for lam in lambdas:
        XtX_reg = X.T @ X + lam * np.eye(p)
        try:
            H = X @ np.linalg.solve(XtX_reg, X.T)
        except np.linalg.LinAlgError:
            continue
        y_hat = H @ y
        h_diag = np.diag(H)
        denom = np.where(np.abs(1.0 - h_diag) < 1e-10, 1e-10, 1.0 - h_diag)
        cv_resid = (y - y_hat) / denom
        ss_cv = np.sum(cv_resid ** 2)
        r2 = 1.0 - ss_cv / ss_tot
        if r2 > best_r2:
            best_r2 = r2

    return float(best_r2)


def pca_reduce(X, n_components):
    """PCA via SVD."""
    X_c = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    nc = min(n_components, U.shape[1])
    return U[:, :nc] * S[:nc]

# ─── Main experiment ──────────────────────────────────────────────────

def main():
    print("E8b -- Multi-Form L-function Amplification\n")
    print("=" * 78)

    # 1. Generate semiprimes
    semiprimes = gen_semiprimes(N_SEMIPRIMES, MIN_P, MAX_P)
    n_semi = len(semiprimes)
    max_N = max(N for N, p, q in semiprimes)
    print("Semiprimes: %d  [%d..%d]  max_N=%d" %
          (n_semi, semiprimes[0][0], semiprimes[-1][0], max_N), flush=True)

    # 2. Compute modular form basis and eigenforms
    # max_terms for AFE: need ~N/(2pi)*3 ~ 0.48*N terms with alpha=1.0
    prec = int(max_N * 0.55) + 100
    forms = compute_forms(WEIGHTS, prec)
    K = len(forms)
    print("\nTotal forms: %d" % K, flush=True)

    J = len(T_CRIT) + len(T_RIGHT)
    print("s-grid: %d points per form (%d critical + %d right)" %
          (J, len(T_CRIT), len(T_RIGHT)), flush=True)

    # 3. Compute L-value matrix
    print("\n" + "=" * 78)
    print("Computing L-values (%d semiprimes x %d forms x %d s-points = %d evals)" %
          (n_semi, K, J, n_semi * K * J))
    print("=" * 78)

    L_log_mag = np.zeros((n_semi, K, J), dtype=np.float64)
    total_evals = n_semi * K * J
    eval_count = 0
    t_start = time.perf_counter()

    for ni, (N, p, q) in enumerate(semiprimes):
        max_terms_N = int(N * 0.5) + 50
        chi = compute_chi_N(N, max_terms_N)

        for fi, form in enumerate(forms):
            k = form['weight']
            coeffs = form['coeffs']
            eps = form['epsilon']
            s_center = k / 2.0

            s_points = [complex(s_center, t) for t in T_CRIT] + \
                       [complex(s_center + 1.0, t) for t in T_RIGHT]

            for ji, s in enumerate(s_points):
                L_val = eval_L_afe(s, coeffs, chi, N, k, eps, max_terms_N)
                mag = abs(L_val)
                L_log_mag[ni, fi, ji] = np.log(mag + 1e-30)
                eval_count += 1

        if (ni + 1) % 5 == 0 or ni == 0 or ni == n_semi - 1:
            elapsed = time.perf_counter() - t_start
            rate = eval_count / (elapsed + 0.001)
            eta = (total_evals - eval_count) / (rate + 0.001)
            print("  N=%6d (%2d/%2d)  %.0f evals/s  ETA %.0fs" %
                  (N, ni + 1, n_semi, rate, eta), flush=True)

    dt_total = time.perf_counter() - t_start
    print("  Total: %d evaluations in %.1fs (%.0f/s)" %
          (eval_count, dt_total, eval_count / dt_total), flush=True)

    # 4. Feature engineering
    print("\n" + "=" * 78)
    print("Feature engineering and regression")
    print("=" * 78)

    y = np.array([np.log(float(p)) for N, p, q in semiprimes])
    y_centered = y - np.mean(y)

    X_raw = L_log_mag.reshape(n_semi, K * J)

    # z-score per feature
    X_mean = np.mean(X_raw, axis=0, keepdims=True)
    X_std = np.std(X_raw, axis=0, keepdims=True)
    X_std = np.where(X_std < 1e-10, 1.0, X_std)
    X_zscore = (X_raw - X_mean) / X_std

    # 5. R^2 vs K (primary metric)
    print("\n--- R^2 vs K (adding forms by increasing weight) ---")
    print("  %-4s %-10s %8s %10s %10s %5s" %
          ("K", "added", "R^2", "null_med", "null_95", "sig"))

    n_pca = min(12, n_semi // 5)
    r2_vs_K = []
    r2_null_vs_K = []

    for k_count in range(1, K + 1):
        cols = []
        for fi in range(k_count):
            cols.extend(range(fi * J, (fi + 1) * J))
        X_k = X_zscore[:, cols]

        n_comp = min(n_pca, X_k.shape[1], n_semi - 2)
        if n_comp < 1:
            n_comp = 1
        X_pca = pca_reduce(X_k, n_comp)
        r2 = ridge_loocv_r2(X_pca, y_centered)
        r2_vs_K.append(r2)

        rng = np.random.RandomState(42)
        null_r2s = [ridge_loocv_r2(X_pca, rng.permutation(y_centered))
                    for _ in range(500)]
        null_med = float(np.median(null_r2s))
        null_95 = float(np.percentile(null_r2s, 95))
        r2_null_vs_K.append((null_med, null_95))

        sig = "***" if r2 > null_95 else ""
        print("  %-4d %-10s %+8.4f %+10.4f %+10.4f %5s" %
              (k_count, forms[k_count - 1]['label'], r2, null_med, null_95, sig),
              flush=True)

    # 6. Diagnostic: critical vs right-of-critical
    print("\n--- Critical line vs slightly-right signal ---")
    col_crit = []
    col_right = []
    for fi in range(K):
        col_crit.extend(range(fi * J, fi * J + len(T_CRIT)))
        col_right.extend(range(fi * J + len(T_CRIT), (fi + 1) * J))

    X_c_pca = pca_reduce(X_zscore[:, col_crit],
                          min(n_pca, len(col_crit), n_semi - 2))
    X_r_pca = pca_reduce(X_zscore[:, col_right],
                          min(n_pca, len(col_right), n_semi - 2))
    r2_crit = ridge_loocv_r2(X_c_pca, y_centered)
    r2_right = ridge_loocv_r2(X_r_pca, y_centered)
    print("  Critical line (Re(s) = k/2):     R^2 = %+.4f" % r2_crit)
    print("  Slightly right (Re(s) = k/2+1):  R^2 = %+.4f" % r2_right)

    # 7. Top individual correlations
    print("\n--- Top-10 individual feature correlations ---")
    print("  %-10s %-16s %8s %10s" % ("Form", "s-point", "r", "p-value"))
    correlations = []
    for fi, form in enumerate(forms):
        for ji in range(J):
            feat = X_zscore[:, fi * J + ji]
            r, pval = stats.pearsonr(feat, y_centered)
            if ji < len(T_CRIT):
                s_desc = "k/2+%.1fi" % T_CRIT[ji]
            else:
                s_desc = "k/2+1+%.1fi" % T_RIGHT[ji - len(T_CRIT)]
            correlations.append((abs(r), r, pval, form['label'], s_desc))
    correlations.sort(reverse=True)

    # Apply Bonferroni correction for multiple testing across all form x s-point pairs
    n_tests = len(correlations)
    print("  %-10s %-16s %8s %10s %12s" % ("Form", "s-point", "r", "p-value", "bonf_p"))
    for _, r, pval, label, s_desc in correlations[:10]:
        bonf_p = min(1.0, pval * n_tests)
        print("  %-10s %-16s %+8.4f %10.2e %12.2e" % (label, s_desc, r, pval, bonf_p))

    # 8. Save results
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'data', 'E8b_multi_form_results.json')
    results = {
        'n_semiprimes': n_semi,
        'n_forms': K,
        'n_s_points': J,
        'n_tests_bonferroni': n_tests,
        'weights': [int(f['weight']) for f in forms],
        'form_labels': [f['label'] for f in forms],
        'r2_vs_K': r2_vs_K,
        'r2_null_vs_K': r2_null_vs_K,
        'r2_critical_line': r2_crit,
        'r2_slightly_right': r2_right,
        'top_correlations': [
            {'form': label, 's': s_desc, 'r': float(r), 'p': float(pval),
             'bonferroni_p': float(min(1.0, pval * n_tests))}
            for _, r, pval, label, s_desc in correlations[:20]
        ],
    }
    safe_json_dump(results, outpath)
    print("\nSaved -> %s" % os.path.abspath(outpath), flush=True)

    # 9. Plot
    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', 'E8b_r2_vs_K.png')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    Ks = np.arange(1, K + 1)
    r2_arr = np.array(r2_vs_K)
    null_med = np.array([x[0] for x in r2_null_vs_K])
    null_95 = np.array([x[1] for x in r2_null_vs_K])

    ax.plot(Ks, r2_arr, 'b-o', linewidth=2, markersize=5, label='Actual R^2')
    ax.plot(Ks, null_med, 'r--', linewidth=1, label='Null median')
    ax.plot(Ks, null_95, 'r:', linewidth=1, label='Null 95th pct')
    ax.fill_between(Ks, null_med, null_95, alpha=0.15, color='red')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('K (number of forms)')
    ax.set_ylabel('LOOCV R^2 for predicting log(min(p,q))')
    ax.set_title('E8b: Multi-Form L-function Amplification\n'
                 '(%d semiprimes, %d s-points/form, PCA+Ridge)' % (n_semi, J))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plotpath, dpi=150)
    print("Saved -> %s" % os.path.abspath(plotpath), flush=True)

    # 10. Verdict
    print("\n" + "=" * 78)
    print("E8b MULTI-FORM AMPLIFICATION -- VERDICT")
    print("=" * 78)

    max_r2 = max(r2_vs_K)
    max_k = r2_vs_K.index(max_r2) + 1
    last_r2 = r2_vs_K[-1]
    last_null_95 = r2_null_vs_K[-1][1]

    if K >= 4:
        first_half = np.mean(r2_vs_K[:K // 2])
        second_half = np.mean(r2_vs_K[K // 2:])
        trend = "INCREASING" if second_half > first_half + 0.02 else \
                "DECREASING" if second_half < first_half - 0.02 else "FLAT"
    else:
        trend = "INSUFFICIENT DATA"

    significant = last_r2 > last_null_95

    print("  %d forms, %d semiprimes" % (K, n_semi))
    print("  Best R^2 = %.4f at K = %d" % (max_r2, max_k))
    print("  Final R^2 (K=%d) = %.4f  (null 95th = %.4f)" %
          (K, last_r2, last_null_95))
    print("  R^2 trend: %s (first half %.4f, second half %.4f)" %
          (trend, first_half, second_half))
    print("  Significant: %s" % ("YES" if significant else "NO"))
    print()

    if significant and trend == "INCREASING":
        print("  AMPLIFICATION DETECTED: R^2 increases with K above null.")
        print("  Each form carries independent factor information.")
    elif significant:
        print("  WEAK SIGNAL: R^2 exceeds null but does not grow with K.")
        print("  All forms see the same limited factor information.")
    else:
        print("  NO SIGNAL: R^2 within null distribution.")
        print("  Multi-form L-values carry no extractable factor information")
        print("  beyond what single-form evaluation provides.")


if __name__ == '__main__':
    main()
