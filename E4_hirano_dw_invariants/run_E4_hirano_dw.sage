"""
E4: Hirano DW Invariants — Jacobi Autocorrelation & Gauss Sums
===============================================================
Tests whether DW-adjacent observables carry factor information for semiprimes.

Key observables:
  1. DW trivial cocycle: gcd(N, m) for m = 1..M (known: trial division)
  2. Jacobi symbol autocorrelation at factor lags p, q vs random lags
  3. Gauss sum |G(1,N)| / sqrt(N) magnitude and phase
  4. Cross-Legendre: (p/q) * (q/p) [quadratic reciprocity]

Targets: hinge scalars S_D(N) = chi_D(p) + chi_D(q)

Analysis: Pearson correlations, paired t-test (factor vs random lags),
permutation test on ridge LOOCV R^2, prime baseline for Gauss sums.
"""
import sys, os, time
import numpy as np
from collections import OrderedDict
from scipy import stats as sp_stats

from sage.all import (
    kronecker_symbol, gcd, isqrt, next_prime, is_prime,
    set_random_seed, ZZ, CC, exp, pi, I
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)
np.random.seed(42)

DISCRIMINANTS = [-3, -4, 5, -7, 8]
DW_M_MAX = 30   # trivial cocycle gauge groups Z/m for m=2..DW_M_MAX
N_RANDOM_LAGS = 30  # random lags for autocorrelation baseline
GAUSS_N_LIMIT = 50000  # max N for O(N) Gauss sum computation


# ============================================================
# Feature computation
# ============================================================

def jacobi_sequence(N):
    """Compute Jacobi symbol sequence (a/N) for a = 0..N-1."""
    N = int(N)
    return np.array([int(kronecker_symbol(a, N)) for a in range(N)], dtype=float)


def autocorrelation_at_lag(seq, lag):
    """Compute normalized autocorrelation of seq at given lag.

    Returns sum_{i=0}^{n-lag-1} seq[i]*seq[i+lag] / (n * var),
    or 0.0 if variance is zero.
    """
    n = len(seq)
    lag = int(lag)
    if lag <= 0 or lag >= n:
        return 0.0
    var = np.var(seq)
    if var < 1e-15:
        return 0.0
    s = np.sum(seq[:n - lag] * seq[lag:])
    return float(s / (n * var))


def compute_jacobi_autocorr(N, p, q, jac_seq):
    """Compute Jacobi autocorrelation at factor lags and random baseline.

    Returns OrderedDict of autocorrelation features.
    """
    N = int(N)
    p = int(p)
    q = int(q)
    features = OrderedDict()

    # Autocorrelation at factor lags
    ac_p = autocorrelation_at_lag(jac_seq, p)
    ac_q = autocorrelation_at_lag(jac_seq, q)
    features['ac_p'] = ac_p
    features['ac_q'] = ac_q
    features['ac_factor_mean'] = (ac_p + ac_q) / 2.0

    # Random lag baseline
    rng = np.random.RandomState(int(N) % (2**31))
    max_lag = min(N // 2, 200)
    if max_lag < 3:
        features['ac_random_mean'] = 0.0
        features['ac_random_std'] = 1.0
        features['ac_p_zscore'] = 0.0
        features['ac_q_zscore'] = 0.0
        return features

    candidate_lags = [L for L in range(2, max_lag) if L != p and L != q]
    if len(candidate_lags) < N_RANDOM_LAGS:
        chosen_lags = candidate_lags
    else:
        chosen_lags = list(rng.choice(candidate_lags, size=int(N_RANDOM_LAGS), replace=False))

    random_acs = [autocorrelation_at_lag(jac_seq, L) for L in chosen_lags]
    ac_rand_mean = float(np.mean(random_acs))
    ac_rand_std = float(np.std(random_acs))
    if ac_rand_std < 1e-15:
        ac_rand_std = 1.0

    features['ac_random_mean'] = ac_rand_mean
    features['ac_random_std'] = ac_rand_std
    features['ac_p_zscore'] = (ac_p - ac_rand_mean) / ac_rand_std
    features['ac_q_zscore'] = (ac_q - ac_rand_mean) / ac_rand_std

    return features


def compute_gauss_features(N):
    """Compute Gauss sum G(1,N) and derived features.

    G(1,N) = sum_{a=1}^{N-1} (a/N) * exp(2*pi*i*a/N)
    WARNING: O(N) computation.
    """
    N = int(N)
    features = OrderedDict()

    if N > GAUSS_N_LIMIT:
        features['gauss_ratio'] = float('nan')
        features['gauss_phase'] = float('nan')
        return features

    # Use numpy for speed: precompute Jacobi symbols and phases
    a_vals = np.arange(1, N)
    jac_vals = np.array([int(kronecker_symbol(int(a), N)) for a in a_vals], dtype=complex)
    phases = np.exp(2j * np.pi * a_vals / N)
    G = np.sum(jac_vals * phases)

    mag = abs(G)
    features['gauss_ratio'] = float(mag / np.sqrt(N))
    features['gauss_phase'] = float(np.angle(G) / (2 * np.pi))

    return features


def compute_dw_features(N):
    """DW trivial cocycle spectrum: gcd(N, m) / N for m=2..M_MAX.

    This is KNOWN to be equivalent to trial division (included as control).
    """
    N = int(N)
    features = OrderedDict()
    for m in range(2, DW_M_MAX + 1):
        features['dw_gcd_%d' % m] = float(gcd(N, m)) / float(N)
    return features


def compute_targets(N, p, q):
    """Compute hinge scalar targets (require knowledge of factors)."""
    targets = OrderedDict()
    for D in DISCRIMINANTS:
        chi_p = int(kronecker_symbol(D, int(p)))
        chi_q = int(kronecker_symbol(D, int(q)))
        targets['S_%d' % D] = chi_p + chi_q
    return targets


# ============================================================
# Statistical analysis
# ============================================================

def ridge_loocv_r2(X, y, lambdas=None):
    """Ridge regression with LOOCV using hat matrix formula."""
    n, p = X.shape
    if p == 0 or n < 5:
        return -999.0, -999.0

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

    return float(best_r2), float(best_lam)


def permutation_test_r2(X, y, n_perm=200, seed=42):
    """Permutation test for ridge LOOCV R^2."""
    rng = np.random.default_rng(int(seed))
    real_r2, real_lam = ridge_loocv_r2(X, y)
    null_r2s = []
    for _ in range(int(n_perm)):
        y_perm = rng.permutation(y)
        r2, _ = ridge_loocv_r2(X, y_perm, lambdas=[real_lam])
        null_r2s.append(r2)
    null_r2s = np.array(null_r2s)
    p_value = float(np.mean(null_r2s >= real_r2))
    return real_r2, p_value, null_r2s


def compute_prime_baseline(bit_sizes, count=100):
    """Compute Gauss sum ratio for primes as a baseline.

    For primes P, |G(1,P)| / sqrt(P) = 1 exactly.
    Returns (mean_ratio, std_ratio, ratios).
    """
    ratios = []
    rng = np.random.RandomState(int(123))
    for bits in bit_sizes:
        lo = 2**(bits - 1)
        hi = 2**bits
        for _ in range(count // len(bit_sizes)):
            p_start = lo + int(rng.randint(0, max(1, hi - lo)))
            P = int(next_prime(p_start))
            if P > GAUSS_N_LIMIT:
                continue
            a_vals = np.arange(1, P)
            jac_vals = np.array([int(kronecker_symbol(int(a), P)) for a in a_vals],
                                dtype=complex)
            phases = np.exp(2j * np.pi * a_vals / P)
            G = np.sum(jac_vals * phases)
            ratios.append(float(abs(G) / np.sqrt(P)))
    ratios = np.array(ratios)
    return float(np.mean(ratios)), float(np.std(ratios)), ratios


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("E4: Hirano DW Invariants -- Jacobi Autocorrelation & Gauss Sums")
    print("=" * 70)

    bit_sizes = [14, 16, 18, 20]
    count_per_size = int(60)

    print("Generating balanced semiprimes...")
    semiprimes = balanced_semiprimes(bit_sizes, count_per_size=count_per_size,
                                     min_ratio=0.3, seed=int(42))
    print("  Total: %d semiprimes across %s bit sizes" % (len(semiprimes), bit_sizes))

    # ----------------------------------------------------------
    # Compute features and targets
    # ----------------------------------------------------------
    print("\nComputing features...")
    t0 = time.time()

    all_features = []
    all_targets = []
    metadata = []
    ac_factor_vals = []
    ac_random_vals = []

    for idx, (N, p, q) in enumerate(semiprimes):
        if idx % 50 == 0 and idx > 0:
            print("  Processing %d/%d..." % (idx, len(semiprimes)))

        features = OrderedDict()

        # Jacobi autocorrelation features
        jac_seq = jacobi_sequence(N)
        ac_feats = compute_jacobi_autocorr(N, p, q, jac_seq)
        features.update(ac_feats)
        ac_factor_vals.append(ac_feats['ac_factor_mean'])
        ac_random_vals.append(ac_feats['ac_random_mean'])

        # Gauss sum features
        gauss_feats = compute_gauss_features(N)
        features.update(gauss_feats)

        # DW trivial cocycle (control -- known trial division)
        dw_feats = compute_dw_features(N)
        features.update(dw_feats)

        all_features.append(features)
        all_targets.append(compute_targets(N, p, q))
        metadata.append({
            'N': int(N), 'p': int(p), 'q': int(q),
            'bits': int(N).bit_length(),
            'ratio': float(p) / float(q)
        })

    elapsed = time.time() - t0
    print("  Feature computation: %.1fs" % elapsed)

    # ----------------------------------------------------------
    # Build feature and target matrices
    # ----------------------------------------------------------
    feature_names = list(all_features[0].keys())
    target_names = list(all_targets[0].keys())
    n = len(all_features)
    d = len(feature_names)

    X = np.zeros((n, d))
    for i, row in enumerate(all_features):
        for j, name in enumerate(feature_names):
            val = row[name]
            X[i, j] = float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0

    # Replace any remaining NaN
    X = np.nan_to_num(X, nan=0.0)

    print("\nFeature matrix: %d samples x %d features" % (n, d))
    print("  Autocorrelation: ac_p, ac_q, ac_factor_mean, ac_random_mean/std, z-scores")
    print("  Gauss sum: gauss_ratio, gauss_phase")
    print("  DW control: dw_gcd_2..dw_gcd_%d (%d features)" % (DW_M_MAX, DW_M_MAX - 1))

    # ----------------------------------------------------------
    # Analysis 1: Per-feature correlations with hinge scalars
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Feature correlations with hinge scalars")
    print("=" * 70)

    target_results = {}
    all_max_r = []

    for tname in target_names:
        y = np.array([float(row[tname]) for row in all_targets])
        correlations = {}
        for j, fname in enumerate(feature_names):
            x = X[:, j]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                correlations[fname] = 0.0
            else:
                correlations[fname] = float(np.corrcoef(x, y)[0, 1])

        top = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        max_abs_r = abs(top[0][1]) if top else 0.0
        all_max_r.append(max_abs_r)

        print("\n  Target: %s" % tname)
        for fname, r in top[:5]:
            print("    %20s: r = %+.4f" % (fname, r))
        print("    max |r| = %.4f" % max_abs_r)

        target_results[tname] = {
            'top_correlations': [(fname, float(r)) for fname, r in top],
            'max_abs_r': float(max_abs_r),
            'all_correlations': {fname: float(r) for fname, r in correlations.items()},
        }

    overall_max_r = max(all_max_r)
    print("\n  Overall max |r| across all targets: %.4f" % overall_max_r)

    # ----------------------------------------------------------
    # Analysis 2: Paired test -- factor lags vs random lags
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Factor-lag vs random-lag autocorrelation (paired)")
    print("=" * 70)

    ac_factor_arr = np.array(ac_factor_vals)
    ac_random_arr = np.array(ac_random_vals)
    diff = ac_factor_arr - ac_random_arr

    t_stat, t_pval = sp_stats.ttest_rel(ac_factor_arr, ac_random_arr)
    print("  Mean ac_factor: %.6f" % np.mean(ac_factor_arr))
    print("  Mean ac_random: %.6f" % np.mean(ac_random_arr))
    print("  Mean difference: %.6f" % np.mean(diff))
    print("  Paired t-test: t=%.4f, p=%.4f" % (float(t_stat), float(t_pval)))
    print("  Significant at 0.05? %s" % ("YES" if float(t_pval) < 0.05 else "NO"))

    # ----------------------------------------------------------
    # Analysis 3: Ridge LOOCV + permutation test per target
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Ridge LOOCV R^2 + permutation test (200 shuffles)")
    print("=" * 70)

    # Use only non-control features (exclude DW)
    non_dw_cols = [j for j, fname in enumerate(feature_names)
                   if not fname.startswith('dw_')]
    X_nodw = X[:, non_dw_cols]
    non_dw_names = [feature_names[j] for j in non_dw_cols]

    print("  Using %d non-control features: %s" % (
        len(non_dw_names), ', '.join(non_dw_names)))

    for tname in target_names:
        y = np.array([float(row[tname]) for row in all_targets])
        r2, pval, null = permutation_test_r2(X_nodw, y, n_perm=int(200))
        p95 = float(np.percentile(null, 95))
        target_results[tname]['r2_cv'] = float(r2)
        target_results[tname]['perm_p_value'] = float(pval)
        target_results[tname]['null_95th'] = float(p95)
        status = "SIGNAL" if (r2 > 0.05 and pval < 0.05) else "FLAT"
        print("  %6s: R^2_CV=%+.6f, p=%.3f, null_95th=%.4f [%s]" % (
            tname, r2, pval, p95, status))

    # ----------------------------------------------------------
    # Analysis 4: Prime baseline for Gauss sum ratio
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Prime baseline for |G(1,N)|/sqrt(N)")
    print("=" * 70)

    prime_mean, prime_std, prime_ratios = compute_prime_baseline(bit_sizes, count=int(100))
    semi_ratios = np.array([row['gauss_ratio'] for row in all_features
                            if not np.isnan(row['gauss_ratio'])])
    print("  Primes:     mean=%.6f, std=%.6f (n=%d)" % (
        prime_mean, prime_std, len(prime_ratios)))
    print("  Semiprimes: mean=%.6f, std=%.6f (n=%d)" % (
        float(np.mean(semi_ratios)), float(np.std(semi_ratios)), len(semi_ratios)))

    if len(semi_ratios) > 5 and len(prime_ratios) > 5:
        ks_stat, ks_p = sp_stats.ks_2samp(prime_ratios, semi_ratios)
        print("  KS test (prime vs semi): D=%.4f, p=%.4f" % (float(ks_stat), float(ks_p)))
    else:
        ks_stat, ks_p = float('nan'), float('nan')
        print("  KS test: insufficient samples")

    # ----------------------------------------------------------
    # Per bit-size breakdown
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER BIT-SIZE BREAKDOWN")
    print("=" * 70)

    per_bit = {}
    for bits in bit_sizes:
        idxs = [i for i, m in enumerate(metadata) if m['bits'] == int(bits)]
        if not idxs:
            continue
        sub_ac_factor = np.array([ac_factor_vals[i] for i in idxs])
        sub_ac_random = np.array([ac_random_vals[i] for i in idxs])
        sub_gauss = np.array([all_features[i]['gauss_ratio'] for i in idxs
                              if not np.isnan(all_features[i]['gauss_ratio'])])
        per_bit[int(bits)] = {
            'n': len(idxs),
            'mean_ac_factor': float(np.mean(sub_ac_factor)),
            'mean_ac_random': float(np.mean(sub_ac_random)),
            'mean_gauss_ratio': float(np.mean(sub_gauss)) if len(sub_gauss) > 0 else float('nan'),
        }
        print("  %d-bit (n=%d): ac_factor=%.4f, ac_random=%.4f, gauss_ratio=%.4f" % (
            bits, len(idxs),
            per_bit[bits]['mean_ac_factor'],
            per_bit[bits]['mean_ac_random'],
            per_bit[bits]['mean_gauss_ratio']))

    # ----------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    hinge_r2s = [target_results[t].get('r2_cv', -999) for t in target_names]
    hinge_pvals = [target_results[t].get('perm_p_value', 1.0) for t in target_names]
    max_r2 = max(hinge_r2s)
    min_pval = min(hinge_pvals)

    for tname in target_names:
        r2 = target_results[tname].get('r2_cv', -999)
        pv = target_results[tname].get('perm_p_value', 1.0)
        status = "SIGNAL" if (r2 > 0.05 and pv < 0.05) else "FLAT"
        print("  %6s: R^2_CV=%+.6f, p=%.3f  [%s]" % (tname, r2, pv, status))

    print()
    print("  Max hinge R^2_CV: %.6f" % max_r2)
    print("  Min hinge p-value: %.4f" % min_pval)
    print("  Overall max |r|:  %.4f" % overall_max_r)
    print("  Paired t-test p:  %.4f" % float(t_pval))

    if max_r2 > 0.05 and min_pval < 0.05:
        verdict = "SIGNAL -- investigate whether known CRT mechanism"
    elif max_r2 > 0.01 and min_pval < 0.05:
        verdict = "WEAK SIGNAL -- marginal R^2, needs investigation"
    else:
        verdict = "BARRIER -- DW/Gauss observables carry no factor info beyond known mechanisms"

    print("\n  VERDICT: %s" % verdict)

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    results = {
        'config': {
            'bit_sizes': bit_sizes,
            'count_per_size': int(count_per_size),
            'n_semiprimes': n,
            'n_features': d,
            'feature_names': feature_names,
            'target_names': target_names,
            'compute_time_s': float(elapsed),
            'dw_m_max': DW_M_MAX,
            'n_random_lags': N_RANDOM_LAGS,
        },
        'targets': target_results,
        'paired_ttest': {
            'mean_ac_factor': float(np.mean(ac_factor_arr)),
            'mean_ac_random': float(np.mean(ac_random_arr)),
            't_stat': float(t_stat),
            'p_value': float(t_pval),
            'significant_05': bool(float(t_pval) < 0.05),
        },
        'prime_baseline': {
            'mean_gauss_ratio': float(prime_mean),
            'std_gauss_ratio': float(prime_std),
            'n_primes': len(prime_ratios),
            'ks_stat': float(ks_stat) if not np.isnan(ks_stat) else None,
            'ks_p_value': float(ks_p) if not np.isnan(ks_p) else None,
        },
        'per_bit_size': per_bit,
        'verdict': verdict,
        'max_hinge_r2': float(max_r2),
        'min_hinge_pval': float(min_pval),
        'overall_max_abs_r': float(overall_max_r),
    }

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'E4_hirano_dw_results.json')
    safe_json_dump(results, json_path)

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('E4: Hirano DW Invariants (%d semiprimes, %s bit)' % (
            n, '-'.join(str(b) for b in bit_sizes)), fontsize=13)

        # Plot 1: Top correlations for S_-3
        ax = axes[0]
        tname = 'S_-3'
        corrs = target_results[tname]['all_correlations']
        top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
        names = [t[0] for t in top]
        vals = [t[1] for t in top]
        colors = ['red' if abs(v) > 0.15 else 'steelblue' for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Pearson r')
        ax.set_title('Correlations with %s' % tname)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvline(0.15, color='red', ls='--', alpha=0.3)
        ax.axvline(-0.15, color='red', ls='--', alpha=0.3)
        ax.invert_yaxis()

        # Plot 2: Factor-lag vs random-lag autocorrelation
        ax = axes[1]
        ax.scatter(ac_random_arr, ac_factor_arr, s=15, alpha=0.5, c='steelblue')
        lims = [min(ac_random_arr.min(), ac_factor_arr.min()) - 0.01,
                max(ac_random_arr.max(), ac_factor_arr.max()) + 0.01]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Mean random-lag autocorrelation')
        ax.set_ylabel('Mean factor-lag autocorrelation')
        ax.set_title('Factor vs Random Lags (t=%.2f, p=%.3f)' % (
            float(t_stat), float(t_pval)))
        ax.legend(fontsize=8)

        # Plot 3: Gauss ratio distributions (handle constant data)
        ax = axes[2]
        all_ratios = np.concatenate([prime_ratios, semi_ratios]) if (
            len(prime_ratios) > 0 and len(semi_ratios) > 0) else (
            prime_ratios if len(prime_ratios) > 0 else semi_ratios)
        ratio_range = float(np.ptp(all_ratios)) if len(all_ratios) > 0 else 0.0
        if ratio_range > 1e-10 and len(all_ratios) > 1:
            if len(prime_ratios) > 0:
                ax.hist(prime_ratios, bins=int(20), alpha=0.5, color='green',
                        label='Primes (n=%d)' % len(prime_ratios), density=True)
            if len(semi_ratios) > 0:
                ax.hist(semi_ratios, bins=int(20), alpha=0.5, color='steelblue',
                        label='Semiprimes (n=%d)' % len(semi_ratios), density=True)
        else:
            # Constant data -- show as text
            ax.text(0.5, 0.5,
                    '|G(1,N)|/sqrt(N) = %.4f\n(constant for all N)' % float(
                        np.mean(all_ratios)) if len(all_ratios) > 0 else 'No data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_xlabel('|G(1,N)| / sqrt(N)')
        ax.set_title('Gauss Sum Ratio Distribution')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E4_hirano_dw.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("Plot saved to %s" % plot_path)
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == '__main__':
    main()
