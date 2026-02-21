"""
E14: Nonlinear ML on Poly(log N) Features
==========================================

E11 tested 111 features with Ridge regression (linear) and found R^2_CV <= 0.025.
But linear methods cannot detect nonlinear encodings (e.g., XOR of Jacobi symbols).

This experiment applies nonlinear classifiers (Random Forest, Gradient Boosting,
K-Nearest Neighbors) to the same feature set, plus pairwise interaction features,
to test whether factor information is encoded nonlinearly.

Targets: S_D(N) = chi_D(p) + chi_D(q) for D in {-3, -4, 5, -7, 8}
    These are categorical: values in {-2, 0, 2}

Models:
    1. RandomForestClassifier (100 trees)
    2. GradientBoostingClassifier (100 trees, depth 3)
    3. KNeighborsClassifier (k=5, 7, 11)

Evaluation:
    - Stratified 5-fold CV accuracy
    - Permutation test (200 shuffles) for statistical significance
    - Benjamini-Hochberg FDR correction across all target x model tests

Success criterion: CV accuracy > 40% (vs 33.3% random baseline for 3-class)
    with BH-adjusted p < 0.01
"""
import sys
import os
import time

import numpy as np
from collections import OrderedDict

from sage.all import (
    kronecker_symbol, next_prime, isqrt, ZZ, gcd,
    power_mod, is_prime, set_random_seed
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from sage_encoding import safe_json_dump

# Reproducibility
set_random_seed(42)
np.random.seed(42)

# Import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ============================================================
# Feature and target computation — adapted from E11
# ============================================================

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
DISCRIMINANTS = [-3, -4, 5, -7, 8]
MODEXP_BASES = [2, 3, 5, 7, 11]


def generate_semiprimes_fixed_bits(bit_size, count, min_ratio=0.3):
    """Generate balanced semiprimes N=pq with exactly `bit_size` bits."""
    results = []
    seen = set()
    half = bit_size // 2
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


def cf_sqrt(N, max_terms=15):
    """Compute CF expansion of sqrt(N) and convergent remainders."""
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


def compute_features(N):
    """Compute ~111 poly(log N) features for semiprime N."""
    N = int(N)
    features = OrderedDict()
    sqrtN = int(isqrt(N))

    # Group 1: Jacobi symbols
    for k in SMALL_PRIMES:
        features['J_%d' % k] = int(kronecker_symbol(k, N))

    # Group 2: Modular exponentiations
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

    # Group 3: Euler residuals
    euler_exp = N - 2 * sqrtN + 1
    for g in MODEXP_BASES:
        g_int = int(g)
        if int(gcd(g_int, N)) > 1:
            features['euler_resid_%d' % g_int] = -1.0
        else:
            val = int(power_mod(g_int, euler_exp, N))
            features['euler_resid_%d' % g_int] = val / float(N)

    # Group 4: Carry/integer features
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

    # Group 5: CF convergent features
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

    # Group 6: Pollard p-1 remnants
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
            if 1 < d < N:
                features['pollard_%d_found' % g_int] = 1
            elif d == N:
                features['pollard_%d_found' % g_int] = -1
            else:
                features['pollard_%d_found' % g_int] = 0
            features['pollard_%d_val' % g_int] = val / float(N) if d == 1 else -1.0

    # Group 7: Mixed/interaction features
    g2N = features.get('modexp_2_N', 0.0)
    g3N = features.get('modexp_3_N', 0.0)
    for k in [2, 3, 5, 7, 11]:
        jk = features.get('J_%d' % k, 0)
        features['mix_J%d_g2N' % k] = jk * g2N
    if g2N >= 0 and g3N >= 0:
        features['mix_g2g3_diff'] = abs(g2N - g3N)
        features['mix_g2g3_xor'] = int.__xor__(int(g2N * N), int(g3N * N)) / float(max(N, 1))
    else:
        features['mix_g2g3_diff'] = 0.0
        features['mix_g2g3_xor'] = 0.0
    er2 = features.get('euler_resid_2', 0.0)
    for k in [2, 3, 5]:
        jk = features.get('J_%d' % k, 0)
        features['mix_er2_J%d' % k] = er2 * jk

    # Group 8: N-arithmetic controls
    for k in [3, 5, 7, 8, 11]:
        features['N_mod_%d' % k] = N % k

    # Group 9: Random controls
    rng = np.random.RandomState(N % (2**31))
    for i in range(5):
        features['random_%d' % i] = rng.random()

    return features


def compute_targets(N, p, q):
    """Compute target variables (requires knowledge of factors)."""
    targets = OrderedDict()
    for D in DISCRIMINANTS:
        chi_p = int(kronecker_symbol(D, int(p)))
        chi_q = int(kronecker_symbol(D, int(q)))
        targets['S_%d' % D] = chi_p + chi_q
    targets['p_normalized'] = float(p) / float(isqrt(N))
    targets['log_ratio'] = float(np.log(float(q) / float(p)))
    targets['sum_normalized'] = float(p + q) / (2.0 * float(isqrt(N)))
    return targets


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        raw_adjusted = sorted_pvals[i] * n / (i + 1)
        if i == n - 1:
            adjusted[i] = min(raw_adjusted, 1.0)
        else:
            adjusted[i] = min(adjusted[i + 1], raw_adjusted)
    result = np.zeros(n)
    result[sorted_indices] = adjusted
    return result


# ============================================================
# Nonlinear classification pipeline
# ============================================================

def stratified_cv_accuracy(clf_factory, X, y, n_splits=5, rng_seed=42):
    """Stratified K-fold CV accuracy for a classifier.

    clf_factory: callable returning a fresh classifier instance.
    Returns mean accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train_s, y_train)
        acc = float(clf.score(X_test_s, y_test))
        accs.append(acc)

    return float(np.mean(accs)), accs


def permutation_test_clf(clf_factory, X, y, n_perm=200, n_splits=5, rng_seed=42):
    """Permutation test for classifier accuracy.

    Returns (real_acc, p_value, null_distribution).
    """
    real_acc, _ = stratified_cv_accuracy(clf_factory, X, y, n_splits, rng_seed)

    rng = np.random.RandomState(rng_seed + 1000)
    null_accs = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_accs[i], _ = stratified_cv_accuracy(
            clf_factory, X, y_perm, n_splits, rng_seed=rng_seed + i + 1)

    p_value = float(np.mean(null_accs >= real_acc))
    return real_acc, p_value, null_accs


def add_pairwise_interactions(X, feature_names, top_k=20, anova_ranking=None):
    """Add pairwise interaction features for top-K features.

    If anova_ranking is provided (list of (name, f_stat)), use that to select top-K.
    Otherwise use variance ranking.

    Returns (X_augmented, augmented_feature_names).
    """
    if anova_ranking is not None:
        top_names = [name for name, _ in anova_ranking[:top_k]]
        top_indices = [feature_names.index(n) for n in top_names if n in feature_names]
    else:
        variances = np.var(X, axis=0)
        top_indices = list(np.argsort(variances)[::-1][:top_k])

    top_indices = top_indices[:top_k]

    new_features = []
    new_names = list(feature_names)

    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            idx_i = top_indices[i]
            idx_j = top_indices[j]
            product = X[:, idx_i] * X[:, idx_j]
            new_features.append(product)
            new_names.append('%s_x_%s' % (feature_names[idx_i], feature_names[idx_j]))

    if new_features:
        X_aug = np.column_stack([X] + new_features)
    else:
        X_aug = X

    return X_aug, new_names


# ============================================================
# Main experiment
# ============================================================

def run_experiment(bit_sizes, count_per_size, output_dir='../data'):
    print("=" * 70, flush=True)
    print("E14: Nonlinear ML on Poly(log N) Features", flush=True)
    print("=" * 70, flush=True)
    print("Bit sizes: %s" % bit_sizes, flush=True)
    print("Semiprimes per size: %d" % count_per_size, flush=True)
    print(flush=True)

    # --- Generate semiprimes ---
    print("Generating semiprimes...", flush=True)
    all_semiprimes = []
    for bits in bit_sizes:
        sp = generate_semiprimes_fixed_bits(bits, count_per_size)
        print("  %d-bit: %d semiprimes (p/q ratio: %.3f to %.3f)" % (
            bits, len(sp),
            min(s[1]/float(s[2]) for s in sp) if sp else 0,
            max(s[1]/float(s[2]) for s in sp) if sp else 0), flush=True)
        all_semiprimes.extend(sp)

    n_total = len(all_semiprimes)
    print("Total: %d semiprimes\n" % n_total, flush=True)

    # --- Compute features and targets ---
    print("Computing features and targets...", flush=True)
    t0 = time.time()

    feature_rows = []
    target_rows = []
    metadata_rows = []

    for idx, (N, p, q) in enumerate(all_semiprimes):
        if idx % 100 == 0 and idx > 0:
            print("  Processing %d/%d..." % (idx, n_total), flush=True)
        feats = compute_features(N)
        targs = compute_targets(N, p, q)
        feature_rows.append(feats)
        target_rows.append(targs)
        metadata_rows.append({'N': N, 'p': p, 'q': q, 'bits': N.bit_length(),
                              'ratio': float(p) / float(q)})

    elapsed = time.time() - t0
    print("Feature computation: %.1fs\n" % elapsed, flush=True)

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

    print("Feature matrix: %d x %d" % (n, d), flush=True)
    print("Targets: %s\n" % target_names, flush=True)

    # --- Compute ANOVA ranking for interaction selection ---
    # Use S_{-3} as reference target for feature ranking
    y_ref = Y['S_-3'].astype(int)
    categories = np.unique(y_ref)
    anova_ranking = []
    for i, name in enumerate(feature_names):
        x = X[:, i]
        grand_mean = np.mean(x)
        ss_between = 0.0
        ss_within = 0.0
        for cat in categories:
            mask = y_ref == cat
            n_cat = int(np.sum(mask))
            if n_cat == 0:
                continue
            cat_mean = np.mean(x[mask])
            ss_between += n_cat * (cat_mean - grand_mean)**2
            ss_within += np.sum((x[mask] - cat_mean)**2)
        if ss_within < 1e-15 or len(categories) <= 1 or n <= len(categories):
            f_stat = 0.0
        else:
            f_stat = (ss_between / (len(categories) - 1)) / (ss_within / (n - len(categories)))
        anova_ranking.append((name, float(f_stat)))

    anova_ranking.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 features by ANOVA F-stat (S_{-3}):", flush=True)
    for name, f_val in anova_ranking[:10]:
        print("  %-25s: F = %.4f" % (name, f_val), flush=True)
    print(flush=True)

    # --- Add pairwise interactions ---
    X_aug, aug_names = add_pairwise_interactions(
        X, feature_names, top_k=20, anova_ranking=anova_ranking)
    n_interactions = len(aug_names) - len(feature_names)
    print("Added %d pairwise interaction features (total: %d)\n" % (
        n_interactions, len(aug_names)), flush=True)

    # --- Define classifiers ---
    classifiers = OrderedDict([
        ('RF_100', lambda: RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
        ('GBT_100', lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
        ('KNN_5', lambda: KNeighborsClassifier(n_neighbors=5)),
        ('KNN_11', lambda: KNeighborsClassifier(n_neighbors=11)),
    ])

    # --- Test on hinge scalar targets (categorical) ---
    hinge_targets = ['S_%d' % D for D in DISCRIMINANTS]

    results = {
        'config': {
            'bit_sizes': bit_sizes,
            'count_per_size': count_per_size,
            'n_semiprimes': n,
            'n_base_features': d,
            'n_interaction_features': n_interactions,
            'n_total_features': len(aug_names),
            'classifiers': list(classifiers.keys()),
            'hinge_targets': hinge_targets,
            'compute_time_s': float(elapsed),
        },
        'anova_ranking_top20': [(name, float(f)) for name, f in anova_ranking[:20]],
        'tests': {},
    }

    # Feature sets: base features only, and base + interactions
    feature_sets = OrderedDict([
        ('base', (X, feature_names)),
        ('base+interactions', (X_aug, aug_names)),
    ])

    all_test_keys = []
    all_raw_pvals = []

    print("=" * 70, flush=True)
    print("CLASSIFICATION RESULTS", flush=True)
    print("=" * 70, flush=True)

    for fset_name, (X_set, fnames) in feature_sets.items():
        print("\n--- Feature set: %s (%d features) ---" % (fset_name, len(fnames)), flush=True)

        for tname in hinge_targets:
            y = Y[tname].astype(int)
            classes = np.unique(y)
            baseline_acc = float(np.max(np.bincount(y + 2)) / len(y))
            random_baseline = 1.0 / len(classes)
            print("\n  Target: %s (classes: %s, random: %.3f, majority: %.3f)" % (
                tname, list(classes), random_baseline, baseline_acc), flush=True)

            for clf_name, clf_factory in classifiers.items():
                t1 = time.time()
                real_acc, p_val, null_dist = permutation_test_clf(
                    clf_factory, X_set, y, n_perm=int(200))
                dt = time.time() - t1

                null_mean = float(np.mean(null_dist))
                null_std = float(np.std(null_dist))

                test_key = '%s_%s_%s' % (fset_name, clf_name, tname)
                test_result = {
                    'feature_set': fset_name,
                    'classifier': clf_name,
                    'target': tname,
                    'accuracy': float(real_acc),
                    'random_baseline': float(random_baseline),
                    'majority_baseline': float(baseline_acc),
                    'perm_p_value': float(p_val),
                    'null_mean': float(null_mean),
                    'null_std': float(null_std),
                    'time_s': float(dt),
                }

                # Feature importances for RF
                if clf_name.startswith('RF'):
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X_set)
                    rf = clf_factory()
                    rf.fit(X_s, y)
                    importances = rf.feature_importances_
                    top_imp = sorted(zip(fnames, importances),
                                     key=lambda x: x[1], reverse=True)[:15]
                    test_result['top_importances'] = [
                        (name, float(imp)) for name, imp in top_imp]

                results['tests'][test_key] = test_result
                all_test_keys.append(test_key)
                all_raw_pvals.append(p_val)

                status = "SIGNAL" if (real_acc > random_baseline + 0.07 and p_val < 0.05) else "flat"
                print("    %-8s: acc=%.4f  (null: %.4f +/- %.4f)  p=%.4f  [%s]  %.1fs" % (
                    clf_name, real_acc, null_mean, null_std, p_val, status, dt), flush=True)

    # --- Benjamini-Hochberg correction ---
    adjusted_pvals = benjamini_hochberg(all_raw_pvals)
    print("\n--- Benjamini-Hochberg FDR Correction (%d tests) ---" % len(all_test_keys), flush=True)
    for i, key in enumerate(all_test_keys):
        results['tests'][key]['bh_adjusted_pvalue'] = float(adjusted_pvals[i])

    # Print significant results
    sig_count = 0
    for i, key in enumerate(all_test_keys):
        if adjusted_pvals[i] < 0.05:
            r = results['tests'][key]
            print("  SIGNIFICANT: %s  acc=%.4f  BH-p=%.4f" % (key, r['accuracy'], adjusted_pvals[i]),
                  flush=True)
            sig_count += 1

    if sig_count == 0:
        print("  No tests significant after BH correction.", flush=True)

    # --- Per-bit-size analysis with best classifier ---
    print("\n--- Per-Bit-Size Analysis (RF_100, base+interactions) ---", flush=True)
    bit_results = {}
    for bits in bit_sizes:
        mask = np.array([m['bits'] == bits for m in metadata_rows])
        n_sub = int(mask.sum())
        if n_sub < 30:
            print("  %d-bit: skipped (n=%d < 30)" % (bits, n_sub), flush=True)
            continue

        X_sub = X_aug[mask]
        bit_result = {'n': n_sub}
        for tname in ['S_-3', 'S_5']:
            y_sub = Y[tname][mask].astype(int)
            acc, _ = stratified_cv_accuracy(
                classifiers['RF_100'], X_sub, y_sub)
            bit_result['acc_%s' % tname] = float(acc)

        print("  %d-bit (n=%d): S_{-3} acc=%.4f, S_5 acc=%.4f" % (
            bits, n_sub, bit_result['acc_S_-3'], bit_result['acc_S_5']), flush=True)
        bit_results[int(bits)] = bit_result

    results['per_bit_size'] = bit_results

    # --- Verdict ---
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    best_acc = 0.0
    best_test = None
    for key, r in results['tests'].items():
        if r['target'].startswith('S_') and r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_test = key

    if best_test:
        best_r = results['tests'][best_test]
        print("  Best hinge accuracy: %.4f (%s)" % (best_acc, best_test), flush=True)
        print("  Random baseline:     %.4f" % best_r['random_baseline'], flush=True)
        print("  Permutation p-value: %.4f" % best_r['perm_p_value'], flush=True)
        print("  BH-adjusted p-value: %.4f" % best_r['bh_adjusted_pvalue'], flush=True)

    if best_acc > 0.40 and sig_count > 0:
        verdict = "SIGNAL DETECTED — nonlinear encoding may exist, extend to larger N"
    elif best_acc > 0.36:
        verdict = "WEAK SIGNAL — marginal improvement over random, needs investigation"
    else:
        verdict = "BARRIER CONFIRMED — nonlinear classifiers also fail to predict hinge scalars"

    print("\n  VERDICT: %s" % verdict, flush=True)
    results['verdict'] = verdict
    results['best_hinge_accuracy'] = float(best_acc)
    results['n_significant_tests'] = sig_count

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'E14_nonlinear_ml_results.json')
    safe_json_dump(results, json_path)
    print(flush=True)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('E14: Nonlinear ML on Poly(log N) Features', fontsize=14)

        # Plot 1: Accuracy by classifier and target (base+interactions)
        ax = axes[0, 0]
        clf_names = list(classifiers.keys())
        x_pos = np.arange(len(hinge_targets))
        width = 0.18
        for i, clf_name in enumerate(clf_names):
            accs = []
            for tname in hinge_targets:
                key = 'base+interactions_%s_%s' % (clf_name, tname)
                accs.append(results['tests'][key]['accuracy'])
            ax.bar(x_pos + i * width, accs, width, label=clf_name, alpha=0.8)
        ax.axhline(y=1.0/3, color='gray', ls='--', lw=1, label='random (1/3)')
        ax.set_xticks(x_pos + width * 1.5)
        ax.set_xticklabels([t.replace('S_', 'S_{') + '}' for t in hinge_targets], fontsize=9)
        ax.set_ylabel('CV Accuracy')
        ax.set_title('Accuracy by Classifier (base+interactions)')
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(0.5, max(accs) + 0.1))

        # Plot 2: Permutation test null distribution for best test
        ax = axes[0, 1]
        if best_test:
            best_r = results['tests'][best_test]
            _, _, null_dist = permutation_test_clf(
                classifiers[best_r['classifier']], X_aug,
                Y[best_r['target']].astype(int), n_perm=int(200))
            ax.hist(null_dist, bins=int(25), color='steelblue', alpha=0.7,
                    edgecolor='black', label='Null (permuted)')
            ax.axvline(best_r['accuracy'], color='red', lw=2, ls='--',
                       label='Observed acc=%.4f' % best_r['accuracy'])
            ax.set_xlabel('CV Accuracy')
            ax.set_ylabel('Count')
            ax.set_title('Permutation Test: %s' % best_test)
            ax.legend(fontsize=8)

        # Plot 3: Base vs base+interactions accuracy comparison
        ax = axes[1, 0]
        base_accs = []
        aug_accs = []
        labels = []
        for tname in hinge_targets:
            key_b = 'base_RF_100_%s' % tname
            key_a = 'base+interactions_RF_100_%s' % tname
            base_accs.append(results['tests'][key_b]['accuracy'])
            aug_accs.append(results['tests'][key_a]['accuracy'])
            labels.append(tname.replace('S_', 'S_{') + '}')

        x_pos = np.arange(len(labels))
        ax.bar(x_pos - 0.15, base_accs, 0.3, label='Base (111 feat)', alpha=0.8, color='steelblue')
        ax.bar(x_pos + 0.15, aug_accs, 0.3, label='+ Interactions', alpha=0.8, color='coral')
        ax.axhline(y=1.0/3, color='gray', ls='--', lw=1, label='random')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('RF_100 CV Accuracy')
        ax.set_title('Effect of Pairwise Interactions (RF)')
        ax.legend(fontsize=8)

        # Plot 4: Accuracy vs bit size
        ax = axes[1, 1]
        if bit_results:
            bits_list = sorted(bit_results.keys())
            for tname, marker, color in [('S_-3', 'o-', 'blue'), ('S_5', 's-', 'red')]:
                acc_key = 'acc_%s' % tname
                vals = [bit_results[b].get(acc_key, 0) for b in bits_list]
                ax.plot(bits_list, vals, marker, color=color, label=tname, markersize=6)
            ax.axhline(y=1.0/3, color='gray', ls='--', lw=1, label='random')
            ax.set_xlabel('Bit size')
            ax.set_ylabel('RF_100 CV Accuracy')
            ax.set_title('Accuracy vs N Bit Size')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'E14_nonlinear_ml.png')
        plt.savefig(plot_path, dpi=int(150))
        plt.close()
        print("Plot saved to %s" % plot_path, flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    results = run_experiment(
        bit_sizes=[16, 18, 20, 22],
        count_per_size=int(150),
        output_dir='../data'
    )
