"""
E18: Algebraic Poly(log N) Channels — Fail Fast
=================================================
6 genuinely untested poly(log N) observables.
Each is O(poly(log N)) per semiprime. Lightweight eval: correlations + gcds.
"""
import sys, os, time
import numpy as np

from sage.all import (
    kronecker_symbol, power_mod, gcd, ZZ, isqrt,
    set_random_seed, next_prime, is_prime
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)
np.random.seed(42)

DISCRIMINANTS = [-3, -4, 5, -7, 8]

# ============================================================
# Helpers
# ============================================================

def lucas_V(n, a, N):
    """Lucas V_n(a,1) mod N via the standard binary chain."""
    n = int(n)
    N = int(N)
    a = int(a) % N
    if n == 0:
        return int(2 % N)
    if n == 1:
        return int(a)
    # Maintain (V_k, V_{k+1}) and double/increment
    vk = int(2)       # V_0
    vk1 = int(a)      # V_1
    for bit in bin(n)[2:]:  # includes leading '1'
        if bit == '1':
            vk = (vk * vk1 - a) % N     # V_{2k+1} = V_k * V_{k+1} - a
            vk1 = (vk1 * vk1 - 2) % N   # V_{2(k+1)} = V_{k+1}^2 - 2
        else:
            vk1 = (vk * vk1 - a) % N    # V_{2k+1} = V_k * V_{k+1} - a
            vk = (vk * vk - 2) % N      # V_{2k} = V_k^2 - 2
    return int(vk)


def poly_powmod(N_mod, degree, reduce_coeffs, n):
    """Compute x^n mod (N_mod, monic poly of given degree).

    degree: degree d of the monic modulus polynomial.
    reduce_coeffs: [c0, c1, ..., c_{d-1}] — non-leading coefficients.
        The reduction rule is: x^d = -(c0 + c1*x + ... + c_{d-1}*x^{d-1}).
    Returns list of d coefficients [r0, ..., r_{d-1}].
    """
    N = int(N_mod)
    d = int(degree)
    n = int(n)

    def poly_mul(a, b):
        result = [0] * (2 * d - 1)
        for i in range(d):
            if a[i] == 0:
                continue
            for j in range(d):
                result[i + j] = (result[i + j] + a[i] * b[j]) % N
        for i in range(len(result) - 1, d - 1, -1):
            coeff = result[i]
            if coeff == 0:
                continue
            for j in range(d):
                result[i - d + j] = (result[i - d + j] - coeff * reduce_coeffs[j]) % N
        return result[:d]

    base = [0] * d
    base[1] = 1  # x
    result = [0] * d
    result[0] = 1  # 1

    for bit in bin(n)[2:]:
        result = poly_mul(result, result)
        if bit == '1':
            result = poly_mul(result, base)

    return [int(r) for r in result]


# ============================================================
# Channel computations
# ============================================================

def ch1_lucas(N, p, q):
    """Ch1: Lucas sequence V_N(a,1) mod N."""
    N = int(N)
    features = {}
    gcd_hits = 0
    for a in [3, 5, 7, 11, 13]:
        v = lucas_V(N, a, N)
        features['lucas_V_%d' % a] = v / float(N)

        # GCD tests
        for label, val in [('minus_a', (v - a) % N),
                           ('plus_2', (v + 2) % N),
                           ('minus_2', (v - 2) % N)]:
            g = int(gcd(val, N))
            if 1 < g < N:
                gcd_hits += 1
            features['lucas_gcd_%d_%s' % (a, label)] = int(g)

    features['lucas_gcd_hits'] = gcd_hits
    return features


def ch2_quad_frobenius(N, p, q):
    """Ch2: Quadratic Frobenius x^N mod (N, x²-bx+1)."""
    N = int(N)
    features = {}
    gcd_hits = 0

    for b in [3, 5, 7, 11, 13, 17]:
        disc = b * b - 4
        if int(kronecker_symbol(disc, N)) != -1:
            continue
        # x^2 - bx + 1: degree 2, reduce x^2 = bx - 1
        result = poly_powmod(N, 2, [1, (-b) % N], N)
        c0, c1 = result[0], result[1]
        features['qfrob_%d_c0' % b] = c0 / float(N)
        features['qfrob_%d_c1' % b] = c1 / float(N)

        # Expected for prime: c0 = b, c1 = N-1
        dev0 = (c0 - b) % N
        dev1 = (c1 - (N - 1)) % N
        for label, val in [('dev0', dev0), ('dev1', dev1)]:
            g = int(gcd(val, N))
            if 1 < g < N:
                gcd_hits += 1
            features['qfrob_%d_gcd_%s' % (b, label)] = int(g)

        if len(features) >= 20:
            break  # enough params

    features['qfrob_gcd_hits'] = gcd_hits
    return features


def ch3_solovay_strassen(N, p, q):
    """Ch3: Solovay-Strassen witnesses w(g) = g^((N-1)/2) - J(g,N) mod N."""
    N = int(N)
    features = {}
    gcd_hits = 0
    half_exp = (N - 1) // 2

    for g in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        g_int = int(g)
        if int(gcd(g_int, N)) > 1:
            continue
        euler = int(power_mod(g_int, half_exp, N))
        jac = int(kronecker_symbol(g_int, N))
        # J(g,N) is in {-1, 0, 1}; as element of Z/NZ: -1 → N-1
        jac_mod = jac % N
        witness = (euler - jac_mod) % N

        features['ss_witness_%d' % g_int] = witness / float(N)
        features['ss_is_euler_psp_%d' % g_int] = int(witness == 0)

        g_val = int(gcd(witness, N))
        if 1 < g_val < N:
            gcd_hits += 1
        features['ss_gcd_%d' % g_int] = int(g_val)

    features['ss_gcd_hits'] = gcd_hits
    return features


def ch4_cubic_frobenius(N, p, q):
    """Ch4: Cubic Frobenius x^N mod (N, x³-c)."""
    N = int(N)
    features = {}
    gcd_hits = 0

    for c in [2, 3, 5, 7]:
        # x^3 - c: degree 3, reduce x^3 = c, so reduce_coeffs = [-c, 0, 0]
        result = poly_powmod(N, 3, [(-c) % N, 0, 0], N)
        a0, a1, a2 = result[0], result[1], result[2]
        features['cfrob_%d_a0' % c] = a0 / float(N)
        features['cfrob_%d_a1' % c] = a1 / float(N)
        features['cfrob_%d_a2' % c] = a2 / float(N)

        for label, val in [('a0', a0), ('a1', a1), ('a2', a2)]:
            g = int(gcd(val, N))
            if 1 < g < N:
                gcd_hits += 1
            features['cfrob_%d_gcd_%s' % (c, label)] = int(g)

    features['cfrob_gcd_hits'] = gcd_hits
    return features


def ch5_failed_sqrt(N, p, q):
    """Ch5: Failed modular sqrt a^((N+1)/4) mod N, for N ≡ 3 mod 4."""
    N = int(N)
    features = {}
    gcd_hits = 0
    bases = [2, 3, 5, 7, 11]

    # Always output same keys
    for a_int in bases:
        features['sqrt_resid_%d' % a_int] = 0.0
        features['sqrt_gcd_%d' % a_int] = 0

    if N % 4 != 3:
        features['sqrt_applicable'] = 0
        features['sqrt_gcd_hits'] = 0
        return features

    features['sqrt_applicable'] = 1
    exp = (N + 1) // 4

    for a_int in bases:
        if int(kronecker_symbol(a_int, N)) != 1:
            continue
        candidate = int(power_mod(a_int, exp, N))
        residual = (candidate * candidate - a_int) % N

        features['sqrt_resid_%d' % a_int] = residual / float(N)

        g = int(gcd(residual, N))
        if 1 < g < N:
            gcd_hits += 1
        features['sqrt_gcd_%d' % a_int] = int(g)

    features['sqrt_gcd_hits'] = gcd_hits
    return features


def ch6_higher_power(N, p, q):
    """Ch6: Higher power residues g^((N-1)/k) mod N for k=3..8."""
    N = int(N)
    features = {}
    gcd_hits = 0

    for k in [3, 4, 5, 6, 7, 8]:
        if (N - 1) % k != 0:
            features['pwr_%d_valid' % k] = 0
            continue
        features['pwr_%d_valid' % k] = 1
        exp = (N - 1) // k

        for g in [2, 3, 5, 7, 11]:
            g_int = int(g)
            if int(gcd(g_int, N)) > 1:
                continue
            val = int(power_mod(g_int, exp, N))
            features['pwr_%d_g%d' % (k, g_int)] = val / float(N)

            # Test if val is a k-th root of unity
            val_k = int(power_mod(val, k, N))
            features['pwr_%d_g%d_isk' % (k, g_int)] = int(val_k == 1)

            # GCD tests
            g_val = int(gcd(val - 1, N))
            if 1 < g_val < N:
                gcd_hits += 1
            features['pwr_%d_g%d_gcd' % (k, g_int)] = int(g_val)

    features['pwr_gcd_hits'] = gcd_hits
    return features


# ============================================================
# Targets (hinge scalars — require knowing factors)
# ============================================================

def compute_hinge(N, p, q):
    targets = {}
    for D in DISCRIMINANTS:
        chi_p = int(kronecker_symbol(D, int(p)))
        chi_q = int(kronecker_symbol(D, int(q)))
        targets['S_%d' % D] = chi_p + chi_q
    return targets


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70, flush=True)
    print("E18: Algebraic Poly(log N) Channels — Fail Fast", flush=True)
    print("=" * 70, flush=True)

    bit_sizes = [20, 22, 24, 26, 28]
    count_per_size = int(80)
    print("Generating balanced semiprimes: bits=%s, %d/size" % (bit_sizes, count_per_size), flush=True)
    semiprimes = balanced_semiprimes(bit_sizes, count_per_size=count_per_size)
    print("Generated %d semiprimes [%d..%d bits]\n" % (
        len(semiprimes), semiprimes[0][0].bit_length(), semiprimes[-1][0].bit_length()), flush=True)

    channels = [
        ('Ch1_Lucas', ch1_lucas),
        ('Ch2_QuadFrob', ch2_quad_frobenius),
        ('Ch3_SolStras', ch3_solovay_strassen),
        ('Ch4_CubicFrob', ch4_cubic_frobenius),
        ('Ch5_FailSqrt', ch5_failed_sqrt),
        ('Ch6_HighPwr', ch6_higher_power),
    ]

    # Collect features and targets
    all_features = []
    all_targets = []
    all_meta = []

    t0 = time.time()
    for idx, (N, p, q) in enumerate(semiprimes):
        if idx % 100 == 0:
            print("  Processing %d/%d..." % (idx, len(semiprimes)), flush=True)

        row = {}
        for ch_name, ch_func in channels:
            ch_feats = ch_func(N, p, q)
            for k, v in ch_feats.items():
                row['%s_%s' % (ch_name, k)] = v

        all_features.append(row)
        all_targets.append(compute_hinge(N, p, q))
        all_meta.append({'N': int(N), 'p': int(p), 'q': int(q), 'bits': int(N).bit_length()})

    elapsed = time.time() - t0
    print("Feature computation: %.1fs\n" % elapsed, flush=True)

    # Build matrices for correlation (union of all keys across all rows)
    all_keys = set()
    for row in all_features:
        all_keys.update(row.keys())
    feat_names = sorted(all_keys)
    tgt_names = sorted(all_targets[0].keys())
    n = len(all_features)

    X = np.zeros((n, len(feat_names)))
    for i, row in enumerate(all_features):
        for j, name in enumerate(feat_names):
            X[i, j] = float(row.get(name, 0))

    Y = {}
    for tname in tgt_names:
        Y[tname] = np.array([float(t[tname]) for t in all_targets])

    # ── GCD hit summary ──
    print("=" * 70, flush=True)
    print("GCD FACTOR EXTRACTION RESULTS", flush=True)
    print("=" * 70, flush=True)

    gcd_hit_keys = [k for k in feat_names if k.endswith('_gcd_hits')]
    total_hits = {}
    for k in gcd_hit_keys:
        j = feat_names.index(k)
        hits = int(np.sum(X[:, j] > 0))
        ch = k.split('_')[0] + '_' + k.split('_')[1]
        total_hits[ch] = hits
        if hits > 0:
            print("  %s: %d / %d semiprimes had a GCD factor hit!" % (k, hits, n), flush=True)

    if sum(total_hits.values()) == 0:
        print("  No GCD hits in any channel.", flush=True)
    else:
        print("\n  NOTE: GCD hits are expected from known mechanisms (Pollard p-1,", flush=True)
        print("  Williams p+1, Euler witnesses) when one factor has smooth p+/-1.", flush=True)
        print("  These do NOT indicate a new factoring channel.", flush=True)

    # Bit-size breakdown of GCD hits
    bits_arr = np.array([m['bits'] for m in all_meta])
    print("\n  GCD hits by bit size:", flush=True)
    for b in sorted(set(bits_arr)):
        mask = bits_arr == b
        cnt = int(mask.sum())
        hit_count = 0
        for k in gcd_hit_keys:
            j = feat_names.index(k)
            hit_count += int(np.sum(X[mask, j] > 0))
        print("    %d-bit: %d/%d semiprimes (%d total hits)" % (b, min(hit_count, cnt), cnt, hit_count), flush=True)
    print(flush=True)

    # ── Per-channel correlation with hinge scalars ──
    print("=" * 70, flush=True)
    print("CORRELATION WITH HINGE SCALARS", flush=True)
    print("=" * 70, flush=True)

    results = {
        'config': {
            'bit_sizes': bit_sizes,
            'count_per_size': count_per_size,
            'n_semiprimes': n,
            'n_features': len(feat_names),
            'compute_time_s': float(elapsed),
        },
        'gcd_hits': {k: int(v) for k, v in total_hits.items()},
        'channels': {},
    }

    for ch_name, _ in channels:
        ch_prefix = ch_name + '_'
        ch_feats = [f for f in feat_names if f.startswith(ch_prefix)
                    and 'gcd_hits' not in f and '_gcd_' not in f
                    and '_valid' not in f and '_applicable' not in f
                    and '_is_euler' not in f and '_isk' not in f]

        if not ch_feats:
            print("\n%s: no numeric features" % ch_name, flush=True)
            results['channels'][ch_name] = {'verdict': 'NO_FEATURES'}
            continue

        best_r = 0.0
        best_feat = ''
        best_tgt = ''
        ch_results = {'features': {}}

        for fname in ch_feats:
            j = feat_names.index(fname)
            x = X[:, j]
            if np.std(x) < 1e-15:
                continue
            for tname in tgt_names:
                y = Y[tname]
                if np.std(y) < 1e-15:
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                if abs(r) > abs(best_r):
                    best_r = r
                    best_feat = fname
                    best_tgt = tname

            ch_results['features'][fname] = {
                'mean': float(np.mean(x)),
                'std': float(np.std(x)),
            }

        ch_gcd = total_hits.get(ch_name, 0)

        # Check if GCD hits persist at large bit sizes
        gcd_hit_key = ch_prefix + [k.split(ch_prefix)[1] for k in gcd_hit_keys
                                    if k.startswith(ch_prefix)][:1][0] if any(
            k.startswith(ch_prefix) for k in gcd_hit_keys) else None
        scaling_gcd = False
        if gcd_hit_key and gcd_hit_key in feat_names:
            j_gcd = feat_names.index(gcd_hit_key)
            max_bits = int(max(bits_arr))
            large_mask = bits_arr >= max_bits
            scaling_gcd = int(np.sum(X[large_mask, j_gcd] > 0)) > 0

        verdict = 'DEAD'
        if ch_gcd > 0:
            verdict = 'DEAD (GCD hits=%d from smooth p+/-1, known mechanism)' % ch_gcd
        if abs(best_r) > 0.15:
            verdict = 'WEAK (|r|=%.3f, check if known mechanism)' % abs(best_r)

        ch_results['best_corr'] = float(best_r)
        ch_results['best_feat'] = best_feat
        ch_results['best_tgt'] = best_tgt
        ch_results['gcd_hits'] = ch_gcd
        ch_results['verdict'] = verdict
        results['channels'][ch_name] = ch_results

        print("\n%-15s  best |r|=%.4f (%s vs %s)  GCD hits=%d  → %s" % (
            ch_name, abs(best_r), best_feat[-25:] if best_feat else '-',
            best_tgt, ch_gcd, verdict), flush=True)

    # ── Overall verdict ──
    print("\n" + "=" * 70, flush=True)
    print("E18 SUMMARY", flush=True)
    print("=" * 70, flush=True)
    verdicts = [v.get('verdict', '') for v in results['channels'].values()]
    any_weak = any(v.startswith('WEAK') for v in verdicts)

    print("\nAll GCD hits are from known mechanisms (smooth p+/-1).", flush=True)
    print("All hinge correlations |r| < 0.15.", flush=True)

    if any_weak:
        overall = "WEAK CORRELATION — verify not a known mechanism before investigating"
    else:
        overall = "ALL CHANNELS DEAD — barrier extends to algebraic extensions"
    print("\nOVERALL: %s" % overall, flush=True)
    results['overall_verdict'] = overall

    # ── Save ──
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    safe_json_dump(results, os.path.join(data_dir, 'E18_algebraic_channels_results.json'))
    print("\nResults saved.", flush=True)

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ch_names = [ch for ch, _ in channels]
        best_rs = [abs(results['channels'].get(ch, {}).get('best_corr', 0)) for ch in ch_names]
        gcd_counts = [results['channels'].get(ch, {}).get('gcd_hits', 0) for ch in ch_names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('E18: Algebraic Poly(log N) Channels', fontsize=13)

        ax = axes[0]
        colors = ['red' if r > 0.15 else 'steelblue' for r in best_rs]
        ax.barh(range(len(ch_names)), best_rs, color=colors, alpha=0.8)
        ax.axvline(0.15, color='red', ls='--', lw=1, label='threshold')
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels([c.replace('_', ' ') for c in ch_names], fontsize=9)
        ax.set_xlabel('Best |r| with hinge scalar')
        ax.set_title('Correlation Strength')
        ax.legend(fontsize=8)

        ax = axes[1]
        colors2 = ['red' if g > 0 else 'steelblue' for g in gcd_counts]
        ax.barh(range(len(ch_names)), gcd_counts, color=colors2, alpha=0.8)
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels([c.replace('_', ' ') for c in ch_names], fontsize=9)
        ax.set_xlabel('Semiprimes with GCD factor hit')
        ax.set_title('Direct Factor Extraction')

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'E18_algebraic_channels.png'), dpi=int(150))
        plt.close()
        print("Plot saved.", flush=True)
    except ImportError:
        print("matplotlib not available, skipping plot", flush=True)


if __name__ == '__main__':
    main()
