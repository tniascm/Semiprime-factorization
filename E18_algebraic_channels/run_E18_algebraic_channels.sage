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
    """Compute V_n(a, 1) mod N via binary ladder. O(log n) mults."""
    n = int(n)
    N = int(N)
    a = int(a)
    if n == 0:
        return 2
    if n == 1:
        return a % N
    V_k = a % N
    V_km1 = 2
    bits = bin(n)[3:]  # skip '0b1', process remaining bits
    for bit in bits:
        if bit == '1':
            V_km1 = (V_k * V_km1 - a) % N
            V_k = (V_k * V_k - 2) % N
        else:
            V_km1 = (V_k * V_km1 - a) % N
            # swap: V_k stays, V_km1 updated
            V_k, V_km1 = (V_k * V_k - 2) % N, V_km1
    # Wait, the standard Lucas chain is:
    # To compute V_n: start with V_1=a, V_0=2
    # For bit=1: V_{2k+1} = V_{k+1}*V_k - a, V_{k+1} = V_{k+1}^2 - 2
    # For bit=0: V_{2k} = V_k^2 - 2, V_{k+1} = V_{k+1}*V_k - a
    # Let me redo this correctly.
    return _lucas_V_correct(n, a, N)


def _lucas_V_correct(n, a, N):
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


def poly_powmod(N_mod, poly_mod, n):
    """Compute x^n mod (N_mod, poly_mod) in (Z/NZ)[x]/(poly_mod).

    poly_mod: list of coefficients [c0, c1, ..., cd] for c0 + c1*x + ... + cd*x^d.
    Returns result as list of d coefficients [r0, r1, ..., r_{d-1}].
    """
    N = int(N_mod)
    d = len(poly_mod) - 1  # degree of modulus
    n = int(n)

    def poly_mul(a, b):
        """Multiply two polynomials mod (N, poly_mod)."""
        # a, b are lists of length d
        result = [0] * (2 * d - 1)
        for i in range(len(a)):
            for j in range(len(b)):
                result[i + j] = (result[i + j] + a[i] * b[j]) % N
        # Reduce mod poly_mod (leading coeff assumed 1)
        for i in range(len(result) - 1, d - 1, -1):
            coeff = result[i]
            if coeff == 0:
                continue
            for j in range(d):
                result[i - d + j] = (result[i - d + j] - coeff * poly_mod[j]) % N
        return result[:d]

    # x = [0, 1, 0, ...] (the polynomial x)
    base = [0] * d
    if d > 1:
        base[1] = 1
    else:
        base[0] = 1  # degenerate

    # Start: result = 1 (constant polynomial)
    result = [0] * d
    result[0] = 1

    # Make base = x
    base = [0] * d
    base[1 % d] = 1

    # Binary square-and-multiply
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
        v = _lucas_V_correct(N, a, N)
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
        # poly_mod for x² - bx + 1: coefficients [1, -b, 1] (constant, x, x²)
        # But our poly_powmod expects leading coeff = 1, so poly_mod = [1, -b % N]
        # and degree d = 2
        poly_mod = [1, (-b) % N]  # x² - bx + 1 has coeffs [1, -b, 1]; leading=1
        # Actually poly_mod should be [c0, c1] where x^2 = c0 + c1*x + ... nah.
        # Let me fix: x^2 ≡ -c0 - c1*x mod (x^2 + c1*x + c0)
        # For x^2 - bx + 1: x^2 = bx - 1, so when reducing x^2, replace with bx - 1
        # In poly_mul, poly_mod = [1, -b] means x^2 = -1 + bx? No...
        # Let me re-think. poly_mod = [1, -b, 1] means 1 - b*x + x^2.
        # When we reduce x^2, we get x^2 = b*x - 1.
        # So poly_mod for reduction: [(-1) % N, b % N] (the replacement for x^d)
        # Actually in my poly_mul, the reduction line does:
        #   result[i-d+j] -= coeff * poly_mod[j]
        # So poly_mod should be the coefficients of the modulus EXCLUDING the leading term.
        # For x^2 - bx + 1 (monic, leading coeff 1):
        #   poly_mod = [1, (-b) % N] = constant and x coefficient
        # Then x^2 = -poly_mod[0] - poly_mod[1]*x = -1 + bx. Wait no:
        # result[i-d+j] -= coeff * poly_mod[j], so for i=2, d=2:
        #   result[0] -= coeff * poly_mod[0] (= coeff * 1)
        #   result[1] -= coeff * poly_mod[1] (= coeff * (-b))
        # So x^2 is being replaced by: -(1)*1 + -((-b))*x = -1 + bx
        # That means x^2 - bx + 1 = 0 => x^2 = bx - 1. YES correct!

        result = poly_powmod(N, [1, (-b) % N], N)
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
        # x^3 - c: monic degree 3, poly_mod = [(-c) % N, 0, 0]
        # x^3 = c, so poly_mod = [(-c) % N, 0, 0] means:
        # result[i-3+0] -= coeff * (-c) => result[i-3] += coeff*c
        # result[i-3+1] -= coeff * 0 = 0
        # result[i-3+2] -= coeff * 0 = 0
        # So x^3 -> c. That's x^3 = c. Correct for x^3 - c = 0.
        # Wait: x^3 - c = 0 means x^3 = c.
        # My reduction: result[i-d+j] -= coeff * poly_mod[j]
        # If poly_mod = [(-c)%N, 0, 0], then for i=3, d=3:
        #   result[0] -= coeff * ((-c)%N) = result[0] + coeff*c
        # That gives x^3 → c. Correct!

        result = poly_powmod(N, [(-c) % N, 0, 0], N)
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

    if N % 4 != 3:
        features['sqrt_applicable'] = 0
        features['sqrt_gcd_hits'] = 0
        return features

    features['sqrt_applicable'] = 1
    exp = (N + 1) // 4

    for a in [2, 3, 5, 7, 11, 13, 17, 19]:
        a_int = int(a)
        if int(kronecker_symbol(a_int, N)) != 1:
            continue
        candidate = int(power_mod(a_int, exp, N))
        residual = (candidate * candidate - a_int) % N

        features['sqrt_resid_%d' % a_int] = residual / float(N)

        g = int(gcd(residual, N))
        if 1 < g < N:
            gcd_hits += 1
        features['sqrt_gcd_%d' % a_int] = int(g)

        if len([k for k in features if k.startswith('sqrt_resid')]) >= 5:
            break

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

    # Build matrices for correlation
    feat_names = sorted(all_features[0].keys())
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
        verdict = 'DEAD'
        if ch_gcd > 0:
            verdict = 'SIGNAL (GCD hits: %d)' % ch_gcd
        elif abs(best_r) > 0.15:
            verdict = 'SIGNAL (|r|=%.3f)' % abs(best_r)

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
    any_signal = any(v.get('verdict', '').startswith('SIGNAL')
                     for v in results['channels'].values())
    if any_signal:
        overall = "SIGNAL DETECTED in at least one channel — investigate further"
    else:
        overall = "ALL CHANNELS DEAD — barrier extends to algebraic extensions"
    print("OVERALL: %s" % overall, flush=True)
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
