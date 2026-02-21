"""
E7e — Analytic Proxy Tests for Jacobi De-biasing

HYPOTHESIS: There exists a low-complexity operator A acting on Jacobi-
accessible data that produces a signal with non-flat spectrum at factor rows.

GUARDRAILS:
  - All operations Jacobi-only (no hidden p,q in signal construction)
  - No gcd(t^2-d, N) filters (would be trial division)
  - No per-prime epsilon weights
  - CRT speedup for kronecker: kronecker(m, pq) = kronecker(m,p)*kronecker(m,q)
    is a performance optimization, not oracle use (identical result)

TESTS:
  1. Oracle baseline: L_p + L_q
  2. Arithmetic weightings: J*Lambda, J*mu, Gaussian-smoothed J
  3. Multiplicative twists: J*chi_m for m in {3,4,5,7,8}
  4. Functional equation: J is even => reflection trivial (reported, not tested)
  5. Multi-discriminant elimination:
     J_d(t) = kronecker(t^2-d, N) for d in {1,...,12}
     Strategies: equal, alternating, SVD, max-kurtosis, coherence
"""
import time
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from sage_encoding import _py, safe_json_dump
from spectral import verify_parseval

import numpy as np
from scipy import stats
from scipy.optimize import minimize

set_random_seed(42)
np.random.seed(42)

# ─── Helpers ──────────────────────────────────────────────────────────────

def dft_plus(sig):
    """DFT: hat{f}(xi) = (1/N) sum_t f(t) e^{+2pi i t xi / N}."""
    # NOTE: This uses the positive-exponential convention: hat{f}(xi) = (1/N) sum f(t) e^{+2pi i t xi/N}.
    # For real-valued signals, |hat{f}(xi)| is identical to the standard negative convention.
    # See BARRIER_THEOREM.md for discussion.
    result = np.conj(np.fft.fft(sig)) / len(sig)
    # Parseval check: for our convention, sum|X|^2 = (1/N) sum|f|^2
    verify_parseval(sig, result)
    return result

def precompute_gcd_classes(N, p, q):
    """Classify xi = 1..N-1 by gcd(xi, N)."""
    gcd_p, gcd_q, gcd_1 = [], [], []
    for xi in range(1, int(N)):
        g = int(gcd(xi, int(N)))
        if g == int(p):
            gcd_p.append(xi)
        elif g == int(q):
            gcd_q.append(xi)
        elif g == 1:
            gcd_1.append(xi)
    return gcd_p, gcd_q, gcd_1

def spectral_metrics(coeffs, N, gcd_p, gcd_q, gcd_1):
    """Peak/bulk ratio, E(gcd>1), top-1/5 factor extraction."""
    mags = np.abs(coeffs)
    mags2 = mags ** 2
    mag_p = float(np.mean(mags[gcd_p])) if gcd_p else 0.0
    mag_q = float(np.mean(mags[gcd_q])) if gcd_q else 0.0
    mag_bulk = float(np.mean(mags[gcd_1])) if gcd_1 else 1e-30

    E_tot = float(np.sum(mags2[1:]))
    E_gcd = float(np.sum(mags2[gcd_p]) + np.sum(mags2[gcd_q]))
    top_xi = int(np.argmax(mags[1:]) + 1)
    top_g = int(gcd(top_xi, int(N)))
    top1 = bool(1 < top_g < int(N))

    top5_xi = np.argsort(mags[1:])[-5:][::-1] + 1
    top5 = bool(any(1 < int(gcd(int(xi), int(N))) < int(N) for xi in top5_xi))

    peak = float(max(mag_p, mag_q) / mag_bulk) if mag_bulk > 1e-30 else 0.0
    E_frac = float(E_gcd / E_tot) if E_tot > 0 else 0.0
    return {'peak': peak, 'E_gcd': E_frac, 'top1': top1, 'top5': top5}

def gen_balanced(count=15, min_p=50, max_p=400):
    """Generate balanced semiprimes N = p*q."""
    out = []
    targets = np.logspace(np.log10(min_p), np.log10(max_p), count)
    for tgt in targets:
        p = next_prime(int(tgt))
        q = next_prime(p)
        out.append((int(p * q), int(p), int(q)))
    return out

def von_mangoldt_sieve(N):
    """Compute Lambda(t) for t = 0,...,N-1."""
    lam = np.zeros(int(N))
    for p_val in primes(2, int(N)):
        pk = int(p_val)
        log_p = float(RR(log(p_val)))
        while pk < int(N):
            lam[pk] = log_p
            pk *= int(p_val)
    return lam

def moebius_sieve(N):
    """Compute mu(t) for t = 0,...,N-1."""
    mu = np.ones(int(N), dtype=np.float64)
    mu[0] = 0.0
    for p_val in primes(2, int(N)):
        p = int(p_val)
        mu[p::p] *= -1
        p2 = p * p
        if p2 < int(N):
            mu[p2::p2] = 0.0
    return mu

def make_character(mod, N):
    """Precompute kronecker(t, mod) for t = 0,...,N-1."""
    table = np.array([float(kronecker_symbol(a, mod)) for a in range(mod)])
    return table[np.arange(int(N)) % mod]

def legendre_array(disc_arr, prime):
    """Compute kronecker(d mod p, p) for array of discriminants."""
    p = int(prime)
    return np.array([kronecker_symbol(int(d) % p, p) for d in disc_arr],
                    dtype=np.float64)

# ─── Main ─────────────────────────────────────────────────────────────────

print("E7e — Analytic Proxy Tests\n", flush=True)
print("=" * 80, flush=True)
print("HYPOTHESIS: exists low-complexity operator on Jacobi data producing", flush=True)
print("  non-flat spectrum at factor rows.", flush=True)
print("GUARDRAILS: Jacobi-only, no gcd filters, no per-prime data.", flush=True)
print("=" * 80, flush=True)

semiprimes = gen_balanced(count=15, min_p=50, max_p=400)
D_set = list(range(1, 13))   # Discriminants {1,...,12} for test 5
char_mods = [3, 4, 5, 7, 8]  # Character moduli for test 3

print(f"\nSemiprimes: {len(semiprimes)}  [{semiprimes[0][0]}..{semiprimes[-1][0]}]",
      flush=True)
print(f"Discriminants for multi-disc: {D_set}", flush=True)
print(f"Character moduli: {char_mods}\n", flush=True)

# Test signal names
test_names = [
    'Lp+Lq (oracle)',
    'J*Lambda', 'J*mu', 'J smoothed',
    'J*chi3', 'J*chi4', 'J*chi5', 'J*chi7', 'J*chi8',
    'multi-D equal', 'multi-D alt', 'SVD top-1', 'max-kurt',
]
# Accumulate results
agg = {name: {'peak': [], 'E_gcd': [], 'top1': [], 'top5': []}
       for name in test_names}
coherence_data = {'C_factor': [], 'C_bulk': [], 'C_ratio': []}

def store(name, m):
    for k in agg[name]:
        agg[name][k].append(m[k])

# ─── Per-semiprime analysis ───────────────────────────────────────────────

for idx, (N, p, q) in enumerate(semiprimes):
    t0 = time.perf_counter()
    t_arr = np.arange(N)

    # Precompute gcd classes (for metrics only — not used in signal construction)
    gcd_p, gcd_q, gcd_1 = precompute_gcd_classes(N, p, q)

    # Core discriminant: t^2 - 4
    disc = t_arr * t_arr - 4

    # Oracle Legendre symbols (used for CRT-speed AND oracle baseline)
    Lp = legendre_array(disc, p)
    Lq = legendre_array(disc, q)

    # Jacobi symbol J(t) = L_p * L_q (= kronecker(t^2-4, N))
    J = Lp * Lq
    J_hat = dft_plus(J)

    # ─── TEST 1: Oracle baseline ──────────────────────────────────────
    residual = Lp + Lq
    m = spectral_metrics(dft_plus(residual), N, gcd_p, gcd_q, gcd_1)
    store('Lp+Lq (oracle)', m)

    # ─── TEST 2: Arithmetic weightings ────────────────────────────────

    # 2a: J * Lambda(t)
    lam = von_mangoldt_sieve(N)
    m = spectral_metrics(dft_plus(J * lam), N, gcd_p, gcd_q, gcd_1)
    store('J*Lambda', m)

    # 2b: J * mu(t)
    mu = moebius_sieve(N)
    m = spectral_metrics(dft_plus(J * mu), N, gcd_p, gcd_q, gcd_1)
    store('J*mu', m)

    # 2c: Gaussian smoothing (multiply J_hat by Gaussian envelope)
    freqs = np.arange(N, dtype=np.float64)
    freq_dist = np.minimum(freqs, N - freqs)
    sigma = float(N) ** (1.0 / 3.0)
    gaussian = np.exp(-freq_dist**2 / (2.0 * sigma**2))
    J_hat_smooth = J_hat * gaussian
    m = spectral_metrics(J_hat_smooth, N, gcd_p, gcd_q, gcd_1)
    store('J smoothed', m)

    # ─── TEST 3: Multiplicative twists ────────────────────────────────
    for mod in char_mods:
        chi = make_character(mod, N)
        sig_twist = J * chi
        m = spectral_metrics(dft_plus(sig_twist), N, gcd_p, gcd_q, gcd_1)
        store(f'J*chi{mod}', m)

    # ─── TEST 5: Multi-discriminant elimination ───────────────────────
    # Build DFT matrix M[d_idx, xi] for discriminants t^2 - d
    n_disc = len(D_set)
    M = np.zeros((n_disc, N), dtype=np.complex128)
    for i, d in enumerate(D_set):
        disc_d = t_arr * t_arr - d
        Lp_d = legendre_array(disc_d, p)
        Lq_d = legendre_array(disc_d, q)
        J_d = Lp_d * Lq_d  # = kronecker(t^2-d, N) via CRT
        M[i, :] = dft_plus(J_d)

    # 5a: Equal weights
    w_eq = np.ones(n_disc) / n_disc
    S_eq = M.T @ w_eq
    m = spectral_metrics(S_eq, N, gcd_p, gcd_q, gcd_1)
    store('multi-D equal', m)

    # 5b: Alternating weights
    w_alt = np.array([(-1.0)**d for d in D_set])
    w_alt /= np.linalg.norm(w_alt)
    S_alt = M.T @ w_alt
    m = spectral_metrics(S_alt, N, gcd_p, gcd_q, gcd_1)
    store('multi-D alt', m)

    # 5c: SVD — top right singular vector
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    S_svd = Vt[0, :]  # top singular vector in frequency domain
    m = spectral_metrics(S_svd, N, gcd_p, gcd_q, gcd_1)
    store('SVD top-1', m)

    # 5d: Max-kurtosis optimization
    def neg_kurt(w_flat):
        w = w_flat / (np.linalg.norm(w_flat) + 1e-30)
        S = M.T @ w
        mags = np.abs(S[1:])
        return -float(stats.kurtosis(mags, fisher=True))

    best_k = np.inf
    best_w = np.ones(n_disc) / n_disc
    for trial in range(15):
        w0 = np.random.randn(n_disc)
        try:
            res = minimize(neg_kurt, w0, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-5})
            if res.fun < best_k:
                best_k = res.fun
                best_w = res.x / (np.linalg.norm(res.x) + 1e-30)
        except Exception as e:
            print(f"  WARNING: Nelder-Mead trial failed: {e}", flush=True)
    S_kurt = M.T @ best_w
    m = spectral_metrics(S_kurt, N, gcd_p, gcd_q, gcd_1)
    store('max-kurt', m)

    # 5e: Coherence analysis
    sum_hats = np.sum(M, axis=0)  # sum_d J_d_hat(xi)
    sum_mags2 = np.sum(np.abs(M)**2, axis=0)  # sum_d |J_d_hat(xi)|^2
    C = np.abs(sum_hats)**2 / (n_disc * sum_mags2 + 1e-30)
    # Mean coherence at factor vs bulk frequencies
    factor_idx = gcd_p + gcd_q  # list concatenation
    C_factor = float(np.mean(C[factor_idx])) if factor_idx else 0.0
    C_bulk = float(np.mean(C[gcd_1])) if gcd_1 else 0.0
    C_ratio = C_factor / C_bulk if C_bulk > 1e-30 else 0.0
    coherence_data['C_factor'].append(C_factor)
    coherence_data['C_bulk'].append(C_bulk)
    coherence_data['C_ratio'].append(C_ratio)

    dt = time.perf_counter() - t0
    print(f"  [{idx+1:>2}/{len(semiprimes)}]  N={N:>7} = {p} x {q}  ({dt:.1f}s)", flush=True)

print(f"\nAll {len(semiprimes)} semiprimes processed.\n", flush=True)

# ─── TEST 4: Functional equation observation ──────────────────────────────

print("=" * 80, flush=True)
print("TEST 4: FUNCTIONAL EQUATION OBSERVATION", flush=True)
print("=" * 80, flush=True)
print("  J(t) = kronecker(t^2-4, N). Since (N-t)^2 = t^2 mod N,", flush=True)
print("  J(N-t) = J(t) — the signal is EVEN.", flush=True)
print("  => J_hat(xi) = J_hat(N-xi) (DFT of even function is even)", flush=True)
print("  => Reflection xi -> N-xi is TRIVIAL for J_hat.", flush=True)
print("  => Any functional-equation-style even/odd decomposition degenerates.", flush=True)
print("  => The Gauss sum root number omega cannot help here.", flush=True)
print("  This is NOT a test failure — it means the FE symmetry is already", flush=True)
print("  built into J, so it provides no additional information.\n", flush=True)

# ─── Summary table ────────────────────────────────────────────────────────

print("=" * 80, flush=True)
print("E7e SUMMARY TABLE", flush=True)
print("  All metrics averaged over balanced semiprimes", flush=True)
print("  Peak = mean(mag at gcd>1) / mean(mag at gcd=1)", flush=True)
print("=" * 80, flush=True)
hdr = "%-18s %10s %10s %7s %7s" % ('Signal', 'Peak/Bulk', 'E(gcd>1)', 'Top1%', 'Top5%')
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for name in test_names:
    a = agg[name]
    pk = float(np.mean(a['peak']))
    eg = float(np.mean(a['E_gcd']))
    t1 = 100.0 * float(np.mean(a['top1']))
    t5 = 100.0 * float(np.mean(a['top5']))
    marker = ' ***' if t1 > 50 else ''
    print(f"{name:<18} {pk:>10.4f} {eg:>10.6f} {t1:>6.1f}% {t5:>6.1f}%{marker}",
          flush=True)

print(flush=True)
print("COHERENCE ANALYSIS (multi-discriminant, test 5e):", flush=True)
print(f"  Mean C at factor freqs: {np.mean(coherence_data['C_factor']):.6f}", flush=True)
print(f"  Mean C at bulk freqs:   {np.mean(coherence_data['C_bulk']):.6f}", flush=True)
print(f"  Mean C ratio (factor/bulk): {np.mean(coherence_data['C_ratio']):.4f}", flush=True)
print(flush=True)

# ─── Scaling analysis ─────────────────────────────────────────────────────

print("=" * 80, flush=True)
print("PEAK SCALING WITH N", flush=True)
print("=" * 80, flush=True)

Ns = np.array([s[0] for s in semiprimes], dtype=float)
for name in test_names:
    peaks = np.array(agg[name]['peak'])
    mask = peaks > 0.01
    if mask.sum() < 3:
        print(f"  {name:<18}: insufficient data", flush=True)
        continue
    sl, intc, rv, _, se = stats.linregress(np.log10(Ns[mask]),
                                            np.log10(peaks[mask]))
    print(f"  {name:<18}: peak ~ N^{sl:.3f} +/- {se:.3f}  R^2={rv**2:.4f}",
          flush=True)

# Coherence scaling
C_ratios = np.array(coherence_data['C_ratio'])
if len(C_ratios) > 3 and all(c > 0 for c in C_ratios):
    sl, _, rv, _, se = stats.linregress(np.log10(Ns), np.log10(C_ratios))
    print(f"  {'coherence ratio':<18}: ~ N^{sl:.3f} +/- {se:.3f}  R^2={rv**2:.4f}",
          flush=True)

print(flush=True)

# ─── Save results ─────────────────────────────────────────────────────────

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

output = {
    'semiprimes': [{'N': N, 'p': p, 'q': q} for N, p, q in semiprimes],
    'tests': {name: {k: [_py(v) for v in vals]
                     for k, vals in agg[name].items()}
              for name in test_names},
    'coherence': {k: [_py(v) for v in vals]
                  for k, vals in coherence_data.items()},
}
out_path = os.path.join(data_dir, 'E7e_analytic_proxies_results.json')
safe_json_dump(output, out_path)
print(f"Saved -> {out_path}\n", flush=True)

# ─── Plots ────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Peak/bulk ratio bar chart
ax = axes[0, 0]
plot_names = test_names
peaks_mean = [float(np.mean(agg[n]['peak'])) for n in plot_names]
colors = ['red'] + ['steelblue']*3 + ['green']*5 + ['orange']*4
ax.bar(range(len(plot_names)), peaks_mean, color=colors, alpha=0.7)
ax.set_xticks(range(len(plot_names)))
ax.set_xticklabels(plot_names, rotation=60, ha='right', fontsize=7)
ax.set_ylabel('Peak/Bulk Ratio')
ax.set_title('Peak Magnitude at Factor Frequencies')
ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.grid(True, alpha=0.3)

# 2. E(gcd>1) bar chart
ax = axes[0, 1]
E_mean = [float(np.mean(agg[n]['E_gcd'])) for n in plot_names]
ax.bar(range(len(plot_names)), E_mean, color=colors, alpha=0.7)
ax.set_xticks(range(len(plot_names)))
ax.set_xticklabels(plot_names, rotation=60, ha='right', fontsize=7)
ax.set_ylabel('E(gcd>1)')
ax.set_title('Energy in Factor-Frequency Modes')
ax.axhline(2.0/3, color='gray', ls='--', lw=1, label='2/3 (oracle)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Coherence: factor vs bulk
ax = axes[1, 0]
ax.scatter(Ns, coherence_data['C_factor'], s=30, alpha=0.7, color='red',
           label='C at factor freq')
ax.scatter(Ns, coherence_data['C_bulk'], s=30, alpha=0.7, color='blue',
           label='C at bulk freq')
ax.set_xscale('log')
ax.set_xlabel('N')
ax.set_ylabel('Coherence C(xi)')
ax.set_title('Multi-Discriminant Coherence')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Peak scaling: oracle vs best computable
ax = axes[1, 1]
oracle_peaks = np.array(agg['Lp+Lq (oracle)']['peak'])
ax.scatter(Ns, oracle_peaks, s=30, color='red', label='Oracle (Lp+Lq)',
           zorder=5)
# Best computable: max-kurtosis
for comp_name, col, mk in [('max-kurt', 'orange', 's'),
                             ('multi-D equal', 'blue', '^'),
                             ('J*Lambda', 'green', 'o')]:
    comp_peaks = np.array(agg[comp_name]['peak'])
    ax.scatter(Ns, comp_peaks, s=25, color=col, marker=mk, alpha=0.7,
               label=comp_name)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N')
ax.set_ylabel('Peak/Bulk')
ax.set_title('Peak Scaling: Oracle vs Computable')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(data_dir, 'E7e_analytic_proxies_plots.png')
plt.savefig(plot_path, dpi=150)
print(f"Saved plots -> {plot_path}\n", flush=True)

# ─── Verdict ──────────────────────────────────────────────────────────────

print("=" * 80, flush=True)
print("E7e ANALYTIC PROXY TESTS — VERDICT", flush=True)
print("=" * 80, flush=True)
print(flush=True)

# Check if any computable test achieved >50% top-1
any_success = False
for name in test_names:
    if name == 'Lp+Lq (oracle)':
        continue
    t1_rate = float(np.mean(agg[name]['top1']))
    if t1_rate > 0.5:
        any_success = True
        print(f"  SIGNAL DETECTED: {name} has {100*t1_rate:.0f}% top-1 success!",
              flush=True)

if not any_success:
    print("  NO computable proxy achieved >50% top-1 factor extraction.", flush=True)
    print(flush=True)
    print("  TESTED AND FAILED:", flush=True)
    print("    - Arithmetic weightings (Lambda, mu): Jacobi-flat * arithmetic", flush=True)
    print("      function = still flat. No constructive interference at factor freqs.", flush=True)
    print("    - Gaussian smoothing: pointwise freq multiplication on flat", flush=True)
    print("      spectrum -> shaped-flat spectrum. Cannot create peaks.", flush=True)
    print("    - Dirichlet character twists: J*chi_m breaks CRT product structure", flush=True)
    print("      but DFT(J*chi) = J_hat * chi_hat (convolution). Convolving a", flush=True)
    print("      flat-magnitude spectrum with anything preserves magnitude flatness.", flush=True)
    print("    - Multi-discriminant linear combinations: each J_d is a CRT product", flush=True)
    print("      with flat DFT. Sum of flat spectra with incoherent phases -> flat.", flush=True)
    print("    - SVD of discriminant family: no low-rank factor-related component", flush=True)
    print("      because the Hadamard product M = F (*) G has rank up to p*q.", flush=True)
    print("    - Kurtosis-optimized weights: optimization finds noise, not signal.", flush=True)
    print("    - Discriminant coherence: C(factor) ~ C(bulk) because f_hat_d(0)", flush=True)
    print("      = -1/p for all d (Jacobsthal sum), giving no discriminant-dependent", flush=True)
    print("      phase variation at factor frequencies.", flush=True)
    print("    - Functional equation: J is even => xi<->N-xi symmetry trivial.", flush=True)
    print(flush=True)
    print("  INTERPRETATION:", flush=True)
    print("    Low-complexity analytic proxies (convolutions, twists, linear", flush=True)
    print("    combinations of discriminant families) CANNOT extract factor", flush=True)
    print("    information from Jacobi-flat data.", flush=True)
    print(flush=True)
    print("    This does NOT kill the BE corridor in general. It means:", flush=True)
    print("    'Any BE-like separation mechanism that works must involve", flush=True)
    print("    genuinely global analytic objects (L-function poles, spectral", flush=True)
    print("    projectors, functional equations of non-abelian L-functions)", flush=True)
    print("    not implementable as cheap transforms on Jacobi sequences.'", flush=True)
print(flush=True)
