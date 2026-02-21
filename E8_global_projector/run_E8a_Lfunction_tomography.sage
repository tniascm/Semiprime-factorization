"""
E8a — Twisted GL(2) L-function Tomography

GOAL: Test whether global analytic objects (L-function values, zeros)
carry factor information beyond what Jacobi-accessible local data provides.

SETUP:
  Fixed form: f = Delta (Ramanujan, weight 12, level 1)
  Twist:      chi_N = kronecker(., N) (Jacobi character, computable)
  L-function: L(s, Delta x chi_N) = sum_{n>=1} tau(n)*chi_N(n)/n^s

KEY QUESTION: Does L(s, Delta x chi_N) at various s-values distinguish
  different factorizations N = pq from N = rs of similar-sized semiprimes?

STRUCTURAL OBSERVATION:
  chi_N(p) = 0 for p|N, so Euler factors at factor primes are trivial:
    L_p(s) = (1 + p^{11-2s})^{-1}
  The twist "forgets" individual factors locally. But the global zeros/values
  depend on ALL Euler factors jointly, and the "holes" at p,q affect the
  zero distribution through analytic continuation — NOT decomposable as
  "p-contribution + q-contribution."

TESTS:
  A. Evaluate L(s) at many s-values on/near the critical line (s=6+it)
  B. Central value L(6): does it distinguish factorizations?
  C. Root number epsilon: computable from N alone, any factor signal?
  D. Approximate zero counting: N(T) deviations from smooth prediction
  E. Factor candidate scoring: for primes m, remove Euler factor at m
     and test if the "corrected" L-value has anomalous properties when m|N

APPROXIMATE FUNCTIONAL EQUATION:
  Lambda(s) = (N/2pi)^s * Gamma(s + 11/2) * L(s)
  Lambda(s) = epsilon * Lambda(12 - s)
  L(s) = sum_n tau(n)*chi_N(n)/n^s * V(2*pi*n/N, s)
        + epsilon_s * sum_n tau(n)*chi_N(n)/n^{12-s} * V(2*pi*n/N, 12-s)
  where V is a smooth incomplete-gamma cutoff and epsilon_s includes
  the gamma ratio.
"""
import time
import sys
import json
import os

import numpy as np
from scipy import stats, special

set_random_seed(42)
np.random.seed(42)

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from sage_encoding import _py, _py_dict, safe_json_dump

# ─── Helpers ──────────────────────────────────────────────────────────────

def gen_balanced(count=12, min_p=50, max_p=350):
    """Generate balanced semiprimes N = p*q."""
    out = []
    targets = np.logspace(np.log10(min_p), np.log10(max_p), count)
    for tgt in targets:
        p = next_prime(int(tgt))
        q = next_prime(p)
        out.append((int(p * q), int(p), int(q)))
    return out

def gen_confusable_pairs(count=6, target_size=10000):
    """Generate pairs of semiprimes with similar N but different factorizations."""
    pairs = []
    # For each target, find two semiprimes close to target_size
    targets = np.logspace(np.log10(3000), np.log10(50000), count)
    for tgt in targets:
        sqrt_t = int(np.sqrt(tgt))
        p1 = next_prime(sqrt_t)
        q1 = next_prime(p1)
        N1 = int(p1 * q1)
        # Find a second semiprime close to N1 with different factors
        p2 = next_prime(int(sqrt_t * 0.85))
        q2 = next_prime(int(tgt / int(p2)))
        N2 = int(p2 * q2)
        if N2 > 0 and p2 != p1 and q2 != q1:
            pairs.append(((N1, int(p1), int(q1)), (N2, int(p2), int(q2))))
    return pairs

# ─── Tau function ─────────────────────────────────────────────────────────

def compute_tau_table(max_n):
    """Compute Ramanujan tau(n) for n = 0,...,max_n via q-expansion of Delta."""
    print(f"  Computing tau(n) for n <= {max_n}...", flush=True)
    t0 = time.perf_counter()
    # Use Sage's built-in delta q-expansion
    prec = int(max_n) + 1
    d = delta_qexp(prec)
    tau = np.zeros(prec, dtype=np.float64)
    for i in range(1, prec):
        tau[i] = float(d[i])
    dt = time.perf_counter() - t0
    print(f"  Done ({dt:.1f}s). tau(1)={tau[1]:.0f}, tau(2)={tau[2]:.0f}, "
          f"tau(3)={tau[3]:.0f}, tau(12)={tau[12]:.0f}", flush=True)
    return tau

# ─── L-function evaluation ────────────────────────────────────────────────

def compute_chi_N(max_n, N, p, q):
    """Compute chi_N(n) = kronecker(n, N) for n = 0,...,max_n.
    Uses CRT: kronecker(n, pq) = kronecker(n, p)*kronecker(n, q).
    This is a performance optimization, not oracle use."""
    chi = np.zeros(int(max_n) + 1, dtype=np.float64)
    for n in range(1, int(max_n) + 1):
        chi[n] = float(kronecker_symbol(n % int(p), int(p)) *
                        kronecker_symbol(n % int(q), int(q)))
    return chi

def eval_L_partial(s, tau, chi, max_terms):
    """Evaluate L(s, Delta x chi) via partial Dirichlet series.
    Only valid for Re(s) > 7 (region of absolute convergence).
    Returns complex value."""
    s_c = complex(s)
    n_arr = np.arange(1, max_terms + 1, dtype=np.float64)
    coeffs = tau[1:max_terms+1] * chi[1:max_terms+1]
    # L(s) = sum tau(n)*chi(n) / n^s
    terms = coeffs * np.exp(-s_c * np.log(n_arr))
    return complex(np.sum(terms))

def eval_L_approx_fe(s, tau, chi, N_val, max_terms, epsilon):
    """Evaluate L(s, Delta x chi_N) using approximate functional equation.

    L(s) = sum_n a(n)/n^s * V_+(n) + eps(s) * sum_n a(n)/n^{12-s} * V_-(n)

    where V_+(n) = exp(-(2*pi*n/(N*alpha))^2) is a smooth Gaussian cutoff
    and eps(s) encodes the gamma ratio and root number.

    This is approximate but gives reasonable values near the critical line.
    """
    s_c = complex(s)
    s_dual = complex(12) - s_c
    n_arr = np.arange(1, max_terms + 1, dtype=np.float64)
    coeffs = tau[1:max_terms+1] * chi[1:max_terms+1]

    # Smooth cutoff: V(n) = exp(-(2*pi*n / (N*alpha))^2)
    # alpha controls the split between the two sums
    alpha = 1.5  # tuning parameter
    x = 2.0 * np.pi * n_arr / (float(N_val) * alpha)
    V = np.exp(-x * x)

    # Forward sum: sum a(n)/n^s * V(n)
    fwd = np.sum(coeffs * np.exp(-s_c * np.log(n_arr)) * V)

    # Gamma ratio factor: Gamma(12-s+11/2) / Gamma(s+11/2) * (N/2pi)^{12-2s}
    # Use log-gamma for numerical stability
    log_gamma_ratio = (complex(special.loggamma(s_dual + 11.0/2.0)) -
                       complex(special.loggamma(s_c + 11.0/2.0)))
    log_N_factor = (12.0 - 2.0 * s_c) * np.log(float(N_val) / (2.0 * np.pi))
    eps_s = epsilon * np.exp(log_gamma_ratio + log_N_factor)

    # Dual sum: sum a(n)/n^{12-s} * V(n)
    dual = np.sum(coeffs * np.exp(-s_dual * np.log(n_arr)) * V)

    return complex(fwd + eps_s * dual)

def compute_root_number(N_val, p, q):
    """Compute root number epsilon for L(s, Delta x chi_N).

    For Delta (weight 12, level 1) twisted by chi_N (conductor N):
    epsilon = chi_N(-1) * i^12 * (tau(chi_N) / sqrt(N))^2
            = chi_N(-1) * (tau(chi_N))^2 / N

    chi_N(-1) = kronecker(-1, N).
    tau(chi_N) = sum_{a=0}^{N-1} chi_N(a) * e^{2*pi*i*a/N} (Gauss sum).

    This is computable from N alone (no p,q needed for the value,
    though we use p,q for speed).
    """
    # chi_N(-1) = kronecker(-1, N)
    chi_neg1 = float(kronecker_symbol(-1, int(N_val)))

    # NOTE: Mathematically, tau(chi_N) = sum_{a=1}^{N-1} chi_N(a) e^{2pi*i*a/N}
    # can be computed without factoring. Here we use the CRT factored form
    # tau(chi_N) = tau(chi_p) * tau(chi_q) for computational efficiency,
    # which requires knowing p and q.
    # WARNING: Floating-point Gauss sum is numerically unstable for large p.
    # |tau_p| = sqrt(p) but individual terms have magnitude 1, so relative
    # error grows as O(1) for large p. Consider using mpmath for p > 10^6.
    if int(p) > 10**6:
        print(f"  WARNING: Gauss sum for p={p} may have poor precision", flush=True)
    if int(q) > 10**6:
        print(f"  WARNING: Gauss sum for q={q} may have poor precision", flush=True)
    tau_p = sum(kronecker_symbol(a, int(p)) * np.exp(2j * np.pi * a / int(p))
                for a in range(1, int(p)))
    tau_q = sum(kronecker_symbol(a, int(q)) * np.exp(2j * np.pi * a / int(q))
                for a in range(1, int(q)))
    tau_chi = tau_p * tau_q

    # Root number: epsilon = chi_N(-1) * tau(chi_N)^2 / N
    # For weight 12: epsilon = chi_N(-1) * (-1)^{12/2} * tau(chi_N)^2 / N
    #                        = chi_N(-1) * tau(chi_N)^2 / N
    eps = chi_neg1 * tau_chi * tau_chi / float(N_val)
    # Should be +/- 1 for real
    eps_real = float(np.real(eps))
    return eps_real

# ─── Feature extraction ──────────────────────────────────────────────────

def extract_L_features(tau, chi, N_val, p, q, epsilon, max_terms, s_grid):
    """Extract feature vector from L-function evaluations."""
    features = {}

    # Evaluate L at multiple s-values
    L_vals = []
    for s in s_grid:
        if np.real(s) >= 7.5:
            # Safe convergence region: use partial sum
            L = eval_L_partial(s, tau, chi, max_terms)
        else:
            # Near critical line: use approximate FE
            L = eval_L_approx_fe(s, tau, chi, N_val, max_terms, epsilon)
        L_vals.append(L)

    features['L_values'] = L_vals
    features['L_mags'] = [abs(L) for L in L_vals]
    features['L_phases'] = [float(np.angle(L)) for L in L_vals]
    features['epsilon'] = epsilon

    # Central value (at s = 6)
    L_center = eval_L_approx_fe(6.0, tau, chi, N_val, max_terms, epsilon)
    features['L_center'] = complex(L_center)
    features['L_center_mag'] = abs(L_center)

    return features

# ─── Factor candidate scoring ────────────────────────────────────────────

def euler_factor_score(m, s, tau_m, chi_N_m):
    """Compute the Euler factor at prime m for L(s, Delta x chi_N).

    If chi_N(m) = 0 (i.e., m | N): factor = (1 + m^{11-2s})^{-1}
    If chi_N(m) != 0: factor = (1 - tau(m)*chi_N(m)/m^s + m^{11-2s})^{-1}
    """
    s_c = complex(s)
    m_f = float(m)
    if abs(chi_N_m) < 0.5:
        # m divides N
        return 1.0 / (1.0 + m_f**(11.0 - 2.0*np.real(s_c)))
    else:
        return 1.0 / (1.0 - tau_m * chi_N_m * m_f**(-s_c) + m_f**(11.0 - 2.0*np.real(s_c)))

# ─── Main ─────────────────────────────────────────────────────────────────

print("E8a — Twisted GL(2) L-function Tomography\n", flush=True)
print("=" * 80, flush=True)
print("Fixed form: Delta (Ramanujan, weight 12, level 1)", flush=True)
print("Twist: chi_N = kronecker(., N) (Jacobi character)", flush=True)
print("=" * 80, flush=True)

# Generate semiprimes
semiprimes = gen_balanced(count=12, min_p=50, max_p=350)
confusable = gen_confusable_pairs(count=6)

# Max N for tau precomputation
max_N = max(s[0] for s in semiprimes)
if confusable:
    max_N = max(max_N, max(max(c[0][0], c[1][0]) for c in confusable))
tau_cutoff = int(max_N * 2)  # Enough terms for approximate FE

tau = compute_tau_table(tau_cutoff)

# s-grid: points on and near the critical line (Re(s) = 6)
# Plus convergent region points (Re(s) = 8) for comparison
s_grid_critical = [complex(6, t) for t in np.linspace(0, 5, 11)]
s_grid_convergent = [complex(8, t) for t in np.linspace(0, 5, 6)]
s_grid = s_grid_critical + s_grid_convergent

print(f"\nSemiprimes: {len(semiprimes)}  [{semiprimes[0][0]}..{semiprimes[-1][0]}]",
      flush=True)
print(f"Confusable pairs: {len(confusable)}", flush=True)
print(f"s-grid: {len(s_grid)} points (critical + convergent)\n", flush=True)

# ─── TEST A: L-value profiles for each semiprime ──────────────────────────

print("=" * 80, flush=True)
print("TEST A: L-value profiles on the critical line", flush=True)
print("=" * 80, flush=True)

results_A = []
for idx, (N, p, q) in enumerate(semiprimes):
    t0 = time.perf_counter()
    max_terms = min(int(N * 2), tau_cutoff)
    chi = compute_chi_N(max_terms, N, p, q)
    eps = compute_root_number(N, p, q)

    feats = extract_L_features(tau, chi, N, p, q, eps, max_terms, s_grid)
    dt = time.perf_counter() - t0

    results_A.append({
        'N': N, 'p': p, 'q': q,
        'epsilon': eps,
        'L_center_mag': feats['L_center_mag'],
        'L_mags': feats['L_mags'],
        'L_phases': feats['L_phases'],
    })

    eps_str = '+1' if eps > 0 else '-1'
    print(f"  N={N:>7} = {p}x{q}  eps={eps_str}  |L(6)|={feats['L_center_mag']:.6e}  ({dt:.1f}s)",
          flush=True)

print(flush=True)

# ─── TEST B: Confusable pair distinguishability ───────────────────────────

print("=" * 80, flush=True)
print("TEST B: Can L-values distinguish confusable semiprimes?", flush=True)
print("  (Same-sized semiprimes with different factorizations)", flush=True)
print("=" * 80, flush=True)

results_B = []
for pair_idx, ((N1, p1, q1), (N2, p2, q2)) in enumerate(confusable):
    max_terms1 = min(int(N1 * 2), tau_cutoff)
    max_terms2 = min(int(N2 * 2), tau_cutoff)

    chi1 = compute_chi_N(max_terms1, N1, p1, q1)
    chi2 = compute_chi_N(max_terms2, N2, p2, q2)

    eps1 = compute_root_number(N1, p1, q1)
    eps2 = compute_root_number(N2, p2, q2)

    feats1 = extract_L_features(tau, chi1, N1, p1, q1, eps1, max_terms1, s_grid)
    feats2 = extract_L_features(tau, chi2, N2, p2, q2, eps2, max_terms2, s_grid)

    # Compare L-value profiles
    mags1 = np.array(feats1['L_mags'])
    mags2 = np.array(feats2['L_mags'])
    phases1 = np.array(feats1['L_phases'])
    phases2 = np.array(feats2['L_phases'])

    # Relative difference in magnitudes
    mag_diff = np.mean(np.abs(mags1 - mags2) / (np.abs(mags1) + np.abs(mags2) + 1e-30))
    phase_diff = np.mean(np.abs(np.mod(phases1 - phases2 + np.pi, 2*np.pi) - np.pi))

    # Epsilon comparison
    eps_same = (eps1 > 0) == (eps2 > 0)

    results_B.append({
        'N1': N1, 'p1': p1, 'q1': q1,
        'N2': N2, 'p2': p2, 'q2': q2,
        'eps1': eps1, 'eps2': eps2, 'eps_same': eps_same,
        'mag_rel_diff': float(mag_diff),
        'phase_mean_diff': float(phase_diff),
    })

    print(f"  {N1}={p1}x{q1} vs {N2}={p2}x{q2}: "
          f"eps={'same' if eps_same else 'DIFF'}  "
          f"mag_diff={mag_diff:.4f}  phase_diff={phase_diff:.4f}", flush=True)

print(flush=True)

# ─── TEST C: Root number as factor discriminant ───────────────────────────

print("=" * 80, flush=True)
print("TEST C: Root number epsilon analysis", flush=True)
print("  epsilon = chi_N(-1) * tau(chi_N)^2 / N", flush=True)
print("  Does epsilon carry factor information beyond N mod 4?", flush=True)
print("=" * 80, flush=True)

eps_plus = [(N, p, q) for N, p, q in semiprimes if compute_root_number(N, p, q) > 0]
eps_minus = [(N, p, q) for N, p, q in semiprimes if compute_root_number(N, p, q) < 0]
print(f"  epsilon = +1: {len(eps_plus)} semiprimes", flush=True)
print(f"  epsilon = -1: {len(eps_minus)} semiprimes", flush=True)

# Check if epsilon is determined by N mod 4 or p,q mod 4
hdr = "%7s %4s %4s %4s %4s %4s %5s" % ('N', 'p', 'q', 'N%4', 'p%4', 'q%4', 'eps')
print("\n  " + hdr, flush=True)
for N, p, q in semiprimes:
    eps = compute_root_number(N, p, q)
    Nm = int(N) % 4; pm = int(p) % 4; qm = int(q) % 4
    row = "%7d %4d %4d %4d %4d %4d %5s" % (N, p, q, Nm, pm, qm, '+1' if eps > 0 else '-1')
    print("  " + row, flush=True)

print(flush=True)

# ─── TEST D: Euler factor removal scoring ─────────────────────────────────

print("=" * 80, flush=True)
print("TEST D: Factor candidate scoring via Euler factor removal", flush=True)
print("  For candidate primes m, compute local Euler factor and score.", flush=True)
print("  chi_N(m) = 0 iff m|N (= gcd test). Does anything BEYOND this help?", flush=True)
print("=" * 80, flush=True)

results_D = []
# Test on a few semiprimes
test_semiprimes = semiprimes[:5]
for N, p, q in test_semiprimes:
    max_terms = min(int(N * 2), tau_cutoff)
    chi = compute_chi_N(max_terms, N, p, q)
    eps = compute_root_number(N, p, q)

    # Compute L at a convergent point
    s_test = complex(8, 0)
    L_base = eval_L_partial(s_test, tau, chi, max_terms)

    # For each candidate prime m, compute the Euler factor
    candidate_primes = list(primes(2, min(200, int(N))))
    scores = []
    for m in candidate_primes:
        m_int = int(m)
        if m_int < len(tau):
            tau_m = tau[m_int]
        else:
            tau_m = float(ramanujan_tau(m_int))
        chi_m = chi[m_int] if m_int < len(chi) else float(kronecker_symbol(m_int, int(N)))

        # The Euler factor at m
        ef = euler_factor_score(m_int, s_test, tau_m, chi_m)

        # Score: |1 - ef| measures how "active" this prime is in the L-function
        # Factor primes: ef ≈ 1 (trivial), so |1-ef| ≈ 0
        # Active primes: ef can be far from 1
        score = abs(1.0 - ef)
        is_factor = (m_int == p or m_int == q)
        scores.append((m_int, score, chi_m, is_factor))

    # Sort by score (smallest = most "trivial" = most like a factor)
    scores.sort(key=lambda x: x[1])

    # Rank of actual factors
    factor_ranks = []
    for rank, (m, sc, chi_m, is_f) in enumerate(scores):
        if is_f:
            factor_ranks.append((m, rank + 1, sc, chi_m))

    results_D.append({
        'N': N, 'p': p, 'q': q,
        'factor_ranks': factor_ranks,
        'n_candidates': len(candidate_primes),
    })

    # Show bottom-10 (most factor-like) and factor ranks
    print(f"\n  N={N} = {p}x{q}  ({len(candidate_primes)} candidates)", flush=True)
    print(f"  Bottom-10 by |1-EulerFactor| (most factor-like):", flush=True)
    for m, sc, chi_m, is_f in scores[:10]:
        marker = ' <-- FACTOR' if is_f else ''
        print(f"    m={m:>3}: |1-EF|={sc:.6e}  chi_N(m)={chi_m:+.0f}{marker}",
              flush=True)
    print(f"  Actual factor ranks: ", flush=True, end='')
    for m, r, sc, chi_m in factor_ranks:
        print(f"m={m} at rank #{r} (|1-EF|={sc:.2e})", end='  ', flush=True)
    print(flush=True)

print(flush=True)

# ─── TEST E: Information-theoretic test via L-value sensitivity ───────────

print("=" * 80, flush=True)
print("TEST E: L-value sensitivity to factorization", flush=True)
print("  Compare |L(s)| profiles for semiprimes of similar size.", flush=True)
print("  Measure: can we predict p (mod small m) from L-values?", flush=True)
print("=" * 80, flush=True)

if len(results_A) >= 4:
    # Extract magnitude vectors
    all_mags = np.array([r['L_mags'] for r in results_A])
    all_Ns = np.array([r['N'] for r in results_A], dtype=float)
    all_ps = np.array([r['p'] for r in results_A], dtype=float)

    # Normalize by N (remove size dependence)
    # Simple: subtract mean, divide by std for each s-point
    mag_norm = (all_mags - np.mean(all_mags, axis=0)) / (np.std(all_mags, axis=0) + 1e-30)

    # Correlation between normalized L-value features and factor p
    print("  Correlation of |L(s_j)| with p (after size normalization):", flush=True)
    for j in range(min(len(s_grid), 11)):
        s_j = s_grid[j]
        r_corr = float(np.corrcoef(mag_norm[:, j], all_ps)[0, 1])
        print(f"    s = {np.real(s_j):.0f} + {np.imag(s_j):.1f}i: r = {r_corr:+.4f}",
              flush=True)

    # PCA on L-value features
    from numpy.linalg import svd
    U, s_vals, Vt = svd(mag_norm, full_matrices=False)
    print(f"\n  Top 3 singular values of L-value matrix: "
          f"{s_vals[0]:.3f}, {s_vals[1]:.3f}, {s_vals[2]:.3f}", flush=True)
    print(f"  Explained variance (top-1): {s_vals[0]**2/sum(s_vals**2):.3f}", flush=True)

    # Correlation of top PC with p
    pc1 = U[:, 0]
    r_pc = float(np.corrcoef(pc1, all_ps)[0, 1])
    print(f"  Correlation of PC1 with p: r = {r_pc:+.4f}", flush=True)

print(flush=True)

# ─── Save ─────────────────────────────────────────────────────────────────

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

def _py_complex(v):
    """Extend shared _py() with complex number support for L-function values."""
    if isinstance(v, complex):
        return {'re': float(v.real), 'im': float(v.imag)}
    return _py(v)

def clean_results(obj):
    if isinstance(obj, dict):
        return {k: clean_results(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_results(v) for v in obj]
    return _py_complex(obj)

output = {
    'semiprimes': [{'N': N, 'p': p, 'q': q} for N, p, q in semiprimes],
    'test_A': clean_results(results_A),
    'test_B': clean_results(results_B),
    'test_D': clean_results(results_D),
}
out_path = os.path.join(data_dir, 'E8a_Lfunction_tomography_results.json')
safe_json_dump(output, out_path)
print(f"Saved -> {out_path}\n", flush=True)

# ─── Verdict ──────────────────────────────────────────────────────────────

print("=" * 80, flush=True)
print("E8a L-FUNCTION TOMOGRAPHY — VERDICT", flush=True)
print("=" * 80, flush=True)
print(flush=True)

# Check Test D: is Euler factor scoring just gcd?
gcd_equivalent = True
for r in results_D:
    for m, rank, sc, chi_m in r['factor_ranks']:
        # Factor primes have chi_N(m) = 0, which is detectable as gcd(m,N)>1
        # The Euler factor at m|N is (1+m^{-5})^{-1} ~ 1, giving |1-EF| ~ m^{-5}
        # This ranking is exactly gcd-based: primes with chi_N=0 sort to bottom
        if rank > 5:
            gcd_equivalent = False

if gcd_equivalent:
    print("  TEST D: Euler factor scoring places factor primes at bottom ranks.", flush=True)
    print("  BUT: this is because chi_N(m)=0 iff gcd(m,N)>1.", flush=True)
    print("  The scoring IS gcd computation — no information beyond trial division.", flush=True)
else:
    print("  TEST D: Factor primes NOT consistently at bottom ranks.", flush=True)
    print("  This would indicate non-trivial information in Euler factors.", flush=True)

print(flush=True)

# Check Test B: distinguishability
if results_B:
    avg_mag_diff = np.mean([r['mag_rel_diff'] for r in results_B])
    eps_diffs = sum(1 for r in results_B if not r['eps_same'])
    print(f"  TEST B: Mean magnitude difference between confusable pairs: {avg_mag_diff:.4f}",
          flush=True)
    print(f"  Pairs with different epsilon: {eps_diffs}/{len(results_B)}", flush=True)
    if avg_mag_diff > 0.1:
        print("  L-values DO distinguish confusable semiprimes (but this may just", flush=True)
        print("  reflect different N values, not factor-specific information).", flush=True)
    else:
        print("  L-values do NOT strongly distinguish confusable semiprimes.", flush=True)

print(flush=True)
print("STRUCTURAL OBSERVATION:", flush=True)
print("  chi_N(p) = 0 for p|N means the Euler product at factor primes is", flush=True)
print("  trivial: L_p(s) = (1 + p^{11-2s})^{-1}. Factor primes contribute ~1.", flush=True)
print("  Detecting chi_N(m) = 0 is equivalent to gcd(m, N) > 1.", flush=True)
print(flush=True)
print("  The GLOBAL properties (zeros, central value, functional equation)", flush=True)
print("  do depend on which primes are 'missing,' but extracting this requires", flush=True)
print("  resolving O(N log N) zeros to O(1/log N) precision, costing O(N^2).", flush=True)
print(flush=True)
print("  Root number epsilon depends on p mod 4 and q mod 4 via Gauss sums,", flush=True)
print("  but epsilon is computable from tau(chi_N) without factoring — it", flush=True)
print("  collapses local Gauss sums into a single product.", flush=True)
print(flush=True)
