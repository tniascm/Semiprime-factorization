"""
E7b — Spectral Sparsity Analysis & GCD-Class Energy Decomposition
Follow-up to E7: replace crude 1% threshold with participation ratio,
k_90 sparsity, and gcd-bucketed energy analysis.
"""
import time
import sys
import json
import os

import numpy as np
from scipy import stats

# ─── orbital integrals (same as E7) ─────────────────────────────────────────

def local_orbital_integral(t, p):
    disc = t*t - 4
    if disc % p == 0:
        return 1
    return 1 + kronecker_symbol(disc, p)

def orbital_product(t, primes):
    r = 1
    for p in primes:
        r *= local_orbital_integral(t, p)
        if r == 0:
            return 0
    return r

def poisson_fourier_all(N, primes):
    orb = np.array([orbital_product(t, primes) for t in range(N)], dtype=np.float64)
    X = np.fft.fft(orb)
    return np.conj(X) / N

# ─── spectral sparsity measures ─────────────────────────────────────────────

def spectral_sparsity(coeffs):
    """
    Compute spectral sparsity measures from Fourier coefficients.
    coeffs[0] is the zero mode; coeffs[1:] are the cuspidal modes.
    """
    mags = np.abs(coeffs)
    N = len(mags)

    # Cuspidal (non-zero modes)
    cusp_mags = mags[1:]
    cusp_energy_vals = cusp_mags**2
    cusp_total = cusp_energy_vals.sum()

    if cusp_total < 1e-30:
        return {'k_50': 0, 'k_67': 0, 'k_90': 0, 'k_95': 0,
                'participation_ratio': 0, 'entropy_rank': 0}

    # Sort by descending energy
    sorted_E = np.sort(cusp_energy_vals)[::-1]
    cumsum = np.cumsum(sorted_E) / cusp_total

    def k_threshold(frac):
        idx = np.searchsorted(cumsum, frac)
        return int(min(idx + 1, len(cumsum)))

    # Participation ratio: r = 1 / Σ p_i^2, where p_i = E_i / Σ E
    p_i = cusp_energy_vals / cusp_total
    herfindahl = np.sum(p_i**2)
    participation = 1.0 / herfindahl if herfindahl > 0 else 0

    # Shannon entropy rank: exp(H) where H = -Σ p_i log p_i
    mask = p_i > 0
    entropy = -np.sum(p_i[mask] * np.log(p_i[mask]))
    entropy_rank = np.exp(entropy)

    return {
        'k_50': k_threshold(0.50),
        'k_67': k_threshold(0.67),
        'k_90': k_threshold(0.90),
        'k_95': k_threshold(0.95),
        'participation_ratio': float(participation),
        'entropy_rank': float(entropy_rank),
    }

# ─── gcd-class energy decomposition ─────────────────────────────────────────

def gcd_class_energy(coeffs, N):
    """
    Bucket Fourier modes by gcd(xi, N) and compute energy in each bucket.
    """
    mags2 = np.abs(coeffs)**2
    total_cusp_E = mags2[1:].sum()

    # Possible gcd values for N = pq are {1, p, q, N}
    # More generally, divisors of N
    divs = divisors(N)
    buckets = {}
    for d in divs:
        buckets[int(d)] = {'count': 0, 'energy': 0.0, 'mean_mag': 0.0}

    for xi in range(1, N):
        g = gcd(xi, N)
        g = int(g)
        mags2_xi = float(mags2[xi])
        if g in buckets:
            buckets[g]['count'] += 1
            buckets[g]['energy'] += mags2_xi

    # Compute fractions
    for d in buckets:
        if buckets[d]['count'] > 0:
            buckets[d]['mean_mag'] = np.sqrt(buckets[d]['energy'] / buckets[d]['count'])
        if total_cusp_E > 0:
            buckets[d]['energy_frac'] = buckets[d]['energy'] / total_cusp_E
        else:
            buckets[d]['energy_frac'] = 0.0

    # Summary: gcd=1 (bulk) vs gcd>1 (divisor-related)
    bulk_E = buckets.get(1, {}).get('energy', 0)
    div_E = total_cusp_E - bulk_E
    div_count = sum(buckets[d]['count'] for d in buckets if d > 1)

    return {
        'buckets': buckets,
        'bulk_energy_frac': float(bulk_E / total_cusp_E) if total_cusp_E > 0 else 0,
        'div_energy_frac': float(div_E / total_cusp_E) if total_cusp_E > 0 else 0,
        'bulk_count': int(buckets.get(1, {}).get('count', 0)),
        'div_count': div_count,
    }

# ─── generate semiprimes (balanced and unbalanced) ──────────────────────────

def gen_balanced(count=25, min_p=50):
    """Balanced semiprimes with p ≈ q, both > min_p."""
    out = []
    p = next_prime(min_p)
    while len(out) < count:
        q = next_prime(p)
        out.append((int(p*q), int(p), int(q)))
        p = next_prime(q + randint(1, 50))
    return out

def gen_unbalanced(small_primes=[2, 3, 5, 7, 11, 13], count_per=6):
    """Unbalanced semiprimes N = r * q with small r."""
    out = []
    for r in small_primes:
        q = next_prime(100)
        for _ in range(count_per):
            q = next_prime(q + randint(50, 500))
            out.append((int(r*q), int(r), int(q)))
    return out

# ─── main ───────────────────────────────────────────────────────────────────

print("E7b — Spectral Sparsity & GCD-Class Energy Analysis\n", flush=True)

balanced = gen_balanced(count=30, min_p=50)
unbalanced = gen_unbalanced(small_primes=[2, 3, 5, 7, 11, 13, 17, 23, 29], count_per=5)
all_semiprimes = balanced + unbalanced
all_semiprimes.sort()

print(f"Total semiprimes: {len(all_semiprimes)} ({len(balanced)} balanced, {len(unbalanced)} unbalanced)")
print(f"Range: [{all_semiprimes[0][0]}..{all_semiprimes[-1][0]}]\n", flush=True)

hdr = (f"{'N':>7} {'p':>5} {'q':>5} {'bal':>4} {'k50':>5} {'k67':>5} {'k90':>6} "
       f"{'partR':>7} {'entR':>7} {'gcd1%':>6} {'gcd>1%':>6} {'time':>7}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

results = []
for N, p, q in all_semiprimes:
    try:
        t0 = time.perf_counter()
        primes = [p, q]
        coeffs = poisson_fourier_all(N, primes)

        ss = spectral_sparsity(coeffs)
        gc = gcd_class_energy(coeffs, N)
        dt = time.perf_counter() - t0

        is_balanced = min(p,q) / max(p,q) > 0.3
        row = {
            'N': int(N), 'p': int(p), 'q': int(q),
            'balanced': is_balanced,
            'ratio_pq': float(min(p,q)/max(p,q)),
            'k_50': ss['k_50'], 'k_67': ss['k_67'],
            'k_90': ss['k_90'], 'k_95': ss['k_95'],
            'participation_ratio': ss['participation_ratio'],
            'entropy_rank': ss['entropy_rank'],
            'bulk_energy_frac': gc['bulk_energy_frac'],
            'div_energy_frac': gc['div_energy_frac'],
            'bulk_count': gc['bulk_count'],
            'div_count': gc['div_count'],
            'time': dt,
        }
        # Per-gcd-bucket energy fractions
        for d, info in gc['buckets'].items():
            row[f'gcd_{d}_energy_frac'] = info['energy_frac']
            row[f'gcd_{d}_count'] = info['count']
            row[f'gcd_{d}_mean_mag'] = info['mean_mag']

        results.append(row)

        bal_tag = " B" if is_balanced else "  "
        print(f"{N:>7} {p:>5} {q:>5} {bal_tag} {ss['k_50']:>5} {ss['k_67']:>5} {ss['k_90']:>6} "
              f"{ss['participation_ratio']:>7.1f} {ss['entropy_rank']:>7.1f} "
              f"{gc['bulk_energy_frac']:>6.3f} {gc['div_energy_frac']:>6.3f} {dt:>7.3f}s",
              flush=True)
    except Exception as e:
        print(f"  N={N}: ERROR — {e}", flush=True)

print(f"\nCompleted {len(results)} / {len(all_semiprimes)}\n", flush=True)

# ─── save ────────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, 'E7b_spectral_sparsity_results.json')

def _py(v):
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return float(v)

with open(out_path, 'w') as f:
    json.dump([{k: _py(v) for k, v in r.items()} for r in results], f, indent=2)
print(f"Saved → {out_path}\n", flush=True)

# ─── analysis ────────────────────────────────────────────────────────────────

Ns_all = np.array([r['N'] for r in results])
bal_mask = np.array([r['balanced'] for r in results])
unbal_mask = ~bal_mask

# Separate balanced vs unbalanced
def fit_and_report(name, Ns, vals, label):
    mask = vals > 0
    if mask.sum() < 3:
        print(f"  {name}: insufficient data", flush=True)
        return None
    sl, intc, rv, _, se = stats.linregress(np.log10(Ns[mask]), np.log10(vals[mask]))
    print(f"  {name:25s}: {label} ~ N^{sl:.3f} ± {se:.3f}  R²={rv**2:.4f}", flush=True)
    return sl

print("=== Scaling Exponents (ALL) ===", flush=True)
k50_all = np.array([r['k_50'] for r in results], dtype=float)
k67_all = np.array([r['k_67'] for r in results], dtype=float)
k90_all = np.array([r['k_90'] for r in results], dtype=float)
pr_all  = np.array([r['participation_ratio'] for r in results])
er_all  = np.array([r['entropy_rank'] for r in results])

fit_and_report("k_50", Ns_all, k50_all, "k_50")
fit_and_report("k_67", Ns_all, k67_all, "k_67")
fit_and_report("k_90", Ns_all, k90_all, "k_90")
fit_and_report("participation_ratio", Ns_all, pr_all, "partR")
fit_and_report("entropy_rank", Ns_all, er_all, "entR")
print(flush=True)

if bal_mask.sum() > 3:
    print("=== Scaling Exponents (BALANCED only) ===", flush=True)
    Ns_b = Ns_all[bal_mask]
    fit_and_report("k_50", Ns_b, k50_all[bal_mask], "k_50")
    fit_and_report("k_67", Ns_b, k67_all[bal_mask], "k_67")
    fit_and_report("k_90", Ns_b, k90_all[bal_mask], "k_90")
    fit_and_report("participation_ratio", Ns_b, pr_all[bal_mask], "partR")
    fit_and_report("entropy_rank", Ns_b, er_all[bal_mask], "entR")
    print(flush=True)

# GCD-class summary for balanced
if bal_mask.sum() > 0:
    print("=== GCD-Class Energy (BALANCED semiprimes) ===", flush=True)
    bulk_fracs = np.array([r['bulk_energy_frac'] for r in results if r['balanced']])
    div_fracs = np.array([r['div_energy_frac'] for r in results if r['balanced']])
    print(f"  Bulk (gcd=1) energy fraction:  mean={bulk_fracs.mean():.4f}  std={bulk_fracs.std():.4f}", flush=True)
    print(f"  Divisor (gcd>1) energy fraction: mean={div_fracs.mean():.4f}  std={div_fracs.std():.4f}", flush=True)
    print(f"  Predicted (analytic): bulk=1/3, div=2/3", flush=True)
    print(flush=True)

# Energy band analysis for unbalanced (by small prime)
print("=== Energy Bands by Small Prime (UNBALANCED) ===", flush=True)
unbal_results = [r for r in results if not r['balanced']]
small_primes_seen = sorted(set(min(r['p'], r['q']) for r in unbal_results))
for sp in small_primes_seen:
    subset = [r for r in unbal_results if min(r['p'], r['q']) == sp]
    if len(subset) < 2:
        continue
    cusp_Es = [1 - r['bulk_energy_frac'] for r in subset]  # div energy = cusp in gcd>1
    # Actually, let's report total cuspidal energy (1 - zero-mode fraction)
    # which we can get from the original E7 formula
    div_fracs = [r['div_energy_frac'] for r in subset]
    bulk_fracs = [r['bulk_energy_frac'] for r in subset]
    print(f"  r={sp:>3}: div_energy_frac = {np.mean(div_fracs):.4f} ± {np.std(div_fracs):.4f}  (n={len(subset)})",
          flush=True)
print(flush=True)

# ─── plots ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. k_90 vs N
ax = axes[0, 0]
ax.scatter(Ns_all[bal_mask], k90_all[bal_mask], s=20, alpha=.7, color='blue', label='Balanced')
ax.scatter(Ns_all[unbal_mask], k90_all[unbal_mask], s=20, alpha=.7, color='orange', label='Unbalanced')
ax.plot(sorted(Ns_all), sorted(Ns_all), 'r--', lw=1, alpha=.4, label='O(N)')
ax.plot(sorted(Ns_all), np.sqrt(sorted(Ns_all))*5, 'g--', lw=1, alpha=.4, label='O(√N)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('k_90')
ax.set_title('k_90: modes for 90% cuspidal energy'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

# 2. Participation ratio vs N
ax = axes[0, 1]
ax.scatter(Ns_all[bal_mask], pr_all[bal_mask], s=20, alpha=.7, color='blue', label='Balanced')
ax.scatter(Ns_all[unbal_mask], pr_all[unbal_mask], s=20, alpha=.7, color='orange', label='Unbalanced')
ax.plot(sorted(Ns_all), np.sqrt(sorted(Ns_all))*4.5, 'g--', lw=1, alpha=.4, label='4.5√N')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('Participation ratio')
ax.set_title('Participation Ratio (effective dim)'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

# 3. div vs bulk energy fraction
ax = axes[0, 2]
div_frac = np.array([r['div_energy_frac'] for r in results])
ax.scatter(Ns_all[bal_mask], div_frac[bal_mask], s=20, alpha=.7, color='blue', label='Balanced')
ax.scatter(Ns_all[unbal_mask], div_frac[unbal_mask], s=20, alpha=.7, color='orange', label='Unbalanced')
ax.axhline(2/3, color='green', lw=1, ls='--', label='Predicted 2/3')
ax.set_xscale('log')
ax.set_xlabel('N'); ax.set_ylabel('gcd>1 energy / cuspidal energy')
ax.set_title('Energy in Divisor-Related Modes'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

# 4. k_67 vs N (this is where we expect ~ p+q ~ 2√N for balanced)
ax = axes[1, 0]
ax.scatter(Ns_all[bal_mask], k67_all[bal_mask], s=20, alpha=.7, color='blue', label='Balanced')
ax.scatter(Ns_all[unbal_mask], k67_all[unbal_mask], s=20, alpha=.7, color='orange', label='Unbalanced')
pq_sum = np.array([r['p'] + r['q'] for r in results])
ax.scatter(Ns_all[bal_mask], pq_sum[bal_mask], s=15, alpha=.5, color='green', marker='x', label='p+q')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('k_67')
ax.set_title('k_67: modes for 67% cuspidal energy'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

# 5. Entropy rank vs N
ax = axes[1, 1]
ax.scatter(Ns_all[bal_mask], er_all[bal_mask], s=20, alpha=.7, color='blue', label='Balanced')
ax.scatter(Ns_all[unbal_mask], er_all[unbal_mask], s=20, alpha=.7, color='orange', label='Unbalanced')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('Entropy rank exp(H)')
ax.set_title('Shannon Entropy Rank'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)

# 6. Per-gcd-bucket energy for a specific balanced example
ax = axes[1, 2]
bal_results = [r for r in results if r['balanced'] and r['N'] > 5000]
if bal_results:
    ex = bal_results[-1]  # largest balanced
    # Extract bucket data
    divs_ex = sorted([int(d) for d in divisors(ex['N'])])
    bucket_E = [ex.get(f'gcd_{d}_energy_frac', 0) for d in divs_ex]
    bucket_labels = [str(d) for d in divs_ex]
    bars = ax.bar(range(len(divs_ex)), bucket_E, alpha=.7, color=['gray','blue','red','purple'][:len(divs_ex)])
    ax.set_xticks(range(len(divs_ex)))
    ax.set_xticklabels(bucket_labels, fontsize=8)
    ax.set_xlabel(f'gcd(ξ, N) class  [N={ex["N"]}={ex["p"]}×{ex["q"]}]')
    ax.set_ylabel('Fraction of cuspidal energy')
    ax.set_title(f'Energy by GCD Class (balanced example)')
else:
    ax.text(0.5, 0.5, 'No balanced N>5000', ha='center', va='center', transform=ax.transAxes)
ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'E7b_spectral_sparsity_plots.png'), dpi=150)
print(f"Saved plots → data/E7b_spectral_sparsity_plots.png\n", flush=True)

# ─── verdict ─────────────────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("E7b SPECTRAL SPARSITY — VERDICT", flush=True)
print("=" * 70, flush=True)
print(flush=True)

print("The 1% threshold (E7) showed eff_rank ~ N^0.94 — misleadingly flat.", flush=True)
print("The spectral sparsity measures reveal the true structure:", flush=True)
print(flush=True)
print("Participation ratio tells us the effective dimensionality.", flush=True)
print("k_67 tells us how many modes capture 2/3 of energy (the div-related share).", flush=True)
print("The gcd-class decomposition tells us WHERE the energy concentrates.", flush=True)
print(flush=True)
print("If participation_ratio ~ √N and 2/3 of energy is in gcd>1 modes,", flush=True)
print("then the spectrum has the predicted 3-tier structure and factor", flush=True)
print("information is concentrated in a √N-sized subset of frequencies.", flush=True)
print(flush=True)
