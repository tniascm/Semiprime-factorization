"""
E7 — Altuğ Beyond Endoscopy Cancellation (runnable sage script)
Measure post-Poisson cancellation in the trace formula geometric side at level N=pq.
"""
import time
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from sage_encoding import _py, _py_dict
from spectral import verify_parseval

import numpy as np

set_random_seed(42)
np.random.seed(42)

# ─── local orbital integrals ────────────────────────────────────────────────

def local_orbital_integral(t, p):
    """
    Local orbital integral O_p(t) for squarefree level at prime p.
    O_p(t) = 1 + kronecker(t^2 - 4, p)  when p ∤ (t^2-4),
           = 1                            when p | (t^2-4).
    """
    disc = t*t - 4
    if disc % p == 0:
        return 1
    return 1 + kronecker_symbol(disc, p)

def orbital_product(t, primes):
    """Product of local orbital integrals over all primes dividing N."""
    r = 1
    for p in primes:
        r *= local_orbital_integral(t, p)
        if r == 0:
            return 0
    return r

# ─── Fourier transform of local orbital data (vectorised over xi) ───────────

def poisson_fourier_all(N, primes):
    """
    Compute hat{O}(xi, N) for all xi in 0..N-1 simultaneously.

    hat{O}(xi) = (1/N) sum_{t=0}^{N-1} O(t,N) e^{2 pi i t xi / N}

    We build the orbital-product vector once, then DFT it.
    Returns a complex numpy array of length N.
    """
    orb = np.array([orbital_product(t, primes) for t in range(N)], dtype=np.float64)
    # NOTE: This uses the positive-exponential convention: hat{f}(xi) = (1/N) sum f(t) e^{+2pi i t xi/N}.
    # For real-valued signals, |hat{f}(xi)| is identical to the standard negative convention.
    # See BARRIER_THEOREM.md for discussion.
    # numpy fft: X[xi] = sum_t x[t] exp(-2pi i t xi / N)
    # So hat{O}(xi) = conj(X[xi]) / N
    X = np.fft.fft(orb)
    result = np.conj(X) / N
    # Parseval check: for our convention, sum|X|^2 = (1/N) sum|f|^2
    verify_parseval(orb, result)
    return result

# ─── cancellation analysis ──────────────────────────────────────────────────

def cancellation_analysis(N):
    fac = factor(N)
    primes = [int(p) for p, _ in fac]

    # Naive: count non-zero orbital integrals mod N
    naive_nonzero = 0
    naive_weight  = 0
    for t in range(N):
        v = orbital_product(t, primes)
        if v > 0:
            naive_nonzero += 1
            naive_weight  += v

    # Poisson-transformed coefficients (via FFT — O(N log N) not O(N^2))
    coeffs = poisson_fourier_all(N, primes)
    mags   = np.abs(coeffs)

    zero_mode   = mags[0]
    nz_mags     = mags[1:]          # non-zero frequencies
    nz_max      = float(nz_mags.max()) if len(nz_mags) else 0.0
    threshold   = nz_max * 0.01 if nz_max > 0 else 1e-10

    eff_rank    = int(np.sum(nz_mags > threshold))
    total_E     = float(np.sum(mags**2))
    cuspidal_E  = float(np.sum(nz_mags**2))
    energy_ratio = cuspidal_E / total_E if total_E > 0 else 0

    # Which non-zero modes are divisor-related?
    div_modes = 0
    for xi in range(1, N):
        if gcd(xi, N) > 1 and mags[xi] > threshold:
            div_modes += 1

    return {
        'N': int(N),
        'p': int(primes[0]) if len(primes) >= 1 else None,
        'q': int(primes[1]) if len(primes) >= 2 else None,
        'naive_nonzero': naive_nonzero,
        'naive_weight':  int(naive_weight),
        'zero_mode_mag': float(zero_mode),
        'eff_rank':      eff_rank,
        'compress':      float(eff_rank) / float(N - 1) if N > 1 else 1.0,
        'cusp_energy':   energy_ratio,
        'nz_max':        nz_max,
        'nz_mean':       float(nz_mags.mean()) if len(nz_mags) else 0,
        'nz_l2':         float(np.sqrt(cuspidal_E)),
        'div_modes':     div_modes,
        'sig_nz':        int(np.sum(nz_mags > 1e-10)),
    }

# ─── generate semiprimes ────────────────────────────────────────────────────

def gen_semiprimes(max_N, count=50):
    targets = np.logspace(np.log10(15), np.log10(max_N), count)
    out, seen = [], set()
    for tgt in targets:
        N0 = int(tgt)
        for off in range(300):
            c = N0 + off
            if c < 6 or c in seen:
                continue
            f = factor(c)
            if len(f) == 2 and all(e == 1 for _, e in f):
                out.append((c, int(f[0][0]), int(f[1][0])))
                seen.add(c)
                break
    return out

# ─── main ───────────────────────────────────────────────────────────────────

print("E7 — Altuğ Beyond Endoscopy Cancellation\n", flush=True)

# Sanity check
print("Sanity check N=15 …", flush=True)
ca15 = cancellation_analysis(15)
print(f"  naive={ca15['naive_nonzero']}  eff_rank={ca15['eff_rank']}  "
      f"compress={ca15['compress']:.4f}  cusp_E={ca15['cusp_energy']:.4f}", flush=True)
print("  OK\n", flush=True)

semiprimes = gen_semiprimes(5000, count=50)
print(f"Semiprimes: {len(semiprimes)}  [{semiprimes[0][0]}..{semiprimes[-1][0]}]\n", flush=True)

hdr = f"{'N':>6} {'p':>5} {'q':>5} {'naive':>6} {'eff_rk':>7} {'compress':>9} {'cusp_E':>8} {'div_m':>5} {'time':>8}"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

results = []
failed_N = []
for N, p, q in semiprimes:
    try:
        t0 = time.perf_counter()
        ca = cancellation_analysis(N)
        dt = time.perf_counter() - t0
        ca['time'] = dt
        results.append(ca)
        print(f"{N:>6} {p:>5} {q:>5} {ca['naive_nonzero']:>6} {ca['eff_rank']:>7} "
              f"{ca['compress']:>9.4f} {ca['cusp_energy']:>8.4f} {ca['div_modes']:>5} {dt:>8.3f}s",
              flush=True)
    except Exception as e:
        print(f"  ERROR for N={N}: {e}", flush=True)
        failed_N.append(N)

print(f"\nCompleted {len(results)} / {len(semiprimes)}", flush=True)
if failed_N:
    print(f"Failed N values: {failed_N}", flush=True)
print(flush=True)

# ─── save ────────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, 'E7_cancellation_results.json')
serial = [_py_dict(r) for r in results]
with open(out_path, 'w') as f:
    json.dump(serial, f, indent=2)
print(f"Saved {len(serial)} rows → {out_path}\n", flush=True)

# ─── fit & verdict ──────────────────────────────────────────────────────────
from scipy import stats

Ns       = np.array([r['N'] for r in results])
eff_rk   = np.array([r['eff_rank'] for r in results], dtype=float)
compress = np.array([r['compress'] for r in results])
cusp_E   = np.array([r['cusp_energy'] for r in results])

mask = eff_rk > 0
if mask.sum() > 2:
    sl, intc, rv, _, se = stats.linregress(np.log10(Ns[mask]), np.log10(eff_rk[mask]))
    print(f"Effective rank scaling: eff_rank ~ N^{sl:.3f} ± {se:.3f}  R²={rv**2:.4f}", flush=True)
else:
    sl = float('nan')
    print("Not enough data for fit", flush=True)

print(f"Compression ratio: mean={np.mean(compress):.4f}  median={np.median(compress):.4f}", flush=True)
print(f"Cuspidal energy ratio: mean={np.mean(cusp_E):.4f}", flush=True)
print(flush=True)

# ─── plots ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
naive_t = np.array([r['naive_nonzero'] for r in results])

ax = axes[0,0]
ax.scatter(Ns, eff_rk, s=20, alpha=.7, color='blue')
ax.plot(Ns, Ns, 'r--', lw=1, alpha=.5, label='O(N)')
if mask.sum() > 2:
    Nf = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
    ax.plot(Nf, 10**intc * Nf**sl, 'b--', lw=1.5, label=f'fit N^{sl:.2f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('Effective rank')
ax.set_title('Post-Cancellation Effective Rank'); ax.legend(fontsize=8); ax.grid(True, alpha=.3)

ax = axes[0,1]
ax.scatter(Ns, compress, s=20, alpha=.7, color='green')
ax.set_xscale('log'); ax.set_xlabel('N'); ax.set_ylabel('eff_rank / (N-1)')
ax.set_title('Compression Ratio vs N'); ax.grid(True, alpha=.3)

ax = axes[1,0]
ax.scatter(Ns, cusp_E, s=20, alpha=.7, color='red')
ax.set_xscale('log'); ax.set_xlabel('N'); ax.set_ylabel('Cuspidal / Total energy')
ax.set_title('Energy in Non-zero Modes'); ax.grid(True, alpha=.3)

ax = axes[1,1]
ax.scatter(naive_t, eff_rk, s=20, alpha=.7, color='purple')
mx = max(naive_t.max(), eff_rk.max())
ax.plot([0, mx], [0, mx], 'r--', lw=1, label='No compression')
ax.set_xlabel('Naive non-zero terms'); ax.set_ylabel('Effective rank')
ax.set_title('Naive vs Post-Cancellation'); ax.legend(fontsize=8); ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'E7_cancellation_plots.png'), dpi=150)
print(f"Saved plots → data/E7_cancellation_plots.png\n", flush=True)

# ─── verdict ─────────────────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("E7 ALTUG BEYOND ENDOSCOPY — VERDICT", flush=True)
print("=" * 70, flush=True)

if not np.isnan(sl):
    if   sl < 0.5:  v = "VERY PROMISING — substantial cancellation, sublinear residual."
    elif sl < 0.9:  v = "INTERESTING — partial cancellation, growth reduced but near-linear."
    elif sl < 1.05: v = "MARGINAL — near-linear growth. Cancellation doesn't help much."
    else:           v = "NO SIGNIFICANT CANCELLATION — eff_rank grows >= N."
    print(f"  eff_rank ~ N^{sl:.2f}  →  {v}", flush=True)

avg_c = np.mean(compress)
if avg_c < 0.3:
    print("  Compression < 30% — significant structural reduction.", flush=True)
elif avg_c < 0.6:
    print("  Compression 30-60% — moderate reduction.", flush=True)
else:
    print("  Compression > 60% — weak reduction.", flush=True)

print(f"\n  Cuspidal energy fraction: {np.mean(cusp_E):.3f}", flush=True)
print("  (Low = most energy in zero-mode/identity = strong cancellation)", flush=True)
print("  (High = energy spread across modes = weak cancellation)", flush=True)
print(flush=True)
