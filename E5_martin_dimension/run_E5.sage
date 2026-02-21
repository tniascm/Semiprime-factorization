"""
E5 — Martin Dimension Test (runnable sage script)
Benchmark dim S_k^new(Gamma_0(N)) via modular symbols for semiprimes.
"""
import time
import sys
import json
import os

import numpy as np
from scipy import stats

set_random_seed(42)

# ─── helpers ────────────────────────────────────────────────────────────────

def generate_semiprimes(max_N, num_samples=60, min_N=100):
    targets = np.logspace(np.log10(min_N), np.log10(max_N), num_samples)
    semiprimes = []
    seen = set()
    for target in targets:
        N_approx = int(target)
        if N_approx < 6:
            continue
        for offset in range(300):
            candidate = N_approx + offset
            if candidate < 6 or candidate in seen:
                continue
            f = factor(candidate)
            if len(f) == 2 and all(e == 1 for _, e in f):
                p, q = int(f[0][0]), int(f[1][0])
                semiprimes.append((candidate, p, q))
                seen.add(candidate)
                break
    semiprimes.sort()
    return semiprimes

# ─── benchmarks ─────────────────────────────────────────────────────────────

def bench_modsym(N, k=2):
    t0 = time.perf_counter()
    M = ModularSymbols(Gamma0(N), k, sign=1)
    S = M.cuspidal_subspace()
    Snew = S.new_subspace()
    dim = Snew.dimension()
    return dim, time.perf_counter() - t0

def mu_star_mu(n):
    divs = divisors(n)
    return sum(moebius(d) * moebius(n // d) for d in divs)

def dim_Sk_new_formula(N, k=2):
    divs = divisors(N)
    return sum(mu_star_mu(N // d) * (Gamma0(d).genus() if k == 2 else dimension_cusp_forms(Gamma0(d), k))
               for d in divs)

def bench_formula(N, k=2):
    t0 = time.perf_counter()
    dim = dim_Sk_new_formula(N, k)
    return dim, time.perf_counter() - t0

def bench_trial_div(N):
    t0 = time.perf_counter()
    n = int(N)
    if n % 2 == 0:
        r = 2
    else:
        i = 3
        while i * i <= n:
            if n % i == 0:
                r = i
                break
            i += 2
        else:
            r = n
    return r, time.perf_counter() - t0

def bench_ecm(N):
    t0 = time.perf_counter()
    r = ecm.factor(N)
    return r, time.perf_counter() - t0

# ─── sanity check ───────────────────────────────────────────────────────────
print("Sanity check on N=77 …", flush=True)
d_ms, t_ms = bench_modsym(77)
d_f, t_f = bench_formula(77)
print(f"  modsym  dim={d_ms}  t={t_ms:.4f}s", flush=True)
print(f"  formula dim={d_f}  t={t_f:.6f}s", flush=True)
assert d_ms == d_f, f"MISMATCH {d_ms} vs {d_f}"
print("  OK — dimensions match.\n", flush=True)

# ─── main sweep ─────────────────────────────────────────────────────────────
semiprimes = generate_semiprimes(50000, num_samples=55)
print(f"Generated {len(semiprimes)} semiprimes  [{semiprimes[0][0]} .. {semiprimes[-1][0]}]\n", flush=True)

hdr = f"{'N':>8} {'p':>6} {'q':>6} {'dim':>5} {'modsym':>10} {'formula':>10} {'td':>10} {'ecm':>10}"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

results = []
failed_N = []
for N, p, q in semiprimes:
    try:
        dim_ms, t_ms = bench_modsym(N)
        dim_f, t_f  = bench_formula(N)
        _, t_td      = bench_trial_div(N)
        _, t_ecm     = bench_ecm(N)

        assert dim_ms == dim_f, f"dim mismatch at N={N}: {dim_ms} vs {dim_f}"

        row = dict(N=int(N), p=int(p), q=int(q), dim_new=int(dim_ms),
                   time_modsym=float(t_ms), time_formula=float(t_f),
                   time_trial_div=float(t_td), time_ecm=float(t_ecm),
                   log_N=float(np.log10(N)))
        results.append(row)

        print(f"{N:>8} {p:>6} {q:>6} {dim_ms:>5} {t_ms:>10.4f} {t_f:>10.6f} {t_td:>10.6f} {t_ecm:>10.6f}",
              flush=True)
    except Exception as e:
        print(f"  ERROR for N={N}: {e}", flush=True)
        failed_N.append(N)

print(f"\nCompleted {len(results)} / {len(semiprimes)}", flush=True)
if failed_N:
    print(f"Failed: {len(failed_N)} semiprimes: {failed_N}", flush=True)
print(flush=True)

# ─── save raw data ──────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, 'E5_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} rows → {out_path}\n", flush=True)

# ─── power-law fit ──────────────────────────────────────────────────────────
Ns       = np.array([r['N'] for r in results])
t_modsym = np.array([r['time_modsym'] for r in results])
t_td     = np.array([r['time_trial_div'] for r in results])
t_ecm    = np.array([r['time_ecm'] for r in results])

mask = t_modsym > 0
sl, intc, rv, _, se = stats.linregress(np.log10(Ns[mask]), np.log10(t_modsym[mask]))
print("=== Modular Symbols Power-Law Fit ===", flush=True)
print(f"  T(N) = {10**intc:.2e} · N^{sl:.3f}", flush=True)
print(f"  α = {sl:.4f} ± {se:.4f}   R² = {rv**2:.6f}", flush=True)

mask_td = t_td > 0
if mask_td.sum() > 2:
    sl_td, intc_td, rv_td, _, se_td = stats.linregress(np.log10(Ns[mask_td]), np.log10(t_td[mask_td]))
    print(f"\n=== Trial Division Power-Law Fit ===", flush=True)
    print(f"  α = {sl_td:.4f} ± {se_td:.4f}   R² = {rv_td**2:.6f}", flush=True)

mask_ecm = t_ecm > 0
if mask_ecm.sum() > 2:
    sl_ecm, intc_ecm, rv_ecm, _, se_ecm = stats.linregress(np.log10(Ns[mask_ecm]), np.log10(t_ecm[mask_ecm]))
    print(f"\n=== ECM Power-Law Fit ===", flush=True)
    print(f"  α = {sl_ecm:.4f} ± {se_ecm:.4f}   R² = {rv_ecm**2:.6f}", flush=True)

# ─── piecewise fit ──────────────────────────────────────────────────────────
med = np.median(Ns)
lo = (Ns < med) & mask; hi = (Ns >= med) & mask
if lo.sum() > 2 and hi.sum() > 2:
    sl_lo, _, rv_lo, _, se_lo = stats.linregress(np.log10(Ns[lo]), np.log10(t_modsym[lo]))
    sl_hi, _, rv_hi, _, se_hi = stats.linregress(np.log10(Ns[hi]), np.log10(t_modsym[hi]))
    print(f"\n=== Piecewise Fit (median N={med:.0f}) ===", flush=True)
    print(f"  low  α = {sl_lo:.4f} ± {se_lo:.4f}  R²={rv_lo**2:.4f}", flush=True)
    print(f"  high α = {sl_hi:.4f} ± {se_hi:.4f}  R²={rv_hi**2:.4f}", flush=True)

# ─── plots ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t_formula = np.array([r['time_formula'] for r in results])
dims      = np.array([r['dim_new'] for r in results])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
ax.scatter(Ns, t_modsym, s=15, alpha=.7, label='Modular Symbols', color='blue')
ax.scatter(Ns, t_formula, s=15, alpha=.7, label='Dim Formula', color='green')
ax.scatter(Ns, t_td,      s=15, alpha=.7, label='Trial Division', color='orange')
ax.scatter(Ns, t_ecm,     s=15, alpha=.7, label='ECM', color='red')
Nf = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 100)
ax.plot(Nf, 10**intc * Nf**sl, 'b--', lw=1.5, label=f'fit N^{sl:.2f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('Time (s)')
ax.set_title('E5: Timing Comparison (log-log)')
ax.legend(fontsize=8); ax.grid(True, alpha=.3)

ax = axes[0,1]
pred = sl * np.log10(Ns[mask]) + intc
resid = np.log10(t_modsym[mask]) - pred
ax.scatter(Ns[mask], resid, s=15, alpha=.7, color='blue')
ax.axhline(0, color='k', lw=.5)
ax.set_xscale('log'); ax.set_xlabel('N'); ax.set_ylabel('log₁₀(T) residual')
ax.set_title(f'Fit Residuals (α={sl:.3f})'); ax.grid(True, alpha=.3)

ax = axes[1,0]
ax.scatter(Ns, dims, s=15, alpha=.7, color='purple')
ax.set_xscale('log'); ax.set_xlabel('N')
ax.set_ylabel('dim S₂ⁿᵉʷ(Γ₀(N))'); ax.set_title('Newform Dimension vs N'); ax.grid(True, alpha=.3)

ax = axes[1,1]
valid = (t_modsym > 0) & (t_td > 0)
ax.scatter(Ns[valid], (t_td/t_modsym)[valid], s=15, alpha=.7, label='TD/MS', color='orange')
valid2 = (t_modsym > 0) & (t_ecm > 0)
ax.scatter(Ns[valid2], (t_ecm/t_modsym)[valid2], s=15, alpha=.7, label='ECM/MS', color='red')
ax.axhline(1, color='k', lw=.5, ls='--', label='break-even')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('N'); ax.set_ylabel('ratio  (factoring / modular symbols)')
ax.set_title('Relative Speed'); ax.legend(fontsize=8); ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'E5_timing_plots.png'), dpi=150)
print(f"\nSaved plots → data/E5_timing_plots.png", flush=True)

# ─── verdict ────────────────────────────────────────────────────────────────
print("\n" + "="*70, flush=True)
print("E5 MARTIN DIMENSION TEST — SUMMARY", flush=True)
print("="*70, flush=True)
print(f"Tested {len(results)} semiprimes N = pq in [{results[0]['N']}, {results[-1]['N']}]", flush=True)
print(f"Modular Symbols scaling: T(N) ∝ N^{sl:.3f}  (R²={rv**2:.4f})", flush=True)

if   sl < 1.0: v = "SURPRISING — sublinear in N! Genuine potential."
elif sl < 2.0: v = "INTERESTING — sub-quadratic. Worth investigating shortcuts."
elif sl < 3.0: v = "EXPECTED — quadratic-to-cubic (polynomial in N ⇒ exponential in log N)."
else:          v = "STEEP — cubic+. Very unlikely to yield a practical speedup."
print(f"VERDICT: {v}\n", flush=True)

last = results[-1]
print(f"At largest N={last['N']}:", flush=True)
print(f"  modsym  {last['time_modsym']:.4f}s", flush=True)
print(f"  trial   {last['time_trial_div']:.6f}s   ({last['time_modsym']/max(last['time_trial_div'],1e-10):.0f}× slower)", flush=True)
print(f"  ECM     {last['time_ecm']:.6f}s   ({last['time_modsym']/max(last['time_ecm'],1e-10):.0f}× slower)", flush=True)
print(f"\nα={sl:.2f} means O(N^{sl:.2f}) = O(exp({sl:.2f}·ln N)).  Trial div is O(N^0.5).", flush=True)
if sl >= 0.5:
    print("⇒ ModSym is SLOWER than trial division in N-scaling.", flush=True)
    print("  A shortcut must reduce the exponent for this path to be viable.", flush=True)
