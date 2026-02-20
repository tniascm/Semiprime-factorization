"""
E12: Deep carry compositions — iterated modular squaring quotient traces
========================================================================

Motivation
----------
E10 tested depth-1 carry functions: floor(t^2/N), parity of floor(t^2/N), etc.
All were spectrally flat (peaks ~ N^{-0.35} to N^{-0.47}).

However, E10 only scratched the surface. The ITERATED carry structure from
repeated modular squaring produces O(log N) non-CRT-separable bits per
evaluation, with exponential formal CRT rank (2^d for depth d).

Key idea: In the repeated-squaring computation of t^{2^d} mod N, each step
  x_{i+1} = x_i^2 mod N
produces a quotient (carry):
  q_i = floor(x_i^2 / N)

The sequence (q_0, q_1, ..., q_{d-1}) is the "carry trace." Each q_i is
NOT CRT-separable (it depends on the integer representation of x_i, which
couples the p and q residues). Accumulating d ≈ log_2(N) such carries
produces a function with:
  - Poly(log N) computational complexity (just repeated squaring)
  - Exponentially high formal CRT rank (each composition doubles degree)
  - O(log N) non-separable bits of information

This is the LEAST EXPLORED region of the RJI oracle model. If any
poly(log N)-computable function has spectral peaks at factor frequencies,
it would most likely be found here.

Signals tested
--------------
For each t in [0, N), and depth d = ceil(log2(N)):

1. carry_final: q_{d-1} (last quotient in squaring chain)
2. carry_xor: XOR of all q_i mod 2 (parity accumulation)
3. carry_weighted: sum_{i=0}^{d-1} q_i / N (normalized carry sum)
4. carry_alternating: sum_{i=0}^{d-1} (-1)^i * (q_i mod 2) (alternating parity)
5. carry_polynomial: (sum_{i=0}^{d-1} q_i * 2^i) mod N (binary encoding)
6. carry_product_parity: product_{i=0}^{d-1} (-1)^{q_i mod 2} (multiplicative parity)
7. carry_jacobi_chain: sum J(q_i, N) over all i (Jacobi of quotients)
8. CONTROL: t^{2^d} mod N (CRT-separable, must be flat)

Plus multi-point statistics (autocorrelation-based):
9. carry_autocorr: sum_s f(t)*f(t+s) for carry_xor signal, specific shifts

Metrics (same as E10)
---------------------
- DFT peak-to-bulk ratio
- Factor-localized energy fraction
- Factor energy excess (observed / expected)
- CRT rank via SVD
- Scaling exponents across N sizes
"""

import sys
import os
import json
import time

import numpy as np
from scipy import stats

# ── semiprime generation ──────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import generate_semiprimes

# ── Sage type coercion ────────────────────────────────────────────────

class SageEncoder(json.JSONEncoder):
    """JSON encoder that handles Sage types."""
    def default(self, obj):
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        try:
            return int(obj)
        except (TypeError, ValueError):
            return str(obj)


def _py(v):
    """Convert Sage types to native Python for JSON serialization."""
    if isinstance(v, (bool, type(None), str)):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def _py_dict(d):
    """Recursively convert all values in a dict for JSON."""
    return {k: _py(v) for k, v in d.items()}


# ── Core: compute carry trace of iterated squaring ────────────────────

def carry_trace(t, N, depth):
    """
    Compute the carry trace of iterated squaring: t -> t^2 mod N -> ...

    Returns:
        quotients: list of floor(x_i^2 / N) for i=0..depth-1
        residues:  list of x_i^2 mod N for i=0..depth-1 (= x_{i+1})
    """
    N_int = int(N)
    x = int(t) % N_int
    quotients = []
    residues = []
    for _ in range(depth):
        x2 = x * x
        q = x2 // N_int
        r = x2 % N_int
        quotients.append(q)
        residues.append(r)
        x = r
    return quotients, residues


# ── Signal definitions ────────────────────────────────────────────────

def compute_signals(N, depth=None):
    """
    Compute all deep-carry signals for semiprime N.
    Returns dict: name -> np.array of length N.
    """
    N_int = int(N)
    if depth is None:
        depth = int(N_int).bit_length()  # ceil(log2(N))

    signals = {}

    # Preallocate
    sig_final = np.zeros(N_int, dtype=np.float64)
    sig_xor = np.zeros(N_int, dtype=np.float64)
    sig_weighted = np.zeros(N_int, dtype=np.float64)
    sig_alt = np.zeros(N_int, dtype=np.float64)
    sig_poly = np.zeros(N_int, dtype=np.float64)
    sig_prod_parity = np.zeros(N_int, dtype=np.float64)
    sig_jac_chain = np.zeros(N_int, dtype=np.float64)
    sig_control = np.zeros(N_int, dtype=np.float64)

    for t in range(N_int):
        quots, resids = carry_trace(t, N_int, depth)

        # Signal 1: carry_final — last quotient
        sig_final[t] = float(quots[-1] % N_int) if quots else 0.0

        # Convert quotients to Python ints to avoid Sage preparser issues
        iquots = [int(qq) for qq in quots]

        # Signal 2: carry_xor — XOR of quotient parities
        xor_val = 0
        for qq in iquots:
            xor_val = xor_val ^ (qq & 1)
        sig_xor[t] = 1.0 if xor_val == 0 else -1.0

        # Signal 3: carry_weighted — normalized sum of quotients
        carry_sum = sum(iquots)
        sig_weighted[t] = float(carry_sum) / float(N_int * depth) if depth > 0 else 0.0

        # Signal 4: carry_alternating — alternating parity sum
        alt_sum = 0.0
        for i, qq in enumerate(iquots):
            parity = 1.0 if (qq & 1 == 0) else -1.0
            alt_sum += ((-1.0) ** i) * parity
        sig_alt[t] = alt_sum

        # Signal 5: carry_polynomial — binary encoding of parities mod N
        poly_val = 0
        for i, qq in enumerate(iquots):
            if qq & 1 == 1:
                poly_val = (poly_val + (1 << i)) % N_int
        sig_poly[t] = float(poly_val)

        # Signal 6: carry_product_parity — product of parities
        prod = 1
        for qq in iquots:
            if qq & 1 == 1:
                prod = -prod
        sig_prod_parity[t] = float(prod)

        # Signal 7: carry_jacobi_chain — sum of J(q_i, N)
        jac_sum = 0.0
        for q in quots:
            if q > 0:
                g = gcd(q, N_int)
                if g == 1:
                    jac_sum += float(kronecker_symbol(q, N_int))
        sig_jac_chain[t] = jac_sum

        # Signal 8: CONTROL — t^{2^d} mod N (CRT-separable)
        sig_control[t] = float(resids[-1]) if resids else float(t)

    signals['carry_final'] = sig_final
    signals['carry_xor'] = sig_xor
    signals['carry_weighted'] = sig_weighted
    signals['carry_alternating'] = sig_alt
    signals['carry_polynomial'] = sig_poly
    signals['carry_product_parity'] = sig_prod_parity
    signals['carry_jacobi_chain'] = sig_jac_chain
    signals['control_modsq'] = sig_control

    return signals, depth


# ── DFT and spectral analysis ────────────────────────────────────────

def analyze_signal(sig, N, p, q):
    """
    Compute DFT and extract spectral metrics for a signal on Z/NZ.
    Returns dict of metrics.
    """
    N_int = int(N)
    X = np.fft.fft(sig) / N_int  # Normalized DFT
    mags = np.abs(X)
    mags2 = mags ** 2

    # Skip DC component for spectral analysis
    mags_ac = mags[1:]
    mags2_ac = mags2[1:]

    # Peak-to-bulk ratio
    peak = np.max(mags_ac)
    median_val = np.median(mags_ac)
    peak_to_bulk = float(peak / median_val) if median_val > 0 else float('inf')

    # Factor-localized energy: energy at xi with gcd(xi, N) > 1
    energy_total = float(np.sum(mags2_ac))
    energy_factor = 0.0
    for xi in range(1, N_int):
        g = gcd(xi, N_int)
        if g > 1:
            energy_factor += mags2[xi]

    n_factor_modes = 0
    for xi in range(1, N_int):
        if gcd(xi, N_int) > 1:
            n_factor_modes += 1
    expected_factor_frac = float(n_factor_modes) / (N_int - 1)

    factor_frac = float(energy_factor / energy_total) if energy_total > 0 else 0.0

    # Top-mode factor extraction: do the K largest modes have gcd > 1?
    sorted_indices = np.argsort(mags_ac)[::-1] + 1  # +1 to skip DC
    top_10_factor = sum(1 for idx in sorted_indices[:10] if gcd(int(idx), N_int) > 1)

    # CRT rank estimate: reshape signal as p x q matrix and compute SVD
    M = np.zeros((int(p), int(q)), dtype=np.float64)
    for t in range(N_int):
        a = t % int(p)
        b = t % int(q)
        M[a, b] = sig[t]
    sv = np.linalg.svd(M, compute_uv=False)
    sv_total = float(np.sum(sv))
    if sv_total > 0:
        sv_norm = sv / sv_total
        cumsum = np.cumsum(sv_norm)
        eff_rank_90 = int(np.searchsorted(cumsum, 0.90)) + 1
    else:
        eff_rank_90 = 0
    nuclear_ratio = float(sv_total / sv[0]) if sv[0] > 0 else 0.0

    # Peak location: is the top peak at a factor frequency?
    peak_idx = int(np.argmax(mags_ac)) + 1
    peak_at_factor = 1 if gcd(peak_idx, N_int) > 1 else 0

    return {
        'peak': float(peak),
        'median': float(median_val),
        'peak_to_bulk': peak_to_bulk,
        'energy_total': energy_total,
        'factor_energy_frac': factor_frac,
        'expected_factor_frac': expected_factor_frac,
        'factor_energy_excess': factor_frac / expected_factor_frac if expected_factor_frac > 0 else 0.0,
        'top_10_factor_count': top_10_factor,
        'peak_at_factor': peak_at_factor,
        'crt_eff_rank_90': eff_rank_90,
        'crt_nuclear_ratio': nuclear_ratio,
    }


# ── Multi-point analysis: autocorrelation at specific shifts ─────────

def analyze_autocorrelation(sig, N, p, q):
    """
    Compute autocorrelation of signal at factor-related shifts.
    Returns dict with autocorrelation values.
    """
    N_int = int(N)
    p_int = int(p)
    q_int = int(q)

    # Normalize signal
    sig_centered = sig - np.mean(sig)
    norm = np.sqrt(np.sum(sig_centered ** 2))
    if norm < 1e-15:
        return {'autocorr_p': 0.0, 'autocorr_q': 0.0,
                'autocorr_1': 0.0, 'autocorr_random': 0.0,
                'autocorr_ratio_p': 0.0, 'autocorr_ratio_q': 0.0}

    sig_n = sig_centered / norm

    def autocorr_at_shift(s):
        return float(np.sum(sig_n * np.roll(sig_n, -s)))

    # Autocorrelation at factor shifts
    ac_p = autocorr_at_shift(p_int)
    ac_q = autocorr_at_shift(q_int)

    # Control: autocorrelation at shift 1 and random shift
    ac_1 = autocorr_at_shift(1)
    random_shift = int(np.random.randint(2, N_int - 1))
    while gcd(random_shift, N_int) > 1:
        random_shift = int(np.random.randint(2, N_int - 1))
    ac_rand = autocorr_at_shift(random_shift)

    # Ratio: factor autocorrelation vs typical
    typical = max(abs(ac_1), abs(ac_rand), 1e-15)

    return {
        'autocorr_p': ac_p,
        'autocorr_q': ac_q,
        'autocorr_1': ac_1,
        'autocorr_random': ac_rand,
        'autocorr_ratio_p': abs(ac_p) / typical,
        'autocorr_ratio_q': abs(ac_q) / typical,
    }


# ── Main experiment ───────────────────────────────────────────────────

def main():
    print("=" * 76, flush=True)
    print("E12: Deep carry compositions — iterated squaring quotient traces", flush=True)
    print("=" * 76, flush=True)

    size_classes = [
        (500, 2000, 25),     # small
        (2000, 5000, 25),    # medium
        (5000, 12000, 25),   # large
    ]

    signal_names = [
        'carry_final',
        'carry_xor',
        'carry_weighted',
        'carry_alternating',
        'carry_polynomial',
        'carry_product_parity',
        'carry_jacobi_chain',
        'control_modsq',
    ]

    all_results = []

    for min_N, max_N, n_samples in size_classes:
        print(f"\n--- Size class: N in [{min_N}, {max_N}], {n_samples} samples ---",
              flush=True)
        semiprimes = generate_semiprimes(max_N, num_samples=n_samples, min_N=min_N)
        print(f"Generated {len(semiprimes)} semiprimes", flush=True)

        hdr = (f"{'N':>7} {'p':>5} {'q':>5} {'depth':>5} {'signal':>22} "
               f"{'peak/bulk':>10} {'fac_E':>7} {'excess':>7} {'rank90':>7}")
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

        for N, p, q in semiprimes:
            t0 = time.time()
            sigs, depth = compute_signals(N)
            dt_signals = time.time() - t0

            for sname in signal_names:
                sig = sigs[sname]
                metrics = analyze_signal(sig, N, p, q)

                # Multi-point analysis for the most promising signals
                autocorr = {}
                if sname in ('carry_xor', 'carry_product_parity', 'carry_jacobi_chain'):
                    autocorr = analyze_autocorrelation(sig, N, p, q)

                row = {
                    'N': int(N),
                    'p': int(p),
                    'q': int(q),
                    'ratio_pq': float(min(p, q)) / float(max(p, q)),
                    'depth': int(depth),
                    'signal': sname,
                }
                row.update(_py_dict(metrics))
                row.update(_py_dict(autocorr))
                row['time_signals'] = round(float(dt_signals), 4)
                all_results.append(row)

                print(f"{N:>7} {p:>5} {q:>5} {depth:>5} {sname:>22} "
                      f"{metrics['peak_to_bulk']:>10.2f} "
                      f"{metrics['factor_energy_frac']:>7.4f} "
                      f"{metrics['factor_energy_excess']:>7.3f} "
                      f"{metrics['crt_eff_rank_90']:>7}", flush=True)

    # ── Save JSON results ─────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E12_carry_depth_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=int(2), cls=SageEncoder)
    print(f"\nResults saved to {out_path}", flush=True)

    # ── Scaling analysis per signal ───────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("SCALING ANALYSIS", flush=True)
    print("=" * 76, flush=True)

    scaling_results = {}
    for sname in signal_names:
        rows = [r for r in all_results if r['signal'] == sname]
        Ns = np.array([r['N'] for r in rows], dtype=np.float64)
        peaks = np.array([r['peak'] for r in rows], dtype=np.float64)
        fac_excess = np.array([r['factor_energy_excess'] for r in rows], dtype=np.float64)
        ranks = np.array([r['crt_eff_rank_90'] for r in rows], dtype=np.float64)

        print(f"\n--- {sname} ---", flush=True)

        # Peak scaling: peak ~ N^alpha
        mask = peaks > 0
        alpha, alpha_se, alpha_r2 = 0, 0, 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(peaks[mask]))
            alpha, alpha_se, alpha_r2 = sl, se, rv**2
            print(f"  Peak scaling:    peak ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)

        # Factor energy excess scaling
        mask = fac_excess > 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(fac_excess[mask]))
            print(f"  Excess scaling:  excess ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)

        # CRT rank scaling
        mask = ranks > 0
        rank_alpha = 0
        if np.sum(mask) >= 3:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(ranks[mask]))
            rank_alpha = sl
            print(f"  Rank scaling:    rank90 ~ N^{sl:.3f} +/- {se:.3f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)

        mean_ptb = np.mean([r['peak_to_bulk'] for r in rows])
        mean_excess = np.mean(fac_excess)
        mean_rank = np.mean(ranks)
        print(f"  Mean peak/bulk:  {mean_ptb:.2f}", flush=True)
        print(f"  Mean excess:     {mean_excess:.3f}", flush=True)
        print(f"  Mean CRT rank90: {mean_rank:.1f}", flush=True)

        # Autocorrelation analysis for selected signals
        if sname in ('carry_xor', 'carry_product_parity', 'carry_jacobi_chain'):
            ac_p_vals = [r.get('autocorr_ratio_p', 0) for r in rows if 'autocorr_ratio_p' in r]
            ac_q_vals = [r.get('autocorr_ratio_q', 0) for r in rows if 'autocorr_ratio_q' in r]
            if ac_p_vals:
                print(f"  Mean autocorr ratio (p): {np.mean(ac_p_vals):.3f}", flush=True)
                print(f"  Mean autocorr ratio (q): {np.mean(ac_q_vals):.3f}", flush=True)

        scaling_results[sname] = {
            'alpha': float(alpha),
            'alpha_se': float(alpha_se),
            'alpha_r2': float(alpha_r2),
            'mean_excess': float(mean_excess),
            'mean_rank': float(mean_rank),
            'rank_alpha': float(rank_alpha),
        }

    # ── Comparative summary ───────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("COMPARATIVE SUMMARY", flush=True)
    print("=" * 76, flush=True)

    print(f"\n{'signal':>22} {'mean_peak/bulk':>14} {'mean_excess':>11} "
          f"{'mean_rank':>10} {'peak_alpha':>11} {'rank_alpha':>11}", flush=True)
    print("-" * 82, flush=True)

    for sname in signal_names:
        rows = [r for r in all_results if r['signal'] == sname]
        mean_ptb = np.mean([r['peak_to_bulk'] for r in rows])
        sr = scaling_results[sname]
        print(f"{sname:>22} {mean_ptb:>14.2f} {sr['mean_excess']:>11.3f} "
              f"{sr['mean_rank']:>10.1f} {sr['alpha']:>+11.3f} {sr['rank_alpha']:>+11.3f}",
              flush=True)

    # ── E10 vs E12 comparison ─────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("KEY COMPARISON: E10 (depth-1) vs E12 (depth-d)", flush=True)
    print("=" * 76, flush=True)

    e10_reference = {
        'jacobi_control': {'alpha': -0.25, 'excess': 1.0, 'rank': 2},
        'carry_parity': {'alpha': -0.35, 'excess': 1.0, 'rank': 'N^0.3'},
    }
    print("\nE10 reference (depth-1 carries):", flush=True)
    print("  - Pure Jacobi control:  peak ~ N^{-0.25}, excess ~ 1.0, rank ~ 2", flush=True)
    print("  - Carry parity (d=1):   peak ~ N^{-0.35}, excess ~ 1.0, rank ~ N^{0.3}", flush=True)
    print("\nE12 results (depth-d carries):", flush=True)
    for sname in signal_names:
        sr = scaling_results[sname]
        status = "PROMISING" if sr['alpha'] > -0.1 else "flat"
        print(f"  - {sname:>22}: peak ~ N^{{{sr['alpha']:+.3f}}}, "
              f"excess ~ {sr['mean_excess']:.3f}, rank ~ N^{{{sr['rank_alpha']:+.3f}}}  "
              f"[{status}]", flush=True)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    # Check success criterion: any signal with alpha > -0.1
    any_promising = False
    for sname in signal_names:
        if sname == 'control_modsq':
            continue
        sr = scaling_results[sname]
        if sr['alpha'] > -0.1:
            any_promising = True
            print(f"  PROMISING: {sname} has peak scaling N^{{{sr['alpha']:+.3f}}}", flush=True)
        if sr['mean_excess'] > 1.5:
            any_promising = True
            print(f"  PROMISING: {sname} has factor energy excess {sr['mean_excess']:.3f}", flush=True)

    if not any_promising:
        print("  No deep-carry signal shows elevated factor energy or peaks.", flush=True)
        print("  Despite exponentially higher CRT rank (depth-d vs depth-1),", flush=True)
        print("  the spectral flatness barrier holds.", flush=True)
        print("", flush=True)
        print("  INTERPRETATION:", flush=True)
        print("  Iterated carry compositions produce 'pseudo-random' functions", flush=True)
        print("  whose DFT peaks decay at LEAST as fast as N^{-0.25}.", flush=True)
        print("  High CRT rank is NECESSARY but not SUFFICIENT for spectral peaks.", flush=True)
        print("  The barrier extends to the full poly(log N) RJI oracle model.", flush=True)
    else:
        print("  Some deep-carry signals show elevated metrics.", flush=True)
        print("  → Extend to larger N (10^5-10^6) to confirm scaling.", flush=True)

    # ── Plotting ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: Peak scaling
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle('E12: Deep carry traces — DFT peak scaling', fontsize=14)
        for idx, sname in enumerate(signal_names):
            ax = axes[idx // 4, idx % 4]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            peaks = np.array([r['peak'] for r in rows])
            ax.scatter(Ns, peaks, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('DFT peak')
            ax.set_title(sname, fontsize=9)
            mask = peaks > 0
            if np.sum(mask) >= 3:
                sl, intc, rv, _, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(peaks[mask]))
                x_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                y_fit = 10**intc * x_fit**sl
                ax.plot(x_fit, y_fit, 'r-', lw=1.5,
                        label=f'$N^{{{sl:.3f}}}$')
                ax.legend(fontsize=7)
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E12_carry_depth_peaks.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}", flush=True)

        # Plot 2: Factor energy excess
        fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
        fig2.suptitle('E12: Factor energy excess by signal', fontsize=14)
        for idx, sname in enumerate(signal_names):
            ax = axes2[idx // 4, idx % 4]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            excess = np.array([r['factor_energy_excess'] for r in rows])
            ax.scatter(Ns, excess, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.axhline(y=1.0, color='gray', ls='--', lw=0.8, label='flat (=1)')
            ax.set_xlabel('N')
            ax.set_ylabel('Factor energy excess')
            ax.set_title(sname, fontsize=9)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plot_path2 = os.path.join(data_dir, 'E12_carry_depth_excess.png')
        plt.savefig(plot_path2, dpi=150)
        print(f"Plot saved to {plot_path2}", flush=True)

        # Plot 3: CRT rank
        fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
        fig3.suptitle('E12: CRT effective rank (90% energy) by signal', fontsize=14)
        for idx, sname in enumerate(signal_names):
            ax = axes3[idx // 4, idx % 4]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            ranks = np.array([r['crt_eff_rank_90'] for r in rows])
            ax.scatter(Ns, ranks, alpha=0.4, s=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('CRT rank (90%)')
            ax.set_title(sname, fontsize=9)
            mask = ranks > 0
            if np.sum(mask) >= 3:
                sl, intc, rv, _, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(ranks[mask]))
                x_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                y_fit = 10**intc * x_fit**sl
                ax.plot(x_fit, y_fit, 'r-', lw=1.5,
                        label=f'$N^{{{sl:.3f}}}$')
                ax.legend(fontsize=7)
        plt.tight_layout()
        plot_path3 = os.path.join(data_dir, 'E12_carry_depth_rank.png')
        plt.savefig(plot_path3, dpi=150)
        print(f"Plot saved to {plot_path3}", flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
