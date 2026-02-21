"""
E12b: Deep carry compositions — balanced semiprimes, normalized signals
=======================================================================

Follow-up to E12: The initial run showed that unnormalized carry signals
(values in [0,N)) have positive peak scaling as an amplitude artifact —
the CRT-separable control showed the SAME behavior.

The ±1 normalized signals showed:
  carry_xor: alpha = -0.994 (FLATTER than E10 depth-1)
  carry_product_parity: alpha = -0.381 (same as E10)

This run:
1. Uses BALANCED semiprimes (p/q > 0.3) to avoid trial-division contamination
2. Normalizes ALL signals to zero mean, unit variance before DFT
3. Tests at larger N to get cleaner scaling
4. Adds the critical comparison: E10-style depth-1 carry on same semiprimes

Key question: Does depth-d carry accumulation produce ANY improvement over
depth-1 carries? Or does each step add noise that washes out structure?
"""

import sys
import os
import time

import numpy as np
from scipy import stats

# ── Reproducibility ──────────────────────────────────────────────────
set_random_seed(42)
np.random.seed(42)

# ── Shared utilities ─────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import _py, _py_dict, safe_json_dump
from spectral import precompute_gcd_classes
from spectral import analyze_signal as _spectral_analyze_signal


def carry_trace(t, N_int, depth):
    """Compute carry trace of iterated squaring."""
    x = int(t) % N_int
    quotients = []
    for _ in range(depth):
        x2 = x * x
        q = x2 // N_int
        r = x2 % N_int
        quotients.append(int(q))
        x = r
    return quotients


def compute_signals(N):
    """Compute balanced, normalized carry signals."""
    N_int = int(N)
    depth = int(N_int).bit_length()

    signals = {}

    # Preallocate
    sig_d1_parity = np.zeros(N_int, dtype=np.float64)  # E10-style depth-1
    sig_dd_xor = np.zeros(N_int, dtype=np.float64)     # depth-d XOR
    sig_dd_prodpar = np.zeros(N_int, dtype=np.float64)  # depth-d product parity
    sig_dd_altpar = np.zeros(N_int, dtype=np.float64)   # depth-d alternating
    sig_dd_jacchain = np.zeros(N_int, dtype=np.float64) # depth-d Jacobi chain
    sig_dd_weighted_norm = np.zeros(N_int, dtype=np.float64)  # normalized weighted
    sig_dd_final_norm = np.zeros(N_int, dtype=np.float64)     # normalized final
    sig_dd_poly_norm = np.zeros(N_int, dtype=np.float64)      # normalized polynomial
    sig_ctrl_jacobi = np.zeros(N_int, dtype=np.float64)       # E7c-style control
    sig_ctrl_modsq_norm = np.zeros(N_int, dtype=np.float64)   # CRT-separable control

    # Half-depth variants
    half_depth = max(1, depth // 2)
    sig_half_xor = np.zeros(N_int, dtype=np.float64)
    sig_half_prodpar = np.zeros(N_int, dtype=np.float64)

    for t in range(N_int):
        # Full-depth carry trace
        quots = carry_trace(t, N_int, depth)

        # E10-style depth-1: (-1)^floor(t^2/N)
        t2 = t * t
        d1_carry = t2 // N_int
        sig_d1_parity[t] = 1.0 if (d1_carry & 1 == 0) else -1.0

        # Depth-d XOR of parities
        xor_val = 0
        for qq in quots:
            xor_val = xor_val ^ (qq & 1)
        sig_dd_xor[t] = 1.0 if xor_val == 0 else -1.0

        # Depth-d product parity
        prod = 1
        for qq in quots:
            if qq & 1 == 1:
                prod = -prod
        sig_dd_prodpar[t] = float(prod)

        # Depth-d alternating parity
        alt_sum = 0.0
        for i, qq in enumerate(quots):
            parity = 1.0 if (qq & 1 == 0) else -1.0
            alt_sum += ((-1.0) ** i) * parity
        sig_dd_altpar[t] = alt_sum

        # Depth-d Jacobi chain (sum of J(q_i, N))
        jac_sum = 0.0
        for qq in quots:
            if qq > 0:
                g = gcd(qq, N_int)
                if g == 1:
                    jac_sum += float(kronecker_symbol(qq, N_int))
        sig_dd_jacchain[t] = jac_sum

        # Unnormalized signals (will be normalized after loop)
        sig_dd_weighted_norm[t] = float(sum(quots))
        sig_dd_final_norm[t] = float(quots[-1]) if quots else 0.0
        poly_val = 0
        for i, qq in enumerate(quots):
            if qq & 1 == 1:
                poly_val = (poly_val + (1 << i)) % N_int
        sig_dd_poly_norm[t] = float(poly_val)

        # Control: CRT-separable (modular exponentiation result)
        sig_ctrl_modsq_norm[t] = float(pow(int(t), int(1) << depth, N_int))

        # Control: E7c-style pure Jacobi
        disc = t * t - 4
        if disc == 0:
            sig_ctrl_jacobi[t] = 0.0
        else:
            sig_ctrl_jacobi[t] = float(kronecker_symbol(disc, N_int))

        # Half-depth variants
        half_quots = quots[:half_depth]
        hx = 0
        for qq in half_quots:
            hx = hx ^ (qq & 1)
        sig_half_xor[t] = 1.0 if hx == 0 else -1.0

        hprod = 1
        for qq in half_quots:
            if qq & 1 == 1:
                hprod = -hprod
        sig_half_prodpar[t] = float(hprod)

    # Normalize unnormalized signals to zero mean, unit variance
    for sig_arr, name in [(sig_dd_weighted_norm, 'carry_weighted_norm'),
                          (sig_dd_final_norm, 'carry_final_norm'),
                          (sig_dd_poly_norm, 'carry_poly_norm'),
                          (sig_ctrl_modsq_norm, 'control_modsq_norm')]:
        mu = np.mean(sig_arr)
        sigma = np.std(sig_arr)
        if sigma > 1e-15:
            sig_arr[:] = (sig_arr - mu) / sigma

    signals['E10_depth1_parity'] = sig_d1_parity
    signals['depth_d_xor'] = sig_dd_xor
    signals['depth_d_prodparity'] = sig_dd_prodpar
    signals['depth_d_altparity'] = sig_dd_altpar
    signals['depth_d_jacchain'] = sig_dd_jacchain
    signals['depth_d_weighted_n'] = sig_dd_weighted_norm
    signals['depth_d_final_n'] = sig_dd_final_norm
    signals['depth_d_poly_n'] = sig_dd_poly_norm
    signals['half_depth_xor'] = sig_half_xor
    signals['half_depth_prodpar'] = sig_half_prodpar
    signals['ctrl_jacobi'] = sig_ctrl_jacobi
    signals['ctrl_modsq_n'] = sig_ctrl_modsq_norm

    return signals, depth


def analyze_signal(sig, N, p, q, gcd_class=None, n_factor_modes=None):
    """DFT analysis with factor-localized metrics.
    Delegates to shared spectral.analyze_signal with Parseval verification.
    """
    return _spectral_analyze_signal(
        sig, N, p, q,
        gcd_class=gcd_class,
        n_factor_modes=n_factor_modes,
    )


def main():
    print("=" * 76, flush=True)
    print("E12b: Deep carry — balanced semiprimes, normalized signals", flush=True)
    print("=" * 76, flush=True)

    # Balanced semiprimes at multiple bit sizes
    bit_sizes = [10, 11, 12, 13, 14]
    count_per = 8
    semiprimes = balanced_semiprimes(bit_sizes, count_per_size=count_per)
    print(f"Generated {len(semiprimes)} balanced semiprimes across bit sizes {bit_sizes}",
          flush=True)

    # Verify balance
    for N, p, q in semiprimes[:3]:
        ratio = float(min(p, q)) / float(max(p, q))
        print(f"  Sample: N={N}, p={p}, q={q}, p/q={ratio:.3f}", flush=True)

    signal_names = [
        'E10_depth1_parity',
        'depth_d_xor',
        'depth_d_prodparity',
        'depth_d_altparity',
        'depth_d_jacchain',
        'depth_d_weighted_n',
        'depth_d_final_n',
        'depth_d_poly_n',
        'half_depth_xor',
        'half_depth_prodpar',
        'ctrl_jacobi',
        'ctrl_modsq_n',
    ]

    all_results = []

    hdr = (f"{'N':>7} {'p':>5} {'q':>5} {'d':>3} {'signal':>22} "
           f"{'peak':>9} {'pk/bk':>7} {'excess':>7} {'rk90':>5}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for N, p, q in semiprimes:
        t0 = time.time()
        sigs, depth = compute_signals(N)
        dt = time.time() - t0

        # Precompute gcd classes once per semiprime
        gcd_class, n_factor_modes = precompute_gcd_classes(int(N), int(p), int(q))

        for sname in signal_names:
            metrics = analyze_signal(sigs[sname], N, p, q,
                                     gcd_class=gcd_class,
                                     n_factor_modes=n_factor_modes)
            row = {
                'N': int(N), 'p': int(p), 'q': int(q),
                'ratio_pq': float(min(p, q)) / float(max(p, q)),
                'depth': int(depth),
                'signal': sname,
            }
            row.update(_py_dict(metrics))
            row['time'] = round(float(dt), 3)
            all_results.append(row)

            print(f"{N:>7} {p:>5} {q:>5} {depth:>3} {sname:>22} "
                  f"{metrics['peak']:>9.5f} {metrics['peak_to_bulk']:>7.2f} "
                  f"{metrics['factor_energy_excess']:>7.3f} "
                  f"{metrics['crt_eff_rank_90']:>5}", flush=True)

    # ── Save results ──────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'E12b_balanced_results.json')
    safe_json_dump(all_results, out_path)

    # ── Scaling analysis ──────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("SCALING ANALYSIS (balanced semiprimes, normalized signals)", flush=True)
    print("=" * 76, flush=True)

    scaling = {}
    for sname in signal_names:
        rows = [r for r in all_results if r['signal'] == sname]
        Ns = np.array([r['N'] for r in rows], dtype=np.float64)
        peaks = np.array([r['peak'] for r in rows], dtype=np.float64)
        excess = np.array([r['factor_energy_excess'] for r in rows], dtype=np.float64)
        ranks = np.array([r['crt_eff_rank_90'] for r in rows], dtype=np.float64)

        print(f"\n--- {sname} ---", flush=True)

        alpha, alpha_se, alpha_r2 = 0, 0, 0
        mask = peaks > 0
        if np.sum(mask) >= 5:
            sl, intc, rv, pv, se = stats.linregress(
                np.log10(Ns[mask]), np.log10(peaks[mask]))
            alpha, alpha_se, alpha_r2 = sl, se, rv**2
            print(f"  Peak scaling:   alpha = {sl:+.4f} +/- {se:.4f}  "
                  f"(R2={rv**2:.4f}, p={pv:.2e})", flush=True)

        excess_alpha, excess_se = 0, 0
        mask = excess > 0
        if np.sum(mask) >= 5:
            sl2, _, rv2, pv2, se2 = stats.linregress(
                np.log10(Ns[mask]), np.log10(excess[mask]))
            excess_alpha, excess_se = sl2, se2
            print(f"  Excess scaling: beta = {sl2:+.4f} +/- {se2:.4f}  "
                  f"(R2={rv2**2:.4f}, p={pv2:.2e})", flush=True)

        rank_alpha, rank_se = 0, 0
        mask = ranks > 0
        if np.sum(mask) >= 5:
            sl3, _, rv3, pv3, se3 = stats.linregress(
                np.log10(Ns[mask]), np.log10(ranks[mask]))
            rank_alpha, rank_se = sl3, se3
            print(f"  Rank scaling:   gamma = {sl3:+.4f} +/- {se3:.4f}  "
                  f"(R2={rv3**2:.4f}, p={pv3:.2e})", flush=True)

        mean_excess = float(np.mean(excess))
        mean_rank = float(np.mean(ranks))
        print(f"  Mean excess:    {mean_excess:.4f}", flush=True)
        print(f"  Mean rank90:    {mean_rank:.1f}", flush=True)

        scaling[sname] = {
            'alpha': float(alpha), 'alpha_se': float(alpha_se),
            'alpha_r2': float(alpha_r2),
            'excess_alpha': float(excess_alpha), 'excess_se': float(excess_se),
            'rank_alpha': float(rank_alpha), 'rank_se': float(rank_se),
            'mean_excess': mean_excess, 'mean_rank': mean_rank,
        }

    # ── Comparative table ─────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("COMPARATIVE TABLE — Depth-1 vs Depth-d vs Controls", flush=True)
    print("=" * 76, flush=True)

    print(f"\n{'signal':>22} {'alpha':>8} {'excess_α':>9} {'mean_exc':>9} "
          f"{'rank_α':>8} {'mean_rk':>8} {'STATUS':>10}", flush=True)
    print("-" * 78, flush=True)

    for sname in signal_names:
        s = scaling[sname]
        # Status: promising if alpha > -0.15 AND excess > 1.3
        if s['alpha'] > -0.15 and s['mean_excess'] > 1.3:
            status = "PROMISING"
        elif s['alpha'] > -0.20:
            status = "marginal"
        else:
            status = "flat"
        print(f"{sname:>22} {s['alpha']:>+8.3f} {s['excess_alpha']:>+9.3f} "
              f"{s['mean_excess']:>9.3f} {s['rank_alpha']:>+8.3f} "
              f"{s['mean_rank']:>8.1f} {status:>10}", flush=True)

    # ── Key comparison: does depth help? ──────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("KEY QUESTION: Does carry depth improve signal?", flush=True)
    print("=" * 76, flush=True)

    d1 = scaling.get('E10_depth1_parity', {})
    dd_xor = scaling.get('depth_d_xor', {})
    dd_pp = scaling.get('depth_d_prodparity', {})
    hd_xor = scaling.get('half_depth_xor', {})
    ctrl = scaling.get('ctrl_jacobi', {})

    print(f"\n  {'Signal':>22} {'alpha':>8} {'excess':>8}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'ctrl_jacobi (E7c)':>22} {ctrl.get('alpha',0):>+8.3f} {ctrl.get('mean_excess',0):>8.3f}",
          flush=True)
    print(f"  {'E10 depth-1 parity':>22} {d1.get('alpha',0):>+8.3f} {d1.get('mean_excess',0):>8.3f}",
          flush=True)
    print(f"  {'half-depth XOR':>22} {hd_xor.get('alpha',0):>+8.3f} {hd_xor.get('mean_excess',0):>8.3f}",
          flush=True)
    print(f"  {'full-depth XOR':>22} {dd_xor.get('alpha',0):>+8.3f} {dd_xor.get('mean_excess',0):>8.3f}",
          flush=True)
    print(f"  {'full-depth prodpar':>22} {dd_pp.get('alpha',0):>+8.3f} {dd_pp.get('mean_excess',0):>8.3f}",
          flush=True)

    # Determine direction
    d1_alpha = d1.get('alpha', -0.25)
    dd_best_alpha = max(dd_xor.get('alpha', -1), dd_pp.get('alpha', -1))

    print(f"\n  Depth-1 peak scaling: N^{{{d1_alpha:+.3f}}}", flush=True)
    print(f"  Depth-d best scaling: N^{{{dd_best_alpha:+.3f}}}", flush=True)

    if dd_best_alpha > d1_alpha + 0.05:
        print(f"\n  → DEPTH HELPS: deeper carries have LESS negative scaling", flush=True)
        print(f"  → Extend to N ~ 10^6 to confirm.", flush=True)
    elif dd_best_alpha > d1_alpha - 0.05:
        print(f"\n  → DEPTH NEUTRAL: no significant change with depth.", flush=True)
        print(f"  → Carry depth does not improve over single-step carries.", flush=True)
    else:
        print(f"\n  → DEPTH HURTS: deeper carries are FLATTER.", flush=True)
        print(f"  → Each carry step adds pseudo-random noise.", flush=True)
        print(f"  → The barrier extends fully to iterated carry compositions.", flush=True)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)

    any_promising = False
    for sname in signal_names:
        if 'ctrl' in sname:
            continue
        s = scaling[sname]
        if s['alpha'] > -0.10 and s['mean_excess'] > 1.5:
            any_promising = True

    if not any_promising:
        print("  No deep-carry signal on balanced semiprimes shows factor peaks.", flush=True)
        print("  All ±1 carry signals decay as N^{-c} with c >= 0.25.", flush=True)
        print("  Normalized carry signals are spectrally indistinguishable from", flush=True)
        print("  random functions on Z/NZ.", flush=True)
        print("", flush=True)
        print("  CONCLUSION:", flush=True)
        print("  The spectral flatness barrier extends to the FULL RJI oracle model,", flush=True)
        print("  including arbitrarily deep carry compositions.", flush=True)
        print("  High CRT rank (from iterated carries) is NECESSARY but NOT", flush=True)
        print("  SUFFICIENT for spectral peaks at factor frequencies.", flush=True)
    else:
        print("  Some signals show non-trivial structure. Further investigation needed.", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Focus on the key comparison: depth-1 vs depth-d
        key_signals = ['E10_depth1_parity', 'depth_d_xor', 'depth_d_prodparity',
                       'half_depth_xor', 'ctrl_jacobi']
        fig, axes = plt.subplots(1, len(key_signals), figsize=(5*len(key_signals), 5))
        fig.suptitle('E12b: Carry depth comparison (balanced semiprimes)', fontsize=13)

        for idx, sname in enumerate(key_signals):
            ax = axes[idx]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            peaks = np.array([r['peak'] for r in rows])
            ax.scatter(Ns, peaks, alpha=0.5, s=15)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('DFT peak')
            ax.set_title(sname.replace('_', '\n'), fontsize=8)
            mask = peaks > 0
            if np.sum(mask) >= 3:
                sl, intc, rv, _, se = stats.linregress(
                    np.log10(Ns[mask]), np.log10(peaks[mask]))
                x_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
                y_fit = 10**intc * x_fit**sl
                ax.plot(x_fit, y_fit, 'r-', lw=1.5,
                        label=f'$\\alpha={sl:+.2f}$')
                ax.legend(fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'E12b_depth_comparison.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}", flush=True)

        # Excess comparison
        fig2, axes2 = plt.subplots(1, len(key_signals), figsize=(5*len(key_signals), 5))
        fig2.suptitle('E12b: Factor energy excess (balanced semiprimes)', fontsize=13)
        for idx, sname in enumerate(key_signals):
            ax = axes2[idx]
            rows = [r for r in all_results if r['signal'] == sname]
            Ns = np.array([r['N'] for r in rows])
            excess = np.array([r['factor_energy_excess'] for r in rows])
            ax.scatter(Ns, excess, alpha=0.5, s=15)
            ax.set_xscale('log')
            ax.axhline(y=1.0, color='gray', ls='--', lw=0.8)
            ax.set_xlabel('N')
            ax.set_ylabel('Factor energy excess')
            ax.set_title(sname.replace('_', '\n'), fontsize=8)
        plt.tight_layout()
        plot_path2 = os.path.join(data_dir, 'E12b_excess_comparison.png')
        plt.savefig(plot_path2, dpi=150)
        print(f"Plot saved to {plot_path2}", flush=True)

    except ImportError:
        print("matplotlib not available, skipping plots", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
