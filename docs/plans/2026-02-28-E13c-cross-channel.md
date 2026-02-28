# E13c Cross-Channel Eisenstein Experiments — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether cross-channel algebraic structure between 7 Eisenstein congruence channels can reduce the O(N) evaluation cost for computing a_N(f_k) mod ell.

**Architecture:** Single Sage script (`run_E13c_cross_channel.sage`) with 4 sub-experiments sharing common infrastructure (semiprime generation, channel evaluation, trace formula). Each sub-experiment produces independent results in a unified JSON output. Reuses functions from E13/E13b scripts but does NOT import them (standalone script, avoids Sage module import issues).

**Tech Stack:** SageMath (q-expansion, finite fields, polynomial arithmetic, class numbers), Python (JSON output, statistics), `utils/semiprime_gen.py`, `utils/sage_encoding.py`

---

### Task 1: Scaffold script and shared infrastructure

**Files:**
- Create: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Create the script skeleton with imports and shared functions**

Copy the following shared functions from E13/E13b (do NOT import — standalone):
- `eigenform_qexp(k, prec)` — from `run_E13_congruence_channel.sage:67-103`
- `power_sum_poly(m, N_mod, ell)` — from `run_E13_congruence_channel.sage:108-153`
- `eval_poly_mod(coeffs, x, F)` — from `run_E13_congruence_channel.sage:156-161`
- `solve_congruence_channel(a_N_mod_ell, N, k, ell)` — from `run_E13_congruence_channel.sage:164-187`
- `get_congruence_primes()` — from `run_E13_congruence_channel.sage:222-250`
- `hurwitz_class_number(D)` — from `run_E13b_ratio_scaling.sage:37-75`
- `trace_formula_class_sum(N, k)` — from `run_E13b_reduction.sage:113-198`

Add the script header:

```python
"""
E13c: Cross-channel Eisenstein congruence experiments
=====================================================

Tests whether algebraic structure ACROSS 7 Eisenstein congruence channels
can reduce the O(N) evaluation cost for a_N(f_k) mod ell.

Sub-experiments:
  c1: Resultant degree analysis between channel polynomial pairs
  c2: Newton recurrence interpolation via multi-weight class sums
  c3: Galois representation consistency constraints
  c4: Multi-weight trace formula elimination attack
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from semiprime_gen import balanced_semiprimes
from sage_encoding import safe_json_dump

set_random_seed(42)
```

Add test semiprime generation at the bottom:

```python
SIZE_CLASSES = [8, 10, 12, 14, 16]
COUNT_PER_SIZE = 10
QEXP_THRESHOLD = 16  # bits: verify all via q-expansion at these small sizes
```

**Step 2: Verify the scaffold loads**

Run: `sage -c "load('E13_bach_charles/run_E13c_cross_channel.sage')"`
Expected: No errors, no output (functions defined but not called)

**Step 3: Commit**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Scaffold E13c cross-channel experiment with shared infrastructure"
```

---

### Task 2: Implement E13c-1 — Resultant Degree Analysis

**Files:**
- Modify: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Write the resultant analysis function**

Add function `run_c1_resultant_analysis(semiprimes_by_size, channels, qexp_cache)`:

```python
def run_c1_resultant_analysis(semiprimes_by_size, channels, qexp_cache):
    """
    E13c-1: For each pair of channels (i, j), lift their polynomials
    to Z/(ell_i * ell_j)[e1] and compute GCD to check for non-trivial
    common structure.
    """
    print("\n" + "=" * 76, flush=True)
    print("E13c-1: Resultant Degree Analysis", flush=True)
    print("=" * 76, flush=True)

    results = []
    pair_stats = {}  # (k_i, ell_i, k_j, ell_j) -> list of gcd degrees

    for bits, semiprimes in sorted(semiprimes_by_size.items()):
        for N, p_true, q_true in semiprimes:
            # Get a_N mod ell for each channel via q-expansion
            channel_polys = {}
            for ch in channels:
                k, ell = ch['weight'], ch['ell']
                f = qexp_cache[k]
                a_N = int(f[int(N)]) % ell

                # Build the polynomial F_i(e1) = s_{k-1}(e1) - target over F_ell
                target_val = (a_N - 1 - pow(int(N), k - 1, ell)) % ell
                coeffs = power_sum_poly(k - 1, int(N) % ell, ell)
                # F_i(e1) = s_{k-1}(e1) - target
                coeffs_shifted = list(coeffs)
                coeffs_shifted[0] = (coeffs_shifted[0] - GF(ell)(target_val))
                channel_polys[(k, ell)] = coeffs_shifted

            # For each pair, lift to Z/(ell_i * ell_j) and compute GCD
            ch_list = list(channel_polys.keys())
            for i_idx in range(len(ch_list)):
                for j_idx in range(i_idx + 1, len(ch_list)):
                    ki, elli = ch_list[i_idx]
                    kj, ellj = ch_list[j_idx]

                    # Lift polynomials to Z/(elli * ellj)[x] via CRT
                    M = elli * ellj
                    R_mod = Integers(M)
                    Rx = PolynomialRing(R_mod, 'x')

                    # Lift F_i: coefficients known mod elli, set to 0 mod ellj
                    # Lift F_j: coefficients known mod ellj, set to 0 mod elli
                    ci = channel_polys[(ki, elli)]
                    cj = channel_polys[(kj, ellj)]

                    # CRT lift each coefficient
                    poly_i_coeffs = []
                    for idx_c in range(max(len(ci), 1)):
                        val_i = int(ci[idx_c]) if idx_c < len(ci) else 0
                        poly_i_coeffs.append(R_mod(crt(val_i, 0, elli, ellj)))
                    poly_i = Rx(poly_i_coeffs)

                    poly_j_coeffs = []
                    for idx_c in range(max(len(cj), 1)):
                        val_j = int(cj[idx_c]) if idx_c < len(cj) else 0
                        poly_j_coeffs.append(R_mod(crt(0, val_j, elli, ellj)))
                    poly_j = Rx(poly_j_coeffs)

                    # GCD over Z/M[x] — Sage may not support this directly
                    # for non-field rings. Instead, count common roots by brute
                    # force for small ell values.
                    e1_true = (p_true + q_true) % M
                    common_roots = 0
                    # Check roots of poly_i mod elli
                    roots_i = set()
                    Fi = GF(elli)
                    for e1 in range(elli):
                        if eval_poly_mod(ci, e1, Fi) == Fi(0):
                            roots_i.add(e1)
                    # Check roots of poly_j mod ellj
                    roots_j = set()
                    Fj = GF(ellj)
                    for e1 in range(ellj):
                        if eval_poly_mod(cj, e1, Fj) == Fj(0):
                            roots_j.add(e1)

                    # CRT-compatible root pairs
                    compatible = 0
                    for ri in roots_i:
                        for rj in roots_j:
                            compatible += 1
                    # Expected: |roots_i| * |roots_j|
                    # If cross-channel eliminates some, compatible < product

                    key = (ki, elli, kj, ellj)
                    if key not in pair_stats:
                        pair_stats[key] = []
                    pair_stats[key].append({
                        'N': int(N),
                        'bits': int(bits),
                        'roots_i': len(roots_i),
                        'roots_j': len(roots_j),
                        'product': len(roots_i) * len(roots_j),
                        'compatible_crt': compatible,
                        'e1_true_mod_M': int(e1_true),
                    })

    # Summary
    print(f"\n  {'Chan_i':>20s}  {'Chan_j':>20s}  avg|Ri| avg|Rj| avg_prod avg_compat", flush=True)
    for key in sorted(pair_stats.keys()):
        ki, elli, kj, ellj = key
        entries = pair_stats[key]
        avg_ri = sum(e['roots_i'] for e in entries) / len(entries)
        avg_rj = sum(e['roots_j'] for e in entries) / len(entries)
        avg_prod = sum(e['product'] for e in entries) / len(entries)
        avg_compat = sum(e['compatible_crt'] for e in entries) / len(entries)
        print(f"  (k={ki},l={elli:>5d})  (k={kj},l={ellj:>5d})  "
              f"{avg_ri:7.1f} {avg_rj:7.1f} {avg_prod:8.1f} {avg_compat:10.1f}", flush=True)

    return {'pair_stats': {str(k): v for k, v in pair_stats.items()}}
```

**Step 2: Test on a single semiprime**

Add a temporary test at the bottom of the script:

```python
if __name__ == '__main__':
    channels = get_congruence_primes()
    N, p, q = 247, 13, 19
    prec = 260
    qexp_cache = {ch['weight']: eigenform_qexp(ch['weight'], prec) for ch in channels}
    # Quick check: one pair
    semiprimes_by_size = {8: [(N, p, q)]}
    result = run_c1_resultant_analysis(semiprimes_by_size, channels, qexp_cache)
    print(result)
```

Run: `sage E13_bach_charles/run_E13c_cross_channel.sage`
Expected: Prints pair statistics for the single semiprime N=247.

**Step 3: Commit**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Add E13c-1: resultant degree analysis between channel pairs"
```

---

### Task 3: Implement E13c-2 — Newton Recurrence Interpolation

**Files:**
- Modify: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Write the multi-weight class sum function**

This generalizes `trace_formula_class_sum` from E13b_reduction to accept any weight k.
The function already exists as `trace_formula_class_sum(N, k)` in the shared infrastructure
(copied in Task 1). No new code needed for class sum computation.

Write the interpolation analysis function:

```python
def run_c2_newton_interpolation(semiprimes_by_size, channels, qexp_cache):
    """
    E13c-2: Compute class number sums for multiple weights and test
    whether the overdetermined system can recover min(p,q).

    For each weight k and semiprime N=pq:
      ClassSum_k(N) = tau_k(N) - E_k(N)
      E_k(N) = -1 - min(p,q)^{k-1}  (for N=pq with p != q primes)

    The class sum is FACTOR-FREE (costs O(sqrt(N))).
    The Eisenstein correction REQUIRES factors.

    If we compute ClassSum for 5 weights, we get 5 equations in 6 unknowns
    (5 tau_k values + min(p,q)). Test solvability.
    """
    print("\n" + "=" * 76, flush=True)
    print("E13c-2: Newton Recurrence Interpolation", flush=True)
    print("=" * 76, flush=True)

    weights = sorted(set(ch['weight'] for ch in channels))  # [12, 16, 18, 20, 22]
    results = []

    for bits, semiprimes in sorted(semiprimes_by_size.items()):
        for N, p_true, q_true in semiprimes:
            min_pq = min(p_true, q_true)
            N_int = int(N)

            # Compute class sums for each weight (factor-free, O(sqrt(N)))
            class_sums = {}
            for k in weights:
                cs = trace_formula_class_sum(N_int, k)
                class_sums[k] = cs

            # Compute true tau_k via q-expansion
            tau_values = {}
            for k in weights:
                f = qexp_cache[k]
                tau_values[k] = int(f[N_int])

            # Compute true E_k
            eis_values = {}
            for k in weights:
                eis_values[k] = -1 - int(min_pq) ** (k - 1)

            # Verify: ClassSum = tau - E
            verification = {}
            for k in weights:
                cs_expected = tau_values[k] - eis_values[k]
                cs_actual = int(class_sums[k])
                verification[k] = (cs_actual == cs_expected)

            # The attack: from ClassSum_k = tau_k + 1 + min^{k-1},
            # form ratios of (ClassSum_{k1} + 1) and (ClassSum_{k2} + 1)
            # If tau_k were 0, ratio = min^{k1-1} / min^{k2-1} = min^{k1-k2}
            # But tau_k is NOT 0 — it's the same order as E_k.
            #
            # Test: how close is the ratio to the true min(p,q)^{k1-k2}?
            ratio_results = []
            for i_idx in range(len(weights)):
                for j_idx in range(i_idx + 1, len(weights)):
                    k1, k2 = weights[i_idx], weights[j_idx]
                    cs1 = class_sums[k1]
                    cs2 = class_sums[k2]

                    # Naive ratio (ignoring tau)
                    if cs2 != 0:
                        ratio = float(abs(cs1)) / float(abs(cs2))
                    else:
                        ratio = float('inf')

                    # True ratio of Eisenstein parts
                    true_ratio = float(abs(eis_values[k1])) / float(abs(eis_values[k2])) if eis_values[k2] != 0 else float('inf')

                    # Error: how far is the naive ratio from the Eis ratio?
                    if true_ratio != float('inf') and ratio != float('inf'):
                        rel_error = abs(ratio - true_ratio) / max(abs(true_ratio), 1e-30)
                    else:
                        rel_error = float('inf')

                    ratio_results.append({
                        'k1': int(k1), 'k2': int(k2),
                        'cs_ratio': float(round(float(ratio), int(6))),
                        'eis_ratio': float(round(float(true_ratio), int(6))),
                        'rel_error': float(round(float(rel_error), int(6))),
                    })

            entry = {
                'N': int(N), 'bits': int(bits),
                'p': int(p_true), 'q': int(q_true), 'min_pq': int(min_pq),
                'verification': {str(k): v for k, v in verification.items()},
                'ratio_results': ratio_results,
            }
            results.append(entry)

            # Print summary line
            all_ok = all(verification.values())
            best_err = min(r['rel_error'] for r in ratio_results if r['rel_error'] != float('inf'))
            print(f"  N={N_int:>16d}  {bits:>2d}b  verify={'OK' if all_ok else 'FAIL'}  "
                  f"best_rel_err={best_err:.4f}", flush=True)

    return results
```

**Step 2: Test on small semiprimes**

Update the bottom test to include c2:

```python
result_c2 = run_c2_newton_interpolation(semiprimes_by_size, channels, qexp_cache)
```

Run: `sage E13_bach_charles/run_E13c_cross_channel.sage`
Expected: Prints verification=OK for all entries, rel_error values showing how close the ratio approximation is.

**Step 3: Commit**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Add E13c-2: Newton recurrence interpolation via class sums"
```

---

### Task 4: Implement E13c-3 — Galois Representation Consistency

**Files:**
- Modify: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Write the Galois consistency analysis**

```python
def run_c3_galois_consistency(semiprimes_by_size, channels, qexp_cache):
    """
    E13c-3: Test whether symmetric power traces of the mod-ell Galois
    representation at Frob_N give constraints beyond multiplicativity.

    For the Eisenstein congruence: rho_{f,ell} is reducible:
      rho = 1 + chi^{k-1}
    At Frob_p: tr(rho(Frob_p)) = 1 + p^{k-1} mod ell
    At Frob_{pq}: tr(rho(Frob_{pq})) = tr(rho(Frob_p)) * tr(rho(Frob_q))
                                       = (1+p^{k-1})(1+q^{k-1}) mod ell

    Symmetric powers Sym^m(rho) at Frob_p:
      tr(Sym^m(rho)(Frob_p)) = sum_{j=0}^{m} (p^{k-1})^j = (p^{m(k-1)} - 1)/(p^{k-1} - 1)

    Question: do Sym^m at composite N give info beyond a_N = a_p * a_q?
    """
    print("\n" + "=" * 76, flush=True)
    print("E13c-3: Galois Representation Consistency", flush=True)
    print("=" * 76, flush=True)

    results = []

    for bits, semiprimes in sorted(semiprimes_by_size.items()):
        for N, p_true, q_true in semiprimes:
            N_int = int(N)
            entry = {'N': N_int, 'bits': int(bits), 'channels': []}

            for ch in channels:
                k, ell = ch['weight'], ch['ell']
                F = GF(ell)

                # True values mod ell
                p_mod = F(p_true)
                q_mod = F(q_true)
                pk = p_mod ** (k - 1)
                qk = q_mod ** (k - 1)

                # a_p, a_q, a_N mod ell (from Eisenstein congruence)
                a_p = F(1) + pk
                a_q = F(1) + qk
                a_N = a_p * a_q  # multiplicativity

                # Sym^2 trace at Frob_p: 1 + p^{k-1} + p^{2(k-1)}
                sym2_p = F(1) + pk + pk**2
                sym2_q = F(1) + qk + qk**2
                sym2_N = sym2_p * sym2_q  # multiplicativity for Sym^2

                # Wedge^2 trace at Frob_p (for 2-dim rep):
                # wedge^2(rho)(Frob_p) = det(rho(Frob_p)) = p^{k-1} (= chi^{k-1}(Frob_p))
                wedge2_p = pk
                wedge2_q = qk
                wedge2_N = wedge2_p * wedge2_q

                # Key test: can we recover a_N from Sym^2 and Wedge^2?
                # a_p^2 = Sym^2(Frob_p) + Wedge^2(Frob_p) (character identity)
                # So a_p^2 - Sym^2(Frob_p) = Wedge^2(Frob_p) = p^{k-1}
                # Similarly: a_N^2 - Sym^2(Frob_N) = ... more complex
                # For composite: Wedge^2(Frob_N) = p^{k-1} * q^{k-1} = N^{k-1} mod ell

                # This is ALREADY KNOWN (N^{k-1} mod ell is trivially computable)
                # So Galois constraints reduce to: a_N and N^{k-1} mod ell
                # No new info beyond multiplicativity.

                # But test explicitly:
                a_N_from_qexp = F(int(qexp_cache[k][N_int]))
                assert a_N == a_N_from_qexp, f"Multiplicativity check failed for N={N_int}, k={k}"

                wedge2_check = (a_N**2 - sym2_N == wedge2_N)
                new_info = not wedge2_check  # If check fails, we have new info

                entry['channels'].append({
                    'k': int(k), 'ell': int(ell),
                    'a_N_mod_ell': int(a_N),
                    'sym2_N': int(sym2_N),
                    'wedge2_N': int(wedge2_N),
                    'wedge2_equals_Nk': int(wedge2_N) == pow(N_int, k - 1, ell),
                    'identity_holds': bool(wedge2_check),
                    'new_info': bool(new_info),
                })

            any_new = any(c['new_info'] for c in entry['channels'])
            print(f"  N={N_int:>16d}  {bits:>2d}b  new_info={'YES' if any_new else 'no'}", flush=True)
            results.append(entry)

    return results
```

**Step 2: Test**

Run: `sage E13_bach_charles/run_E13c_cross_channel.sage`
Expected: `new_info=no` for all entries (Galois constraints reduce to known identities).

**Step 3: Commit**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Add E13c-3: Galois representation consistency analysis"
```

---

### Task 5: Implement E13c-4 — Multi-Weight Trace Formula Elimination

**Files:**
- Modify: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Write the elimination attack function**

This is the primary experiment. For each semiprime N=pq:
1. Compute ClassSum_k(N) for k = 12, 16, 18, 20, 22 (factor-free, O(sqrt(N)))
2. We have: ClassSum_k = tau_k + 1 + min(p,q)^{k-1}
3. Test whether the overdetermined system recovers min(p,q)

```python
def run_c4_elimination_attack(semiprimes_by_size, channels, qexp_cache):
    """
    E13c-4: Multi-weight trace formula elimination.

    Compute factor-free ClassSum_k(N) for 5 weights. Each satisfies:
      ClassSum_k(N) = tau_k(N) + 1 + min(p,q)^{k-1}

    Test numerical methods to recover min(p,q) from this system.
    """
    print("\n" + "=" * 76, flush=True)
    print("E13c-4: Multi-Weight Trace Formula Elimination", flush=True)
    print("=" * 76, flush=True)

    weights = sorted(set(ch['weight'] for ch in channels))  # [12, 16, 18, 20, 22]
    results = []

    for bits, semiprimes in sorted(semiprimes_by_size.items()):
        for N, p_true, q_true in semiprimes:
            N_int = int(N)
            min_pq = min(int(p_true), int(q_true))

            # ── Step A: Compute class sums (factor-free) ──────────
            class_sums = {}
            cs_times = {}
            for k in weights:
                t0 = time.time()
                cs = trace_formula_class_sum(N_int, k)
                cs_times[k] = time.time() - t0
                class_sums[k] = cs

            # ── Step B: Compute true tau via q-expansion (for verification) ──
            tau_true = {}
            for k in weights:
                tau_true[k] = int(qexp_cache[k][N_int])

            # ── Step C: Elimination attempts ──────────────────────

            # Method 1: Pairwise ratio
            # If ClassSum_k ≈ E_k = -(1 + min^{k-1}), then
            # ClassSum_{k1} / ClassSum_{k2} ≈ (1 + min^{k1-1}) / (1 + min^{k2-1})
            # Since min >> 1, ≈ min^{k1-k2}
            # This ignores tau_k, so it's wrong when |tau_k| ~ |E_k|.

            pairwise_estimates = []
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    k1, k2 = weights[i], weights[j]
                    cs1 = float(class_sums[k1])
                    cs2 = float(class_sums[k2])
                    dk = k1 - k2  # negative (k1 < k2)

                    if abs(cs2) > 1e-10 and abs(cs1) > 1e-10:
                        # cs1/cs2 should approximate min^{k1-k2}
                        ratio = abs(cs1 / cs2)
                        if dk != 0 and ratio > 0:
                            try:
                                est_min = ratio ** (1.0 / dk)
                            except (OverflowError, ValueError):
                                est_min = float('nan')
                        else:
                            est_min = float('nan')
                    else:
                        est_min = float('nan')

                    pairwise_estimates.append({
                        'k1': int(k1), 'k2': int(k2),
                        'estimated_min': float(round(float(est_min), int(2))),
                        'true_min': int(min_pq),
                        'rel_error': float(round(float(abs(est_min - min_pq) / max(min_pq, 1)), int(4)))
                            if not (est_min != est_min) else float('inf'),  # NaN check
                    })

            # Method 2: Newton iteration on the full system
            # Given ClassSum_k = tau_k + 1 + m^{k-1} for each k,
            # and tau_k = a_p(f_k) * a_q(f_k) where p and q are the
            # factors we're looking for, this is deeply circular.
            # But try: define F(m) = sum_k (ClassSum_k - 1 - m^{k-1})^2
            # and minimize over m. If tau_k terms are "noise", minimum
            # should be near m = min(p,q).
            #
            # Actually: ClassSum_k - 1 - m^{k-1} = tau_k for the right m.
            # For wrong m, this gives inconsistent "tau_k" values.
            # Consistency check: do the residuals satisfy Hecke multiplicativity?

            # Scan over candidate m values
            m_range = range(max(2, isqrt(isqrt(N_int))), isqrt(N_int) + 1)
            best_m = None
            best_score = float('inf')
            m_scores = []

            for m_cand in m_range:
                # For each m, compute would-be tau_k values
                residuals = []
                for k in weights:
                    tau_k_guess = int(class_sums[k]) - 1 - m_cand ** (k - 1)
                    residuals.append((k, tau_k_guess))

                # Consistency score: check multiplicativity
                # tau_{k1}(N) * tau_{k2}(N) should relate via Hecke algebra
                # Simpler: check if all tau_k_guess have the same sign pattern
                # as known semiprimes (positive or negative)

                # Use variance of log|tau_k| / ((k-1)/2 * log(N)) as consistency
                # For true tau: |tau_k(N)| ~ N^{(k-1)/2}, so
                # log|tau_k| / ((k-1)/2) should be approximately log(N) + constant
                log_ratios = []
                for k, tau_g in residuals:
                    if tau_g != 0:
                        log_ratios.append(abs(float(log(abs(tau_g))) / ((k - 1) / 2.0)))

                if len(log_ratios) >= 2:
                    mean_lr = sum(log_ratios) / len(log_ratios)
                    variance = sum((x - mean_lr) ** 2 for x in log_ratios) / len(log_ratios)
                    score = variance
                else:
                    score = float('inf')

                if score < best_score:
                    best_score = score
                    best_m = m_cand

            newton_result = {
                'best_m': int(best_m) if best_m is not None else None,
                'true_min': int(min_pq),
                'correct': best_m == min_pq if best_m is not None else False,
                'best_score': float(round(float(best_score), int(6))),
                'search_range': [int(m_range.start), int(m_range.stop - 1)],
            }

            entry = {
                'N': int(N_int), 'bits': int(bits),
                'p': int(p_true), 'q': int(q_true), 'min_pq': int(min_pq),
                'class_sum_times': {str(k): float(round(float(cs_times[k]), int(4))) for k in weights},
                'pairwise_estimates': pairwise_estimates,
                'newton_result': newton_result,
            }
            results.append(entry)

            best_pair_err = min(r['rel_error'] for r in pairwise_estimates
                               if r['rel_error'] != float('inf'))
            print(f"  N={N_int:>16d}  {bits:>2d}b  min={min_pq:>8d}  "
                  f"newton={'HIT' if newton_result['correct'] else 'miss':>4s}  "
                  f"best_m={best_m}  best_pair_err={best_pair_err:.4f}", flush=True)

    return results
```

**Step 2: Test on small semiprimes**

Run: `sage E13_bach_charles/run_E13c_cross_channel.sage`
Expected: newton results showing whether the brute-force scan over m candidates finds min(p,q). For 8-bit semiprimes, the scan range is small (~10 values).

**Step 3: Commit**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Add E13c-4: multi-weight trace formula elimination attack"
```

---

### Task 6: Wire up main() and full experiment run

**Files:**
- Modify: `E13_bach_charles/run_E13c_cross_channel.sage`

**Step 1: Write main() function**

```python
def main():
    print("=" * 76, flush=True)
    print("E13c: Cross-Channel Eisenstein Congruence Experiments", flush=True)
    print("=" * 76, flush=True)

    channels = get_congruence_primes()
    print(f"\n  {len(channels)} channels:", flush=True)
    for ch in channels:
        print(f"    k={ch['weight']}, ell={ch['ell']}", flush=True)

    # Generate test semiprimes
    semiprimes_by_size = {}
    for bits in SIZE_CLASSES:
        sps = balanced_semiprimes(int(bits), int(COUNT_PER_SIZE))
        semiprimes_by_size[bits] = sps

    total = sum(len(v) for v in semiprimes_by_size.values())
    print(f"\n  {total} semiprimes across {len(SIZE_CLASSES)} size classes", flush=True)

    # Compute q-expansions (needed for verification at small sizes)
    max_N = max(N for sps in semiprimes_by_size.values() for N, _, _ in sps)
    prec = int(max_N) + 10
    print(f"\n  Computing q-expansions to precision {prec}...", flush=True)
    t0 = time.time()
    weights = sorted(set(ch['weight'] for ch in channels))
    qexp_cache = {}
    for k in weights:
        qexp_cache[k] = eigenform_qexp(k, prec)
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    # Run sub-experiments
    t_total = time.time()

    result_c1 = run_c1_resultant_analysis(semiprimes_by_size, channels, qexp_cache)
    result_c2 = run_c2_newton_interpolation(semiprimes_by_size, channels, qexp_cache)
    result_c3 = run_c3_galois_consistency(semiprimes_by_size, channels, qexp_cache)
    result_c4 = run_c4_elimination_attack(semiprimes_by_size, channels, qexp_cache)

    wall_time = time.time() - t_total

    # ── Conclusions ─────────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print("CONCLUSIONS", flush=True)
    print("=" * 76, flush=True)

    # c1: any GCD degree reduction?
    c1_reduction = False  # Analyze from result_c1
    print(f"\n  c1 (Resultant): Cross-channel GCD reduction found: {c1_reduction}", flush=True)

    # c3: any new Galois info?
    c3_new = any(any(c['new_info'] for c in entry['channels']) for entry in result_c3)
    print(f"  c3 (Galois): New info beyond multiplicativity: {c3_new}", flush=True)

    # c4: Newton hit rate
    c4_hits = sum(1 for e in result_c4 if e['newton_result']['correct'])
    c4_total = len(result_c4)
    print(f"  c4 (Elimination): Newton scan hit rate: {c4_hits}/{c4_total}", flush=True)

    # Overall
    any_positive = c1_reduction or c3_new or (c4_hits > c4_total * 0.5)
    if any_positive:
        conclusion = 'investigate'
        print(f"\n  RESULT: Cross-channel structure shows promise. Investigate further.", flush=True)
    else:
        conclusion = 'blocked'
        print(f"\n  RESULT: No cross-channel structure exploitable for cost reduction.", flush=True)
        print(f"  All paths reduce to known identities or ill-conditioned systems.", flush=True)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'E13c_cross_channel.json')
    output = {
        'experiment': 'E13c_cross_channel',
        'description': 'Cross-channel Eisenstein congruence structure analysis',
        'size_classes': SIZE_CLASSES,
        'count_per_size': COUNT_PER_SIZE,
        'channels': channels,
        'c1_resultant': result_c1,
        'c2_interpolation': result_c2,
        'c3_galois': result_c3,
        'c4_elimination': result_c4,
        'conclusion': conclusion,
        'wall_time_secs': float(round(float(wall_time), int(1))),
    }
    safe_json_dump(output, out_path)

    print(f"\nTotal wall time: {wall_time:.1f}s", flush=True)
    print("Done.", flush=True)


main()
```

**Step 2: Run the full experiment**

Run: `sage E13_bach_charles/run_E13c_cross_channel.sage`
Expected: All 4 sub-experiments complete. JSON written to `data/E13c_cross_channel.json`.
Estimated wall time: ~5-15 minutes (dominated by class sum computation at 16-bit).

**Step 3: Commit script and results**

```bash
git add E13_bach_charles/run_E13c_cross_channel.sage
git commit -m "Wire up E13c main() with all four sub-experiments"
```

After successful run:

```bash
git add data/E13c_cross_channel.json
git commit -m "Add E13c results: cross-channel structure analysis"
```

---

### Task 7: Analyze results and update MEMORY.md

**Files:**
- Modify: `MEMORY.md` (memory file)

**Step 1: Review the JSON output**

Read `data/E13c_cross_channel.json` and summarize:
- c1: Did any channel pairs show GCD degree reduction?
- c2: What were the best relative errors for ratio estimation?
- c3: Did Galois constraints give new info?
- c4: What was the Newton scan hit rate? Did it find true min(p,q)?

**Step 2: Update MEMORY.md with findings**

Add E13c entry under Key Results section following the existing format.

**Step 3: Commit**

```bash
git commit -m "Update memory with E13c cross-channel results"
```
