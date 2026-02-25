//! E21: Dominant eigenvector multiplicative character structure verification.
//!
//! ## Theoretical background
//!
//! For prime ℓ and weight k, define:
//!   g(p)  = (1 + p^{k−1}) mod ℓ
//!   M[p][q] = (σ_{k−1}(pq) mod ℓ) mod 2  =  parity(g(p)·g(q) mod ℓ)
//!
//! The factorisation M[p][q] = h(g(p)·g(q) mod ℓ) with h = parity means that
//! M has the Schur-product structure of a matrix indexed by a cyclic group.
//! Its eigenvectors are (restrictions to the prime set of) multiplicative
//! characters χ_r of (ℤ/ℓℤ)*.
//!
//! ## E21 tests
//!
//! For each (n_bits, channel):
//!   1. Build M and extract top-k eigenvectors u₁, u₂, …
//!   2. For u_i: find r* = argmax |⟨u_i, χ_r(g(·))⟩|  (character scan)
//!   3. Report Pearson corr(u_i(p), Re(χ_{r*}(g(p))))  ← "eigenvector ≈ character?"
//!   4. Product test A: corr(u_i(p)·u_i(q), Re(χ_{r*}(σ_{k−1}(N) mod ℓ)))
//!      ← predicted = Re(χ_{r*}(g(p)·g(q) mod ℓ)) = Re(χ_{r*}(σ_{k−1}(N) mod ℓ))
//!   5. Product test B: corr(u_i(p)·u_i(q), Re(χ_{r*}(N^{k−1} mod ℓ)))
//!      ← tests whether the product is determined by N alone (computable from N)
//!
//! Interpretation:
//!   Test 3 ≈ 1   : eigenvector has multiplicative character structure (expected)
//!   Test 4 ≈ 1   : product is function of σ_{k−1}(N) mod ℓ (expected)
//!   Test 5 ≈ 1   : product is function of N^{k−1} mod ℓ alone (would open corridor)
//!   Test 5 ≈ 0   : barrier holds — σ_{k−1}(N) mod ℓ cannot be read from N alone

pub mod algebra;
pub mod eigen;

use eisenstein_hunt::{
    arith::{is_prime_u64, mod_pow},
    Channel, Semiprime,
};
use algebra::im_char;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;

// E24 NFS imports
use classical_nfs::polynomial::{select_polynomial, NfsPolynomial};
use classical_nfs::sieve::{eval_homogeneous_abs, gcd_u64, sieve_primes};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Per-eigenvector character-matching result.
#[derive(Debug, Clone, Serialize)]
pub struct EigenvectorCharResult {
    /// 0 = top eigenvector (largest |λ|).
    pub eigenvector_idx: usize,
    /// Signed Rayleigh quotient v^T M v.
    pub eigenvalue: f64,
    /// Character index r* with maximum amplitude.
    pub best_char_r: usize,
    /// Amplitude |⟨u, χ_{r*}(g(·))⟩| / (‖u‖ · √n_valid).  Range [0, 1].
    pub best_char_amp: f64,
    /// Number of character frequencies scanned (= (ℓ−1)/2 + 1).
    pub n_chars_scanned: usize,
    /// Pearson corr(u(p), Re(χ_{r*}(g(p)))).
    /// Near ±1: eigenvector is the real part of a multiplicative character.
    pub corr_re_best: f64,
    /// Pearson corr(u(p), Im(χ_{r*}(g(p)))).
    /// Non-zero: eigenvector has an imaginary-part component (complex character).
    pub corr_im_best: f64,
    /// Product test A: Pearson corr(u(p)·u(q), Re(χ_{r*}(σ_{k−1}(N) mod ℓ))).
    /// σ_{k−1}(N) = g(p)·g(q) mod ℓ requires knowing p, q.
    pub product_corr_sigma: f64,
    /// Product test B: Pearson corr(u(p)·u(q), Re(χ_{r*}(N^{k−1} mod ℓ))).
    /// N^{k−1} mod ℓ IS computable from N alone.  Near 0 = barrier intact.
    pub product_corr_nk: f64,
    /// Number of valid pairs used in product tests (both g(p) and g(q) ≠ 0).
    pub n_pairs_used: usize,
    /// Top-10 characters by amplitude: [(r, amplitude)].
    pub top_characters: Vec<(usize, f64)>,
}

/// Full per-(n_bits, channel) result.
#[derive(Debug, Clone, Serialize)]
pub struct CharacterAuditResult {
    pub n_bits: u32,
    pub channel_weight: u32,
    pub channel_ell: u64,
    /// Number of distinct primes in the n_bits-bit balanced-semiprime set.
    pub n_primes: usize,
    /// Number of valid (p, q) pairs forming n_bits-bit balanced semiprimes.
    pub n_valid_pairs: usize,
    /// Number of primes p where g(p) = 0 mod ℓ (excluded from character scan).
    pub n_zero_g: usize,
    /// Primitive root of ℓ used for the discrete-log table.
    pub primitive_root: u64,
    /// Per-eigenvector results (length = min(n_eigenvectors, n_primes)).
    pub eigenvectors: Vec<EigenvectorCharResult>,
}

// ---------------------------------------------------------------------------
// Semiprime enumeration (re-implemented to avoid path-sum-degree dependency)
// ---------------------------------------------------------------------------

/// Enumerate all balanced semiprimes N = p·q that are exactly `n_bits` bits,
/// with p ≤ q and p/q ≥ 0.3.  Returns sorted (p, q) pairs.
///
/// Only feasible for n_bits ≤ 24.
pub fn enumerate_balanced_semiprimes(n_bits: u32) -> Vec<(u64, u64)> {
    assert!(n_bits >= 4 && n_bits <= 24, "only feasible for n_bits ≤ 24");
    let half = n_bits / 2;
    let lo   = 1u64 << (half - 1);
    let hi   = 1u64 << (half + 1);

    let primes: Vec<u64> = (lo..hi).filter(|&x| x % 2 == 1 && is_prime_u64(x)).collect();

    let n_lo = 1u64 << (n_bits - 1);
    let n_hi = 1u64 << n_bits;

    let mut pairs = Vec::new();
    for (i, &p) in primes.iter().enumerate() {
        for &q in &primes[i..] {
            if p == q {
                continue;
            }
            let n = p as u128 * q as u128;
            if n < n_lo as u128 || n >= n_hi as u128 {
                continue;
            }
            if (p as f64) / (q as f64) < 0.3 {
                continue;
            }
            pairs.push((p, q));
        }
    }
    pairs
}

/// Sample `count` distinct odd primes in `[lo, hi)` using a seeded RNG.
///
/// Used for bit sizes > 24 where full enumeration is infeasible.
/// The RNG is deterministic (seeded), so results are reproducible.
fn sample_primes_in_range(lo: u64, hi: u64, count: usize, seed: u64) -> Vec<u64> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut primes = std::collections::BTreeSet::new();
    let max_attempts = count * 200;
    let mut attempts = 0;
    while primes.len() < count && attempts < max_attempts {
        let candidate = rng.gen_range(lo..hi) | 1; // ensure odd
        if candidate >= lo && candidate < hi && is_prime_u64(candidate) {
            primes.insert(candidate);
        }
        attempts += 1;
    }
    primes.into_iter().collect()
}

/// Generate primes and balanced-semiprime pairs for a given bit size.
///
/// For `n_bits ≤ 24`: enumerates all balanced semiprimes exactly.
/// For `n_bits > 24`: samples `target_primes` random primes in the half-bit
/// range and forms pairs, capped at `max_pairs` for runtime.
fn generate_primes_and_pairs(
    n_bits: u32,
    target_primes: usize,
    max_pairs: usize,
    seed: u64,
) -> (Vec<u64>, Vec<(u64, u64)>) {
    if n_bits <= 24 {
        let pairs = enumerate_balanced_semiprimes(n_bits);
        let mut primes: Vec<u64> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
        primes.sort_unstable();
        primes.dedup();
        (primes, pairs)
    } else {
        assert!(n_bits <= 60, "n_bits must be ≤ 60 to avoid u64 overflow in p*q");
        let half = n_bits / 2;
        let lo = 1u64 << (half - 1);
        let hi = 1u64 << (half + 1);
        let primes = sample_primes_in_range(lo, hi, target_primes, seed);

        let n_lo = 1u128 << (n_bits - 1);
        let n_hi = 1u128 << n_bits;
        let mut pairs = Vec::new();
        for (i, &p) in primes.iter().enumerate() {
            for &q in &primes[i + 1..] {
                let n = p as u128 * q as u128;
                if n < n_lo || n >= n_hi {
                    continue;
                }
                let (small, big) = if p <= q { (p, q) } else { (q, p) };
                if (small as f64) / (big as f64) < 0.3 {
                    continue;
                }
                pairs.push((small, big));
            }
        }
        // Cap pairs for runtime; shuffle for representative sampling.
        if pairs.len() > max_pairs {
            use rand::seq::SliceRandom;
            let mut rng = StdRng::seed_from_u64(seed ^ 0xCAFE);
            pairs.shuffle(&mut rng);
            pairs.truncate(max_pairs);
        }
        (primes, pairs)
    }
}

// ---------------------------------------------------------------------------
// Core audit function
// ---------------------------------------------------------------------------

/// Run the E21 character audit for a single (n_bits, channel) pair.
///
/// Extracts `n_eigenvectors` dominant eigenvectors from the Eisenstein CRT
/// matrix and tests each against the multiplicative character prediction.
pub fn run_character_audit(
    n_bits: u32,
    ch: &Channel,
    n_eigenvectors: usize,
    seed: u64,
) -> CharacterAuditResult {
    use algebra::{
        best_character, build_dlog_table, pearson_corr, primitive_root, re_char,
    };
    use eigen::top_k_eigenvectors;

    // ------------------------------------------------------------------
    // 1. Enumerate pairs and build prime set
    // ------------------------------------------------------------------
    let pairs = enumerate_balanced_semiprimes(n_bits);
    let n_valid_pairs = pairs.len();

    let mut prime_set: Vec<u64> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
    prime_set.sort_unstable();
    prime_set.dedup();
    let np = prime_set.len();

    // ------------------------------------------------------------------
    // 2. Build the binary CRT matrix M ∈ ℝ^{np × np}
    // ------------------------------------------------------------------
    let mut mat = vec![vec![0.0f64; np]; np];
    for &(p, q) in &pairs {
        let pi = prime_set.partition_point(|&x| x < p);
        let qi = prime_set.partition_point(|&x| x < q);
        let sp  = Semiprime { n: p * q, p, q };
        let val = (eisenstein_hunt::ground_truth(&sp, ch) % 2) as f64;
        mat[pi][qi] = val;
        mat[qi][pi] = val;
    }

    // ------------------------------------------------------------------
    // 3. Extract top-k eigenvectors
    // ------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(seed);
    let n_iters = 120; // more iterations for better convergence
    let eigenpairs = top_k_eigenvectors(&mat, np, n_eigenvectors, n_iters, &mut rng);

    // ------------------------------------------------------------------
    // 4. Prepare character infrastructure for this channel
    // ------------------------------------------------------------------
    let ell    = ch.ell;
    let k1     = (ch.weight - 1) as u64;
    let order  = (ell - 1) as usize;
    let prim_g = primitive_root(ell);
    let dlog   = build_dlog_table(ell, prim_g);

    // g(p) = (1 + p^{k−1}) mod ℓ  for each prime
    let g_vals: Vec<u64> = prime_set
        .iter()
        .map(|&p| (1 + mod_pow(p, k1, ell)) % ell)
        .collect();

    // Discrete logs of g(p); u32::MAX if g(p) = 0 (shouldn't happen in practice
    // but handled for robustness).
    let dlogs_g: Vec<u32> = g_vals
        .iter()
        .map(|&gp| if gp == 0 { u32::MAX } else { dlog[gp as usize] })
        .collect();

    let n_zero_g = dlogs_g.iter().filter(|&&d| d == u32::MAX).count();

    // ------------------------------------------------------------------
    // 5. For each eigenvector: character scan + product tests
    // ------------------------------------------------------------------
    let mut ev_results = Vec::with_capacity(eigenpairs.len());

    for (idx, (eigenvalue, u)) in eigenpairs.iter().enumerate() {
        // --- 5a. Character amplitude scan (Pearson-based, skips r=0) ---
        let (best_r, best_amp, best_corr_re_from_scan, top_chars) =
            best_character(u, &dlogs_g, order, 10);

        // Recompute corr_re and corr_im at best_r with explicit Pearson.
        let (u_valid, dlog_valid): (Vec<f64>, Vec<u32>) = dlogs_g
            .iter()
            .zip(u.iter())
            .filter(|(&d, _)| d != u32::MAX)
            .map(|(&d, &ui)| (ui, d))
            .unzip();

        let (corr_re_best, corr_im_best) = if u_valid.len() >= 2 {
            let re_vals: Vec<f64> = dlog_valid.iter().map(|&d| re_char(d, best_r, order)).collect();
            let im_vals: Vec<f64> = dlog_valid.iter().map(|&d| im_char(d, best_r, order)).collect();
            (pearson_corr(&u_valid, &re_vals), pearson_corr(&u_valid, &im_vals))
        } else {
            (best_corr_re_from_scan, 0.0)
        };

        // --- 5b. Product tests over valid pairs ---
        // actual[i]         = u[pi] * u[qi]
        // pred_sigma[i]     = Re(χ_{r*}(g(p)·g(q) mod ℓ)) = Re(χ_{r*}(σ_{k−1}(N) mod ℓ))
        // pred_nk[i]        = Re(χ_{r*}(N^{k−1} mod ℓ))
        let mut actual:     Vec<f64> = Vec::new();
        let mut pred_sigma: Vec<f64> = Vec::new();
        let mut pred_nk:    Vec<f64> = Vec::new();

        for &(p, q) in &pairs {
            let pi = prime_set.partition_point(|&x| x < p);
            let qi = prime_set.partition_point(|&x| x < q);

            let dlog_gp = dlogs_g[pi];
            let dlog_gq = dlogs_g[qi];
            if dlog_gp == u32::MAX || dlog_gq == u32::MAX {
                continue;
            }

            let prod_u = u[pi] * u[qi];

            // Predicted via σ_{k−1}(N) = g(p)·g(q) mod ℓ
            let sigma_val = g_vals[pi] * g_vals[qi] % ell;
            let dlog_sigma = dlog[sigma_val as usize];
            let re_sigma = if dlog_sigma != u32::MAX {
                re_char(dlog_sigma, best_r, order)
            } else {
                // sigma_val = 0 (very unlikely for prime ℓ; skip)
                continue;
            };

            // Predicted via N^{k−1} mod ℓ
            let n_val   = p * q; // fits u64 for n_bits ≤ 20
            let nk_val  = mod_pow(n_val, k1, ell);
            let dlog_nk = dlog[nk_val as usize];
            let re_nk   = if dlog_nk != u32::MAX {
                re_char(dlog_nk, best_r, order)
            } else {
                continue;
            };

            actual.push(prod_u);
            pred_sigma.push(re_sigma);
            pred_nk.push(re_nk);
        }

        let n_pairs_used = actual.len();
        let product_corr_sigma = if n_pairs_used >= 2 {
            pearson_corr(&actual, &pred_sigma)
        } else {
            0.0
        };
        let product_corr_nk = if n_pairs_used >= 2 {
            pearson_corr(&actual, &pred_nk)
        } else {
            0.0
        };

        ev_results.push(EigenvectorCharResult {
            eigenvector_idx: idx,
            eigenvalue: *eigenvalue,
            best_char_r: best_r,
            best_char_amp: best_amp,
            n_chars_scanned: order / 2,  // r = 1 … order/2
            corr_re_best,
            corr_im_best,
            product_corr_sigma,
            product_corr_nk,
            n_pairs_used,
            top_characters: top_chars,
        });
    }

    CharacterAuditResult {
        n_bits,
        channel_weight: ch.weight,
        channel_ell: ch.ell,
        n_primes: np,
        n_valid_pairs,
        n_zero_g,
        primitive_root: prim_g,
        eigenvectors: ev_results,
    }
}

// ---------------------------------------------------------------------------
// Full-group control experiment
// ---------------------------------------------------------------------------

/// Result of the full-group control: eigenstructure of H[a][b] = parity(ab mod ℓ)
/// over all of (ℤ/ℓℤ)*, without prime restriction.
///
/// Over the full group, the characters χ_r are EXACT eigenvectors with
/// eigenvalue (ℓ−1)·ĥ(r), so the character scan should return amp ≈ 1.
/// If it doesn't, the pipeline is broken; if it does, the pipeline is
/// validated and the prime-restriction drop is real.
#[derive(Debug, Clone, Serialize)]
pub struct FullGroupControlResult {
    pub ell: u64,
    pub order: usize,
    pub primitive_root: u64,
    /// Per-eigenvector results over the full group.
    pub eigenvectors: Vec<FullGroupEigResult>,
}

/// Per-eigenvector result for the full-group control.
#[derive(Debug, Clone, Serialize)]
pub struct FullGroupEigResult {
    pub eigenvector_idx: usize,
    pub eigenvalue: f64,
    pub best_char_r: usize,
    pub best_char_amp: f64,
    pub corr_re: f64,
    pub corr_im: f64,
}

/// Build H[a][b] = parity(ab mod ℓ) for all a,b ∈ {1,…,ℓ−1} and extract
/// eigenvectors, then run the character scan over the full group.
///
/// This validates that the character scan pipeline correctly finds characters
/// when they truly are the eigenvectors (no prime restriction, no g() mapping).
pub fn run_full_group_control(ell: u64, n_eigenvectors: usize, seed: u64) -> FullGroupControlResult {
    use algebra::{best_character, build_dlog_table, pearson_corr, primitive_root};
    use eigen::top_k_eigenvectors;

    let order  = (ell - 1) as usize;
    let prim_g = primitive_root(ell);
    let dlog   = build_dlog_table(ell, prim_g);

    // Build the full (ℓ-1) × (ℓ-1) parity matrix: H[a][b] = (ab mod ℓ) mod 2.
    // Index 0 → group element 1, index i → group element (i+1).
    let n = order;
    let mut mat = vec![vec![0.0f64; n]; n];
    for a in 1..=order {
        for b in a..=order {
            let prod = (a as u64) * (b as u64) % ell;
            let val  = (prod % 2) as f64;
            let ai   = a - 1;
            let bi   = b - 1;
            mat[ai][bi] = val;
            mat[bi][ai] = val;
        }
    }

    // Extract top eigenvectors.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eigenpairs = top_k_eigenvectors(&mat, n, n_eigenvectors, 150, &mut rng);

    // Build dlog table for group elements 1..ℓ-1 (no g() mapping — identity).
    // dlogs[i] = dlog_table[i+1]  (group element = index + 1).
    let dlogs: Vec<u32> = (1..=order as u64).map(|a| dlog[a as usize]).collect();

    let mut ev_results = Vec::with_capacity(eigenpairs.len());
    for (idx, (eigenvalue, u)) in eigenpairs.iter().enumerate() {
        let (best_r, best_amp, best_corr_re, _top) = best_character(u, &dlogs, order, 5);

        // Also compute corr_im for reporting.
        let u_valid: Vec<f64> = u.clone();
        let im_vals: Vec<f64> = dlogs.iter().map(|&d| algebra::im_char(d, best_r, order)).collect();
        let corr_im = pearson_corr(&u_valid, &im_vals);

        ev_results.push(FullGroupEigResult {
            eigenvector_idx: idx,
            eigenvalue: *eigenvalue,
            best_char_r: best_r,
            best_char_amp: best_amp,
            corr_re: best_corr_re,
            corr_im,
        });
    }

    FullGroupControlResult {
        ell,
        order,
        primitive_root: prim_g,
        eigenvectors: ev_results,
    }
}

// ---------------------------------------------------------------------------
// Fourier scaling: centered parity spectrum on the full group
// ---------------------------------------------------------------------------

/// Result of the Fourier scaling analysis for a single prime ℓ.
///
/// Computes the DFT of the centered parity function h̃(a) = 2(a mod 2) − 1
/// on the full cyclic group (ℤ/ℓℤ)*, giving exact Fourier coefficients
/// ĥ̃(r) for r = 0, …, ℓ−2.
#[derive(Debug, Clone, Serialize)]
pub struct FourierScalingResult {
    pub ell: u64,
    pub order: usize,
    /// A_max = max_{r≥1} |ĥ̃(r)|  (excluding r=0 DC component).
    pub a_max: f64,
    /// A_max · √ℓ — should be O(1) if ĥ̃(r) = O(1/√ℓ).
    pub a_max_scaled: f64,
    /// ĥ̃(0) = mean of h̃ over the group.  Should be ≈ 0 for centered parity.
    pub dc_component: f64,
    /// Top-k character amplitudes: [(r, |ĥ̃(r)|)], sorted descending by amplitude.
    pub top_amplitudes: Vec<(usize, f64)>,
}

/// Full Fourier scaling analysis result across multiple primes.
#[derive(Debug, Clone, Serialize)]
pub struct FourierScalingAnalysis {
    /// Per-prime results.
    pub results: Vec<FourierScalingResult>,
    /// Fitted power law: A_max ≈ C · ℓ^slope.
    pub fitted_slope: f64,
    /// Fitted prefactor C in A_max ≈ C · ℓ^slope.
    pub fitted_prefactor: f64,
    /// R² of the log-log fit.
    pub r_squared: f64,
    /// Corrected fit: A_max · √ℓ / √(log ℓ) — should be ≈ constant if
    /// the √(log ℓ) extremal-value correction explains the deviation from -1/2.
    pub corrected_ratio_mean: f64,
    /// Standard deviation of A_max · √ℓ / √(log ℓ).
    pub corrected_ratio_std: f64,
}

/// Compute the full Fourier spectrum of the centered parity function
/// h̃(a) = 2(a mod 2) − 1 on (ℤ/ℓℤ)* for prime ℓ.
///
/// Returns the Fourier coefficients ĥ̃(r) for r = 0, …, ℓ−2:
///   ĥ̃(r) = (1/(ℓ−1)) · Σ_{a=1}^{ℓ−1} h̃(a) · χ̄_r(a)
///
/// Using the discrete log: let f(k) = h̃(g^k), then
///   ĥ̃(r) = (1/n) · Σ_{k=0}^{n−1} f(k) · exp(−2πi·r·k/n)
///
/// Returns Vec of (|ĥ̃(r)|², r) for efficiency; caller takes sqrt if needed.
/// Only computes r = 0 … n/2 (conjugate symmetry for real h̃).
pub fn centered_parity_spectrum(ell: u64, top_k: usize) -> FourierScalingResult {
    use algebra::primitive_root;

    let order = (ell - 1) as usize;
    let prim_g = primitive_root(ell);

    // Build the centered parity sequence in dlog order: f[k] = h̃(g^k mod ℓ)
    // where h̃(a) = 2*(a%2) - 1 ∈ {-1, +1}.
    let mut f = vec![0.0f64; order];
    let mut val = 1u64; // g^0 = 1
    for k in 0..order {
        let a = val;
        f[k] = 2.0 * (a % 2) as f64 - 1.0; // +1 if odd, -1 if even
        val = val * prim_g % ell;
    }
    debug_assert_eq!(val, 1, "primitive root cycle failed");

    // Compute DFT: ĥ̃(r) = (1/n) · Σ_k f(k) · exp(-2πi·r·k/n)
    // For r = 0 … n/2 (conjugate symmetry: |ĥ̃(r)| = |ĥ̃(n-r)| for real f).
    let n = order;
    let n_f = n as f64;
    let half = n / 2;

    let mut amplitudes: Vec<(usize, f64)> = Vec::with_capacity(half + 1);

    // r = 0: DC component = mean of h̃.
    let dc: f64 = f.iter().sum::<f64>() / n_f;

    // r = 1 … n/2
    for r in 1..=half {
        let mut re_sum = 0.0f64;
        let mut im_sum = 0.0f64;
        let phase_step = std::f64::consts::TAU * (r as f64) / n_f;
        for k in 0..n {
            let phase = phase_step * (k as f64);
            re_sum += f[k] * phase.cos();
            im_sum -= f[k] * phase.sin(); // conjugate: exp(-iθ)
        }
        let amp = (re_sum * re_sum + im_sum * im_sum).sqrt() / n_f;
        amplitudes.push((r, amp));
    }

    // Sort by amplitude descending.
    amplitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let a_max = amplitudes.first().map(|x| x.1).unwrap_or(0.0);
    let a_max_scaled = a_max * (ell as f64).sqrt();

    amplitudes.truncate(top_k);

    FourierScalingResult {
        ell,
        order,
        a_max,
        a_max_scaled,
        dc_component: dc,
        top_amplitudes: amplitudes,
    }
}

// ---------------------------------------------------------------------------
// Generic DFT on (ℤ/ℓℤ)*
// ---------------------------------------------------------------------------

/// Compute the DFT of an arbitrary real function f on (ℤ/ℓℤ)* (given in
/// discrete-log order) and return the top-k Fourier amplitudes.
///
/// Input: `f_dlog[k]` = f(g^k mod ℓ) for k = 0, …, ℓ−2.
/// The function is automatically centered (mean subtracted) before DFT.
///
/// Returns `(dc, amplitudes)` where dc = mean(f) and amplitudes are
/// sorted by |ĥ(r)| descending.
fn dft_on_cyclic_group(f_dlog: &[f64], top_k: usize) -> (f64, Vec<(usize, f64)>) {
    let n = f_dlog.len();
    let n_f = n as f64;
    let half = n / 2;

    // DC component = mean.
    let dc: f64 = f_dlog.iter().sum::<f64>() / n_f;

    // Center the function for the DFT.
    let centered: Vec<f64> = f_dlog.iter().map(|&x| x - dc).collect();

    // DFT of centered function: ĥ(r) = (1/n) · Σ_k f̃(k) · exp(-2πi·r·k/n)
    let mut amplitudes: Vec<(usize, f64)> = Vec::with_capacity(half + 1);

    for r in 1..=half {
        let mut re_sum = 0.0f64;
        let mut im_sum = 0.0f64;
        let phase_step = std::f64::consts::TAU * (r as f64) / n_f;
        for k in 0..n {
            let phase = phase_step * (k as f64);
            re_sum += centered[k] * phase.cos();
            im_sum -= centered[k] * phase.sin();
        }
        let amp = (re_sum * re_sum + im_sum * im_sum).sqrt() / n_f;
        amplitudes.push((r, amp));
    }

    amplitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    amplitudes.truncate(top_k);

    (dc, amplitudes)
}

/// Build the dlog-ordered sequence of a function h: {1,…,ℓ−1} → ℝ.
///
/// Returns f[k] = h(g^k mod ℓ) for k = 0, …, ℓ−2, plus the primitive root g.
fn build_dlog_sequence(ell: u64, h: impl Fn(u64) -> f64) -> (Vec<f64>, u64) {
    use algebra::primitive_root;

    let order = (ell - 1) as usize;
    let prim_g = primitive_root(ell);

    let mut f = vec![0.0f64; order];
    let mut val = 1u64;
    for k in 0..order {
        f[k] = h(val);
        val = val * prim_g % ell;
    }
    debug_assert_eq!(val, 1, "primitive root cycle failed");

    (f, prim_g)
}

// ---------------------------------------------------------------------------
// Smoothness indicator DFT
// ---------------------------------------------------------------------------

/// Result of the Fourier scaling analysis for one (ℓ, B) pair.
#[derive(Debug, Clone, Serialize)]
pub struct SmoothnessSpectrumResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    /// Fraction of elements in {1,…,ℓ-1} that are B-smooth.
    pub smooth_fraction: f64,
    /// A_max = max_{r≥1} |ŝ̃_B(r)| for centered smoothness indicator.
    pub a_max: f64,
    /// A_max · √ℓ.
    pub a_max_scaled: f64,
    /// A_max · √ℓ / √(log ℓ).
    pub a_max_corrected: f64,
    /// For comparison: A_max of centered parity on the same group.
    pub parity_a_max: f64,
    /// Ratio: smoothness A_max / parity A_max. >1 means smoothness has MORE bias.
    pub ratio_to_parity: f64,
    /// Expected A_max for a RANDOM subset of the same density p in (ℤ/ℓℤ)*:
    ///   null = √(p(1−p) · 2·ln(n/2) / n)  where n = ℓ−1, p = smooth_fraction.
    /// This is the extreme-value-corrected null for the maximum of ~n/2 i.i.d.
    /// Gaussian random variables with variance p(1−p)/n.
    pub null_a_max: f64,
    /// Excess ratio: a_max / null_a_max.  >1 means genuine multiplicative structure
    /// beyond what a random subset of the same density would produce.
    pub excess_ratio: f64,
    /// Head energy fraction: Σ_{top-k} |ĥ(r)|² / Σ_{all r≠0} |ĥ(r)|².
    /// By Parseval, Σ_{r≠0} |ĥ(r)|² = Var(s̃_B) = p(1−p).
    /// High head energy = spectral concentration in few modes (multiplicative bias).
    pub head_energy_fraction: f64,
    /// Parity head energy fraction for comparison.
    pub parity_head_energy: f64,
    /// Top-k character amplitudes.
    pub top_amplitudes: Vec<(usize, f64)>,
}

/// Full smoothness Fourier scaling analysis across multiple primes.
#[derive(Debug, Clone, Serialize)]
pub struct SmoothnessScalingAnalysis {
    pub smoothness_bound: u64,
    pub results: Vec<SmoothnessSpectrumResult>,
    /// Fitted power law: A_max ≈ C · ℓ^slope.
    pub fitted_slope: f64,
    pub fitted_prefactor: f64,
    pub r_squared: f64,
    /// Corrected ratio: A_max · √ℓ / √(log ℓ).
    pub corrected_ratio_mean: f64,
    pub corrected_ratio_std: f64,
    /// Parity baseline slope for comparison.
    pub parity_slope: f64,
}

/// Check if n is B-smooth (all prime factors ≤ B).
fn is_b_smooth(mut n: u64, bound: u64) -> bool {
    if n <= 1 {
        return true;
    }
    let mut d = 2u64;
    while d * d <= n && d <= bound {
        while n % d == 0 {
            n /= d;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    // If n > 1 after trial division up to min(√n, B), then n has a prime factor.
    // That factor is n itself, and it's > B only if n > bound.
    n <= bound
}

/// Compute the Fourier spectrum of the centered B-smoothness indicator on (ℤ/ℓℤ)*.
///
/// s_B(a) = 1 if a is B-smooth, 0 otherwise.
/// Centered: s̃_B(a) = s_B(a) − mean(s_B).
///
/// Also computes the parity baseline for comparison.
pub fn smoothness_spectrum(ell: u64, bound: u64, top_k: usize) -> SmoothnessSpectrumResult {
    let order = (ell - 1) as usize;

    // Build smoothness indicator in dlog order.
    let (f_smooth, _prim_g) = build_dlog_sequence(ell, |a| {
        if is_b_smooth(a, bound) { 1.0 } else { 0.0 }
    });

    // DFT of centered smoothness indicator (returns top-k by amplitude).
    let (dc, amplitudes) = dft_on_cyclic_group(&f_smooth, top_k);

    let a_max = amplitudes.first().map(|x| x.1).unwrap_or(0.0);
    let ell_f = ell as f64;
    let n_f = order as f64;
    let a_max_scaled = a_max * ell_f.sqrt();
    let a_max_corrected = a_max * ell_f.sqrt() / ell_f.ln().sqrt();

    // Parity baseline: use the existing function (request top_k amplitudes for head energy).
    let parity_result = centered_parity_spectrum(ell, top_k);
    let parity_a_max = parity_result.a_max;

    let ratio = if parity_a_max > 1e-15 { a_max / parity_a_max } else { 0.0 };

    // Density-normalized null: expected A_max for a RANDOM subset of density p.
    // For n i.i.d. centered Bernoulli(p) variables, each DFT coefficient has
    // variance p(1-p)/n. The max over ~n/2 such coefficients (Gaussian approx):
    //   null_a_max ≈ √(p(1-p) · 2·ln(n/2) / n)
    let p = dc; // smooth_fraction = DC component = mean of indicator
    let pq = p * (1.0 - p);
    let null_a_max = if pq > 1e-15 && n_f > 4.0 {
        (pq * 2.0 * (n_f / 2.0).ln() / n_f).sqrt()
    } else {
        0.0
    };
    let excess_ratio = if null_a_max > 1e-15 { a_max / null_a_max } else { 0.0 };

    // Head energy fraction: Σ_{top-k} |ĥ(r)|² / Σ_{all r≠0} |ĥ(r)|².
    // By Parseval on centered function: Σ_{r≠0} |ĥ(r)|² = Var(f̃) = p(1-p).
    let head_energy_sum: f64 = amplitudes.iter().map(|&(_, amp)| amp * amp).sum();
    let head_energy_fraction = if pq > 1e-15 { head_energy_sum / pq } else { 0.0 };

    // Parity head energy: centered parity h̃ ∈ {−1,+1} has Parseval total ≈ 1.0.
    // (Not 0.25: the {0,1} indicator has Var=0.25, but h̃ = 2h−1 has Var=1.0.)
    let parity_head_sum: f64 = parity_result.top_amplitudes.iter().map(|&(_, amp)| amp * amp).sum();
    let parity_head_energy = parity_head_sum; // Parseval total for {−1,+1} = 1.0

    SmoothnessSpectrumResult {
        ell,
        order,
        smoothness_bound: bound,
        smooth_fraction: dc,
        a_max,
        a_max_scaled,
        a_max_corrected,
        parity_a_max,
        ratio_to_parity: ratio,
        null_a_max,
        excess_ratio,
        head_energy_fraction,
        parity_head_energy,
        top_amplitudes: amplitudes,
    }
}

// ---------------------------------------------------------------------------
// Prime-restricted smoothness character diagnostic (E21b Step 2)
// ---------------------------------------------------------------------------

/// Result of testing whether the smoothness-tuned character r* survives
/// restriction to the prime image set {g(p) : p in balanced-semiprime set}.
///
/// Full group: the smoothness indicator s_B has a dominant character r* with
/// excess ratio > 1 (confirmed in E21b Step 1).
/// Prime-restricted: do the same characters still show signal when evaluated
/// only on {g(p)} instead of all of (ℤ/ℓℤ)*?
#[derive(Debug, Clone, Serialize)]
pub struct PrimeRestrictedResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    pub n_bits: u32,
    pub channel_weight: u32,
    // ─── Full-group baseline ───
    /// Dominant character from full-group smoothness DFT.
    pub full_r_star: usize,
    /// Full-group A_max of the smoothness indicator.
    pub full_a_max: f64,
    /// Full-group excess ratio (A_max / null_A_max).
    pub full_excess: f64,
    // ─── Restricted set info ───
    /// Number of distinct primes in the balanced-semiprime set.
    pub n_primes: usize,
    /// Number of primes with g(p) ≠ 0 mod ℓ (valid for character evaluation).
    pub n_valid: usize,
    /// Fraction of g(p) values that are B-smooth (restricted smoothness density).
    pub restricted_smooth_fraction: f64,
    // ─── Fixed-r* test ───
    /// Pearson amplitude of χ_{full_r_star} against s_B(g(·)) on the prime set.
    pub fixed_r_amplitude: f64,
    /// Null amplitude for a single pre-specified character: √(2/(n_valid−2)).
    pub fixed_r_null: f64,
    /// Excess: fixed_r_amplitude / fixed_r_null.
    pub fixed_r_excess: f64,
    // ─── Full scan on restricted set ───
    /// Best character found by scanning all r on the restricted set.
    pub scanned_r_star: usize,
    /// Pearson amplitude of the best character on the restricted set.
    pub scanned_amplitude: f64,
    /// Null for scanned-max: √(2·ln(order/2) / (n_valid−2)).
    pub scanned_null: f64,
    /// Excess: scanned_amplitude / scanned_null.
    pub scanned_excess: f64,
    // ─── Product tests (at scanned_r_star) ───
    /// corr(s_B(g(p))·s_B(g(q)), Re(χ_{r*}(σ_{k−1}(N) mod ℓ)))  [needs p,q]
    pub product_corr_sigma: f64,
    /// corr(s_B(g(p))·s_B(g(q)), Re(χ_{r*}(N^{k−1} mod ℓ)))      [from N only]
    pub product_corr_nk: f64,
    /// corr using σ_approx = 1 + 2·⌊√N⌋^{k−1} + N^{k−1} mod ℓ    [N-only approx]
    /// This exploits p ≈ q ≈ √N for balanced semiprimes.
    pub product_corr_approx: f64,
    /// Mean |σ_approx − σ_true| / ℓ — measures approximation quality.
    pub sigma_approx_error_mean: f64,
    /// Number of pairs used in product tests.
    pub n_pairs: usize,
}

/// Run the prime-restricted smoothness character diagnostic for a single
/// (n_bits, channel, B) combination.
///
/// Steps:
/// 1. Compute full-group smoothness DFT on (ℤ/ℓℤ)* → find r* and A_max.
/// 2. Enumerate primes from balanced semiprimes of n_bits bits.
/// 3. For each prime p, compute g(p) = (1 + p^{k−1}) mod ℓ and s_B(g(p)).
/// 4. Fixed-r* test: Pearson amplitude of χ_{r*} against s_B on restricted set.
/// 5. Full scan: find best character on restricted set.
/// 6. Product tests at the scanned best character.
pub fn run_prime_restricted_smoothness(
    n_bits: u32,
    ch: &Channel,
    bound: u64,
    top_k: usize,
) -> PrimeRestrictedResult {
    use algebra::{
        best_character, build_dlog_table, im_char, pearson_corr, primitive_root, re_char,
    };

    let ell = ch.ell;
    let k1 = (ch.weight - 1) as u64;
    let order = (ell - 1) as usize;

    // 1. Full-group smoothness spectrum.
    let full = smoothness_spectrum(ell, bound, top_k);
    let full_r_star = full.top_amplitudes.first().map(|&(r, _)| r).unwrap_or(1);

    // 2. Generate primes and pairs (enumerate for n≤24, sample for n>24).
    let seed = 0xE21b_5a3d ^ (n_bits as u64 * 0x10000 + ell);
    let (prime_set, pairs) = generate_primes_and_pairs(n_bits, 500, 20_000, seed);
    let n_primes = prime_set.len();

    // 3. Compute g(p) and smoothness for each prime.
    let prim_g = primitive_root(ell);
    let dlog = build_dlog_table(ell, prim_g);

    let g_vals: Vec<u64> = prime_set
        .iter()
        .map(|&p| (1 + mod_pow(p, k1, ell)) % ell)
        .collect();

    let dlogs_g: Vec<u32> = g_vals
        .iter()
        .map(|&gp| if gp == 0 { u32::MAX } else { dlog[gp as usize] })
        .collect();

    // s_B(g(p)) = 1 if g(p) is B-smooth, 0 otherwise.
    let smooth_vals: Vec<f64> = g_vals
        .iter()
        .zip(dlogs_g.iter())
        .map(|(&gp, &d)| {
            if d != u32::MAX && gp > 0 && is_b_smooth(gp, bound) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // Valid primes (g(p) ≠ 0 mod ℓ).
    let valid_mask: Vec<bool> = dlogs_g.iter().map(|&d| d != u32::MAX).collect();
    let n_valid = valid_mask.iter().filter(|&&m| m).count();

    let smooth_count: f64 = (0..n_primes)
        .filter(|&i| valid_mask[i])
        .map(|i| smooth_vals[i])
        .sum();
    let restricted_smooth_fraction = if n_valid > 0 {
        smooth_count / n_valid as f64
    } else {
        0.0
    };

    // Extract valid (smooth_val, dlog) pairs for correlation tests.
    let (valid_smooth, valid_dlogs): (Vec<f64>, Vec<u32>) = smooth_vals
        .iter()
        .zip(dlogs_g.iter())
        .filter(|(_, &d)| d != u32::MAX)
        .map(|(&s, &d)| (s, d))
        .unzip();

    // 4. Fixed-r* test: amplitude of full-group's r* on the restricted set.
    let fixed_r_amplitude = if valid_smooth.len() >= 2 {
        let re_vals: Vec<f64> = valid_dlogs
            .iter()
            .map(|&d| re_char(d, full_r_star, order))
            .collect();
        let im_vals: Vec<f64> = valid_dlogs
            .iter()
            .map(|&d| im_char(d, full_r_star, order))
            .collect();
        let cr = pearson_corr(&valid_smooth, &re_vals);
        let ci = pearson_corr(&valid_smooth, &im_vals);
        (cr * cr + ci * ci).sqrt()
    } else {
        0.0
    };

    // Null for a single pre-specified r: Rayleigh with σ = 1/√(n−2).
    let fixed_r_null = if n_valid > 2 {
        (2.0 / (n_valid - 2) as f64).sqrt()
    } else {
        1.0
    };
    let fixed_r_excess = if fixed_r_null > 1e-15 {
        fixed_r_amplitude / fixed_r_null
    } else {
        0.0
    };

    // 5. Full character scan on the restricted set.
    let (scanned_r_star, scanned_amplitude, _, _) =
        best_character(&valid_smooth, &valid_dlogs, order, top_k);

    // Null for max over ~order/2 characters: √(2·ln(order/2) / (n_valid−2)).
    let scanned_null = if n_valid > 2 && order > 2 {
        (2.0 * ((order as f64 / 2.0).ln()) / (n_valid - 2) as f64).sqrt()
    } else {
        1.0
    };
    let scanned_excess = if scanned_null > 1e-15 {
        scanned_amplitude / scanned_null
    } else {
        0.0
    };

    // 6. Product tests at scanned_r_star.
    let mut actual: Vec<f64> = Vec::new();
    let mut pred_sigma: Vec<f64> = Vec::new();
    let mut pred_nk: Vec<f64> = Vec::new();
    let mut pred_approx: Vec<f64> = Vec::new();
    let mut sigma_errors: Vec<f64> = Vec::new();

    for &(p, q) in &pairs {
        let pi = prime_set.partition_point(|&x| x < p);
        let qi = prime_set.partition_point(|&x| x < q);

        if !valid_mask[pi] || !valid_mask[qi] {
            continue;
        }

        // Product of smoothness values.
        let prod = smooth_vals[pi] * smooth_vals[qi];

        // χ_{r*}(σ_{k−1}(N) mod ℓ) = χ_{r*}(g(p)·g(q) mod ℓ)
        let sigma_val = g_vals[pi] * g_vals[qi] % ell;
        let dlog_sigma = dlog[sigma_val as usize];
        if dlog_sigma == u32::MAX {
            continue;
        }
        let re_sigma = re_char(dlog_sigma, scanned_r_star, order);

        // χ_{r*}(N^{k−1} mod ℓ)
        let n_val = p * q;
        let nk_val = mod_pow(n_val, k1, ell);
        let dlog_nk = dlog[nk_val as usize];
        if dlog_nk == u32::MAX {
            continue;
        }
        let re_nk = re_char(dlog_nk, scanned_r_star, order);

        // σ-approximation: σ_approx = 1 + 2·⌊√N⌋^{k−1} + N^{k−1}  mod ℓ
        // This exploits p ≈ q ≈ √N for balanced semiprimes.
        let sqrt_n = (n_val as f64).sqrt() as u64;
        let sqrt_k1 = mod_pow(sqrt_n, k1, ell);
        let sigma_approx = (1 + 2 * sqrt_k1 + nk_val) % ell;

        // Measure approximation error: |σ_approx − σ_true| / ℓ
        let err = if sigma_approx >= sigma_val {
            sigma_approx - sigma_val
        } else {
            sigma_val - sigma_approx
        };
        let err_wrapped = err.min(ell - err); // circular distance
        sigma_errors.push(err_wrapped as f64 / ell as f64);

        let dlog_approx = dlog[sigma_approx as usize];
        let re_approx = if dlog_approx != u32::MAX {
            re_char(dlog_approx, scanned_r_star, order)
        } else {
            // σ_approx = 0 mod ℓ — skip this pair for approx but keep others
            0.0
        };

        actual.push(prod);
        pred_sigma.push(re_sigma);
        pred_nk.push(re_nk);
        pred_approx.push(re_approx);
    }

    let n_pairs = actual.len();
    let product_corr_sigma = if n_pairs >= 2 {
        pearson_corr(&actual, &pred_sigma)
    } else {
        0.0
    };
    let product_corr_nk = if n_pairs >= 2 {
        pearson_corr(&actual, &pred_nk)
    } else {
        0.0
    };
    let product_corr_approx = if n_pairs >= 2 {
        pearson_corr(&actual, &pred_approx)
    } else {
        0.0
    };
    let sigma_approx_error_mean = if !sigma_errors.is_empty() {
        sigma_errors.iter().sum::<f64>() / sigma_errors.len() as f64
    } else {
        0.0
    };

    PrimeRestrictedResult {
        ell,
        order,
        smoothness_bound: bound,
        n_bits,
        channel_weight: ch.weight,
        full_r_star,
        full_a_max: full.a_max,
        full_excess: full.excess_ratio,
        n_primes,
        n_valid,
        restricted_smooth_fraction,
        fixed_r_amplitude,
        fixed_r_null,
        fixed_r_excess,
        scanned_r_star,
        scanned_amplitude,
        scanned_null,
        scanned_excess,
        product_corr_sigma,
        product_corr_nk,
        product_corr_approx,
        sigma_approx_error_mean,
        n_pairs,
    }
}

/// Run smoothness Fourier scaling across many primes for a fixed B.
pub fn run_smoothness_scaling(
    primes: &[u64],
    bound: u64,
    top_k: usize,
) -> SmoothnessScalingAnalysis {
    let results: Vec<SmoothnessSpectrumResult> = primes
        .iter()
        .map(|&ell| smoothness_spectrum(ell, bound, top_k))
        .collect();

    // Filter out primes where A_max ≈ 0 (all elements B-smooth → constant function).
    // log(0) = -inf would corrupt the fit.
    let valid: Vec<&SmoothnessSpectrumResult> = results
        .iter()
        .filter(|r| r.a_max > 1e-15)
        .collect();

    // Log-log fit for smoothness (on valid points only).
    let (slope, intercept, r_sq) = if valid.len() >= 2 {
        let xs: Vec<f64> = valid.iter().map(|r| (r.ell as f64).ln()).collect();
        let ys: Vec<f64> = valid.iter().map(|r| r.a_max.ln()).collect();
        log_log_fit(&xs, &ys)
    } else {
        (-0.5, 0.0, 0.0)
    };

    // Corrected ratio (only for valid points).
    let ratios: Vec<f64> = valid
        .iter()
        .map(|r| {
            let ell_f = r.ell as f64;
            r.a_max * ell_f.sqrt() / ell_f.ln().sqrt()
        })
        .collect();
    let n = ratios.len() as f64;
    let mean = if n > 0.0 { ratios.iter().sum::<f64>() / n } else { 0.0 };
    let std_dev = if n > 1.0 {
        (ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };

    // Parity baseline slope (computed on valid points to match ℓ range).
    let parity_slope = if valid.len() >= 2 {
        let xs: Vec<f64> = valid.iter().map(|r| (r.ell as f64).ln()).collect();
        let ys: Vec<f64> = valid.iter().map(|r| r.parity_a_max.ln()).collect();
        log_log_fit(&xs, &ys).0
    } else {
        -0.5
    };

    SmoothnessScalingAnalysis {
        smoothness_bound: bound,
        results,
        fitted_slope: slope,
        fitted_prefactor: intercept.exp(),
        r_squared: r_sq,
        corrected_ratio_mean: mean,
        corrected_ratio_std: std_dev,
        parity_slope,
    }
}

/// Run the Fourier scaling analysis across many primes ℓ.
///
/// For each prime, computes the exact DFT of centered parity h̃ on (ℤ/ℓℤ)*
/// and records A_max(ℓ).  Then fits log(A_max) = slope·log(ℓ) + log(C)
/// via least-squares regression.
pub fn run_fourier_scaling(primes: &[u64], top_k: usize) -> FourierScalingAnalysis {
    let results: Vec<FourierScalingResult> = primes
        .iter()
        .map(|&ell| centered_parity_spectrum(ell, top_k))
        .collect();

    build_fourier_scaling_analysis(results)
}

/// Build a `FourierScalingAnalysis` from pre-computed per-prime results.
///
/// Fits log(A_max) = slope·log(ℓ) + log(C) via least-squares, and also
/// computes the √(log ℓ)-corrected ratio A_max·√ℓ/√(log ℓ) to check
/// whether the deviation from slope = -1/2 is explained by extremal
/// value statistics.
pub fn build_fourier_scaling_analysis(results: Vec<FourierScalingResult>) -> FourierScalingAnalysis {
    let (slope, intercept, r_sq) = if results.len() >= 2 {
        let xs: Vec<f64> = results.iter().map(|r| (r.ell as f64).ln()).collect();
        let ys: Vec<f64> = results.iter().map(|r| r.a_max.ln()).collect();
        log_log_fit(&xs, &ys)
    } else {
        (-0.5, 0.0, 0.0)
    };

    // Compute corrected ratio: A_max · √ℓ / √(log ℓ).
    // If A_max = C · √(log ℓ) / √ℓ, this should be roughly constant.
    let ratios: Vec<f64> = results
        .iter()
        .map(|r| {
            let ell_f = r.ell as f64;
            r.a_max * ell_f.sqrt() / ell_f.ln().sqrt()
        })
        .collect();
    let n = ratios.len() as f64;
    let mean = if n > 0.0 { ratios.iter().sum::<f64>() / n } else { 0.0 };
    let std_dev = if n > 1.0 {
        (ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };

    FourierScalingAnalysis {
        results,
        fitted_slope: slope,
        fitted_prefactor: intercept.exp(),
        r_squared: r_sq,
        corrected_ratio_mean: mean,
        corrected_ratio_std: std_dev,
    }
}

/// Simple least-squares fit: y = slope·x + intercept.
/// Returns (slope, intercept, R²).
fn log_log_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f64 = x.iter().map(|a| a * a).sum();
    let _syy: f64 = y.iter().map(|b| b * b).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return (0.0, sy / n, 0.0);
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    // R²
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let pred = slope * xi + intercept;
            (yi - pred).powi(2)
        })
        .sum();
    let mean_y = sy / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let r_sq = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 1.0 };

    (slope, intercept, r_sq)
}

/// Generate a list of primes for the scaling sweep.
///
/// Picks primes near logarithmically-spaced targets from `lo` to `hi`.
pub fn scaling_primes(lo: u64, hi: u64, n_target: usize) -> Vec<u64> {
    let log_lo = (lo as f64).ln();
    let log_hi = (hi as f64).ln();
    let step = (log_hi - log_lo) / (n_target.max(2) - 1) as f64;

    let mut primes = Vec::with_capacity(n_target);
    for i in 0..n_target {
        let target = (log_lo + step * i as f64).exp() as u64;
        // Find the smallest prime >= target.
        let p = next_prime(target.max(3));
        if primes.last() != Some(&p) {
            primes.push(p);
        }
    }
    primes
}

/// Find the smallest prime ≥ n.
fn next_prime(n: u64) -> u64 {
    let mut candidate = if n % 2 == 0 { n + 1 } else { n };
    while !is_prime_u64(candidate) {
        candidate += 2;
    }
    candidate
}

// ---------------------------------------------------------------------------
// E21b Step 3: Stress tests
// ---------------------------------------------------------------------------

/// Full (untruncated) DFT of the smoothness indicator on (ℤ/ℓℤ)*.
/// Returns all coefficients for r = 1..order/2 (not just top-k).
/// Needed for the DFT-weighted multi-character N-score test.
#[derive(Debug, Clone)]
pub struct FullSmoothnessSpectrum {
    pub ell: u64,
    pub order: usize,
    pub smooth_fraction: f64,
    /// (r, re_coeff, im_coeff, amplitude) for r = 1..order/2.
    pub coefficients: Vec<(usize, f64, f64, f64)>,
}

/// Compute the full (untruncated) DFT of the B-smoothness indicator on (ℤ/ℓℤ)*.
pub fn smoothness_spectrum_full(ell: u64, bound: u64) -> FullSmoothnessSpectrum {
    let order = (ell - 1) as usize;
    let (f_smooth, _prim_g) = build_dlog_sequence(ell, |a| {
        if is_b_smooth(a, bound) {
            1.0
        } else {
            0.0
        }
    });

    let n = order;
    let n_f = n as f64;
    let half = n / 2;

    let dc: f64 = f_smooth.iter().sum::<f64>() / n_f;
    let centered: Vec<f64> = f_smooth.iter().map(|&x| x - dc).collect();

    let mut coefficients = Vec::with_capacity(half);
    for r in 1..=half {
        let mut re_sum = 0.0f64;
        let mut im_sum = 0.0f64;
        let phase_step = std::f64::consts::TAU * (r as f64) / n_f;
        for k in 0..n {
            let phase = phase_step * (k as f64);
            re_sum += centered[k] * phase.cos();
            im_sum -= centered[k] * phase.sin();
        }
        let re_coeff = re_sum / n_f;
        let im_coeff = im_sum / n_f;
        let amp = (re_coeff * re_coeff + im_coeff * im_coeff).sqrt();
        coefficients.push((r, re_coeff, im_coeff, amp));
    }

    FullSmoothnessSpectrum {
        ell,
        order,
        smooth_fraction: dc,
        coefficients,
    }
}

/// Intermediate data for a single (n_bits, channel, B) block.
/// Shared across all stress tests.
struct BlockData {
    ell: u64,
    k1: u64,
    order: usize,
    #[allow(dead_code)]
    bound: u64,
    #[allow(dead_code)]
    prim_g: u64,
    dlog: Vec<u32>,
    prime_set: Vec<u64>,
    pairs: Vec<(u64, u64)>,
    #[allow(dead_code)]
    g_vals: Vec<u64>,
    #[allow(dead_code)]
    dlogs_g: Vec<u32>,
    smooth_vals: Vec<f64>,
    valid_mask: Vec<bool>,
    n_valid: usize,
    valid_smooth: Vec<f64>,
    valid_dlogs: Vec<u32>,
    full_r_star: usize,
    scanned_r_star: usize,
    #[allow(dead_code)]
    fixed_r_amplitude: f64,
    fixed_r_null: f64,
    fixed_r_excess: f64,
    channel_weight: u32,
}

/// Prepare block data for stress tests (shared setup extracted from
/// `run_prime_restricted_smoothness`).
fn prepare_block(n_bits: u32, ch: &Channel, bound: u64, top_k: usize) -> BlockData {
    use algebra::{best_character, build_dlog_table, im_char, pearson_corr, primitive_root, re_char};

    let ell = ch.ell;
    let k1 = (ch.weight - 1) as u64;
    let order = (ell - 1) as usize;

    // 1. Full-group smoothness spectrum → find r*.
    let full = smoothness_spectrum(ell, bound, top_k);
    let full_r_star = full.top_amplitudes.first().map(|&(r, _)| r).unwrap_or(1);

    // 2. Generate primes and pairs.
    let seed = 0xE21b_5a3d ^ (n_bits as u64 * 0x10000 + ell);
    let (prime_set, pairs) = generate_primes_and_pairs(n_bits, 500, 20_000, seed);

    // 3. Compute g(p) and smoothness for each prime.
    let prim_g = primitive_root(ell);
    let dlog = build_dlog_table(ell, prim_g);

    let g_vals: Vec<u64> = prime_set
        .iter()
        .map(|&p| (1 + mod_pow(p, k1, ell)) % ell)
        .collect();

    let dlogs_g: Vec<u32> = g_vals
        .iter()
        .map(|&gp| if gp == 0 { u32::MAX } else { dlog[gp as usize] })
        .collect();

    let smooth_vals: Vec<f64> = g_vals
        .iter()
        .zip(dlogs_g.iter())
        .map(|(&gp, &d)| {
            if d != u32::MAX && gp > 0 && is_b_smooth(gp, bound) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let valid_mask: Vec<bool> = dlogs_g.iter().map(|&d| d != u32::MAX).collect();
    let n_valid = valid_mask.iter().filter(|&&m| m).count();

    let (valid_smooth, valid_dlogs): (Vec<f64>, Vec<u32>) = smooth_vals
        .iter()
        .zip(dlogs_g.iter())
        .filter(|(_, &d)| d != u32::MAX)
        .map(|(&s, &d)| (s, d))
        .unzip();

    // 4. Fixed-r* test.
    let fixed_r_amplitude = if valid_smooth.len() >= 2 {
        let re_vals: Vec<f64> = valid_dlogs
            .iter()
            .map(|&d| re_char(d, full_r_star, order))
            .collect();
        let im_vals: Vec<f64> = valid_dlogs
            .iter()
            .map(|&d| im_char(d, full_r_star, order))
            .collect();
        let cr = pearson_corr(&valid_smooth, &re_vals);
        let ci = pearson_corr(&valid_smooth, &im_vals);
        (cr * cr + ci * ci).sqrt()
    } else {
        0.0
    };

    let fixed_r_null = if n_valid > 2 {
        (2.0 / (n_valid - 2) as f64).sqrt()
    } else {
        1.0
    };
    let fixed_r_excess = if fixed_r_null > 1e-15 {
        fixed_r_amplitude / fixed_r_null
    } else {
        0.0
    };

    // 5. Full character scan on the restricted set.
    let (scanned_r_star, _scanned_amplitude, _, _) =
        best_character(&valid_smooth, &valid_dlogs, order, top_k);

    BlockData {
        ell,
        k1,
        order,
        bound,
        prim_g,
        dlog,
        prime_set,
        pairs,
        g_vals,
        dlogs_g,
        smooth_vals,
        valid_mask,
        n_valid,
        valid_smooth,
        valid_dlogs,
        full_r_star,
        scanned_r_star,
        fixed_r_amplitude,
        fixed_r_null,
        fixed_r_excess,
        channel_weight: ch.weight,
    }
}

// ─── Stress test result types ───

/// Test 1: Permutation null controls for the product correlation.
#[derive(Debug, Clone, Serialize)]
pub struct PermutationNullResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    pub n_bits: u32,
    pub channel_weight: u32,
    pub observed_corr_nk: f64,
    pub n_permutations: usize,
    pub null_mean: f64,
    pub null_std: f64,
    pub z_score: f64,
    pub empirical_p_value: f64,
    pub n_pairs: usize,
    pub scanned_r_star: usize,
}

/// Test 2: One entry in the cross-n transfer test.
#[derive(Debug, Clone, Serialize)]
pub struct CrossNTransferEntry {
    pub n_bits: u32,
    pub local_r_star: usize,
    pub source_r_star: usize,
    pub local_fix_excess: f64,
    pub transfer_fix_excess: f64,
    pub consistency_ratio: f64,
    pub n_valid: usize,
}

/// Test 2: Cross-n transfer of r* for one (channel, B) combination.
#[derive(Debug, Clone, Serialize)]
pub struct CrossNTransferResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    pub channel_weight: u32,
    pub source_n_bits: u32,
    pub source_r_star: usize,
    pub entries: Vec<CrossNTransferEntry>,
}

/// Test 3: Multi-character N-score (can ANY linear combination extract from N?).
#[derive(Debug, Clone, Serialize)]
pub struct MultiCharacterScoreResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    pub n_bits: u32,
    pub channel_weight: u32,
    /// 3a: DFT-weighted score using all r (algebraically ≡ 3c).
    pub corr_dft_weighted_all: f64,
    /// 3b: DFT-weighted score using top-10 r only.
    pub corr_dft_weighted_top10: f64,
    /// 3c: Direct smoothness: corr(product, s_B(N^{k-1} mod ℓ)).
    pub corr_direct_smoothness: f64,
    /// 3d: Held-out R² from train/test split with learned weights.
    pub train_test_r_squared: f64,
    pub n_chars_all: usize,
    pub n_pairs: usize,
    pub n_train: usize,
    pub n_test: usize,
}

/// Test 4: Bootstrap confidence intervals.
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapCIResult {
    pub ell: u64,
    pub order: usize,
    pub smoothness_bound: u64,
    pub n_bits: u32,
    pub channel_weight: u32,
    pub fix_excess_mean: f64,
    pub fix_excess_std: f64,
    pub fix_excess_ci_lo: f64,
    pub fix_excess_ci_hi: f64,
    pub corr_nk_mean: f64,
    pub corr_nk_std: f64,
    pub corr_nk_ci_lo: f64,
    pub corr_nk_ci_hi: f64,
    pub n_bootstrap: usize,
    pub n_pairs: usize,
}

/// Top-level stress test result.
#[derive(Debug, Clone, Serialize)]
pub struct StressTestResult {
    pub permutation_null: Vec<PermutationNullResult>,
    pub cross_n_transfer: Vec<CrossNTransferResult>,
    pub multi_character_score: Vec<MultiCharacterScoreResult>,
    pub bootstrap_ci: Vec<BootstrapCIResult>,
}

// ─── Stress test implementations ───

/// Test 1: Permutation null for the product correlation.
///
/// Randomly re-pairs primes to generate a null distribution for corr_Nk,
/// verifying that the observed correlation is consistent with random pairing.
pub fn run_permutation_null(
    n_bits: u32,
    ch: &Channel,
    bound: u64,
    top_k: usize,
    n_permutations: usize,
    seed: u64,
) -> PermutationNullResult {
    use algebra::{pearson_corr, re_char};
    use rand::seq::SliceRandom;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let blk = prepare_block(n_bits, ch, bound, top_k);

    // Build original product test vectors.
    let mut actual = Vec::new();
    let mut pred_nk = Vec::new();

    for &(p, q) in &blk.pairs {
        let pi = blk.prime_set.partition_point(|&x| x < p);
        let qi = blk.prime_set.partition_point(|&x| x < q);
        if !blk.valid_mask[pi] || !blk.valid_mask[qi] {
            continue;
        }
        let prod = blk.smooth_vals[pi] * blk.smooth_vals[qi];
        let n_val = p * q;
        let nk_val = mod_pow(n_val, blk.k1, blk.ell);
        let dlog_nk = blk.dlog[nk_val as usize];
        if dlog_nk == u32::MAX {
            continue;
        }
        actual.push(prod);
        pred_nk.push(re_char(dlog_nk, blk.scanned_r_star, blk.order));
    }

    let n_pairs = actual.len();
    let observed_corr_nk = if n_pairs >= 2 {
        pearson_corr(&actual, &pred_nk)
    } else {
        0.0
    };

    // Collect p and q indices from valid pairs for permutation.
    let mut p_indices = Vec::new();
    let mut q_indices = Vec::new();
    for &(p, q) in &blk.pairs {
        let pi = blk.prime_set.partition_point(|&x| x < p);
        let qi = blk.prime_set.partition_point(|&x| x < q);
        if !blk.valid_mask[pi] || !blk.valid_mask[qi] {
            continue;
        }
        let n_val = p * q;
        let nk_val = mod_pow(n_val, blk.k1, blk.ell);
        if blk.dlog[nk_val as usize] == u32::MAX {
            continue;
        }
        p_indices.push(pi);
        q_indices.push(qi);
    }

    // Run permutations.
    let mut rng = StdRng::seed_from_u64(seed ^ 0xAE12_0011);
    let mut perm_corrs = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        let mut q_perm = q_indices.clone();
        q_perm.shuffle(&mut rng);

        let mut actual_perm = Vec::with_capacity(n_pairs);
        let mut pred_perm = Vec::with_capacity(n_pairs);

        for (i, &pi) in p_indices.iter().enumerate() {
            let qi = q_perm[i];
            let prod = blk.smooth_vals[pi] * blk.smooth_vals[qi];
            // Compute N_perm = p * q_shuffled
            let p_val = blk.prime_set[pi];
            let q_val = blk.prime_set[qi];
            let n_val = p_val * q_val;
            let nk_val = mod_pow(n_val, blk.k1, blk.ell);
            let dlog_nk = blk.dlog[nk_val as usize];
            if dlog_nk == u32::MAX {
                continue;
            }
            actual_perm.push(prod);
            pred_perm.push(re_char(dlog_nk, blk.scanned_r_star, blk.order));
        }

        let corr = if actual_perm.len() >= 2 {
            pearson_corr(&actual_perm, &pred_perm)
        } else {
            0.0
        };
        perm_corrs.push(corr);
    }

    // Aggregate.
    let n_p = perm_corrs.len() as f64;
    let null_mean = perm_corrs.iter().sum::<f64>() / n_p;
    let null_std = if n_p > 1.0 {
        (perm_corrs
            .iter()
            .map(|&c| (c - null_mean).powi(2))
            .sum::<f64>()
            / (n_p - 1.0))
            .sqrt()
    } else {
        0.0
    };
    let z_score = if null_std > 1e-15 {
        (observed_corr_nk - null_mean) / null_std
    } else {
        0.0
    };
    let empirical_p_value = perm_corrs
        .iter()
        .filter(|&&c| c.abs() >= observed_corr_nk.abs())
        .count() as f64
        / n_p;

    PermutationNullResult {
        ell: blk.ell,
        order: blk.order,
        smoothness_bound: bound,
        n_bits,
        channel_weight: blk.channel_weight,
        observed_corr_nk,
        n_permutations,
        null_mean,
        null_std,
        z_score,
        empirical_p_value,
        n_pairs,
        scanned_r_star: blk.scanned_r_star,
    }
}

/// Test 2: Cross-n transfer of r*.
///
/// Tests whether the scanned r* from one bit size (source) retains its
/// signal at other bit sizes, confirming the character is stable.
pub fn run_cross_n_transfer(
    ch: &Channel,
    bound: u64,
    top_k: usize,
    bit_sizes: &[u32],
    source_n_bits: u32,
) -> CrossNTransferResult {
    use algebra::{im_char, pearson_corr, re_char};

    // Get source r* from the source bit size's restricted scan.
    let source_blk = prepare_block(source_n_bits, ch, bound, top_k);
    let source_r_star = source_blk.scanned_r_star;

    let mut entries = Vec::with_capacity(bit_sizes.len());

    for &n_bits in bit_sizes {
        let blk = prepare_block(n_bits, ch, bound, top_k);

        // Compute transfer amplitude: Pearson amplitude of χ_{source_r_star}
        // against s_B(g(p)) on the TARGET prime set.
        let transfer_fix_amplitude = if blk.valid_smooth.len() >= 2 {
            let re_vals: Vec<f64> = blk
                .valid_dlogs
                .iter()
                .map(|&d| re_char(d, source_r_star, blk.order))
                .collect();
            let im_vals: Vec<f64> = blk
                .valid_dlogs
                .iter()
                .map(|&d| im_char(d, source_r_star, blk.order))
                .collect();
            let cr = pearson_corr(&blk.valid_smooth, &re_vals);
            let ci = pearson_corr(&blk.valid_smooth, &im_vals);
            (cr * cr + ci * ci).sqrt()
        } else {
            0.0
        };

        let transfer_fix_excess = if blk.fixed_r_null > 1e-15 {
            transfer_fix_amplitude / blk.fixed_r_null
        } else {
            0.0
        };

        let local_fix_excess = blk.fixed_r_excess;
        let consistency_ratio = if local_fix_excess > 1e-15 {
            transfer_fix_excess / local_fix_excess
        } else {
            0.0
        };

        entries.push(CrossNTransferEntry {
            n_bits,
            local_r_star: blk.scanned_r_star,
            source_r_star,
            local_fix_excess,
            transfer_fix_excess,
            consistency_ratio,
            n_valid: blk.n_valid,
        });
    }

    CrossNTransferResult {
        ell: ch.ell,
        order: (ch.ell - 1) as usize,
        smoothness_bound: bound,
        channel_weight: ch.weight,
        source_n_bits,
        source_r_star,
        entries,
    }
}

/// Test 3: Multi-character N-score.
///
/// Tests whether ANY linear combination of characters can extract the
/// smoothness product from N alone. This is the key test.
pub fn run_multi_character_score(
    n_bits: u32,
    ch: &Channel,
    bound: u64,
    top_k: usize,
    _seed: u64,
) -> MultiCharacterScoreResult {
    use algebra::{im_char, pearson_corr, re_char};

    let blk = prepare_block(n_bits, ch, bound, top_k);
    let full_dft = smoothness_spectrum_full(blk.ell, bound);
    let n_chars_all = full_dft.coefficients.len();

    // Build pair-level data.
    let mut actual = Vec::new();
    let mut nk_dlogs: Vec<u32> = Vec::new();
    let mut nk_vals_raw: Vec<u64> = Vec::new();

    for &(p, q) in &blk.pairs {
        let pi = blk.prime_set.partition_point(|&x| x < p);
        let qi = blk.prime_set.partition_point(|&x| x < q);
        if !blk.valid_mask[pi] || !blk.valid_mask[qi] {
            continue;
        }
        let n_val = p * q;
        let nk_val = mod_pow(n_val, blk.k1, blk.ell);
        let dlog_nk = blk.dlog[nk_val as usize];
        if dlog_nk == u32::MAX {
            continue;
        }
        actual.push(blk.smooth_vals[pi] * blk.smooth_vals[qi]);
        nk_dlogs.push(dlog_nk);
        nk_vals_raw.push(nk_val);
    }

    let n_pairs = actual.len();

    // 3a: DFT-weighted score using all r.
    // Mathematical identity: Σ_r ŝ_B(r)·χ_r(x) = s̃_B(x).
    // So score_all[i] = s_B(N^{k-1} mod ℓ) - smooth_fraction.
    let score_all: Vec<f64> = nk_vals_raw
        .iter()
        .map(|&nk| is_b_smooth(nk, bound) as u64 as f64 - full_dft.smooth_fraction)
        .collect();
    let corr_dft_weighted_all = if n_pairs >= 2 {
        pearson_corr(&actual, &score_all)
    } else {
        0.0
    };

    // 3b: DFT-weighted score using top-10 r only.
    let mut sorted_coeffs = full_dft.coefficients.clone();
    sorted_coeffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let top10: Vec<_> = sorted_coeffs.iter().take(10).cloned().collect();

    let score_top10: Vec<f64> = nk_dlogs
        .iter()
        .map(|&d| {
            top10
                .iter()
                .map(|&(r, re_c, im_c, _)| {
                    re_c * re_char(d, r, blk.order) + im_c * im_char(d, r, blk.order)
                })
                .sum::<f64>()
        })
        .collect();
    let corr_dft_weighted_top10 = if n_pairs >= 2 {
        pearson_corr(&actual, &score_top10)
    } else {
        0.0
    };

    // 3c: Direct smoothness: s_B(N^{k-1} mod ℓ) as 0/1.
    let pred_direct: Vec<f64> = nk_vals_raw
        .iter()
        .map(|&nk| is_b_smooth(nk, bound) as u64 as f64)
        .collect();
    let corr_direct_smoothness = if n_pairs >= 2 {
        pearson_corr(&actual, &pred_direct)
    } else {
        0.0
    };

    // 3d: Train/test split with learned character weights.
    let (n_train, n_test, train_test_r_squared) = if n_pairs >= 20 {
        let n_train = n_pairs / 2;
        let n_test = n_pairs - n_train;

        let actual_train = &actual[..n_train];
        let actual_test = &actual[n_train..];
        let nk_dlogs_train = &nk_dlogs[..n_train];
        let nk_dlogs_test = &nk_dlogs[n_train..];

        // Learn weights: w_r = corr(actual_train, Re(χ_r(nk))) for top characters.
        let max_chars = (blk.order / 2).min(500);
        let mut weights: Vec<(usize, f64)> = Vec::with_capacity(max_chars);

        for r in 1..=max_chars {
            let re_vals: Vec<f64> = nk_dlogs_train
                .iter()
                .map(|&d| re_char(d, r, blk.order))
                .collect();
            let w = pearson_corr(actual_train, &re_vals);
            if w.is_finite() {
                weights.push((r, w));
            }
        }

        // Evaluate on test set.
        let score_test: Vec<f64> = nk_dlogs_test
            .iter()
            .map(|&d| {
                weights
                    .iter()
                    .map(|&(r, w)| w * re_char(d, r, blk.order))
                    .sum::<f64>()
            })
            .collect();

        // R² = 1 - SS_res / SS_tot
        let mean_test = actual_test.iter().sum::<f64>() / n_test as f64;
        let ss_tot: f64 = actual_test.iter().map(|&y| (y - mean_test).powi(2)).sum();
        let ss_res: f64 = actual_test
            .iter()
            .zip(score_test.iter())
            .map(|(&y, &yhat)| (y - yhat).powi(2))
            .sum();
        let r_sq = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        (n_train, n_test, r_sq)
    } else {
        (0, 0, 0.0)
    };

    MultiCharacterScoreResult {
        ell: blk.ell,
        order: blk.order,
        smoothness_bound: bound,
        n_bits,
        channel_weight: blk.channel_weight,
        corr_dft_weighted_all,
        corr_dft_weighted_top10,
        corr_direct_smoothness,
        train_test_r_squared,
        n_chars_all,
        n_pairs,
        n_train,
        n_test,
    }
}

/// Test 4: Bootstrap confidence intervals for fix_excess and corr_Nk.
pub fn run_bootstrap_ci(
    n_bits: u32,
    ch: &Channel,
    bound: u64,
    top_k: usize,
    n_bootstrap: usize,
    seed: u64,
) -> BootstrapCIResult {
    use algebra::{im_char, pearson_corr, re_char};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let blk = prepare_block(n_bits, ch, bound, top_k);

    // Build pair-level data for corr_Nk bootstrap.
    let mut pair_actual = Vec::new();
    let mut pair_pred_nk = Vec::new();

    for &(p, q) in &blk.pairs {
        let pi = blk.prime_set.partition_point(|&x| x < p);
        let qi = blk.prime_set.partition_point(|&x| x < q);
        if !blk.valid_mask[pi] || !blk.valid_mask[qi] {
            continue;
        }
        let n_val = p * q;
        let nk_val = mod_pow(n_val, blk.k1, blk.ell);
        let dlog_nk = blk.dlog[nk_val as usize];
        if dlog_nk == u32::MAX {
            continue;
        }
        pair_actual.push(blk.smooth_vals[pi] * blk.smooth_vals[qi]);
        pair_pred_nk.push(re_char(dlog_nk, blk.scanned_r_star, blk.order));
    }

    let n_pairs = pair_actual.len();
    let n_valid = blk.n_valid;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xB007_C1);

    let mut fix_samples = Vec::with_capacity(n_bootstrap);
    let mut corr_samples = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Bootstrap fix_excess: resample primes with replacement.
        if n_valid >= 2 {
            let mut re_boot = Vec::with_capacity(n_valid);
            let mut im_boot = Vec::with_capacity(n_valid);
            let mut smooth_boot = Vec::with_capacity(n_valid);
            for _ in 0..n_valid {
                let idx = rng.gen_range(0..n_valid);
                smooth_boot.push(blk.valid_smooth[idx]);
                re_boot.push(re_char(blk.valid_dlogs[idx], blk.full_r_star, blk.order));
                im_boot.push(im_char(blk.valid_dlogs[idx], blk.full_r_star, blk.order));
            }
            let cr = pearson_corr(&smooth_boot, &re_boot);
            let ci = pearson_corr(&smooth_boot, &im_boot);
            let amp = (cr * cr + ci * ci).sqrt();
            let null = (2.0 / (n_valid - 2) as f64).sqrt();
            fix_samples.push(if null > 1e-15 { amp / null } else { 0.0 });
        } else {
            fix_samples.push(0.0);
        }

        // Bootstrap corr_Nk: resample pairs with replacement.
        if n_pairs >= 2 {
            let mut a_boot = Vec::with_capacity(n_pairs);
            let mut p_boot = Vec::with_capacity(n_pairs);
            for _ in 0..n_pairs {
                let idx = rng.gen_range(0..n_pairs);
                a_boot.push(pair_actual[idx]);
                p_boot.push(pair_pred_nk[idx]);
            }
            corr_samples.push(pearson_corr(&a_boot, &p_boot));
        } else {
            corr_samples.push(0.0);
        }
    }

    // Aggregate.
    let fix_mean = fix_samples.iter().sum::<f64>() / n_bootstrap as f64;
    let fix_std = (fix_samples
        .iter()
        .map(|&x| (x - fix_mean).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1).max(1) as f64)
        .sqrt();
    fix_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fix_ci_lo = fix_samples[(0.025 * n_bootstrap as f64) as usize];
    let fix_ci_hi = fix_samples[(0.975 * n_bootstrap as f64).min(n_bootstrap as f64 - 1.0) as usize];

    let corr_mean = corr_samples.iter().sum::<f64>() / n_bootstrap as f64;
    let corr_std = (corr_samples
        .iter()
        .map(|&x| (x - corr_mean).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1).max(1) as f64)
        .sqrt();
    corr_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let corr_ci_lo = corr_samples[(0.025 * n_bootstrap as f64) as usize];
    let corr_ci_hi = corr_samples[(0.975 * n_bootstrap as f64).min(n_bootstrap as f64 - 1.0) as usize];

    BootstrapCIResult {
        ell: blk.ell,
        order: blk.order,
        smoothness_bound: bound,
        n_bits,
        channel_weight: blk.channel_weight,
        fix_excess_mean: fix_mean,
        fix_excess_std: fix_std,
        fix_excess_ci_lo: fix_ci_lo,
        fix_excess_ci_hi: fix_ci_hi,
        corr_nk_mean: corr_mean,
        corr_nk_std: corr_std,
        corr_nk_ci_lo: corr_ci_lo,
        corr_nk_ci_hi: corr_ci_hi,
        n_bootstrap,
        n_pairs,
    }
}

// ===========================================================================
// E21c: Joint cross-channel N-only tests
// ===========================================================================

// ---------------------------------------------------------------------------
// OLS solver via Cholesky decomposition
// ---------------------------------------------------------------------------

/// Cholesky decomposition: A = L L^T for symmetric positive-definite A.
///
/// Returns lower-triangular L, or None if A is not positive-definite.
/// Input `a` is n×n stored as `Vec<Vec<f64>>`.
fn cholesky(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let mut diag = a[j][j];
        for k in 0..j {
            diag -= l[j][k] * l[j][k];
        }
        if diag <= 0.0 {
            return None;
        }
        l[j][j] = diag.sqrt();
        for i in (j + 1)..n {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            l[i][j] = sum / l[j][j];
        }
    }
    Some(l)
}

/// Solve L L^T x = b given lower-triangular L (Cholesky factor).
///
/// Forward substitution: L y = b, then backward substitution: L^T x = y.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    // Forward: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }
    // Backward: L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Ordinary least squares: minimise ‖Xβ − y‖² via normal equations
/// (X^T X + λI)β = X^T y with Tikhonov regularisation λ = 1e-6.
///
/// `x_rows[i]` is the i-th feature vector (length p).  `y` has length n.
/// Returns coefficient vector β (length p), or None if decomposition fails.
fn ols_solve(x_rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let n = x_rows.len();
    if n == 0 {
        return None;
    }
    let p = x_rows[0].len();
    if p == 0 {
        return None;
    }

    // Build X^T X (p × p) with Tikhonov regularisation.
    let lambda = 1e-6;
    let mut xtx = vec![vec![0.0f64; p]; p];
    for row in x_rows {
        for j in 0..p {
            for k in j..p {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }
    // Symmetrise and add regularisation.
    for j in 0..p {
        xtx[j][j] += lambda;
        for k in (j + 1)..p {
            xtx[k][j] = xtx[j][k];
        }
    }

    // Build X^T y (length p).
    let mut xty = vec![0.0f64; p];
    for (row, &yi) in x_rows.iter().zip(y.iter()) {
        for j in 0..p {
            xty[j] += row[j] * yi;
        }
    }

    let l = cholesky(&xtx, p)?;
    Some(cholesky_solve(&l, &xty, p))
}

/// Compute R² = 1 − SS_res/SS_tot for predictions Xβ vs actual y.
fn ols_r_squared(x_rows: &[Vec<f64>], y: &[f64], beta: &[f64]) -> f64 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    if ss_tot < 1e-15 {
        return 0.0;
    }
    let ss_res: f64 = x_rows
        .iter()
        .zip(y.iter())
        .map(|(row, &yi)| {
            let pred: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
            (yi - pred).powi(2)
        })
        .sum();
    1.0 - ss_res / ss_tot
}

// ---------------------------------------------------------------------------
// Binned mutual information estimator
// ---------------------------------------------------------------------------

/// Assign values to quantile bins.  Returns a vector of bin indices in 0..n_bins.
///
/// Ties are broken by position: values sharing the same rank are assigned
/// to successive bins in input order.
fn assign_quantile_bins(values: &[u32], n_bins: usize) -> Vec<usize> {
    let n = values.len();
    if n == 0 || n_bins == 0 {
        return vec![0; n];
    }
    // Rank by value (stable sort preserves order within ties).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| values[i]);

    let mut bins = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        bins[idx] = (rank * n_bins) / n;
    }
    bins
}

/// Compute binned mutual information I(target; bins_i, bins_j) where
/// target is binary (0.0 or 1.0) and bins_i, bins_j are quantile bin indices.
///
/// Builds a Q × Q × 2 contingency table and computes:
///   MI = H(target) − H(target | bins_i, bins_j)
///
/// Returns MI in nats.  For independent variables, MI ≈ 0.
fn binned_mutual_information(
    target: &[f64],
    bins_i: &[usize],
    bins_j: &[usize],
    n_bins: usize,
) -> f64 {
    let n = target.len();
    debug_assert_eq!(n, bins_i.len());
    debug_assert_eq!(n, bins_j.len());
    if n == 0 {
        return 0.0;
    }

    // Count(bin_i, bin_j, target_class).
    let n_cells = n_bins * n_bins;
    let mut count_0 = vec![0u64; n_cells]; // target = 0
    let mut count_1 = vec![0u64; n_cells]; // target = 1

    for idx in 0..n {
        let cell = bins_i[idx] * n_bins + bins_j[idx];
        if target[idx] > 0.5 {
            count_1[cell] += 1;
        } else {
            count_0[cell] += 1;
        }
    }

    // H(target): marginal entropy.
    let n_pos: u64 = count_1.iter().sum();
    let n_neg: u64 = count_0.iter().sum();
    let n_f = n as f64;
    let p_pos = n_pos as f64 / n_f;
    let p_neg = n_neg as f64 / n_f;
    let h_target = -safe_plogp(p_pos) - safe_plogp(p_neg);

    // H(target | bins): conditional entropy.
    let mut h_cond = 0.0f64;
    for cell in 0..n_cells {
        let c0 = count_0[cell] as f64;
        let c1 = count_1[cell] as f64;
        let total = c0 + c1;
        if total < 0.5 {
            continue;
        }
        let weight = total / n_f;
        let p0 = c0 / total;
        let p1 = c1 / total;
        h_cond += weight * (-safe_plogp(p0) - safe_plogp(p1));
    }

    h_target - h_cond
}

/// Safe p·log(p): returns 0.0 when p ≤ 0 (avoids NaN from log(0)).
#[inline]
fn safe_plogp(p: f64) -> f64 {
    if p <= 0.0 {
        0.0
    } else {
        p * p.ln()
    }
}

// ---------------------------------------------------------------------------
// Cross-channel data structures
// ---------------------------------------------------------------------------

/// Per-channel data within a cross-channel block.
struct ChannelData {
    ell: u64,
    k1: u64,
    order: usize,
    #[allow(dead_code)]
    prim_g: u64,
    dlog: Vec<u32>,
    #[allow(dead_code)]
    g_vals: Vec<u64>,
    dlogs_g: Vec<u32>,
    smooth_vals: Vec<f64>,
    scanned_r_star: usize,
    #[allow(dead_code)]
    full_r_star: usize,
    channel_weight: u32,
}

/// Shared cross-channel block: same primes/pairs across all 7 channels.
struct CrossChannelBlock {
    n_bits: u32,
    bound: u64,
    prime_set: Vec<u64>,
    pairs: Vec<(u64, u64)>,
    channels: Vec<ChannelData>,
    /// Pair indices where ALL channels have valid g(p) and g(q).
    valid_indices: Vec<usize>,
    /// Target: s_B(g(p))·s_B(g(q)) averaged across channels for valid pairs.
    #[allow(dead_code)]
    target_mean: Vec<f64>,
    /// Target from reference channel (idx 5: k=22, ℓ=131).
    target_ref: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Cross-channel result structs
// ---------------------------------------------------------------------------

/// C1: Pairwise interaction correlation results.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelPairwiseResult {
    pub n_bits: u32,
    pub smoothness_bound: u64,
    pub n_pairs: usize,
    pub max_abs_corr: f64,
    pub mean_abs_corr: f64,
    pub max_corr_ch_i: usize,
    pub max_corr_ch_j: usize,
    pub max_corr_type: String,
    pub bonferroni_threshold: f64,
    pub n_tests: usize,
}

/// C2: OLS regression with holdout results.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelOLSResult {
    pub n_bits: u32,
    pub smoothness_bound: u64,
    pub n_features: usize,
    pub n_train: usize,
    pub n_test: usize,
    pub test_r_squared: f64,
    pub best_single_r_squared: f64,
}

/// C3: Binned mutual information results.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelMIResult {
    pub n_bits: u32,
    pub smoothness_bound: u64,
    pub channel_i: usize,
    pub channel_j: usize,
    pub n_bins: usize,
    pub observed_mi: f64,
    pub null_mean: f64,
    pub null_std: f64,
    pub z_score: f64,
    pub empirical_p_value: f64,
    pub n_permutations: usize,
    pub n_pairs: usize,
}

/// C4: Permutation null on strongest cross-channel feature.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelPermResult {
    pub n_bits: u32,
    pub smoothness_bound: u64,
    pub feature_desc: String,
    pub observed_corr: f64,
    pub null_mean: f64,
    pub null_std: f64,
    pub z_score: f64,
    pub empirical_p_value: f64,
    pub n_permutations: usize,
    pub n_pairs: usize,
}

/// Combined results for a single cross-channel block.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelBlockResult {
    pub n_bits: u32,
    pub smoothness_bound: u64,
    pub n_primes: usize,
    pub n_pairs: usize,
    pub pairwise: CrossChannelPairwiseResult,
    pub ols: CrossChannelOLSResult,
    pub mi: Option<CrossChannelMIResult>,
    pub perm_null: CrossChannelPermResult,
}

/// Top-level E21c result.
#[derive(Debug, Clone, Serialize)]
pub struct CrossChannelResult {
    pub blocks: Vec<CrossChannelBlockResult>,
}

// ---------------------------------------------------------------------------
// Cross-channel block preparation
// ---------------------------------------------------------------------------

/// Prepare a cross-channel block with shared primes/pairs across all 7 channels.
///
/// Uses a channel-independent seed `0xE21c_0000 + n_bits` so all channels
/// operate on the SAME (p, q) pairs — critical for cross-channel tests.
fn prepare_cross_channel_block(
    n_bits: u32,
    bound: u64,
    top_k: usize,
) -> CrossChannelBlock {
    use algebra::{best_character, build_dlog_table, primitive_root};

    let seed = 0xE21c_0000u64 + n_bits as u64;
    let (prime_set, pairs) = generate_primes_and_pairs(n_bits, 500, 20_000, seed);

    let mut channels = Vec::with_capacity(7);
    for ch in eisenstein_hunt::CHANNELS {
        let ell = ch.ell;
        let k1 = (ch.weight - 1) as u64;
        let order = (ell - 1) as usize;

        // Full-group smoothness spectrum → find r*.
        let full = smoothness_spectrum(ell, bound, top_k);
        let full_r_star = full.top_amplitudes.first().map(|&(r, _)| r).unwrap_or(1);

        let prim_g = primitive_root(ell);
        let dlog = build_dlog_table(ell, prim_g);

        let g_vals: Vec<u64> = prime_set
            .iter()
            .map(|&p| (1 + mod_pow(p, k1, ell)) % ell)
            .collect();

        let dlogs_g: Vec<u32> = g_vals
            .iter()
            .map(|&gp| if gp == 0 { u32::MAX } else { dlog[gp as usize] })
            .collect();

        let smooth_vals: Vec<f64> = g_vals
            .iter()
            .zip(dlogs_g.iter())
            .map(|(&gp, &d)| {
                if d != u32::MAX && gp > 0 && is_b_smooth(gp, bound) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Character scan on valid primes.
        let (valid_smooth, valid_dlogs): (Vec<f64>, Vec<u32>) = smooth_vals
            .iter()
            .zip(dlogs_g.iter())
            .filter(|(_, &d)| d != u32::MAX)
            .map(|(&s, &d)| (s, d))
            .unzip();

        let (scanned_r_star, _, _, _) =
            best_character(&valid_smooth, &valid_dlogs, order, top_k);

        channels.push(ChannelData {
            ell,
            k1,
            order,
            prim_g,
            dlog,
            g_vals,
            dlogs_g,
            smooth_vals,
            scanned_r_star,
            full_r_star,
            channel_weight: ch.weight,
        });
    }

    // Find valid pairs: g(p) ≠ 0 AND g(q) ≠ 0 in ALL 7 channels.
    let mut valid_indices = Vec::new();
    let prime_idx = |p: u64| -> usize { prime_set.partition_point(|&x| x < p) };

    for (pair_idx, &(p, q)) in pairs.iter().enumerate() {
        let pi = prime_idx(p);
        let qi = prime_idx(q);
        let all_valid = channels
            .iter()
            .all(|ch| ch.dlogs_g[pi] != u32::MAX && ch.dlogs_g[qi] != u32::MAX);
        if all_valid {
            valid_indices.push(pair_idx);
        }
    }

    // Build targets for valid pairs.
    let n_valid = valid_indices.len();
    let n_channels = channels.len();
    let mut target_ref = Vec::with_capacity(n_valid);
    let mut target_mean = Vec::with_capacity(n_valid);

    for &pair_idx in &valid_indices {
        let (p, q) = pairs[pair_idx];
        let pi = prime_idx(p);
        let qi = prime_idx(q);

        // Reference channel (idx 5: k=22, ℓ=131).
        let sp = channels[5].smooth_vals[pi] * channels[5].smooth_vals[qi];
        target_ref.push(sp);

        // Mean across all channels.
        let mean: f64 = channels
            .iter()
            .map(|ch| ch.smooth_vals[pi] * ch.smooth_vals[qi])
            .sum::<f64>()
            / n_channels as f64;
        target_mean.push(mean);
    }

    CrossChannelBlock {
        n_bits,
        bound,
        prime_set,
        pairs,
        channels,
        valid_indices,
        target_mean,
        target_ref,
    }
}

// ---------------------------------------------------------------------------
// Cross-channel N-only feature computation
// ---------------------------------------------------------------------------

/// Compute 14-element N-only feature vectors for each valid pair.
///
/// For each valid pair (p, q) and each channel i (0..7):
///   features[2*i]   = Re(χ_{r*_i}(N^{k_i-1} mod ℓ_i))
///   features[2*i+1] = Im(χ_{r*_i}(N^{k_i-1} mod ℓ_i))
///
/// r* is the `scanned_r_star` from character scan on the restricted set.
fn compute_n_only_features(block: &CrossChannelBlock) -> Vec<Vec<f64>> {
    use algebra::{im_char, re_char};

    let n_channels = block.channels.len();
    let n_features = 2 * n_channels; // 14 for 7 channels

    block
        .valid_indices
        .iter()
        .map(|&pair_idx| {
            let (p, q) = block.pairs[pair_idx];
            let n_val = p as u128 * q as u128;

            let mut feats = Vec::with_capacity(n_features);
            for ch in &block.channels {
                // N^{k-1} mod ℓ — use u128 intermediate for large N.
                let nk_val = mod_pow_u128(n_val, ch.k1 as u128, ch.ell as u128) as u64;
                let dlog_nk = if nk_val == 0 {
                    u32::MAX
                } else {
                    ch.dlog[nk_val as usize]
                };
                if dlog_nk == u32::MAX {
                    feats.push(0.0);
                    feats.push(0.0);
                } else {
                    feats.push(re_char(dlog_nk, ch.scanned_r_star, ch.order));
                    feats.push(im_char(dlog_nk, ch.scanned_r_star, ch.order));
                }
            }
            feats
        })
        .collect()
}

/// Modular exponentiation for u128 base/modulus (needed for N > 2^64).
fn mod_pow_u128(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u128;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

// ---------------------------------------------------------------------------
// Test C1: Pairwise interaction correlations
// ---------------------------------------------------------------------------

/// For each channel pair (i, j) with i < j, compute 4 cross-products:
///   Re_i·Re_j, Re_i·Im_j, Im_i·Re_j, Im_i·Im_j
/// and correlate each with `target`.
///
/// Total tests: C(7,2) × 4 = 84.  Bonferroni threshold at α = 0.05.
fn run_pairwise_interactions(
    block: &CrossChannelBlock,
    features: &[Vec<f64>],
) -> CrossChannelPairwiseResult {
    use algebra::pearson_corr;

    let n_channels = block.channels.len();
    let n_pairs = features.len();
    let target = &block.target_ref;
    let cross_names = ["Re·Re", "Re·Im", "Im·Re", "Im·Im"];

    let mut max_abs_corr = 0.0f64;
    let mut sum_abs_corr = 0.0f64;
    let mut n_tests = 0usize;
    let mut max_ch_i = 0;
    let mut max_ch_j = 0;
    let mut max_type = String::new();

    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            // Indices into the 14-element feature vector.
            let ri = 2 * i;     // Re_i
            let ii = 2 * i + 1; // Im_i
            let rj = 2 * j;     // Re_j
            let ij = 2 * j + 1; // Im_j

            let cross_pairs = [(ri, rj), (ri, ij), (ii, rj), (ii, ij)];

            for (idx, &(fi, fj)) in cross_pairs.iter().enumerate() {
                let cross_product: Vec<f64> = features
                    .iter()
                    .map(|f| f[fi] * f[fj])
                    .collect();
                let corr = pearson_corr(&cross_product, target);
                let abs_corr = corr.abs();
                sum_abs_corr += abs_corr;
                n_tests += 1;

                if abs_corr > max_abs_corr {
                    max_abs_corr = abs_corr;
                    max_ch_i = i;
                    max_ch_j = j;
                    max_type = cross_names[idx].to_string();
                }
            }
        }
    }

    let mean_abs_corr = if n_tests > 0 {
        sum_abs_corr / n_tests as f64
    } else {
        0.0
    };

    // Bonferroni threshold: two-sided test at α/n_tests.
    // For Pearson r with n-2 df, approximate: threshold ≈ sqrt(2 * ln(2*n_tests/α) / (n-2)).
    let alpha = 0.05;
    let bonferroni_threshold = if n_pairs > 2 {
        (2.0 * (2.0 * n_tests as f64 / alpha).ln() / (n_pairs - 2) as f64).sqrt()
    } else {
        1.0
    };

    CrossChannelPairwiseResult {
        n_bits: block.n_bits,
        smoothness_bound: block.bound,
        n_pairs,
        max_abs_corr,
        mean_abs_corr,
        max_corr_ch_i: max_ch_i,
        max_corr_ch_j: max_ch_j,
        max_corr_type: max_type,
        bonferroni_threshold,
        n_tests,
    }
}

// ---------------------------------------------------------------------------
// Test C2: OLS regression with holdout
// ---------------------------------------------------------------------------

/// Build 35 features per pair (14 linear + 21 Re_i·Re_j products),
/// fit OLS on first half, evaluate R² on second half.
fn run_cross_channel_ols(
    block: &CrossChannelBlock,
    features: &[Vec<f64>],
) -> CrossChannelOLSResult {
    use algebra::pearson_corr;

    let n = features.len();
    let n_channels = block.channels.len();
    let target = &block.target_ref;

    // Build augmented feature matrix: 14 linear + 21 products = 35 features.
    let n_products = n_channels * (n_channels - 1) / 2; // C(7,2) = 21
    let n_feats = 2 * n_channels + n_products;           // 14 + 21 = 35

    let augmented: Vec<Vec<f64>> = features
        .iter()
        .map(|f| {
            let mut row = f.clone();
            // Add Re_i · Re_j products for i < j.
            for i in 0..n_channels {
                for j in (i + 1)..n_channels {
                    row.push(f[2 * i] * f[2 * j]);
                }
            }
            row
        })
        .collect();

    // 50/50 train/test split (deterministic: first half train).
    let n_train = n / 2;
    let n_test = n - n_train;

    let (train_x, test_x) = augmented.split_at(n_train);
    let (train_y, test_y) = target.split_at(n_train);

    let test_r_squared = if n_train > n_feats && n_test > 2 {
        match ols_solve(train_x, train_y) {
            Some(beta) => ols_r_squared(test_x, test_y, &beta),
            None => f64::NEG_INFINITY,
        }
    } else {
        // Not enough samples for OLS; return negative R².
        f64::NEG_INFINITY
    };

    // Best single-channel R² (max over 7 of pearson_corr(Re_i, target)²).
    let best_single_r_squared = (0..n_channels)
        .map(|i| {
            let re_vals: Vec<f64> = features.iter().map(|f| f[2 * i]).collect();
            let c = pearson_corr(&re_vals, target);
            c * c
        })
        .fold(0.0f64, f64::max);

    CrossChannelOLSResult {
        n_bits: block.n_bits,
        smoothness_bound: block.bound,
        n_features: n_feats,
        n_train,
        n_test,
        test_r_squared,
        best_single_r_squared,
    }
}

// ---------------------------------------------------------------------------
// Test C3: Binned mutual information
// ---------------------------------------------------------------------------

/// Compute MI between binned N-derived dlogs (from two channels) and
/// binarised smoothness target, with permutation null.
fn run_cross_channel_mi(
    block: &CrossChannelBlock,
    features: &[Vec<f64>],
    n_bins: usize,
    n_permutations: usize,
    seed: u64,
) -> Option<CrossChannelMIResult> {
    use rand::seq::SliceRandom;

    let n = features.len();

    // Require sufficient bin occupancy: n ≥ n_bins² × 4.
    if n < n_bins * n_bins * 4 {
        return None;
    }

    // Use channels with smallest ℓ for best resolution:
    // idx 5 (ℓ=131) and idx 3 (ℓ=283).
    let ch_i = 5;
    let ch_j = 3;

    // Get N^{k-1} mod ℓ dlogs for each channel.
    let mut dlogs_i = Vec::with_capacity(n);
    let mut dlogs_j = Vec::with_capacity(n);

    for &pair_idx in &block.valid_indices {
        let (p, q) = block.pairs[pair_idx];
        let n_val = p as u128 * q as u128;

        let nk_i = mod_pow_u128(
            n_val,
            block.channels[ch_i].k1 as u128,
            block.channels[ch_i].ell as u128,
        ) as u64;
        let nk_j = mod_pow_u128(
            n_val,
            block.channels[ch_j].k1 as u128,
            block.channels[ch_j].ell as u128,
        ) as u64;

        let dlog_i = if nk_i == 0 { 0u32 } else { block.channels[ch_i].dlog[nk_i as usize] };
        let dlog_j = if nk_j == 0 { 0u32 } else { block.channels[ch_j].dlog[nk_j as usize] };
        dlogs_i.push(dlog_i);
        dlogs_j.push(dlog_j);
    }

    let bins_i = assign_quantile_bins(&dlogs_i, n_bins);
    let bins_j = assign_quantile_bins(&dlogs_j, n_bins);

    // Binarise target (threshold at mean).
    let mean_target: f64 = block.target_ref.iter().sum::<f64>() / n as f64;
    let binary_target: Vec<f64> = block
        .target_ref
        .iter()
        .map(|&t| if t > mean_target { 1.0 } else { 0.0 })
        .collect();

    let observed_mi = binned_mutual_information(&binary_target, &bins_i, &bins_j, n_bins);

    // Permutation null: shuffle target n_permutations times.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut null_mis = Vec::with_capacity(n_permutations);
    let mut shuffled = binary_target.clone();

    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let mi = binned_mutual_information(&shuffled, &bins_i, &bins_j, n_bins);
        null_mis.push(mi);
    }

    let null_mean: f64 = null_mis.iter().sum::<f64>() / n_permutations as f64;
    let null_var: f64 = null_mis
        .iter()
        .map(|&m| (m - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    let z_score = if null_std > 1e-15 {
        (observed_mi - null_mean) / null_std
    } else {
        0.0
    };

    let empirical_p_value = null_mis
        .iter()
        .filter(|&&m| m >= observed_mi)
        .count() as f64
        / n_permutations as f64;

    Some(CrossChannelMIResult {
        n_bits: block.n_bits,
        smoothness_bound: block.bound,
        channel_i: ch_i,
        channel_j: ch_j,
        n_bins,
        observed_mi,
        null_mean,
        null_std,
        z_score,
        empirical_p_value,
        n_permutations,
        n_pairs: n,
    })
}

// ---------------------------------------------------------------------------
// Test C4: Permutation null on strongest cross-channel feature
// ---------------------------------------------------------------------------

/// Permutation null on the maximum |correlation| across ALL 84 cross-channel features.
///
/// Properly accounts for selection bias by computing max|corr| over all features
/// for each permutation, matching the selection process in C1.
fn run_cross_channel_perm(
    block: &CrossChannelBlock,
    features: &[Vec<f64>],
    best_i: usize,
    best_j: usize,
    best_type: &str,
    n_permutations: usize,
    seed: u64,
) -> CrossChannelPermResult {
    use algebra::pearson_corr;
    use rand::seq::SliceRandom;

    let n_channels = block.channels.len();
    let n = features.len();
    let target = &block.target_ref;

    // Precompute all 84 cross-product feature vectors.
    let mut all_cross: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            let pairs = [
                (2 * i, 2 * j),
                (2 * i, 2 * j + 1),
                (2 * i + 1, 2 * j),
                (2 * i + 1, 2 * j + 1),
            ];
            for &(fi, fj) in &pairs {
                let cross: Vec<f64> = features.iter().map(|f| f[fi] * f[fj]).collect();
                all_cross.push(cross);
            }
        }
    }

    // Observed max |corr| across all 84 features.
    let observed_max: f64 = all_cross
        .iter()
        .map(|cross| pearson_corr(cross, target).abs())
        .fold(0.0f64, f64::max);

    // Identify the specific observed correlation for the named best feature.
    let (fi, fj) = match best_type {
        "Re·Re" => (2 * best_i, 2 * best_j),
        "Re·Im" => (2 * best_i, 2 * best_j + 1),
        "Im·Re" => (2 * best_i + 1, 2 * best_j),
        "Im·Im" => (2 * best_i + 1, 2 * best_j + 1),
        _ => (2 * best_i, 2 * best_j),
    };
    let specific_cross: Vec<f64> = features.iter().map(|f| f[fi] * f[fj]).collect();
    let observed_corr = pearson_corr(&specific_cross, target);

    // Permutation null: for each permutation, compute max|corr| over all 84 features.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut null_max_corrs = Vec::with_capacity(n_permutations);
    let mut shuffled = target.clone();

    for _ in 0..n_permutations {
        shuffled.shuffle(&mut rng);
        let perm_max: f64 = all_cross
            .iter()
            .map(|cross| pearson_corr(cross, &shuffled).abs())
            .fold(0.0f64, f64::max);
        null_max_corrs.push(perm_max);
    }

    let null_mean: f64 = null_max_corrs.iter().sum::<f64>() / n_permutations as f64;
    let null_var: f64 = null_max_corrs
        .iter()
        .map(|&c| (c - null_mean).powi(2))
        .sum::<f64>()
        / n_permutations as f64;
    let null_std = null_var.sqrt();

    let z_score = if null_std > 1e-15 {
        (observed_max - null_mean) / null_std
    } else {
        0.0
    };

    // p-value: fraction of permutations with max|corr| ≥ observed max.
    let empirical_p_value = null_max_corrs
        .iter()
        .filter(|&&c| c >= observed_max)
        .count() as f64
        / n_permutations as f64;

    let feature_desc = format!(
        "max_84 (best: ch{}(k={},ℓ={})×ch{}(k={},ℓ={}) {})",
        best_i,
        block.channels[best_i].channel_weight,
        block.channels[best_i].ell,
        best_j,
        block.channels[best_j].channel_weight,
        block.channels[best_j].ell,
        best_type,
    );

    CrossChannelPermResult {
        n_bits: block.n_bits,
        smoothness_bound: block.bound,
        feature_desc,
        observed_corr,
        null_mean,
        null_std,
        z_score,
        empirical_p_value,
        n_permutations,
        n_pairs: n,
    }
}

// ---------------------------------------------------------------------------
// Top-level cross-channel runner
// ---------------------------------------------------------------------------

/// Run the full E21c cross-channel test suite across multiple bit sizes and bounds.
pub fn run_cross_channel_tests(
    bit_sizes: &[u32],
    bounds: &[u64],
    top_k: usize,
    n_permutations: usize,
    n_bins: usize,
    seed: u64,
) -> CrossChannelResult {
    let mut blocks = Vec::new();

    for &n_bits in bit_sizes {
        for &bound in bounds {
            eprintln!(
                "[E21c] Preparing cross-channel block: n_bits={}, B={}",
                n_bits, bound,
            );

            let block = prepare_cross_channel_block(n_bits, bound, top_k);
            let n_primes = block.prime_set.len();
            let n_valid = block.valid_indices.len();

            eprintln!(
                "  {} primes, {} pairs, {} valid (all channels)",
                n_primes,
                block.pairs.len(),
                n_valid,
            );

            if n_valid < 10 {
                eprintln!("  Skipping: too few valid pairs ({n_valid})");
                continue;
            }

            let features = compute_n_only_features(&block);

            // C1: Pairwise interaction correlations.
            let pairwise = run_pairwise_interactions(&block, &features);
            eprintln!(
                "  C1: max|corr|={:.4}, mean|corr|={:.4}, Bonf={:.4}",
                pairwise.max_abs_corr, pairwise.mean_abs_corr, pairwise.bonferroni_threshold,
            );

            // C2: OLS with holdout.
            let ols = run_cross_channel_ols(&block, &features);
            eprintln!(
                "  C2: test R²={:.4}, best single R²={:.4}",
                ols.test_r_squared, ols.best_single_r_squared,
            );

            // C3: Binned mutual information.
            let mi = run_cross_channel_mi(
                &block,
                &features,
                n_bins,
                n_permutations,
                seed ^ (n_bits as u64 * 0x1000 + bound),
            );
            if let Some(ref m) = mi {
                eprintln!(
                    "  C3: MI={:.6}, z={:.2}, p={:.3}",
                    m.observed_mi, m.z_score, m.empirical_p_value,
                );
            } else {
                eprintln!("  C3: skipped (insufficient pairs for binning)");
            }

            // C4: Permutation null on strongest feature.
            let perm_null = run_cross_channel_perm(
                &block,
                &features,
                pairwise.max_corr_ch_i,
                pairwise.max_corr_ch_j,
                &pairwise.max_corr_type,
                n_permutations,
                seed ^ (n_bits as u64 * 0x2000 + bound),
            );
            eprintln!(
                "  C4: obs={:.4}, z={:.2}, p={:.3}",
                perm_null.observed_corr, perm_null.z_score, perm_null.empirical_p_value,
            );

            blocks.push(CrossChannelBlockResult {
                n_bits,
                smoothness_bound: bound,
                n_primes,
                n_pairs: n_valid,
                pairwise,
                ols,
                mi,
                perm_null,
            });
        }
    }

    CrossChannelResult { blocks }
}

// ===========================================================================
// E22: Eisenstein-Scored Sieve Enrichment
// ===========================================================================
//
// Tests whether the found Eisenstein smoothness bias (2-6× fix_excess at group
// level, confirmed in E21b) can accelerate a quadratic sieve.
//
// Key insight: For QS polynomial Q(x) = (x + ⌊√N⌋)² − N, the residue
// Q(x) mod ℓ IS computable from public information (no factorisation needed).
// The group-level enrichment tells us which cosets in (ℤ/ℓℤ)* are smooth-rich.
// The question is whether scoring Q(x) via its character position predicts
// smoothness of the FULL Q(x) when Q(x) >> ℓ.
//
// Phase 1: Group-level enrichment profile (exact computation on (ℤ/ℓℤ)*)
// Phase 2: QS polynomial enrichment (does char score predict smoothness of Q(x)?)
// Phase 3: Joint multi-channel scoring (CRT across channels)
// Phase 4: Direct sieve speedup measurement (T_random vs T_scored)

/// Integer square root via Newton's method.
pub fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    // Initial guess: 2^(ceil(bits/2))
    let bits = 128 - n.leading_zeros();
    let mut x = 1u128 << ((bits + 1) / 2);
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            return x;
        }
        x = x1;
    }
}

/// Check if n is B-smooth using trial division up to bound.
/// Works for u128 values (needed for QS polynomial values).
fn is_b_smooth_u128(mut n: u128, bound: u64) -> bool {
    if n <= 1 {
        return true;
    }
    let mut d = 2u64;
    while (d as u128) * (d as u128) <= n && d <= bound {
        while n % (d as u128) == 0 {
            n /= d as u128;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    // After trial division, if n > 1 and n > bound, it has a large factor.
    n == 1 || n <= bound as u128
}

// ---------------------------------------------------------------------------
// Phase 1: Group-level enrichment profile
// ---------------------------------------------------------------------------

/// Per-channel group-level enrichment result.
///
/// Measures the smoothness enrichment at the best character frequency r*
/// on the FULL group (ℤ/ℓℤ)*.  This is the theoretical ceiling — the maximum
/// enrichment achievable IF Q(x) mod ℓ were uniformly distributed in the
/// smooth-rich coset.
#[derive(Debug, Clone, Serialize)]
pub struct GroupEnrichmentResult {
    pub ell: u64,
    pub weight: u32,
    pub order: usize,
    pub smooth_bound: u64,
    /// Fraction of (ℤ/ℓℤ)* that is B-smooth.
    pub smooth_fraction: f64,
    /// Best character index from full-group DFT.
    pub full_r_star: usize,
    /// Amplitude of best character (Fourier coefficient magnitude).
    pub full_amplitude: f64,
    /// Enrichment ratio: P(smooth | top quartile) / P(smooth | overall).
    pub enrichment_top_quartile: f64,
    /// Enrichment ratio: P(smooth | top decile) / P(smooth | overall).
    pub enrichment_top_decile: f64,
    /// Enrichment ratio: P(smooth | top ventile) / P(smooth | overall).
    pub enrichment_top_ventile: f64,
    /// Number of quantile bins used.
    pub n_bins: usize,
    /// Enrichment per bin: (bin_idx, fraction_smooth, relative_enrichment).
    pub bin_enrichments: Vec<(usize, f64, f64)>,
}

/// Compute group-level enrichment profile for one channel at one smoothness bound.
///
/// Partitions (ℤ/ℓℤ)* into quantile bins by Re(χ_{r*}(a)) and measures
/// the smoothness rate in each bin.
pub fn compute_group_enrichment(
    ell: u64,
    weight: u32,
    smooth_bound: u64,
    n_bins: usize,
) -> GroupEnrichmentResult {
    use algebra::{build_dlog_table, primitive_root, re_char};

    let order = (ell - 1) as usize;
    let prim_g = primitive_root(ell);
    let dlog = build_dlog_table(ell, prim_g);

    // Full-group smoothness spectrum to find r*.
    let spectrum = smoothness_spectrum_full(ell, smooth_bound);
    let full_r_star = spectrum
        .coefficients
        .iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .map(|&(r, _, _, _)| r)
        .unwrap_or(1);
    let full_amplitude = spectrum
        .coefficients
        .iter()
        .find(|&&(r, _, _, _)| r == full_r_star)
        .map(|&(_, _, _, amp)| amp)
        .unwrap_or(0.0);

    // For each a in 1..ell, compute Re(χ_{r*}(a)) and whether a is B-smooth.
    let mut scored_elements: Vec<(f64, bool)> = Vec::with_capacity(order);
    for a in 1..ell {
        let d = dlog[a as usize];
        let re_val = re_char(d, full_r_star, order);
        let smooth = is_b_smooth(a, smooth_bound);
        scored_elements.push((re_val, smooth));
    }

    let overall_smooth: f64 =
        scored_elements.iter().filter(|(_, s)| *s).count() as f64 / order as f64;

    // Sort by score descending, then partition into quantile bins.
    scored_elements.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let bin_size = order / n_bins;
    let mut bin_enrichments = Vec::with_capacity(n_bins);
    for bin_idx in 0..n_bins {
        let start = bin_idx * bin_size;
        let end = if bin_idx == n_bins - 1 {
            order
        } else {
            (bin_idx + 1) * bin_size
        };
        let bin_slice = &scored_elements[start..end];
        let n_smooth = bin_slice.iter().filter(|(_, s)| *s).count();
        let frac = n_smooth as f64 / bin_slice.len() as f64;
        let enrichment = if overall_smooth > 1e-15 {
            frac / overall_smooth
        } else {
            0.0
        };
        bin_enrichments.push((bin_idx, frac, enrichment));
    }

    // Compute specific quantile enrichments.
    let top_quartile_n = order / 4;
    let top_decile_n = order / 10;
    let top_ventile_n = order / 20;

    let enrichment_at = |top_n: usize| -> f64 {
        let n_smooth = scored_elements[..top_n].iter().filter(|(_, s)| *s).count();
        let frac = n_smooth as f64 / top_n as f64;
        if overall_smooth > 1e-15 {
            frac / overall_smooth
        } else {
            0.0
        }
    };

    GroupEnrichmentResult {
        ell,
        weight,
        order,
        smooth_bound,
        smooth_fraction: overall_smooth,
        full_r_star,
        full_amplitude,
        enrichment_top_quartile: enrichment_at(top_quartile_n),
        enrichment_top_decile: enrichment_at(top_decile_n),
        enrichment_top_ventile: enrichment_at(top_ventile_n),
        n_bins,
        bin_enrichments,
    }
}

// ---------------------------------------------------------------------------
// Phase 2: QS polynomial enrichment
// ---------------------------------------------------------------------------

/// Per-block QS polynomial enrichment result for one channel.
#[derive(Debug, Clone, Serialize)]
pub struct QSChannelEnrichmentResult {
    pub ell: u64,
    pub weight: u32,
    pub full_r_star: usize,
    /// Number of QS polynomial values tested.
    pub n_tested: usize,
    /// Number of Q(x) values that are B-smooth.
    pub n_smooth: usize,
    /// Overall smoothness rate.
    pub smooth_rate: f64,
    /// Smoothness rate in top-quartile by character score.
    pub smooth_rate_top_q: f64,
    /// Enrichment: smooth_rate_top_q / smooth_rate.
    pub enrichment_top_q: f64,
    /// Smoothness rate in bottom-quartile by character score.
    pub smooth_rate_bot_q: f64,
    /// Enrichment in bottom quartile.
    pub enrichment_bot_q: f64,
    /// Pearson correlation between character score and smoothness indicator.
    pub pearson_corr_score_smooth: f64,
    /// Average log2 of |Q(x)| (measures how far Q(x) >> ℓ).
    pub avg_log2_qx: f64,
    /// Ratio log2(Q(x)) / log2(ℓ) — measures the "overflow" factor.
    pub overflow_ratio: f64,
}

/// Per-block QS polynomial enrichment result.
#[derive(Debug, Clone, Serialize)]
pub struct QSBlockResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_values: usize,
    /// Number of Q(x) that are B-smooth.
    pub n_smooth_total: usize,
    pub overall_smooth_rate: f64,
    /// Per-channel results.
    pub channels: Vec<QSChannelEnrichmentResult>,
    /// Best single-channel enrichment (top quartile).
    pub best_single_enrichment: f64,
    pub best_single_channel_idx: usize,
}

/// Generate QS polynomial values Q(x) = (x + floor(sqrt(N)))^2 - N for a
/// given semiprime N, scanning x = 1, 2, ...
///
/// Returns Vec<(x, Q(x))> for the first `count` values where Q(x) > 0.
fn generate_qs_values(n: u128, count: usize) -> Vec<(u64, u128)> {
    let sqrt_n = isqrt(n);
    let mut results = Vec::with_capacity(count);
    for x in 1..=(count as u128 * 2 + 100) {
        let val = (sqrt_n + x) * (sqrt_n + x);
        if val > n {
            let qx = val - n;
            results.push((x as u64, qx));
            if results.len() >= count {
                break;
            }
        }
    }
    results
}

/// Compute QS enrichment for one block (one N, one smoothness bound).
///
/// For each channel, scores Q(x) mod ℓ by Re(χ_{r*}(Q(x) mod ℓ)),
/// then measures whether high-scoring Q(x) values are more likely to be smooth.
fn compute_qs_block(
    n_bits: u32,
    smooth_bound: u64,
    n_values: usize,
    seed: u64,
) -> QSBlockResult {
    use algebra::{build_dlog_table, primitive_root, re_char};

    // Generate a semiprime N of the right size.
    let n = generate_semiprime(n_bits, seed);

    // Generate QS polynomial values.
    let qs_values = generate_qs_values(n, n_values);
    let actual_count = qs_values.len();

    // Determine smoothness of each Q(x).
    let smooth_flags: Vec<bool> = qs_values
        .iter()
        .map(|&(_, qx)| is_b_smooth_u128(qx, smooth_bound))
        .collect();
    let n_smooth_total = smooth_flags.iter().filter(|&&s| s).count();
    let overall_smooth_rate = n_smooth_total as f64 / actual_count as f64;

    // Average log2(|Q(x)|).
    let avg_log2_qx: f64 = qs_values
        .iter()
        .map(|&(_, qx)| (qx as f64).log2())
        .sum::<f64>()
        / actual_count as f64;

    // Per-channel enrichment.
    let mut channel_results = Vec::with_capacity(7);
    let mut best_single_enrichment = 0.0f64;
    let mut best_single_channel_idx = 0usize;

    for (ch_idx, ch) in eisenstein_hunt::CHANNELS.iter().enumerate() {
        let ell = ch.ell;
        let order = (ell - 1) as usize;
        let prim_g = primitive_root(ell);
        let dlog = build_dlog_table(ell, prim_g);

        // Find r* from full-group smoothness spectrum.
        let spectrum = smoothness_spectrum_full(ell, smooth_bound);
        let full_r_star = spectrum
            .coefficients
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .map(|&(r, _, _, _)| r)
            .unwrap_or(1);

        // Score each Q(x): compute Re(χ_{r*}(Q(x) mod ℓ)).
        let mut scores: Vec<f64> = Vec::with_capacity(actual_count);
        let mut valid_indices: Vec<usize> = Vec::with_capacity(actual_count);

        for (i, &(_, qx)) in qs_values.iter().enumerate() {
            let qx_mod = (qx % ell as u128) as u64;
            if qx_mod == 0 {
                // Q(x) ≡ 0 (mod ℓ) — skip (not in (ℤ/ℓℤ)*)
                continue;
            }
            let d = dlog[qx_mod as usize];
            if d == u32::MAX {
                continue;
            }
            scores.push(re_char(d, full_r_star, order));
            valid_indices.push(i);
        }

        let n_tested = valid_indices.len();
        if n_tested < 4 {
            channel_results.push(QSChannelEnrichmentResult {
                ell,
                weight: ch.weight,
                full_r_star,
                n_tested,
                n_smooth: 0,
                smooth_rate: 0.0,
                smooth_rate_top_q: 0.0,
                enrichment_top_q: 0.0,
                smooth_rate_bot_q: 0.0,
                enrichment_bot_q: 0.0,
                pearson_corr_score_smooth: 0.0,
                avg_log2_qx,
                overflow_ratio: avg_log2_qx / (ell as f64).log2(),
            });
            continue;
        }

        // Sort by score descending.
        let mut sorted_idx: Vec<usize> = (0..n_tested).collect();
        sorted_idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let smooth_for_valid: Vec<f64> = valid_indices
            .iter()
            .map(|&i| if smooth_flags[i] { 1.0 } else { 0.0 })
            .collect();

        let n_smooth = smooth_for_valid.iter().filter(|&&s| s > 0.5).count();
        let smooth_rate = n_smooth as f64 / n_tested as f64;

        // Top and bottom quartile enrichment.
        let q_size = n_tested / 4;
        let top_q_smooth = sorted_idx[..q_size]
            .iter()
            .filter(|&&idx| smooth_for_valid[idx] > 0.5)
            .count();
        let bot_q_smooth = sorted_idx[n_tested - q_size..]
            .iter()
            .filter(|&&idx| smooth_for_valid[idx] > 0.5)
            .count();

        let smooth_rate_top_q = top_q_smooth as f64 / q_size as f64;
        let smooth_rate_bot_q = bot_q_smooth as f64 / q_size as f64;
        let enrichment_top_q = if smooth_rate > 1e-15 {
            smooth_rate_top_q / smooth_rate
        } else {
            0.0
        };
        let enrichment_bot_q = if smooth_rate > 1e-15 {
            smooth_rate_bot_q / smooth_rate
        } else {
            0.0
        };

        // Pearson correlation between score and smoothness.
        let pearson_corr_score_smooth =
            algebra::pearson_corr(&scores, &smooth_for_valid);

        if enrichment_top_q > best_single_enrichment {
            best_single_enrichment = enrichment_top_q;
            best_single_channel_idx = ch_idx;
        }

        channel_results.push(QSChannelEnrichmentResult {
            ell,
            weight: ch.weight,
            full_r_star,
            n_tested,
            n_smooth,
            smooth_rate,
            smooth_rate_top_q,
            enrichment_top_q,
            smooth_rate_bot_q,
            enrichment_bot_q,
            pearson_corr_score_smooth,
            avg_log2_qx,
            overflow_ratio: avg_log2_qx / (ell as f64).log2(),
        });
    }

    QSBlockResult {
        n_bits,
        smooth_bound,
        n_values: actual_count,
        n_smooth_total,
        overall_smooth_rate,
        channels: channel_results,
        best_single_enrichment,
        best_single_channel_idx,
    }
}

/// Generate a random semiprime N = p*q where N has approximately n_bits bits.
///
/// Uses deterministic RNG seeded by `seed` for reproducibility.
fn generate_semiprime(n_bits: u32, seed: u64) -> u128 {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);

    let half = n_bits / 2;
    let lo = 1u64 << (half - 1);
    let hi = 1u64 << half;

    loop {
        let p_cand = lo + rng.gen_range(0..(hi - lo));
        if !is_prime_u64(p_cand) {
            continue;
        }
        let q_cand = lo + rng.gen_range(0..(hi - lo));
        if !is_prime_u64(q_cand) || q_cand == p_cand {
            continue;
        }
        let n = p_cand as u128 * q_cand as u128;
        let n_actual_bits = 128 - n.leading_zeros();
        if n_actual_bits == n_bits {
            return n;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Joint multi-channel scoring
// ---------------------------------------------------------------------------

/// Joint scoring result across multiple channels.
#[derive(Debug, Clone, Serialize)]
pub struct JointScoringResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_values: usize,
    pub overall_smooth_rate: f64,
    /// Best single-channel enrichment (top quartile).
    pub best_single_enrichment: f64,
    /// Joint score enrichment: average Re(χ_{r*}(Q(x) mod ℓ)) over channels.
    pub joint_enrichment_top_q: f64,
    /// Joint score enrichment (top decile).
    pub joint_enrichment_top_d: f64,
    /// Joint Pearson correlation.
    pub joint_pearson_corr: f64,
    /// Number of channels contributing to joint score.
    pub n_channels_used: usize,
    /// Per-channel weight in joint score (correlation with smoothness).
    pub channel_weights: Vec<(usize, f64)>,
}

/// Compute joint multi-channel scoring for QS polynomial values.
///
/// Scores each Q(x) by the average Re(χ_{r*_i}(Q(x) mod ℓ_i)) across channels,
/// weighting channels by their group-level enrichment amplitude.
fn compute_joint_scoring(
    n_bits: u32,
    smooth_bound: u64,
    n_values: usize,
    seed: u64,
) -> JointScoringResult {
    use algebra::{build_dlog_table, primitive_root, re_char};

    let n = generate_semiprime(n_bits, seed);
    let qs_values = generate_qs_values(n, n_values);
    let actual_count = qs_values.len();

    let smooth_flags: Vec<bool> = qs_values
        .iter()
        .map(|&(_, qx)| is_b_smooth_u128(qx, smooth_bound))
        .collect();
    let overall_smooth_rate =
        smooth_flags.iter().filter(|&&s| s).count() as f64 / actual_count as f64;

    // Per-channel infrastructure.
    struct ChannelInfo {
        ell: u64,
        order: usize,
        dlog: Vec<u32>,
        full_r_star: usize,
        group_amplitude: f64,
    }

    let mut channel_infos: Vec<ChannelInfo> = Vec::with_capacity(7);
    for ch in eisenstein_hunt::CHANNELS {
        let ell = ch.ell;
        let order = (ell - 1) as usize;
        let prim_g = primitive_root(ell);
        let dlog_tbl = build_dlog_table(ell, prim_g);

        let spectrum = smoothness_spectrum_full(ell, smooth_bound);
        let (full_r_star, group_amplitude) = spectrum
            .coefficients
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .map(|&(r, _, _, amp)| (r, amp))
            .unwrap_or((1, 0.0));

        channel_infos.push(ChannelInfo {
            ell,
            order,
            dlog: dlog_tbl,
            full_r_star,
            group_amplitude,
        });
    }

    // Compute per-channel scores for each Q(x).
    let n_channels = channel_infos.len();
    let mut per_channel_scores: Vec<Vec<f64>> = vec![Vec::with_capacity(actual_count); n_channels];

    for &(_, qx) in &qs_values {
        for (ch_idx, ci) in channel_infos.iter().enumerate() {
            let qx_mod = (qx % ci.ell as u128) as u64;
            if qx_mod == 0 {
                per_channel_scores[ch_idx].push(0.0);
            } else {
                let d = ci.dlog[qx_mod as usize];
                if d == u32::MAX {
                    per_channel_scores[ch_idx].push(0.0);
                } else {
                    per_channel_scores[ch_idx].push(re_char(d, ci.full_r_star, ci.order));
                }
            }
        }
    }

    // Compute per-channel Pearson correlations with smoothness.
    let smooth_f64: Vec<f64> = smooth_flags
        .iter()
        .map(|&s| if s { 1.0 } else { 0.0 })
        .collect();

    let mut channel_weights: Vec<(usize, f64)> = Vec::with_capacity(n_channels);
    for ch_idx in 0..n_channels {
        let corr = algebra::pearson_corr(&per_channel_scores[ch_idx], &smooth_f64);
        channel_weights.push((ch_idx, corr));
    }

    // Joint score: weighted average (weight = group_amplitude).
    let total_weight: f64 = channel_infos.iter().map(|ci| ci.group_amplitude).sum();
    let joint_scores: Vec<f64> = (0..actual_count)
        .map(|i| {
            let weighted_sum: f64 = channel_infos
                .iter()
                .enumerate()
                .map(|(ch_idx, ci)| ci.group_amplitude * per_channel_scores[ch_idx][i])
                .sum();
            if total_weight > 1e-15 {
                weighted_sum / total_weight
            } else {
                0.0
            }
        })
        .collect();

    let joint_pearson_corr = algebra::pearson_corr(&joint_scores, &smooth_f64);

    // Sort by joint score descending, measure enrichment.
    let mut sorted_idx: Vec<usize> = (0..actual_count).collect();
    sorted_idx.sort_by(|&a, &b| {
        joint_scores[b]
            .partial_cmp(&joint_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let q_size = actual_count / 4;
    let d_size = actual_count / 10;

    let top_q_smooth = sorted_idx[..q_size]
        .iter()
        .filter(|&&idx| smooth_flags[idx])
        .count();
    let joint_enrichment_top_q = if overall_smooth_rate > 1e-15 {
        (top_q_smooth as f64 / q_size as f64) / overall_smooth_rate
    } else {
        0.0
    };

    let top_d_smooth = sorted_idx[..d_size]
        .iter()
        .filter(|&&idx| smooth_flags[idx])
        .count();
    let joint_enrichment_top_d = if overall_smooth_rate > 1e-15 {
        (top_d_smooth as f64 / d_size as f64) / overall_smooth_rate
    } else {
        0.0
    };

    // Best single-channel enrichment (from per-channel scores).
    let mut best_single = 0.0f64;
    for ch_idx in 0..n_channels {
        let mut ch_sorted: Vec<usize> = (0..actual_count).collect();
        ch_sorted.sort_by(|&a, &b| {
            per_channel_scores[ch_idx][b]
                .partial_cmp(&per_channel_scores[ch_idx][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ch_top_smooth = ch_sorted[..q_size]
            .iter()
            .filter(|&&idx| smooth_flags[idx])
            .count();
        let ch_enrichment = if overall_smooth_rate > 1e-15 {
            (ch_top_smooth as f64 / q_size as f64) / overall_smooth_rate
        } else {
            0.0
        };
        if ch_enrichment > best_single {
            best_single = ch_enrichment;
        }
    }

    JointScoringResult {
        n_bits,
        smooth_bound,
        n_values: actual_count,
        overall_smooth_rate,
        best_single_enrichment: best_single,
        joint_enrichment_top_q,
        joint_enrichment_top_d,
        joint_pearson_corr,
        n_channels_used: n_channels,
        channel_weights,
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Direct sieve speedup measurement
// ---------------------------------------------------------------------------

/// Direct sieve speedup measurement result.
#[derive(Debug, Clone, Serialize)]
pub struct SieveSpeedupResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    /// Number of smooth Q(x) values needed (target for sieve).
    pub target_smooth: usize,
    /// Number of Q(x) tested to find `target_smooth` smooth values (random scan).
    pub random_tested: usize,
    /// Number of Q(x) tested to find `target_smooth` smooth values (scored scan).
    pub scored_tested: usize,
    /// Speedup factor: random_tested / scored_tested.
    pub speedup_factor: f64,
    /// Random smooth rate.
    pub random_smooth_rate: f64,
    /// Scored smooth rate.
    pub scored_smooth_rate: f64,
}

/// Measure direct sieve speedup: how many Q(x) must be tested to find
/// `target_smooth` B-smooth values, comparing random scan vs scored scan.
///
/// Random scan: test Q(x) for x = 1, 2, 3, ...
/// Scored scan: sort Q(x) by joint character score, test in score-descending order.
fn measure_sieve_speedup(
    n_bits: u32,
    smooth_bound: u64,
    n_pool: usize,
    target_smooth: usize,
    seed: u64,
) -> SieveSpeedupResult {
    use algebra::{build_dlog_table, primitive_root, re_char};

    let n = generate_semiprime(n_bits, seed);
    let qs_values = generate_qs_values(n, n_pool);
    let actual_pool = qs_values.len();

    // Build per-channel scoring infrastructure.
    struct ChInfo {
        ell: u64,
        order: usize,
        dlog: Vec<u32>,
        full_r_star: usize,
        group_amplitude: f64,
    }

    let mut ch_infos: Vec<ChInfo> = Vec::with_capacity(7);
    for ch in eisenstein_hunt::CHANNELS {
        let ell = ch.ell;
        let order = (ell - 1) as usize;
        let prim_g = primitive_root(ell);
        let dlog_tbl = build_dlog_table(ell, prim_g);
        let spectrum = smoothness_spectrum_full(ell, smooth_bound);
        let (r_star, amp) = spectrum
            .coefficients
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .map(|&(r, _, _, a)| (r, a))
            .unwrap_or((1, 0.0));
        ch_infos.push(ChInfo {
            ell,
            order,
            dlog: dlog_tbl,
            full_r_star: r_star,
            group_amplitude: amp,
        });
    }

    let total_weight: f64 = ch_infos.iter().map(|ci| ci.group_amplitude).sum();

    // Score each Q(x).
    let joint_scores: Vec<f64> = qs_values
        .iter()
        .map(|&(_, qx)| {
            let weighted_sum: f64 = ch_infos
                .iter()
                .map(|ci| {
                    let qx_mod = (qx % ci.ell as u128) as u64;
                    if qx_mod == 0 {
                        0.0
                    } else {
                        let d = ci.dlog[qx_mod as usize];
                        if d == u32::MAX {
                            0.0
                        } else {
                            ci.group_amplitude * re_char(d, ci.full_r_star, ci.order)
                        }
                    }
                })
                .sum();
            if total_weight > 1e-15 {
                weighted_sum / total_weight
            } else {
                0.0
            }
        })
        .collect();

    // Random scan: test in order x = 1, 2, 3, ...
    let mut random_tested = 0usize;
    let mut random_found = 0usize;
    for &(_, qx) in &qs_values {
        random_tested += 1;
        if is_b_smooth_u128(qx, smooth_bound) {
            random_found += 1;
            if random_found >= target_smooth {
                break;
            }
        }
    }

    // Scored scan: sort by joint score descending, test in that order.
    let mut scored_order: Vec<usize> = (0..actual_pool).collect();
    scored_order.sort_by(|&a, &b| {
        joint_scores[b]
            .partial_cmp(&joint_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut scored_tested = 0usize;
    let mut scored_found = 0usize;
    for &idx in &scored_order {
        scored_tested += 1;
        if is_b_smooth_u128(qs_values[idx].1, smooth_bound) {
            scored_found += 1;
            if scored_found >= target_smooth {
                break;
            }
        }
    }

    let speedup_factor = if scored_tested > 0 {
        random_tested as f64 / scored_tested as f64
    } else {
        0.0
    };

    SieveSpeedupResult {
        n_bits,
        smooth_bound,
        target_smooth,
        random_tested,
        scored_tested,
        speedup_factor,
        random_smooth_rate: if random_tested > 0 {
            random_found as f64 / random_tested as f64
        } else {
            0.0
        },
        scored_smooth_rate: if scored_tested > 0 {
            scored_found as f64 / scored_tested as f64
        } else {
            0.0
        },
    }
}

// ---------------------------------------------------------------------------
// Top-level E22 result types
// ---------------------------------------------------------------------------

/// Full E22 experiment result.
#[derive(Debug, Clone, Serialize)]
pub struct SieveEnrichmentResult {
    /// Phase 1: Group-level enrichment profiles.
    pub group_profiles: Vec<GroupEnrichmentResult>,
    /// Phase 2: QS polynomial enrichment per block.
    pub qs_blocks: Vec<QSBlockResult>,
    /// Phase 3: Joint multi-channel scoring per block.
    pub joint_scoring: Vec<JointScoringResult>,
    /// Phase 4: Direct sieve speedup measurements.
    pub sieve_speedup: Vec<SieveSpeedupResult>,
}

/// Run the full E22 experiment with checkpointing.
///
/// Writes intermediate results to `checkpoint_path` after each phase completes.
pub fn run_sieve_enrichment(
    bit_sizes: &[u32],
    smooth_bounds: &[u64],
    n_qs_values: usize,
    n_pool: usize,
    target_smooth: usize,
    n_bins: usize,
    seed: u64,
    checkpoint_path: Option<&str>,
) -> SieveEnrichmentResult {
    // Phase 1: Group-level enrichment profiles.
    eprintln!("\n=== E22 Phase 1: Group-level enrichment profiles ===\n");

    let mut group_profiles = Vec::new();
    for &bound in smooth_bounds {
        for ch in eisenstein_hunt::CHANNELS {
            eprint!(
                "  ℓ={:6}, k={:2}, B={:3} … ",
                ch.ell, ch.weight, bound
            );
            let result = compute_group_enrichment(ch.ell, ch.weight, bound, n_bins);
            eprintln!(
                "smooth={:.4}, enrich_q4={:.2}×, enrich_d10={:.2}×",
                result.smooth_fraction,
                result.enrichment_top_quartile,
                result.enrichment_top_decile,
            );
            group_profiles.push(result);
        }
    }

    // Checkpoint after phase 1.
    if let Some(path) = checkpoint_path {
        let partial = SieveEnrichmentResult {
            group_profiles: group_profiles.clone(),
            qs_blocks: vec![],
            joint_scoring: vec![],
            sieve_speedup: vec![],
        };
        write_checkpoint(path, &partial);
        eprintln!("  [checkpoint: Phase 1 written to {path}]");
    }

    // Phase 2: QS polynomial enrichment.
    eprintln!("\n=== E22 Phase 2: QS polynomial enrichment ===\n");

    let mut qs_blocks = Vec::new();
    for &n_bits in bit_sizes {
        for &bound in smooth_bounds {
            let block_seed = seed ^ (0xE22_0002u64 + n_bits as u64 * 1000 + bound);
            eprint!(
                "  n_bits={:2}, B={:3}, {} Q(x) values … ",
                n_bits, bound, n_qs_values
            );
            let result = compute_qs_block(n_bits, bound, n_qs_values, block_seed);
            eprintln!(
                "smooth_rate={:.6}, best_enrich={:.3}×",
                result.overall_smooth_rate,
                result.best_single_enrichment,
            );
            qs_blocks.push(result);
        }
    }

    // Checkpoint after phase 2.
    if let Some(path) = checkpoint_path {
        let partial = SieveEnrichmentResult {
            group_profiles: group_profiles.clone(),
            qs_blocks: qs_blocks.clone(),
            joint_scoring: vec![],
            sieve_speedup: vec![],
        };
        write_checkpoint(path, &partial);
        eprintln!("  [checkpoint: Phase 2 written to {path}]");
    }

    // Phase 3: Joint multi-channel scoring.
    eprintln!("\n=== E22 Phase 3: Joint multi-channel scoring ===\n");

    let mut joint_scoring = Vec::new();
    for &n_bits in bit_sizes {
        for &bound in smooth_bounds {
            let block_seed = seed ^ (0xE22_0003u64 + n_bits as u64 * 1000 + bound);
            eprint!(
                "  n_bits={:2}, B={:3} … ",
                n_bits, bound
            );
            let result = compute_joint_scoring(n_bits, bound, n_qs_values, block_seed);
            eprintln!(
                "joint_enrich_q4={:.3}×, joint_corr={:.6}, best_single={:.3}×",
                result.joint_enrichment_top_q,
                result.joint_pearson_corr,
                result.best_single_enrichment,
            );
            joint_scoring.push(result);
        }
    }

    // Checkpoint after phase 3.
    if let Some(path) = checkpoint_path {
        let partial = SieveEnrichmentResult {
            group_profiles: group_profiles.clone(),
            qs_blocks: qs_blocks.clone(),
            joint_scoring: joint_scoring.clone(),
            sieve_speedup: vec![],
        };
        write_checkpoint(path, &partial);
        eprintln!("  [checkpoint: Phase 3 written to {path}]");
    }

    // Phase 4: Direct sieve speedup.
    eprintln!("\n=== E22 Phase 4: Direct sieve speedup ===\n");

    let mut sieve_speedup = Vec::new();
    for &n_bits in bit_sizes {
        for &bound in smooth_bounds {
            let block_seed = seed ^ (0xE22_0004u64 + n_bits as u64 * 1000 + bound);
            eprint!(
                "  n_bits={:2}, B={:3}, pool={}, target={} … ",
                n_bits, bound, n_pool, target_smooth
            );
            let result = measure_sieve_speedup(n_bits, bound, n_pool, target_smooth, block_seed);
            eprintln!(
                "speedup={:.3}×, random={}/{}, scored={}/{}",
                result.speedup_factor,
                result.random_tested,
                target_smooth,
                result.scored_tested,
                target_smooth,
            );
            sieve_speedup.push(result);
        }
    }

    let full = SieveEnrichmentResult {
        group_profiles,
        qs_blocks,
        joint_scoring,
        sieve_speedup,
    };

    // Final checkpoint.
    if let Some(path) = checkpoint_path {
        write_checkpoint(path, &full);
        eprintln!("  [checkpoint: FINAL written to {path}]");
    }

    full
}

/// Write a checkpoint JSON file.
fn write_checkpoint(path: &str, result: &SieveEnrichmentResult) {
    if let Ok(json) = serde_json::to_string_pretty(result) {
        let _ = std::fs::write(path, json);
    }
}

// ===========================================================================
// E23: Local Smoothness Dependence in QS Polynomial Neighborhoods
// ===========================================================================
//
// Tests whether smoothness of QS polynomial values Q(x) = (x + ⌊√N⌋)² − N
// has local structure beyond what the standard sieve already captures.
//
// Phase 1: Binary smoothness autocorrelation C(δ)
// Phase 2: Continuous partial-smooth-fraction Pearson autocorrelation
// Phase 3: Cofactor decomposition — does the sieve capture all local structure?
// Phase 4: Matched-size random control

/// One lag entry for binary smoothness autocorrelation (Phase 1).
#[derive(Debug, Clone, Serialize)]
pub struct AutocorrLag {
    pub delta: usize,
    /// Number of positions where both Q(x) and Q(x+δ) are B-smooth.
    pub n_both_smooth: usize,
    /// Number of positions where Q(x) is B-smooth (denominator).
    pub n_base_smooth: usize,
    /// C(δ) = P(Q(x+δ) smooth | Q(x) smooth) / P(smooth overall).
    /// Values > 1 indicate positive local correlation.
    pub c_delta: f64,
}

/// Phase 1 result: binary smoothness autocorrelation across lags.
#[derive(Debug, Clone, Serialize)]
pub struct SmoothnessAutocorrResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_pool: usize,
    pub overall_smooth_rate: f64,
    pub lags: Vec<AutocorrLag>,
}

/// Phase 2 result: partial-fraction Pearson autocorrelation across lags.
#[derive(Debug, Clone, Serialize)]
pub struct PartialFracAutocorrResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_pool: usize,
    pub mean_partial_frac: f64,
    /// (lag, pearson_r) pairs.
    pub lag_correlations: Vec<(usize, f64)>,
}

/// Phase 3 result: cofactor decomposition separating sieve-known from unknown structure.
#[derive(Debug, Clone, Serialize)]
pub struct SievePredictedResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_pool: usize,
    /// (delta, partial_frac_corr, cofactor_corr, residual_ratio) per lag.
    /// residual_ratio = cofactor_corr / partial_frac_corr — near 0 means sieve captures all.
    pub lag_comparisons: Vec<(usize, f64, f64, f64)>,
    pub n_sieve_primes: usize,
}

/// Phase 4 result: random control (matched-size random integers).
#[derive(Debug, Clone, Serialize)]
pub struct RandomControlResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_pool: usize,
    pub overall_smooth_rate: f64,
    pub lags: Vec<AutocorrLag>,
    pub lag_correlations: Vec<(usize, f64)>,
}

/// Combined result for one (n_bits, smooth_bound) block.
#[derive(Debug, Clone, Serialize)]
pub struct LocalBlockResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub seed: u64,
    pub phase1: SmoothnessAutocorrResult,
    pub phase2: PartialFracAutocorrResult,
    pub phase3: SievePredictedResult,
    pub phase4: RandomControlResult,
}

/// Top-level E23 result aggregating all blocks.
#[derive(Debug, Clone, Serialize)]
pub struct LocalSmoothnessResult {
    pub blocks: Vec<LocalBlockResult>,
}

/// Overflow-safe modular multiplication via u128.
#[inline]
fn mul_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: base^exp mod modulus, using u128 intermediates.
fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod_u64(result, base, modulus);
        }
        exp >>= 1;
        base = mul_mod_u64(base, base, modulus);
    }
    result
}

/// Tonelli-Shanks algorithm: compute x such that x² ≡ n (mod p).
/// Returns None if n is not a quadratic residue mod p. Requires p to be prime.
fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if p == 2 {
        return Some(n % 2);
    }
    let n = n % p;
    if n == 0 {
        return Some(0);
    }

    // Euler's criterion: n^((p-1)/2) ≡ 1 (mod p) iff n is a QR
    if mod_pow_u64(n, (p - 1) / 2, p) != 1 {
        return None;
    }

    // Factor out powers of 2: p - 1 = q · 2^s with q odd
    let mut q = p - 1;
    let mut s: u32 = 0;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    if s == 1 {
        // p ≡ 3 (mod 4) — simple case
        return Some(mod_pow_u64(n, (p + 1) / 4, p));
    }

    // Find a quadratic non-residue z
    let mut z = 2u64;
    while mod_pow_u64(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
        if z >= p {
            return None;
        }
    }

    let mut m = s;
    let mut c = mod_pow_u64(z, q, p);
    let mut t = mod_pow_u64(n, q, p);
    let mut r = mod_pow_u64(n, (q + 1) / 2, p);

    loop {
        if t == 0 {
            return Some(0);
        }
        if t == 1 {
            return Some(r);
        }

        // Find least i such that t^(2^i) ≡ 1 (mod p)
        let mut i: u32 = 1;
        let mut temp = mul_mod_u64(t, t, p);
        while temp != 1 {
            temp = mul_mod_u64(temp, temp, p);
            i += 1;
            if i >= m {
                return None;
            }
        }

        let b = mod_pow_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod_u64(b, b, p);
        t = mul_mod_u64(t, c, p);
        r = mul_mod_u64(r, b, p);
    }
}

/// Find roots of Q(x) ≡ 0 (mod p) for QS polynomial Q(x) = (x + s)² − N.
///
/// Q(x) ≡ 0 (mod p) means (x + s)² ≡ N (mod p), so x ≡ ±√(N mod p) − s (mod p).
/// Returns 0 roots if N is a NQR mod p, 1 if the two roots coincide, 2 otherwise.
fn qs_roots_mod_p(n: u128, p: u64) -> Vec<u64> {
    let n_mod_p = (n % p as u128) as u64;
    let s = (isqrt(n) % p as u128) as u64;

    match tonelli_shanks(n_mod_p, p) {
        None => vec![],
        Some(r) => {
            // x ≡ r - s (mod p) and x ≡ -r - s ≡ p - r - s (mod p)
            let root1 = (r + p - s % p) % p;
            let root2 = (p - r + p - s % p) % p;
            if root1 == root2 {
                vec![root1]
            } else {
                vec![root1, root2]
            }
        }
    }
}

/// Compute the log-fraction of n explained by primes ≤ bound.
///
/// Returns sum(v_p(n) · ln(p)) / ln(n) for all primes p ≤ bound dividing n.
/// Result is in [0, 1]: 1.0 means fully B-smooth, 0.0 means no small-prime factors.
fn partial_smooth_fraction(mut n: u128, bound: u64) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let total_log = (n as f64).ln();
    let mut smooth_log = 0.0f64;

    let mut d = 2u64;
    while (d as u128) * (d as u128) <= n && d <= bound {
        if n % (d as u128) == 0 {
            let d_log = (d as f64).ln();
            while n % (d as u128) == 0 {
                smooth_log += d_log;
                n /= d as u128;
            }
        }
        d += if d == 2 { 1 } else { 2 };
    }
    // If n > 1 and n <= bound, it's a prime factor within the bound.
    if n > 1 && n <= bound as u128 {
        smooth_log += (n as f64).ln();
    }

    smooth_log / total_log
}

/// Compute log₂ of the cofactor of n after dividing out all primes ≤ bound.
///
/// cofactor = n / (B-smooth part of n). Returns log₂(cofactor).
/// 0.0 means n is fully B-smooth.
fn cofactor_log(mut n: u128, bound: u64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut d = 2u64;
    while (d as u128) * (d as u128) <= n && d <= bound {
        while n % (d as u128) == 0 {
            n /= d as u128;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    // If remaining n <= bound, it's a small prime factor.
    if n > 1 && n <= bound as u128 {
        n = 1;
    }
    if n <= 1 {
        0.0
    } else {
        (n as f64).log2()
    }
}

// ---------------------------------------------------------------------------
// E23 Phase 1: Binary smoothness autocorrelation
// ---------------------------------------------------------------------------

/// Compute binary smoothness autocorrelation: C(δ) = P(smooth(x+δ) | smooth(x)) / P(smooth).
///
/// For each lag δ in 1..=max_lag, counts how often Q(x+δ) is smooth given Q(x) is smooth,
/// normalized by the overall smooth rate. C(δ) > 1 means positive local correlation.
pub fn compute_smoothness_autocorrelation(
    qs_values: &[u128],
    smooth_bound: u64,
    max_lag: usize,
    n_bits: u32,
) -> SmoothnessAutocorrResult {
    let n_pool = qs_values.len();

    // Precompute smooth flags.
    let smooth_flags: Vec<bool> = qs_values
        .iter()
        .map(|&v| is_b_smooth_u128(v, smooth_bound))
        .collect();

    let n_smooth: usize = smooth_flags.iter().filter(|&&s| s).count();
    let overall_smooth_rate = n_smooth as f64 / n_pool as f64;

    let mut lags = Vec::with_capacity(max_lag);

    for delta in 1..=max_lag {
        let mut n_base_smooth = 0usize;
        let mut n_both_smooth = 0usize;

        for i in 0..(n_pool - delta) {
            if smooth_flags[i] {
                n_base_smooth += 1;
                if smooth_flags[i + delta] {
                    n_both_smooth += 1;
                }
            }
        }

        let conditional_rate = if n_base_smooth > 0 {
            n_both_smooth as f64 / n_base_smooth as f64
        } else {
            0.0
        };

        let c_delta = if overall_smooth_rate > 0.0 {
            conditional_rate / overall_smooth_rate
        } else {
            0.0
        };

        lags.push(AutocorrLag {
            delta,
            n_both_smooth,
            n_base_smooth,
            c_delta,
        });
    }

    SmoothnessAutocorrResult {
        n_bits,
        smooth_bound,
        n_pool,
        overall_smooth_rate,
        lags,
    }
}

// ---------------------------------------------------------------------------
// E23 Phase 2: Partial-fraction Pearson autocorrelation
// ---------------------------------------------------------------------------

/// Compute Pearson autocorrelation of partial smooth fractions at each lag.
///
/// Uses the continuous metric partial_smooth_fraction(Q(x), B) for higher
/// statistical power than binary smoothness.
pub fn compute_partial_frac_autocorrelation(
    qs_values: &[u128],
    smooth_bound: u64,
    max_lag: usize,
    n_bits: u32,
) -> PartialFracAutocorrResult {
    let n_pool = qs_values.len();

    // Precompute partial fractions.
    let partial_fracs: Vec<f64> = qs_values
        .iter()
        .map(|&v| partial_smooth_fraction(v, smooth_bound))
        .collect();

    let mean_partial_frac = partial_fracs.iter().sum::<f64>() / n_pool as f64;

    let mut lag_correlations = Vec::with_capacity(max_lag);

    for delta in 1..=max_lag {
        let len = n_pool - delta;
        let x: Vec<f64> = partial_fracs[..len].to_vec();
        let y: Vec<f64> = partial_fracs[delta..delta + len].to_vec();
        let r = algebra::pearson_corr(&x, &y);
        lag_correlations.push((delta, r));
    }

    PartialFracAutocorrResult {
        n_bits,
        smooth_bound,
        n_pool,
        mean_partial_frac,
        lag_correlations,
    }
}

// ---------------------------------------------------------------------------
// E23 Phase 3: Cofactor decomposition (sieve-predicted null)
// ---------------------------------------------------------------------------

/// Decompose local structure into sieve-explained and cofactor components.
///
/// partial_frac captures what the sieve knows (small-prime divisibility).
/// cofactor_log captures what the sieve doesn't know.
/// If cofactor autocorrelation ≈ 0, the sieve captures all local structure.
pub fn compute_cofactor_decomposition(
    qs_values: &[u128],
    smooth_bound: u64,
    max_lag: usize,
    n_bits: u32,
) -> SievePredictedResult {
    let n_pool = qs_values.len();

    // Precompute both metrics.
    let partial_fracs: Vec<f64> = qs_values
        .iter()
        .map(|&v| partial_smooth_fraction(v, smooth_bound))
        .collect();
    let cofactor_logs: Vec<f64> = qs_values
        .iter()
        .map(|&v| cofactor_log(v, smooth_bound))
        .collect();

    // Count sieve primes.
    let n_sieve_primes = {
        let mut count = 0;
        let mut d = 2u64;
        while d <= smooth_bound {
            count += 1;
            d += if d == 2 { 1 } else { 2 };
        }
        count
    };

    let mut lag_comparisons = Vec::with_capacity(max_lag);

    for delta in 1..=max_lag {
        let len = n_pool - delta;
        let pf_x: Vec<f64> = partial_fracs[..len].to_vec();
        let pf_y: Vec<f64> = partial_fracs[delta..delta + len].to_vec();
        let pf_corr = algebra::pearson_corr(&pf_x, &pf_y);

        let cf_x: Vec<f64> = cofactor_logs[..len].to_vec();
        let cf_y: Vec<f64> = cofactor_logs[delta..delta + len].to_vec();
        let cf_corr = algebra::pearson_corr(&cf_x, &cf_y);

        let residual_ratio = if pf_corr.abs() > 1e-12 {
            cf_corr / pf_corr
        } else {
            0.0
        };

        lag_comparisons.push((delta, pf_corr, cf_corr, residual_ratio));
    }

    SievePredictedResult {
        n_bits,
        smooth_bound,
        n_pool,
        lag_comparisons,
        n_sieve_primes,
    }
}

// ---------------------------------------------------------------------------
// E23 Phase 4: Random control
// ---------------------------------------------------------------------------

/// Random control: compute autocorrelation on matched-size random integers.
///
/// Generates random u128 values in the same range as the QS polynomial,
/// expects C(δ) ≈ 1.0 and partial-frac correlations ≈ 0.
pub fn compute_random_control(
    n: u128,
    smooth_bound: u64,
    n_pool: usize,
    max_lag: usize,
    seed: u64,
    n_bits: u32,
) -> RandomControlResult {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let s = isqrt(n);
    let q_min = {
        let v = (1u128 + s) * (1u128 + s);
        v.saturating_sub(n).max(1)
    };
    let q_max = {
        let v = (n_pool as u128 + s) * (n_pool as u128 + s);
        v.saturating_sub(n).max(q_min + 1)
    };
    let range = q_max - q_min;

    // Simple deterministic PRNG seeded from seed.
    let mut state = seed;
    let mut rng_next = move || -> u128 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        let h1 = hasher.finish();
        state = h1;
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        let h2 = hasher2.finish();
        let val = ((h1 as u128) << 64) | (h2 as u128);
        q_min + (val % range)
    };

    let random_values: Vec<u128> = (0..n_pool).map(|_| rng_next()).collect();

    let smooth_flags: Vec<bool> = random_values
        .iter()
        .map(|&v| is_b_smooth_u128(v, smooth_bound))
        .collect();
    let n_smooth = smooth_flags.iter().filter(|&&s| s).count();
    let overall_smooth_rate = n_smooth as f64 / n_pool as f64;

    let partial_fracs: Vec<f64> = random_values
        .iter()
        .map(|&v| partial_smooth_fraction(v, smooth_bound))
        .collect();

    let mut lags = Vec::with_capacity(max_lag);
    let mut lag_correlations = Vec::with_capacity(max_lag);

    for delta in 1..=max_lag {
        // Binary autocorrelation.
        let mut n_base_smooth = 0usize;
        let mut n_both_smooth = 0usize;
        for i in 0..(n_pool - delta) {
            if smooth_flags[i] {
                n_base_smooth += 1;
                if smooth_flags[i + delta] {
                    n_both_smooth += 1;
                }
            }
        }
        let conditional_rate = if n_base_smooth > 0 {
            n_both_smooth as f64 / n_base_smooth as f64
        } else {
            0.0
        };
        let c_delta = if overall_smooth_rate > 0.0 {
            conditional_rate / overall_smooth_rate
        } else {
            0.0
        };
        lags.push(AutocorrLag {
            delta,
            n_both_smooth,
            n_base_smooth,
            c_delta,
        });

        // Partial-frac Pearson.
        let len = n_pool - delta;
        let x: Vec<f64> = partial_fracs[..len].to_vec();
        let y: Vec<f64> = partial_fracs[delta..delta + len].to_vec();
        let r = algebra::pearson_corr(&x, &y);
        lag_correlations.push((delta, r));
    }

    RandomControlResult {
        n_bits,
        smooth_bound,
        n_pool,
        overall_smooth_rate,
        lags,
        lag_correlations,
    }
}

// ---------------------------------------------------------------------------
// E23 Top-level runner
// ---------------------------------------------------------------------------

/// Compute all 4 phases for a single (n_bits, smooth_bound) block.
///
/// Generates QS values once and precomputes all shared data.
pub fn compute_local_block(
    n_bits: u32,
    smooth_bound: u64,
    n_pool: usize,
    max_lag: usize,
    seed: u64,
) -> LocalBlockResult {
    let n = generate_semiprime(n_bits, seed);
    let qs_pairs = generate_qs_values(n, n_pool);
    let qs_values: Vec<u128> = qs_pairs.iter().map(|&(_, v)| v).collect();

    eprintln!("    Phase 1: binary smoothness autocorrelation ...");
    let phase1 = compute_smoothness_autocorrelation(&qs_values, smooth_bound, max_lag, n_bits);
    eprintln!(
        "      smooth_rate={:.6}, C(1)={:.4}",
        phase1.overall_smooth_rate,
        phase1.lags.first().map(|l| l.c_delta).unwrap_or(0.0),
    );

    eprintln!("    Phase 2: partial-fraction Pearson autocorrelation ...");
    let phase2 = compute_partial_frac_autocorrelation(&qs_values, smooth_bound, max_lag, n_bits);
    eprintln!(
        "      mean_pf={:.4}, r(1)={:.6}",
        phase2.mean_partial_frac,
        phase2.lag_correlations.first().map(|&(_, r)| r).unwrap_or(0.0),
    );

    eprintln!("    Phase 3: cofactor decomposition ...");
    let phase3 = compute_cofactor_decomposition(&qs_values, smooth_bound, max_lag, n_bits);
    let (cf_corr_1, resid_1) = phase3
        .lag_comparisons
        .first()
        .map(|&(_, _, cf, rr)| (cf, rr))
        .unwrap_or((0.0, 0.0));
    eprintln!(
        "      cofactor_corr(1)={:.6}, residual_ratio(1)={:.4}",
        cf_corr_1, resid_1,
    );

    eprintln!("    Phase 4: random control ...");
    let phase4 = compute_random_control(n, smooth_bound, n_pool, max_lag, seed ^ 0xE23_FFFF_0000, n_bits);
    eprintln!(
        "      random_smooth_rate={:.6}, random_r(1)={:.6}",
        phase4.overall_smooth_rate,
        phase4.lag_correlations.first().map(|&(_, r)| r).unwrap_or(0.0),
    );

    LocalBlockResult {
        n_bits,
        smooth_bound,
        seed,
        phase1,
        phase2,
        phase3,
        phase4,
    }
}

fn write_local_checkpoint(path: &str, result: &LocalSmoothnessResult) {
    if let Ok(json) = serde_json::to_string_pretty(result) {
        let _ = std::fs::write(path, json);
    }
}

/// Run the full E23 local smoothness experiment across all bit sizes and bounds.
pub fn run_local_smoothness(
    bit_sizes: &[u32],
    smooth_bounds: &[u64],
    n_pool: usize,
    max_lag: usize,
    seed: u64,
    checkpoint_path: Option<&str>,
) -> LocalSmoothnessResult {
    let mut blocks = Vec::new();

    let total = bit_sizes.len() * smooth_bounds.len();
    let mut done = 0;

    for &n_bits in bit_sizes {
        for &bound in smooth_bounds {
            done += 1;
            let block_seed = seed ^ (n_bits as u64 * 10000 + bound);
            eprintln!(
                "\n  [{done}/{total}] n_bits={n_bits}, B={bound}, pool={n_pool}, max_lag={max_lag}"
            );

            let block = compute_local_block(n_bits, bound, n_pool, max_lag, block_seed);
            blocks.push(block);

            // Checkpoint after each block.
            if let Some(path) = checkpoint_path {
                let partial = LocalSmoothnessResult {
                    blocks: blocks.clone(),
                };
                write_local_checkpoint(path, &partial);
                eprintln!("    [checkpoint: {done}/{total} blocks written to {path}]");
            }
        }
    }

    LocalSmoothnessResult { blocks }
}

// ===========================================================================
// E24: NFS 2D Lattice Locality — Cofactor Autocorrelation in Algebraic Norm
//      Neighborhoods
// ===========================================================================
//
// Extends E23 from 1D QS polynomials to 2D NFS lattice norms.
// Tests whether cofactors of algebraic norms F(a,b) have 2D spatial
// autocorrelation beyond what the lattice sieve already captures.
//
// Phase 1: 2D binary smoothness autocorrelation across displacement vectors
// Phase 2: 2D continuous partial-smooth-fraction Pearson autocorrelation
// Phase 3: 2D cofactor decomposition — does the sieve capture all structure?
// Phase 4: 2D matched-size random control
// Phase 5: Joint rational+algebraic cofactor correlation (novel dual-norm test)

/// 2D displacement vector for lattice autocorrelation.
#[derive(Debug, Clone, Serialize)]
pub struct Displacement2D {
    pub da: i64,
    pub db: i64,
    pub label: String,
}

/// One displacement entry for 2D binary smoothness autocorrelation (Phase 1).
#[derive(Debug, Clone, Serialize)]
pub struct Autocorr2DLag {
    pub da: i64,
    pub db: i64,
    pub label: String,
    /// Number of pairs where both base and displaced norms are B-smooth.
    pub n_both_smooth: usize,
    /// Number of pairs where the base norm is B-smooth.
    pub n_base_smooth: usize,
    /// C(Δa,Δb) = P(neighbor smooth | base smooth) / P(smooth overall).
    pub c_delta: f64,
}

/// Phase 1 result: 2D binary smoothness autocorrelation.
#[derive(Debug, Clone, Serialize)]
pub struct NfsAutocorr2DResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub overall_smooth_rate: f64,
    pub lags: Vec<Autocorr2DLag>,
}

/// Phase 2 result: 2D partial-fraction Pearson autocorrelation.
#[derive(Debug, Clone, Serialize)]
pub struct NfsPartialFrac2DResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub mean_partial_frac: f64,
    /// (label, pearson_r) pairs per displacement.
    pub displacement_correlations: Vec<(String, f64)>,
}

/// Phase 3 result: 2D cofactor decomposition.
#[derive(Debug, Clone, Serialize)]
pub struct NfsCofactor2DResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// (label, partial_frac_corr, cofactor_corr, residual_ratio) per displacement.
    pub displacement_comparisons: Vec<(String, f64, f64, f64)>,
    pub n_sieve_primes: usize,
}

/// Phase 4 result: 2D random control.
#[derive(Debug, Clone, Serialize)]
pub struct NfsRandom2DResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub overall_smooth_rate: f64,
    pub lags: Vec<Autocorr2DLag>,
}

/// Phase 5 result: joint rational+algebraic cofactor correlation (novel).
#[derive(Debug, Clone, Serialize)]
pub struct NfsDualNormResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// Pearson correlation between cofactor_log(algebraic) and cofactor_log(rational).
    pub rational_alg_cofactor_corr: f64,
    pub rational_mean_partial: f64,
    pub algebraic_mean_partial: f64,
}

/// Combined result for one (n_bits, smooth_bound) block across all 5 phases.
#[derive(Debug, Clone, Serialize)]
pub struct NfsBlockResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub seed: u64,
    pub n_grid: usize,
    pub phase1: NfsAutocorr2DResult,
    pub phase2: NfsPartialFrac2DResult,
    pub phase3: NfsCofactor2DResult,
    pub phase4: NfsRandom2DResult,
    pub phase5: NfsDualNormResult,
}

/// Top-level E24 result aggregating all blocks.
#[derive(Debug, Clone, Serialize)]
pub struct NfsLatticeResult {
    pub blocks: Vec<NfsBlockResult>,
}

// ===========================================================================
// E24b: Artifact Validation Controls
// ===========================================================================
//
// Five controls to verify whether E24's cofactor autocorrelation is genuine
// or an artifact of norm magnitude gradients across the (a,b) grid.
//
// Control A: Norm-residualized cofactor autocorrelation
//   Regress cofactor_log against log2(alg_norm), use residuals.
//   If signal survives → not a magnitude gradient artifact.
//
// Control B: Per-side (rational vs algebraic) cofactor analysis
//   Separate analysis for each norm side.
//
// Control C: Magnitude-bin shuffle null (geometry-preserving)
//   Shuffle cofactor_logs within bins of similar log2(norm).
//   Preserves magnitude profile, destroys genuine local structure.
//
// Control D: Displacement decay analysis
//   Extended displacements at multiple radii to test if correlation
//   decays smoothly with |δ|.
//
// Control E: Conditional cofactor corr (binned by partial_smooth_fraction)
//   Among points with similar sieve scores, does cofactor corr persist?

/// Control A result: norm-residualized cofactor autocorrelation.
#[derive(Debug, Clone, Serialize)]
pub struct NormResidualizedResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// R² of cofactor_log ~ log2(norm) regression.
    pub norm_r_squared: f64,
    /// Displacement comparisons using norm-residualized cofactor_log.
    /// (label, raw_cofactor_corr, residualized_cofactor_corr)
    pub displacement_comparisons: Vec<(String, f64, f64)>,
}

/// Control B result: per-side cofactor autocorrelation.
#[derive(Debug, Clone, Serialize)]
pub struct PerSideCofactorResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// (label, alg_cf_corr, rat_cf_corr, alg_residualized, rat_residualized)
    pub displacement_comparisons: Vec<(String, f64, f64, f64, f64)>,
}

/// Control C result: magnitude-bin shuffle null.
#[derive(Debug, Clone, Serialize)]
pub struct MagnitudeBinShuffleResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub n_bins: usize,
    /// (label, original_cf_corr, shuffled_cf_corr_mean, shuffled_cf_corr_std)
    pub displacement_comparisons: Vec<(String, f64, f64, f64)>,
}

/// Control D result: displacement decay analysis.
#[derive(Debug, Clone, Serialize)]
pub struct DisplacementDecayResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// (displacement_norm, mean_cf_corr, mean_residualized_cf_corr, n_displacements)
    pub decay_by_radius: Vec<(f64, f64, f64, usize)>,
}

/// Control E result: conditional cofactor correlation.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionalCofactorResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub n_bins: usize,
    /// (pf_bin_center, n_in_bin, mean_cf_corr_within_bin) for displacement (1,0).
    pub bin_results: Vec<(f64, usize, f64)>,
    /// Overall weighted mean cofactor corr across bins.
    pub conditional_cf_corr: f64,
}

/// Combined validation result for one block.
#[derive(Debug, Clone, Serialize)]
pub struct NfsValidationBlockResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub seed: u64,
    pub control_a: NormResidualizedResult,
    pub control_b: PerSideCofactorResult,
    pub control_c: MagnitudeBinShuffleResult,
    pub control_d: DisplacementDecayResult,
    pub control_e: ConditionalCofactorResult,
}

/// Top-level E24b validation result.
#[derive(Debug, Clone, Serialize)]
pub struct NfsValidationResult {
    pub blocks: Vec<NfsValidationBlockResult>,
}

// ===========================================================================
// E24c: Robustness Check Result Structs
// ===========================================================================

/// Check 1 result: nonlinear (binned) residualization vs OLS residualization.
#[derive(Debug, Clone, Serialize)]
pub struct NonlinearResidResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub n_bins: usize,
    /// (label, raw_corr, ols_resid_corr, bin_resid_corr) per displacement.
    pub displacement_comparisons: Vec<(String, f64, f64, f64)>,
}

/// Check 2 result: cross-validated residualization.
#[derive(Debug, Clone, Serialize)]
pub struct CrossValidatedResidResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// Number of points in left half (a < 0).
    pub n_left: usize,
    /// Number of points in right half (a >= 0).
    pub n_right: usize,
    /// (label, in_sample_resid_corr, held_out_left_resid_corr, held_out_right_resid_corr)
    pub displacement_comparisons: Vec<(String, f64, f64, f64)>,
}

/// Check 3 result: partial correlation controlling both endpoint norms.
#[derive(Debug, Clone, Serialize)]
pub struct PartialCorrelationResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    /// Mean R² of 2-predictor model (cf ~ log_norm_base + log_norm_neighbor).
    pub r_squared_2d: f64,
    /// Mean R² of 1-predictor model (cf ~ log_norm_base) for comparison.
    pub r_squared_1d: f64,
    /// (label, raw_corr, ols_1d_resid_corr, partial_corr_both_norms) per displacement.
    pub displacement_comparisons: Vec<(String, f64, f64, f64)>,
}

/// One transform variant result for Check 4.
#[derive(Debug, Clone, Serialize)]
pub struct TransformVariantResult {
    pub transform_name: String,
    /// (label, raw_corr, resid_corr) per displacement.
    pub displacement_comparisons: Vec<(String, f64, f64)>,
}

/// Check 4 result: alternative transforms of cofactor_log.
#[derive(Debug, Clone, Serialize)]
pub struct AlternativeTransformResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub n_grid: usize,
    pub variants: Vec<TransformVariantResult>,
}

/// Combined E24c robustness result for one (n_bits, smooth_bound) block.
#[derive(Debug, Clone, Serialize)]
pub struct NfsRobustnessBlockResult {
    pub n_bits: u32,
    pub smooth_bound: u64,
    pub seed: u64,
    pub n_grid: usize,
    pub check1_nonlinear: NonlinearResidResult,
    pub check2_crossval: CrossValidatedResidResult,
    pub check3_partial_corr: PartialCorrelationResult,
    pub check4_transforms: AlternativeTransformResult,
}

/// Top-level E24c robustness result.
#[derive(Debug, Clone, Serialize)]
pub struct NfsRobustnessResult {
    pub blocks: Vec<NfsRobustnessBlockResult>,
}

// ---------------------------------------------------------------------------
// E24 helper functions
// ---------------------------------------------------------------------------

/// Convert a BigUint to u128, clamping to u128::MAX on overflow.
/// The log₂ of a clamped value is still meaningful for cofactor analysis.
fn biguint_to_u128_clamped(v: &BigUint) -> u128 {
    v.to_u128().unwrap_or(u128::MAX)
}

/// Compute the rational norm |a + b·m| as u128.
fn rational_norm_u128(a: i64, b: i64, m: u64) -> u128 {
    let bm = b as i128 * m as i128;
    let val = a as i128 + bm;
    val.unsigned_abs()
}

/// Standard 2D displacement vectors for lattice autocorrelation.
///
/// Covers nearest neighbors in all cardinal and diagonal directions,
/// plus a few second-order displacements.
fn standard_displacements() -> Vec<Displacement2D> {
    vec![
        Displacement2D { da: 1, db: 0, label: "(1,0)".into() },
        Displacement2D { da: 0, db: 1, label: "(0,1)".into() },
        Displacement2D { da: 1, db: 1, label: "(1,1)".into() },
        Displacement2D { da: 1, db: -1, label: "(1,-1)".into() },
        Displacement2D { da: 2, db: 0, label: "(2,0)".into() },
        Displacement2D { da: 0, db: 2, label: "(0,2)".into() },
        Displacement2D { da: 2, db: 1, label: "(2,1)".into() },
        Displacement2D { da: 1, db: 2, label: "(1,2)".into() },
    ]
}

/// Build a HashMap from (a,b) coordinates to index in the grid for O(1) neighbor lookup.
fn build_grid_index(grid: &[(i64, i64, u128, u128)]) -> HashMap<(i64, i64), usize> {
    grid.iter()
        .enumerate()
        .map(|(i, &(a, b, _, _))| ((a, b), i))
        .collect()
}

/// Generate an NFS grid of coprime (a,b) pairs with their algebraic and rational norms.
///
/// For a semiprime N of the given bit size:
/// 1. Select NFS polynomial via base-m method
/// 2. Enumerate (a,b) with a ∈ [-sieve_area, sieve_area], b ∈ [1, max_b], gcd(|a|,b)=1
/// 3. Compute algebraic norm F(a,b) and rational norm |a + b·m|
///
/// Returns the polynomial and grid as Vec<(a, b, alg_norm, rat_norm)>.
fn generate_nfs_grid(
    n_bits: u32,
    seed: u64,
    sieve_area: i64,
    max_b: i64,
) -> (NfsPolynomial, Vec<(i64, i64, u128, u128)>) {
    let n_u128 = generate_semiprime(n_bits, seed);
    let n_big = BigUint::from(n_u128);

    // Choose polynomial degree based on bit size
    let degree = if n_bits <= 90 { 3 } else if n_bits <= 120 { 4 } else { 5 };
    let poly = select_polynomial(&n_big, degree);
    let m_u64 = poly.m.to_u64().expect("m must fit in u64 for our bit sizes");

    let mut grid = Vec::new();
    for b in 1..=max_b {
        let b_u64 = b as u64;
        for a in -sieve_area..=sieve_area {
            if a == 0 {
                continue;
            }
            // coprimality requirement
            if gcd_u64(a.unsigned_abs(), b_u64) != 1 {
                continue;
            }
            let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);
            let alg_norm = biguint_to_u128_clamped(&alg_norm_big);
            let rat_norm = rational_norm_u128(a, b, m_u64);

            // skip trivial norms
            if alg_norm <= 1 || rat_norm <= 1 {
                continue;
            }
            grid.push((a, b, alg_norm, rat_norm));
        }
    }

    (poly, grid)
}

// ---------------------------------------------------------------------------
// E24 Phase 1: 2D binary smoothness autocorrelation
// ---------------------------------------------------------------------------

/// Compute 2D binary smoothness autocorrelation for NFS algebraic norms.
///
/// For each displacement (Δa, Δb):
///   C(Δa,Δb) = P(F(a+Δa, b+Δb) smooth | F(a,b) smooth) / P(smooth overall)
///
/// Values > 1 indicate positive local correlation.
fn compute_nfs_2d_autocorrelation(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    smooth_flags: &[bool],
    smooth_bound: u64,
    n_bits: u32,
) -> NfsAutocorr2DResult {
    let n_grid = grid.len();
    let n_smooth: usize = smooth_flags.iter().filter(|&&s| s).count();
    let overall_smooth_rate = n_smooth as f64 / n_grid as f64;

    let displacements = standard_displacements();
    let mut lags = Vec::new();

    for disp in &displacements {
        let mut n_base_smooth = 0usize;
        let mut n_both_smooth = 0usize;

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            if !smooth_flags[i] {
                continue;
            }
            n_base_smooth += 1;

            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                if smooth_flags[j] {
                    n_both_smooth += 1;
                }
            }
        }

        let c_delta = if n_base_smooth > 0 && overall_smooth_rate > 0.0 {
            let cond_prob = n_both_smooth as f64 / n_base_smooth as f64;
            cond_prob / overall_smooth_rate
        } else {
            0.0
        };

        lags.push(Autocorr2DLag {
            da: disp.da,
            db: disp.db,
            label: disp.label.clone(),
            n_both_smooth,
            n_base_smooth,
            c_delta,
        });
    }

    NfsAutocorr2DResult {
        n_bits,
        smooth_bound,
        n_grid,
        overall_smooth_rate,
        lags,
    }
}

// ---------------------------------------------------------------------------
// E24 Phase 2: 2D partial-fraction Pearson autocorrelation
// ---------------------------------------------------------------------------

/// Compute 2D partial-fraction Pearson autocorrelation for NFS algebraic norms.
///
/// For each displacement, collects paired (partial_frac[base], partial_frac[displaced])
/// vectors and computes their Pearson correlation.
fn compute_nfs_2d_partial_frac(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    partial_fracs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> NfsPartialFrac2DResult {
    let n_grid = grid.len();
    let mean_partial_frac = if n_grid > 0 {
        partial_fracs.iter().sum::<f64>() / n_grid as f64
    } else {
        0.0
    };

    let displacements = standard_displacements();
    let mut displacement_correlations = Vec::new();

    for disp in &displacements {
        let mut base_vals = Vec::new();
        let mut disp_vals = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_vals.push(partial_fracs[i]);
                disp_vals.push(partial_fracs[j]);
            }
        }

        let r = if base_vals.len() >= 3 {
            algebra::pearson_corr(&base_vals, &disp_vals)
        } else {
            0.0
        };

        displacement_correlations.push((disp.label.clone(), r));
    }

    NfsPartialFrac2DResult {
        n_bits,
        smooth_bound,
        n_grid,
        mean_partial_frac,
        displacement_correlations,
    }
}

// ---------------------------------------------------------------------------
// E24 Phase 3: 2D cofactor decomposition (sieve-predicted null)
// ---------------------------------------------------------------------------

/// Compute 2D cofactor decomposition for NFS algebraic norms.
///
/// For each displacement:
/// - partial_frac_corr = Pearson autocorrelation of partial_smooth_fraction (sieve-explained)
/// - cofactor_corr = Pearson autocorrelation of cofactor_log (beyond-sieve)
/// - residual_ratio = |cofactor_corr| / max(|partial_frac_corr|, ε)
///
/// If cofactor_corr ≈ 0, the NFS sieve captures all local structure.
fn compute_nfs_2d_cofactor_decomposition(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    partial_fracs: &[f64],
    cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> NfsCofactor2DResult {
    let n_grid = grid.len();
    let primes = sieve_primes(smooth_bound);
    let n_sieve_primes = primes.len();

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();

    for disp in &displacements {
        let mut base_pf = Vec::new();
        let mut disp_pf = Vec::new();
        let mut base_cf = Vec::new();
        let mut disp_cf = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_pf.push(partial_fracs[i]);
                disp_pf.push(partial_fracs[j]);
                base_cf.push(cofactor_logs[i]);
                disp_cf.push(cofactor_logs[j]);
            }
        }

        let (pf_corr, cf_corr) = if base_pf.len() >= 3 {
            (
                algebra::pearson_corr(&base_pf, &disp_pf),
                algebra::pearson_corr(&base_cf, &disp_cf),
            )
        } else {
            (0.0, 0.0)
        };

        let residual_ratio = cf_corr.abs() / pf_corr.abs().max(1e-12);

        displacement_comparisons.push((disp.label.clone(), pf_corr, cf_corr, residual_ratio));
    }

    NfsCofactor2DResult {
        n_bits,
        smooth_bound,
        n_grid,
        displacement_comparisons,
        n_sieve_primes,
    }
}

// ---------------------------------------------------------------------------
// E24 Phase 4: 2D random control
// ---------------------------------------------------------------------------

/// Compute 2D random control: random integers in the same magnitude range.
///
/// Generates random u128 values at the same (a,b) grid positions.
/// Expected: C(Δa,Δb) ≈ 1.0 (no spatial structure).
fn compute_nfs_2d_random_control(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    smooth_bound: u64,
    n_bits: u32,
    seed: u64,
) -> NfsRandom2DResult {
    let n_grid = grid.len();

    // Determine magnitude range from actual algebraic norms
    let max_norm = grid.iter().map(|&(_, _, a, _)| a).max().unwrap_or(1);
    let min_norm = grid.iter().map(|&(_, _, a, _)| a).filter(|&v| v > 1).min().unwrap_or(1);

    // Generate random values in [min_norm, max_norm]
    let mut rng = StdRng::seed_from_u64(seed);
    let range = if max_norm > min_norm { max_norm - min_norm } else { 1 };
    let random_vals: Vec<u128> = (0..n_grid)
        .map(|_| {
            let r: u64 = rng.gen();
            min_norm + (r as u128 % range).max(2)
        })
        .collect();

    // Compute smooth flags for random values
    let random_smooth: Vec<bool> = random_vals.iter()
        .map(|&v| is_b_smooth_u128(v, smooth_bound))
        .collect();

    let n_smooth: usize = random_smooth.iter().filter(|&&s| s).count();
    let overall_smooth_rate = n_smooth as f64 / n_grid as f64;

    let displacements = standard_displacements();
    let mut lags = Vec::new();

    for disp in &displacements {
        let mut n_base_smooth = 0usize;
        let mut n_both_smooth = 0usize;

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            if !random_smooth[i] {
                continue;
            }
            n_base_smooth += 1;

            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                if random_smooth[j] {
                    n_both_smooth += 1;
                }
            }
        }

        let c_delta = if n_base_smooth > 0 && overall_smooth_rate > 0.0 {
            let cond_prob = n_both_smooth as f64 / n_base_smooth as f64;
            cond_prob / overall_smooth_rate
        } else {
            0.0
        };

        lags.push(Autocorr2DLag {
            da: disp.da,
            db: disp.db,
            label: disp.label.clone(),
            n_both_smooth,
            n_base_smooth,
            c_delta,
        });
    }

    NfsRandom2DResult {
        n_bits,
        smooth_bound,
        n_grid,
        overall_smooth_rate,
        lags,
    }
}

// ---------------------------------------------------------------------------
// E24 Phase 5: Joint dual-norm cofactor correlation (novel)
// ---------------------------------------------------------------------------

/// Compute cross-correlation between cofactors of algebraic and rational norms.
///
/// For each (a,b) pair, computes cofactor_log of the algebraic norm F(a,b)
/// and cofactor_log of the rational norm |a+bm|. Tests whether these
/// cofactors are correlated (which would imply dual-norm sieving could help).
///
/// This test is impossible in QS, which has only one polynomial.
fn compute_nfs_dual_norm(
    grid: &[(i64, i64, u128, u128)],
    alg_partial_fracs: &[f64],
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> NfsDualNormResult {
    let n_grid = grid.len();

    // Compute rational-side partial fractions and cofactor logs
    let rat_partial_fracs: Vec<f64> = grid.iter()
        .map(|&(_, _, _, rat)| partial_smooth_fraction(rat, smooth_bound))
        .collect();
    let rat_cofactor_logs: Vec<f64> = grid.iter()
        .map(|&(_, _, _, rat)| cofactor_log(rat, smooth_bound))
        .collect();

    let rational_alg_cofactor_corr = if n_grid >= 3 {
        algebra::pearson_corr(alg_cofactor_logs, &rat_cofactor_logs)
    } else {
        0.0
    };

    let rational_mean_partial = if n_grid > 0 {
        rat_partial_fracs.iter().sum::<f64>() / n_grid as f64
    } else {
        0.0
    };
    let algebraic_mean_partial = if n_grid > 0 {
        alg_partial_fracs.iter().sum::<f64>() / n_grid as f64
    } else {
        0.0
    };

    NfsDualNormResult {
        n_bits,
        smooth_bound,
        n_grid,
        rational_alg_cofactor_corr,
        rational_mean_partial,
        algebraic_mean_partial,
    }
}

// ---------------------------------------------------------------------------
// E24 block computation + runner
// ---------------------------------------------------------------------------

/// Compute a single NFS lattice locality block for one (n_bits, smooth_bound) pair.
///
/// Generates the NFS grid ONCE, precomputes all per-point data, then runs
/// all 5 phases on the shared precomputed data.
fn compute_nfs_block(
    n_bits: u32,
    smooth_bound: u64,
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsBlockResult {
    eprintln!(
        "  E24 block: n_bits={}, B={}, area={}, max_b={}, seed=0x{:X}",
        n_bits, smooth_bound, sieve_area, max_b, seed
    );

    // Step 1: Generate NFS grid once
    let (_poly, grid) = generate_nfs_grid(n_bits, seed, sieve_area, max_b);
    let n_grid = grid.len();
    eprintln!("    grid size: {} coprime (a,b) pairs", n_grid);

    // Step 2: Build spatial index
    let index = build_grid_index(&grid);

    // Step 3: Precompute per-point algebraic norm data
    let alg_smooth_flags: Vec<bool> = grid.iter()
        .map(|&(_, _, alg, _)| is_b_smooth_u128(alg, smooth_bound))
        .collect();
    let alg_partial_fracs: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| partial_smooth_fraction(alg, smooth_bound))
        .collect();
    let alg_cofactor_logs: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| cofactor_log(alg, smooth_bound))
        .collect();

    let n_smooth: usize = alg_smooth_flags.iter().filter(|&&s| s).count();
    eprintln!(
        "    smooth rate: {:.4} ({}/{})",
        n_smooth as f64 / n_grid as f64,
        n_smooth,
        n_grid
    );

    // Phase 1: 2D binary smoothness autocorrelation
    let phase1 = compute_nfs_2d_autocorrelation(
        &grid, &index, &alg_smooth_flags, smooth_bound, n_bits,
    );
    eprintln!("    Phase 1 done: {} displacements", phase1.lags.len());

    // Phase 2: 2D partial-fraction Pearson autocorrelation
    let phase2 = compute_nfs_2d_partial_frac(
        &grid, &index, &alg_partial_fracs, smooth_bound, n_bits,
    );
    eprintln!("    Phase 2 done: mean_pf={:.4}", phase2.mean_partial_frac);

    // Phase 3: 2D cofactor decomposition
    let phase3 = compute_nfs_2d_cofactor_decomposition(
        &grid, &index, &alg_partial_fracs, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Phase 3 done: {} sieve primes", phase3.n_sieve_primes);

    // Phase 4: 2D random control
    let phase4 = compute_nfs_2d_random_control(
        &grid, &index, smooth_bound, n_bits, seed.wrapping_add(0x1000),
    );
    eprintln!("    Phase 4 done: random smooth rate={:.4}", phase4.overall_smooth_rate);

    // Phase 5: Joint dual-norm cofactor correlation
    let phase5 = compute_nfs_dual_norm(
        &grid, &alg_partial_fracs, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!(
        "    Phase 5 done: dual-norm cofactor corr={:.6}",
        phase5.rational_alg_cofactor_corr
    );

    NfsBlockResult {
        n_bits,
        smooth_bound,
        seed,
        n_grid,
        phase1,
        phase2,
        phase3,
        phase4,
        phase5,
    }
}

/// Write E24 checkpoint to JSON file.
fn write_nfs_checkpoint(path: &str, result: &NfsLatticeResult) {
    let json = serde_json::to_string_pretty(result).expect("serialize nfs lattice result");
    std::fs::write(path, json).expect("write nfs checkpoint");
}

/// Run the full E24 NFS 2D lattice locality experiment.
///
/// Iterates over all (bit_size, bound) combinations, computes a block for each,
/// and writes a checkpoint after every block completes.
pub fn run_nfs_lattice(
    bit_sizes: &[u32],
    bounds: &[u64],
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsLatticeResult {
    let checkpoint_path = "data/E24_nfs_lattice.json";
    let mut blocks = Vec::new();

    let total = bit_sizes.len() * bounds.len();
    let mut count = 0;

    for &n_bits in bit_sizes {
        for &bound in bounds {
            count += 1;
            eprintln!(
                "\n=== E24 block {}/{}: n_bits={}, B={} ===",
                count, total, n_bits, bound
            );

            let block_seed = seed
                .wrapping_add(n_bits as u64 * 0x1_0000)
                .wrapping_add(bound);
            let block = compute_nfs_block(n_bits, bound, sieve_area, max_b, block_seed);
            blocks.push(block);

            // Checkpoint after each block
            let partial = NfsLatticeResult { blocks: blocks.clone() };
            write_nfs_checkpoint(checkpoint_path, &partial);
            eprintln!("  checkpoint saved ({}/{})", count, total);
        }
    }

    NfsLatticeResult { blocks }
}

// ===========================================================================
// E24b: Artifact Validation Control Functions
// ===========================================================================

/// Simple OLS regression: y = a + b*x, returns (intercept, slope, r_squared).
fn simple_ols(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    if x.len() < 3 {
        return (0.0, 0.0, 0.0);
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }
    if ss_xx < 1e-30 {
        return (mean_y, 0.0, 0.0);
    }
    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = if ss_yy > 1e-30 { (ss_xy * ss_xy) / (ss_xx * ss_yy) } else { 0.0 };
    (intercept, slope, r_squared)
}

/// Compute OLS residuals: y - (intercept + slope * x).
fn ols_residuals(x: &[f64], y: &[f64], intercept: f64, slope: f64) -> Vec<f64> {
    x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - (intercept + slope * xi))
        .collect()
}

// ===========================================================================
// E24c: Robustness Check Helper Functions
// ===========================================================================

/// Monotone-bin residualization: sort y by x into k equal-frequency bins,
/// subtract within-bin mean of y. Returns residualized y values in original order.
///
/// Unlike OLS, this removes any monotone (not just linear) relationship
/// between x and y.
fn binned_residualize(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n == 0 || k == 0 {
        return y.to_vec();
    }
    let k = k.min(n);

    // Sort indices by x
    let mut indexed: Vec<(usize, f64)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign each index to a bin (equal-frequency)
    let bin_size = n / k;
    let mut residuals = vec![0.0f64; n];
    let mut start = 0;
    for bin in 0..k {
        let end = if bin == k - 1 { n } else { start + bin_size };
        let bin_indices: Vec<usize> = indexed[start..end].iter().map(|&(i, _)| i).collect();
        let bin_mean = if bin_indices.is_empty() {
            0.0
        } else {
            bin_indices.iter().map(|&i| y[i]).sum::<f64>() / bin_indices.len() as f64
        };
        for &i in &bin_indices {
            residuals[i] = y[i] - bin_mean;
        }
        start = end;
    }

    residuals
}

/// 2-predictor OLS regression: y = a + b1*x1 + b2*x2.
/// Returns (intercept, b1, b2, r_squared).
///
/// Solves the normal equations via centered 2x2 system (Cramer's rule).
fn ols_2d(x1: &[f64], x2: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    assert_eq!(x1.len(), x2.len());
    assert_eq!(x1.len(), y.len());
    let n = x1.len() as f64;
    if x1.len() < 4 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let m1 = x1.iter().sum::<f64>() / n;
    let m2 = x2.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut s11 = 0.0f64;
    let mut s22 = 0.0f64;
    let mut s12 = 0.0f64;
    let mut s1y = 0.0f64;
    let mut s2y = 0.0f64;
    let mut syy = 0.0f64;

    for i in 0..x1.len() {
        let d1 = x1[i] - m1;
        let d2 = x2[i] - m2;
        let dy = y[i] - my;
        s11 += d1 * d1;
        s22 += d2 * d2;
        s12 += d1 * d2;
        s1y += d1 * dy;
        s2y += d2 * dy;
        syy += dy * dy;
    }

    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-30 {
        // Singular: fall back to simple OLS on x1 only
        let (intercept, slope, r_sq) = simple_ols(x1, y);
        return (intercept, slope, 0.0, r_sq);
    }

    let b1 = (s22 * s1y - s12 * s2y) / det;
    let b2 = (s11 * s2y - s12 * s1y) / det;
    let intercept = my - b1 * m1 - b2 * m2;

    let ss_res: f64 = (0..x1.len())
        .map(|i| {
            let pred = intercept + b1 * x1[i] + b2 * x2[i];
            (y[i] - pred).powi(2)
        })
        .sum();
    let r_squared = if syy > 1e-30 { 1.0 - ss_res / syy } else { 0.0 };

    (intercept, b1, b2, r_squared)
}

/// Compute 2-predictor OLS residuals: y - (intercept + b1*x1 + b2*x2).
fn ols_2d_residuals(x1: &[f64], x2: &[f64], y: &[f64], intercept: f64, b1: f64, b2: f64) -> Vec<f64> {
    x1.iter().zip(x2.iter()).zip(y.iter())
        .map(|((&x1i, &x2i), &yi)| yi - (intercept + b1 * x1i + b2 * x2i))
        .collect()
}

/// Replace values with their ranks (0-indexed). Ties get average rank.
fn rank_transform(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-30 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Winsorize a slice at the given lower and upper percentiles (e.g., 0.05 and 0.95).
/// Returns a new vector with extreme values clipped to the percentile boundaries.
fn winsorize(values: &[f64], lower_pct: f64, upper_pct: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = ((n as f64 * lower_pct).floor() as usize).min(n - 1);
    let hi_idx = ((n as f64 * upper_pct).ceil() as usize).min(n - 1);
    let lo_val = sorted[lo_idx];
    let hi_val = sorted[hi_idx];

    values.iter().map(|&v| v.clamp(lo_val, hi_val)).collect()
}

/// Control A: Norm-residualized cofactor autocorrelation.
///
/// Regresses cofactor_log against log2(alg_norm) to remove the magnitude
/// gradient. If the cofactor autocorrelation signal is just from nearby points
/// having similar norm sizes, residualized autocorrelation → 0.
fn compute_norm_residualized(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> NormResidualizedResult {
    let n_grid = grid.len();

    // Regress cofactor_log against log2(alg_norm)
    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();

    let (intercept, slope, norm_r_squared) = simple_ols(&log_norms, alg_cofactor_logs);
    let residualized = ols_residuals(&log_norms, alg_cofactor_logs, intercept, slope);

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();

    for disp in &displacements {
        let mut base_raw = Vec::new();
        let mut disp_raw = Vec::new();
        let mut base_resid = Vec::new();
        let mut disp_resid = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_raw.push(alg_cofactor_logs[i]);
                disp_raw.push(alg_cofactor_logs[j]);
                base_resid.push(residualized[i]);
                disp_resid.push(residualized[j]);
            }
        }

        let raw_corr = if base_raw.len() >= 3 {
            algebra::pearson_corr(&base_raw, &disp_raw)
        } else {
            0.0
        };
        let resid_corr = if base_resid.len() >= 3 {
            algebra::pearson_corr(&base_resid, &disp_resid)
        } else {
            0.0
        };

        displacement_comparisons.push((disp.label.clone(), raw_corr, resid_corr));
    }

    NormResidualizedResult {
        n_bits,
        smooth_bound,
        n_grid,
        norm_r_squared,
        displacement_comparisons,
    }
}

/// Control B: Per-side (rational vs algebraic) cofactor analysis with residualization.
fn compute_per_side_cofactor(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    smooth_bound: u64,
    n_bits: u32,
) -> PerSideCofactorResult {
    let n_grid = grid.len();

    // Algebraic side
    let alg_cf: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| cofactor_log(alg, smooth_bound))
        .collect();
    let alg_log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();
    let (ai, as_, _) = simple_ols(&alg_log_norms, &alg_cf);
    let alg_resid = ols_residuals(&alg_log_norms, &alg_cf, ai, as_);

    // Rational side
    let rat_cf: Vec<f64> = grid.iter()
        .map(|&(_, _, _, rat)| cofactor_log(rat, smooth_bound))
        .collect();
    let rat_log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, _, rat)| if rat > 1 { (rat as f64).log2() } else { 0.0 })
        .collect();
    let (ri, rs_, _) = simple_ols(&rat_log_norms, &rat_cf);
    let rat_resid = ols_residuals(&rat_log_norms, &rat_cf, ri, rs_);

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();

    for disp in &displacements {
        let mut base_alg = Vec::new();
        let mut disp_alg = Vec::new();
        let mut base_rat = Vec::new();
        let mut disp_rat = Vec::new();
        let mut base_alg_r = Vec::new();
        let mut disp_alg_r = Vec::new();
        let mut base_rat_r = Vec::new();
        let mut disp_rat_r = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_alg.push(alg_cf[i]);
                disp_alg.push(alg_cf[j]);
                base_rat.push(rat_cf[i]);
                disp_rat.push(rat_cf[j]);
                base_alg_r.push(alg_resid[i]);
                disp_alg_r.push(alg_resid[j]);
                base_rat_r.push(rat_resid[i]);
                disp_rat_r.push(rat_resid[j]);
            }
        }

        let alg_corr = if base_alg.len() >= 3 { algebra::pearson_corr(&base_alg, &disp_alg) } else { 0.0 };
        let rat_corr = if base_rat.len() >= 3 { algebra::pearson_corr(&base_rat, &disp_rat) } else { 0.0 };
        let alg_resid_corr = if base_alg_r.len() >= 3 { algebra::pearson_corr(&base_alg_r, &disp_alg_r) } else { 0.0 };
        let rat_resid_corr = if base_rat_r.len() >= 3 { algebra::pearson_corr(&base_rat_r, &disp_rat_r) } else { 0.0 };

        displacement_comparisons.push((disp.label.clone(), alg_corr, rat_corr, alg_resid_corr, rat_resid_corr));
    }

    PerSideCofactorResult {
        n_bits,
        smooth_bound,
        n_grid,
        displacement_comparisons,
    }
}

/// Control C: Magnitude-bin shuffle null.
///
/// Bins grid points by log2(alg_norm) into equal-frequency bins, then shuffles
/// cofactor_logs within each bin. This preserves the magnitude profile while
/// destroying any genuine local structure. Repeated n_shuffles times.
fn compute_magnitude_bin_shuffle(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
    n_bins: usize,
    n_shuffles: usize,
    seed: u64,
) -> MagnitudeBinShuffleResult {
    let n_grid = grid.len();

    // Compute log norms and sort indices by log norm
    let mut indexed_log_norms: Vec<(usize, f64)> = grid.iter()
        .enumerate()
        .map(|(i, &(_, _, alg, _))| (i, if alg > 1 { (alg as f64).log2() } else { 0.0 }))
        .collect();
    indexed_log_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign each point to a bin (equal-frequency bins)
    let bin_size = n_grid / n_bins.max(1);
    let mut bin_assignments = vec![0usize; n_grid];
    for (rank, &(orig_idx, _)) in indexed_log_norms.iter().enumerate() {
        bin_assignments[orig_idx] = (rank / bin_size.max(1)).min(n_bins - 1);
    }

    // Build bin → indices mapping
    let mut bin_indices: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
    for (i, &bin) in bin_assignments.iter().enumerate() {
        bin_indices[bin].push(i);
    }

    let displacements = standard_displacements();

    // Compute original cofactor autocorrelations
    let mut orig_corrs = Vec::new();
    for disp in &displacements {
        let mut base_cf = Vec::new();
        let mut disp_cf = Vec::new();
        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_cf.push(alg_cofactor_logs[i]);
                disp_cf.push(alg_cofactor_logs[j]);
            }
        }
        let corr = if base_cf.len() >= 3 { algebra::pearson_corr(&base_cf, &disp_cf) } else { 0.0 };
        orig_corrs.push(corr);
    }

    // Shuffle within bins and recompute autocorrelations
    let mut shuffled_corrs: Vec<Vec<f64>> = vec![Vec::new(); displacements.len()];

    for s in 0..n_shuffles {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(s as u64));
        let mut shuffled_cf = alg_cofactor_logs.to_vec();

        // Fisher-Yates shuffle within each bin
        for bin_idxs in &bin_indices {
            let n = bin_idxs.len();
            if n <= 1 { continue; }
            for k in (1..n).rev() {
                let j: usize = rng.gen::<usize>() % (k + 1);
                let idx_k = bin_idxs[k];
                let idx_j = bin_idxs[j];
                shuffled_cf.swap(idx_k, idx_j);
            }
        }

        for (d_idx, disp) in displacements.iter().enumerate() {
            let mut base_cf = Vec::new();
            let mut disp_cf = Vec::new();
            for (i, &(a, b, _, _)) in grid.iter().enumerate() {
                let na = a + disp.da;
                let nb = b + disp.db;
                if let Some(&j) = index.get(&(na, nb)) {
                    base_cf.push(shuffled_cf[i]);
                    disp_cf.push(shuffled_cf[j]);
                }
            }
            let corr = if base_cf.len() >= 3 { algebra::pearson_corr(&base_cf, &disp_cf) } else { 0.0 };
            shuffled_corrs[d_idx].push(corr);
        }
    }

    let mut displacement_comparisons = Vec::new();
    for (d_idx, disp) in displacements.iter().enumerate() {
        let mean_shuffled = if !shuffled_corrs[d_idx].is_empty() {
            shuffled_corrs[d_idx].iter().sum::<f64>() / shuffled_corrs[d_idx].len() as f64
        } else { 0.0 };
        let std_shuffled = if shuffled_corrs[d_idx].len() >= 2 {
            let var: f64 = shuffled_corrs[d_idx].iter()
                .map(|&c| (c - mean_shuffled).powi(2))
                .sum::<f64>() / (shuffled_corrs[d_idx].len() - 1) as f64;
            var.sqrt()
        } else { 0.0 };

        displacement_comparisons.push((disp.label.clone(), orig_corrs[d_idx], mean_shuffled, std_shuffled));
    }

    MagnitudeBinShuffleResult {
        n_bits,
        smooth_bound,
        n_grid,
        n_bins,
        displacement_comparisons,
    }
}

/// Control D: Displacement decay analysis.
///
/// Generates displacements at multiple radii (1, sqrt(2), 2, sqrt(5), ..., 10)
/// and measures how cofactor autocorrelation (raw + residualized) decays with
/// displacement distance.
fn compute_displacement_decay(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> DisplacementDecayResult {
    let n_grid = grid.len();

    // Residualize cofactor_log against log2(norm)
    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();
    let (intercept, slope, _) = simple_ols(&log_norms, alg_cofactor_logs);
    let residualized = ols_residuals(&log_norms, alg_cofactor_logs, intercept, slope);

    // Generate displacements at various radii
    // Group by approximate radius and average
    let max_r = 10i64;
    let mut all_disps: Vec<(i64, i64, f64)> = Vec::new(); // (da, db, norm)
    for da in -max_r..=max_r {
        for db in -max_r..=max_r {
            if da == 0 && db == 0 { continue; }
            let norm = ((da * da + db * db) as f64).sqrt();
            if norm <= max_r as f64 {
                all_disps.push((da, db, norm));
            }
        }
    }

    // Group by binned radius (0.5-wide bins)
    let mut radius_bins: HashMap<u32, Vec<(i64, i64, f64)>> = HashMap::new();
    for &(da, db, norm) in &all_disps {
        let bin = (norm * 2.0).round() as u32; // 0.5-wide bins
        radius_bins.entry(bin).or_default().push((da, db, norm));
    }

    let mut decay_by_radius: Vec<(f64, f64, f64, usize)> = Vec::new();

    let mut sorted_bins: Vec<u32> = radius_bins.keys().cloned().collect();
    sorted_bins.sort();

    for bin in sorted_bins {
        let disps = &radius_bins[&bin];
        let mean_radius = disps.iter().map(|&(_, _, r)| r).sum::<f64>() / disps.len() as f64;

        let mut raw_corrs = Vec::new();
        let mut resid_corrs = Vec::new();

        for &(da, db, _) in disps {
            let mut base_raw = Vec::new();
            let mut disp_raw = Vec::new();
            let mut base_res = Vec::new();
            let mut disp_res = Vec::new();

            for (i, &(a, b, _, _)) in grid.iter().enumerate() {
                let na = a + da;
                let nb = b + db;
                if let Some(&j) = index.get(&(na, nb)) {
                    base_raw.push(alg_cofactor_logs[i]);
                    disp_raw.push(alg_cofactor_logs[j]);
                    base_res.push(residualized[i]);
                    disp_res.push(residualized[j]);
                }
            }

            if base_raw.len() >= 3 {
                raw_corrs.push(algebra::pearson_corr(&base_raw, &disp_raw));
                resid_corrs.push(algebra::pearson_corr(&base_res, &disp_res));
            }
        }

        if !raw_corrs.is_empty() {
            let mean_raw = raw_corrs.iter().sum::<f64>() / raw_corrs.len() as f64;
            let mean_resid = resid_corrs.iter().sum::<f64>() / resid_corrs.len() as f64;
            decay_by_radius.push((mean_radius, mean_raw, mean_resid, raw_corrs.len()));
        }
    }

    DisplacementDecayResult {
        n_bits,
        smooth_bound,
        n_grid,
        decay_by_radius,
    }
}

/// Control E: Conditional cofactor correlation (binned by partial_smooth_fraction).
///
/// Among points with similar sieve scores (partial_smooth_fraction), does
/// cofactor autocorrelation still persist? Tests displacement (1,0) specifically.
fn compute_conditional_cofactor(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_partial_fracs: &[f64],
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
    n_bins: usize,
) -> ConditionalCofactorResult {
    let n_grid = grid.len();

    // Sort partial fracs to determine bin edges (equal-frequency bins)
    let mut sorted_pf: Vec<f64> = alg_partial_fracs.to_vec();
    sorted_pf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let bin_size = n_grid / n_bins.max(1);
    let bin_edges: Vec<f64> = (0..n_bins)
        .map(|i| sorted_pf[(i * bin_size).min(n_grid - 1)])
        .collect();

    // Assign each point to a pf bin
    let get_bin = |pf: f64| -> usize {
        for i in (0..n_bins).rev() {
            if pf >= bin_edges[i] {
                return i;
            }
        }
        0
    };

    let pf_bins: Vec<usize> = alg_partial_fracs.iter().map(|&pf| get_bin(pf)).collect();

    // Build bin → indices
    let mut bin_indices: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
    for (i, &bin) in pf_bins.iter().enumerate() {
        bin_indices[bin].push(i);
    }

    // For displacement (1,0), compute cofactor autocorrelation WITHIN each pf bin
    let da = 1i64;
    let db = 0i64;

    let mut bin_results = Vec::new();
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;

    for bin in 0..n_bins {
        let idxs = &bin_indices[bin];
        let n_in_bin = idxs.len();
        if n_in_bin < 3 { continue; }

        let bin_center = idxs.iter().map(|&i| alg_partial_fracs[i]).sum::<f64>() / n_in_bin as f64;

        // Collect cofactor pairs where BOTH base and displaced are in this bin
        let mut base_cf = Vec::new();
        let mut disp_cf = Vec::new();

        for &i in idxs {
            let (a, b, _, _) = grid[i];
            let na = a + da;
            let nb = b + db;
            if let Some(&j) = index.get(&(na, nb)) {
                if pf_bins[j] == bin {
                    base_cf.push(alg_cofactor_logs[i]);
                    disp_cf.push(alg_cofactor_logs[j]);
                }
            }
        }

        let cf_corr = if base_cf.len() >= 3 {
            algebra::pearson_corr(&base_cf, &disp_cf)
        } else {
            f64::NAN
        };

        bin_results.push((bin_center, n_in_bin, cf_corr));
        if cf_corr.is_finite() && base_cf.len() >= 3 {
            weighted_sum += cf_corr * base_cf.len() as f64;
            total_weight += base_cf.len() as f64;
        }
    }

    let conditional_cf_corr = if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 };

    ConditionalCofactorResult {
        n_bits,
        smooth_bound,
        n_grid,
        n_bins,
        bin_results,
        conditional_cf_corr,
    }
}

// ===========================================================================
// E24c: Robustness Check Compute Functions
// ===========================================================================

/// Check 1: Nonlinear (monotone-bin) residualization.
///
/// OLS assumes linear cofactor_log ~ log2(norm). If the true relationship is
/// nonlinear, OLS under-removes the gradient. Monotone-bin residualization
/// (k equal-frequency bins sorted by log2(norm), subtract within-bin mean)
/// makes no linearity assumption.
fn compute_nonlinear_resid(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
    n_bins: usize,
) -> NonlinearResidResult {
    let n_grid = grid.len();

    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();

    // OLS residualization (same as E24b Control A)
    let (intercept, slope, _) = simple_ols(&log_norms, alg_cofactor_logs);
    let ols_resid = ols_residuals(&log_norms, alg_cofactor_logs, intercept, slope);

    // Bin residualization (nonlinear)
    let bin_resid = binned_residualize(&log_norms, alg_cofactor_logs, n_bins);

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();

    for disp in &displacements {
        let mut base_raw = Vec::new();
        let mut disp_raw = Vec::new();
        let mut base_ols = Vec::new();
        let mut disp_ols = Vec::new();
        let mut base_bin = Vec::new();
        let mut disp_bin = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_raw.push(alg_cofactor_logs[i]);
                disp_raw.push(alg_cofactor_logs[j]);
                base_ols.push(ols_resid[i]);
                disp_ols.push(ols_resid[j]);
                base_bin.push(bin_resid[i]);
                disp_bin.push(bin_resid[j]);
            }
        }

        let raw_corr = if base_raw.len() >= 3 { algebra::pearson_corr(&base_raw, &disp_raw) } else { 0.0 };
        let ols_corr = if base_ols.len() >= 3 { algebra::pearson_corr(&base_ols, &disp_ols) } else { 0.0 };
        let bin_corr = if base_bin.len() >= 3 { algebra::pearson_corr(&base_bin, &disp_bin) } else { 0.0 };

        displacement_comparisons.push((disp.label.clone(), raw_corr, ols_corr, bin_corr));
    }

    NonlinearResidResult {
        n_bits,
        smooth_bound,
        n_grid,
        n_bins,
        displacement_comparisons,
    }
}

/// Check 2: Cross-validated residualization.
///
/// Split grid into spatial halves (a < 0 vs a >= 0). Fit OLS on one half,
/// compute residuals on the other. Measure autocorrelation on held-out half only.
/// If held-out autocorrelations match in-sample ones, no overfitting.
fn compute_crossval_resid(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> CrossValidatedResidResult {
    let n_grid = grid.len();

    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();

    // Split indices by a-coordinate
    let left_mask: Vec<bool> = grid.iter().map(|&(a, _, _, _)| a < 0).collect();
    let n_left = left_mask.iter().filter(|&&m| m).count();
    let n_right = n_grid - n_left;

    // Collect left and right subsets for OLS
    let left_x: Vec<f64> = (0..n_grid).filter(|&i| left_mask[i]).map(|i| log_norms[i]).collect();
    let left_y: Vec<f64> = (0..n_grid).filter(|&i| left_mask[i]).map(|i| alg_cofactor_logs[i]).collect();
    let right_x: Vec<f64> = (0..n_grid).filter(|&i| !left_mask[i]).map(|i| log_norms[i]).collect();
    let right_y: Vec<f64> = (0..n_grid).filter(|&i| !left_mask[i]).map(|i| alg_cofactor_logs[i]).collect();

    // Fit OLS on left half
    let (li, ls, _) = simple_ols(&left_x, &left_y);
    // Fit OLS on right half
    let (ri, rs, _) = simple_ols(&right_x, &right_y);

    // In-sample residuals (full data, same as E24b)
    let (fi, fs, _) = simple_ols(&log_norms, alg_cofactor_logs);
    let full_resid = ols_residuals(&log_norms, alg_cofactor_logs, fi, fs);

    // Cross-validated residuals: for each point, use the OTHER half's model
    let cv_resid: Vec<f64> = (0..n_grid).map(|i| {
        if left_mask[i] {
            // Point is in left half -> use right's model
            alg_cofactor_logs[i] - (ri + rs * log_norms[i])
        } else {
            // Point is in right half -> use left's model
            alg_cofactor_logs[i] - (li + ls * log_norms[i])
        }
    }).collect();

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();

    for disp in &displacements {
        let mut base_full = Vec::new();
        let mut disp_full = Vec::new();
        let mut base_ho_left = Vec::new();
        let mut disp_ho_left = Vec::new();
        let mut base_ho_right = Vec::new();
        let mut disp_ho_right = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                base_full.push(full_resid[i]);
                disp_full.push(full_resid[j]);

                if left_mask[i] && left_mask[j] {
                    base_ho_left.push(cv_resid[i]);
                    disp_ho_left.push(cv_resid[j]);
                }
                if !left_mask[i] && !left_mask[j] {
                    base_ho_right.push(cv_resid[i]);
                    disp_ho_right.push(cv_resid[j]);
                }
            }
        }

        let full_corr = if base_full.len() >= 3 { algebra::pearson_corr(&base_full, &disp_full) } else { 0.0 };
        let left_corr = if base_ho_left.len() >= 3 { algebra::pearson_corr(&base_ho_left, &disp_ho_left) } else { 0.0 };
        let right_corr = if base_ho_right.len() >= 3 { algebra::pearson_corr(&base_ho_right, &disp_ho_right) } else { 0.0 };

        displacement_comparisons.push((disp.label.clone(), full_corr, left_corr, right_corr));
    }

    CrossValidatedResidResult {
        n_bits,
        smooth_bound,
        n_grid,
        n_left,
        n_right,
        displacement_comparisons,
    }
}

/// Check 3: Partial correlation controlling for both endpoint norms.
///
/// E24b Control A only regresses against the base point's log2(norm).
/// Here we do 2-predictor OLS: regress each side against both log_norm_i
/// AND log_norm_j, then correlate the two residual vectors. This is proper
/// partial correlation controlling for both endpoints' magnitudes.
fn compute_partial_correlation(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> PartialCorrelationResult {
    let n_grid = grid.len();

    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();

    // 1D OLS for reference (same as E24b)
    let (fi, fs, _) = simple_ols(&log_norms, alg_cofactor_logs);
    let ols_1d_resid = ols_residuals(&log_norms, alg_cofactor_logs, fi, fs);

    let displacements = standard_displacements();
    let mut displacement_comparisons = Vec::new();
    let mut r_sq_2d_sum = 0.0f64;
    let mut r_sq_1d_sum = 0.0f64;
    let mut n_disps = 0usize;

    for disp in &displacements {
        let mut cf_base = Vec::new();
        let mut cf_neigh = Vec::new();
        let mut ln_base = Vec::new();
        let mut ln_neigh = Vec::new();
        let mut ols1d_base = Vec::new();
        let mut ols1d_neigh = Vec::new();

        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            let na = a + disp.da;
            let nb = b + disp.db;
            if let Some(&j) = index.get(&(na, nb)) {
                cf_base.push(alg_cofactor_logs[i]);
                cf_neigh.push(alg_cofactor_logs[j]);
                ln_base.push(log_norms[i]);
                ln_neigh.push(log_norms[j]);
                ols1d_base.push(ols_1d_resid[i]);
                ols1d_neigh.push(ols_1d_resid[j]);
            }
        }

        if cf_base.len() < 4 {
            displacement_comparisons.push((disp.label.clone(), 0.0, 0.0, 0.0));
            continue;
        }

        let raw_corr = algebra::pearson_corr(&cf_base, &cf_neigh);
        let ols1d_corr = algebra::pearson_corr(&ols1d_base, &ols1d_neigh);

        // 2-predictor OLS: regress cf_base on (ln_base, ln_neigh)
        let (int_b, b1_b, b2_b, r2_b) = ols_2d(&ln_base, &ln_neigh, &cf_base);
        let resid_base = ols_2d_residuals(&ln_base, &ln_neigh, &cf_base, int_b, b1_b, b2_b);

        // 2-predictor OLS: regress cf_neigh on (ln_base, ln_neigh)
        let (int_n, b1_n, b2_n, _r2_n) = ols_2d(&ln_base, &ln_neigh, &cf_neigh);
        let resid_neigh = ols_2d_residuals(&ln_base, &ln_neigh, &cf_neigh, int_n, b1_n, b2_n);

        let partial_corr = algebra::pearson_corr(&resid_base, &resid_neigh);

        // Track R² for reporting
        let (_, _, r2_1d) = simple_ols(&ln_base, &cf_base);
        r_sq_1d_sum += r2_1d;
        r_sq_2d_sum += r2_b;
        n_disps += 1;

        displacement_comparisons.push((disp.label.clone(), raw_corr, ols1d_corr, partial_corr));
    }

    PartialCorrelationResult {
        n_bits,
        smooth_bound,
        n_grid,
        r_squared_2d: if n_disps > 0 { r_sq_2d_sum / n_disps as f64 } else { 0.0 },
        r_squared_1d: if n_disps > 0 { r_sq_1d_sum / n_disps as f64 } else { 0.0 },
        displacement_comparisons,
    }
}

/// Check 4: Alternative transforms of cofactor_log.
///
/// Tests robustness of residualization to the choice of metric:
/// (a) Rank cofactor: replace cofactor_log with ranks (most robust to any monotone transform)
/// (b) Winsorized cofactor_log: clip at 5th/95th percentiles before residualization
/// (c) Raw cofactor bits: ceil(cofactor_log) if > 0, else 0 — integer discretization
fn compute_alternative_transforms(
    grid: &[(i64, i64, u128, u128)],
    index: &HashMap<(i64, i64), usize>,
    alg_cofactor_logs: &[f64],
    smooth_bound: u64,
    n_bits: u32,
) -> AlternativeTransformResult {
    let n_grid = grid.len();

    let log_norms: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| if alg > 1 { (alg as f64).log2() } else { 0.0 })
        .collect();

    let transforms: Vec<(&str, Vec<f64>)> = vec![
        ("rank_cofactor", rank_transform(alg_cofactor_logs)),
        ("winsorized_cofactor_log", winsorize(alg_cofactor_logs, 0.05, 0.95)),
        ("cofactor_bits", alg_cofactor_logs.iter()
            .map(|&v| if v > 0.0 { v.ceil() } else { 0.0 })
            .collect()),
    ];

    let displacements = standard_displacements();
    let mut variants = Vec::new();

    for (name, transformed) in &transforms {
        // OLS residualization of transformed values against log2(norm)
        let (intercept, slope, _) = simple_ols(&log_norms, transformed);
        let residualized = ols_residuals(&log_norms, transformed, intercept, slope);

        let mut displacement_comparisons = Vec::new();

        for disp in &displacements {
            let mut base_raw = Vec::new();
            let mut disp_raw = Vec::new();
            let mut base_res = Vec::new();
            let mut disp_res = Vec::new();

            for (i, &(a, b, _, _)) in grid.iter().enumerate() {
                let na = a + disp.da;
                let nb = b + disp.db;
                if let Some(&j) = index.get(&(na, nb)) {
                    base_raw.push(transformed[i]);
                    disp_raw.push(transformed[j]);
                    base_res.push(residualized[i]);
                    disp_res.push(residualized[j]);
                }
            }

            let raw_corr = if base_raw.len() >= 3 { algebra::pearson_corr(&base_raw, &disp_raw) } else { 0.0 };
            let res_corr = if base_res.len() >= 3 { algebra::pearson_corr(&base_res, &disp_res) } else { 0.0 };

            displacement_comparisons.push((disp.label.clone(), raw_corr, res_corr));
        }

        variants.push(TransformVariantResult {
            transform_name: name.to_string(),
            displacement_comparisons,
        });
    }

    AlternativeTransformResult {
        n_bits,
        smooth_bound,
        n_grid,
        variants,
    }
}

/// Compute full validation block for one (n_bits, smooth_bound) configuration.
fn compute_nfs_validation_block(
    n_bits: u32,
    smooth_bound: u64,
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsValidationBlockResult {
    eprintln!(
        "  E24b validation: n_bits={}, B={}, area={}, max_b={}, seed=0x{:X}",
        n_bits, smooth_bound, sieve_area, max_b, seed
    );

    let (_poly, grid) = generate_nfs_grid(n_bits, seed, sieve_area, max_b);
    let n_grid = grid.len();
    let index = build_grid_index(&grid);
    eprintln!("    grid size: {} coprime pairs", n_grid);

    let alg_partial_fracs: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| partial_smooth_fraction(alg, smooth_bound))
        .collect();
    let alg_cofactor_logs: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| cofactor_log(alg, smooth_bound))
        .collect();

    // Control A: norm-residualized
    let control_a = compute_norm_residualized(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Control A done: R²(cf~log_norm) = {:.4}", control_a.norm_r_squared);

    // Control B: per-side
    let control_b = compute_per_side_cofactor(&grid, &index, smooth_bound, n_bits);
    eprintln!("    Control B done: per-side analysis complete");

    // Control C: magnitude-bin shuffle (20 bins, 50 shuffles)
    let control_c = compute_magnitude_bin_shuffle(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
        20, 50, seed.wrapping_add(0xC000),
    );
    eprintln!("    Control C done: {} bins, 50 shuffles", control_c.n_bins);

    // Control D: displacement decay
    let control_d = compute_displacement_decay(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Control D done: {} radius bins", control_d.decay_by_radius.len());

    // Control E: conditional cofactor (10 pf-bins)
    let control_e = compute_conditional_cofactor(
        &grid, &index, &alg_partial_fracs, &alg_cofactor_logs, smooth_bound, n_bits, 10,
    );
    eprintln!("    Control E done: conditional cf_corr = {:.6}", control_e.conditional_cf_corr);

    NfsValidationBlockResult {
        n_bits,
        smooth_bound,
        seed,
        control_a,
        control_b,
        control_c,
        control_d,
        control_e,
    }
}

/// Run the full E24b validation experiment.
pub fn run_nfs_validation(
    bit_sizes: &[u32],
    bounds: &[u64],
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsValidationResult {
    let checkpoint_path = "data/E24b_nfs_validation.json";
    let mut blocks = Vec::new();

    let total = bit_sizes.len() * bounds.len();
    let mut count = 0;

    for &n_bits in bit_sizes {
        for &bound in bounds {
            count += 1;
            eprintln!(
                "\n=== E24b validation {}/{}: n_bits={}, B={} ===",
                count, total, n_bits, bound
            );

            let block_seed = seed
                .wrapping_add(n_bits as u64 * 0x1_0000)
                .wrapping_add(bound);
            let block = compute_nfs_validation_block(n_bits, bound, sieve_area, max_b, block_seed);
            blocks.push(block);

            let partial = NfsValidationResult { blocks: blocks.clone() };
            let json = serde_json::to_string_pretty(&partial).expect("serialize");
            std::fs::write(checkpoint_path, json).expect("write checkpoint");
            eprintln!("  checkpoint saved ({}/{})", count, total);
        }
    }

    NfsValidationResult { blocks }
}

// ===========================================================================
// E24c: Block Computation and Runner
// ===========================================================================

/// Compute full E24c robustness block for one (n_bits, smooth_bound) configuration.
fn compute_nfs_robustness_block(
    n_bits: u32,
    smooth_bound: u64,
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsRobustnessBlockResult {
    eprintln!(
        "  E24c robustness: n_bits={}, B={}, area={}, max_b={}, seed=0x{:X}",
        n_bits, smooth_bound, sieve_area, max_b, seed
    );

    let (_poly, grid) = generate_nfs_grid(n_bits, seed, sieve_area, max_b);
    let n_grid = grid.len();
    let index = build_grid_index(&grid);
    eprintln!("    grid size: {} coprime pairs", n_grid);

    let alg_cofactor_logs: Vec<f64> = grid.iter()
        .map(|&(_, _, alg, _)| cofactor_log(alg, smooth_bound))
        .collect();

    // Check 1: nonlinear residualization (50 bins)
    let check1 = compute_nonlinear_resid(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits, 50,
    );
    eprintln!("    Check 1 (nonlinear resid) done");

    // Check 2: cross-validated residualization
    let check2 = compute_crossval_resid(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Check 2 (cross-val) done: n_left={}, n_right={}", check2.n_left, check2.n_right);

    // Check 3: partial correlation (both norms)
    let check3 = compute_partial_correlation(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Check 3 (partial corr) done: R²_1d={:.4}, R²_2d={:.4}", check3.r_squared_1d, check3.r_squared_2d);

    // Check 4: alternative transforms
    let check4 = compute_alternative_transforms(
        &grid, &index, &alg_cofactor_logs, smooth_bound, n_bits,
    );
    eprintln!("    Check 4 (alt transforms) done: {} variants", check4.variants.len());

    NfsRobustnessBlockResult {
        n_bits,
        smooth_bound,
        seed,
        n_grid,
        check1_nonlinear: check1,
        check2_crossval: check2,
        check3_partial_corr: check3,
        check4_transforms: check4,
    }
}

/// Run the full E24c robustness experiment.
pub fn run_nfs_robustness(
    bit_sizes: &[u32],
    bounds: &[u64],
    sieve_area: i64,
    max_b: i64,
    seed: u64,
) -> NfsRobustnessResult {
    let checkpoint_path = "data/E24c_nfs_robustness.json";
    let mut blocks = Vec::new();

    let total = bit_sizes.len() * bounds.len();
    let mut count = 0;

    for &n_bits in bit_sizes {
        for &bound in bounds {
            count += 1;
            eprintln!(
                "\n=== E24c robustness {}/{}: n_bits={}, B={} ===",
                count, total, n_bits, bound
            );

            let block_seed = seed
                .wrapping_add(n_bits as u64 * 0x1_0000)
                .wrapping_add(bound);
            let block = compute_nfs_robustness_block(n_bits, bound, sieve_area, max_b, block_seed);
            blocks.push(block);

            let partial = NfsRobustnessResult { blocks: blocks.clone() };
            let json = serde_json::to_string_pretty(&partial).expect("serialize");
            std::fs::write(checkpoint_path, json).expect("write checkpoint");
            eprintln!("  checkpoint saved ({}/{})", count, total);
        }
    }

    NfsRobustnessResult { blocks }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use eisenstein_hunt::CHANNELS;

    #[test]
    fn test_enumerate_16bit() {
        let pairs = enumerate_balanced_semiprimes(16);
        assert!(!pairs.is_empty(), "should find some 16-bit balanced semiprimes");
        for &(p, q) in &pairs {
            assert!(is_prime_u64(p) && is_prime_u64(q));
            let n = p as u128 * q as u128;
            assert!(n >= (1u128 << 15) && n < (1u128 << 16));
            assert!((p as f64) / (q as f64) >= 0.3);
            assert!(p != q);
        }
    }

    #[test]
    fn test_run_character_audit_smoke() {
        // Quick smoke test: 14-bit, channel 0 (k=12, ℓ=691)
        let ch = &CHANNELS[0];
        let result = run_character_audit(14, ch, 2, 42);
        assert!(result.n_primes > 0);
        assert!(!result.eigenvectors.is_empty());
        let ev = &result.eigenvectors[0];
        // Eigenvalue and correlation should be finite
        assert!(ev.eigenvalue.is_finite());
        assert!(ev.corr_re_best.is_finite());
        assert!(ev.product_corr_sigma.is_finite());
        assert!(ev.product_corr_nk.is_finite());
        // Correlation must be in [-1, 1]
        assert!(ev.corr_re_best.abs() <= 1.0 + 1e-9);
        assert!(ev.product_corr_sigma.abs() <= 1.0 + 1e-9);
        assert!(ev.product_corr_nk.abs() <= 1.0 + 1e-9);
    }

    #[test]
    fn test_run_character_audit_channel5() {
        // Channel 5 has the smallest ell (131), easiest to scan
        let ch = &CHANNELS[5]; // k=22, ell=131
        let result = run_character_audit(16, ch, 1, 0xc0ffee);
        assert!(result.n_primes > 0);
        let ev = &result.eigenvectors[0];
        // For ℓ=131 and n=16: the character structure should be visible.
        // We just check it runs and produces finite values.
        assert!(ev.best_char_amp.is_finite() && ev.best_char_amp >= 0.0);
    }

    #[test]
    fn test_character_structure_prediction() {
        // Verify the key algebraic identity:
        // g(p)·g(q) mod ℓ = σ_{k−1}(pq) mod ℓ
        let ch   = &CHANNELS[0]; // k=12, ell=691
        let ell  = ch.ell;
        let k1   = (ch.weight - 1) as u64;
        let pairs = enumerate_balanced_semiprimes(14);
        for &(p, q) in pairs.iter().take(20) {
            let gp     = (1 + mod_pow(p, k1, ell)) % ell;
            let gq     = (1 + mod_pow(q, k1, ell)) % ell;
            let sigma  = eisenstein_hunt::ground_truth(
                &Semiprime { n: p * q, p, q }, ch,
            );
            // g(p)·g(q) mod ℓ should equal σ_{k−1}(pq) mod ℓ
            assert_eq!(gp * gq % ell, sigma % ell,
                "identity failed at p={p}, q={q}");
        }
    }

    #[test]
    fn test_centered_parity_spectrum_dc_near_zero() {
        // For prime ℓ, half the elements of {1,…,ℓ-1} are odd, half even.
        // Centered parity: DC = (n_odd - n_even) / (ℓ-1).
        // For ℓ > 2, DC should be ±1/(ℓ-1) (exactly one more odd than even,
        // or vice versa).
        let result = centered_parity_spectrum(131, 5);
        assert!(
            result.dc_component.abs() < 0.02,
            "DC should be near zero for centered parity, got {}",
            result.dc_component
        );
    }

    #[test]
    fn test_centered_parity_spectrum_scaling() {
        // For ℓ = 131, A_max · √ℓ should be O(1) — roughly 1–2.
        let result = centered_parity_spectrum(131, 10);
        assert!(
            result.a_max_scaled > 0.5 && result.a_max_scaled < 5.0,
            "A_max·√ℓ should be O(1), got {}",
            result.a_max_scaled
        );
        assert!(
            !result.top_amplitudes.is_empty(),
            "should have at least one non-trivial amplitude"
        );
    }

    #[test]
    fn test_fourier_scaling_regression() {
        // Run on a few small primes and check that fitted slope is negative.
        let primes = vec![101, 251, 503, 1009];
        let analysis = run_fourier_scaling(&primes, 5);
        assert_eq!(analysis.results.len(), 4);
        assert!(
            analysis.fitted_slope < 0.0,
            "slope should be negative (amplitudes decay), got {}",
            analysis.fitted_slope
        );
        assert!(
            analysis.fitted_slope > -1.0,
            "slope should be > -1 (not decaying faster than 1/ℓ), got {}",
            analysis.fitted_slope
        );
    }

    #[test]
    fn test_scaling_primes() {
        let primes = scaling_primes(100, 10000, 8);
        assert!(primes.len() >= 6, "should generate several primes");
        // All should be prime
        for &p in &primes {
            assert!(is_prime_u64(p), "{p} is not prime");
        }
        // Should be sorted ascending
        for w in primes.windows(2) {
            assert!(w[0] < w[1], "primes not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_is_b_smooth() {
        // 1 is trivially smooth.
        assert!(is_b_smooth(1, 2));
        // 2 = 2 is 2-smooth.
        assert!(is_b_smooth(2, 2));
        // 12 = 2² · 3 is 3-smooth.
        assert!(is_b_smooth(12, 3));
        // 12 is NOT 2-smooth (has factor 3).
        assert!(!is_b_smooth(12, 2));
        // 30 = 2 · 3 · 5 is 5-smooth.
        assert!(is_b_smooth(30, 5));
        // 30 is NOT 3-smooth.
        assert!(!is_b_smooth(30, 3));
        // 17 (prime) is 17-smooth but not 16-smooth.
        assert!(is_b_smooth(17, 17));
        assert!(!is_b_smooth(17, 16));
        // 100 = 2² · 5² is 5-smooth.
        assert!(is_b_smooth(100, 5));
        // 210 = 2 · 3 · 5 · 7 is 7-smooth.
        assert!(is_b_smooth(210, 7));
        assert!(!is_b_smooth(210, 5));
    }

    #[test]
    fn test_dft_on_cyclic_group_constant() {
        // A constant function should have all Fourier coefficients = 0
        // (after centering, the function is identically zero).
        let f = vec![1.0; 100];
        let (dc, amps) = dft_on_cyclic_group(&f, 5);
        assert!((dc - 1.0).abs() < 1e-10, "DC should be 1.0, got {dc}");
        for &(_, amp) in &amps {
            assert!(amp < 1e-10, "constant function should have zero Fourier coefficients");
        }
    }

    #[test]
    fn test_dft_on_cyclic_group_pure_cosine() {
        // f(k) = cos(2π·3·k/n) — should have a single peak at r=3.
        let n = 64;
        let f: Vec<f64> = (0..n)
            .map(|k| (std::f64::consts::TAU * 3.0 * k as f64 / n as f64).cos())
            .collect();
        let (dc, amps) = dft_on_cyclic_group(&f, 5);
        assert!(dc.abs() < 1e-10, "cosine has zero mean");
        // The top amplitude should be at r=3.
        let (top_r, top_amp) = amps[0];
        assert_eq!(top_r, 3, "peak should be at r=3, got r={top_r}");
        assert!(top_amp > 0.4, "amplitude at r=3 should be large, got {top_amp}");
        // All other amplitudes should be much smaller.
        if amps.len() > 1 {
            assert!(
                amps[1].1 < top_amp * 0.01,
                "second amplitude should be much smaller: {:.6} vs {:.6}",
                amps[1].1,
                top_amp,
            );
        }
    }

    #[test]
    fn test_smoothness_spectrum_basic() {
        // For ℓ = 131, B = 10: the smoothness spectrum should produce finite values.
        let result = smoothness_spectrum(131, 10, 5);
        assert_eq!(result.ell, 131);
        assert_eq!(result.order, 130);
        assert_eq!(result.smoothness_bound, 10);
        // smooth_fraction should be between 0 and 1.
        assert!(
            result.smooth_fraction > 0.0 && result.smooth_fraction < 1.0,
            "smooth_fraction should be in (0,1), got {}",
            result.smooth_fraction,
        );
        // A_max should be positive and finite.
        assert!(
            result.a_max > 0.0 && result.a_max.is_finite(),
            "a_max should be positive finite, got {}",
            result.a_max,
        );
        // Parity baseline should also be positive.
        assert!(result.parity_a_max > 0.0);
        // ratio_to_parity should be positive.
        assert!(result.ratio_to_parity > 0.0);
    }

    #[test]
    fn test_smoothness_spectrum_large_b_approaches_constant() {
        // When B ≥ ℓ, ALL elements in {1,…,ℓ-1} are B-smooth (since ℓ-1 < ℓ
        // and we only consider elements up to ℓ-1).
        // So smooth_fraction ≈ 1.0 and the DFT of a (nearly) constant function
        // should have very small Fourier coefficients.
        let result = smoothness_spectrum(131, 200, 5);
        assert!(
            result.smooth_fraction > 0.95,
            "with B ≥ ℓ, almost all should be smooth, got {}",
            result.smooth_fraction,
        );
        // A_max should be very small (constant function → flat spectrum).
        assert!(
            result.a_max < 0.05,
            "constant-like function should have small A_max, got {}",
            result.a_max,
        );
    }

    #[test]
    fn test_run_smoothness_scaling_smoke() {
        let primes = vec![101, 251, 503];
        let analysis = run_smoothness_scaling(&primes, 10, 5);
        assert_eq!(analysis.results.len(), 3);
        assert_eq!(analysis.smoothness_bound, 10);
        // Slope should be negative (amplitudes decay with ℓ).
        assert!(
            analysis.fitted_slope < 0.0,
            "slope should be negative, got {}",
            analysis.fitted_slope,
        );
        // Parity slope should also be negative.
        assert!(analysis.parity_slope < 0.0);
    }

    #[test]
    fn test_prime_restricted_smoothness_smoke() {
        // Smoke test: 14-bit, channel 5 (k=22, ℓ=131), B=10.
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_prime_restricted_smoothness(14, ch, 10, 5);
        assert_eq!(result.ell, 131);
        assert_eq!(result.smoothness_bound, 10);
        assert!(result.n_primes > 0, "should find primes");
        assert!(result.n_valid > 0, "should have valid primes");
        // Full-group should have finite values.
        assert!(result.full_a_max > 0.0 && result.full_a_max.is_finite());
        assert!(result.full_excess > 0.0 && result.full_excess.is_finite());
        // Restricted amplitudes should be finite and non-negative.
        assert!(result.fixed_r_amplitude >= 0.0 && result.fixed_r_amplitude.is_finite());
        assert!(result.scanned_amplitude >= 0.0 && result.scanned_amplitude.is_finite());
        // Product correlations should be in [-1, 1].
        assert!(result.product_corr_sigma.abs() <= 1.0 + 1e-9);
        assert!(result.product_corr_nk.abs() <= 1.0 + 1e-9);
    }

    #[test]
    fn test_prime_restricted_smoothness_barrier_expected() {
        // For 16-bit, channel 0 (k=12, ℓ=691), B=30:
        // The product test B (corr_Nk) should be near zero — barrier intact.
        let ch = &CHANNELS[0]; // k=12, ℓ=691
        let result = run_prime_restricted_smoothness(16, ch, 30, 5);
        assert!(result.n_pairs > 10, "need enough pairs for meaningful test");
        assert!(
            result.product_corr_nk.abs() < 0.5,
            "product test B should be near zero (barrier), got {:.4}",
            result.product_corr_nk,
        );
    }

    #[test]
    fn test_sample_primes_in_range() {
        let primes = sample_primes_in_range(1000, 2000, 50, 42);
        assert_eq!(primes.len(), 50, "should sample exactly 50 primes");
        for &p in &primes {
            assert!(p >= 1000 && p < 2000, "prime {p} out of range");
            assert!(is_prime_u64(p), "{p} is not prime");
        }
        // Should be sorted (BTreeSet output).
        for w in primes.windows(2) {
            assert!(w[0] < w[1], "not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_generate_primes_and_pairs_large() {
        // Test that sampling works for n=28 bits.
        let (primes, pairs) = generate_primes_and_pairs(28, 100, 5000, 0xBEEF);
        assert!(primes.len() >= 80, "should sample many primes, got {}", primes.len());
        assert!(!pairs.is_empty(), "should form some pairs");
        // Verify pair properties.
        for &(p, q) in pairs.iter().take(20) {
            assert!(is_prime_u64(p) && is_prime_u64(q));
            let n = p as u128 * q as u128;
            assert!(n >= (1u128 << 27) && n < (1u128 << 28),
                "product {n} not in 28-bit range");
            assert!((p.min(q) as f64) / (p.max(q) as f64) >= 0.3);
        }
    }

    #[test]
    fn test_prime_restricted_smoothness_large_n() {
        // Smoke test: 28-bit, channel 5 (k=22, ℓ=131), B=10.
        // Uses sampling, should still produce valid results.
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_prime_restricted_smoothness(28, ch, 10, 5);
        assert_eq!(result.ell, 131);
        assert!(result.n_primes > 50, "should have many sampled primes");
        assert!(result.n_valid > 50, "should have many valid primes");
        assert!(result.fixed_r_amplitude.is_finite());
        assert!(result.product_corr_nk.abs() <= 1.0 + 1e-9);
    }

    // ─── Stress test unit tests ───

    #[test]
    fn test_smoothness_spectrum_full_matches_truncated() {
        let ell = 131u64;
        let bound = 10u64;
        let truncated = smoothness_spectrum(ell, bound, 5);
        let full = smoothness_spectrum_full(ell, bound);
        assert_eq!(full.order, truncated.order);
        assert!(
            (full.smooth_fraction - truncated.smooth_fraction).abs() < 1e-10,
            "smooth fractions differ: {} vs {}",
            full.smooth_fraction,
            truncated.smooth_fraction,
        );
        let full_a_max = full
            .coefficients
            .iter()
            .map(|&(_, _, _, amp)| amp)
            .fold(0.0f64, f64::max);
        assert!(
            (full_a_max - truncated.a_max).abs() < 1e-10,
            "full a_max {} != truncated a_max {}",
            full_a_max,
            truncated.a_max,
        );
    }

    #[test]
    fn test_permutation_null_smoke() {
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_permutation_null(14, ch, 10, 5, 50, 0xBEEF);
        assert!(result.null_std >= 0.0);
        assert!(
            result.empirical_p_value >= 0.0 && result.empirical_p_value <= 1.0,
            "p-value out of range: {}",
            result.empirical_p_value,
        );
        assert!(result.z_score.is_finite());
        assert!(
            result.z_score.abs() < 5.0,
            "z_score should be moderate (barrier holds), got {}",
            result.z_score,
        );
    }

    #[test]
    fn test_cross_n_transfer_smoke() {
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_cross_n_transfer(ch, 10, 5, &[14, 16, 18, 20], 20);
        assert_eq!(result.source_n_bits, 20);
        assert!(!result.entries.is_empty());
        for entry in &result.entries {
            assert!(entry.consistency_ratio.is_finite());
            assert!(entry.local_fix_excess >= 0.0);
            assert!(entry.transfer_fix_excess >= 0.0);
        }
    }

    #[test]
    fn test_multi_character_score_smoke() {
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_multi_character_score(14, ch, 10, 5, 0xCAFE);
        assert!(result.corr_dft_weighted_all.abs() <= 1.0 + 1e-9);
        assert!(result.corr_dft_weighted_top10.abs() <= 1.0 + 1e-9);
        assert!(result.corr_direct_smoothness.abs() <= 1.0 + 1e-9);
        assert!(result.train_test_r_squared.is_finite());
        // Verify 3a ≈ 3c (algebraic identity: DFT-weighted all = direct smoothness).
        assert!(
            (result.corr_dft_weighted_all - result.corr_direct_smoothness).abs() < 0.01,
            "DFT-weighted all should equal direct smoothness: {} vs {}",
            result.corr_dft_weighted_all,
            result.corr_direct_smoothness,
        );
    }

    #[test]
    fn test_bootstrap_ci_smoke() {
        let ch = &CHANNELS[5]; // k=22, ℓ=131
        let result = run_bootstrap_ci(14, ch, 10, 5, 100, 0xF00D);
        assert!(result.fix_excess_ci_lo <= result.fix_excess_ci_hi);
        assert!(result.corr_nk_ci_lo <= result.corr_nk_ci_hi);
        assert!(result.fix_excess_mean.is_finite());
        assert!(result.corr_nk_mean.is_finite());
        assert!(result.fix_excess_std >= 0.0);
        assert!(result.corr_nk_std >= 0.0);
    }

    #[test]
    fn test_full_group_control_small() {
        // For ℓ = 131 (order = 130), the full group matrix is 130×130.
        // The SECOND eigenvector (index 1) should match a non-trivial character
        // with amp near 1.0 (top eigenvector is the constant mode / r=0).
        let result = run_full_group_control(131, 3, 42);
        assert_eq!(result.order, 130);
        assert!(result.eigenvectors.len() >= 2);

        // The second eigenvector (after sorting by |eigenvalue|) should have
        // character amplitude near 1.0 — characters ARE exact eigenvectors
        // over the full group.
        let ev1 = &result.eigenvectors[1]; // 2nd-largest eigenvalue
        assert!(
            ev1.best_char_amp > 0.90,
            "full-group control: 2nd eigenvector should match a character, got amp={:.3}",
            ev1.best_char_amp,
        );
    }

    // -----------------------------------------------------------------------
    // E21c: OLS solver tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cholesky_identity() {
        // Cholesky of I₃ should be I₃.
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let l = cholesky(&a, 3).expect("Cholesky should succeed on identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (l[i][j] - expected).abs() < 1e-12,
                    "L[{i}][{j}] = {}, expected {expected}",
                    l[i][j]
                );
            }
        }
    }

    #[test]
    fn test_ols_solve_exact() {
        // y = 2·x₁ + 3·x₂ exactly on 4 data points.
        let x_rows = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let y: Vec<f64> = x_rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1]).collect();
        let beta = ols_solve(&x_rows, &y).expect("OLS should succeed");
        assert!(
            (beta[0] - 2.0).abs() < 1e-4,
            "beta[0] = {}, expected 2.0",
            beta[0]
        );
        assert!(
            (beta[1] - 3.0).abs() < 1e-4,
            "beta[1] = {}, expected 3.0",
            beta[1]
        );
        let r2 = ols_r_squared(&x_rows, &y, &beta);
        assert!(
            r2 > 0.999,
            "R² should be ≈ 1 for exact fit, got {r2}"
        );
    }

    // -----------------------------------------------------------------------
    // E21c: Binned MI test
    // -----------------------------------------------------------------------

    #[test]
    fn test_n_only_features_shape() {
        // Verify that compute_n_only_features returns 14 features per pair.
        let block = prepare_cross_channel_block(14, 10, 5);
        let features = compute_n_only_features(&block);
        assert_eq!(features.len(), block.valid_indices.len());
        for feats in &features {
            assert_eq!(feats.len(), 14, "expected 14 features (2 per channel × 7 channels)");
        }
        // Check features are finite and in [-1, 1].
        for feats in &features {
            for &f in feats {
                assert!(f.is_finite(), "feature is not finite: {f}");
                assert!(f.abs() <= 1.0 + 1e-10, "feature out of range: {f}");
            }
        }
    }

    #[test]
    fn test_cross_channel_smoke() {
        // Full pipeline at n=14, B=10, 20 permutations.
        let result = run_cross_channel_tests(&[14], &[10], 5, 20, 5, 0xE21C);
        assert_eq!(result.blocks.len(), 1);
        let b = &result.blocks[0];
        assert_eq!(b.n_bits, 14);
        assert_eq!(b.smoothness_bound, 10);
        assert!(b.n_pairs > 0, "should have valid pairs");

        // C1: max |corr| should be finite.
        assert!(b.pairwise.max_abs_corr.is_finite());
        assert!(b.pairwise.n_tests == 84);

        // C2: R² should be finite (may be negative for noise-only).
        assert!(b.ols.test_r_squared.is_finite());
        assert_eq!(b.ols.n_features, 35);

        // C4: permutation null should have valid z-score.
        assert!(b.perm_null.z_score.is_finite());
        assert!(b.perm_null.empirical_p_value >= 0.0);
        assert!(b.perm_null.empirical_p_value <= 1.0);
    }

    #[test]
    fn test_binned_mi_independent_is_near_zero() {
        // Target and bins are independent: MI should be ≈ 0.
        // Construct 200 data points with uniform bins and random-ish target.
        let n = 200;
        let n_bins = 5;
        let bins_i: Vec<usize> = (0..n).map(|i| i % n_bins).collect();
        let bins_j: Vec<usize> = (0..n).map(|i| (i * 3 + 1) % n_bins).collect();
        // Target alternates 0/1 in a pattern unrelated to bin assignment.
        let target: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mi = binned_mutual_information(&target, &bins_i, &bins_j, n_bins);
        assert!(
            mi.abs() < 0.05,
            "MI for independent bins/target should be near 0, got {mi}"
        );
    }

    // -----------------------------------------------------------------------
    // E22: Sieve enrichment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(15), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(100), 10);
        // Large value: 10^18
        let big = 1_000_000_000_000_000_000u128;
        assert_eq!(isqrt(big), 1_000_000_000);
        // Verify: s² ≤ n < (s+1)²
        for n in [2u128, 3, 7, 10, 99, 101, 1000, 999999] {
            let s = isqrt(n);
            assert!(s * s <= n, "isqrt({n}) = {s}: s² > n");
            assert!((s + 1) * (s + 1) > n, "isqrt({n}) = {s}: (s+1)² ≤ n");
        }
    }

    #[test]
    fn test_is_b_smooth_u128() {
        assert!(is_b_smooth_u128(1, 2));
        assert!(is_b_smooth_u128(12, 3)); // 2² · 3
        assert!(!is_b_smooth_u128(12, 2));
        assert!(is_b_smooth_u128(30, 5)); // 2 · 3 · 5
        assert!(!is_b_smooth_u128(30, 3));
        // Large smooth number: 2^40 = 1099511627776
        assert!(is_b_smooth_u128(1u128 << 40, 2));
        // Large non-smooth: 2^40 + 1 (has large prime factor)
        assert!(!is_b_smooth_u128((1u128 << 40) + 1, 100));
    }

    #[test]
    fn test_generate_semiprime() {
        for &n_bits in &[20u32, 24, 28, 32] {
            let n = generate_semiprime(n_bits, 0xE220_7E57 + n_bits as u64);
            let actual_bits = 128 - n.leading_zeros();
            assert_eq!(
                actual_bits, n_bits,
                "generate_semiprime({n_bits}) produced {actual_bits}-bit number"
            );
        }
    }

    #[test]
    fn test_generate_qs_values() {
        // N = 15 (3 × 5), sqrt(15) ≈ 3.87, floor = 3
        // Q(x) = (x+3)² - 15
        // Q(1) = 16 - 15 = 1
        // Q(2) = 25 - 15 = 10
        // Q(3) = 36 - 15 = 21
        let qs = generate_qs_values(15, 5);
        assert!(!qs.is_empty(), "should generate QS values");
        // All Q(x) should be > 0.
        for &(_, qx) in &qs {
            assert!(qx > 0, "Q(x) should be positive");
        }
    }

    #[test]
    fn test_group_enrichment_smoke() {
        // Small ℓ = 131, B = 10.
        let result = compute_group_enrichment(131, 22, 10, 10);
        assert!(result.smooth_fraction > 0.0, "some elements should be smooth");
        assert!(result.smooth_fraction < 1.0, "not all elements are smooth");
        assert!(result.full_r_star > 0, "r* should be > 0");
        assert!(
            result.enrichment_top_quartile >= 0.0,
            "enrichment should be non-negative"
        );
        assert_eq!(result.bin_enrichments.len(), 10);
        // Sum of bin sizes should cover the group.
        assert!(result.bin_enrichments.iter().all(|&(_, f, _)| f >= 0.0 && f <= 1.0));
    }

    #[test]
    fn test_qs_block_smoke() {
        // Small: 20-bit N, B=10, 100 QS values.
        let result = compute_qs_block(20, 10, 100, 0xE220_5000);
        assert_eq!(result.n_bits, 20);
        assert!(result.n_values > 0);
        assert_eq!(result.channels.len(), 7);
        for ch in &result.channels {
            assert!(ch.n_tested > 0, "should have tested values for ℓ={}", ch.ell);
            assert!(ch.avg_log2_qx > 0.0);
            assert!(ch.overflow_ratio > 0.0);
            assert!(ch.enrichment_top_q.is_finite());
        }
    }

    #[test]
    fn test_sieve_enrichment_full_smoke() {
        // Minimal run: 1 bit size, 1 bound, few values.
        let result = run_sieve_enrichment(
            &[20],
            &[10],
            50,   // n_qs_values
            100,  // n_pool
            5,    // target_smooth
            5,    // n_bins
            0xE220_F000,
            None,
        );
        assert_eq!(result.group_profiles.len(), 7); // 7 channels × 1 bound
        assert_eq!(result.qs_blocks.len(), 1);      // 1 bit size × 1 bound
        assert_eq!(result.joint_scoring.len(), 1);
        assert_eq!(result.sieve_speedup.len(), 1);
        assert!(result.sieve_speedup[0].speedup_factor.is_finite());
    }

    // -----------------------------------------------------------------------
    // E23: Local smoothness helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tonelli_shanks() {
        // sqrt(2) mod 7: 3² = 9 ≡ 2 (mod 7), so sqrt(2) = 3 or 4.
        let r = tonelli_shanks(2, 7).expect("2 is QR mod 7");
        assert!(r == 3 || r == 4, "sqrt(2) mod 7 should be 3 or 4, got {r}");
        assert_eq!(mul_mod_u64(r, r, 7), 2);

        // sqrt(4) mod 13: 2² = 4, so sqrt(4) = 2 or 11.
        let r = tonelli_shanks(4, 13).expect("4 is QR mod 13");
        assert_eq!(mul_mod_u64(r, r, 13), 4);

        // 3 is NQR mod 7.
        assert!(tonelli_shanks(3, 7).is_none(), "3 is NQR mod 7");

        // sqrt(0) mod 13 = 0.
        assert_eq!(tonelli_shanks(0, 13), Some(0));

        // p = 2 edge case.
        assert_eq!(tonelli_shanks(1, 2), Some(1));
        assert_eq!(tonelli_shanks(0, 2), Some(0));

        // 5 is NQR mod 997 (Legendre symbol = -1).
        assert!(tonelli_shanks(5, 997).is_none(), "5 is NQR mod 997");

        // Large prime: p = 997 (≡ 1 mod 4, needs full Tonelli-Shanks).
        // 100² mod 997 = 30, so 30 is a QR mod 997.
        let r = tonelli_shanks(30, 997).expect("30 is QR mod 997");
        assert_eq!(mul_mod_u64(r, r, 997), 30);
    }

    #[test]
    fn test_qs_roots_mod_p() {
        // N = 15 (3 × 5). Q(x) = (x+3)² - 15.
        // Q(x) ≡ 0 (mod 7) means (x+3)² ≡ 15 ≡ 1 (mod 7), so x+3 ≡ ±1 (mod 7).
        // x ≡ -2 ≡ 5 (mod 7) or x ≡ -4 ≡ 3 (mod 7).
        let roots = qs_roots_mod_p(15, 7);
        assert_eq!(roots.len(), 2, "should have 2 roots mod 7");
        // Verify each root: (root + 3)² ≡ 15 (mod 7)
        for &r in &roots {
            let val = mul_mod_u64((r + 3) % 7, (r + 3) % 7, 7);
            assert_eq!(val, 15 % 7, "root {r} doesn't satisfy Q(x) ≡ 0 (mod 7)");
        }
    }

    #[test]
    fn test_partial_smooth_fraction() {
        // 12 = 2² · 3, B=3 → fully smooth → 1.0
        let f = partial_smooth_fraction(12, 3);
        assert!((f - 1.0).abs() < 1e-10, "12 is 3-smooth, got {f}");

        // 14 = 2 · 7, B=5 → only factor 2 is ≤ 5 → ln(2)/ln(14)
        let f = partial_smooth_fraction(14, 5);
        let expected = (2.0f64).ln() / (14.0f64).ln();
        assert!(
            (f - expected).abs() < 1e-6,
            "partial_smooth_fraction(14, 5) = {f}, expected {expected}"
        );

        // 7, B=5 → no small factors → 0.0
        let f = partial_smooth_fraction(7, 5);
        assert!((f - 0.0).abs() < 1e-10, "7 has no factors ≤ 5, got {f}");

        // 1 → 1.0 (trivially smooth)
        assert!((partial_smooth_fraction(1, 2) - 1.0).abs() < 1e-10);

        // 30 = 2·3·5, B=5 → fully smooth → 1.0
        let f = partial_smooth_fraction(30, 5);
        assert!((f - 1.0).abs() < 1e-10, "30 is 5-smooth, got {f}");

        // 210 = 2·3·5·7, B=5 → partial: ln(2·3·5)/ln(210) = ln(30)/ln(210)
        let f = partial_smooth_fraction(210, 5);
        let expected = (30.0f64).ln() / (210.0f64).ln();
        assert!(
            (f - expected).abs() < 1e-6,
            "partial_smooth_fraction(210, 5) = {f}, expected {expected}"
        );
    }

    #[test]
    fn test_autocorrelation_smoke() {
        // Small N (20-bit), B=30, pool=500. Just check it runs and returns sane values.
        let n = generate_semiprime(20, 0xE230_0001);
        let qs_pairs = generate_qs_values(n, 500);
        let qs: Vec<u128> = qs_pairs.iter().map(|&(_, v)| v).collect();
        let result = compute_smoothness_autocorrelation(&qs, 30, 10, 20);

        assert_eq!(result.lags.len(), 10);
        assert!(result.overall_smooth_rate > 0.0, "should have some smooth values");
        // C(δ) should be positive for all lags.
        for lag in &result.lags {
            assert!(lag.c_delta >= 0.0, "C({}) = {} should be non-negative", lag.delta, lag.c_delta);
        }
    }

    #[test]
    fn test_partial_frac_autocorrelation_finite() {
        let n = generate_semiprime(20, 0xE230_0002);
        let qs_pairs = generate_qs_values(n, 500);
        let qs: Vec<u128> = qs_pairs.iter().map(|&(_, v)| v).collect();
        let result = compute_partial_frac_autocorrelation(&qs, 30, 10, 20);

        assert_eq!(result.lag_correlations.len(), 10);
        assert!(result.mean_partial_frac > 0.0, "mean partial frac should be positive");
        // All Pearson correlations should be finite.
        for &(delta, r) in &result.lag_correlations {
            assert!(r.is_finite(), "r({delta}) is not finite");
            assert!(r.abs() <= 1.0 + 1e-6, "r({delta}) = {r} out of [-1,1]");
        }
    }

    #[test]
    fn test_random_control_near_one() {
        // Use larger B bound so random values have a chance of being smooth.
        let n = generate_semiprime(20, 0xE230_0004);
        let result = compute_random_control(n, 500, 500, 10, 0xE230_C000, 20);

        // Verify structure: correct number of lags.
        assert_eq!(result.lags.len(), 10);
        assert_eq!(result.lag_correlations.len(), 10);

        // For random data with large enough smooth bound:
        // C(δ) should be near 1.0 IF there are enough smooth values.
        for lag in &result.lags {
            if lag.n_base_smooth >= 10 {
                assert!(
                    lag.c_delta > 0.2 && lag.c_delta < 5.0,
                    "Random C({}) = {} far from 1.0",
                    lag.delta,
                    lag.c_delta,
                );
            }
        }
        // Partial-frac correlations should be near 0 (no structure in random data).
        for &(delta, r) in &result.lag_correlations {
            assert!(r.is_finite(), "r({delta}) not finite");
            assert!(
                r.abs() < 0.5,
                "Random partial-frac r({delta}) = {r} too far from 0",
            );
        }
    }

    #[test]
    fn test_local_block_smoke() {
        // Quick smoke test: 20-bit, B=30, pool=200, max_lag=5.
        let block = compute_local_block(20, 30, 200, 5, 0xE230_B000);

        assert_eq!(block.n_bits, 20);
        assert_eq!(block.smooth_bound, 30);
        assert_eq!(block.phase1.lags.len(), 5);
        assert_eq!(block.phase2.lag_correlations.len(), 5);
        assert_eq!(block.phase3.lag_comparisons.len(), 5);
        assert_eq!(block.phase4.lags.len(), 5);

        // Phase 1: smooth rate should be positive for 20-bit with B=30.
        assert!(block.phase1.overall_smooth_rate > 0.0);

        // Phase 2: all correlations should be finite.
        for &(_, r) in &block.phase2.lag_correlations {
            assert!(r.is_finite());
        }

        // Phase 3: cofactor correlations should be finite.
        for &(_, _, cf_corr, _) in &block.phase3.lag_comparisons {
            assert!(cf_corr.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // E24 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rational_norm() {
        // |3 + 2*10| = 23
        assert_eq!(rational_norm_u128(3, 2, 10), 23);
        // |-5 + 1*3| = |-5+3| = 2
        assert_eq!(rational_norm_u128(-5, 1, 3), 2);
        // |0 + 1*7| = 7  — but a=0 is excluded from grid, test the function itself
        assert_eq!(rational_norm_u128(0, 1, 7), 7);
    }

    #[test]
    fn test_biguint_to_u128_clamped() {
        assert_eq!(biguint_to_u128_clamped(&BigUint::from(42u64)), 42);
        assert_eq!(biguint_to_u128_clamped(&BigUint::from(u128::MAX)), u128::MAX);
        // A value > u128::MAX should clamp
        let huge = BigUint::from(u128::MAX) + BigUint::from(1u64);
        assert_eq!(biguint_to_u128_clamped(&huge), u128::MAX);
    }

    #[test]
    fn test_standard_displacements() {
        let disps = standard_displacements();
        assert_eq!(disps.len(), 8);
        assert_eq!(disps[0].da, 1);
        assert_eq!(disps[0].db, 0);
        assert_eq!(disps[1].da, 0);
        assert_eq!(disps[1].db, 1);
    }

    #[test]
    fn test_generate_nfs_grid() {
        // Small grid: 40-bit N, area=10, max_b=3
        let (poly, grid) = generate_nfs_grid(40, 0xE240_0001, 10, 3);

        // Polynomial should be degree 3 for 40-bit N
        assert_eq!(poly.degree, 3);

        // Grid should have coprime (a,b) pairs with non-trivial norms
        assert!(!grid.is_empty(), "grid must be non-empty");

        // All entries should have a != 0
        for &(a, b, alg, rat) in &grid {
            assert_ne!(a, 0);
            assert!(b >= 1);
            assert!(alg > 1);
            assert!(rat > 1);
            // coprimality
            assert_eq!(gcd_u64(a.unsigned_abs(), b as u64), 1);
        }
    }

    #[test]
    fn test_grid_index_lookup() {
        let (_, grid) = generate_nfs_grid(40, 0xE240_0002, 5, 2);
        let index = build_grid_index(&grid);

        // Every grid entry should be in the index
        for (i, &(a, b, _, _)) in grid.iter().enumerate() {
            assert_eq!(index[&(a, b)], i);
        }

        // Non-existent point should not be in index
        assert!(!index.contains_key(&(9999, 9999)));
    }

    #[test]
    fn test_nfs_2d_autocorrelation_smoke() {
        // Small NFS grid, verify Phase 1 produces meaningful output.
        let (_, grid) = generate_nfs_grid(40, 0xE240_1000, 20, 5);
        let index = build_grid_index(&grid);
        let smooth_bound: u64 = 500;

        let smooth_flags: Vec<bool> = grid.iter()
            .map(|&(_, _, alg, _)| is_b_smooth_u128(alg, smooth_bound))
            .collect();

        let result = compute_nfs_2d_autocorrelation(
            &grid, &index, &smooth_flags, smooth_bound, 40,
        );

        assert_eq!(result.lags.len(), 8); // 8 standard displacements
        assert!(result.overall_smooth_rate >= 0.0);
        assert!(result.n_grid > 0);

        // All C(δ) should be non-negative
        for lag in &result.lags {
            assert!(lag.c_delta >= 0.0, "C({}) should be non-negative", lag.label);
        }
    }

    #[test]
    fn test_nfs_2d_partial_frac_finite() {
        let (_, grid) = generate_nfs_grid(40, 0xE240_2000, 15, 3);
        let index = build_grid_index(&grid);
        let smooth_bound: u64 = 200;

        let partial_fracs: Vec<f64> = grid.iter()
            .map(|&(_, _, alg, _)| partial_smooth_fraction(alg, smooth_bound))
            .collect();

        let result = compute_nfs_2d_partial_frac(
            &grid, &index, &partial_fracs, smooth_bound, 40,
        );

        assert_eq!(result.displacement_correlations.len(), 8);
        // All correlations should be finite
        for &(ref _label, r) in &result.displacement_correlations {
            assert!(r.is_finite(), "Pearson correlation must be finite");
        }
        // Mean partial fraction should be in (0, 1]
        assert!(result.mean_partial_frac > 0.0);
        assert!(result.mean_partial_frac <= 1.0);
    }

    #[test]
    fn test_nfs_2d_random_control() {
        let (_, grid) = generate_nfs_grid(40, 0xE240_4000, 20, 5);
        let index = build_grid_index(&grid);
        let smooth_bound: u64 = 500;

        let result = compute_nfs_2d_random_control(
            &grid, &index, smooth_bound, 40, 0xE240_4001,
        );

        assert_eq!(result.lags.len(), 8);
        // Random control: all C(δ) should be defined (though may be 0 if few smooth)
        for lag in &result.lags {
            assert!(lag.c_delta >= 0.0 || lag.c_delta == 0.0);
        }
    }

    #[test]
    fn test_nfs_block_smoke() {
        // Quick smoke test: 40-bit, B=200, small area, verify all 5 phases run.
        let block = compute_nfs_block(40, 200, 15, 3, 0xE240_B000);

        assert_eq!(block.n_bits, 40);
        assert_eq!(block.smooth_bound, 200);
        assert!(block.n_grid > 0);

        // Phase 1: 8 displacement lags
        assert_eq!(block.phase1.lags.len(), 8);
        assert!(block.phase1.overall_smooth_rate >= 0.0);

        // Phase 2: 8 displacement correlations, all finite
        assert_eq!(block.phase2.displacement_correlations.len(), 8);
        for &(ref _label, r) in &block.phase2.displacement_correlations {
            assert!(r.is_finite());
        }

        // Phase 3: 8 displacement comparisons, all finite
        assert_eq!(block.phase3.displacement_comparisons.len(), 8);
        for &(ref _label, pf, cf, _rr) in &block.phase3.displacement_comparisons {
            assert!(pf.is_finite());
            assert!(cf.is_finite());
        }

        // Phase 4: random control has 8 lags
        assert_eq!(block.phase4.lags.len(), 8);

        // Phase 5: dual-norm correlation is finite
        assert!(block.phase5.rational_alg_cofactor_corr.is_finite());
        assert!(block.phase5.rational_mean_partial >= 0.0);
        assert!(block.phase5.algebraic_mean_partial >= 0.0);
    }

    #[test]
    fn test_simple_ols() {
        // Perfect linear: y = 2 + 3x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];
        let (intercept, slope, r_sq) = simple_ols(&x, &y);
        assert!((slope - 3.0).abs() < 1e-10);
        assert!((intercept - 2.0).abs() < 1e-10);
        assert!((r_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nfs_validation_block_smoke() {
        // Quick validation smoke: small grid, verify all 5 controls run.
        let block = compute_nfs_validation_block(40, 200, 15, 3, 0xE24B_0000);

        // Control A: norm-residualized
        assert!(block.control_a.norm_r_squared >= 0.0);
        assert_eq!(block.control_a.displacement_comparisons.len(), 8);
        for &(ref _label, raw, resid) in &block.control_a.displacement_comparisons {
            assert!(raw.is_finite());
            assert!(resid.is_finite());
        }

        // Control B: per-side
        assert_eq!(block.control_b.displacement_comparisons.len(), 8);

        // Control C: magnitude-bin shuffle
        assert_eq!(block.control_c.displacement_comparisons.len(), 8);
        for &(ref _label, orig, shuf_mean, shuf_std) in &block.control_c.displacement_comparisons {
            assert!(orig.is_finite());
            assert!(shuf_mean.is_finite());
            assert!(shuf_std.is_finite());
        }

        // Control D: displacement decay
        assert!(!block.control_d.decay_by_radius.is_empty());

        // Control E: conditional cofactor
        assert!(block.control_e.conditional_cf_corr.is_finite());
    }
}
