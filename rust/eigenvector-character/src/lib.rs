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
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

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
}
