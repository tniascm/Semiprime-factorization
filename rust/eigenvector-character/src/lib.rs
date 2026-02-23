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
    g_vals: Vec<u64>,
    dlogs_g: Vec<u32>,
    smooth_vals: Vec<f64>,
    scanned_r_star: usize,
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
}
