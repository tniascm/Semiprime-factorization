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

    // 2. Enumerate primes from balanced semiprimes.
    let pairs = enumerate_balanced_semiprimes(n_bits);
    let mut prime_set: Vec<u64> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
    prime_set.sort_unstable();
    prime_set.dedup();
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
