/// Boolean polynomial degree measurement for functions over balanced semiprimes.
///
/// Two complementary measurements:
///
/// 1. **Correlation lower bound**: for each degree d, sample random degree-d
///    monomials and measure their correlation with f(N) = target(N) mod 2.
///    If any monomial has significant correlation, deg(f) ≥ d.
///
/// 2. **Min fitting degree** (for d ≤ 2): build the full monomial feature matrix
///    over F2 and check via Gaussian elimination whether a degree-d polynomial fits.
///
/// 3. **CRT rank** (for small n): build the full prime×prime matrix and find rank.

use eisenstein_hunt::{
    arith::is_prime_u64,
    Channel, Semiprime,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of the correlation lower-bound test for a single (n_bits, degree, channel).
#[derive(Debug, Clone, Serialize)]
pub struct CorrelationResult {
    pub n_bits: u32,
    pub degree: u32,
    /// Quantile used for the calibrated null threshold (0.0 = fixed 3/√m).
    pub null_quantile: f64,
    pub channel_weight: u32,
    pub channel_ell: u64,
    /// Maximum |correlation| found over sampled monomials (in ±1 encoding).
    pub max_abs_corr: f64,
    /// Number of monomials sampled.
    pub n_monomials: usize,
    /// Number of semiprimes used.
    pub n_samples: usize,
    /// Significance threshold at this sample size (3 / sqrt(n_samples)).
    pub threshold: f64,
    /// Whether max_abs_corr exceeds the threshold.
    pub significant: bool,
}

/// Result of the CRT rank estimation for a single (n_bits, channel).
#[derive(Debug, Clone, Serialize)]
pub struct CrtRankResult {
    pub n_bits: u32,
    pub channel_weight: u32,
    pub channel_ell: u64,
    /// Number of distinct primes found in the p-range.
    pub n_primes: usize,
    /// Total semiprime entries in the matrix.
    pub matrix_entries: usize,
    /// Rank of the matrix over F2.
    pub rank: usize,
    /// Fraction rank / n_primes (1.0 = full rank).
    pub rank_fraction: f64,
}

/// Result of the minimum fitting degree test (d ≤ 2) for a single (n_bits, channel).
#[derive(Debug, Clone, Serialize)]
pub struct FitDegreeResult {
    pub n_bits: u32,
    pub channel_weight: u32,
    pub channel_ell: u64,
    pub n_samples: usize,
    /// Minimum d in {0,1,2} for which a degree-d polynomial over F2 fits all samples,
    /// or None if even degree-2 doesn't fit.
    pub min_fitting_degree: Option<u32>,
}

/// Real-valued spectral analysis of M[p][q] = f_{k,ℓ}(pq) ∈ {0,1} viewed over ℝ.
///
/// M is symmetric (f(pq) = f(qp)), so singular values equal |eigenvalues|.
///
/// Key metrics:
/// - `stable_rank` = ‖M‖_F² / σ_max² ∈ [1, n_primes].
///   ≈ n_primes → flat spectrum (no compression; consistent with random function).
///   « n_primes → heavy-tailed spectrum (low effective rank; potential structure).
/// - `real_rank` = number of eigenvalues above 1e-6 (numerical rank over ℝ).
///   Differs from F2 rank: can be lower (some F2-independent directions cancel over ℝ)
///   or higher (F2 dependencies that don't hold over ℝ).
/// - `top_eigenvalues` sorted descending by absolute value.
///   Empty for n_primes > 250 (too slow without BLAS; use stable_rank instead).
#[derive(Debug, Clone, Serialize)]
pub struct RealSpectrum {
    pub n_bits: u32,
    pub channel_weight: u32,
    pub channel_ell: u64,
    /// Number of distinct primes in the valid-pair set.
    pub n_primes: usize,
    /// Number of valid (p, q) pairs that fill entries of M.
    pub n_valid_pairs: usize,
    /// ‖M‖_F² = number of 1-entries in M (since entries ∈ {0,1}).
    pub frobenius_sq: f64,
    /// σ_max = largest singular value = largest |eigenvalue| (power iteration).
    pub spectral_norm: f64,
    /// ‖M‖_F² / σ_max²: effective number of dimensions (1 = rank-1, n = flat).
    pub stable_rank: f64,
    /// Numerical rank over ℝ via Gram-Schmidt (0 if n_primes > 300).
    pub real_rank: usize,
    /// Top min(20, n_primes) eigenvalues by |λ|, sorted descending.
    /// Empty if n_primes > 250.
    pub top_eigenvalues: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Target function
// ---------------------------------------------------------------------------

/// Ground truth: σ_{k-1}(N) mod ℓ mod 2.
///
/// This is the bit we measure: the parity of the Eisenstein channel value.
/// A constant-0 or constant-1 function would have degree 0; any non-trivial
/// dependence on the bits of N implies degree ≥ 1.
pub fn target_bit(sp: &Semiprime, ch: &Channel) -> u8 {
    let gt = eisenstein_hunt::ground_truth(sp, ch);
    (gt % 2) as u8
}

// ---------------------------------------------------------------------------
// Semiprime generation (enum for small n, random sampling for large n)
// ---------------------------------------------------------------------------

/// Return up to `count` balanced semiprimes of exactly `n_bits` bits.
///
/// For n_bits ≤ 20, enumerates the full population and subsamples so we never
/// request more semiprimes than exist (which would loop forever in rejection
/// sampling).  For n_bits > 20, delegates to `generate_semiprimes`.
pub(crate) fn get_semiprimes(n_bits: u32, count: usize, seed: u64) -> Vec<Semiprime> {
    if n_bits <= 20 {
        let pairs = enumerate_balanced_semiprimes(n_bits);
        // Shuffle and subsample using seeded RNG.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..pairs.len()).collect();
        // Partial Fisher-Yates to pick min(count, len) elements.
        let take = count.min(pairs.len());
        for i in 0..take {
            let j = i + rng.gen_range(0..(pairs.len() - i));
            indices.swap(i, j);
        }
        indices[..take]
            .iter()
            .map(|&idx| {
                let (p, q) = pairs[idx];
                Semiprime { n: p * q, p, q }
            })
            .collect()
    } else {
        eisenstein_hunt::generate_semiprimes(count, n_bits, n_bits, seed)
    }
}

// ---------------------------------------------------------------------------
// Semiprime enumeration for CRT rank (small n only)
// ---------------------------------------------------------------------------

/// Enumerate all balanced semiprimes N = p*q that are exactly n_bits bits wide,
/// with p ≤ q and p/q ≥ 0.3.  Returns (p, q) pairs sorted by (p, q).
///
/// Only feasible for n_bits ≤ 24.
pub fn enumerate_balanced_semiprimes(n_bits: u32) -> Vec<(u64, u64)> {
    assert!(n_bits >= 4 && n_bits <= 24, "CRT rank only feasible for n_bits ≤ 24");
    let half = n_bits / 2;
    let lo = 1u64 << (half - 1);
    let hi = 1u64 << (half + 1);

    // Collect primes in [lo, hi)
    let primes: Vec<u64> = (lo..hi)
        .filter(|&x| x % 2 == 1 && is_prime_u64(x))
        .collect();

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
// CRT rank estimation
// ---------------------------------------------------------------------------

/// Estimate the CRT rank of f(N) = target_bit(N, ch) for n-bit balanced semiprimes.
///
/// Builds the matrix M[p_idx][q_idx] over F2 and computes its rank
/// via Gaussian elimination.
pub fn crt_rank(n_bits: u32, ch: &Channel) -> CrtRankResult {
    let pairs = enumerate_balanced_semiprimes(n_bits);

    // Collect the set of primes involved
    let mut prime_set: Vec<u64> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
    prime_set.sort_unstable();
    prime_set.dedup();

    let np = prime_set.len();

    // Build matrix: rows = p index, cols = q index (both into prime_set)
    // Values over F2
    let mut matrix = vec![vec![0u8; np]; np];
    for &(p, q) in &pairs {
        let pi = prime_set.partition_point(|&x| x < p);
        let qi = prime_set.partition_point(|&x| x < q);
        let sp = Semiprime { n: p * q, p, q };
        let bit = target_bit(&sp, ch);
        matrix[pi][qi] = bit;
        matrix[qi][pi] = bit; // symmetric since f(pq) = f(qp)
    }

    // Gaussian elimination over F2
    let rank = f2_rank(&mut matrix, np, np);

    CrtRankResult {
        n_bits,
        channel_weight: ch.weight,
        channel_ell: ch.ell,
        n_primes: np,
        matrix_entries: pairs.len(),
        rank,
        rank_fraction: if np == 0 { 0.0 } else { rank as f64 / np as f64 },
    }
}

// ---------------------------------------------------------------------------
// F2 Gaussian elimination (in-place, returns rank)
// ---------------------------------------------------------------------------

/// Gaussian elimination over F2 on matrix `m` (n_rows × n_cols).
/// Modifies `m` in place to row-echelon form.  Returns the rank.
pub fn f2_rank(m: &mut Vec<Vec<u8>>, n_rows: usize, n_cols: usize) -> usize {
    let mut pivot_row = 0;
    for col in 0..n_cols {
        // Find a row at or below pivot_row with a 1 in this column
        let maybe_row = (pivot_row..n_rows).find(|&r| m[r][col] == 1);
        let Some(row) = maybe_row else { continue };
        m.swap(pivot_row, row);
        // Eliminate this column from all other rows
        for r in 0..n_rows {
            if r != pivot_row && m[r][col] == 1 {
                for c in 0..n_cols {
                    m[r][c] ^= m[pivot_row][c];
                }
            }
        }
        pivot_row += 1;
    }
    pivot_row
}

// ---------------------------------------------------------------------------
// Min fitting degree for d ≤ 2 (F2 linear systems)
// ---------------------------------------------------------------------------

/// Check if a degree-d polynomial over F2 in the bits of N fits all (N_i, y_i) pairs.
///
/// Enumerates all monomials of degree ≤ d in the n_bits bits of each N,
/// builds the feature matrix Φ ∈ F2^{m × C(n,≤d)}, and checks if y ∈ colspan(Φ).
///
/// Only feasible for d ≤ 2 with n_bits ≤ 48 and m ≤ 10_000.
pub fn min_fitting_degree(
    semiprimes: &[u64],
    targets: &[u8],
    n_bits: u32,
    max_degree: u32,
) -> FitDegreeResult {
    assert_eq!(semiprimes.len(), targets.len());
    assert!(max_degree <= 2, "min_fitting_degree only supports d ≤ 2");

    let m = semiprimes.len();
    let ch_dummy = Channel { weight: 0, ell: 0 }; // placeholder for result

    // Try each degree starting from 0
    for d in 0..=max_degree {
        // Build the list of monomials for degree ≤ d
        let monomials = monomials_up_to_degree(n_bits, d);
        let n_cols = monomials.len();

        // Build Φ ∈ F2^{m × n_cols} and augment with y as last column [Φ | y]
        let aug_cols = n_cols + 1;
        let mut aug = vec![vec![0u8; aug_cols]; m];
        for (i, &n) in semiprimes.iter().enumerate() {
            for (j, &mask) in monomials.iter().enumerate() {
                aug[i][j] = monomial_value(n, mask);
            }
            aug[i][n_cols] = targets[i];
        }

        // Gaussian elimination: if the augmented system has the same rank as Φ,
        // then y ∈ colspan(Φ) and a degree-d polynomial exists.
        let rank_aug = f2_rank(&mut aug, m, aug_cols);

        // Rebuild Φ alone to get its rank
        let mut phi = vec![vec![0u8; n_cols]; m];
        for (i, &n) in semiprimes.iter().enumerate() {
            for (j, &mask) in monomials.iter().enumerate() {
                phi[i][j] = monomial_value(n, mask);
            }
        }
        let rank_phi = f2_rank(&mut phi, m, n_cols);

        if rank_aug == rank_phi {
            // Consistent: y ∈ colspan(Φ), degree d suffices
            return FitDegreeResult {
                n_bits,
                channel_weight: ch_dummy.weight,
                channel_ell: ch_dummy.ell,
                n_samples: m,
                min_fitting_degree: Some(d),
            };
        }
    }

    FitDegreeResult {
        n_bits,
        channel_weight: ch_dummy.weight,
        channel_ell: ch_dummy.ell,
        n_samples: m,
        min_fitting_degree: None,
    }
}

/// Monomial value: m_S(N) = ∏_{i ∈ S} bit_i(N), where S is encoded as a bitmask.
/// Returns 1 if all bits in mask are set in n, else 0.
#[inline]
pub fn monomial_value(n: u64, mask: u64) -> u8 {
    ((n & mask) == mask) as u8
}

/// Generate all bitmasks for monomials of degree exactly `d` over n_bits bits.
fn monomials_of_degree(n_bits: u32, d: u32) -> Vec<u64> {
    if d == 0 {
        return vec![0u64]; // constant monomial, mask = 0 → always value 1
    }
    let mut result = Vec::new();
    // Enumerate subsets of size d from {0, ..., n_bits-1}
    subsets_of_size(n_bits as usize, d as usize, &mut result);
    result
}

/// Generate all bitmasks for monomials of degree ≤ d.
pub fn monomials_up_to_degree(n_bits: u32, d: u32) -> Vec<u64> {
    let mut result = Vec::new();
    for k in 0..=d {
        result.extend(monomials_of_degree(n_bits, k));
    }
    result
}

/// Enumerate all k-element subsets of {0, ..., n-1} as bitmasks, appending to `out`.
fn subsets_of_size(n: usize, k: usize, out: &mut Vec<u64>) {
    if k == 0 {
        out.push(0);
        return;
    }
    if k > n {
        return;
    }
    // Gosper's hack for lexicographic enumeration of k-subsets
    let mut mask = (1u64 << k) - 1;
    let limit = 1u64 << n;
    while mask < limit {
        out.push(mask);
        // Next k-subset via Gosper's hack
        let c = mask & mask.wrapping_neg();
        let r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }
}

// ---------------------------------------------------------------------------
// Correlation lower bound
// ---------------------------------------------------------------------------

/// Sample `n_monomials` random degree-d monomials and measure their correlation
/// with f(N) = target_bit(N, ch) in ±1 encoding.
///
/// Returns max |correlation| found.  If this exceeds `3 / sqrt(n_samples)`,
/// deg(f) ≥ d is confirmed with high probability.
pub fn correlation_lower_bound(
    semiprimes: &[u64],
    targets_pm1: &[i8], // ±1 encoding: 1 for target_bit=0, -1 for target_bit=1
    n_bits: u32,
    degree: u32,
    n_monomials: usize,
    rng: &mut StdRng,
) -> f64 {
    let m = semiprimes.len() as f64;
    let mut max_corr: f64 = 0.0;

    for _ in 0..n_monomials {
        // Sample a random degree-d subset of bit positions
        let mask = random_degree_mask(n_bits, degree, rng);

        // Compute correlation in ±1 encoding
        // m_S(N) = monomial_value(N, mask) ∈ {0,1} → ±1 encoding: 1 - 2*monomial_value
        let corr: f64 = semiprimes
            .iter()
            .zip(targets_pm1.iter())
            .map(|(&n, &y)| {
                let mx = 1i64 - 2 * monomial_value(n, mask) as i64;
                (mx * y as i64) as f64
            })
            .sum::<f64>()
            / m;

        if corr.abs() > max_corr {
            max_corr = corr.abs();
        }
    }

    max_corr
}

/// Sample a random bitmask representing a degree-d subset of {0, ..., n_bits-1}.
fn random_degree_mask(n_bits: u32, degree: u32, rng: &mut StdRng) -> u64 {
    if degree == 0 {
        return 0;
    }
    // Fisher-Yates to pick `degree` distinct positions
    let mut positions: Vec<u32> = (0..n_bits).collect();
    for i in 0..degree as usize {
        let j = rng.gen_range(i..n_bits as usize);
        positions.swap(i, j);
    }
    let mut mask = 0u64;
    for &pos in &positions[..degree as usize] {
        mask |= 1u64 << pos;
    }
    mask
}

/// Run the full correlation lower bound scan for a single channel at a given bit size.
///
/// `n_null_sims > 0` enables the calibrated permutation threshold (recommended).
/// Set `n_null_sims = 0` to fall back to the fixed 3/√m threshold.
pub fn run_correlation_scan(
    ch: &Channel,
    n_bits: u32,
    max_degree: u32,
    n_semiprimes: usize,
    n_monomials: usize,
    seed: u64,
    n_null_sims: usize,
    null_quantile: f64,
) -> Vec<CorrelationResult> {
    let sps = get_semiprimes(n_bits, n_semiprimes, seed);
    let actual_n = sps.len();

    let ns: Vec<u64> = sps.iter().map(|sp| sp.n).collect();
    let bits: Vec<u8> = sps.iter().map(|sp| target_bit(sp, ch)).collect();
    // Convert to ±1: 0 → +1, 1 → -1
    let pm1: Vec<i8> = bits.iter().map(|&b| 1 - 2 * b as i8).collect();

    // Separate RNGs so null simulation draws don't consume monomial seeds.
    let mut rng_mono = StdRng::seed_from_u64(seed ^ 0xdeadbeef);
    let mut rng_null = StdRng::seed_from_u64(seed ^ 0xcafe_babe);

    (1..=max_degree)
        .map(|d| {
            // Calibrated threshold via permutation test; fallback to 3/√m.
            let threshold = if n_null_sims > 0 && actual_n > 0 {
                null_max_corr_quantile(&ns, n_bits, d, n_monomials, null_quantile, n_null_sims, &mut rng_null)
            } else {
                3.0 / (actual_n as f64).sqrt().max(1.0)
            };
            let reported_quantile = if n_null_sims > 0 { null_quantile } else { 0.0 };

            let max_corr = correlation_lower_bound(&ns, &pm1, n_bits, d, n_monomials, &mut rng_mono);
            CorrelationResult {
                n_bits,
                degree: d,
                null_quantile: reported_quantile,
                channel_weight: ch.weight,
                channel_ell: ch.ell,
                max_abs_corr: max_corr,
                n_monomials,
                n_samples: actual_n,
                threshold,
                significant: max_corr > threshold,
            }
        })
        .collect()
}

/// Run the min_fitting_degree test for d ≤ 2 for a single channel at a given bit size.
pub fn run_fit_degree(
    ch: &Channel,
    n_bits: u32,
    n_semiprimes: usize,
    seed: u64,
) -> FitDegreeResult {
    let sps = get_semiprimes(n_bits, n_semiprimes, seed);
    let ns: Vec<u64> = sps.iter().map(|sp| sp.n).collect();
    let bits: Vec<u8> = sps.iter().map(|sp| target_bit(sp, ch)).collect();

    let mut result = min_fitting_degree(&ns, &bits, n_bits, 2);
    result.channel_weight = ch.weight;
    result.channel_ell = ch.ell;
    result
}

// ---------------------------------------------------------------------------
// Calibrated null threshold (permutation test)
// ---------------------------------------------------------------------------

/// Estimate the `quantile`-th quantile of the null distribution of
/// `max |correlation|` over `n_monomials` random degree-`degree` monomials.
///
/// Under H₀ the labels are i.i.d. ±1 independent of the input bits.  We fix
/// the semiprime N-values and re-draw labels for each simulation, making this
/// a permutation-style calibration.
///
/// This corrects the ~42% false-positive rate that the fixed 3/√m threshold
/// suffers when 200+ monomials are tested: the expected max of 200 null
/// correlations already exceeds 3/√m when m=2000.
pub fn null_max_corr_quantile(
    ns: &[u64],
    n_bits: u32,
    degree: u32,
    n_monomials: usize,
    quantile: f64,
    n_sims: usize,
    rng: &mut StdRng,
) -> f64 {
    if n_sims == 0 || ns.is_empty() {
        return 3.0 / (ns.len() as f64).sqrt().max(1.0);
    }
    let m_f = ns.len() as f64;
    let mut maxima = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        // Random ±1 labels — null hypothesis: no signal.
        let pm1: Vec<i8> = (0..ns.len())
            .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
            .collect();

        let max_c: f64 = (0..n_monomials)
            .map(|_| {
                let mask = random_degree_mask(n_bits, degree, rng);
                ns.iter()
                    .zip(pm1.iter())
                    .map(|(&n, &y)| {
                        let mx = 1i64 - 2 * monomial_value(n, mask) as i64;
                        (mx * y as i64) as f64
                    })
                    .sum::<f64>()
                    .abs()
                    / m_f
            })
            .fold(0.0f64, f64::max);

        maxima.push(max_c);
    }

    maxima.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((quantile * n_sims as f64) as usize).min(n_sims - 1);
    maxima[idx]
}

// ---------------------------------------------------------------------------
// Real-valued spectral analysis
// ---------------------------------------------------------------------------

/// Build the real-valued CRT matrix and compute spectral metrics.
///
/// For n_primes ≤ 250: computes top-20 eigenvalues via deflated power
/// iteration.  For n_primes > 250: only stable_rank and spectral_norm
/// (power iteration is O(n²·iters) and stays fast at any supported size).
pub fn real_spectrum(n_bits: u32, ch: &Channel) -> RealSpectrum {
    let pairs = enumerate_balanced_semiprimes(n_bits);
    let n_valid_pairs = pairs.len();

    let mut prime_set: Vec<u64> = pairs.iter().flat_map(|&(p, q)| [p, q]).collect();
    prime_set.sort_unstable();
    prime_set.dedup();
    let np = prime_set.len();

    // Build M ∈ ℝ^{np × np} with entries in {0.0, 1.0}.
    let mut m = vec![vec![0.0f64; np]; np];
    let mut n_ones = 0u64;
    for &(p, q) in &pairs {
        let pi = prime_set.partition_point(|&x| x < p);
        let qi = prime_set.partition_point(|&x| x < q);
        let sp = Semiprime { n: p * q, p, q };
        let val = target_bit(&sp, ch) as f64;
        m[pi][qi] = val;
        m[qi][pi] = val;
        // Count ones, avoiding double-count on diagonal (p≠q guaranteed by enum).
        if val > 0.5 { n_ones += 2; }
    }
    // frobenius_sq = ∑_{i,j} m_{ij}^2 = number of 1-entries (since entries ∈ {0,1}).
    let frobenius_sq = n_ones as f64;

    let mut rng = StdRng::seed_from_u64(0xabcd_1234 ^ ch.ell ^ (n_bits as u64 * 0x1111));

    let spectral_norm = power_iter_spectral_norm(&m, np, 80, &mut rng);

    let stable_rank = if spectral_norm > 1e-10 {
        frobenius_sq / (spectral_norm * spectral_norm)
    } else {
        0.0
    };

    let real_rank = if np <= 300 {
        gs_real_rank(&m, np, 1e-6)
    } else {
        0 // skipped — too expensive without BLAS
    };

    let top_eigenvalues = if np <= 250 {
        top_k_eigenvalues(&m, np, 20, 80, &mut rng)
    } else {
        vec![]
    };

    RealSpectrum {
        n_bits,
        channel_weight: ch.weight,
        channel_ell: ch.ell,
        n_primes: np,
        n_valid_pairs,
        frobenius_sq,
        spectral_norm,
        stable_rank,
        real_rank,
        top_eigenvalues,
    }
}

/// Power iteration for the spectral norm (largest |eigenvalue|) of a
/// symmetric real matrix.  Runs `n_iters` Rayleigh-quotient iterations.
fn power_iter_spectral_norm(m: &[Vec<f64>], n: usize, n_iters: usize, rng: &mut StdRng) -> f64 {
    if n == 0 { return 0.0; }
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    let vn = vec_norm(&v);
    if vn < 1e-10 { return 0.0; }
    for x in &mut v { *x /= vn; }

    for _ in 0..n_iters {
        let w = sym_matvec(m, &v, n);
        let wn = vec_norm(&w);
        if wn < 1e-10 { break; }
        for j in 0..n { v[j] = w[j] / wn; }
    }
    // Rayleigh quotient = v^T M v = eigenvalue (can be negative for symmetric M).
    let mv = sym_matvec(m, &v, n);
    let ray: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
    ray.abs()
}

/// Numerical rank of M over ℝ via Gram-Schmidt on the rows.
/// A row is linearly dependent if its residual 2-norm < `tol`.
fn gs_real_rank(m: &[Vec<f64>], n: usize, tol: f64) -> usize {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for row in m.iter().take(n) {
        let mut v = row.clone();
        for b in &basis {
            let dot: f64 = v.iter().zip(b.iter()).map(|(a, x)| a * x).sum();
            for j in 0..n { v[j] -= dot * b[j]; }
        }
        let vn = vec_norm(&v);
        if vn > tol {
            for x in &mut v { *x /= vn; }
            basis.push(v);
        }
    }
    basis.len()
}

/// Top-k eigenvalues by |λ| of symmetric M, via deflated power iteration
/// with re-orthogonalisation against all previously found eigenvectors.
fn top_k_eigenvalues(m: &[Vec<f64>], n: usize, k: usize, n_iters: usize, rng: &mut StdRng) -> Vec<f64> {
    if n == 0 { return vec![]; }
    let k = k.min(n);
    let mut eigenvalues = Vec::with_capacity(k);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);

    for _ in 0..k {
        let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
        orth_against(&mut v, &basis, n);
        let vn = vec_norm(&v);
        if vn < 1e-10 { break; }
        for x in &mut v { *x /= vn; }

        for _ in 0..n_iters {
            let mut w = sym_matvec(m, &v, n);
            orth_against(&mut w, &basis, n);
            let wn = vec_norm(&w);
            if wn < 1e-10 { break; }
            for j in 0..n { v[j] = w[j] / wn; }
        }
        let mv = sym_matvec(m, &v, n);
        let ray: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        eigenvalues.push(ray.abs());
        basis.push(v);
    }
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

#[inline]
fn sym_matvec(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n { w[i] += m[i][j] * v[j]; }
    }
    w
}

#[inline]
fn vec_norm(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }

fn orth_against(v: &mut Vec<f64>, basis: &[Vec<f64>], n: usize) {
    for b in basis {
        let dot: f64 = v.iter().zip(b.iter()).map(|(a, x)| a * x).sum();
        for j in 0..n { v[j] -= dot * b[j]; }
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
    fn test_monomial_value_constant() {
        // mask = 0: constant monomial → always 1
        assert_eq!(monomial_value(0, 0), 1);
        assert_eq!(monomial_value(u64::MAX, 0), 1);
    }

    #[test]
    fn test_monomial_value_single_bit() {
        // mask = 0b1 (bit 0): 1 for odd N, 0 for even N
        assert_eq!(monomial_value(5, 1), 1); // 5 = ...101, bit0 = 1
        assert_eq!(monomial_value(4, 1), 0); // 4 = ...100, bit0 = 0
    }

    #[test]
    fn test_monomial_value_two_bits() {
        // mask = 0b11 (bits 0,1): 1 iff both bit0 and bit1 are set
        assert_eq!(monomial_value(3, 3), 1); // 3 = 11, both set
        assert_eq!(monomial_value(5, 3), 0); // 5 = 101, bit1 = 0
    }

    #[test]
    fn test_subsets_of_size_zero() {
        let mut out = Vec::new();
        subsets_of_size(4, 0, &mut out);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn test_subsets_of_size_one() {
        let mut out = Vec::new();
        subsets_of_size(4, 1, &mut out);
        assert_eq!(out.len(), 4);
        for mask in &out {
            assert_eq!(mask.count_ones(), 1);
        }
    }

    #[test]
    fn test_subsets_of_size_two() {
        let mut out = Vec::new();
        subsets_of_size(4, 2, &mut out);
        assert_eq!(out.len(), 6); // C(4,2) = 6
        for mask in &out {
            assert_eq!(mask.count_ones(), 2);
        }
    }

    #[test]
    fn test_subsets_count() {
        let counts = [(4, 2, 6usize), (6, 3, 20), (8, 4, 70)];
        for (n, k, expected) in counts {
            let mut out = Vec::new();
            subsets_of_size(n, k, &mut out);
            assert_eq!(out.len(), expected, "C({n},{k}) should be {expected}");
        }
    }

    #[test]
    fn test_f2_rank_identity() {
        let mut m = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        assert_eq!(f2_rank(&mut m, 3, 3), 3);
    }

    #[test]
    fn test_f2_rank_zero() {
        let mut m = vec![vec![0u8; 3]; 3];
        assert_eq!(f2_rank(&mut m, 3, 3), 0);
    }

    #[test]
    fn test_f2_rank_rank_one() {
        // All rows are [1,0,1]: rank 1
        let mut m = vec![vec![1, 0, 1], vec![1, 0, 1], vec![1, 0, 1]];
        assert_eq!(f2_rank(&mut m, 3, 3), 1);
    }

    #[test]
    fn test_enumerate_balanced_semiprimes_16bit() {
        let pairs = enumerate_balanced_semiprimes(16);
        assert!(!pairs.is_empty());
        for &(p, q) in &pairs {
            assert!(is_prime_u64(p));
            assert!(is_prime_u64(q));
            let n = p as u128 * q as u128;
            assert!(n >= 1 << 15 && n < 1 << 16, "N = {n} should be 16-bit");
            assert!((p as f64) / (q as f64) >= 0.3);
        }
    }

    #[test]
    fn test_target_bit_range() {
        let ch = &CHANNELS[0];
        let sps = eisenstein_hunt::generate_semiprimes(20, 16, 20, 42);
        for sp in &sps {
            let bit = target_bit(sp, ch);
            assert!(bit == 0 || bit == 1);
        }
    }

    #[test]
    fn test_crt_rank_16bit() {
        let ch = &CHANNELS[0]; // k=12, ℓ=691
        let result = crt_rank(16, ch);
        // rank_fraction should be in [0,1]
        assert!(result.rank_fraction >= 0.0 && result.rank_fraction <= 1.0);
        assert!(result.n_primes > 0);
    }

    #[test]
    fn test_min_fitting_degree_constant_function() {
        // If all targets are 0, degree 0 should fit.
        let ns: Vec<u64> = vec![15, 21, 35, 77, 143];
        let targets = vec![0u8; 5];
        let result = min_fitting_degree(&ns, &targets, 8, 2);
        assert_eq!(result.min_fitting_degree, Some(0));
    }

    #[test]
    fn test_min_fitting_degree_linear_function() {
        // f(N) = bit_0(N): this is a degree-1 function.
        // Should NOT fit degree-0 but SHOULD fit degree-1.
        let ns: Vec<u64> = vec![15, 21, 35, 77, 143, 10, 22, 34];
        let targets: Vec<u8> = ns.iter().map(|&n| (n & 1) as u8).collect();
        let result = min_fitting_degree(&ns, &targets, 8, 2);
        // degree-1 polynomial should fit (it's literally a degree-1 monomial)
        assert!(
            result.min_fitting_degree.map(|d| d <= 1).unwrap_or(false),
            "linear function should fit degree ≤ 1, got {:?}",
            result.min_fitting_degree
        );
    }

    #[test]
    fn test_correlation_scan_returns_results() {
        let ch = &CHANNELS[6]; // k=22, ℓ=593
        // n_null_sims=0 → fast fixed-threshold mode (calibration skipped in tests)
        let results = run_correlation_scan(ch, 16, 3, 200, 50, 42, 0, 0.99);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.max_abs_corr >= 0.0 && r.max_abs_corr <= 1.0);
        }
    }
}
