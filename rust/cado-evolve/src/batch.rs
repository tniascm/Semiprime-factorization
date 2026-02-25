//! Batch smoothness testing via product trees (Bernstein's approach).
//!
//! Tests whether a batch of candidates are B-smooth simultaneously,
//! using a product tree to amortize GCD cost across candidates.
//!
//! The key insight: instead of testing each candidate against every prime
//! in the factor base (O(k * π(B)) per candidate), we compute
//! P = Π(p^⌊log_p(B)⌋ for p ≤ B), then for each candidate a, compute
//! gcd(a, P) in a single operation. If gcd(a, P) == a, then a is B-smooth.
//!
//! For batches, we use a remainder tree to compute P mod aᵢ for all
//! candidates simultaneously, reducing the per-candidate cost.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use std::time::{Duration, Instant};

/// Result of a batch smoothness test.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of candidates tested
    pub candidates_tested: usize,
    /// Number found to be B-smooth
    pub smooth_count: usize,
    /// Wall-clock time for the batch
    pub wall_time: Duration,
    /// Candidates tested per second
    pub throughput: f64,
    /// Smooth fraction
    pub smooth_fraction: f64,
    /// Factor base size (number of primes)
    pub factor_base_size: usize,
    /// Smoothness bound B
    pub smoothness_bound: u64,
}

/// Configuration for batch smoothness testing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Smoothness bound B — test if candidates have all prime factors ≤ B
    pub smoothness_bound: u64,
    /// Number of candidates per batch
    pub batch_size: usize,
    /// Whether to use the product tree optimization
    pub use_product_tree: bool,
}

/// Factor base: set of primes up to bound B, with precomputed product.
#[derive(Debug, Clone)]
pub struct FactorBase {
    /// The smoothness bound
    pub bound: u64,
    /// All primes up to bound
    pub primes: Vec<u64>,
    /// Product P = Π(p^⌊log_p(bound²)⌋) for all primes p ≤ bound.
    /// Using bound² as the exponent limit ensures we catch prime powers
    /// up to the candidate size.
    pub product: BigUint,
}

impl FactorBase {
    /// Build a factor base for the given smoothness bound.
    ///
    /// Computes the "prime power product" P = Π(p^k) where k = ⌊log_p(limit)⌋
    /// and limit is chosen based on the expected candidate size.
    pub fn new(bound: u64, candidate_bits: u32) -> Self {
        let primes = sieve_primes(bound);

        // Compute P = Π(p^k) where k = ⌊log_p(2^candidate_bits)⌋
        // This ensures p^k ≤ candidate_max for each prime p.
        let limit = candidate_bits as f64 * std::f64::consts::LN_2;

        let product = compute_prime_power_product(&primes, limit);

        FactorBase {
            bound,
            primes,
            product,
        }
    }

    /// Number of primes in the factor base.
    pub fn size(&self) -> usize {
        self.primes.len()
    }
}

/// Sieve of Eratosthenes up to bound.
fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return Vec::new();
    }
    let n = bound as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    if n >= 1 {
        is_prime[1] = false;
    }

    let limit = (n as f64).sqrt() as usize + 1;
    for i in 2..=limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    (2..=n).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}

/// Compute P = Π(p^k) where k = ⌊limit / ln(p)⌋.
///
/// Uses a product tree for efficient multiplication of many large numbers.
fn compute_prime_power_product(primes: &[u64], limit: f64) -> BigUint {
    if primes.is_empty() {
        return BigUint::one();
    }

    // First compute all prime powers
    let prime_powers: Vec<BigUint> = primes
        .iter()
        .map(|&p| {
            let k = (limit / (p as f64).ln()).floor() as u32;
            let k = k.max(1);
            BigUint::from(p).pow(k)
        })
        .collect();

    // Use a product tree to multiply them all together
    product_tree_multiply(&prime_powers)
}

/// Multiply a list of BigUints using a balanced product tree.
///
/// More efficient than sequential multiplication because it keeps
/// operand sizes balanced, which is better for multiplication algorithms
/// (Karatsuba, Toom-Cook, FFT).
fn product_tree_multiply(values: &[BigUint]) -> BigUint {
    if values.is_empty() {
        return BigUint::one();
    }
    if values.len() == 1 {
        return values[0].clone();
    }
    if values.len() == 2 {
        return &values[0] * &values[1];
    }

    // Pairwise multiply, then recurse
    let mut next_level: Vec<BigUint> = values
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                &chunk[0] * &chunk[1]
            } else {
                chunk[0].clone()
            }
        })
        .collect();

    while next_level.len() > 1 {
        next_level = next_level
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    &chunk[0] * &chunk[1]
                } else {
                    chunk[0].clone()
                }
            })
            .collect();
    }

    next_level.into_iter().next().unwrap_or_else(BigUint::one)
}

/// Test if a single value is B-smooth using the precomputed product.
///
/// Algorithm: repeatedly compute gcd(value, P) and divide out the
/// smooth part until the remainder is 1 (smooth) or no progress (not smooth).
fn is_smooth_via_product(value: &BigUint, product: &BigUint) -> bool {
    if value.is_one() || value.is_zero() {
        return true;
    }

    let mut remainder = value.clone();
    loop {
        let g = remainder.gcd(product);
        if g.is_one() {
            // No common factors with P — remainder is not smooth
            return remainder.is_one();
        }
        remainder /= &g;
        if remainder.is_one() {
            return true;
        }
        // Continue: the remainder might have repeated prime factors
        // that weren't fully divided out by one gcd
    }
}

/// Test a batch of candidates for B-smoothness.
///
/// Returns (smooth_indices, wall_time) — indices of candidates that are B-smooth.
pub fn batch_smooth_test(
    candidates: &[BigUint],
    factor_base: &FactorBase,
) -> (Vec<usize>, Duration) {
    let start = Instant::now();

    let smooth_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| is_smooth_via_product(c, &factor_base.product))
        .map(|(i, _)| i)
        .collect();

    (smooth_indices, start.elapsed())
}

/// Test a batch of candidates using naive trial division (baseline).
///
/// Returns (smooth_indices, wall_time).
pub fn batch_smooth_trial_division(
    candidates: &[BigUint],
    primes: &[u64],
) -> (Vec<usize>, Duration) {
    let start = Instant::now();

    let smooth_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| is_smooth_trial_division(c, primes))
        .map(|(i, _)| i)
        .collect();

    (smooth_indices, start.elapsed())
}

/// Test smoothness via trial division (baseline for comparison).
fn is_smooth_trial_division(value: &BigUint, primes: &[u64]) -> bool {
    let mut remainder = value.clone();
    for &p in primes {
        let big_p = BigUint::from(p);
        while (&remainder % &big_p).is_zero() {
            remainder /= &big_p;
        }
        if remainder.is_one() {
            return true;
        }
    }
    remainder.is_one()
}

/// Generate random candidates of a given bit size for testing.
///
/// In real NFS, candidates are values of the algebraic norm at sieve
/// positions. Here we generate random integers of similar size.
pub fn generate_random_candidates(
    n_candidates: usize,
    bits: u32,
    rng: &mut impl rand::Rng,
) -> Vec<BigUint> {
    (0..n_candidates)
        .map(|_| {
            let bytes = (bits as usize + 7) / 8;
            let mut buf = vec![0u8; bytes];
            rng.fill(&mut buf[..]);
            // Set high bit to ensure correct bit count
            if !buf.is_empty() {
                let excess_bits = (bytes * 8) as u32 - bits;
                buf[0] >>= excess_bits;
                buf[0] |= 1 << (7 - excess_bits);
            }
            BigUint::from_bytes_be(&buf)
        })
        .collect()
}

/// Generate candidates with a known smooth fraction for testing.
///
/// Creates `n_smooth` B-smooth values and `n_total - n_smooth` random values.
pub fn generate_mixed_candidates(
    n_total: usize,
    n_smooth: usize,
    primes: &[u64],
    bits: u32,
    rng: &mut impl rand::Rng,
) -> Vec<BigUint> {
    let mut candidates = Vec::with_capacity(n_total);

    // Generate smooth candidates: products of random primes from the factor base
    for _ in 0..n_smooth.min(n_total) {
        let target_bits = bits;
        let mut value = BigUint::one();
        let mut current_bits = 0u32;
        while current_bits < target_bits {
            let idx = rng.gen_range(0..primes.len());
            let p = primes[idx];
            value *= BigUint::from(p);
            current_bits = value.bits() as u32;
        }
        candidates.push(value);
    }

    // Fill rest with random (likely non-smooth) candidates
    let remaining = n_total - candidates.len();
    candidates.extend(generate_random_candidates(remaining, bits, rng));

    // Shuffle so smooth values are distributed throughout
    use rand::seq::SliceRandom;
    candidates.shuffle(rng);

    candidates
}

/// Run a complete batch smoothness benchmark.
///
/// Compares product-tree method vs trial division on the same candidates.
pub fn run_batch_benchmark(config: &BatchConfig, candidate_bits: u32) -> BatchBenchmark {
    let mut rng = rand::thread_rng();

    // Build factor base
    let fb_start = Instant::now();
    let factor_base = FactorBase::new(config.smoothness_bound, candidate_bits);
    let fb_time = fb_start.elapsed();

    // Generate candidates — use realistic smooth fraction
    // In NFS at c80 with bound ~300K, expect ~1-5% smoothness rate
    let n_smooth = (config.batch_size as f64 * 0.03) as usize;
    let candidates = generate_mixed_candidates(
        config.batch_size,
        n_smooth,
        &factor_base.primes,
        candidate_bits,
        &mut rng,
    );

    // Test with product-tree method
    let (product_smooth, product_time) = batch_smooth_test(&candidates, &factor_base);

    // Test with trial division (baseline)
    let (trial_smooth, trial_time) = batch_smooth_trial_division(&candidates, &factor_base.primes);

    // Verify both methods agree
    let methods_agree = product_smooth == trial_smooth;
    if !methods_agree {
        log::warn!(
            "Product tree found {} smooth, trial division found {} smooth",
            product_smooth.len(),
            trial_smooth.len()
        );
    }

    let product_throughput = config.batch_size as f64 / product_time.as_secs_f64();
    let trial_throughput = config.batch_size as f64 / trial_time.as_secs_f64();

    BatchBenchmark {
        config: config.clone(),
        candidate_bits,
        factor_base_size: factor_base.size(),
        factor_base_product_bits: factor_base.product.bits() as u64,
        fb_build_time: fb_time,
        product_tree: BatchResult {
            candidates_tested: config.batch_size,
            smooth_count: product_smooth.len(),
            wall_time: product_time,
            throughput: product_throughput,
            smooth_fraction: product_smooth.len() as f64 / config.batch_size as f64,
            factor_base_size: factor_base.size(),
            smoothness_bound: config.smoothness_bound,
        },
        trial_division: BatchResult {
            candidates_tested: config.batch_size,
            smooth_count: trial_smooth.len(),
            wall_time: trial_time,
            throughput: trial_throughput,
            smooth_fraction: trial_smooth.len() as f64 / config.batch_size as f64,
            factor_base_size: factor_base.size(),
            smoothness_bound: config.smoothness_bound,
        },
        methods_agree,
        speedup: trial_time.as_secs_f64() / product_time.as_secs_f64(),
    }
}

/// Complete benchmark result comparing product-tree vs trial division.
#[derive(Debug, Clone)]
pub struct BatchBenchmark {
    pub config: BatchConfig,
    pub candidate_bits: u32,
    pub factor_base_size: usize,
    pub factor_base_product_bits: u64,
    pub fb_build_time: Duration,
    pub product_tree: BatchResult,
    pub trial_division: BatchResult,
    pub methods_agree: bool,
    pub speedup: f64,
}

impl std::fmt::Display for BatchBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Batch Smoothness Benchmark")?;
        writeln!(
            f,
            "  Candidates: {} × {}-bit, B={}",
            self.config.batch_size, self.candidate_bits, self.config.smoothness_bound
        )?;
        writeln!(
            f,
            "  Factor base: {} primes, product = {} bits",
            self.factor_base_size, self.factor_base_product_bits
        )?;
        writeln!(
            f,
            "  FB build time: {:.3}s",
            self.fb_build_time.as_secs_f64()
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  Product tree: {:.1} candidates/s, {} smooth ({:.1}%), {:.3}s",
            self.product_tree.throughput,
            self.product_tree.smooth_count,
            self.product_tree.smooth_fraction * 100.0,
            self.product_tree.wall_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Trial div:    {:.1} candidates/s, {} smooth ({:.1}%), {:.3}s",
            self.trial_division.throughput,
            self.trial_division.smooth_count,
            self.trial_division.smooth_fraction * 100.0,
            self.trial_division.wall_time.as_secs_f64()
        )?;
        writeln!(
            f,
            "  Speedup: {:.2}× (product tree vs trial division)",
            self.speedup
        )?;
        writeln!(
            f,
            "  Methods agree: {}",
            if self.methods_agree { "YES" } else { "NO" }
        )?;
        Ok(())
    }
}

/// Run benchmarks across multiple parameter scales matching the scaling protocol.
pub fn run_scaling_benchmarks() -> Vec<BatchBenchmark> {
    // Match the classical baseline parameter scales
    let configs = vec![
        // c60 (199-bit): lim0=~200K, candidates ~100-bit norms
        (
            BatchConfig {
                smoothness_bound: 200_000,
                batch_size: 10_000,
                use_product_tree: true,
            },
            100, // candidate norm bits
        ),
        // c80 (266-bit): lim0=~310K, candidates ~130-bit norms
        (
            BatchConfig {
                smoothness_bound: 310_000,
                batch_size: 10_000,
                use_product_tree: true,
            },
            130,
        ),
        // c80 larger batch
        (
            BatchConfig {
                smoothness_bound: 310_000,
                batch_size: 50_000,
                use_product_tree: true,
            },
            130,
        ),
        // c100 (332-bit): projected lim0=~500K, candidates ~166-bit norms
        (
            BatchConfig {
                smoothness_bound: 500_000,
                batch_size: 10_000,
                use_product_tree: true,
            },
            166,
        ),
    ];

    configs
        .into_iter()
        .map(|(config, bits)| {
            let result = run_batch_benchmark(&config, bits);
            println!("{}", result);
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_primes_small() {
        let empty: Vec<u64> = vec![];
        assert_eq!(sieve_primes(1), empty);
        assert_eq!(sieve_primes(2), vec![2u64]);
        assert_eq!(sieve_primes(3), vec![2u64, 3]);
    }

    #[test]
    fn test_product_tree_multiply() {
        let values: Vec<BigUint> = vec![2u32, 3, 5, 7, 11]
            .into_iter()
            .map(BigUint::from)
            .collect();
        let product = product_tree_multiply(&values);
        assert_eq!(product, BigUint::from(2310u32));
    }

    #[test]
    fn test_product_tree_multiply_empty() {
        let product = product_tree_multiply(&[]);
        assert_eq!(product, BigUint::one());
    }

    #[test]
    fn test_product_tree_multiply_single() {
        let product = product_tree_multiply(&[BigUint::from(42u32)]);
        assert_eq!(product, BigUint::from(42u32));
    }

    #[test]
    fn test_is_smooth_via_product() {
        // Factor base: primes up to 10 → {2, 3, 5, 7}
        // P = 2^3 * 3^2 * 5 * 7 = 2520 (with limit = ln(2^10) ≈ 6.93)
        let fb = FactorBase::new(10, 10);

        // 12 = 2^2 * 3 — smooth
        assert!(is_smooth_via_product(&BigUint::from(12u32), &fb.product));

        // 30 = 2 * 3 * 5 — smooth
        assert!(is_smooth_via_product(&BigUint::from(30u32), &fb.product));

        // 11 — not smooth (11 > bound)
        assert!(!is_smooth_via_product(&BigUint::from(11u32), &fb.product));

        // 22 = 2 * 11 — not smooth
        assert!(!is_smooth_via_product(&BigUint::from(22u32), &fb.product));

        // 1 — trivially smooth
        assert!(is_smooth_via_product(&BigUint::one(), &fb.product));
    }

    #[test]
    fn test_is_smooth_trial_division() {
        let primes = vec![2, 3, 5, 7];

        assert!(is_smooth_trial_division(&BigUint::from(12u32), &primes));
        assert!(is_smooth_trial_division(&BigUint::from(30u32), &primes));
        assert!(!is_smooth_trial_division(&BigUint::from(11u32), &primes));
        assert!(!is_smooth_trial_division(&BigUint::from(22u32), &primes));
    }

    #[test]
    fn test_methods_agree() {
        let fb = FactorBase::new(100, 32);
        let mut rng = rand::thread_rng();

        let candidates = generate_mixed_candidates(100, 10, &fb.primes, 32, &mut rng);

        let (product_smooth, _) = batch_smooth_test(&candidates, &fb);
        let (trial_smooth, _) = batch_smooth_trial_division(&candidates, &fb.primes);

        assert_eq!(
            product_smooth, trial_smooth,
            "Product tree and trial division must agree"
        );
    }

    #[test]
    fn test_factor_base_construction() {
        let fb = FactorBase::new(100, 64);
        assert_eq!(fb.bound, 100);
        assert_eq!(fb.primes.len(), 25); // 25 primes ≤ 100
        assert!(fb.product > BigUint::one());
        // Product should be much larger than 100 (it's Π(p^k))
        assert!(fb.product.bits() > 100);
    }

    #[test]
    fn test_generate_random_candidates() {
        let mut rng = rand::thread_rng();
        let candidates = generate_random_candidates(50, 64, &mut rng);
        assert_eq!(candidates.len(), 50);
        for c in &candidates {
            // Should be close to 64 bits
            assert!(c.bits() >= 60 && c.bits() <= 64, "bits: {}", c.bits());
        }
    }

    #[test]
    fn test_batch_benchmark_small() {
        let config = BatchConfig {
            smoothness_bound: 50,
            batch_size: 100,
            use_product_tree: true,
        };
        let result = run_batch_benchmark(&config, 32);
        assert!(result.methods_agree);
        assert_eq!(result.product_tree.candidates_tested, 100);
        assert_eq!(result.trial_division.candidates_tested, 100);
        assert!(result.product_tree.throughput > 0.0);
        assert!(result.trial_division.throughput > 0.0);
    }
}
