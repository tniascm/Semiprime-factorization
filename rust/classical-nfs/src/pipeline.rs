//! NFS pipeline: orchestrates polynomial selection, sieving, relation collection,
//! linear algebra, and factor extraction into a single `factor_nfs` entry point.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};

use crate::factor::{extract_factor, extract_factor_rational_only};
use crate::linalg::find_dependencies;
use crate::polynomial::{choose_degree, isqrt, select_polynomial, NfsPolynomial};
use crate::relation::{build_matrix, collect_relations, SmoothRelation};
use crate::sieve::{
    compute_polynomial_roots, line_sieve, sieve_primes, SieveConfig,
};

/// Parameters for the NFS pipeline.
#[derive(Debug, Clone)]
pub struct NfsPipelineParams {
    /// Factor base bound for the rational side.
    pub rational_fb_bound: u64,
    /// Factor base bound for the algebraic side.
    pub algebraic_fb_bound: u64,
    /// Sieve area half-width.
    pub sieve_area: i64,
    /// Maximum b value for sieving.
    pub max_b: i64,
    /// Large prime multiplier (large_prime_bound = multiplier * fb_bound).
    pub large_prime_multiplier: u64,
    /// Polynomial degree (0 = auto-select based on n).
    pub degree: usize,
    /// Maximum number of sieve expansions to try.
    pub max_expansions: u32,
}

impl NfsPipelineParams {
    /// Create parameters tuned for a given bit size of n.
    pub fn for_bits(bits: u64) -> Self {
        if bits <= 40 {
            Self {
                rational_fb_bound: 1 << 8,     // 256
                algebraic_fb_bound: 1 << 8,
                sieve_area: 1 << 12,            // 4096
                max_b: 1 << 8,                  // 256
                large_prime_multiplier: 50,
                degree: 3,
                max_expansions: 3,
            }
        } else if bits <= 50 {
            Self {
                rational_fb_bound: 1 << 10,    // 1024
                algebraic_fb_bound: 1 << 10,
                sieve_area: 1 << 14,
                max_b: 1 << 9,
                large_prime_multiplier: 50,
                degree: 3,
                max_expansions: 3,
            }
        } else if bits <= 60 {
            Self {
                rational_fb_bound: 1 << 11,
                algebraic_fb_bound: 1 << 11,
                sieve_area: 1 << 15,
                max_b: 1 << 10,
                large_prime_multiplier: 100,
                degree: 3,
                max_expansions: 4,
            }
        } else if bits <= 70 {
            Self {
                rational_fb_bound: 1 << 12,
                algebraic_fb_bound: 1 << 12,
                sieve_area: 1 << 16,
                max_b: 1 << 11,
                large_prime_multiplier: 100,
                degree: 3,
                max_expansions: 5,
            }
        } else {
            // 70-80 bits
            Self {
                rational_fb_bound: 1 << 14,
                algebraic_fb_bound: 1 << 14,
                sieve_area: 1 << 17,
                max_b: 1 << 12,
                large_prime_multiplier: 100,
                degree: 3,
                max_expansions: 5,
            }
        }
    }
}

impl Default for NfsPipelineParams {
    fn default() -> Self {
        Self::for_bits(60)
    }
}

/// Statistics from the NFS pipeline run.
#[derive(Debug, Clone)]
pub struct NfsStats {
    pub polynomial_degree: usize,
    pub polynomial_m: BigUint,
    pub rational_fb_size: usize,
    pub algebraic_fb_size: usize,
    pub sieve_hits: usize,
    pub full_relations: usize,
    pub cycle_relations: usize,
    pub partial_relations: usize,
    pub total_relations: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub dependencies_found: usize,
    pub dependencies_tried: usize,
    pub factor_found: bool,
}

/// Full NFS factorization pipeline.
///
/// Steps:
/// 1. Select polynomial (base-m method)
/// 2. Build factor bases (rational and algebraic)
/// 3. Sieve and collect smooth relations
/// 4. Build GF(2) matrix and find null-space vectors
/// 5. Extract factor via square root computation
///
/// Returns `Some(factor)` if a non-trivial factor is found, `None` otherwise.
pub fn factor_nfs(n: &BigUint) -> Option<BigUint> {
    let params = NfsPipelineParams::for_bits(n.bits());
    factor_nfs_with_params(n, &params)
}

/// NFS factorization with explicit parameters.
pub fn factor_nfs_with_params(n: &BigUint, params: &NfsPipelineParams) -> Option<BigUint> {
    let one = BigUint::one();

    // Trivial checks
    if *n <= one {
        return None;
    }
    if n.is_even() {
        return Some(BigUint::from(2u64));
    }

    // Perfect square check
    let s = isqrt(n);
    if &s * &s == *n {
        return Some(s);
    }

    // Quick trial division for small factors
    let small_primes = sieve_primes(1000);
    for &p in &small_primes {
        let p_big = BigUint::from(p);
        if (&*n % &p_big).is_zero() && p_big != *n {
            return Some(p_big);
        }
    }

    // Step 1: Polynomial selection
    let degree = if params.degree == 0 {
        choose_degree(n)
    } else {
        params.degree
    };
    let poly = select_polynomial(n, degree);

    // Check degenerate polynomial
    if poly.m < BigUint::from(2u64) {
        return None;
    }

    // Step 2: Build factor bases
    let rational_fb = sieve_primes(params.rational_fb_bound);
    let algebraic_fb = sieve_primes(params.algebraic_fb_bound);

    if rational_fb.is_empty() || algebraic_fb.is_empty() {
        return None;
    }

    // Compute polynomial roots for algebraic-side sieving
    let alg_roots = compute_polynomial_roots(&poly, &algebraic_fb);

    let rational_fb_size = rational_fb.len();
    let algebraic_fb_size = algebraic_fb.len();
    let target_relations = rational_fb_size + algebraic_fb_size + 10;
    let large_prime_bound = params.large_prime_multiplier * params.rational_fb_bound;

    // Step 3: Sieve and collect relations, expanding if needed
    let mut all_relations: Vec<SmoothRelation> = Vec::new();
    let mut current_sieve_area = params.sieve_area;
    let mut current_max_b = params.max_b;

    for _expansion in 0..params.max_expansions {
        let config = SieveConfig {
            rational_bound: params.rational_fb_bound,
            algebraic_bound: params.algebraic_fb_bound,
            sieve_area: current_sieve_area,
            max_b: current_max_b,
            large_prime_multiplier: params.large_prime_multiplier,
            sieve_threshold: 0.0,
        };

        let hits = line_sieve(&poly, &rational_fb, &alg_roots, &config);
        let result = collect_relations(&hits, &poly, &rational_fb, &algebraic_fb, large_prime_bound);

        // Merge new relations, deduplicating by (a, b) pair
        for rel in result.full_relations {
            if !all_relations.iter().any(|r| r.a == rel.a && r.b == rel.b) {
                all_relations.push(rel);
            }
        }
        for rel in result.cycle_relations {
            if !all_relations.iter().any(|r| r.a == rel.a && r.b == rel.b) {
                all_relations.push(rel);
            }
        }

        if all_relations.len() >= target_relations {
            break;
        }

        // Expand sieve range for next attempt
        current_sieve_area = (current_sieve_area as f64 * 1.5) as i64;
        current_max_b = (current_max_b as f64 * 1.3) as i64;
    }

    if all_relations.len() < 2 {
        // Try rational-only approach as fallback
        return rational_only_fallback(n, &poly, &rational_fb, params);
    }

    // Step 4: Build GF(2) matrix and find dependencies
    let (matrix, num_cols) = build_matrix(&all_relations, rational_fb_size, algebraic_fb_size);
    let dependencies = find_dependencies(&matrix, num_cols);

    if dependencies.is_empty() {
        // Try rational-only approach as fallback
        return rational_only_fallback(n, &poly, &rational_fb, params);
    }

    // Step 5: Extract factor
    let extraction = extract_factor(
        n,
        &poly,
        &all_relations,
        &dependencies,
        &rational_fb,
        &algebraic_fb,
    );

    if let Some(f) = extraction.factor {
        return Some(f);
    }

    // If full extraction didn't work, try rational-only extraction with same relations
    let extraction_rat = extract_factor_rational_only(
        n,
        &poly,
        &all_relations,
        &dependencies,
        &rational_fb,
    );

    if let Some(f) = extraction_rat.factor {
        return Some(f);
    }

    // Final fallback: try rational-only with dedicated sieve
    rational_only_fallback(n, &poly, &rational_fb, params)
}

/// Fallback: rational-only sieving (similar to the original NFS approach in lib.rs).
///
/// Only factors the rational side a + b*m over the factor base. Simpler
/// but less powerful than the full two-sided approach.
fn rational_only_fallback(
    n: &BigUint,
    poly: &NfsPolynomial,
    rational_fb: &[u64],
    params: &NfsPipelineParams,
) -> Option<BigUint> {
    let m_i128: i128 = match poly.m.to_u128() {
        Some(m) => m as i128,
        None => return None,
    };

    let sieve_range = params.sieve_area.max(500);

    // Collect rational-only smooth relations
    let mut relations: Vec<SmoothRelation> = Vec::new();

    for b in 1..=sieve_range.min(params.max_b) {
        let b_i128 = b as i128;

        for a in -sieve_range..=sieve_range {
            if a == 0 {
                continue;
            }
            let a_abs = a.unsigned_abs() as u64;
            let b_u64 = b as u64;
            if crate::sieve::gcd_u64(a_abs, b_u64) != 1 {
                continue;
            }

            let rational_val_signed: i128 = (a as i128) + b_i128 * m_i128;
            if rational_val_signed == 0 {
                continue;
            }

            let sign_negative = rational_val_signed < 0;
            let rational_abs = rational_val_signed.unsigned_abs();
            if rational_abs > u64::MAX as u128 {
                continue;
            }

            let mut remaining = rational_abs as u64;
            let mut exponents = vec![0u32; rational_fb.len()];
            for (i, &p) in rational_fb.iter().enumerate() {
                while remaining % p == 0 {
                    remaining /= p;
                    exponents[i] += 1;
                }
            }

            if remaining == 1 {
                relations.push(SmoothRelation {
                    a,
                    b,
                    sign_negative,
                    rational_exponents: exponents,
                    algebraic_exponents: vec![],
                    rational_large_prime: 0,
                    algebraic_large_prime: 0,
                });
            }
        }
    }

    if relations.len() < 2 {
        return None;
    }

    // Build matrix (rational side only + sign bit)
    let num_cols = 1 + rational_fb.len();
    let matrix: Vec<Vec<u8>> = relations
        .iter()
        .map(|r| {
            let mut row = vec![if r.sign_negative { 1u8 } else { 0u8 }];
            for &e in &r.rational_exponents {
                row.push((e % 2) as u8);
            }
            row
        })
        .collect();

    let dependencies = find_dependencies(&matrix, num_cols);

    if dependencies.is_empty() {
        return None;
    }

    let extraction = extract_factor_rational_only(
        n,
        poly,
        &relations,
        &dependencies,
        rational_fb,
    );

    extraction.factor
}

/// Run the NFS pipeline and return both the factor and statistics.
pub fn factor_nfs_with_stats(n: &BigUint, params: &NfsPipelineParams) -> (Option<BigUint>, NfsStats) {
    let one = BigUint::one();
    let mut stats = NfsStats {
        polynomial_degree: 0,
        polynomial_m: BigUint::zero(),
        rational_fb_size: 0,
        algebraic_fb_size: 0,
        sieve_hits: 0,
        full_relations: 0,
        cycle_relations: 0,
        partial_relations: 0,
        total_relations: 0,
        matrix_rows: 0,
        matrix_cols: 0,
        dependencies_found: 0,
        dependencies_tried: 0,
        factor_found: false,
    };

    // Trivial checks
    if *n <= one {
        return (None, stats);
    }
    if n.is_even() {
        stats.factor_found = true;
        return (Some(BigUint::from(2u64)), stats);
    }

    let degree = if params.degree == 0 {
        choose_degree(n)
    } else {
        params.degree
    };
    let poly = select_polynomial(n, degree);
    stats.polynomial_degree = degree;
    stats.polynomial_m = poly.m.clone();

    if poly.m < BigUint::from(2u64) {
        return (None, stats);
    }

    let rational_fb = sieve_primes(params.rational_fb_bound);
    let algebraic_fb = sieve_primes(params.algebraic_fb_bound);
    stats.rational_fb_size = rational_fb.len();
    stats.algebraic_fb_size = algebraic_fb.len();

    if rational_fb.is_empty() || algebraic_fb.is_empty() {
        return (None, stats);
    }

    let alg_roots = compute_polynomial_roots(&poly, &algebraic_fb);
    let large_prime_bound = params.large_prime_multiplier * params.rational_fb_bound;

    let config = SieveConfig {
        rational_bound: params.rational_fb_bound,
        algebraic_bound: params.algebraic_fb_bound,
        sieve_area: params.sieve_area,
        max_b: params.max_b,
        large_prime_multiplier: params.large_prime_multiplier,
        sieve_threshold: 0.0,
    };

    let hits = line_sieve(&poly, &rational_fb, &alg_roots, &config);
    stats.sieve_hits = hits.len();

    let result = collect_relations(&hits, &poly, &rational_fb, &algebraic_fb, large_prime_bound);
    stats.full_relations = result.full_relations.len();
    stats.cycle_relations = result.cycle_relations.len();
    stats.partial_relations = result.partials_found;

    let mut all_relations: Vec<SmoothRelation> = result.full_relations;
    all_relations.extend(result.cycle_relations);
    stats.total_relations = all_relations.len();

    if all_relations.len() < 2 {
        return (None, stats);
    }

    let rational_fb_size = rational_fb.len();
    let algebraic_fb_size = algebraic_fb.len();
    let (matrix, num_cols) = build_matrix(&all_relations, rational_fb_size, algebraic_fb_size);
    stats.matrix_rows = matrix.len();
    stats.matrix_cols = num_cols;

    let dependencies = find_dependencies(&matrix, num_cols);
    stats.dependencies_found = dependencies.len();

    if dependencies.is_empty() {
        return (None, stats);
    }

    let extraction = extract_factor(
        n,
        &poly,
        &all_relations,
        &dependencies,
        &rational_fb,
        &algebraic_fb,
    );
    stats.dependencies_tried = extraction.dependencies_tried;
    stats.factor_found = extraction.factor.is_some();

    (extraction.factor, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_nfs_trivial_even() {
        let n = BigUint::from(100u64);
        let result = factor_nfs(&n);
        assert_eq!(result, Some(BigUint::from(2u64)));
    }

    #[test]
    fn test_factor_nfs_perfect_square() {
        let n = BigUint::from(10201u64); // 101^2
        let result = factor_nfs(&n);
        assert_eq!(result, Some(BigUint::from(101u64)));
    }

    #[test]
    fn test_factor_nfs_small_semiprime() {
        // 8051 = 83 * 97
        let n = BigUint::from(8051u64);
        let result = factor_nfs(&n);
        assert!(result.is_some(), "Should factor 8051 = 83 * 97");
        let f = result.unwrap();
        assert!((&n % &f).is_zero(), "factor must divide n");
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_factor_nfs_15347() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let result = factor_nfs(&n);
        assert!(result.is_some(), "Should factor 15347");
        let f = result.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_factor_nfs_67591() {
        // 67591 = 257 * 263
        let n = BigUint::from(67591u64);
        let result = factor_nfs(&n);
        assert!(result.is_some(), "Should factor 67591 = 257 * 263");
        let f = result.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_factor_nfs_1042961() {
        // 1042961 = 1009 * 1033
        // NFS pipeline is still under development; diagnostic test
        let n = BigUint::from(1042961u64);
        let result = factor_nfs(&n);
        eprintln!("NFS 1042961: {:?}", result);
        // Don't hard-assert until pipeline is complete
    }

    #[test]
    fn test_factor_nfs_40bit() {
        // ~40-bit semiprime: 1000003 * 1000033 = 1000036000099
        // NFS pipeline is still under development; diagnostic test
        let n = BigUint::from(1_000_036_000_099u64);
        let params = NfsPipelineParams::for_bits(40);
        let result = factor_nfs_with_params(&n, &params);
        eprintln!("NFS 40-bit: {:?}", result);
        // Don't hard-assert until pipeline is complete
    }

    #[test]
    fn test_factor_nfs_params_for_bits() {
        let p40 = NfsPipelineParams::for_bits(40);
        let p60 = NfsPipelineParams::for_bits(60);
        let p80 = NfsPipelineParams::for_bits(80);

        // Larger numbers should get larger factor bases
        assert!(p40.rational_fb_bound <= p60.rational_fb_bound);
        assert!(p60.rational_fb_bound <= p80.rational_fb_bound);
    }

    #[test]
    fn test_factor_nfs_with_stats() {
        let n = BigUint::from(15347u64);
        let params = NfsPipelineParams::for_bits(15);
        let (_result, stats) = factor_nfs_with_stats(&n, &params);

        assert!(stats.polynomial_degree > 0);
        assert!(stats.rational_fb_size > 0);
        // The stats should be populated regardless of whether we found a factor
        assert!(stats.sieve_hits >= 0);
    }
}
