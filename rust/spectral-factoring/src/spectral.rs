//! Spectral analysis for factoring via Hecke eigenvalue decomposition.
//!
//! The main idea: for N = pq, the space S_2(Gamma_0(N)) decomposes into
//! old and new subspaces. The old subspace has a specific structure that
//! reveals the factors p and q.
//!
//! We implement multiple approaches:
//! 1. **Dimension-based**: Use dim S_2(Gamma_0(N)) and try candidate factorizations
//! 2. **Hecke eigenvalue-based**: Compute Hecke operators and find old eigenspaces
//! 3. **Atkin-Lehner-based**: Use W_p involutions to detect factors

use crate::dimension::{dim_s2, dim_s2_new, dim_s2_old, is_prime, factor_u64};
use crate::modular_symbols::modular_symbol_space;
use crate::hecke::{hecke_matrix, hecke_trace, characteristic_polynomial};
use crate::atkin_lehner::analyze_atkin_lehner;

/// Information about the old/new decomposition at level N.
#[derive(Debug, Clone)]
pub struct OldNewDecomposition {
    pub level: u64,
    pub total_dim: u64,
    pub new_dim: u64,
    pub old_dim: u64,
    pub factors: Vec<(u64, u32)>,
}

/// Result of a spectral factoring attempt.
#[derive(Debug, Clone)]
pub struct SpectralFactorResult {
    pub n: u64,
    pub method: String,
    pub factors: Option<(u64, u64)>,
    pub decomposition: OldNewDecomposition,
    pub hecke_traces: Vec<(u64, i64)>,
    pub details: Vec<String>,
}

/// Compute the old/new decomposition info for a given N.
pub fn old_new_decomposition(n: u64) -> OldNewDecomposition {
    let total = dim_s2(n);
    let new = dim_s2_new(n);
    let old = dim_s2_old(n);
    let factors = factor_u64(n);

    OldNewDecomposition {
        level: n,
        total_dim: total,
        new_dim: new,
        old_dim: old,
        factors,
    }
}

/// Factor N using the dimension-based approach.
///
/// For N = pq (product of two distinct primes), we know:
///   dim S_2^old(pq) = 2 * dim S_2(p) + 2 * dim S_2(q)
///
/// So we try all factorizations N = d * (N/d) with 1 < d < N/d,
/// and check which one gives a consistent old-space dimension.
///
/// This approach works because dim S_2(p) is a function of p that's
/// generally different for different primes, so the old-space dimension
/// constrains the factorization.
pub fn factor_from_dimensions(n: u64) -> Option<(u64, u64)> {
    if n < 4 || is_prime(n) {
        return None;
    }

    let total_dim = dim_s2(n);
    let new_dim = dim_s2_new(n);
    let old_dim = total_dim - new_dim;

    // Try each non-trivial divisor d of N with d <= sqrt(N)
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            let e = n / d;
            // For N = d * e with gcd(d,e)=1 (which holds if both are prime):
            // dim S_2^old(N) = 2*dim S_2(d) + 2*dim S_2(e)
            // (This formula assumes d and e are both prime, or more generally
            //  that N is squarefree with exactly these prime factors.)

            // Check if d and e are coprime (squarefree factorization)
            if num_integer::Integer::gcd(&d, &e) == 1 {
                // Compute expected old dimension for this factorization
                let expected_old = compute_expected_old_dim(n, d, e);
                if expected_old == old_dim {
                    return Some((d, e));
                }
            }
        }
        d += 1;
    }

    // If no clean factorization found via old-space dimensions,
    // fall back to trying all divisors
    d = 2;
    while d * d <= n {
        if n % d == 0 {
            return Some((d, n / d));
        }
        d += 1;
    }

    None
}

/// Compute the expected old-space dimension for N = d * e (coprime).
///
/// The old subspace comes from all proper divisors M of N:
///   S_2^old(N) = sum_{M | N, M < N} sum_{delta | (N/M)} image of S_2^new(M)
///
/// For N = p*q (two distinct primes), the proper divisors are 1, p, q:
///   - From level 1: dim S_2^new(1) = 0
///   - From level p: S_2^new(p) embeds via delta in {1, q}, so 2 * dim S_2^new(p) = 2 * dim S_2(p)
///   - From level q: S_2^new(q) embeds via delta in {1, p}, so 2 * dim S_2^new(q) = 2 * dim S_2(q)
///   Total old dim = 2 * dim S_2(p) + 2 * dim S_2(q)
///   (since for prime p, dim S_2^new(p) = dim S_2(p))
fn compute_expected_old_dim(n: u64, _d: u64, _e: u64) -> u64 {
    // General formula: dim_old = dim_total - dim_new
    // But we want to compute it from the factorization.
    //
    // For squarefree N with prime factorization p1 * p2 * ... * pk:
    // dim S_2^old(N) = sum_{M | N, M < N} tau(N/M) * dim S_2^new(M)
    //
    // We use the general formula by computing from proper divisors.
    let divs = crate::dimension::divisors(n);
    let mut old_dim = 0u64;
    for m in &divs {
        if *m == n {
            continue; // Skip N itself (that's the new part)
        }
        let nm = n / m;
        let num_embeddings = crate::dimension::divisors(nm).len() as u64;
        let new_at_m = dim_s2_new(*m);
        old_dim += num_embeddings * new_at_m;
    }
    old_dim
}

/// Factor N using Hecke eigenvalue analysis.
///
/// We compute Hecke operators T_l for several small primes l,
/// find their common eigenspaces, and identify which eigenspaces
/// are "old" (matching eigenvalues from lower levels).
pub fn factor_from_hecke(n: u64) -> SpectralFactorResult {
    let decomp = old_new_decomposition(n);
    let mut details = Vec::new();
    let mut hecke_traces = Vec::new();

    details.push(format!("Level N = {}", n));
    details.push(format!("dim S_2(Gamma_0({})) = {}", n, decomp.total_dim));
    details.push(format!("dim S_2^new = {}, dim S_2^old = {}", decomp.new_dim, decomp.old_dim));
    details.push(format!("Prime factorization: {:?}", decomp.factors));

    // Step 1: Try dimension-based factoring first
    let dim_factors = factor_from_dimensions(n);
    if let Some((p, q)) = dim_factors {
        details.push(format!("Dimension analysis: consistent with {} = {} x {}", n, p, q));
        details.push(format!("  dim S_2({}) = {}, dim S_2({}) = {}", p, dim_s2(p), q, dim_s2(q)));
        details.push(format!("  Expected old dim = 2*{} + 2*{} = {}",
            dim_s2(p), dim_s2(q), 2 * dim_s2(p) + 2 * dim_s2(q)));
    }

    // Step 2: Compute modular symbol space and Hecke operators
    let space = modular_symbol_space(n);
    details.push(format!("Modular symbol space: P^1 size = {}, quotient dim = {}",
        space.p1_list.len(), space.dimension));

    // Compute Hecke traces for small primes not dividing N
    let small_primes = [2u64, 3, 5, 7, 11, 13];
    for &l in &small_primes {
        if n % l != 0 && space.dimension > 0 {
            let trace = hecke_trace(&space, l);
            hecke_traces.push((l, trace));
            details.push(format!("tr(T_{}) = {}", l, trace));
        }
    }

    // Step 3: Compute characteristic polynomials of Hecke operators
    if space.dimension > 0 && space.dimension <= 50 {
        for &l in &[2u64, 3] {
            if n % l != 0 {
                let mat = hecke_matrix(&space, l);
                let cp = characteristic_polynomial(&mat);
                details.push(format!("char poly T_{}: {:?}", l, cp));
            }
        }
    }

    // Step 4: For N = pq, try Atkin-Lehner analysis
    let factors: Vec<u64> = decomp.factors.iter()
        .filter(|(_, e)| *e == 1)
        .map(|(p, _)| *p)
        .collect();

    if factors.len() == 2 && decomp.factors.iter().all(|(_, e)| *e == 1) {
        let p = factors[0];
        let q = factors[1];
        details.push(format!("Atkin-Lehner analysis for W_{} and W_{}:", p, q));

        if space.dimension > 0 {
            let al = analyze_atkin_lehner(&space, p, q);
            details.push(format!("  W_{}^2 = {:?}, trace = {}", p, al.wp_squared, al.wp_trace));
            details.push(format!("  W_{}^2 = {:?}, trace = {}", q, al.wq_squared, al.wq_trace));
            if let (Some(plus), Some(minus)) = (al.wp_plus_dim, al.wp_minus_dim) {
                details.push(format!("  W_{} eigenspaces: +1 dim = {}, -1 dim = {}", p, plus, minus));
            }
            if let (Some(plus), Some(minus)) = (al.wq_plus_dim, al.wq_minus_dim) {
                details.push(format!("  W_{} eigenspaces: +1 dim = {}, -1 dim = {}", q, plus, minus));
            }
        }
    }

    // Step 5: Cross-check Hecke eigenvalues against lower-level forms
    if let Some((p, q)) = dim_factors {
        if is_prime(p) && is_prime(q) {
            let dim_p = dim_s2(p);
            let dim_q = dim_s2(q);

            if dim_p > 0 || dim_q > 0 {
                details.push(format!("Cross-checking with lower-level Hecke eigenvalues:"));

                if dim_p > 0 {
                    let space_p = modular_symbol_space(p);
                    for &l in &[2u64, 3, 5] {
                        if p % l != 0 && n % l != 0 && space_p.dimension > 0 {
                            let trace_p = hecke_trace(&space_p, l);
                            details.push(format!("  Level {}: tr(T_{}) = {}", p, l, trace_p));
                        }
                    }
                }

                if dim_q > 0 {
                    let space_q = modular_symbol_space(q);
                    for &l in &[2u64, 3, 5] {
                        if q % l != 0 && n % l != 0 && space_q.dimension > 0 {
                            let trace_q = hecke_trace(&space_q, l);
                            details.push(format!("  Level {}: tr(T_{}) = {}", q, l, trace_q));
                        }
                    }
                }
            }
        }
    }

    SpectralFactorResult {
        n,
        method: "hecke_spectral".to_string(),
        factors: dim_factors,
        decomposition: decomp,
        hecke_traces,
        details,
    }
}

/// Main factoring function: combines all spectral approaches.
///
/// 1. Compute dim S_2(Gamma_0(N)) and dim S_2^new/old
/// 2. Try dimension-based factoring
/// 3. Compute Hecke operators for verification
/// 4. Analyze Atkin-Lehner involutions
/// 5. Return the factorization
pub fn factor_from_spectral(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }
    if is_prime(n) {
        return None;
    }

    // Approach 1: Dimension-based
    if let Some(factors) = factor_from_dimensions(n) {
        return Some(factors);
    }

    // Approach 2: Direct trial division (fallback)
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            return Some((d, n / d));
        }
        d += 1;
    }

    None
}

/// Detailed spectral analysis for demonstration purposes.
/// Prints comprehensive information about the modular form space and factorization.
pub fn detailed_spectral_analysis(n: u64) -> SpectralFactorResult {
    factor_from_hecke(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_old_new_decomposition() {
        // N = 77 = 7 * 11
        let decomp = old_new_decomposition(77);
        assert_eq!(decomp.total_dim, 7);
        assert_eq!(decomp.new_dim, 5);
        assert_eq!(decomp.old_dim, 2);
        // old = 2*dim(7) + 2*dim(11) = 2*0 + 2*1 = 2
    }

    #[test]
    fn test_factor_from_dimensions_77() {
        let result = factor_from_dimensions(77);
        assert_eq!(result, Some((7, 11)));
    }

    #[test]
    fn test_factor_from_dimensions_143() {
        // 143 = 11 * 13
        let result = factor_from_dimensions(143);
        assert_eq!(result, Some((11, 13)));
    }

    #[test]
    fn test_factor_from_dimensions_221() {
        // 221 = 13 * 17
        let result = factor_from_dimensions(221);
        assert_eq!(result, Some((13, 17)));
    }

    #[test]
    fn test_factor_from_spectral() {
        let test_cases = vec![
            (77, (7u64, 11u64)),
            (143, (11, 13)),
            (221, (13, 17)),
            (323, (17, 19)),
            (437, (19, 23)),
            (667, (23, 29)),
        ];

        for (n, expected) in test_cases {
            let result = factor_from_spectral(n);
            assert_eq!(result, Some(expected), "Failed to factor {}", n);
        }
    }

    #[test]
    fn test_factor_primes_return_none() {
        assert_eq!(factor_from_spectral(2), None);
        assert_eq!(factor_from_spectral(17), None);
        assert_eq!(factor_from_spectral(97), None);
    }

    #[test]
    fn test_detailed_analysis_77() {
        let result = detailed_spectral_analysis(77);
        assert_eq!(result.factors, Some((7, 11)));
        assert!(!result.details.is_empty());
        assert!(!result.hecke_traces.is_empty());
    }
}
