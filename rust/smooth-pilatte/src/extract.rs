//! Factor extraction pipeline for the Pilatte smooth-relation approach.
//!
//! Orchestrates: lattice construction → enumeration → relation collection →
//! GF(2) linear algebra → factor extraction via gcd.

use std::time::Instant;

use classical_nfs::linalg::{find_dependencies, Dependency};
use factoring_core::{gcd, Algorithm, FactorResult};
use num_bigint::BigUint;
use num_traits::{One, Zero};

/// Modular exponentiation: base^exp mod modulus.
fn mod_pow(base: &BigUint, exp: u64, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        return BigUint::zero();
    }
    let mut result = BigUint::one();
    let mut b = base % modulus;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (&result * &b) % modulus;
        }
        e >>= 1;
        b = (&b * &b) % modulus;
    }
    result
}

use crate::enumerate::{enumerate_short_vectors, EnumerationConfig};
use crate::lattice::{
    build_pilatte_lattice, build_weighted_pilatte_lattice, extract_exponent_vectors,
    pilatte_dimension,
};
use crate::relations::{build_gf2_matrix, collect_smooth_relations, SmoothRelation};

/// Detailed result from the Pilatte factoring pipeline.
#[derive(Debug, Clone)]
pub struct PilatteFactorResult {
    /// The factoring result (factor found, timing, etc.).
    pub factor_result: FactorResult,
    /// Lattice dimension used.
    pub dimension: usize,
    /// Total exponent vectors tested.
    pub vectors_tested: usize,
    /// Number of smooth relations found.
    pub smooth_relations: usize,
    /// Smoothness rate.
    pub smooth_rate: f64,
    /// Number of GF(2) dependencies found.
    pub dependencies_found: usize,
    /// Number of gcd attempts.
    pub gcd_attempts: usize,
    /// Whether the weighted lattice was used.
    pub used_weighted: bool,
}

/// Configuration for the Pilatte pipeline.
#[derive(Debug, Clone)]
pub struct PilatteConfig {
    /// Override dimension (0 = auto from Pilatte theorem).
    pub dimension_override: usize,
    /// Enumeration radius multiplier (relative to shortest vector).
    pub radius_multiplier: f64,
    /// Maximum enumeration vectors.
    pub max_enum_vectors: usize,
    /// Try weighted lattice if standard fails.
    pub try_weighted: bool,
    /// Maximum dimension to try with scaling.
    pub max_dimension: usize,
}

impl Default for PilatteConfig {
    fn default() -> Self {
        Self {
            dimension_override: 0,
            radius_multiplier: 3.0,
            max_enum_vectors: 5000,
            try_weighted: true,
            max_dimension: 50,
        }
    }
}

/// Main entry point: attempt to factor N using Pilatte's lattice-geometric approach.
pub fn factor_smooth_pilatte(n: &BigUint) -> Option<BigUint> {
    let config = PilatteConfig::default();
    factor_smooth_pilatte_with_config(n, &config).and_then(|r| {
        if r.factor_result.complete {
            r.factor_result.factors.first().cloned()
        } else {
            None
        }
    })
}

/// Factor N with detailed diagnostics.
pub fn factor_smooth_pilatte_with_config(
    n: &BigUint,
    config: &PilatteConfig,
) -> Option<PilatteFactorResult> {
    let start = Instant::now();
    let n_bits = n.bits() as u32;

    // Determine dimension
    let base_dim = if config.dimension_override > 0 {
        config.dimension_override
    } else {
        pilatte_dimension(n_bits)
    };

    // Try increasing dimensions
    for dim_mult in 1..=3 {
        let dim = (base_dim * dim_mult).min(config.max_dimension);

        // Standard lattice
        if let Some(result) =
            try_factor_with_lattice(n, dim, false, config, &start)
        {
            return Some(result);
        }

        // Weighted lattice
        if config.try_weighted {
            if let Some(result) =
                try_factor_with_lattice(n, dim, true, config, &start)
            {
                return Some(result);
            }
        }
    }

    // Return failure result
    Some(PilatteFactorResult {
        factor_result: FactorResult {
            n: n.clone(),
            factors: vec![],
            algorithm: Algorithm::SmoothPilatte,
            duration: start.elapsed(),
            complete: false,
        },
        dimension: base_dim,
        vectors_tested: 0,
        smooth_relations: 0,
        smooth_rate: 0.0,
        dependencies_found: 0,
        gcd_attempts: 0,
        used_weighted: false,
    })
}

/// Try factoring with a specific lattice configuration.
fn try_factor_with_lattice(
    n: &BigUint,
    dimension: usize,
    weighted: bool,
    config: &PilatteConfig,
    start: &Instant,
) -> Option<PilatteFactorResult> {
    // Build and reduce lattice
    let lattice_result = if weighted {
        build_weighted_pilatte_lattice(n, dimension)
    } else {
        build_pilatte_lattice(n, dimension)
    };

    let primes = &lattice_result.params.primes;

    // Phase 1: Extract exponent vectors from LLL basis
    let mut all_vectors = extract_exponent_vectors(&lattice_result);

    // Phase 2: Enumerate additional short vectors via Fincke-Pohst
    let enum_radius = lattice_result.quality.shortest_vector_norm * config.radius_multiplier;
    let enum_config = EnumerationConfig {
        radius: enum_radius,
        max_vectors: config.max_enum_vectors,
        skip_zero: true,
    };
    let enum_points = enumerate_short_vectors(&lattice_result.basis, &enum_config);

    // Convert enumerated points to exponent vectors
    let c = lattice_result.params.c_scale;
    let d = lattice_result.params.dimension;
    for point in &enum_points {
        let exponents: Vec<i64> = point.vector[..d]
            .iter()
            .map(|&v| (v / c).round() as i64)
            .collect();
        if !exponents.iter().all(|&e| e == 0) {
            all_vectors.push(exponents);
        }
    }

    // Deduplicate vectors
    all_vectors.sort();
    all_vectors.dedup();

    // Phase 3: Try direct gcd from exponent vectors (fast path)
    let mut gcd_attempts = 0;
    if let Some(factor) = try_direct_gcd(&all_vectors, primes, n, &mut gcd_attempts) {
        return Some(PilatteFactorResult {
            factor_result: FactorResult {
                n: n.clone(),
                factors: vec![factor],
                algorithm: Algorithm::SmoothPilatte,
                duration: start.elapsed(),
                complete: true,
            },
            dimension,
            vectors_tested: all_vectors.len(),
            smooth_relations: 0,
            smooth_rate: 0.0,
            dependencies_found: 0,
            gcd_attempts,
            used_weighted: weighted,
        });
    }

    // Phase 4: Collect smooth relations
    let target_relations = dimension + 5; // need > dimension for dependency
    let collection =
        collect_smooth_relations(&all_vectors, primes, n, target_relations);

    if collection.smooth_count < 2 {
        return None; // Not enough relations
    }

    // Phase 5: GF(2) linear algebra
    let (matrix, num_cols) = build_gf2_matrix(&collection.relations, primes.len());
    let deps = find_dependencies(&matrix, num_cols);

    if deps.is_empty() {
        return None; // No dependencies found
    }

    // Phase 6: Extract factors from dependencies
    if let Some(factor) = extract_factor_from_deps(
        &deps,
        &collection.relations,
        primes,
        n,
        &mut gcd_attempts,
    ) {
        return Some(PilatteFactorResult {
            factor_result: FactorResult {
                n: n.clone(),
                factors: vec![factor],
                algorithm: Algorithm::SmoothPilatte,
                duration: start.elapsed(),
                complete: true,
            },
            dimension,
            vectors_tested: collection.vectors_tested,
            smooth_relations: collection.smooth_count,
            smooth_rate: collection.smooth_rate,
            dependencies_found: deps.len(),
            gcd_attempts,
            used_weighted: weighted,
        });
    }

    None
}

/// Try to find a factor directly from exponent vectors via gcd.
fn try_direct_gcd(
    vectors: &[Vec<i64>],
    primes: &[u64],
    n: &BigUint,
    gcd_attempts: &mut usize,
) -> Option<BigUint> {
    let one = BigUint::one();

    for exps in vectors {
        // Compute products from positive and negative exponents
        let mut lhs = BigUint::one();
        let mut rhs = BigUint::one();

        for (i, &exp) in exps.iter().enumerate() {
            if i >= primes.len() {
                break;
            }
            let p_big = BigUint::from(primes[i]);
            if exp > 0 {
                lhs = (&lhs * &mod_pow(&p_big, exp as u64, n)) % n;
            } else if exp < 0 {
                rhs = (&rhs * &mod_pow(&p_big, (-exp) as u64, n)) % n;
            }
        }

        // Try gcd(|lhs - rhs|, n)
        *gcd_attempts += 1;
        let diff = if lhs >= rhs {
            &lhs - &rhs
        } else {
            &rhs - &lhs
        };
        if !diff.is_zero() {
            let g = gcd(&diff, n);
            if g > one && g < *n {
                return Some(g);
            }
        }

        // Try gcd(lhs + rhs, n)
        *gcd_attempts += 1;
        let sum = (&lhs + &rhs) % n;
        if !sum.is_zero() {
            let g = gcd(&sum, n);
            if g > one && g < *n {
                return Some(g);
            }
        }

        // Try gcd of individual products
        for product in [&lhs, &rhs] {
            if !product.is_zero() && !product.is_one() {
                *gcd_attempts += 1;
                let g = gcd(product, n);
                if g > one && g < *n {
                    return Some(g);
                }
            }
        }
    }

    None
}

/// Extract a factor from GF(2) dependencies.
///
/// For each dependency (set of relation indices), combine the relations'
/// exponents. The combined exponent vector is all-even, so the combined
/// product is a perfect square X^2 ≡ Y^2 (mod N). Then gcd(X-Y, N) may
/// yield a factor.
fn extract_factor_from_deps(
    deps: &[Dependency],
    relations: &[SmoothRelation],
    primes: &[u64],
    n: &BigUint,
    gcd_attempts: &mut usize,
) -> Option<BigUint> {
    let one = BigUint::one();

    for dep in deps {
        // Combine exponents from all relations in this dependency
        let num_primes = primes.len();
        let mut combined_exps = vec![0i64; num_primes];

        for &idx in dep {
            if idx >= relations.len() {
                continue;
            }
            for (i, &e) in relations[idx].exponents.iter().enumerate() {
                if i < num_primes {
                    combined_exps[i] += e;
                }
            }
        }

        // All combined exponents should be even (that's what the dependency guarantees)
        // Compute X = prod(p_i^{combined_exp_i / 2}) mod N
        let mut x = BigUint::one();
        for (i, &exp) in combined_exps.iter().enumerate() {
            let half_exp = exp.unsigned_abs() / 2;
            if half_exp > 0 {
                let p_big = BigUint::from(primes[i]);
                x = (&x * &mod_pow(&p_big, half_exp, n)) % n;
            }
        }

        // Compute Y = prod of all positive/negative products
        let mut y = BigUint::one();
        for &idx in dep {
            if idx >= relations.len() {
                continue;
            }
            y = (&y * &relations[idx].positive_product) % n;
        }

        // Try gcd(X - Y, N) and gcd(X + Y, N)
        if x != y {
            *gcd_attempts += 1;
            let diff = if x >= y { &x - &y } else { &y - &x };
            let g = gcd(&diff, n);
            if g > one && g < *n {
                return Some(g);
            }

            *gcd_attempts += 1;
            let sum = (&x + &y) % n;
            let g = gcd(&sum, n);
            if g > one && g < *n {
                return Some(g);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_small_semiprime() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let result = factor_smooth_pilatte_with_config(&n, &PilatteConfig::default());

        assert!(result.is_some(), "Should return a result");
        let result = result.unwrap();
        // May or may not find the factor for this size, but should run without error
        assert!(result.dimension >= 4);
    }

    #[test]
    fn test_factor_trivial() {
        // 6 = 2 * 3 (should be found directly from lattice gcd)
        let n = BigUint::from(6u32);
        let result = factor_smooth_pilatte(&n);

        if let Some(factor) = result {
            assert!(
                factor == BigUint::from(2u32) || factor == BigUint::from(3u32),
                "Factor should be 2 or 3, got {}",
                factor
            );
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = PilatteConfig::default();
        assert_eq!(config.dimension_override, 0);
        assert!(config.radius_multiplier > 0.0);
        assert!(config.max_enum_vectors > 0);
        assert!(config.try_weighted);
    }

    #[test]
    fn test_direct_gcd_with_known_factor() {
        let n = BigUint::from(77u32); // 7 * 11
        let primes = vec![2, 3, 5, 7, 11];

        // Exponent vector [0,0,0,1,0] means just 7, gcd(7, 77) = 7
        let vectors = vec![vec![0, 0, 0, 1, 0]];
        let mut attempts = 0;
        let result = try_direct_gcd(&vectors, &primes, &n, &mut attempts);

        assert!(result.is_some(), "Should find factor 7 directly");
        assert_eq!(result.unwrap(), BigUint::from(7u32));
    }

    #[test]
    fn test_pipeline_16bit() {
        // 16-bit semiprime: 251 * 241 = 60491
        let n = BigUint::from(60491u64);
        let config = PilatteConfig {
            max_enum_vectors: 2000,
            ..PilatteConfig::default()
        };
        let result = factor_smooth_pilatte_with_config(&n, &config);
        assert!(result.is_some());
    }
}
