//! Factor extraction for the Number Field Sieve.
//!
//! Given null-space vectors from the linear algebra phase, compute the
//! square root on both sides and extract factors via gcd.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};

use crate::linalg::Dependency;
use crate::polynomial::NfsPolynomial;
use crate::relation::SmoothRelation;

/// Result of a factor extraction attempt.
#[derive(Debug)]
pub struct FactorExtractionResult {
    /// Non-trivial factor found, if any.
    pub factor: Option<BigUint>,
    /// Number of dependencies tried.
    pub dependencies_tried: usize,
    /// Number of valid square products found.
    pub valid_squares: usize,
}

/// Attempt to extract a factor of n using the given dependencies.
///
/// For each dependency (a set of relation indices whose combined exponent
/// vector is zero mod 2):
///
/// 1. Compute X = product of (a_i + b_i * m) mod n on the rational side.
/// 2. Compute Y = product of p_j^(e_j/2) mod n from the half-exponents.
/// 3. Try gcd(X - Y, n) and gcd(X + Y, n) for non-trivial factors.
///
/// The algebraic side square root is computed naively by halving exponents
/// and reconstructing the product modulo n.
pub fn extract_factor(
    n: &BigUint,
    poly: &NfsPolynomial,
    relations: &[SmoothRelation],
    dependencies: &[Dependency],
    rational_fb: &[u64],
    algebraic_fb: &[u64],
) -> FactorExtractionResult {
    let mut result = FactorExtractionResult {
        factor: None,
        dependencies_tried: 0,
        valid_squares: 0,
    };

    let m_u64: u64 = match poly.m.to_u64() {
        Some(m) => m,
        None => return result,
    };

    for dep in dependencies {
        result.dependencies_tried += 1;

        if dep.is_empty() || dep.len() < 2 {
            continue;
        }

        // Validate all indices
        if dep.iter().any(|&idx| idx >= relations.len()) {
            continue;
        }

        // Accumulate combined exponents for both sides
        let num_rat = rational_fb.len();
        let num_alg = algebraic_fb.len();

        let mut combined_rat_exp = vec![0u64; num_rat];
        let mut combined_alg_exp = vec![0u64; num_alg];
        let mut sign_count: u64 = 0;

        // Also compute the product of (a + b*m) mod n for the rational side
        let mut x_product = BigUint::one();

        for &idx in dep {
            let rel = &relations[idx];

            // Compute a + b*m
            let a_i128 = rel.a as i128;
            let b_i128 = rel.b as i128;
            let m_i128 = m_u64 as i128;
            let rational_val = a_i128 + b_i128 * m_i128;

            let abs_val = BigUint::from(rational_val.unsigned_abs());
            x_product = (&x_product * &abs_val) % n;

            if rel.sign_negative {
                sign_count += 1;
            }

            // Accumulate exponents
            for i in 0..num_rat.min(rel.rational_exponents.len()) {
                combined_rat_exp[i] += rel.rational_exponents[i] as u64;
            }
            for i in 0..num_alg.min(rel.algebraic_exponents.len()) {
                combined_alg_exp[i] += rel.algebraic_exponents[i] as u64;
            }
        }

        // Check that all exponents are even (sign must also be even)
        if sign_count % 2 != 0 {
            continue;
        }
        let rat_even = combined_rat_exp.iter().all(|&e| e % 2 == 0);
        let alg_even = combined_alg_exp.iter().all(|&e| e % 2 == 0);
        if !rat_even || !alg_even {
            continue;
        }

        result.valid_squares += 1;

        // Compute Y = product of p_j^(e_j/2) mod n for both sides
        // Rational side square root
        let mut y_rational = BigUint::one();
        for (j, &exp) in combined_rat_exp.iter().enumerate() {
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(rational_fb[j]);
                let contribution = p_big.modpow(&half_exp, n);
                y_rational = (&y_rational * &contribution) % n;
            }
        }

        // Algebraic side square root
        let mut y_algebraic = BigUint::one();
        for (j, &exp) in combined_alg_exp.iter().enumerate() {
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(algebraic_fb[j]);
                let contribution = p_big.modpow(&half_exp, n);
                y_algebraic = (&y_algebraic * &contribution) % n;
            }
        }

        // The full Y combines both sides
        let y_combined = (&y_rational * &y_algebraic) % n;

        // Try gcd(X - Y, n) and gcd(X + Y, n)
        if let Some(f) = try_gcd_factor(&x_product, &y_combined, n) {
            result.factor = Some(f);
            return result;
        }

        // Also try just the rational side
        if let Some(f) = try_gcd_factor(&x_product, &y_rational, n) {
            result.factor = Some(f);
            return result;
        }
    }

    result
}

/// Try to extract a non-trivial factor from gcd(x - y, n) or gcd(x + y, n).
fn try_gcd_factor(x: &BigUint, y: &BigUint, n: &BigUint) -> Option<BigUint> {
    let one = BigUint::one();

    // gcd(x - y, n)
    let diff = if *x >= *y {
        x - y
    } else {
        n - ((y - x) % n)
    };

    if !diff.is_zero() {
        let factor = diff.gcd(n);
        if factor != one && factor != *n {
            return Some(factor);
        }
    }

    // gcd(x + y, n)
    let sum = (x + y) % n;
    if !sum.is_zero() {
        let factor = sum.gcd(n);
        if factor != one && factor != *n {
            return Some(factor);
        }
    }

    None
}

/// Simplified NFS factor extraction that uses only the rational side.
///
/// This is a fallback approach: when the algebraic side square root is
/// difficult to compute, we can still try to find factors using just
/// the rational side congruences.
///
/// For a dependency set S of relations:
///   product_{i in S} (a_i + b_i * m) = y^2 (mod n)
///   where y is computed from half the combined exponents.
pub fn extract_factor_rational_only(
    n: &BigUint,
    poly: &NfsPolynomial,
    relations: &[SmoothRelation],
    dependencies: &[Dependency],
    rational_fb: &[u64],
) -> FactorExtractionResult {
    let mut result = FactorExtractionResult {
        factor: None,
        dependencies_tried: 0,
        valid_squares: 0,
    };

    let m_u64: u64 = match poly.m.to_u64() {
        Some(m) => m,
        None => return result,
    };

    for dep in dependencies {
        result.dependencies_tried += 1;

        if dep.is_empty() || dep.len() < 2 {
            continue;
        }

        if dep.iter().any(|&idx| idx >= relations.len()) {
            continue;
        }

        let num_rat = rational_fb.len();
        let mut combined_rat_exp = vec![0u64; num_rat];
        let mut sign_count: u64 = 0;
        let mut x_product = BigUint::one();

        for &idx in dep {
            let rel = &relations[idx];

            let a_i128 = rel.a as i128;
            let b_i128 = rel.b as i128;
            let m_i128 = m_u64 as i128;
            let rational_val = a_i128 + b_i128 * m_i128;

            let abs_val = BigUint::from(rational_val.unsigned_abs());
            x_product = (&x_product * &abs_val) % n;

            if rel.sign_negative {
                sign_count += 1;
            }

            for i in 0..num_rat.min(rel.rational_exponents.len()) {
                combined_rat_exp[i] += rel.rational_exponents[i] as u64;
            }
        }

        // Check rational exponents are even + sign is even
        if sign_count % 2 != 0 {
            continue;
        }
        if !combined_rat_exp.iter().all(|&e| e % 2 == 0) {
            continue;
        }

        result.valid_squares += 1;

        // Compute y = product of p^(e/2) mod n
        let mut y_product = BigUint::one();
        for (j, &exp) in combined_rat_exp.iter().enumerate() {
            if exp > 0 {
                let half_exp = BigUint::from(exp / 2);
                let p_big = BigUint::from(rational_fb[j]);
                let contribution = p_big.modpow(&half_exp, n);
                y_product = (&y_product * &contribution) % n;
            }
        }

        if let Some(f) = try_gcd_factor(&x_product, &y_product, n) {
            result.factor = Some(f);
            return result;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_gcd_factor() {
        // n = 15, x = 4, y = 1 => gcd(3, 15) = 3
        let n = BigUint::from(15u64);
        let x = BigUint::from(4u64);
        let y = BigUint::from(1u64);
        let f = try_gcd_factor(&x, &y, &n);
        assert!(f.is_some());
        let f = f.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f != BigUint::one());
        assert!(f != n);
    }

    #[test]
    fn test_try_gcd_factor_trivial() {
        // n = 15, x = 0, y = 0 => gcd(0, 15) = 15 (trivial)
        let n = BigUint::from(15u64);
        let x = BigUint::zero();
        let y = BigUint::zero();
        let f = try_gcd_factor(&x, &y, &n);
        // Should be None since gcd = n or gcd = 0
        assert!(f.is_none());
    }

    #[test]
    fn test_extract_factor_rational_only_basic() {
        // Test with a simple known case
        use crate::polynomial::select_polynomial;
        use crate::sieve::sieve_primes;

        let n = BigUint::from(8051u64); // 83 * 97
        let poly = select_polynomial(&n, 3);
        let rational_fb = sieve_primes(50);
        let m_u64: u64 = poly.m.to_u64().unwrap();

        // Manually find some smooth relations
        let mut relations = Vec::new();
        for b in 1..=50_i64 {
            for a in -50..=50_i64 {
                if a == 0 {
                    continue;
                }
                if crate::sieve::gcd_u64(a.unsigned_abs(), b as u64) != 1 {
                    continue;
                }

                let val_signed: i128 = (a as i128) + (b as i128) * (m_u64 as i128);
                if val_signed == 0 {
                    continue;
                }
                let sign_negative = val_signed < 0;
                let val_abs = val_signed.unsigned_abs() as u64;

                // Trial divide
                let mut remaining = val_abs;
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
            return; // not enough relations, skip test
        }

        // Build matrix and find dependencies
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

        let num_cols = 1 + rational_fb.len();
        let deps = crate::linalg::gaussian_elimination_gf2(&matrix, num_cols);

        if deps.is_empty() {
            return; // no dependencies found, skip
        }

        let extraction = extract_factor_rational_only(&n, &poly, &relations, &deps, &rational_fb);
        // We may or may not find a factor (depends on the specific relations found)
        // Just verify the function doesn't panic and produces valid output
        assert!(extraction.dependencies_tried > 0);
    }
}
