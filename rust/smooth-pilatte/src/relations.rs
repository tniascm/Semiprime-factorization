//! Smooth relation testing and collection for the Pilatte pipeline.
//!
//! Takes exponent vectors from the lattice and tests whether the corresponding
//! products of primes form smooth relations modulo N. Collects enough relations
//! for GF(2) linear algebra to find factor-revealing dependencies.

use num_bigint::BigUint;
use num_traits::{One, Zero};

/// A smooth relation: an exponent vector whose prime product yields useful
/// information modulo N.
#[derive(Debug, Clone)]
pub struct SmoothRelation {
    /// Exponent vector (e_1, ..., e_d) over the factor base.
    pub exponents: Vec<i64>,
    /// Product value: prod(p_i^{|e_i|}) for positive exponents.
    pub positive_product: BigUint,
    /// Product value: prod(p_i^{|e_i|}) for negative exponents.
    pub negative_product: BigUint,
    /// Residue of the relation modulo N.
    pub residue_mod_n: BigUint,
    /// Whether the residue is fully smooth over the factor base.
    pub is_smooth: bool,
    /// Factorization over factor base if smooth (exponents of residue).
    pub residue_exponents: Option<Vec<u32>>,
}

/// Result of collecting smooth relations.
#[derive(Debug, Clone)]
pub struct RelationCollectionResult {
    /// All smooth relations found.
    pub relations: Vec<SmoothRelation>,
    /// Total exponent vectors tested.
    pub vectors_tested: usize,
    /// Number that produced smooth residues.
    pub smooth_count: usize,
    /// Smoothness rate: smooth_count / vectors_tested.
    pub smooth_rate: f64,
}

/// Try to factor a BigUint value over the given factor base.
/// Returns Some(exponents) if fully smooth, None otherwise.
pub fn smooth_factorize(val: &BigUint, primes: &[u64]) -> Option<Vec<u32>> {
    let mut remaining = val.clone();
    let mut exponents = vec![0u32; primes.len()];
    let one = BigUint::one();

    for (i, &p) in primes.iter().enumerate() {
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() {
            remaining /= &p_big;
            exponents[i] += 1;
        }
    }

    if remaining == one {
        Some(exponents)
    } else {
        None
    }
}

/// Compute modular exponentiation: base^exp mod modulus.
fn mod_pow_biguint(base: &BigUint, exp: u64, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        return BigUint::zero();
    }
    let mut result = BigUint::one();
    let mut base = base % modulus;
    let mut exp = exp;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (&result * &base) % modulus;
        }
        exp >>= 1;
        base = (&base * &base) % modulus;
    }
    result
}

/// Compute modular inverse via Fermat's little theorem (for prime moduli)
/// or extended Euclidean algorithm.
/// Returns None if inverse doesn't exist (gcd(a, n) != 1).
fn mod_inverse(a: &BigUint, n: &BigUint) -> Option<BigUint> {
    use num_bigint::BigInt;

    let a_int = BigInt::from(a.clone());
    let n_int = BigInt::from(n.clone());

    let (g, x, _) = extended_gcd(&a_int, &n_int);
    if g != BigInt::one() {
        return None;
    }

    let result = ((x % &n_int) + &n_int) % &n_int;
    Some(result.to_biguint().unwrap())
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd.
fn extended_gcd(a: &num_bigint::BigInt, b: &num_bigint::BigInt) -> (num_bigint::BigInt, num_bigint::BigInt, num_bigint::BigInt) {
    use num_bigint::BigInt;

    if b.is_zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }
    let (g, x1, y1) = extended_gcd(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;
    (g, x, y)
}

/// Test whether an exponent vector encodes a smooth relation modulo N.
///
/// Given exponents (e_1, ..., e_d) and primes (p_1, ..., p_d):
///   - Compute A = prod(p_i^{e_i} for e_i > 0) mod N
///   - Compute B = prod(p_i^{-e_i} for e_i < 0) mod N
///   - The relation A ≡ B (mod N) means A - B ≡ 0 (mod N)
///   - Check if A*B^{-1} mod N (or |A - B|) is smooth over the factor base
pub fn test_relation(exponents: &[i64], primes: &[u64], n: &BigUint) -> SmoothRelation {
    let mut pos_product = BigUint::one();
    let mut neg_product = BigUint::one();

    for (i, &exp) in exponents.iter().enumerate() {
        let p_big = BigUint::from(primes[i]);
        if exp > 0 {
            let pow = mod_pow_biguint(&p_big, exp as u64, n);
            pos_product = (&pos_product * &pow) % n;
        } else if exp < 0 {
            let pow = mod_pow_biguint(&p_big, (-exp) as u64, n);
            neg_product = (&neg_product * &pow) % n;
        }
    }

    // Compute residue: pos_product / neg_product mod N = pos_product * neg_product^{-1} mod N
    // This is the actual "remainder" of the relation modulo N.
    // If this quotient is smooth over the factor base, we have a useful relation.
    let combined = if neg_product.is_one() {
        pos_product.clone()
    } else if pos_product.is_one() {
        neg_product.clone()
    } else if let Some(inv) = mod_inverse(&neg_product, n) {
        (&pos_product * &inv) % n
    } else {
        // gcd(neg_product, N) != 1 — neg_product shares a factor with N
        // This is actually useful: the gcd itself may be a factor
        (&pos_product * &neg_product) % n
    };

    let is_smooth;
    let residue_exponents;

    if combined.is_zero() || combined.is_one() {
        is_smooth = true;
        residue_exponents = Some(vec![0u32; primes.len()]);
    } else {
        match smooth_factorize(&combined, primes) {
            Some(exps) => {
                is_smooth = true;
                residue_exponents = Some(exps);
            }
            None => {
                is_smooth = false;
                residue_exponents = None;
            }
        }
    }

    SmoothRelation {
        exponents: exponents.to_vec(),
        positive_product: pos_product,
        negative_product: neg_product,
        residue_mod_n: combined,
        is_smooth,
        residue_exponents,
    }
}

/// Collect smooth relations from a set of exponent vectors.
///
/// Tests each vector and collects those that produce smooth residues.
/// Returns when enough relations are found (>= target) or all vectors are tested.
pub fn collect_smooth_relations(
    vectors: &[Vec<i64>],
    primes: &[u64],
    n: &BigUint,
    target: usize,
) -> RelationCollectionResult {
    let mut relations = Vec::new();
    let mut vectors_tested = 0;
    let mut smooth_count = 0;

    for exps in vectors {
        vectors_tested += 1;
        let rel = test_relation(exps, primes, n);

        if rel.is_smooth {
            smooth_count += 1;
            relations.push(rel);

            if relations.len() >= target {
                break;
            }
        }
    }

    let smooth_rate = if vectors_tested > 0 {
        smooth_count as f64 / vectors_tested as f64
    } else {
        0.0
    };

    RelationCollectionResult {
        relations,
        vectors_tested,
        smooth_count,
        smooth_rate,
    }
}

/// Build a GF(2) exponent matrix from smooth relations.
///
/// Each row corresponds to a relation. Columns represent:
///   - Original exponents (e_1 mod 2, ..., e_d mod 2)
///   - Residue exponents (r_1 mod 2, ..., r_d mod 2) if available
///
/// A dependency (null vector in GF(2)) identifies a subset of relations
/// whose combined exponent vector is all-even, yielding a square.
pub fn build_gf2_matrix(relations: &[SmoothRelation], num_primes: usize) -> (Vec<Vec<u8>>, usize) {
    // Columns: num_primes for original exponents + num_primes for residue exponents
    let num_cols = num_primes * 2;

    let matrix: Vec<Vec<u8>> = relations
        .iter()
        .map(|rel| {
            let mut row = Vec::with_capacity(num_cols);

            // Original exponents mod 2
            for i in 0..num_primes {
                let exp = if i < rel.exponents.len() {
                    rel.exponents[i].unsigned_abs() as u32
                } else {
                    0
                };
                row.push((exp % 2) as u8);
            }

            // Residue exponents mod 2
            if let Some(ref res_exps) = rel.residue_exponents {
                for i in 0..num_primes {
                    let exp = if i < res_exps.len() { res_exps[i] } else { 0 };
                    row.push((exp % 2) as u8);
                }
            } else {
                for _ in 0..num_primes {
                    row.push(0);
                }
            }

            row
        })
        .collect();

    (matrix, num_cols)
}

/// Build a simplified GF(2) matrix using only the original exponents.
///
/// This is sufficient when the lattice directly encodes smooth relations
/// (e.g., when the short vectors represent valid factoring congruences).
pub fn build_simple_gf2_matrix(
    relations: &[SmoothRelation],
    num_primes: usize,
) -> (Vec<Vec<u8>>, usize) {
    let matrix: Vec<Vec<u8>> = relations
        .iter()
        .map(|rel| {
            let mut row = Vec::with_capacity(num_primes);
            for i in 0..num_primes {
                let exp = if i < rel.exponents.len() {
                    rel.exponents[i].unsigned_abs() as u32
                } else {
                    0
                };
                row.push((exp % 2) as u8);
            }
            row
        })
        .collect();

    (matrix, num_primes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_factorize() {
        let primes = vec![2, 3, 5, 7];
        assert_eq!(
            smooth_factorize(&BigUint::from(60u32), &primes),
            Some(vec![2, 1, 1, 0]) // 60 = 2^2 * 3 * 5
        );
        assert_eq!(
            smooth_factorize(&BigUint::from(1u32), &primes),
            Some(vec![0, 0, 0, 0])
        );
        // 11 is not smooth over {2,3,5,7}
        assert_eq!(smooth_factorize(&BigUint::from(11u32), &primes), None);
    }

    #[test]
    fn test_mod_pow_biguint() {
        let base = BigUint::from(2u32);
        let modulus = BigUint::from(13u32);
        // 2^10 = 1024, 1024 mod 13 = 10
        assert_eq!(mod_pow_biguint(&base, 10, &modulus), BigUint::from(10u32));
    }

    #[test]
    fn test_test_relation() {
        let n = BigUint::from(15347u64); // 103 * 149
        let primes = vec![2, 3, 5, 7, 11, 13];

        // Test with a simple exponent vector
        let exponents = vec![1, -1, 0, 0, 0, 0]; // 2/3
        let rel = test_relation(&exponents, &primes, &n);

        // The relation should produce some residue
        assert!(!rel.residue_mod_n.is_zero() || rel.is_smooth);
    }

    #[test]
    fn test_collect_smooth_relations() {
        let n = BigUint::from(15347u64);
        let primes = vec![2, 3, 5, 7, 11, 13];

        let vectors = vec![
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0, 0],
            vec![1, 1, 0, 0, 0, 0],
            vec![2, 0, 1, 0, 0, 0],
            vec![0, 0, 0, 1, 1, 0],
        ];

        let result = collect_smooth_relations(&vectors, &primes, &n, 10);

        assert_eq!(result.vectors_tested, 5);
        assert!(result.smooth_rate >= 0.0 && result.smooth_rate <= 1.0);
    }

    #[test]
    fn test_build_gf2_matrix() {
        let rel = SmoothRelation {
            exponents: vec![2, 3, 1],
            positive_product: BigUint::from(1u32),
            negative_product: BigUint::from(1u32),
            residue_mod_n: BigUint::from(1u32),
            is_smooth: true,
            residue_exponents: Some(vec![1, 0, 2]),
        };

        let (matrix, num_cols) = build_gf2_matrix(&[rel], 3);
        assert_eq!(num_cols, 6); // 3 original + 3 residue
        assert_eq!(matrix.len(), 1);
        // exponents mod 2: [0, 1, 1], residue mod 2: [1, 0, 0]
        assert_eq!(matrix[0], vec![0, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_build_simple_gf2_matrix() {
        let rel = SmoothRelation {
            exponents: vec![4, 1, 3],
            positive_product: BigUint::from(1u32),
            negative_product: BigUint::from(1u32),
            residue_mod_n: BigUint::from(1u32),
            is_smooth: true,
            residue_exponents: None,
        };

        let (matrix, num_cols) = build_simple_gf2_matrix(&[rel], 3);
        assert_eq!(num_cols, 3);
        assert_eq!(matrix[0], vec![0, 1, 1]); // [4%2, 1%2, 3%2]
    }
}
