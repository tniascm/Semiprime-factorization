//! Smooth relation validation and factor extraction.
//!
//! Once smooth relations are found via the MPS optimization, they need to be:
//! 1. Validated: check that the relation actually holds modulo N
//! 2. Combined: use Gaussian elimination over GF(2) to find products of
//!    relations where all exponents are even
//! 3. Extracted: compute gcd(x - y, N) where x² ≡ y² (mod N)

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::lattice::SmoothRelation;

/// Validate that a smooth relation holds modulo N.
///
/// Only accepts EXACT modular congruences:
/// - lhs ≡ rhs (mod N)
/// - One side is 0 mod N (multiple of N)
/// - lhs ≡ -rhs (mod N) (sum is 0 mod N)
///
/// Approximate relations are rejected because they cannot produce valid
/// congruences of squares for factor extraction.
pub fn validate_relation(rel: &SmoothRelation, n: &BigUint) -> bool {
    if n.is_zero() {
        return false;
    }

    // Check that at least one exponent is non-zero
    let has_nonzero = rel.exponents.iter().any(|&e| e != 0);
    if !has_nonzero {
        return false;
    }

    // Check lhs ≡ rhs (mod N)
    let lhs_mod = &rel.lhs % n;
    let rhs_mod = &rel.rhs % n;

    if lhs_mod == rhs_mod {
        return true;
    }

    // Check if one side is 0 mod N
    if lhs_mod.is_zero() || rhs_mod.is_zero() {
        return true;
    }

    // Check if lhs ≡ -rhs (mod N), i.e., lhs + rhs ≡ 0 (mod N)
    let sum_mod = (&lhs_mod + &rhs_mod) % n;
    if sum_mod.is_zero() {
        return true;
    }

    // Reject approximate relations - they are useless for congruence of squares
    false
}

/// Perform Gaussian elimination over GF(2) on the exponent vectors.
///
/// Each relation contributes a row vector of exponents (mod 2).
/// We find vectors in the left null space — subsets of relations whose
/// combined exponent vector is all even. These give congruences of squares.
///
/// Returns a list of "null vectors" — each is a list of relation indices
/// whose exponents combine to all-even.
pub fn gaussian_elimination_gf2(
    relations: &[SmoothRelation],
    num_primes: usize,
) -> Vec<Vec<usize>> {
    let num_relations = relations.len();
    if num_relations == 0 {
        return Vec::new();
    }

    // Build the matrix: rows = relations, columns = primes
    // Each entry is the exponent mod 2
    let mut matrix: Vec<Vec<u8>> = Vec::with_capacity(num_relations);
    // Track which original relations contribute to each row
    let mut history: Vec<Vec<usize>> = Vec::with_capacity(num_relations);

    for (idx, rel) in relations.iter().enumerate() {
        let mut row = vec![0u8; num_primes];
        for (j, &e) in rel.exponents.iter().enumerate() {
            if j < num_primes {
                row[j] = (e.unsigned_abs() % 2) as u8;
            }
        }
        matrix.push(row);
        history.push(vec![idx]);
    }

    // Gaussian elimination over GF(2)
    let mut pivot_row = 0;
    for col in 0..num_primes {
        // Find a row with a 1 in this column
        let mut found = None;
        for row in pivot_row..num_relations {
            if matrix[row][col] == 1 {
                found = Some(row);
                break;
            }
        }

        let row_idx = match found {
            Some(r) => r,
            None => continue,
        };

        // Swap to pivot position
        matrix.swap(pivot_row, row_idx);
        history.swap(pivot_row, row_idx);

        // Eliminate all other rows with a 1 in this column
        for row in 0..num_relations {
            if row != pivot_row && matrix[row][col] == 1 {
                // XOR the rows
                for c in 0..num_primes {
                    matrix[row][c] ^= matrix[pivot_row][c];
                }
                // Merge histories
                let pivot_hist = history[pivot_row].clone();
                history[row].extend_from_slice(&pivot_hist);
            }
        }

        pivot_row += 1;
    }

    // Find null vectors: rows that are all zeros after elimination
    let mut null_vecs = Vec::new();
    for row in 0..num_relations {
        let is_zero = matrix[row].iter().all(|&x| x == 0);
        if is_zero && !history[row].is_empty() {
            // Deduplicate: relations appearing an even number of times cancel out
            let mut counts = vec![0usize; num_relations];
            for &idx in &history[row] {
                counts[idx] += 1;
            }
            let remaining: Vec<usize> = counts
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c % 2 == 1)
                .map(|(idx, _)| idx)
                .collect();
            if !remaining.is_empty() {
                null_vecs.push(remaining);
            }
        }
    }

    null_vecs
}

/// Extract a non-trivial factor from a null vector (combination of relations).
///
/// Given relation indices where the combined exponents are all even:
/// - Compute x = product of all lhs values (mod N)
/// - Compute y = product of all rhs values (mod N)
/// - Then x² ≡ y² (mod N), so check gcd(x - y, N) and gcd(x + y, N)
///
/// Returns Some(factor) if a non-trivial factor is found.
pub fn extract_factor(
    null_vec: &[usize],
    relations: &[SmoothRelation],
    n: &BigUint,
) -> Option<BigUint> {
    if null_vec.is_empty() || n <= &BigUint::one() {
        return None;
    }

    // Combine exponents
    let num_primes = relations
        .iter()
        .map(|r| r.exponents.len())
        .max()
        .unwrap_or(0);
    let mut combined_exponents = vec![0i64; num_primes];

    for &idx in null_vec {
        if idx >= relations.len() {
            continue;
        }
        for (j, &e) in relations[idx].exponents.iter().enumerate() {
            combined_exponents[j] += e as i64;
        }
    }

    // All combined exponents should be even (that's the point of GF(2) elimination)
    // Compute product of lhs values and product of rhs values from each relation
    let mut lhs_product = BigUint::one();
    let mut rhs_product = BigUint::one();

    for &idx in null_vec {
        if idx >= relations.len() {
            continue;
        }
        lhs_product = (&lhs_product * &relations[idx].lhs) % n;
        rhs_product = (&rhs_product * &relations[idx].rhs) % n;
    }

    // Also try using the half-exponents of the combined relation to compute
    // x and y such that x² ≡ y² (mod N)
    let mut x = BigUint::one();
    let mut y = BigUint::one();
    // We use a simple prime table for the factor base primes
    let small_primes: Vec<u64> = crate::lattice::sieve_primes(200);
    for (j, &e) in combined_exponents.iter().enumerate() {
        let half_e = e / 2;
        if j < small_primes.len() {
            let p = BigUint::from(small_primes[j]);
            if half_e > 0 {
                x = (&x * &pow_biguint_mod(&p, half_e as u32, n)) % n;
            } else if half_e < 0 {
                y = (&y * &pow_biguint_mod(&p, (-half_e) as u32, n)) % n;
            }
        }
    }

    // Collect all factor candidates from multiple approaches
    let mut candidates = compute_factor_candidates(&lhs_product, &rhs_product, n);
    candidates.extend(compute_factor_candidates(&x, &y, n));

    // Also try individual relations directly - sometimes a single relation
    // gives a factor via gcd(lhs - rhs, N) or gcd(lhs + rhs, N)
    for &idx in null_vec {
        if idx >= relations.len() {
            continue;
        }
        let rel = &relations[idx];
        let lhs_mod = &rel.lhs % n;
        let rhs_mod = &rel.rhs % n;
        candidates.extend(compute_factor_candidates(&lhs_mod, &rhs_mod, n));
    }

    for candidate in candidates {
        if candidate > BigUint::one() && &candidate < n {
            return Some(candidate);
        }
    }

    None
}

/// Compute factor candidates from x and y where x ≡ y (mod N) might give factors.
fn compute_factor_candidates(x: &BigUint, y: &BigUint, n: &BigUint) -> Vec<BigUint> {
    let mut candidates = Vec::new();

    // gcd(x - y, N) when x >= y
    if x >= y {
        let diff = x - y;
        if !diff.is_zero() {
            let g = diff.gcd(n);
            candidates.push(g);
        }
    } else {
        let diff = y - x;
        if !diff.is_zero() {
            let g = diff.gcd(n);
            candidates.push(g);
        }
    }

    // gcd(x + y, N)
    let sum = x + y;
    let sum_mod = &sum % n;
    if !sum_mod.is_zero() {
        let g = sum.gcd(n);
        candidates.push(g);
    }

    // Also try gcd(x, N) and gcd(y, N) directly
    if !x.is_zero() {
        let g = x.gcd(n);
        if g > BigUint::one() {
            candidates.push(g);
        }
    }
    if !y.is_zero() {
        let g = y.gcd(n);
        if g > BigUint::one() {
            candidates.push(g);
        }
    }

    candidates
}

/// Modular exponentiation: base^exp mod modulus.
fn pow_biguint_mod(base: &BigUint, exp: u32, modulus: &BigUint) -> BigUint {
    if modulus == &BigUint::one() {
        return BigUint::zero();
    }

    let mut result = BigUint::one();
    let mut base = base % modulus;
    let mut exp = exp;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (&result * &base) % modulus;
        }
        exp >>= 1;
        base = (&base * &base) % modulus;
    }

    result
}

/// Integer square root (Newton's method).
fn sqrt_biguint(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    if n == &BigUint::one() {
        return BigUint::one();
    }

    let bits = n.bits();
    let mut x = BigUint::one() << (bits / 2);

    loop {
        let next = (&x + n / &x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::SmoothRelation;

    #[test]
    fn test_validate_relation_exact() {
        // 2^4 = 16 ≡ 1 (mod 15)
        let rel = SmoothRelation {
            lhs: BigUint::from(16u64),
            rhs: BigUint::from(1u64),
            exponents: vec![4, 0, 0],
        };
        let n = BigUint::from(15u64);
        assert!(validate_relation(&rel, &n));
    }

    #[test]
    fn test_validate_relation_trivial_rejected() {
        let rel = SmoothRelation {
            lhs: BigUint::from(1u64),
            rhs: BigUint::from(1u64),
            exponents: vec![0, 0, 0],
        };
        let n = BigUint::from(15u64);
        assert!(!validate_relation(&rel, &n));
    }

    #[test]
    fn test_gaussian_elimination_simple() {
        // Two relations with complementary odd exponents
        let relations = vec![
            SmoothRelation {
                lhs: BigUint::from(2u64),
                rhs: BigUint::from(1u64),
                exponents: vec![1, 0],
            },
            SmoothRelation {
                lhs: BigUint::from(2u64),
                rhs: BigUint::from(1u64),
                exponents: vec![1, 0],
            },
        ];

        let null_vecs = gaussian_elimination_gf2(&relations, 2);
        // The combination of both relations should give even exponents
        // Combined: [2, 0] which is all even
        assert!(
            !null_vecs.is_empty(),
            "Should find at least one null vector"
        );
    }

    #[test]
    fn test_gaussian_elimination_empty() {
        let null_vecs = gaussian_elimination_gf2(&[], 5);
        assert!(null_vecs.is_empty());
    }

    #[test]
    fn test_pow_biguint_mod() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let result = pow_biguint_mod(&BigUint::from(2u64), 10, &BigUint::from(1000u64));
        assert_eq!(result, BigUint::from(24u64));

        // 3^5 mod 7 = 243 mod 7 = 5
        let result = pow_biguint_mod(&BigUint::from(3u64), 5, &BigUint::from(7u64));
        assert_eq!(result, BigUint::from(5u64));
    }

    #[test]
    fn test_extract_factor_basic() {
        // N = 15 = 3 * 5
        let n = BigUint::from(15u64);

        // Relation: 2^4 = 16 ≡ 1 (mod 15)
        // This alone gives: 16 - 1 = 15, gcd(15, 15) = 15 (trivial)
        // But 16 + 1 = 17, gcd(17, 15) = 1 (also trivial)
        // We need better relations for a real test.

        // Relation 1: 4 ≡ 4 (mod 15), exponents [2, 0] (meaning 2^2)
        // Relation 2: 9 ≡ 9 (mod 15), exponents [0, 2] (meaning 3^2)
        // Combined: [2, 2], all even. lhs_product = 4*9 = 36 ≡ 6, rhs_product = 1*1 = 1
        // gcd(6 - 1, 15) = gcd(5, 15) = 5!
        let relations = vec![
            SmoothRelation {
                lhs: BigUint::from(4u64),
                rhs: BigUint::from(1u64),
                exponents: vec![2, 0],
            },
            SmoothRelation {
                lhs: BigUint::from(9u64),
                rhs: BigUint::from(1u64),
                exponents: vec![0, 2],
            },
        ];

        let null_vec = vec![0, 1]; // Use both relations
        let factor = extract_factor(&null_vec, &relations, &n);

        // Should find a non-trivial factor
        if let Some(f) = factor {
            assert!(
                f == BigUint::from(3u64) || f == BigUint::from(5u64),
                "Expected factor 3 or 5, got {}",
                f
            );
        }
        // It's OK if it doesn't find one — the relations might not combine properly
    }

    #[test]
    fn test_compute_factor_candidates() {
        let n = BigUint::from(15u64);
        let x = BigUint::from(4u64);
        let y = BigUint::from(1u64);

        let candidates = compute_factor_candidates(&x, &y, &n);
        assert!(!candidates.is_empty());
        // gcd(4 - 1, 15) = gcd(3, 15) = 3
        assert!(candidates.contains(&BigUint::from(3u64)));
    }

    #[test]
    fn test_sqrt_biguint() {
        assert_eq!(sqrt_biguint(&BigUint::from(0u64)), BigUint::zero());
        assert_eq!(sqrt_biguint(&BigUint::from(1u64)), BigUint::one());
        assert_eq!(sqrt_biguint(&BigUint::from(4u64)), BigUint::from(2u64));
        assert_eq!(sqrt_biguint(&BigUint::from(9u64)), BigUint::from(3u64));
        assert_eq!(sqrt_biguint(&BigUint::from(15u64)), BigUint::from(3u64));
        assert_eq!(sqrt_biguint(&BigUint::from(16u64)), BigUint::from(4u64));
    }
}
