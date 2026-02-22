//! Relation collection for the Number Field Sieve.
//!
//! Stores smooth relations found during sieving, handles large prime variation,
//! and builds the relation matrix for linear algebra.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::collections::HashMap;

use crate::polynomial::NfsPolynomial;
use crate::sieve::{eval_homogeneous_abs, SieveHit};

/// A smooth relation: (a, b) pair with factorizations on both sides.
#[derive(Debug, Clone)]
pub struct SmoothRelation {
    /// The a value of the coprime pair.
    pub a: i64,
    /// The b value, always > 0.
    pub b: i64,
    /// Whether a + b*m is negative.
    pub sign_negative: bool,
    /// Exponents of primes on the rational side (indexed by rational factor base).
    pub rational_exponents: Vec<u32>,
    /// Exponents of (prime, root) pairs on the algebraic side.
    pub algebraic_exponents: Vec<u32>,
    /// Large prime on rational side (0 if fully smooth).
    pub rational_large_prime: u64,
    /// Large prime on algebraic side (0 if fully smooth).
    pub algebraic_large_prime: u64,
}

impl SmoothRelation {
    /// Returns true if this relation is fully smooth (no large primes).
    pub fn is_full(&self) -> bool {
        self.rational_large_prime == 0 && self.algebraic_large_prime == 0
    }

    /// Returns true if this is a partial relation (has exactly one large prime on one side).
    pub fn is_partial(&self) -> bool {
        !self.is_full()
    }
}

/// Result of collecting relations from sieve hits.
#[derive(Debug)]
pub struct RelationCollectionResult {
    /// Full smooth relations (both sides factor completely over factor bases).
    pub full_relations: Vec<SmoothRelation>,
    /// Relations formed by combining partials sharing large primes.
    pub cycle_relations: Vec<SmoothRelation>,
    /// Total number of partial relations found.
    pub partials_found: usize,
    /// Number of cycles (combined partials) found.
    pub cycles_found: usize,
}

/// Try to factor value over the given factor base.
/// Returns (exponents, cofactor) where cofactor is the remaining unfactored part.
fn trial_divide(mut val: u64, factor_base: &[u64]) -> (Vec<u32>, u64) {
    let mut exponents = vec![0u32; factor_base.len()];

    for (i, &p) in factor_base.iter().enumerate() {
        if p == 0 {
            continue;
        }
        while val % p == 0 {
            val /= p;
            exponents[i] += 1;
        }
    }

    (exponents, val)
}

/// Try to factor a BigUint value over the factor base.
/// Returns (exponents, cofactor_u64) if cofactor fits in u64, None otherwise.
fn trial_divide_big(val: &BigUint, factor_base: &[u64]) -> Option<(Vec<u32>, u64)> {
    let mut remaining = val.clone();
    let mut exponents = vec![0u32; factor_base.len()];

    for (i, &p) in factor_base.iter().enumerate() {
        if p == 0 {
            continue;
        }
        let p_big = BigUint::from(p);
        while (&remaining % &p_big).is_zero() {
            remaining /= &p_big;
            exponents[i] += 1;
        }
    }

    // Try to convert cofactor to u64
    if let Some(cofactor) = remaining.to_u64() {
        Some((exponents, cofactor))
    } else {
        // Cofactor is too large
        None
    }
}

/// Collect smooth relations from sieve hits by trial dividing both sides.
///
/// For each sieve hit (a, b):
///   - Rational side: factor |a + b*m| over rational_fb
///   - Algebraic side: factor |F(a, b)| over algebraic_fb
///   - If both sides are smooth (or have a single large prime within bounds),
///     record the relation.
pub fn collect_relations(
    hits: &[SieveHit],
    poly: &NfsPolynomial,
    rational_fb: &[u64],
    algebraic_fb: &[u64],
    large_prime_bound: u64,
) -> RelationCollectionResult {
    let m_u64: u64 = poly.m.to_u64().expect("m must fit in u64");
    let m_i128 = m_u64 as i128;

    let mut full_relations = Vec::new();
    let mut partial_relations: Vec<SmoothRelation> = Vec::new();

    for hit in hits {
        let a = hit.a;
        let b = hit.b;

        // Compute rational norm: a + b*m
        let rational_val_signed: i128 = (a as i128) + (b as i128) * m_i128;
        if rational_val_signed == 0 {
            continue;
        }
        let sign_negative = rational_val_signed < 0;
        let rational_abs = rational_val_signed.unsigned_abs();

        // Rational side: try to factor over rational factor base
        let (rational_exp, rational_cofactor) = if rational_abs <= u64::MAX as u128 {
            trial_divide(rational_abs as u64, rational_fb)
        } else {
            let big_val = BigUint::from(rational_abs);
            match trial_divide_big(&big_val, rational_fb) {
                Some((exp, cof)) => (exp, cof),
                None => continue,
            }
        };

        // Algebraic side: compute |F(a, b)| and factor over algebraic factor base
        let alg_norm = eval_homogeneous_abs(&poly.coefficients, a, b);
        if alg_norm.is_zero() {
            continue;
        }

        let (algebraic_exp, algebraic_cofactor) = match trial_divide_big(&alg_norm, algebraic_fb) {
            Some((exp, cof)) => (exp, cof),
            None => continue,
        };

        // Check smoothness: both cofactors must be 1 (full) or a single prime <= large_prime_bound (partial)
        let rat_ok = rational_cofactor == 1 || rational_cofactor <= large_prime_bound;
        let alg_ok = algebraic_cofactor == 1 || algebraic_cofactor <= large_prime_bound;

        if !rat_ok || !alg_ok {
            continue;
        }

        // For simplicity in the large-prime approach, only allow one large prime total
        // (either on rational or algebraic side, not both)
        let rational_lp = if rational_cofactor > 1 {
            rational_cofactor
        } else {
            0
        };
        let algebraic_lp = if algebraic_cofactor > 1 {
            algebraic_cofactor
        } else {
            0
        };

        // Allow at most one large prime across both sides for partial relations
        if rational_lp > 0 && algebraic_lp > 0 {
            // Double-large-prime: skip for now (too complex for baseline)
            continue;
        }

        let rel = SmoothRelation {
            a,
            b,
            sign_negative,
            rational_exponents: rational_exp,
            algebraic_exponents: algebraic_exp,
            rational_large_prime: rational_lp,
            algebraic_large_prime: algebraic_lp,
        };

        if rel.is_full() {
            full_relations.push(rel);
        } else {
            partial_relations.push(rel);
        }
    }

    // Combine partial relations sharing the same large prime
    let (cycle_rels, cycles_found) = combine_partials(&partial_relations, rational_fb, algebraic_fb);

    RelationCollectionResult {
        full_relations,
        cycle_relations: cycle_rels,
        partials_found: partial_relations.len(),
        cycles_found,
    }
}

/// Combine partial relations that share the same large prime into full relations.
///
/// If two relations share the same large prime L, their product eliminates L
/// (since L appears to an even power). We combine their exponent vectors.
fn combine_partials(
    partials: &[SmoothRelation],
    rational_fb: &[u64],
    algebraic_fb: &[u64],
) -> (Vec<SmoothRelation>, usize) {
    // Group partials by their large prime
    let mut by_large_prime: HashMap<u64, Vec<usize>> = HashMap::new();

    for (idx, rel) in partials.iter().enumerate() {
        let lp = if rel.rational_large_prime > 0 {
            rel.rational_large_prime
        } else {
            rel.algebraic_large_prime
        };
        if lp > 1 {
            by_large_prime.entry(lp).or_default().push(idx);
        }
    }

    let mut combined = Vec::new();
    let mut cycles = 0;

    for (_lp, indices) in &by_large_prime {
        if indices.len() < 2 {
            continue;
        }

        // Combine pairs
        for pair in indices.windows(2) {
            let r1 = &partials[pair[0]];
            let r2 = &partials[pair[1]];

            // Combined relation: add exponents
            let mut combined_rat_exp = vec![0u32; rational_fb.len()];
            let mut combined_alg_exp = vec![0u32; algebraic_fb.len()];

            for i in 0..rational_fb.len().min(r1.rational_exponents.len()) {
                combined_rat_exp[i] = r1.rational_exponents[i] + r2.rational_exponents[i];
            }
            for i in 0..algebraic_fb.len().min(r1.algebraic_exponents.len()) {
                combined_alg_exp[i] = r1.algebraic_exponents[i] + r2.algebraic_exponents[i];
            }

            // The large prime appears twice (once in each partial), so it cancels in mod 2
            combined.push(SmoothRelation {
                a: r1.a,
                b: r1.b,
                sign_negative: r1.sign_negative ^ r2.sign_negative,
                rational_exponents: combined_rat_exp,
                algebraic_exponents: combined_alg_exp,
                rational_large_prime: 0,
                algebraic_large_prime: 0,
            });
            cycles += 1;
        }
    }

    (combined, cycles)
}

/// Build the GF(2) exponent matrix from relations.
///
/// Each row represents a relation. The columns are:
///   [sign_bit, rational_exp_0 mod 2, ..., rational_exp_k mod 2,
///    algebraic_exp_0 mod 2, ..., algebraic_exp_j mod 2]
///
/// Returns (matrix, num_cols).
pub fn build_matrix(
    relations: &[SmoothRelation],
    rational_fb_size: usize,
    algebraic_fb_size: usize,
) -> (Vec<Vec<u8>>, usize) {
    let num_cols = 1 + rational_fb_size + algebraic_fb_size;

    let matrix: Vec<Vec<u8>> = relations
        .iter()
        .map(|rel| {
            let mut row = Vec::with_capacity(num_cols);

            // Sign bit
            row.push(if rel.sign_negative { 1 } else { 0 });

            // Rational exponents mod 2
            for i in 0..rational_fb_size {
                let exp = if i < rel.rational_exponents.len() {
                    rel.rational_exponents[i]
                } else {
                    0
                };
                row.push((exp % 2) as u8);
            }

            // Algebraic exponents mod 2
            for i in 0..algebraic_fb_size {
                let exp = if i < rel.algebraic_exponents.len() {
                    rel.algebraic_exponents[i]
                } else {
                    0
                };
                row.push((exp % 2) as u8);
            }

            row
        })
        .collect();

    (matrix, num_cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::select_polynomial;
    use crate::sieve::{
        compute_polynomial_roots, line_sieve, sieve_primes, SieveConfig,
    };

    #[test]
    fn test_trial_divide() {
        let (exp, cofactor) = trial_divide(60, &[2, 3, 5]);
        assert_eq!(exp, vec![2, 1, 1]); // 60 = 2^2 * 3 * 5
        assert_eq!(cofactor, 1);
    }

    #[test]
    fn test_trial_divide_with_cofactor() {
        let (exp, cofactor) = trial_divide(70, &[2, 3, 5]);
        assert_eq!(exp, vec![1, 0, 1]); // 70 = 2 * 5 * 7
        assert_eq!(cofactor, 7);
    }

    #[test]
    fn test_collect_relations() {
        let n = BigUint::from(15347u64);
        let poly = select_polynomial(&n, 3);
        let rational_fb = sieve_primes(100);
        let algebraic_fb = sieve_primes(100);
        let alg_roots = compute_polynomial_roots(&poly, &algebraic_fb);

        let config = SieveConfig {
            rational_bound: 100,
            algebraic_bound: 100,
            sieve_area: 500,
            max_b: 50,
            large_prime_multiplier: 100,
            sieve_threshold: 0.0,
        };

        let hits = line_sieve(&poly, &rational_fb, &alg_roots, &config);
        let result = collect_relations(&hits, &poly, &rational_fb, &algebraic_fb, 10000);

        // We should get at least some relations for this small number
        assert!(
            result.full_relations.len() + result.cycle_relations.len() > 0,
            "Should find some relations for n=15347, full={}, cycles={}",
            result.full_relations.len(),
            result.cycles_found
        );
    }

    #[test]
    fn test_build_matrix() {
        let rel = SmoothRelation {
            a: 1,
            b: 1,
            sign_negative: false,
            rational_exponents: vec![2, 1, 0],
            algebraic_exponents: vec![1, 3],
            rational_large_prime: 0,
            algebraic_large_prime: 0,
        };

        let (matrix, num_cols) = build_matrix(&[rel], 3, 2);
        assert_eq!(num_cols, 6); // 1 sign + 3 rational + 2 algebraic
        assert_eq!(matrix.len(), 1);
        // sign=0, rat=[0,1,0], alg=[1,1]
        assert_eq!(matrix[0], vec![0, 0, 1, 0, 1, 1]);
    }
}
