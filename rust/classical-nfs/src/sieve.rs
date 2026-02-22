//! Lattice sieve for the Number Field Sieve.
//!
//! Implements line sieving on both the rational and algebraic sides.
//! For each prime q in the factor base, we sieve over (a, b) pairs to find
//! smooth values of the rational norm a + b*m and algebraic norm F(a, b).

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

use crate::polynomial::NfsPolynomial;

/// Configuration for the sieve phase.
#[derive(Debug, Clone)]
pub struct SieveConfig {
    /// Factor base bound for the rational side.
    pub rational_bound: u64,
    /// Factor base bound for the algebraic side.
    pub algebraic_bound: u64,
    /// Sieve array half-width: we sieve a in [-sieve_area, sieve_area] for each b.
    pub sieve_area: i64,
    /// Maximum value of b to sieve.
    pub max_b: i64,
    /// Large prime bound (as a multiplier of the factor base bound).
    pub large_prime_multiplier: u64,
    /// Sieve threshold log2 -- positions with sieve sum >= threshold are smoothness candidates.
    pub sieve_threshold: f64,
}

impl Default for SieveConfig {
    fn default() -> Self {
        Self {
            rational_bound: 1 << 12,
            algebraic_bound: 1 << 12,
            sieve_area: 1 << 14,
            max_b: 1 << 10,
            large_prime_multiplier: 100,
            sieve_threshold: 0.0, // computed dynamically
        }
    }
}

/// A raw sieve hit: an (a, b) pair that passed the sieve threshold.
#[derive(Debug, Clone)]
pub struct SieveHit {
    pub a: i64,
    pub b: i64,
    /// Approximate log2 of the rational norm accumulated during sieving.
    pub rational_log_sum: f64,
    /// Approximate log2 of the algebraic norm accumulated during sieving.
    pub algebraic_log_sum: f64,
}

/// A root of the polynomial f(x) modulo p: the value r such that f(r) = 0 (mod p).
#[derive(Debug, Clone)]
pub struct PolynomialRoot {
    pub prime: u64,
    pub root: u64,
    pub log_p: f64,
}

/// Compute the roots of the polynomial f(x) modulo each prime p in the algebraic factor base.
///
/// For each prime p, find all r in [0, p) such that f(r) = 0 (mod p).
/// These roots are used for algebraic-side sieving: the algebraic norm
/// F(a, b) = b^d * f(a/b) is divisible by p when a = -r*b (mod p).
pub fn compute_polynomial_roots(poly: &NfsPolynomial, primes: &[u64]) -> Vec<PolynomialRoot> {
    let mut roots = Vec::new();

    for &p in primes {
        if p == 0 {
            continue;
        }
        let log_p = (p as f64).ln() / (2.0_f64).ln();

        // Evaluate f(x) mod p for x in [0, p)
        for x in 0..p {
            let val = eval_poly_mod(&poly.coefficients, x, p);
            if val == 0 {
                roots.push(PolynomialRoot {
                    prime: p,
                    root: x,
                    log_p,
                });
            }
        }
    }

    roots
}

/// Evaluate polynomial with BigUint coefficients at x modulo p.
/// coefficients are [c_0, c_1, ..., c_d] (low-degree first).
fn eval_poly_mod(coeffs: &[BigUint], x: u64, p: u64) -> u64 {
    if coeffs.is_empty() || p == 0 {
        return 0;
    }
    let p128 = p as u128;
    let x128 = x as u128;

    // Horner's method mod p, from highest degree down
    let d = coeffs.len() - 1;
    let mut result: u128 = coeff_mod(&coeffs[d], p);

    for i in (0..d).rev() {
        result = (result * x128 + coeff_mod(&coeffs[i], p)) % p128;
    }

    result as u64
}

/// Reduce a BigUint coefficient modulo p.
fn coeff_mod(c: &BigUint, p: u64) -> u128 {
    let p_big = BigUint::from(p);
    let r = c % &p_big;
    r.to_u64().unwrap_or(0) as u128
}

/// Perform line sieving over (a, b) pairs.
///
/// For each b in [1, max_b]:
///   - Initialize a sieve array of size 2*sieve_area + 1 (for a in [-sieve_area, sieve_area])
///   - For each rational prime q: sieve positions where a + b*m = 0 (mod q)
///   - For each algebraic root (p, r): sieve positions where a + r*b = 0 (mod p)
///   - Collect positions where both rational and algebraic sums exceed their thresholds
///
/// Returns a vector of SieveHit candidates for trial division.
pub fn line_sieve(
    poly: &NfsPolynomial,
    rational_fb: &[u64],
    algebraic_roots: &[PolynomialRoot],
    config: &SieveConfig,
) -> Vec<SieveHit> {
    let m_u64: u64 = poly.m.to_u64().expect("m must fit in u64 for sieving");
    let area = config.sieve_area;
    let array_size = (2 * area + 1) as usize;

    // Compute expected norms to set thresholds
    let rational_threshold = compute_rational_threshold(m_u64, area, config.rational_bound);
    let algebraic_threshold = compute_algebraic_threshold(poly, area, config.algebraic_bound);

    let mut hits = Vec::new();

    for b in 1..=config.max_b {
        let b_u64 = b as u64;

        // Initialize sieve arrays
        let mut rational_sieve_arr = vec![0.0_f64; array_size];
        let mut algebraic_sieve_arr = vec![0.0_f64; array_size];

        // Rational-side sieve: for each prime q, find a such that a + b*m = 0 (mod q)
        // => a = -b*m (mod q)
        for &q in rational_fb {
            if q == 0 {
                continue;
            }
            let log_q = (q as f64).ln() / (2.0_f64).ln();
            let bm_mod_q = ((b_u64 as u128 * m_u64 as u128) % q as u128) as u64;
            let start_a_mod_q = if bm_mod_q == 0 { 0 } else { q - bm_mod_q };

            // The first a >= -area with a = start_a_mod_q (mod q)
            let a_min = -area;
            let a_min_mod_q = ((a_min % q as i64) + q as i64) as u64 % q;
            let offset = if start_a_mod_q >= a_min_mod_q {
                start_a_mod_q - a_min_mod_q
            } else {
                q - a_min_mod_q + start_a_mod_q
            };

            let mut idx = offset as usize;
            while idx < array_size {
                rational_sieve_arr[idx] += log_q;
                idx += q as usize;
            }
        }

        // Algebraic-side sieve: for each root (p, r), find a such that a + r*b = 0 (mod p)
        // => a = -r*b (mod p)
        for root in algebraic_roots {
            let p = root.prime;
            let r = root.root;
            let rb_mod_p = ((r as u128 * b_u64 as u128) % p as u128) as u64;
            let start_a_mod_p = if rb_mod_p == 0 { 0 } else { p - rb_mod_p };

            let a_min = -area;
            let a_min_mod_p = ((a_min % p as i64) + p as i64) as u64 % p;
            let offset = if start_a_mod_p >= a_min_mod_p {
                start_a_mod_p - a_min_mod_p
            } else {
                p - a_min_mod_p + start_a_mod_p
            };

            let mut idx = offset as usize;
            while idx < array_size {
                algebraic_sieve_arr[idx] += root.log_p;
                idx += p as usize;
            }
        }

        // Collect hits where both sides exceed threshold
        for i in 0..array_size {
            let a = (i as i64) + (-area);
            if a == 0 {
                continue;
            }
            // gcd(|a|, b) must be 1
            if gcd_u64(a.unsigned_abs(), b_u64) != 1 {
                continue;
            }

            if rational_sieve_arr[i] >= rational_threshold
                && algebraic_sieve_arr[i] >= algebraic_threshold
            {
                hits.push(SieveHit {
                    a,
                    b,
                    rational_log_sum: rational_sieve_arr[i],
                    algebraic_log_sum: algebraic_sieve_arr[i],
                });
            }
        }
    }

    hits
}

/// Compute the sieve threshold for the rational side.
///
/// The rational norm is |a + b*m|. For typical (a, b) in the sieve range,
/// log2|a + b*m| ~ log2(max_b * m). We want the sieve sum to account for
/// most of this, leaving a small cofactor. Threshold = norm_log - slack.
fn compute_rational_threshold(m: u64, sieve_area: i64, fb_bound: u64) -> f64 {
    let expected_norm = ((sieve_area as f64).abs() * (m as f64)).max(1.0);
    let norm_log = expected_norm.log2();
    let slack = (fb_bound as f64).log2() * 2.0; // allow cofactor up to fb_bound^2
    (norm_log - slack).max(0.0)
}

/// Compute the sieve threshold for the algebraic side.
fn compute_algebraic_threshold(poly: &NfsPolynomial, sieve_area: i64, fb_bound: u64) -> f64 {
    // Rough estimate: algebraic norm ~ max(|c_i|) * area^d
    let d = poly.degree;
    let max_coeff = poly
        .coefficients
        .iter()
        .filter_map(|c| c.to_f64())
        .fold(1.0_f64, |a, b| a.max(b));
    let expected_norm = max_coeff * (sieve_area as f64).powi(d as i32);
    let norm_log = expected_norm.log2();
    let slack = (fb_bound as f64).log2() * 2.5;
    (norm_log - slack).max(0.0)
}

/// GCD for u64 values.
pub fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Generate primes up to `bound` using the Sieve of Eratosthenes.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return vec![];
    }
    let limit = bound as usize;
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit >= 1 {
        is_prime[1] = false;
    }
    let mut p = 2;
    while p * p <= limit {
        if is_prime[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                is_prime[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=limit)
        .filter(|&i| is_prime[i])
        .map(|i| i as u64)
        .collect()
}

/// Evaluate the homogeneous polynomial f_hom(a, b) = sum_i c_i * a^i * b^(d-i).
/// Returns the absolute value |f_hom(a, b)| as a BigUint.
pub fn eval_homogeneous_abs(coeffs: &[BigUint], a: i64, b: i64) -> BigUint {
    if coeffs.is_empty() {
        return BigUint::zero();
    }

    let d = coeffs.len() - 1;
    let a_abs = BigUint::from(a.unsigned_abs());
    let b_abs = BigUint::from(b.unsigned_abs());
    let a_neg = a < 0;
    let b_neg = b < 0;

    let mut positive_sum = BigUint::zero();
    let mut negative_sum = BigUint::zero();

    for i in 0..=d {
        if coeffs[i].is_zero() {
            continue;
        }
        let a_pow = pow_biguint(&a_abs, i as u32);
        let b_pow = pow_biguint(&b_abs, (d - i) as u32);
        let term_mag = &coeffs[i] * &a_pow * &b_pow;

        let term_neg = (a_neg && (i % 2 == 1)) ^ (b_neg && ((d - i) % 2 == 1));

        if term_neg {
            negative_sum += term_mag;
        } else {
            positive_sum += term_mag;
        }
    }

    if positive_sum >= negative_sum {
        positive_sum - negative_sum
    } else {
        negative_sum - positive_sum
    }
}

/// Compute base^exp for BigUint.
fn pow_biguint(base: &BigUint, exp: u32) -> BigUint {
    if exp == 0 {
        return BigUint::one();
    }
    let mut result = BigUint::one();
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= &b;
        }
        b = &b * &b;
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::select_polynomial;

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_eval_poly_mod() {
        // f(x) = x^2 + 1, coefficients = [1, 0, 1]
        let coeffs = vec![
            BigUint::from(1u64),
            BigUint::zero(),
            BigUint::from(1u64),
        ];
        // f(2) mod 5 = (4 + 1) mod 5 = 0
        assert_eq!(eval_poly_mod(&coeffs, 2, 5), 0);
        // f(3) mod 5 = (9 + 1) mod 5 = 0
        assert_eq!(eval_poly_mod(&coeffs, 3, 5), 0);
        // f(1) mod 5 = 2
        assert_eq!(eval_poly_mod(&coeffs, 1, 5), 2);
    }

    #[test]
    fn test_compute_polynomial_roots() {
        // f(x) = x^2 + 1: roots mod 5 are 2, 3 since 2^2+1=5 and 3^2+1=10
        let coeffs = vec![
            BigUint::from(1u64),
            BigUint::zero(),
            BigUint::from(1u64),
        ];
        let poly = NfsPolynomial {
            coefficients: coeffs,
            m: BigUint::from(10u64),
            degree: 2,
        };
        let roots = compute_polynomial_roots(&poly, &[5]);
        let root_vals: Vec<u64> = roots.iter().map(|r| r.root).collect();
        assert!(root_vals.contains(&2));
        assert!(root_vals.contains(&3));
    }

    #[test]
    fn test_line_sieve_produces_hits() {
        let n = BigUint::from(15347u64);
        let poly = select_polynomial(&n, 3);
        let rational_fb = sieve_primes(50);
        let algebraic_fb = sieve_primes(50);
        let alg_roots = compute_polynomial_roots(&poly, &algebraic_fb);

        let config = SieveConfig {
            rational_bound: 50,
            algebraic_bound: 50,
            sieve_area: 200,
            max_b: 20,
            large_prime_multiplier: 10,
            sieve_threshold: 0.0,
        };

        let hits = line_sieve(&poly, &rational_fb, &alg_roots, &config);
        // We should get some sieve hits for this small number
        assert!(
            !hits.is_empty(),
            "line_sieve should produce hits for n=15347"
        );
    }

    #[test]
    fn test_eval_homogeneous_abs() {
        // f(x) = x^2 + 1, f_hom(a, b) = a^2 + b^2
        let coeffs = vec![
            BigUint::from(1u64),
            BigUint::zero(),
            BigUint::from(1u64),
        ];
        assert_eq!(eval_homogeneous_abs(&coeffs, 3, 4), BigUint::from(25u64));
        assert_eq!(eval_homogeneous_abs(&coeffs, -3, 4), BigUint::from(25u64));
    }

    #[test]
    fn test_gcd_u64() {
        assert_eq!(gcd_u64(12, 8), 4);
        assert_eq!(gcd_u64(17, 13), 1);
        assert_eq!(gcd_u64(0, 5), 5);
    }
}
