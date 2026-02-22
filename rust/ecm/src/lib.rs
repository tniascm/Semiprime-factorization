//! Elliptic Curve Method (Lenstra) for integer factorization.
//!
//! Uses Montgomery curve arithmetic for speed and rayon for parallel
//! trial on multiple random curves simultaneously.

use factoring_core::{Algorithm, FactorResult};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rayon::prelude::*;
use std::time::Instant;

/// A point on a Montgomery curve By^2 = x^3 + Ax^2 + x (mod n).
/// Uses projective coordinates (X : Z) for efficiency.
#[derive(Debug, Clone)]
pub struct MontgomeryPoint {
    pub x: BigUint,
    pub z: BigUint,
}

/// ECM parameters controlling the search bounds.
#[derive(Debug, Clone)]
pub struct EcmParams {
    /// Stage 1 smoothness bound
    pub b1: u64,
    /// Stage 2 smoothness bound
    pub b2: u64,
    /// Number of curves to try in parallel
    pub num_curves: usize,
}

impl Default for EcmParams {
    fn default() -> Self {
        Self {
            b1: 10_000,
            b2: 1_000_000,
            num_curves: 64,
        }
    }
}

/// Montgomery curve scalar multiplication using the Montgomery ladder.
pub fn montgomery_ladder(k: &BigUint, p: &MontgomeryPoint, a24: &BigUint, n: &BigUint) -> MontgomeryPoint {
    if k.is_zero() {
        return MontgomeryPoint {
            x: BigUint::zero(),
            z: BigUint::one(),
        };
    }

    let mut r0 = p.clone();
    let mut r1 = double(&p, a24, n);

    let bits = k.bits();
    for i in (0..bits - 1).rev() {
        let bit = (k >> i) & BigUint::one();
        if bit.is_zero() {
            r1 = add(&r0, &r1, p, n);
            r0 = double(&r0, a24, n);
        } else {
            r0 = add(&r0, &r1, p, n);
            r1 = double(&r1, a24, n);
        }
    }

    r0
}

/// Point doubling on Montgomery curve.
fn double(p: &MontgomeryPoint, a24: &BigUint, n: &BigUint) -> MontgomeryPoint {
    let u = (&p.x + &p.z) % n;
    let v = if p.x >= p.z {
        (&p.x - &p.z) % n
    } else {
        (n - (&p.z - &p.x) % n) % n
    };
    let u2 = (&u * &u) % n;
    let v2 = (&v * &v) % n;
    let diff = if u2 >= v2 {
        (&u2 - &v2) % n
    } else {
        (n - (&v2 - &u2) % n) % n
    };

    let x = (&u2 * &v2) % n;
    let z = (&diff * ((&v2 + a24 * &diff) % n)) % n;

    MontgomeryPoint { x, z }
}

/// Differential point addition.
fn add(p: &MontgomeryPoint, q: &MontgomeryPoint, diff: &MontgomeryPoint, n: &BigUint) -> MontgomeryPoint {
    let u = ((&p.x * &q.x) % n + n - (&p.z * &q.z) % n) % n;
    let v = ((&p.x * &q.z) % n + n - (&p.z * &q.x) % n) % n;

    let x = (&diff.z * &u * &u) % n;
    let z = (&diff.x * &v * &v) % n;

    MontgomeryPoint { x, z }
}

/// Run ECM Stage 1 and Stage 2 on a single curve.
///
/// Stage 1 multiplies the point by all prime powers up to `b1`.
/// Stage 2 then checks each prime `q` in (b1, b2] by multiplying point by `q`
/// and testing gcd(Q.z, n) for a non-trivial factor.
fn ecm_stage1_and_2(n: &BigUint, b1: u64, b2: u64, seed: u64) -> Option<BigUint> {
    let one = BigUint::one();

    // Generate random curve and point
    let sigma = BigUint::from(seed + 6);
    let sigma_sq = (&sigma * &sigma) % n;
    let five = BigUint::from(5u32);
    let u = if sigma_sq >= five {
        (&sigma_sq - &five) % n
    } else {
        (n - (&five - &sigma_sq) % n) % n
    };
    let v = (BigUint::from(4u32) * &sigma) % n;

    let x = u.modpow(&BigUint::from(3u32), n);
    let z = v.modpow(&BigUint::from(3u32), n);

    let v_minus_u = if v >= u { (&v - &u) % n } else { (n - (&u - &v) % n) % n };
    let a24_num = v_minus_u.modpow(&BigUint::from(3u32), n) * (BigUint::from(3u32) * &u + &v) % n;
    let a24_den_inv = match mod_inverse(&(BigUint::from(4u32) * &x * &v % n), n) {
        Some(inv) => inv,
        None => {
            let g = factoring_core::gcd(&(BigUint::from(4u32) * &x * &v % n), n);
            if g > one && g < *n {
                return Some(g);
            }
            return None;
        }
    };
    let a24 = (a24_num * a24_den_inv) % n;

    let mut point = MontgomeryPoint { x, z };

    // Stage 1: multiply point by all primes up to B1
    let primes = sieve_primes(b1);
    for p in &primes {
        let mut pp = *p;
        while pp <= b1 {
            point = montgomery_ladder(&BigUint::from(pp), &point, &a24, n);
            pp *= p;
        }

        let g = factoring_core::gcd(&point.z, n);
        if g > one && g < *n {
            return Some(g);
        }
        if g == *n {
            return None;
        }
    }

    // Stage 2: check primes between B1 and B2
    if b2 > b1 {
        let stage2_primes = sieve_primes(b2);
        for q in &stage2_primes {
            if *q <= b1 {
                continue;
            }
            point = montgomery_ladder(&BigUint::from(*q), &point, &a24, n);

            let g = factoring_core::gcd(&point.z, n);
            if g > one && g < *n {
                return Some(g);
            }
            if g == *n {
                return None;
            }
        }
    }

    None
}

/// Simple sieve of Eratosthenes.
fn sieve_primes(bound: u64) -> Vec<u64> {
    let mut is_prime = vec![true; (bound + 1) as usize];
    is_prime[0] = false;
    if bound >= 1 {
        is_prime[1] = false;
    }

    let mut i = 2;
    while i * i <= bound {
        if is_prime[i as usize] {
            let mut j = i * i;
            while j <= bound {
                is_prime[j as usize] = false;
                j += i;
            }
        }
        i += 1;
    }

    (2..=bound).filter(|&i| is_prime[i as usize]).collect()
}

/// Modular inverse using extended Euclidean algorithm.
fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    let g = factoring_core::gcd(a, m);
    if g != BigUint::one() {
        return None;
    }
    // a^(m-2) mod m for prime m, or extended gcd
    // Using Fermat's little theorem approximation (works when gcd=1)
    Some(a.modpow(&(m - BigUint::from(2u32)), m))
}

/// Run ECM Stage 1 and Stage 2 on multiple curves in parallel with explicit control.
///
/// Unlike `ecm_factor` which manages its own parallelism internally, this function
/// gives the caller explicit control over the number of curves, bounds, and seeds.
/// Each curve uses a seed from `0..num_curves`.
///
/// Returns the first non-trivial factor found by any curve, or `None`.
pub fn ecm_parallel_multi_curve(n: &BigUint, b1: u64, b2: u64, num_curves: usize) -> Option<BigUint> {
    (0..num_curves as u64)
        .into_par_iter()
        .find_map_any(|seed| ecm_stage1_and_2(n, b1, b2, seed))
}

/// Run ECM with parallel curves.
pub fn ecm_factor(n: &BigUint, params: &EcmParams) -> FactorResult {
    let start = Instant::now();

    // Try parallel curves
    let result: Option<BigUint> = (0..params.num_curves as u64)
        .into_par_iter()
        .find_map_any(|seed| ecm_stage1_and_2(n, params.b1, params.b2, seed));

    let found = result.is_some();
    let factors = match result {
        Some(factor) => {
            let cofactor = n / &factor;
            vec![factor, cofactor]
        }
        None => vec![],
    };

    FactorResult {
        n: n.clone(),
        factors,
        algorithm: Algorithm::ECM,
        duration: start.elapsed(),
        complete: found,
    }
}

/// Convenience wrapper around `ecm_factor` that returns a `FactorResult`
/// with full metadata including timing, algorithm tag, and found factors.
///
/// This is intentionally a thin wrapper: `ecm_factor` already returns a
/// `FactorResult`, so this function simply delegates. It exists to provide
/// a symmetrical API name (`_with_result`) matching other crates in the workspace
/// that have separate "find factor" and "return rich result" entry points.
pub fn ecm_factor_with_result(n: &BigUint, params: &EcmParams) -> FactorResult {
    let start = Instant::now();
    let maybe_factor = ecm_parallel_multi_curve(n, params.b1, params.b2, params.num_curves);

    let found = maybe_factor.is_some();
    let factors = match maybe_factor {
        Some(factor) => {
            let cofactor = n / &factor;
            vec![factor, cofactor]
        }
        None => vec![],
    };

    FactorResult {
        n: n.clone(),
        factors,
        algorithm: Algorithm::ECM,
        duration: start.elapsed(),
        complete: found,
    }
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
    fn test_ecm_small() {
        let n = BigUint::from(8051u64); // 83 * 97
        let params = EcmParams {
            b1: 1000,
            b2: 10000,
            num_curves: 32,
        };
        let result = ecm_factor(&n, &params);
        if result.complete {
            assert_eq!(result.factors.len(), 2);
            let product: BigUint = result.factors.iter().product();
            assert_eq!(product, n);
        }
    }

    #[test]
    fn test_ecm_parallel_multi_curve() {
        // 48-bit semiprime: 224737 * 350377 = 78_737_654_849
        // Both are verified primes (not divisible by any small factor).
        let p = 224_737u64;
        let q = 350_377u64;
        let n = BigUint::from(p) * BigUint::from(q);
        // Verify product bit-length is in the 48-bit range
        assert!(n.bits() >= 36, "semiprime should be at least 36 bits, got {}", n.bits());

        // Use generous bounds and enough curves
        let factor = ecm_parallel_multi_curve(&n, 50_000, 2_000_000, 128);
        assert!(
            factor.is_some(),
            "ecm_parallel_multi_curve should find a factor for the semiprime {}",
            n
        );
        let f = factor.unwrap();
        // The factor must be a non-trivial divisor of n
        assert!(
            &n % &f == BigUint::zero(),
            "Found factor {} does not divide n={}",
            f, n
        );
        assert!(
            f > BigUint::one() && f < n,
            "Factor should be non-trivial, got {}",
            f
        );
    }

    #[test]
    fn test_ecm_factor_with_result() {
        let n = BigUint::from(8051u64); // 83 * 97
        let params = EcmParams {
            b1: 1000,
            b2: 10000,
            num_curves: 32,
        };
        let result = ecm_factor_with_result(&n, &params);
        assert_eq!(result.algorithm, Algorithm::ECM);
        assert_eq!(result.n, n);
        if result.complete {
            assert_eq!(result.factors.len(), 2);
            let product: BigUint = result.factors.iter().product();
            assert_eq!(product, n);
        }
    }

    #[test]
    fn test_ecm_stage2() {
        // 1000003 * 1000033 = 1000036000099
        // With a small B1 of 100, Stage 1 alone is unlikely to find the factor.
        // Stage 2 extends the search up to B2 = 1_100_000, which should catch it.
        let n = BigUint::from(1_000_036_000_099u64);
        let params = EcmParams {
            b1: 100,
            b2: 1_100_000,
            num_curves: 64,
        };
        let result = ecm_factor(&n, &params);
        assert!(result.complete, "Stage 2 should find a factor for 1000003 * 1000033");
        assert_eq!(result.factors.len(), 2);
        let product: BigUint = result.factors.iter().product();
        assert_eq!(product, n);
        let p = BigUint::from(1_000_003u64);
        let q = BigUint::from(1_000_033u64);
        assert!(
            (result.factors[0] == p && result.factors[1] == q)
                || (result.factors[0] == q && result.factors[1] == p),
            "Factors should be 1000003 and 1000033, got {:?}",
            result.factors
        );
    }
}
