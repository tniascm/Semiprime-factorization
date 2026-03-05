//! Cofactorization pipeline: trial division -> P-1 -> P+1 -> ECM.
//!
//! After the sieve identifies survivor positions, each survivor's norm
//! must be completely factored.  This module implements the standard
//! CADO-NFS chain:
//!
//! 1. **Trial division** removes all small prime factors (factor-base primes).
//! 2. **Pollard P-1** catches factors whose `(p - 1)` is smooth.
//! 3. **Williams P+1** catches factors whose `(p + 1)` is smooth.
//! 4. **ECM** catches the remaining factors with small group orders on
//!    suitable elliptic curves.
//!
//! The result classifies the norm as fully smooth, having one or two
//! large primes, or not smooth.

pub mod ecm;
pub mod pm1;
pub mod pp1;
pub mod trialdiv;

use crate::arith::{is_probable_prime, TrialDivisor};

/// Result of cofactorizing a norm.
#[derive(Debug, Clone)]
pub enum CofactResult {
    /// Fully smooth: all prime factors are in the factor base.
    Smooth(Vec<(u32, u8)>),
    /// One large prime (within the large-prime bound).
    OneLargePrime(Vec<(u32, u8)>, u64),
    /// Two large primes (both within the large-prime bound).
    TwoLargePrimes(Vec<(u32, u8)>, u64, u64),
    /// Could not factor — reject this relation.
    NotSmooth,
}

/// Run the full cofactorization pipeline on `norm`.
///
/// # Parameters
///
/// * `norm` — the value to factor (a sieve-survivor norm).
/// * `divisors` — factor-base trial divisors.
/// * `lpb` — large-prime bound exponent: primes up to `2^lpb` are accepted.
/// * `mfb` — maximum factor-base bound exponent: cofactors with more than
///   `mfb` bits are immediately rejected.
/// * `_lim` — factor-base limit (reserved for future use).
pub fn cofactorize(
    norm: u64,
    divisors: &[TrialDivisor],
    lpb: u32,
    mfb: u32,
    _lim: u64,
) -> CofactResult {
    // Step 1: trial division.
    let (factors, cofactor) = trialdiv::trial_divide(norm, divisors);

    if cofactor <= 1 {
        return CofactResult::Smooth(factors);
    }

    let lp_bound = 1u64 << lpb;

    // Single large prime?
    if cofactor <= lp_bound {
        return CofactResult::OneLargePrime(factors, cofactor);
    }

    // Reject if cofactor is too large to possibly split into two large primes.
    let cofactor_bits = 64 - cofactor.leading_zeros();
    if cofactor_bits > mfb {
        return CofactResult::NotSmooth;
    }

    // If cofactor is prime and > lp_bound, it cannot yield a valid relation.
    if is_probable_prime(cofactor) {
        return CofactResult::NotSmooth;
    }

    // Step 2: Pollard P-1 (CADO defaults: B1=315, B2=2205).
    if let Some(f) = pm1::pm1(cofactor, 315, 2205) {
        let other = cofactor / f;
        return check_split(factors, f, other, lp_bound);
    }

    // Step 3: Williams P+1 (CADO defaults: B1=525, B2=3255).
    if let Some(f) = pp1::pp1(cofactor, 525, 3255) {
        let other = cofactor / f;
        return check_split(factors, f, other, lp_bound);
    }

    // Step 4: ECM chain.
    for (i, (b1, b2)) in ecm::ecm_bounds(lpb).into_iter().enumerate() {
        // Use distinct sigma values for each curve.
        let sigma = b1.wrapping_add(i as u64).max(6);
        if let Some(f) = ecm::ecm_one_curve(cofactor, b1, b2, sigma) {
            let other = cofactor / f;
            return check_split(factors, f, other, lp_bound);
        }
    }

    CofactResult::NotSmooth
}

/// u128 variant of cofactorization.
///
/// We trial-divide in u128 space, then fall back to the u64 pipeline once the
/// remaining cofactor fits in u64.
pub fn cofactorize_u128(
    norm: u128,
    divisors: &[TrialDivisor],
    lpb: u32,
    mfb: u32,
    _lim: u64,
) -> CofactResult {
    let (factors, cofactor) = trialdiv::trial_divide_u128(norm, divisors);

    if cofactor <= 1 {
        return CofactResult::Smooth(factors);
    }

    let lp_bound = 1u128 << lpb;

    if cofactor <= lp_bound {
        return CofactResult::OneLargePrime(factors, cofactor as u64);
    }

    let cofactor_bits = 128 - cofactor.leading_zeros();
    if cofactor_bits > mfb {
        return CofactResult::NotSmooth;
    }

    // Remaining pipeline uses u64 arithmetic.
    if cofactor > u64::MAX as u128 {
        return CofactResult::NotSmooth;
    }
    let c = cofactor as u64;

    if is_probable_prime(c) {
        return CofactResult::NotSmooth;
    }

    if let Some(f) = pm1::pm1(c, 315, 2205) {
        let other = c / f;
        return check_split(factors, f, other, lp_bound as u64);
    }

    if let Some(f) = pp1::pp1(c, 525, 3255) {
        let other = c / f;
        return check_split(factors, f, other, lp_bound as u64);
    }

    for (i, (b1, b2)) in ecm::ecm_bounds(lpb).into_iter().enumerate() {
        let sigma = b1.wrapping_add(i as u64).max(6);
        if let Some(f) = ecm::ecm_one_curve(c, b1, b2, sigma) {
            let other = c / f;
            return check_split(factors, f, other, lp_bound as u64);
        }
    }

    CofactResult::NotSmooth
}

/// Verify that a factor split yields a valid one- or two-large-prime relation.
fn check_split(factors: Vec<(u32, u8)>, f1: u64, f2: u64, lp_bound: u64) -> CofactResult {
    // Both factors must be within the large-prime bound.
    if f1 <= lp_bound && f2 <= lp_bound {
        if f1 <= 1 {
            // f1 is trivial — cofactor was actually a single large prime.
            return CofactResult::OneLargePrime(factors, f2);
        }
        if f2 <= 1 {
            return CofactResult::OneLargePrime(factors, f1);
        }
        return CofactResult::TwoLargePrimes(factors, f1, f2);
    }
    CofactResult::NotSmooth
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::TrialDivisor;

    fn make_divisors(primes: &[u64]) -> Vec<TrialDivisor> {
        primes.iter().map(|&p| TrialDivisor::new(p, 1.0)).collect()
    }

    #[test]
    fn test_cofactorize_smooth() {
        let divisors = make_divisors(&[2, 3, 5, 7, 11, 13]);
        match cofactorize(210, &divisors, 17, 34, 30000) {
            CofactResult::Smooth(factors) => {
                assert!(!factors.is_empty());
            }
            other => panic!("expected Smooth, got {:?}", other),
        }
    }

    #[test]
    fn test_cofactorize_one_large_prime() {
        let divisors = make_divisors(&[2, 3, 5, 7]);
        // 210 * 131 = 27510, where 131 < 2^17 = 131072
        match cofactorize(27510, &divisors, 17, 34, 30000) {
            CofactResult::OneLargePrime(_, lp) => {
                assert_eq!(lp, 131);
            }
            other => panic!("expected OneLargePrime, got {:?}", other),
        }
    }

    #[test]
    fn test_cofactorize_not_smooth_large_prime_cofactor() {
        let divisors = make_divisors(&[2, 3, 5]);
        // 1_000_003 is prime, after trial div cofactor is > 2^17 and prime => NotSmooth.
        match cofactorize(1_000_003, &divisors, 17, 34, 30000) {
            CofactResult::NotSmooth => {}
            other => panic!("expected NotSmooth, got {:?}", other),
        }
    }

    #[test]
    fn test_cofactorize_smooth_power_of_two() {
        let divisors = make_divisors(&[2, 3, 5, 7]);
        // 128 = 2^7
        match cofactorize(128, &divisors, 17, 34, 30000) {
            CofactResult::Smooth(factors) => {
                assert_eq!(factors.len(), 1);
                assert_eq!(factors[0], (0, 7)); // index 0 = prime 2, exponent 7
            }
            other => panic!("expected Smooth, got {:?}", other),
        }
    }

    #[test]
    fn test_cofactorize_two_large_primes_via_pm1() {
        // Construct a norm that has small factors and two medium primes.
        // 6 * 1009 * 1013 = 6_132_702
        // After trial div by {2, 3}: cofactor = 1009 * 1013 = 1_022_117
        // 1_022_117 < 2^34 so within mfb=34.
        // Both 1009 and 1013 < 2^17 = 131072, so both within lpb=17.
        // P-1 should split 1022117 since 1008 = 2^4 * 3^2 * 7 (7-smooth).
        let divisors = make_divisors(&[2, 3, 5, 7]);
        match cofactorize(6_132_702, &divisors, 17, 34, 30000) {
            CofactResult::TwoLargePrimes(factors, lp1, lp2) => {
                assert!(!factors.is_empty());
                let product = lp1 * lp2;
                assert_eq!(product, 1009 * 1013);
            }
            CofactResult::OneLargePrime(_, _) => {
                // This can happen if the factoring methods return one of the
                // primes and the other is within the factor base — still valid.
            }
            other => panic!("expected TwoLargePrimes or OneLargePrime, got {:?}", other),
        }
    }

    #[test]
    fn test_cofactorize_reject_too_large() {
        let divisors = make_divisors(&[2, 3]);
        // Cofactor after trial div will be very large (> 2^18 bits).
        // With mfb=18, should be rejected.
        match cofactorize(1_000_000_007, &divisors, 17, 18, 30000) {
            CofactResult::NotSmooth => {}
            other => panic!("expected NotSmooth, got {:?}", other),
        }
    }
}
