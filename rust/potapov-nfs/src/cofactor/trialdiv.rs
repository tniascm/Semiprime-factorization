//! Batch trial division using precomputed Montgomery inverses.
//!
//! Each `TrialDivisor` carries a precomputed modular inverse that turns a
//! divisibility test into a single multiply + compare (no actual division
//! until we confirm a hit).

use crate::arith::TrialDivisor;

/// Trial divide `n` by every prime in `divisors`.
///
/// Returns `(factored_exponents, cofactor)` where each element of
/// `factored_exponents` is `(index_into_divisors, exponent)` and `cofactor`
/// is the remaining unfactored part of `n`.
///
/// All divisors are checked (using the Montgomery-inverse fast divisibility
/// test — one multiply + compare per prime, no actual division until a hit).
/// Once the remaining cofactor reaches 1 the loop exits early.
pub fn trial_divide(mut n: u64, divisors: &[TrialDivisor]) -> (Vec<(u32, u8)>, u64) {
    let mut factors = Vec::new();

    for (i, td) in divisors.iter().enumerate() {
        if n <= 1 {
            break;
        }
        let p = td.p;
        if td.divides(n) {
            let mut exp = 0u8;
            while n % p == 0 {
                n /= p;
                exp += 1;
            }
            factors.push((i as u32, exp));
        }
        // Early exit: if p² > n, remaining n is 1 or prime.
        // If n is a FB prime, find it via binary search so it's not
        // misclassified as a large prime cofactor.
        if p.wrapping_mul(p) > n {
            if n > 1 {
                if let Ok(j) = divisors[i + 1..].binary_search_by_key(&n, |d| d.p) {
                    factors.push(((i + 1 + j) as u32, 1));
                    n = 1;
                }
            }
            break;
        }
    }

    (factors, n)
}

/// Trial divide a u128 value by u64 factor-base divisors.
///
/// This is used for larger algebraic norms (e.g. c45) where exact norms can
/// exceed u64, but cofactors after trial division are still typically small.
pub fn trial_divide_u128(mut n: u128, divisors: &[TrialDivisor]) -> (Vec<(u32, u8)>, u128) {
    let mut factors = Vec::new();

    for (i, td) in divisors.iter().enumerate() {
        if n <= 1 {
            break;
        }
        let p64 = td.p;
        let p = p64 as u128;
        // Use Montgomery fast test when n fits in u64, else fall back to u128 mod.
        let divisible = if n <= u64::MAX as u128 {
            td.divides(n as u64)
        } else {
            n % p == 0
        };
        if divisible {
            let mut exp = 0u8;
            while n % p == 0 {
                n /= p;
                exp += 1;
            }
            factors.push((i as u32, exp));
        }
        // Early exit: if p² > n, remaining n is 1 or prime.
        // If n is a FB prime, find it via binary search.
        if (p * p) > n {
            if n > 1 {
                let n64 = n as u64;
                if let Ok(j) = divisors[i + 1..].binary_search_by_key(&n64, |d| d.p) {
                    factors.push(((i + 1 + j) as u32, 1));
                    n = 1;
                }
            }
            break;
        }
    }

    (factors, n)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_divisors(primes: &[u64]) -> Vec<TrialDivisor> {
        primes.iter().map(|&p| TrialDivisor::new(p, 1.0)).collect()
    }

    #[test]
    fn test_trial_divide_fully_smooth() {
        let divisors = make_divisors(&[2, 3, 5, 7, 11]);
        let (factors, cofactor) = trial_divide(2310, &divisors); // 2*3*5*7*11
        assert_eq!(cofactor, 1);
        assert_eq!(factors.len(), 5);
        // Verify each exponent is 1
        for &(_idx, exp) in &factors {
            assert_eq!(exp, 1);
        }
    }

    #[test]
    fn test_trial_divide_with_cofactor() {
        let divisors = make_divisors(&[2, 3, 5]);
        let (factors, cofactor) = trial_divide(210, &divisors); // 2*3*5*7
                                                                // 7 is larger than 5 and 5^2=25 > 7, so loop exits before checking 7.
                                                                // Actually 5^2=25 > 7 isn't reached because 7 is not in divisors.
                                                                // After removing 2,3,5 we get cofactor=7.
        assert_eq!(cofactor, 7);
        assert_eq!(factors.len(), 3);
    }

    #[test]
    fn test_trial_divide_prime_input() {
        let divisors = make_divisors(&[2, 3, 5, 7, 11, 13]);
        let (factors, cofactor) = trial_divide(97, &divisors);
        // 97 is prime; 11^2 = 121 > 97 so we stop after checking up to 7.
        // None of 2,3,5,7 divide 97 => cofactor = 97, no factors found.
        assert!(factors.is_empty());
        assert_eq!(cofactor, 97);
    }

    #[test]
    fn test_trial_divide_perfect_power() {
        let divisors = make_divisors(&[2, 3, 5]);
        let (factors, cofactor) = trial_divide(72, &divisors); // 2^3 * 3^2
        assert_eq!(cofactor, 1);
        // index 0 = prime 2, exponent 3
        assert_eq!(factors[0], (0, 3));
        // index 1 = prime 3, exponent 2
        assert_eq!(factors[1], (1, 2));
    }

    #[test]
    fn test_trial_divide_one() {
        let divisors = make_divisors(&[2, 3, 5]);
        let (factors, cofactor) = trial_divide(1, &divisors);
        assert!(factors.is_empty());
        assert_eq!(cofactor, 1);
    }
}
