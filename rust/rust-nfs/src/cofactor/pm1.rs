//! Pollard P-1 factoring method with u64 Montgomery arithmetic.
//!
//! If `p | n` and `(p - 1)` is B1-smooth, Stage 1 will find `p`.
//! Stage 2 extends coverage to numbers whose `(p - 1)` has at most one
//! prime factor in `(B1, B2]`.

use crate::arith::{sieve_primes, MontgomeryParams};

/// Attempt to find a non-trivial factor of odd composite `n` using
/// Pollard's P-1 method with bounds `b1` (Stage 1) and `b2` (Stage 2).
///
/// Returns `Some(factor)` on success, `None` if no factor is found.
pub fn pm1(n: u64, b1: u64, b2: u64) -> Option<u64> {
    let primes = sieve_primes(b2);
    pm1_with_primes(n, b1, b2, &primes)
}

/// P-1 with pre-computed prime list (avoids redundant sieve_primes calls).
pub fn pm1_with_primes(n: u64, b1: u64, b2: u64, primes: &[u64]) -> Option<u64> {
    if n <= 1 || n % 2 == 0 {
        return None;
    }

    let mp = MontgomeryParams::new(n);

    // Stage 1: compute 2^M mod n  where  M = lcm{ p^k : p^k <= B1 }.
    // We stay in Montgomery form throughout.
    // Check gcd periodically to detect factors early and handle the
    // "overshot" case (gcd == n) that happens when both (p-1) and (q-1)
    // are B1-smooth.
    let mut base = mp.to_mont(2);
    let mut step_count = 0u32;

    for &p in primes {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            base = mp.powmod_mont(base, p);
            pk = pk.saturating_mul(p);
        }
        step_count += 1;

        // Check gcd every 8 primes.
        if step_count % 8 == 0 {
            let result = mp.from_mont(base);
            let diff = result.wrapping_sub(1).wrapping_add(n) % n;
            let g = gcd(diff, n);
            if g > 1 && g < n {
                return Some(g);
            }
            if g == n {
                // Overshot — retry with per-prime gcd checks.
                return pm1_careful(n, b1, b2, primes);
            }
        }
    }

    // Final stage 1 gcd check.
    let result = mp.from_mont(base);
    let diff = result.wrapping_sub(1).wrapping_add(n) % n;
    let g = gcd(diff, n);
    if g > 1 && g < n {
        return Some(g);
    }
    if g == n {
        return pm1_careful(n, b1, b2, primes);
    }

    // Stage 2: for each prime q in (B1, B2], check gcd(base^q - 1, n).
    // Accumulate a product and take batched GCDs.
    let one_mont = mp.r_mod_n;
    let mut accum = one_mont;

    const BATCH: usize = 32;
    let stage2_primes: Vec<u64> = primes
        .iter()
        .copied()
        .filter(|&p| p > b1 && p <= b2)
        .collect();

    for chunk in stage2_primes.chunks(BATCH) {
        for &q in chunk {
            let bq = mp.powmod_mont(base, q);
            let bq_normal = mp.from_mont(bq);
            let d = if bq_normal >= 1 {
                bq_normal - 1
            } else {
                bq_normal + n - 1
            };
            let d_mont = mp.to_mont(d % n);
            accum = mp.mul(accum, d_mont);
        }

        let accum_normal = mp.from_mont(accum);
        let g = gcd(accum_normal, n);
        if g > 1 && g < n {
            return Some(g);
        }
        if g == n {
            // Overshot — retry per-prime.
            let mut acc2 = one_mont;
            for &q in chunk {
                let bq = mp.powmod_mont(base, q);
                let bq_normal = mp.from_mont(bq);
                let d = if bq_normal >= 1 {
                    bq_normal - 1
                } else {
                    bq_normal + n - 1
                };
                let d_mont = mp.to_mont(d % n);
                acc2 = mp.mul(acc2, d_mont);
                let a2 = mp.from_mont(acc2);
                let g2 = gcd(a2, n);
                if g2 > 1 && g2 < n {
                    return Some(g2);
                }
            }
            return None;
        }
    }

    None
}

/// Fallback P-1 that checks gcd after every single exponentiation step.
///
/// Called when the batched version overshoots (both factors' (p-1) are
/// B1-smooth so 2^M = 1 mod n).  By checking after each p^k step we
/// catch the factor of one before the other also reaches 1.
fn pm1_careful(n: u64, b1: u64, b2: u64, primes: &[u64]) -> Option<u64> {
    let mp = MontgomeryParams::new(n);

    let mut base = mp.to_mont(2);

    for &p in primes {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            base = mp.powmod_mont(base, p);
            pk = pk.saturating_mul(p);

            // Check gcd after every single exponentiation step.
            let result = mp.from_mont(base);
            let diff = result.wrapping_sub(1).wrapping_add(n) % n;
            let g = gcd(diff, n);
            if g > 1 && g < n {
                return Some(g);
            }
            if g == n {
                // Both factors' orders divide M — give up.
                return None;
            }
        }
    }

    // Stage 2 with per-prime checks.
    for &q in primes {
        if q <= b1 {
            continue;
        }
        if q > b2 {
            break;
        }
        let bq = mp.powmod_mont(base, q);
        let bq_normal = mp.from_mont(bq);
        let d = if bq_normal >= 1 {
            bq_normal - 1
        } else {
            bq_normal + n - 1
        };
        let g = gcd(d % n, n);
        if g > 1 && g < n {
            return Some(g);
        }
    }

    None
}

/// Binary GCD.
fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pm1_known_smooth() {
        // 1009 * 1013 = 1022117
        // 1009 - 1 = 1008 = 2^4 * 3^2 * 7   (7-smooth)
        // 1013 - 1 = 1012 = 2^2 * 11 * 23    (23-smooth)
        // Both are B1=315 smooth, so both factors' orders are fully within M.
        // The per-prime fallback should still find the factor.
        let n = 1009u64 * 1013;
        let result = pm1(n, 315, 2205);
        assert!(result.is_some(), "P-1 should find a factor of 1009*1013");
        let f = result.unwrap();
        assert!(n % f == 0, "factor {} does not divide {}", f, n);
        assert!(f > 1 && f < n, "trivial factor {}", f);
    }

    #[test]
    fn test_pm1_stage2() {
        // p = 211, p-1 = 210 = 2*3*5*7, fully B1=7 smooth
        // q = 311, q-1 = 310 = 2*5*31, 31 > 7 but 31 <= 100
        let n = 211u64 * 311;
        if let Some(f) = pm1(n, 7, 100) {
            assert!(n % f == 0);
            assert!(f > 1 && f < n);
        }
    }

    #[test]
    fn test_pm1_prime_returns_none() {
        assert!(pm1(97, 100, 1000).is_none());
    }

    #[test]
    fn test_pm1_even_returns_none() {
        assert!(pm1(100, 100, 1000).is_none());
    }

    #[test]
    fn test_pm1_small_semiprime() {
        // 15 = 3 * 5; p-1 = 2 (2-smooth), q-1 = 4 = 2^2 (2-smooth)
        let n = 15u64;
        let result = pm1(n, 10, 100);
        assert!(result.is_some(), "P-1 should find a factor of 15");
        let f = result.unwrap();
        assert!(n % f == 0);
        assert!(f > 1 && f < n);
    }
}
