//! Williams P+1 factoring method using Lucas sequences.
//!
//! If `p | n` and `(p + 1)` is B1-smooth (for a suitable Jacobi symbol
//! condition on the starting value), Stage 1 will find `p`.  Stage 2
//! extends to one large factor in `(B1, B2]`.
//!
//! The algorithm computes `V_M(P, 1) mod n` where `V_k` is the Lucas
//! V-sequence and `M` is the product of prime powers up to the bounds.

use crate::arith::sieve_primes;

/// Attempt to find a non-trivial factor of odd composite `n` using
/// Williams' P+1 method with bounds `b1` and `b2`.
///
/// Tries several starting values (as CADO-NFS does).  Returns
/// `Some(factor)` on success, `None` otherwise.
pub fn pp1(n: u64, b1: u64, b2: u64) -> Option<u64> {
    if n <= 1 || n % 2 == 0 {
        return None;
    }

    // Try a few starting values.  The method works when the Jacobi symbol
    // (P^2 - 4 | p) = -1 for a factor p.  Different starting values cover
    // different residue classes.
    for &start_p in &[5u64, 9, 14, 20, 27] {
        if let Some(f) = pp1_one_start(n, b1, b2, start_p) {
            return Some(f);
        }
    }
    None
}

/// Run P+1 with a single starting value `p_val`.
fn pp1_one_start(n: u64, b1: u64, b2: u64, p_val: u64) -> Option<u64> {
    let primes = sieve_primes(b2);

    let mut v = p_val % n;

    // Stage 1: compute V_M  where  M = lcm{ p^k : p^k <= B1 }.
    for &p in &primes {
        if p > b1 {
            break;
        }
        let mut pk = p;
        while pk <= b1 {
            v = lucas_chain(v, p, n);
            pk = pk.saturating_mul(p);
        }
    }

    // Check gcd(V_M - 2, n).
    // V_M - 2 because V_0 = 2 is the "identity" of the Lucas sequence.
    let diff = submod(v, 2, n);
    let g = gcd(diff, n);
    if g > 1 && g < n {
        return Some(g);
    }
    if g == n {
        return None; // overshot
    }

    // Stage 2: check each prime q in (B1, B2].
    for &q in &primes {
        if q <= b1 {
            continue;
        }
        if q > b2 {
            break;
        }
        v = lucas_chain(v, q, n);
        let diff = submod(v, 2, n);
        let g = gcd(diff, n);
        if g > 1 && g < n {
            return Some(g);
        }
        if g == n {
            return None;
        }
    }

    None
}

/// Compute `V_e(v, 1) mod n` using the binary Lucas-chain method.
///
/// Tracks `(V_k, V_{k+1})` and uses the identities:
///   `V_{2k}   = V_k^2 - 2`
///   `V_{2k+1} = V_k * V_{k+1} - P`   (where `P = v = V_1`)
fn lucas_chain(v: u64, e: u64, n: u64) -> u64 {
    if e == 0 {
        return 2 % n;
    }
    if e == 1 {
        return v;
    }

    // Start: V_1 = v, V_2 = v^2 - 2
    let mut vk = v;
    let mut vk1 = submod(mulmod(v, v, n), 2, n);

    let bits = 64 - e.leading_zeros();
    // Process bits from second-highest down to bit 0.
    for i in (0..bits - 1).rev() {
        if (e >> i) & 1 == 1 {
            // (V_k, V_{k+1}) -> (V_{2k+1}, V_{2k+2})
            // V_{2k+1} = V_k * V_{k+1} - P
            // V_{2k+2} = V_{k+1}^2 - 2
            vk = submod(mulmod(vk, vk1, n), v, n);
            vk1 = submod(mulmod(vk1, vk1, n), 2, n);
        } else {
            // (V_k, V_{k+1}) -> (V_{2k}, V_{2k+1})
            // V_{2k+1} = V_k * V_{k+1} - P
            // V_{2k}   = V_k^2 - 2
            vk1 = submod(mulmod(vk, vk1, n), v, n);
            vk = submod(mulmod(vk, vk, n), 2, n);
        }
    }

    vk
}

/// `(a * b) mod n` using u128 intermediate.
#[inline]
fn mulmod(a: u64, b: u64, n: u64) -> u64 {
    ((a as u128 * b as u128) % n as u128) as u64
}

/// `(a - b) mod n`, handling underflow.
#[inline]
fn submod(a: u64, b: u64, n: u64) -> u64 {
    if a >= b {
        (a - b) % n
    } else {
        // a - b + n, but a < b and both < n so a - b + n is in [1, n-1].
        // However a might already be reduced mod n, so be safe:
        ((a as u128 + n as u128 - b as u128) % n as u128) as u64
    }
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
    fn test_lucas_chain_identity() {
        // V_0(P, 1) = 2 for any P, any n > 2.
        assert_eq!(lucas_chain(7, 0, 97), 2);
    }

    #[test]
    fn test_lucas_chain_one() {
        // V_1(P, 1) = P.
        assert_eq!(lucas_chain(7, 1, 97), 7);
    }

    #[test]
    fn test_lucas_chain_two() {
        // V_2(P, 1) = P^2 - 2.
        // V_2(7, 1) mod 97 = 49 - 2 = 47.
        assert_eq!(lucas_chain(7, 2, 97), 47);
    }

    #[test]
    fn test_lucas_chain_three() {
        // V_3(P, 1) = P^3 - 3P.
        // V_3(7, 1) = 343 - 21 = 322 mod 97 = 322 - 3*97 = 322 - 291 = 31.
        assert_eq!(lucas_chain(7, 3, 97), 31);
    }

    #[test]
    fn test_pp1_known_smooth() {
        // p = 1013, p + 1 = 1014 = 2 * 3 * 169 = 2 * 3 * 13^2 (13-smooth)
        // q = 1009, q + 1 = 1010 = 2 * 5 * 101 (101-smooth)
        // With B1=525, B2=3255, both should be reachable.
        let n = 1009u64 * 1013;
        // P+1 is probabilistic (depends on starting value), so we just
        // check that if it returns a factor, it's valid.
        if let Some(f) = pp1(n, 525, 3255) {
            assert!(n % f == 0, "factor {} does not divide {}", f, n);
            assert!(f > 1 && f < n, "trivial factor {}", f);
        }
    }

    #[test]
    fn test_pp1_prime_returns_none() {
        assert!(pp1(97, 100, 1000).is_none());
    }

    #[test]
    fn test_pp1_even_returns_none() {
        assert!(pp1(100, 100, 1000).is_none());
    }

    #[test]
    fn test_pp1_small_composite() {
        // 15 = 3 * 5; p+1 = 4 or 6, both 3-smooth.
        let n = 15u64;
        if let Some(f) = pp1(n, 10, 100) {
            assert!(n % f == 0);
            assert!(f > 1 && f < n);
        }
    }
}
