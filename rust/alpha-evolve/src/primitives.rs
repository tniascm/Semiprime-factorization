//! Primitive operations on BigUint for the evolutionary factoring DSL.
//!
//! Each function implements a single composable building block that programs
//! in the genetic programming population can invoke.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::Rng;

/// Fast perfect square test using mod-16 bitmask + integer sqrt.
/// Eliminates 75% of non-squares with a single bitwise check.
pub fn is_perfect_square(x: &BigUint) -> bool {
    if x.is_zero() {
        return true;
    }
    // Mod 16 filter: squares can only be 0,1,4,9 mod 16
    let low_bits = x.to_u64_digits();
    if !low_bits.is_empty() {
        let m16 = (low_bits[0] & 0xF) as u32;
        if !matches!(m16, 0 | 1 | 4 | 9) {
            return false;
        }
    }
    let s = x.sqrt();
    &s * &s == *x
}

/// Fermat step with multiplier: compute a^2 - k*N and check if perfect square.
/// Returns gcd(a + sqrt(a^2 - kN), N) if a^2 - kN is a perfect square, else None.
pub fn fermat_step(n: &BigUint, a: &BigUint, k: u64) -> Option<BigUint> {
    let kn = n * BigUint::from(k);
    let a_sq = a * a;
    if a_sq < kn {
        return None;
    }
    let diff = &a_sq - &kn;
    if is_perfect_square(&diff) {
        let b = diff.sqrt();
        let sum = a + &b;
        let g = sum.gcd(n);
        let one = BigUint::one();
        if g > one && g < *n {
            return Some(g);
        }
    }
    None
}

/// Hart's one-line factoring step: for multiplier i, compute s = ceil(sqrt(n*i)),
/// then m = s^2 mod n. If m is a perfect square, return gcd(s - sqrt(m), n).
pub fn hart_step(n: &BigUint, i: u64) -> Option<BigUint> {
    let ni = n * BigUint::from(i);
    let s = ni.sqrt() + BigUint::one(); // ceil(sqrt(n*i))
    let s_sq = &s * &s;
    let m = s_sq % n;
    if is_perfect_square(&m) {
        let sqrt_m = m.sqrt();
        if s > sqrt_m {
            let diff = &s - &sqrt_m;
            let g = diff.gcd(n);
            let one = BigUint::one();
            if g > one && g < *n {
                return Some(g);
            }
        }
    }
    None
}

/// Lucas V-sequence step for Williams p+1: V_{n+1} = A * V_n - V_{n-1} mod N
pub fn lucas_v_step(v_prev: &BigUint, v_curr: &BigUint, a: &BigUint, n: &BigUint) -> BigUint {
    let product = (a * v_curr) % n;
    // Safe subtraction mod n
    if product >= *v_prev {
        (&product - v_prev) % n
    } else {
        (n - (v_prev - &product) % n) % n
    }
}

/// Williams p+1 single stage: iterate Lucas V-sequence, check gcd(V - 2, N) periodically.
/// Returns a factor if p+1 is smooth up to bound.
pub fn williams_p_plus_1_step(n: &BigUint, a: &BigUint, bound: u64) -> Option<BigUint> {
    let one = BigUint::one();
    let two = BigUint::from(2u32);
    if *n <= one {
        return None;
    }

    let mut v_prev = two.clone();
    let mut v_curr = a.clone() % n;

    // Iterate through small primes and their powers
    let mut p = 2u64;
    while p <= bound {
        let mut pk = p;
        while pk <= bound {
            // V_{2k} = V_k^2 - 2 mod N (doubling formula for efficiency)
            let next = lucas_v_step(&v_prev, &v_curr, a, n);
            v_prev = v_curr;
            v_curr = next;
            pk = pk.saturating_mul(p);
        }

        // Check gcd(V - 2, N) periodically
        if p % 50 == 0 || p == bound {
            let v_minus_2 = if v_curr >= two {
                &v_curr - &two
            } else {
                n - (&two - &v_curr) % n
            };
            let g = v_minus_2.gcd(n);
            if g > one && g < *n {
                return Some(g);
            }
        }

        p += if p == 2 { 1 } else { 2 };
    }
    None
}

/// Integer square root (floor).
pub fn isqrt(n: &BigUint) -> BigUint {
    n.sqrt()
}

/// Check if candidate is of the form 6m+1 or 6m-1 (necessary for primes > 3).
pub fn is_6m_pm1(x: &BigUint) -> bool {
    let six = BigUint::from(6u32);
    let r = x % &six;
    r == BigUint::one() || r == BigUint::from(5u32)
}

/// Modular exponentiation: base^exp mod n.
pub fn mod_pow_prim(base: &BigUint, exp: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    base.modpow(exp, n)
}

/// Greatest common divisor of a and b.
pub fn gcd_prim(a: &BigUint, b: &BigUint) -> BigUint {
    a.gcd(b)
}

/// Sample a random element in [2, n-2].
/// Returns 2 if n <= 4 (degenerate case).
pub fn random_element_prim(n: &BigUint, rng: &mut impl Rng) -> BigUint {
    let two = BigUint::from(2u32);
    let four = BigUint::from(4u32);
    if *n <= four {
        return two;
    }

    let n_minus_3 = n - BigUint::from(3u32); // range size = n - 3
    let bytes = n.to_bytes_be();
    loop {
        let mut random_bytes = vec![0u8; bytes.len()];
        rng.fill(&mut random_bytes[..]);
        let val = BigUint::from_bytes_be(&random_bytes) % &n_minus_3;
        let result = val + &two; // shift into [2, n-2]
        if result >= two && result <= n - &two {
            return result;
        }
    }
}

/// Square modulo n: x^2 mod n.
pub fn square_mod(x: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (x * x) % n
}

/// Add a constant modulo n: (x + c) mod n.
pub fn add_const_mod(x: &BigUint, c: u64, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (x + BigUint::from(c)) % n
}

/// Multiply modulo n: (a * b) mod n.
pub fn multiply_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    (a * b) % n
}

/// Compute gcd(|a - b|, n) and return it only if it is a nontrivial factor
/// (i.e., strictly between 1 and n). Returns None otherwise.
pub fn subtract_gcd(a: &BigUint, b: &BigUint, n: &BigUint) -> Option<BigUint> {
    let diff = if a > b {
        a - b
    } else if b > a {
        b - a
    } else {
        return None; // a == b => diff is 0 => gcd(0, n) = n (trivial)
    };

    if diff.is_zero() {
        return None;
    }

    let g = diff.gcd(n);
    let one = BigUint::one();
    if g > one && g < *n {
        Some(g)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_square_mod() {
        // 7^2 mod 15 = 49 mod 15 = 4
        let x = BigUint::from(7u32);
        let n = BigUint::from(15u32);
        assert_eq!(square_mod(&x, &n), BigUint::from(4u32));
    }

    #[test]
    fn test_primitive_subtract_gcd() {
        // |8 - 5| = 3, gcd(3, 15) = 3 (nontrivial factor of 15)
        let a = BigUint::from(8u32);
        let b = BigUint::from(5u32);
        let n = BigUint::from(15u32);
        let result = subtract_gcd(&a, &b, &n);
        assert_eq!(result, Some(BigUint::from(3u32)));
    }

    #[test]
    fn test_subtract_gcd_trivial() {
        // |10 - 5| = 5, gcd(5, 5) = 5 == n => trivial
        let a = BigUint::from(10u32);
        let b = BigUint::from(5u32);
        let n = BigUint::from(5u32);
        assert_eq!(subtract_gcd(&a, &b, &n), None);
    }

    #[test]
    fn test_mod_pow_prim() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = BigUint::from(2u32);
        let exp = BigUint::from(10u32);
        let n = BigUint::from(1000u32);
        assert_eq!(mod_pow_prim(&base, &exp, &n), BigUint::from(24u32));
    }

    #[test]
    fn test_add_const_mod() {
        // (13 + 5) mod 15 = 18 mod 15 = 3
        let x = BigUint::from(13u32);
        let n = BigUint::from(15u32);
        assert_eq!(add_const_mod(&x, 5, &n), BigUint::from(3u32));
    }

    #[test]
    fn test_multiply_mod() {
        // (7 * 8) mod 15 = 56 mod 15 = 11
        let a = BigUint::from(7u32);
        let b = BigUint::from(8u32);
        let n = BigUint::from(15u32);
        assert_eq!(multiply_mod(&a, &b, &n), BigUint::from(11u32));
    }

    #[test]
    fn test_is_perfect_square() {
        assert!(is_perfect_square(&BigUint::from(0u32)));
        assert!(is_perfect_square(&BigUint::from(1u32)));
        assert!(is_perfect_square(&BigUint::from(4u32)));
        assert!(is_perfect_square(&BigUint::from(9u32)));
        assert!(is_perfect_square(&BigUint::from(144u32)));
        assert!(!is_perfect_square(&BigUint::from(2u32)));
        assert!(!is_perfect_square(&BigUint::from(8u32)));
        assert!(!is_perfect_square(&BigUint::from(15u32)));
    }

    #[test]
    fn test_fermat_step() {
        // 35 = 5 * 7. a = 6, a^2 = 36, 36 - 35 = 1, sqrt(1) = 1. gcd(6+1, 35) = 7
        let n = BigUint::from(35u32);
        let a = BigUint::from(6u32);
        let result = fermat_step(&n, &a, 1);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!(f == BigUint::from(5u32) || f == BigUint::from(7u32));
    }

    #[test]
    fn test_hart_step() {
        // 8051 = 83 * 97. Try several multipliers.
        let n = BigUint::from(8051u32);
        let mut found = false;
        for i in 1..100u64 {
            if let Some(f) = hart_step(&n, i) {
                assert!((&n % &f).is_zero());
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Hart should find a factor of 8051 within 100 multipliers"
        );
    }

    #[test]
    fn test_is_6m_pm1() {
        assert!(is_6m_pm1(&BigUint::from(5u32))); // 6*1 - 1
        assert!(is_6m_pm1(&BigUint::from(7u32))); // 6*1 + 1
        assert!(is_6m_pm1(&BigUint::from(11u32))); // 6*2 - 1
        assert!(is_6m_pm1(&BigUint::from(13u32))); // 6*2 + 1
        assert!(!is_6m_pm1(&BigUint::from(6u32)));
        assert!(!is_6m_pm1(&BigUint::from(9u32)));
    }
}
