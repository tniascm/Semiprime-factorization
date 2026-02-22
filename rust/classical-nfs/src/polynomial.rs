//! NFS Polynomial selection (base-m method).
//!
//! Selects a polynomial f(x) such that f(m) = 0 (mod n) where m = floor(n^(1/(d+1))).
//! The polynomial is used to define a number field Q[x]/(f(x)) in which we sieve
//! for smooth algebraic norms.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// A polynomial f(x) = c_d * x^d + c_{d-1} * x^{d-1} + ... + c_0
/// chosen so that f(m) = 0 (mod n) where m = floor(n^(1/(d+1))).
#[derive(Debug, Clone)]
pub struct NfsPolynomial {
    /// Coefficients in order [c_0, c_1, ..., c_d] (low-degree first).
    pub coefficients: Vec<BigUint>,
    /// The base m such that f(m) = 0 (mod n).
    pub m: BigUint,
    /// The degree d of the polynomial.
    pub degree: usize,
}

/// Select a polynomial for NFS using the base-m method.
///
/// Given n and a target degree d, compute m = floor(n^(1/(d+1))) and express
/// n in base m: n = c_d * m^d + c_{d-1} * m^{d-1} + ... + c_0.
/// Then f(x) = c_d * x^d + ... + c_0, with f(m) = n = 0 (mod n).
pub fn select_polynomial(n: &BigUint, degree: usize) -> NfsPolynomial {
    let m = nth_root(n, (degree + 1) as u32);

    // Express n in base m: n = c_d * m^d + ... + c_0
    let mut coefficients = Vec::new();
    let mut remaining = n.clone();

    if m.is_zero() || m == BigUint::one() {
        // Degenerate case: m is 0 or 1, just store n as the constant coeff
        coefficients.push(remaining);
        for _ in 1..=degree {
            coefficients.push(BigUint::zero());
        }
    } else {
        // Standard base-m decomposition
        loop {
            let (quot, rem) = remaining.div_rem(&m);
            coefficients.push(rem);
            remaining = quot;
            if remaining.is_zero() {
                break;
            }
        }
        // Pad to exactly degree+1 coefficients
        while coefficients.len() <= degree {
            coefficients.push(BigUint::zero());
        }
    }

    NfsPolynomial {
        coefficients,
        m,
        degree,
    }
}

/// Choose the optimal polynomial degree based on the bit size of n.
///
/// Heuristic based on standard NFS recommendations:
/// - 40-60 bits: degree 3
/// - 60-90 bits: degree 3 (could be 4 but 3 is simpler)
/// - 90-120 bits: degree 4
/// - 120+ bits: degree 5
pub fn choose_degree(n: &BigUint) -> usize {
    let bits = n.bits();
    if bits <= 90 {
        3
    } else if bits <= 120 {
        4
    } else {
        5
    }
}

/// Evaluate the polynomial f(x) at a given integer x (may be negative).
///
/// f(x) = c_0 + c_1 * x + c_2 * x^2 + ... + c_d * x^d
/// using Horner's method. Returns |f(x)| as BigUint.
pub fn eval_polynomial(poly: &NfsPolynomial, x: i64) -> BigUint {
    let x_abs = BigUint::from(x.unsigned_abs());
    let x_neg = x < 0;

    if poly.coefficients.is_empty() {
        return BigUint::zero();
    }

    horner_eval_abs(&poly.coefficients, &x_abs, x_neg)
}

/// Horner's evaluation of polynomial with BigUint coefficients at signed x.
/// Returns |f(x)|.
///
/// coefficients are [c_0, c_1, ..., c_d] (low-degree first).
fn horner_eval_abs(coeffs: &[BigUint], x_abs: &BigUint, x_neg: bool) -> BigUint {
    if coeffs.is_empty() {
        return BigUint::zero();
    }

    let d = coeffs.len() - 1;
    let mut mag = coeffs[d].clone();
    let mut neg = false;

    for i in (0..d).rev() {
        // Multiply by x
        mag *= x_abs;
        neg ^= x_neg;

        // Add c_i (non-negative)
        if neg {
            if coeffs[i] >= mag {
                mag = &coeffs[i] - &mag;
                neg = false;
            } else {
                mag = &mag - &coeffs[i];
            }
        } else {
            mag = &mag + &coeffs[i];
        }
    }

    mag
}

/// Compute floor(n^(1/k)) -- the integer k-th root of n.
pub fn nth_root(n: &BigUint, k: u32) -> BigUint {
    if n.is_zero() || k == 0 {
        return BigUint::zero();
    }
    if k == 1 {
        return n.clone();
    }
    if k == 2 {
        return isqrt(n);
    }

    let k_big = BigUint::from(k);
    let k_minus_1 = BigUint::from(k - 1);

    let bits = n.bits();
    let init_bits = (bits / k as u64).max(1);
    let mut x = BigUint::one() << init_bits as usize;

    loop {
        let x_pow = pow_biguint(&x, k - 1);
        if x_pow.is_zero() {
            return BigUint::one();
        }
        let x_new = (&k_minus_1 * &x + n / &x_pow) / &k_big;

        if x_new >= x {
            break;
        }
        x = x_new;
    }

    while pow_biguint(&(&x + BigUint::one()), k) <= *n {
        x += BigUint::one();
    }
    while &pow_biguint(&x, k) > n {
        if x.is_zero() {
            break;
        }
        x -= BigUint::one();
    }

    x
}

/// Compute the integer square root (floor) of a BigUint.
pub fn isqrt(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    let mut x = n.clone();
    let mut y = (&x + BigUint::one()) >> 1u32;
    while y < x {
        x = y.clone();
        y = (&x + n / &x) >> 1u32;
    }
    x
}

/// Compute base^exp for BigUint.
pub fn pow_biguint(base: &BigUint, exp: u32) -> BigUint {
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

    #[test]
    fn test_nth_root() {
        assert_eq!(nth_root(&BigUint::from(1024u64), 3), BigUint::from(10u64));
        assert_eq!(nth_root(&BigUint::from(1000u64), 3), BigUint::from(10u64));
        assert_eq!(nth_root(&BigUint::from(999u64), 3), BigUint::from(9u64));
        assert_eq!(nth_root(&BigUint::from(8u64), 3), BigUint::from(2u64));
        assert_eq!(nth_root(&BigUint::from(27u64), 3), BigUint::from(3u64));
        assert_eq!(nth_root(&BigUint::from(16u64), 4), BigUint::from(2u64));
        assert_eq!(nth_root(&BigUint::from(100u64), 2), BigUint::from(10u64));
    }

    #[test]
    fn test_polynomial_selection() {
        let n = BigUint::from(15347u64);
        let poly = select_polynomial(&n, 3);

        assert_eq!(poly.degree, 3);

        // Verify f(m) = n
        let mut reconstructed = BigUint::zero();
        let mut m_power = BigUint::one();
        for coeff in &poly.coefficients {
            reconstructed += coeff * &m_power;
            m_power *= &poly.m;
        }
        assert_eq!(reconstructed, n, "f(m) must equal n");
    }

    #[test]
    fn test_eval_polynomial_simple() {
        let poly = NfsPolynomial {
            coefficients: vec![
                BigUint::from(5u64),
                BigUint::from(3u64),
                BigUint::from(2u64),
            ],
            m: BigUint::from(10u64),
            degree: 2,
        };
        assert_eq!(eval_polynomial(&poly, 1), BigUint::from(10u64));
        assert_eq!(eval_polynomial(&poly, 2), BigUint::from(19u64));
        assert_eq!(eval_polynomial(&poly, 0), BigUint::from(5u64));
        assert_eq!(eval_polynomial(&poly, -1), BigUint::from(4u64));
    }

    #[test]
    fn test_choose_degree() {
        assert_eq!(choose_degree(&BigUint::from(1u64 << 40)), 3);
        assert_eq!(choose_degree(&BigUint::from(1u64 << 60)), 3);
        // For larger numbers, need BigUint
        let big = BigUint::from(1u64) << 100;
        assert_eq!(choose_degree(&big), 4);
        let very_big = BigUint::from(1u64) << 130;
        assert_eq!(choose_degree(&very_big), 5);
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(&BigUint::from(0u32)), BigUint::zero());
        assert_eq!(isqrt(&BigUint::from(1u32)), BigUint::one());
        assert_eq!(isqrt(&BigUint::from(4u32)), BigUint::from(2u32));
        assert_eq!(isqrt(&BigUint::from(10u32)), BigUint::from(3u32));
        assert_eq!(isqrt(&BigUint::from(100u32)), BigUint::from(10u32));
    }
}
