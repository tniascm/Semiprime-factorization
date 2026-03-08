pub mod alpha;
pub mod dickman;
pub mod rotation;
pub use alpha::murphy_alpha;

use crate::arith::nth_root;
use crate::types::PolynomialPair;
use rug::ops::Pow;
use rug::Integer;

/// Choose polynomial degree based on digit count of N.
pub fn choose_degree(digits: u32) -> u32 {
    if digits <= 50 {
        3
    } else if digits <= 100 {
        4
    } else {
        5
    }
}

/// Base-m polynomial selection.
///
/// Given N and degree d, compute m = floor(N^(1/d)), then express N in base m:
///   N = c_d * m^d + c_{d-1} * m^{d-1} + ... + c_0
/// This gives f(x) = c_d * x^d + ... + c_0 with f(m) = N ≡ 0 (mod N).
/// Using m = floor(N^{1/d}) ensures the leading coefficient c_d = 1 (monic)
/// for all N > 2^d, which is required for correct algebraic arithmetic in Z[α]/(f).
/// The rational polynomial is g(x) = x - m, so g(m) = 0.
/// We store: g0 = -m, g1 = 1.
pub fn select_base_m(n: &Integer, degree: u32) -> PolynomialPair {
    select_base_m_variant(n, degree, 0)
}

/// Base-m polynomial selection with variant index.
///
/// Variant 0: standard m = floor(N^{1/d}) (adjusted up for monic).
/// Variant k > 0: try m = standard_m - k, using a different number field.
/// This is critical when the standard polynomial produces a degenerate number
/// field where the algebraic square root always gives trivial gcd.
pub fn select_base_m_variant(n: &Integer, degree: u32, variant: u32) -> PolynomialPair {
    let d = degree as usize;
    let mut m = nth_root(n, degree);
    // Ensure monic: if leading coefficient > 1, increment m until it becomes 1
    loop {
        let m_pow_d = Integer::from(m.clone().pow(degree));
        let lead = Integer::from(n / &m_pow_d);
        if lead <= 1 {
            break;
        }
        m += 1;
    }

    // For variant > 0, decrease m to get a different polynomial
    if variant > 0 {
        m -= Integer::from(variant);
        // Verify still monic (leading coefficient = 1)
        let m_pow_d = Integer::from(m.clone().pow(degree));
        let lead = Integer::from(n / &m_pow_d);
        if lead != 1 {
            // Non-monic with this m, fall back to standard
            return select_base_m_variant(n, degree, 0);
        }
    }

    // Express N in base m
    let mut coeffs = Vec::with_capacity(d + 1);
    let mut remaining = n.clone();

    if m <= 1 {
        // Degenerate: just use N as constant coefficient
        coeffs.push(remaining);
        for _ in 1..=d {
            coeffs.push(Integer::from(0));
        }
    } else {
        loop {
            let (quot, rem) = remaining.div_rem_euc(m.clone());
            coeffs.push(rem);
            remaining = quot;
            if remaining == 0 {
                break;
            }
        }
        // Pad to exactly d+1 coefficients
        while coeffs.len() <= d {
            coeffs.push(Integer::from(0));
        }
    }

    // Verify: f(m) should equal N
    debug_assert!({
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        val == *n
    });

    // Verify monic: leading coefficient must be 1
    debug_assert_eq!(coeffs[d], Integer::from(1), "Polynomial must be monic");

    let neg_m = Integer::from(-&m);
    PolynomialPair::new(&coeffs, &neg_m, &Integer::from(1), &m, n)
}

/// Generate a polynomial pair with given leading coefficient `ad`.
///
/// Computes m = floor((N/ad)^(1/d)), then constructs f(x) = ad*x^d + ... + c_0
/// via base-m expansion such that f(m) = N.
pub fn select_polynomial_with_ad(n: &Integer, degree: u32, ad: u64) -> Option<PolynomialPair> {
    let d = degree as usize;
    let ad_int = Integer::from(ad);

    // m = floor((N / ad)^(1/d))
    let n_over_ad = Integer::from(n / &ad_int);
    let m = nth_root(&n_over_ad, degree);

    if m < 2 {
        return None;
    }

    // Try m and m-1 to find a valid base-m expansion
    for m_candidate in [m.clone(), Integer::from(&m - 1)] {
        if m_candidate < 2 {
            continue;
        }
        let m_pow_d = Integer::from(m_candidate.clone().pow(degree));
        let remaining = Integer::from(n - &ad_int * &m_pow_d);
        if remaining < 0 {
            continue;
        }

        if let Some(poly) = build_poly_from_remainder(n, &m_candidate, &ad_int, d, &remaining) {
            return Some(poly);
        }
    }

    None
}

fn build_poly_from_remainder(
    n: &Integer,
    m: &Integer,
    ad: &Integer,
    d: usize,
    remainder: &Integer,
) -> Option<PolynomialPair> {
    let mut coeffs = Vec::with_capacity(d + 1);
    let mut rem = remainder.clone();

    // Extract d coefficients c_0 through c_{d-1} via base-m expansion
    for _ in 0..d {
        let (quot, r) = rem.div_rem_euc(m.clone());
        coeffs.push(r);
        rem = quot;
    }

    // rem should be 0 if expansion is exact
    if rem != 0 {
        return None;
    }

    coeffs.push(ad.clone());

    // Verify: f(m) = N
    debug_assert!({
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= m;
        }
        val == *n
    });

    let neg_m = Integer::from(-m);
    Some(PolynomialPair::new(&coeffs, &neg_m, &Integer::from(1), m, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_base_m_selection_small() {
        // 8051 = 83 * 97
        let n = Integer::from(8051);
        let poly = select_base_m(&n, 3);
        // Verify f(m) ≡ 0 (mod N)
        let f = poly.f_coeffs();
        let m = poly.m();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &f {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val % &n, 0, "f(m) must be 0 mod N");
    }

    #[test]
    fn test_base_m_selection_g_linear() {
        let n = Integer::from(15347);
        let poly = select_base_m(&n, 3);
        // g(m) = g0 + g1 * m should be divisible by N
        let g0 = poly.g0();
        let g1 = poly.g1();
        let m = poly.m();
        let gm = Integer::from(&g0 + &g1 * &m);
        assert_eq!(gm % &n, 0, "g(m) must be 0 mod N");
    }

    #[test]
    fn test_choose_degree() {
        assert_eq!(choose_degree(50), 3);
        assert_eq!(choose_degree(100), 4);
        assert_eq!(choose_degree(150), 5);
        assert_eq!(choose_degree(200), 5);
        assert_eq!(choose_degree(300), 5);
    }

    #[test]
    fn test_base_m_60_digit() {
        // A 60-digit semiprime
        let n: Integer = "523022617466601111760007224100074291200000259"
            .parse()
            .unwrap();
        let d = choose_degree(n.significant_bits() as u32 * 3 / 10);
        let poly = select_base_m(&n, d);
        let f = poly.f_coeffs();
        let m = poly.m();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &f {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val % &n, 0);
    }

    #[test]
    fn test_select_polynomial_with_ad_monic() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        // ad=1 should produce a valid polynomial
        let poly = select_polynomial_with_ad(&n, 3, 1).unwrap();
        assert_eq!(poly.degree, 3);
        // Verify f(m) = N
        let m = poly.m();
        let coeffs = poly.f_coeffs();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val, n, "f(m) must equal N");
    }

    #[test]
    fn test_nonmonic_polynomial_ad_60() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let poly = select_polynomial_with_ad(&n, 3, 60).unwrap();
        assert_eq!(poly.degree, 3);
        // Leading coefficient should be 60
        let lead = poly.f_coeffs().last().unwrap().clone();
        assert_eq!(lead, Integer::from(60));
        // Verify f(m) = N
        let m = poly.m();
        let coeffs = poly.f_coeffs();
        let mut val = Integer::from(0);
        let mut m_pow = Integer::from(1);
        for c in &coeffs {
            val += c * &m_pow;
            m_pow *= &m;
        }
        assert_eq!(val, n, "f(m) must equal N for ad=60");
    }

    #[test]
    fn test_nonmonic_sweep_produces_diverse_polys() {
        let n = Integer::from_str_radix("684217602914977371691118975023", 10).unwrap();
        let mut valid = 0;
        for ad in (20..=200).step_by(20) {
            if select_polynomial_with_ad(&n, 3, ad).is_some() {
                valid += 1;
            }
        }
        assert!(
            valid >= 5,
            "At least half of ad values should produce valid polys, got {}",
            valid
        );
    }
}
