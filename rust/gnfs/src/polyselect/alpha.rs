use crate::arith::sieve_primes;

/// Compute Murphy's alpha for polynomial f over primes up to `bound`.
///
/// Alpha measures the "root property": how often small primes divide f(x,y).
/// More negative alpha = better polynomial (more smooth values).
///
/// Formula: alpha(f) = sum_{p <= B} (1 - q_p * p/(p+1)) * log(p)/(p-1)
/// where q_p = number of roots of f mod p (affine + projective).
///
/// Coefficients are stored low-degree first: `[c0, c1, ..., cd]` represents
/// `f(x) = c0 + c1*x + ... + cd*x^d`.
pub fn murphy_alpha(f_coeffs: &[i64], bound: u64) -> f64 {
    if f_coeffs.is_empty() {
        return 0.0;
    }
    let primes = sieve_primes(bound);
    let mut alpha = 0.0;
    for &p in &primes {
        let pf = p as f64;
        let q_p = count_roots_mod_p(f_coeffs, p) as f64;
        alpha += (1.0 - q_p * pf / (pf + 1.0)) * pf.ln() / (pf - 1.0);
    }
    alpha
}

/// Count the number of roots of f(x) mod p (affine + projective).
///
/// Affine roots: evaluate f(x) mod p for x in 0..p.
/// Projective root: if p divides the leading coefficient, add 1 (the point at infinity).
///
/// Coefficients are stored low-degree first: `[c0, c1, ..., cd]`.
fn count_roots_mod_p(f_coeffs: &[i64], p: u64) -> u64 {
    let p_i128 = p as i128;
    let mut count = 0u64;

    // Affine roots: evaluate f(x) mod p for x in 0..p using Horner's method
    for x in 0..p {
        let x_i128 = x as i128;
        // Horner's: iterate from highest-degree coefficient down
        let mut val: i128 = 0;
        for &c in f_coeffs.iter().rev() {
            val = (val * x_i128 + c as i128).rem_euclid(p_i128);
        }
        if val == 0 {
            count += 1;
        }
    }

    // Projective root: check if p divides leading coefficient
    let d = f_coeffs.len() - 1;
    if d > 0 {
        let lead = f_coeffs[d] as i128;
        if lead.rem_euclid(p_i128) == 0 {
            count += 1;
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_linear_polynomial() {
        // f(x) = x - 1 has exactly 1 affine root mod every prime (root = 1),
        // no projective root. q_p = 1 for all p.
        // Contribution per prime: (1 - p/(p+1)) * ln(p)/(p-1) = ln(p)/((p+1)(p-1)) > 0
        // So alpha should be positive (1 root is the "expected" baseline).
        let alpha = murphy_alpha(&[-1, 1], 200);
        assert!(alpha > 0.0, "Linear poly with 1 root/prime should have positive alpha: {}", alpha);
        // Should be a modest positive value
        assert!(alpha < 2.0, "Linear poly alpha should be moderate: {}", alpha);
    }

    #[test]
    fn test_alpha_irreducible_poly() {
        // f(x) = x^2 + 1 has 0 roots mod p when p=3 mod 4, 2 roots when p=1 mod 4
        let alpha = murphy_alpha(&[1, 0, 1], 200);
        assert!(alpha.abs() < 2.0, "x^2+1 alpha should be moderate: {}", alpha);
    }

    #[test]
    fn test_alpha_many_roots_is_better() {
        // f(x) = x*(x-1)*(x-2) = x^3 - 3x^2 + 2x has 3 roots mod most primes
        let alpha_good = murphy_alpha(&[0, 2, -3, 1], 200);
        // f(x) = x^3 + x + 1 has fewer roots mod most primes
        let alpha_bad = murphy_alpha(&[1, 1, 0, 1], 200);
        assert!(
            alpha_good < alpha_bad,
            "Poly with more roots should have more negative alpha: {} vs {}",
            alpha_good, alpha_bad
        );
    }

    #[test]
    fn test_count_roots_mod_p() {
        // x^2 - 1 = (x-1)(x+1) has 2 roots mod 5: x=1, x=4
        assert_eq!(count_roots_mod_p(&[-1, 0, 1], 5), 2);
        // x^2 + 1 has 0 roots mod 3 (QNR)
        assert_eq!(count_roots_mod_p(&[1, 0, 1], 3), 0);
        // x^2 + 1 has 2 roots mod 5: x=2, x=3
        assert_eq!(count_roots_mod_p(&[1, 0, 1], 5), 2);
        // x has 1 root mod any prime: x=0
        assert_eq!(count_roots_mod_p(&[0, 1], 7), 1);
    }

    #[test]
    fn test_projective_root() {
        // f(x) = 2*x^2 + x + 1: leading coeff 2, divisible by p=2
        // Affine: x=0: f(0)=1 mod 2 = 1 (not root). x=1: f(1)=4 mod 2 = 0 (root).
        // Plus projective root since 2 | 2. Total = 2.
        let roots = count_roots_mod_p(&[1, 1, 2], 2);
        assert_eq!(roots, 2);
    }

    #[test]
    fn test_alpha_empty_coeffs() {
        assert_eq!(murphy_alpha(&[], 200), 0.0);
    }

    #[test]
    fn test_count_roots_constant_poly() {
        // f(x) = 3, a constant; no affine roots mod 5, no projective root (degree 0)
        assert_eq!(count_roots_mod_p(&[3], 5), 0);
        // f(x) = 0, a constant zero; every x is a root mod 5
        assert_eq!(count_roots_mod_p(&[0], 5), 5);
    }
}
