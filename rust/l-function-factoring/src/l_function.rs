//! L-function evaluation.
//!
//! Computes Dirichlet L-functions L(s, χ) = Σ_{n=1}^∞ χ(n)/n^s
//! via partial sums, Euler products, and the functional equation.

use crate::characters::DirichletChar;
use crate::complex::{self, Complex};
use std::f64::consts::PI;

/// Compute L(s, χ) via partial sum: L(s, χ) ≈ Σ_{n=1}^{B} χ(n)/n^s.
///
/// For s > 1 and non-principal χ, the error is O(B^{1-s}).
/// For s = 1 and non-principal χ, the error is O(1/B) (conditional convergence).
pub fn l_function_partial(chi: &DirichletChar, s: f64, num_terms: u64) -> Complex {
    let mut sum = complex::ZERO;

    for n in 1..=num_terms {
        let chi_n = chi.eval(n);
        if complex::cnorm_sq(chi_n) < 1e-20 {
            continue;
        }
        let n_s = (n as f64).powf(s);
        let term = complex::cscale(1.0 / n_s, chi_n);
        sum = complex::cadd(sum, term);
    }

    sum
}

/// Compute L(1, χ) for a real character χ.
///
/// Uses partial sums with Richardson extrapolation for better convergence.
/// Returns only the real part since χ is real-valued.
pub fn l_function_at_1(chi: &DirichletChar, num_terms: u64) -> f64 {
    if chi.is_principal {
        // L(1, χ_0) diverges for the principal character
        return f64::INFINITY;
    }

    // Use two partial sums of different lengths for extrapolation
    let s1 = l_function_partial(chi, 1.0, num_terms);
    let s2 = l_function_partial(chi, 1.0, num_terms * 2);

    // Richardson extrapolation: better estimate is 2*S(2B) - S(B)
    // (for O(1/B) error terms)
    let extrapolated = 2.0 * s2.0 - s1.0;

    // Use the longer sum if extrapolation gives a weird result
    if extrapolated.is_finite() {
        extrapolated
    } else {
        s2.0
    }
}

/// Compute L(s, χ) via the Euler product: L(s, χ) = Π_p (1 - χ(p) p^{-s})^{-1}.
///
/// The product is over all primes p, truncated to the provided list of primes.
/// Converges faster than partial sums for s > 1.
pub fn l_function_euler_product(chi: &DirichletChar, s: f64, primes: &[u64]) -> Complex {
    let mut product = complex::ONE;

    for &p in primes {
        let chi_p = chi.eval(p);
        if complex::cnorm_sq(chi_p) < 1e-20 {
            // χ(p) = 0, so this Euler factor is 1
            continue;
        }

        let p_s = (p as f64).powf(s);
        let term = complex::csub(complex::ONE, complex::cscale(1.0 / p_s, chi_p));

        // Invert: (1 - χ(p)p^{-s})^{-1}
        let inv = complex::cdiv(complex::ONE, term);
        product = complex::cmul(product, inv);
    }

    product
}

/// Compute L(s, χ) using the functional equation for acceleration.
///
/// The functional equation relates L(s, χ) to L(1-s, χ̄):
/// L(s, χ) = ε(χ) * (f/π)^{(1-2s)/2} * Γ((1-s+a)/2) / Γ((s+a)/2) * L(1-s, χ̄)
///
/// where a = 0 if χ(-1) = 1, a = 1 if χ(-1) = -1, f is the conductor.
///
/// This is useful when s < 1/2 (compute the RHS where 1-s > 1/2 converges faster).
pub fn l_function_functional_equation(
    chi: &DirichletChar,
    s: f64,
    conductor: u64,
    num_terms: u64,
) -> Complex {
    // Determine a: the parity of the character
    let chi_minus_1 = chi.eval(chi.modulus - 1);
    let a = if (chi_minus_1.0 - 1.0).abs() < 1e-8 {
        0.0 // even character: χ(-1) = 1
    } else {
        1.0 // odd character: χ(-1) = -1
    };

    // Compute L(1-s, χ̄) which converges better when s < 1/2
    let s_dual = 1.0 - s;
    let chi_bar = conjugate_character(chi);
    let l_dual = l_function_partial(&chi_bar, s_dual, num_terms);

    // Root number ε(χ) - for simplicity we compute it from τ(χ)
    let tau = crate::gauss_sums::gauss_sum(chi);
    let f = conductor as f64;
    let epsilon = complex::cscale(1.0 / f.sqrt(), tau);

    // Gamma ratio: Γ((1-s+a)/2) / Γ((s+a)/2)
    let gamma_ratio = gamma_ratio_approx(s, a);

    // Power factor: (f/π)^{(1-2s)/2}
    let power_factor = (f / PI).powf((1.0 - 2.0 * s) / 2.0);

    // L(s, χ) = ε(χ) * power_factor * gamma_ratio * L(1-s, χ̄)
    let result = complex::cmul(epsilon, complex::cscale(power_factor * gamma_ratio, l_dual));
    result
}

/// Create the conjugate character χ̄.
fn conjugate_character(chi: &DirichletChar) -> DirichletChar {
    DirichletChar {
        modulus: chi.modulus,
        values: chi.values.iter().map(|&v| complex::conj(v)).collect(),
        is_principal: chi.is_principal,
    }
}

/// Approximate the ratio Γ((1-s+a)/2) / Γ((s+a)/2) using Stirling's approximation.
///
/// For our purposes, we only need moderate accuracy.
fn gamma_ratio_approx(s: f64, a: f64) -> f64 {
    let x1 = (1.0 - s + a) / 2.0;
    let x2 = (s + a) / 2.0;

    // Use the gamma function approximation via Lanczos
    let g1 = lanczos_gamma(x1);
    let g2 = lanczos_gamma(x2);

    if g2.abs() < 1e-100 {
        return 0.0;
    }
    g1 / g2
}

/// Lanczos approximation for the gamma function Γ(x) for positive real x.
fn lanczos_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        // Use reflection formula
        let sinpix = (PI * x).sin();
        if sinpix.abs() < 1e-100 {
            return f64::INFINITY;
        }
        return PI / (sinpix * lanczos_gamma(1.0 - x));
    }

    // Lanczos coefficients (g=7, n=9)
    let p: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let g = 7.0;
    let x = x - 1.0;

    let mut a = p[0];
    let t = x + g + 0.5;

    for i in 1..9 {
        a += p[i] / (x + i as f64);
    }

    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

/// Generate a list of primes up to a bound using the sieve of Eratosthenes.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 {
        return vec![];
    }

    let bound = bound as usize;
    let mut is_prime = vec![true; bound + 1];
    is_prime[0] = false;
    if bound >= 1 {
        is_prime[1] = false;
    }

    let mut p = 2;
    while p * p <= bound {
        if is_prime[p] {
            let mut multiple = p * p;
            while multiple <= bound {
                is_prime[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }

    is_prime
        .iter()
        .enumerate()
        .filter_map(|(i, &is_p)| if is_p { Some(i as u64) } else { None })
        .collect()
}

/// Result of L-function evaluation at multiple points.
#[derive(Debug, Clone)]
pub struct LFunctionProfile {
    /// The modulus of the character.
    pub modulus: u64,
    /// Character index (position in the enumeration).
    pub char_index: usize,
    /// L(2, χ) value.
    pub l_at_2: Complex,
    /// L(3, χ) value.
    pub l_at_3: Complex,
    /// L(4, χ) value.
    pub l_at_4: Complex,
    /// L(1, χ) approximation (for real characters).
    pub l_at_1: Option<f64>,
}

/// Compute L-function profiles for all characters mod N.
///
/// This evaluates L(s, χ) at s = 2, 3, 4 and optionally s = 1,
/// which can be used for factoring via L-function decomposition.
pub fn compute_l_profiles(n: u64, num_terms: u64) -> Vec<LFunctionProfile> {
    let characters = crate::characters::enumerate_characters(n);
    let mut profiles = Vec::new();

    for (idx, chi) in characters.iter().enumerate() {
        if chi.is_principal {
            continue;
        }

        let l_at_2 = l_function_partial(chi, 2.0, num_terms);
        let l_at_3 = l_function_partial(chi, 3.0, num_terms);
        let l_at_4 = l_function_partial(chi, 4.0, num_terms);

        let l_at_1 = if chi.is_real() {
            Some(l_function_at_1(chi, num_terms))
        } else {
            None
        };

        profiles.push(LFunctionProfile {
            modulus: n,
            char_index: idx,
            l_at_2,
            l_at_3,
            l_at_4,
            l_at_1,
        });
    }

    profiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characters;

    #[test]
    fn test_l_function_partial_convergence() {
        // L(2, χ_0) for the principal character mod 1 should approximate π²/6 ≈ 1.6449
        // But we use non-principal characters instead.
        // For the non-trivial character mod 4 (χ(-1) = -1):
        // L(1, χ) = π/4 ≈ 0.7854
        let chi_4 = characters::kronecker_character(-4, 4);
        let l_1_short = l_function_partial(&chi_4, 1.0, 100);
        let l_1_long = l_function_partial(&chi_4, 1.0, 10000);

        // Should converge toward π/4
        let expected = PI / 4.0;
        assert!(
            (l_1_long.0 - expected).abs() < 0.01,
            "L(1, chi(-4)) ≈ {}, expected ≈ {}",
            l_1_long.0,
            expected
        );
        // Longer sum should be more accurate
        assert!(
            (l_1_long.0 - expected).abs() < (l_1_short.0 - expected).abs(),
            "Longer partial sum should be more accurate"
        );
    }

    #[test]
    fn test_l_function_at_2() {
        // For the non-trivial character mod 4:
        // L(2, χ_{-4}) = Catalan's constant G ≈ 0.9159655941
        let chi_4 = characters::kronecker_character(-4, 4);
        let l_2 = l_function_partial(&chi_4, 2.0, 10000);

        let catalan = 0.9159655941;
        assert!(
            (l_2.0 - catalan).abs() < 0.001,
            "L(2, chi(-4)) ≈ {}, expected ≈ {}",
            l_2.0,
            catalan
        );
    }

    #[test]
    fn test_euler_product_vs_partial_sum() {
        let chi_4 = characters::kronecker_character(-4, 100);
        let primes = sieve_primes(1000);

        let l_euler = l_function_euler_product(&chi_4, 2.0, &primes);
        let l_partial = l_function_partial(&chi_4, 2.0, 10000);

        // Both should give approximately the same value
        assert!(
            (l_euler.0 - l_partial.0).abs() < 0.01,
            "Euler product L(2) = {}, partial sum L(2) = {}",
            l_euler.0,
            l_partial.0
        );
    }

    #[test]
    fn test_l_function_at_1_extrapolated() {
        let chi_4 = characters::kronecker_character(-4, 4);
        let l_1 = l_function_at_1(&chi_4, 5000);

        let expected = PI / 4.0;
        assert!(
            (l_1 - expected).abs() < 0.01,
            "L(1, chi(-4)) via extrapolation ≈ {}, expected ≈ {}",
            l_1,
            expected
        );
    }

    #[test]
    fn test_sieve_primes() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_lanczos_gamma() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6
        assert!((lanczos_gamma(1.0) - 1.0).abs() < 1e-8);
        assert!((lanczos_gamma(2.0) - 1.0).abs() < 1e-8);
        assert!((lanczos_gamma(3.0) - 2.0).abs() < 1e-8);
        assert!((lanczos_gamma(4.0) - 6.0).abs() < 1e-8);
        // Γ(1/2) = √π
        assert!((lanczos_gamma(0.5) - PI.sqrt()).abs() < 1e-8);
    }
}
