//! Class number formula and related computations.
//!
//! For an imaginary quadratic field Q(√D) with D < 0, the class number h(D) counts
//! the number of equivalence classes of binary quadratic forms of discriminant D.
//!
//! The class number formula gives:
//!   h(D) = (w * √|D|) / (2π) * L(1, χ_D)
//! where w is the number of roots of unity (w=2 for |D|>4, w=4 for D=-4, w=6 for D=-3)
//! and χ_D = (D|·) is the Kronecker symbol.

use crate::characters;
use crate::l_function;
use std::f64::consts::PI;

/// Result of class number analysis.
#[derive(Debug, Clone)]
pub struct ClassNumberAnalysis {
    /// The discriminant D.
    pub discriminant: i64,
    /// Class number computed via the formula h = w√|D|/(2π) * L(1, χ_D).
    pub h_formula: f64,
    /// Class number computed by counting reduced forms (for small |D|).
    pub h_direct: Option<u64>,
    /// The L-function value L(1, χ_D).
    pub l_value: f64,
    /// The number of roots of unity w.
    pub w: u64,
}

/// Compute the class number h(D) via the analytic class number formula.
///
/// h(D) = (w * √|D|) / (2π) * L(1, χ_D)
///
/// where D is a negative fundamental discriminant.
pub fn class_number_formula(d: i64) -> f64 {
    assert!(d < 0, "Discriminant must be negative");

    let abs_d = d.unsigned_abs();
    let w = roots_of_unity(d);

    // Create the Kronecker character χ_D
    // We evaluate L(1, χ_D) as Σ (D|n)/n
    let chi = characters::kronecker_character(d, abs_d);
    let l_value = l_function::l_function_at_1(&chi, 50000);

    (w as f64) * (abs_d as f64).sqrt() / (2.0 * PI) * l_value
}

/// Compute L(1, χ_D) directly using the Kronecker symbol.
///
/// L(1, χ_D) = Σ_{n=1}^∞ (D|n)/n
///
/// This is computed without the DirichletChar machinery for efficiency.
pub fn l_value_kronecker(d: i64, num_terms: u64) -> f64 {
    let mut sum = 0.0;

    for n in 1..=num_terms {
        let ks = characters::kronecker_symbol(d, n);
        if ks != 0 {
            sum += (ks as f64) / (n as f64);
        }
    }

    sum
}

/// Compute L(1, χ_D) with Richardson extrapolation for better convergence.
pub fn l_value_kronecker_extrapolated(d: i64, num_terms: u64) -> f64 {
    let s1 = l_value_kronecker(d, num_terms);
    let s2 = l_value_kronecker(d, num_terms * 2);

    // Richardson extrapolation for O(1/B) error
    2.0 * s2 - s1
}

/// Compute the class number h(D) via L(1, χ_D) using direct Kronecker computation.
pub fn class_number_via_kronecker(d: i64, num_terms: u64) -> f64 {
    assert!(d < 0, "Discriminant must be negative");

    let abs_d = d.unsigned_abs();
    let w = roots_of_unity(d);
    let l_value = l_value_kronecker_extrapolated(d, num_terms);

    (w as f64) * (abs_d as f64).sqrt() / (2.0 * PI) * l_value
}

/// Compute the class number by directly counting reduced binary quadratic forms.
///
/// A form ax² + bxy + cy² of discriminant D = b² - 4ac is reduced if:
///   -a < b ≤ a < c, or 0 ≤ b ≤ a = c.
///
/// This is only practical for small |D|.
pub fn class_number_direct(d: i64) -> u64 {
    assert!(d < 0, "Discriminant must be negative");

    let abs_d = (-d) as u64;
    let mut count = 0u64;

    // b² - 4ac = D, so b² + 4ac = |D|, so b² ≡ D (mod 4)
    // Since D < 0: 4ac = b² - D = b² + |D|

    // b ranges: b² < |D| + 4ac → we need b² ≤ |D|/3 (from a ≤ c condition)
    // More precisely, for reduced forms: |b| ≤ a, and a² ≤ |D|/3
    let a_max = ((abs_d as f64 / 3.0).sqrt()) as u64 + 1;

    for a in 1..=a_max {
        // b ranges from -a to a, but b ≡ D (mod 2)
        let b_start = -(a as i64);
        let b_end = a as i64;

        for b in b_start..=b_end {
            // Check b ≡ D (mod 2)
            let b_mod2 = ((b % 2) + 2) % 2;
            let d_mod2 = ((d % 2) + 2) % 2;
            if b_mod2 != d_mod2 {
                continue;
            }

            // c = (b² - D) / (4a)
            let numerator = (b as i64) * (b as i64) - d;
            if numerator <= 0 {
                continue;
            }
            let numerator = numerator as u64;
            if numerator % (4 * a) != 0 {
                continue;
            }
            let c = numerator / (4 * a);

            if c == 0 {
                continue;
            }

            // Check reduced form conditions
            let abs_b = b.unsigned_abs();

            // Standard reduced form: -a < b ≤ a ≤ c, with b ≥ 0 if a = c
            if a < c {
                if abs_b <= a {
                    // -a < b ≤ a is our condition for a < c
                    if b > -(a as i64) {
                        count += 1;
                    }
                }
            } else if a == c {
                // 0 ≤ b ≤ a
                if b >= 0 && (b as u64) <= a {
                    count += 1;
                }
            }
        }
    }

    count
}

/// Number of roots of unity in Q(√D).
fn roots_of_unity(d: i64) -> u64 {
    match d {
        -3 => 6,
        -4 => 4,
        _ => 2,
    }
}

/// Perform a class number analysis comparing formula vs direct computation.
pub fn class_number_analysis(d: i64) -> ClassNumberAnalysis {
    assert!(d < 0, "Discriminant must be negative");

    let abs_d = d.unsigned_abs();
    let w = roots_of_unity(d);
    let l_value = l_value_kronecker_extrapolated(d, 50000);
    let h_formula = (w as f64) * (abs_d as f64).sqrt() / (2.0 * PI) * l_value;

    let h_direct = if abs_d < 100000 {
        Some(class_number_direct(d))
    } else {
        None
    };

    ClassNumberAnalysis {
        discriminant: d,
        h_formula,
        h_direct,
        l_value,
        w,
    }
}

/// Compare class numbers for D = -4N with candidate factorizations of N.
///
/// For N = pq, we can compute h(-4N) and compare it with information derived
/// from the factorization. The class number h(-4N) relates to the class numbers
/// of sub-fields in a genus-theory way.
pub fn class_number_vs_factorization(n: u64) -> ClassNumberFactoringResult {
    let d = -(4 * n as i64);
    let h = class_number_via_kronecker(d, 50000);
    let h_rounded = h.round() as u64;

    // For each candidate factorization p * q = N (with p ≤ √N),
    // compute h(-4p) and h(-4q) and see if they are consistent
    let mut candidates = Vec::new();
    let sqrt_n = (n as f64).sqrt() as u64;

    for p in 2..=sqrt_n {
        if n % p == 0 {
            let q = n / p;
            let h_p = class_number_via_kronecker(-(4 * p as i64), 20000);
            let h_q = class_number_via_kronecker(-(4 * q as i64), 20000);

            candidates.push(ClassNumberCandidate {
                p,
                q,
                h_4p: h_p.round() as u64,
                h_4q: h_q.round() as u64,
            });
        }
    }

    ClassNumberFactoringResult {
        n,
        h_4n: h_rounded,
        h_4n_exact: h,
        candidates,
    }
}

/// Result of class number factoring analysis.
#[derive(Debug, Clone)]
pub struct ClassNumberFactoringResult {
    /// The number to factor.
    pub n: u64,
    /// Class number h(-4N) (rounded).
    pub h_4n: u64,
    /// Class number h(-4N) (exact floating point).
    pub h_4n_exact: f64,
    /// Candidate factorizations with their class numbers.
    pub candidates: Vec<ClassNumberCandidate>,
}

/// A candidate factorization with associated class numbers.
#[derive(Debug, Clone)]
pub struct ClassNumberCandidate {
    /// First factor.
    pub p: u64,
    /// Second factor.
    pub q: u64,
    /// Class number h(-4p).
    pub h_4p: u64,
    /// Class number h(-4q).
    pub h_4q: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_number_direct_small() {
        // Known class numbers for small negative discriminants:
        // h(-3) = 1, h(-4) = 1, h(-7) = 1, h(-8) = 1
        // h(-11) = 1, h(-15) = 2, h(-20) = 2, h(-23) = 3
        assert_eq!(class_number_direct(-3), 1);
        assert_eq!(class_number_direct(-4), 1);
        assert_eq!(class_number_direct(-7), 1);
        assert_eq!(class_number_direct(-8), 1);
        assert_eq!(class_number_direct(-11), 1);
        assert_eq!(class_number_direct(-15), 2);
        assert_eq!(class_number_direct(-20), 2);
        assert_eq!(class_number_direct(-23), 3);
    }

    #[test]
    fn test_class_number_formula_vs_direct() {
        // Compare formula and direct computation for several discriminants
        for &d in &[-3, -4, -7, -8, -11, -15, -20, -23, -24, -35, -40] {
            let h_direct = class_number_direct(d);
            let h_formula = class_number_via_kronecker(d, 50000);

            assert!(
                (h_formula - h_direct as f64).abs() < 0.5,
                "For D={}: h_formula={}, h_direct={}",
                d,
                h_formula,
                h_direct
            );
        }
    }

    #[test]
    fn test_l_value_kronecker_chi_minus4() {
        // L(1, χ_{-4}) = π/4 ≈ 0.7854
        let l = l_value_kronecker_extrapolated(-4, 50000);
        let expected = PI / 4.0;
        assert!(
            (l - expected).abs() < 0.01,
            "L(1, chi(-4)) = {}, expected {}",
            l,
            expected
        );
    }

    #[test]
    fn test_roots_of_unity() {
        assert_eq!(roots_of_unity(-3), 6);
        assert_eq!(roots_of_unity(-4), 4);
        assert_eq!(roots_of_unity(-7), 2);
        assert_eq!(roots_of_unity(-100), 2);
    }
}
