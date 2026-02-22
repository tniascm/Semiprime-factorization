//! Gauss sums and conductor detection.
//!
//! The Gauss sum τ(χ) = Σ_{a=1}^{N} χ(a) * e^{2πia/N} encodes information
//! about the conductor of a character. For primitive characters, |τ(χ)|² = N.
//! For imprimitive characters with conductor f, |τ(χ)|² relates to f.

use crate::characters::DirichletChar;
use crate::complex::{self, Complex};
use num_integer::Integer;
use std::f64::consts::PI;

/// Compute the Gauss sum τ(χ) = Σ_{a=0}^{N-1} χ(a) * e^{2πia/N}.
pub fn gauss_sum(chi: &DirichletChar) -> Complex {
    let n = chi.modulus;
    let mut sum = complex::ZERO;

    for a in 0..n {
        let chi_a = chi.eval(a);
        if complex::cnorm_sq(chi_a) < 1e-20 {
            continue;
        }
        let angle = 2.0 * PI * (a as f64) / (n as f64);
        let exp = complex::cis(angle);
        sum = complex::cadd(sum, complex::cmul(chi_a, exp));
    }

    sum
}

/// Compute |τ(χ)|² = the squared magnitude of the Gauss sum.
pub fn gauss_sum_magnitude_sq(chi: &DirichletChar) -> f64 {
    let tau = gauss_sum(chi);
    complex::cnorm_sq(tau)
}

/// Detect the conductor of a Dirichlet character χ mod N.
///
/// The conductor is the smallest positive integer f such that χ is induced
/// from a (primitive) character mod f. Equivalently, f is the smallest
/// modulus m | N such that χ(a) = χ(b) whenever a ≡ b (mod m) and gcd(a, N) = 1.
pub fn detect_conductor(chi: &DirichletChar) -> u64 {
    let n = chi.modulus;

    if chi.is_principal {
        return 1;
    }

    // Get all divisors of N in ascending order
    let divisors = get_divisors(n);

    // For each divisor d of N (from smallest to largest),
    // check if χ is induced from a character mod d
    for &d in &divisors {
        if d == 1 {
            // Only the principal character has conductor 1
            if chi.is_principal {
                return 1;
            }
            continue;
        }
        if d == n {
            // Always works
            return n;
        }

        if is_induced_from(chi, d) {
            return d;
        }
    }

    n // Fallback: conductor = modulus (primitive character)
}

/// Check if character χ mod N is induced from a character mod d.
///
/// χ is induced from mod d if χ(a) = χ(b) whenever a ≡ b (mod d) and gcd(a, N) = gcd(b, N) = 1.
fn is_induced_from(chi: &DirichletChar, d: u64) -> bool {
    let n = chi.modulus;

    // For each residue class mod d, check that χ is constant on
    // elements of (Z/NZ)* in that class
    for r in 0..d {
        let mut first_value: Option<Complex> = None;

        let mut a = r;
        while a < n {
            if a.gcd(&n) == 1 {
                let val = chi.eval(a);
                match first_value {
                    None => first_value = Some(val),
                    Some(fv) => {
                        if complex::cnorm_sq(complex::csub(val, fv)) > 1e-8 {
                            return false;
                        }
                    }
                }
            }
            a += d;
        }
    }

    true
}

/// Detect conductor using Gauss sum magnitude.
///
/// For a primitive character mod f, |τ(χ)|² = f.
/// For an imprimitive character χ mod N induced from a primitive character χ₀ mod f,
/// |τ(χ)|² depends on f and how χ relates to χ₀.
///
/// This is a heuristic method: we compute |τ(χ)|² and check which divisor of N
/// it is closest to.
pub fn conductor_via_gauss(chi: &DirichletChar) -> u64 {
    let n = chi.modulus;
    let mag_sq = gauss_sum_magnitude_sq(chi);

    // The Gauss sum magnitude squared for a primitive character mod f is exactly f.
    // For imprimitive characters, it may be 0 or relate to f differently.

    // If |τ(χ)|² ≈ 0, the character is imprimitive
    if mag_sq < 0.5 {
        // Fall back to direct conductor detection
        return detect_conductor(chi);
    }

    // Check which divisor of N is closest to |τ(χ)|²
    let divisors = get_divisors(n);
    let mut best_div = n;
    let mut best_diff = f64::MAX;

    for &d in &divisors {
        let diff = (mag_sq - d as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best_div = d;
        }
    }

    best_div
}

/// The main factoring function: find a character mod N with conductor < N.
///
/// If N = pq, some characters have conductor p and some have conductor q.
/// Finding any such character reveals a factor.
pub fn factor_via_conductor(n: u64) -> Option<(u64, u64)> {
    if n <= 3 {
        return None;
    }

    // Quick check: is N even?
    if n % 2 == 0 {
        return Some((2, n / 2));
    }

    let characters = crate::characters::enumerate_characters(n);

    for chi in &characters {
        if chi.is_principal {
            continue;
        }

        let conductor = detect_conductor(chi);

        if conductor > 1 && conductor < n {
            let g = n.gcd(&conductor);
            if g > 1 && g < n {
                return Some((g, n / g));
            }
            // If gcd(conductor, N) is trivial, the conductor itself divides N
            // since conductor | N by definition
            // So conductor is a non-trivial divisor
            if n % conductor == 0 {
                return Some((conductor, n / conductor));
            }
        }
    }

    None
}

/// Get all divisors of n in ascending order.
pub fn get_divisors(n: u64) -> Vec<u64> {
    let mut divisors = Vec::new();
    let mut i = 1;
    while i * i <= n {
        if n % i == 0 {
            divisors.push(i);
            if i != n / i {
                divisors.push(n / i);
            }
        }
        i += 1;
    }
    divisors.sort();
    divisors
}

/// Result of conductor analysis for a character.
#[derive(Debug, Clone)]
pub struct ConductorAnalysis {
    /// The modulus of the character.
    pub modulus: u64,
    /// The detected conductor (via direct check).
    pub conductor_direct: u64,
    /// The estimated conductor (via Gauss sum).
    pub conductor_gauss: u64,
    /// The Gauss sum magnitude squared.
    pub gauss_mag_sq: f64,
    /// Whether the character is primitive.
    pub is_primitive: bool,
}

/// Analyze all characters mod N and return their conductor information.
pub fn analyze_conductors(n: u64) -> Vec<ConductorAnalysis> {
    let characters = crate::characters::enumerate_characters(n);
    let mut results = Vec::new();

    for chi in &characters {
        if chi.is_principal {
            continue;
        }

        let conductor_direct = detect_conductor(chi);
        let conductor_gauss = conductor_via_gauss(chi);
        let gauss_mag_sq = gauss_sum_magnitude_sq(chi);

        results.push(ConductorAnalysis {
            modulus: n,
            conductor_direct,
            conductor_gauss,
            gauss_mag_sq,
            is_primitive: conductor_direct == n,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characters;

    #[test]
    fn test_gauss_sum_primitive() {
        // For a primitive character mod p (prime), |τ(χ)|² = p
        let chars = characters::enumerate_characters(7);
        for chi in &chars {
            if chi.is_principal {
                continue;
            }
            let mag_sq = gauss_sum_magnitude_sq(chi);
            assert!(
                (mag_sq - 7.0).abs() < 0.5,
                "Primitive char mod 7: |τ(χ)|² = {}, expected ≈ 7",
                mag_sq
            );
        }
    }

    #[test]
    fn test_conductor_prime_modulus() {
        // All non-principal characters mod a prime are primitive (conductor = p)
        let chars = characters::enumerate_characters(7);
        for chi in &chars {
            let cond = detect_conductor(chi);
            if chi.is_principal {
                assert_eq!(cond, 1);
            } else {
                assert_eq!(cond, 7, "Non-principal char mod 7 should have conductor 7");
            }
        }
    }

    #[test]
    fn test_conductor_composite() {
        // For N = 15 = 3 * 5, some characters should have conductor 3 or 5
        let chars = characters::enumerate_characters(15);
        let mut conductors: Vec<u64> = chars.iter().map(|c| detect_conductor(c)).collect();
        conductors.sort();
        conductors.dedup();

        // Should find conductors: 1 (principal), 3, 5, 15
        assert!(
            conductors.contains(&1),
            "Should have conductor 1 (principal)"
        );
        assert!(conductors.contains(&3), "Should have conductor 3");
        assert!(conductors.contains(&5), "Should have conductor 5");
        assert!(
            conductors.contains(&15),
            "Should have conductor 15 (primitive)"
        );
    }

    #[test]
    fn test_factor_via_conductor_77() {
        // N = 77 = 7 * 11
        let result = factor_via_conductor(77);
        assert!(result.is_some(), "Should factor 77");
        let (a, b) = result.unwrap();
        assert_eq!(a * b, 77);
        assert!(a > 1 && b > 1, "Should give non-trivial factors");
    }

    #[test]
    fn test_factor_via_conductor_small() {
        // Test on several small semiprimes
        for &(n, p, q) in &[(15, 3, 5), (21, 3, 7), (35, 5, 7), (77, 7, 11)] {
            let result = factor_via_conductor(n);
            assert!(result.is_some(), "Should factor {}", n);
            let (a, b) = result.unwrap();
            assert_eq!(a * b, n, "Factors of {} should multiply to {}", n, n);
            assert!(
                (a == p && b == q) || (a == q && b == p),
                "Factors of {} should be {} and {}, got {} and {}",
                n,
                p,
                q,
                a,
                b
            );
        }
    }

    #[test]
    fn test_get_divisors() {
        assert_eq!(get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(get_divisors(77), vec![1, 7, 11, 77]);
    }
}
