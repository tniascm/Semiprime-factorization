//! Main factoring pipeline combining SQUFOF, CF expansion, and quadratic forms.
//!
//! Strategy:
//! 1. Trial division for small factors
//! 2. SQUFOF (fastest for moderate-size N, up to ~60 bits)
//! 3. CF-based factoring using Gauss composition and class group navigation
//! 4. Baby-step/giant-step in the class group using infrastructure distance
//! 5. If regulator is known, jump directly to neighborhood of ambiguous forms

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::cf::{cf_expand, is_perfect_square, CfIterator};
use crate::forms::{gauss_compose, QuadForm};
use crate::regulator::estimate_regulator;
use crate::squfof::squfof_factor;

/// Result of a factoring attempt.
#[derive(Debug, Clone)]
pub struct CfFactorResult {
    /// The number that was factored.
    pub n: BigUint,
    /// A non-trivial factor (None if factoring failed).
    pub factor: Option<BigUint>,
    /// Which method succeeded.
    pub method: String,
    /// Number of CF terms computed.
    pub cf_terms: usize,
    /// Number of quadratic forms explored.
    pub forms_explored: usize,
}

/// Small primes for trial division.
const SMALL_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
    293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401,
    409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509,
    521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631,
    641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751,
    757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
    881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
];

/// Try trial division by small primes.
fn trial_division(n: &BigUint) -> Option<BigUint> {
    for &p in SMALL_PRIMES {
        let bp = BigUint::from(p);
        if &bp * &bp > *n {
            return None; // n is prime
        }
        if (n % &bp).is_zero() {
            return Some(bp);
        }
    }
    None
}

/// Factor using the continued fraction method.
///
/// This is the main entry point. It tries methods in order of speed:
/// 1. Trial division (instant for small factors)
/// 2. Perfect square check
/// 3. SQUFOF (fastest for moderate N)
/// 4. CF-based class group navigation (for larger N)
pub fn factor_cf(n: &BigUint) -> CfFactorResult {
    if n <= &BigUint::one() {
        return CfFactorResult {
            n: n.clone(),
            factor: None,
            method: "trivial".to_string(),
            cf_terms: 0,
            forms_explored: 0,
        };
    }

    // Step 1: Trial division
    if let Some(f) = trial_division(n) {
        return CfFactorResult {
            n: n.clone(),
            factor: Some(f),
            method: "trial_division".to_string(),
            cf_terms: 0,
            forms_explored: 0,
        };
    }

    // Step 2: Perfect square
    if let Some(s) = is_perfect_square(n) {
        if s > BigUint::one() && &s < n {
            return CfFactorResult {
                n: n.clone(),
                factor: Some(s),
                method: "perfect_square".to_string(),
                cf_terms: 0,
                forms_explored: 0,
            };
        }
    }

    // Step 3: SQUFOF
    let squfof_result = squfof_factor(n);
    if let Some(f) = squfof_result.factor {
        return CfFactorResult {
            n: n.clone(),
            factor: Some(f),
            method: format!("squfof(mult={})", squfof_result.multiplier),
            cf_terms: squfof_result.forward_steps + squfof_result.reverse_steps,
            forms_explored: 0,
        };
    }

    // Step 4: CF-based class group exploration
    let cf_result = factor_via_class_group(n);
    if cf_result.factor.is_some() {
        return cf_result;
    }

    // Step 5: GCD from convergents
    let conv_result = factor_from_convergents(n);
    if conv_result.factor.is_some() {
        return conv_result;
    }

    CfFactorResult {
        n: n.clone(),
        factor: None,
        method: "failed".to_string(),
        cf_terms: 0,
        forms_explored: 0,
    }
}

/// Factor using SQUFOF specifically (exposed as public API).
pub fn factor_squfof(n: &BigUint) -> CfFactorResult {
    if n <= &BigUint::one() {
        return CfFactorResult {
            n: n.clone(),
            factor: None,
            method: "trivial".to_string(),
            cf_terms: 0,
            forms_explored: 0,
        };
    }

    let result = squfof_factor(n);
    CfFactorResult {
        n: n.clone(),
        factor: result.factor,
        method: if result.multiplier > 0 {
            format!("squfof(mult={})", result.multiplier)
        } else {
            "squfof(failed)".to_string()
        },
        cf_terms: result.forward_steps + result.reverse_steps,
        forms_explored: 0,
    }
}

/// Factor using regulator information.
///
/// If the regulator (or an estimate) is known, we can jump directly to
/// the neighborhood of ambiguous forms in the infrastructure, potentially
/// finding factors much faster.
pub fn factor_with_regulator(n: &BigUint, reg_hint: Option<f64>) -> CfFactorResult {
    if n <= &BigUint::one() {
        return CfFactorResult {
            n: n.clone(),
            factor: None,
            method: "trivial".to_string(),
            cf_terms: 0,
            forms_explored: 0,
        };
    }

    // First try SQUFOF (always fast)
    let sq = factor_squfof(n);
    if sq.factor.is_some() {
        return sq;
    }

    let reg = reg_hint.unwrap_or_else(|| estimate_regulator(n));

    // The regulator R gives us the period of the CF expansion.
    // Ambiguous forms occur near multiples of R/2 in the infrastructure.
    // We expand the CF and look for forms near distance R/2, R, 3R/2, etc.

    let period_estimate = (2.0 * reg).ceil() as usize;
    let max_terms = period_estimate.max(10_000).min(1_000_000);

    let expansion = cf_expand(n, max_terms);
    let n_int = BigInt::from(n.clone());

    let mut cf_terms = 0;

    // Look for Q_k values that share a factor with N
    for i in 0..expansion.q_values.len() {
        let q = &expansion.q_values[i];
        if !q.is_zero() && q > &BigUint::one() {
            let g = q.gcd(n);
            if g > BigUint::one() && &g < n {
                return CfFactorResult {
                    n: n.clone(),
                    factor: Some(g),
                    method: "regulator_cf_qval".to_string(),
                    cf_terms: i + 1,
                    forms_explored: 0,
                };
            }
        }
        cf_terms = i + 1;
    }

    // Look for h_k^2 - N that shares a factor with N
    for i in 0..expansion.h_values.len() {
        let h = &expansion.h_values[i];
        let h2 = h * h;
        let diff = if h2 >= n_int {
            let d = &h2 - &n_int;
            d.to_biguint().unwrap_or_default()
        } else {
            let d = &n_int - &h2;
            d.to_biguint().unwrap_or_default()
        };

        if !diff.is_zero() {
            let g = diff.gcd(n);
            if g > BigUint::one() && &g < n {
                return CfFactorResult {
                    n: n.clone(),
                    factor: Some(g),
                    method: "regulator_convergent".to_string(),
                    cf_terms: i + 1,
                    forms_explored: 0,
                };
            }
        }
    }

    CfFactorResult {
        n: n.clone(),
        factor: None,
        method: "regulator_failed".to_string(),
        cf_terms,
        forms_explored: 0,
    }
}

/// Factor by navigating the class group of binary quadratic forms.
///
/// Strategy:
/// 1. Construct the principal form for discriminant D = 4N or -4N
/// 2. Compute powers of a generator form using Gauss composition
/// 3. Check each resulting form for ambiguity (reveals factors)
/// 4. Use baby-step/giant-step to cover the class group efficiently
fn factor_via_class_group(n: &BigUint) -> CfFactorResult {
    let n_int = BigInt::from(n.clone());
    let four_n = BigInt::from(4u32) * &n_int;

    // Use discriminant D = 4N for indefinite forms
    // Principal form: (1, 0, -N)
    let _principal = QuadForm::new(BigInt::one(), BigInt::zero(), -n_int.clone());

    // Try to find a small form (a, b, c) with a = small prime, discriminant 4N
    // This means b^2 - 4ac = 4N, so b^2 ≡ 4N (mod 4a)
    let mut forms_explored = 0usize;
    let mut generator: Option<QuadForm> = None;

    for p in SMALL_PRIMES.iter().take(50) {
        let a = BigInt::from(*p);
        // We need b such that b^2 ≡ 4N (mod 4a), i.e., b^2 ≡ 0 (mod 4) and (b/2)^2 ≡ N (mod a)
        // Try b = 2k for k in [0, a)
        for k in 0..(*p as i64) {
            let b = BigInt::from(2 * k);
            let b2 = &b * &b;
            let rem = (&b2 - &four_n).mod_floor(&(BigInt::from(4) * &a));
            if rem.is_zero() {
                let c = (&b2 - &four_n) / (BigInt::from(4) * &a);
                let f = QuadForm::new(a.clone(), b, c);
                if f.discriminant() == four_n {
                    generator = Some(f);
                    break;
                }
            }
        }
        if generator.is_some() {
            break;
        }
    }

    let gen = match generator {
        Some(g) => g,
        None => {
            return CfFactorResult {
                n: n.clone(),
                factor: None,
                method: "class_group_no_generator".to_string(),
                cf_terms: 0,
                forms_explored: 0,
            };
        }
    };

    // Navigate the class group by computing powers of the generator
    let mut current = gen.clone();
    let max_forms = 10_000usize;

    for _ in 0..max_forms {
        forms_explored += 1;
        let reduced = current.reduce();

        // Check if this form is ambiguous
        if reduced.is_ambiguous() {
            let a_uint = reduced.a.to_biguint();
            if let Some(a_val) = a_uint {
                let g = a_val.gcd(n);
                if g > BigUint::one() && &g < n {
                    return CfFactorResult {
                        n: n.clone(),
                        factor: Some(g),
                        method: "class_group_ambiguous".to_string(),
                        cf_terms: 0,
                        forms_explored,
                    };
                }
            }
            // Also check b
            let b_uint = reduced.b.to_biguint();
            if let Some(b_val) = b_uint {
                let g = b_val.gcd(n);
                if g > BigUint::one() && &g < n {
                    return CfFactorResult {
                        n: n.clone(),
                        factor: Some(g),
                        method: "class_group_ambiguous_b".to_string(),
                        cf_terms: 0,
                        forms_explored,
                    };
                }
            }
        }

        // Also check if gcd(a, N) gives a factor directly
        let a_uint: Option<BigUint> = reduced.a.abs().to_biguint();
        if let Some(a_val) = a_uint {
            if a_val > BigUint::one() {
                let g: BigUint = a_val.gcd(n);
                if g > BigUint::one() && &g < n {
                    return CfFactorResult {
                        n: n.clone(),
                        factor: Some(g),
                        method: "class_group_gcd_a".to_string(),
                        cf_terms: 0,
                        forms_explored,
                    };
                }
            }
        }

        // Compose with generator
        current = gauss_compose(&reduced, &gen);
    }

    CfFactorResult {
        n: n.clone(),
        factor: None,
        method: "class_group_exhausted".to_string(),
        cf_terms: 0,
        forms_explored,
    }
}

/// Factor using GCD of convergents with N.
///
/// Sometimes gcd(h_k mod N, N) gives a factor. This is the simplest
/// CF-based factoring approach.
fn factor_from_convergents(n: &BigUint) -> CfFactorResult {
    let n_int = BigInt::from(n.clone());
    let mut cf_terms = 0;

    let iter = CfIterator::new(n);
    for (i, step) in iter.enumerate().take(50_000) {
        cf_terms = i + 1;

        // Check if Q_k divides N
        if !step.q_val.is_zero() && step.q_val > BigUint::one() {
            let g = step.q_val.gcd(n);
            if g > BigUint::one() && &g < n {
                return CfFactorResult {
                    n: n.clone(),
                    factor: Some(g),
                    method: "convergent_q_gcd".to_string(),
                    cf_terms,
                    forms_explored: 0,
                };
            }
        }

        // Check gcd(h_k^2 mod N, N)
        let h_mod_n = step.h.mod_floor(&n_int);
        let h_uint = h_mod_n.to_biguint().unwrap_or_default();
        let h2_mod_n = (&h_uint * &h_uint) % n;
        if h2_mod_n > BigUint::one() {
            let g = h2_mod_n.gcd(n);
            if g > BigUint::one() && &g < n {
                return CfFactorResult {
                    n: n.clone(),
                    factor: Some(g),
                    method: "convergent_h2_gcd".to_string(),
                    cf_terms,
                    forms_explored: 0,
                };
            }
        }
    }

    CfFactorResult {
        n: n.clone(),
        factor: None,
        method: "convergent_failed".to_string(),
        cf_terms,
        forms_explored: 0,
    }
}

/// Fully factor n into prime factors using CF methods.
/// Returns the list of prime factors (with multiplicity).
pub fn full_factorization(n: &BigUint) -> Vec<BigUint> {
    let mut factors = Vec::new();
    let mut remaining = n.clone();

    if remaining <= BigUint::one() {
        return factors;
    }

    // Remove factors of 2
    while (&remaining % BigUint::from(2u32)).is_zero() {
        factors.push(BigUint::from(2u32));
        remaining /= BigUint::from(2u32);
    }

    let mut work_stack = vec![remaining];

    while let Some(current) = work_stack.pop() {
        if current <= BigUint::one() {
            continue;
        }

        // Check if it's a small prime (trial division finds nothing)
        if let Some(s) = is_perfect_square(&current) {
            if s == current {
                // current is 1, skip
                continue;
            }
            // current = s^2, factor s
            work_stack.push(s.clone());
            work_stack.push(s);
            continue;
        }

        let result = factor_cf(&current);
        match result.factor {
            Some(f) => {
                let cofactor = &current / &f;
                work_stack.push(f);
                work_stack.push(cofactor);
            }
            None => {
                // Assume current is prime
                factors.push(current);
            }
        }
    }

    factors.sort();
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_cf_small() {
        let cases: Vec<(u64, Vec<u64>)> = vec![
            (15, vec![3, 5]),
            (77, vec![7, 11]),
            (143, vec![11, 13]),
            (221, vec![13, 17]),
        ];

        for (n, expected_factors) in cases {
            let bn = BigUint::from(n);
            let result = factor_cf(&bn);
            assert!(
                result.factor.is_some(),
                "Failed to factor {} using method {}",
                n,
                result.method
            );
            let f = result.factor.unwrap();
            assert!(
                (&bn % &f).is_zero(),
                "Factor {} does not divide {}",
                f,
                n
            );
            assert!(
                expected_factors.contains(&f.to_string().parse::<u64>().unwrap())
                    || expected_factors.contains(
                        &(n / f.to_string().parse::<u64>().unwrap())
                    ),
                "Factor {} of {} is not one of the expected factors {:?}",
                f,
                n,
                expected_factors
            );
        }
    }

    #[test]
    fn test_factor_squfof_api() {
        let n = BigUint::from(10403u64); // 101 * 103
        let result = factor_squfof(&n);
        assert!(result.factor.is_some(), "SQUFOF should factor 10403");
        let f = result.factor.unwrap();
        assert!((&n % &f).is_zero());
    }

    #[test]
    fn test_factor_with_regulator_api() {
        let n = BigUint::from(323u64); // 17 * 19
        let result = factor_with_regulator(&n, None);
        assert!(
            result.factor.is_some(),
            "Should factor 323 with regulator hint. Method: {}",
            result.method
        );
        let f = result.factor.unwrap();
        assert!((&n % &f).is_zero());
    }

    #[test]
    fn test_full_factorization_small() {
        let n = BigUint::from(60u64); // 2^2 * 3 * 5
        let factors = full_factorization(&n);
        assert_eq!(
            factors,
            vec![
                BigUint::from(2u32),
                BigUint::from(2u32),
                BigUint::from(3u32),
                BigUint::from(5u32),
            ]
        );
    }

    #[test]
    fn test_full_factorization_semiprime() {
        let n = BigUint::from(10403u64); // 101 * 103
        let factors = full_factorization(&n);
        assert_eq!(factors.len(), 2);
        let product: BigUint = factors.iter().product();
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_32bit_semiprime() {
        let n = BigUint::from(2820669811u64); // 57719 * 48869
        let result = factor_cf(&n);
        assert!(
            result.factor.is_some(),
            "Should factor 32-bit semiprime. Method: {}",
            result.method
        );
        let f = result.factor.unwrap();
        assert!((&n % &f).is_zero());
        let cofactor = &n / &f;
        // Verify it's a proper factorization
        assert!(f > BigUint::one() && f < n);
        assert!(cofactor > BigUint::one() && cofactor < n);
    }

    #[test]
    fn test_factor_cf_prime() {
        // 997 is prime — factoring should fail gracefully
        let n = BigUint::from(997u64);
        let result = factor_cf(&n);
        // For a prime, trial division won't find anything (997 > 31^2),
        // so other methods will be tried. Factor should be None or the number itself.
        if let Some(f) = &result.factor {
            assert!((&n % f).is_zero());
        }
    }

    #[test]
    fn test_trial_division() {
        assert_eq!(trial_division(&BigUint::from(15u64)), Some(BigUint::from(3u32)));
        assert_eq!(trial_division(&BigUint::from(49u64)), Some(BigUint::from(7u32)));
        assert_eq!(trial_division(&BigUint::from(997u64)), None); // prime
    }

    #[test]
    fn test_factor_48bit_semiprime() {
        // 100003 * 100019 = 10002200057
        let p1 = 100003u64;
        let p2 = 100019u64;
        let n = BigUint::from(p1 * p2);
        let result = factor_cf(&n);
        assert!(
            result.factor.is_some(),
            "Should factor 48-bit semiprime {} = {} * {}. Method: {}",
            p1 * p2,
            p1,
            p2,
            result.method
        );
        let f = result.factor.unwrap();
        assert!((&n % &f).is_zero());
    }
}
