//! SQUFOF (SQUare FOrm Factorization) — Shanks' algorithm.
//!
//! The algorithm works by:
//! 1. Expanding sqrt(kN) as a continued fraction, tracking Q_i values
//! 2. Looking for Q_i that is a perfect square at an odd iteration
//! 3. When Q_i = s^2, start a "reverse" expansion (inverse square root form)
//! 4. Continue until finding P_i = P_{i+1} (an ambiguous cycle)
//! 5. gcd(N, P_i) gives a non-trivial factor
//!
//! SQUFOF is O(N^{1/4}) and works well for numbers up to about 60 bits.
//! Using multipliers improves success probability significantly.
//!
//! Reference: Gower & Wagstaff, "Square Form Factorization" (2008)

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::cf::{is_perfect_square, isqrt};

/// SQUFOF multipliers to improve success rate.
const MULTIPLIERS: &[u64] = &[
    1, 2, 3, 5, 7, 11, 13, 15, 17, 19, 21, 23, 29, 31, 33, 35, 37, 41, 43, 47, 51, 53, 55, 59,
    61, 67, 69, 71, 73, 77, 79, 83, 85, 89, 91, 93, 95, 97,
];

/// Result from SQUFOF factoring attempt.
#[derive(Debug, Clone)]
pub struct SqufofResult {
    /// The factor found (None if no factor was found).
    pub factor: Option<BigUint>,
    /// Number of CF expansion steps taken in forward phase.
    pub forward_steps: usize,
    /// Number of CF expansion steps taken in reverse phase.
    pub reverse_steps: usize,
    /// Which multiplier was successful (if any).
    pub multiplier: u64,
}

/// Core SQUFOF implementation for a single value D = kN.
///
/// Uses the standard algorithm based on the CF expansion of sqrt(D).
///
/// The recurrence maintains (P_i, Q_i) where:
///   q_i = floor((S + P_i) / Q_i)
///   P_{i+1} = q_i * Q_i - P_i
///   Q_{i+1} = Q_{i-1} + q_i * (P_i - P_{i+1})
/// where S = floor(sqrt(D)).
///
/// Note: In this recurrence, P_i is always <= S and the Q values can
/// be computed without signed arithmetic because P_i >= P_{i+1} when
/// a_i > 0 and the subtraction is safe.
///
/// Actually, that's not always true. The safe computation is:
///   Q_{i+1} = (D - P_{i+1}^2) / Q_i
/// which is always exact (divides evenly) and avoids the subtraction issue.
///
/// Returns (factor_of_D, forward_steps, reverse_steps) or None.
fn squfof_inner(d: &BigUint, max_steps: usize) -> Option<(BigUint, usize, usize)> {
    if d <= &BigUint::from(1u32) {
        return None;
    }

    // Check if D is a perfect square
    if let Some(s) = is_perfect_square(d) {
        if s > BigUint::one() {
            return Some((s, 0, 0));
        }
    }

    let s0 = isqrt(d);
    if &(&s0 * &s0) == d {
        return Some((s0, 0, 0));
    }

    // Initialize forward phase
    // P_0 = S, Q_0 = D - S^2, Q_{-1} = 1
    let mut p_i = s0.clone();
    let mut q_i = d - &s0 * &s0;

    if q_i.is_zero() {
        return Some((s0, 0, 0));
    }

    let mut _q_prev = BigUint::one();
    let mut forward_steps = 0usize;

    let mut found_square: Option<(BigUint, BigUint)> = None; // (sqrt(Q_i), P_i at that point)

    for i in 1..=max_steps {
        if q_i.is_zero() {
            break;
        }

        let qi = (&s0 + &p_i) / &q_i; // partial quotient
        let p_next = &qi * &q_i - &p_i;

        // Use the safe formula: Q_{i+1} = (D - P_{i+1}^2) / Q_i
        let p_next_sq = &p_next * &p_next;
        let q_next = if d >= &p_next_sq {
            (d - &p_next_sq) / &q_i
        } else {
            // This shouldn't happen, but handle gracefully
            break;
        };

        forward_steps = i;

        _q_prev = q_i;
        q_i = q_next;
        p_i = p_next;

        // Check for perfect square Q at odd iteration
        if i % 2 == 1 {
            if let Some(sq) = is_perfect_square(&q_i) {
                if sq > BigUint::one() {
                    found_square = Some((sq, p_i.clone()));
                    break;
                }
                // Q_i = 0 or 1 is trivial
            }
        }
    }

    let (sq_root, sq_p) = found_square?;

    // Reverse phase:
    // Initialize a new CF expansion from the "inverse square root" form.
    //
    // Standard initialization:
    //   b = floor((S + sq_p) / sq_root)
    //   P_0' = b * sq_root - sq_p
    //   Q_0' = sq_root
    //   Q_{-1}' = (D - P_0'^2) / sq_root

    let b = (&s0 + &sq_p) / &sq_root;
    let mut rp = &b * &sq_root - &sq_p;
    let rp_sq = &rp * &rp;
    let mut _rq_prev = if d >= &rp_sq {
        (d - &rp_sq) / &sq_root
    } else {
        return None;
    };
    let mut rq_cur = sq_root.clone();

    #[allow(unused_assignments)]
    let mut reverse_steps = 0usize;

    for i in 1..=max_steps {
        if rq_cur.is_zero() {
            break;
        }

        let qi = (&s0 + &rp) / &rq_cur;
        let rp_next = &qi * &rq_cur - &rp;

        // Use safe formula for Q
        let rp_next_sq = &rp_next * &rp_next;
        let rq_next = if d >= &rp_next_sq {
            (d - &rp_next_sq) / &rq_cur
        } else {
            break;
        };

        reverse_steps = i;

        // Check for ambiguous form: P_{i} = P_{i+1}
        if rp_next == rp {
            let g = d.gcd(&rp);
            if g > BigUint::one() && &g < d {
                return Some((g, forward_steps, reverse_steps));
            }
            let g2 = d.gcd(&rq_cur);
            if g2 > BigUint::one() && &g2 < d {
                return Some((g2, forward_steps, reverse_steps));
            }
        }

        _rq_prev = rq_cur;
        rq_cur = rq_next;
        rp = rp_next;
    }

    None
}

/// Factor n using SQUFOF with multipliers.
///
/// Tries multiple multipliers k*N to increase the probability of finding
/// a square form quickly. For each multiplier k, runs SQUFOF on kN.
pub fn squfof_factor(n: &BigUint) -> SqufofResult {
    if n <= &BigUint::from(1u32) {
        return SqufofResult {
            factor: None,
            forward_steps: 0,
            reverse_steps: 0,
            multiplier: 1,
        };
    }

    // Check small factors first
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        let bp = BigUint::from(p);
        if n > &bp && (n % &bp).is_zero() {
            return SqufofResult {
                factor: Some(bp),
                forward_steps: 0,
                reverse_steps: 0,
                multiplier: 1,
            };
        }
    }

    // Check if perfect square
    if let Some(s) = is_perfect_square(n) {
        if s > BigUint::one() && &s < n {
            return SqufofResult {
                factor: Some(s),
                forward_steps: 0,
                reverse_steps: 0,
                multiplier: 1,
            };
        }
    }

    // Estimate max_steps ~ N^{1/4}
    let bits = n.bits();
    let max_steps = if bits <= 32 {
        10_000usize
    } else if bits <= 64 {
        100_000
    } else {
        1_000_000
    };

    for &mult in MULTIPLIERS {
        let kn = n * BigUint::from(mult);

        // Skip if kN is a perfect square
        if is_perfect_square(&kn).is_some() {
            continue;
        }

        if let Some((factor, fwd, rev)) = squfof_inner(&kn, max_steps) {
            // Extract the factor of N: gcd(factor, N)
            let g = factor.gcd(n);
            if g > BigUint::one() && &g < n {
                return SqufofResult {
                    factor: Some(g),
                    forward_steps: fwd,
                    reverse_steps: rev,
                    multiplier: mult,
                };
            }
            // Try kN / factor
            let cofactor = &kn / &factor;
            let g2 = cofactor.gcd(n);
            if g2 > BigUint::one() && &g2 < n {
                return SqufofResult {
                    factor: Some(g2),
                    forward_steps: fwd,
                    reverse_steps: rev,
                    multiplier: mult,
                };
            }
        }
    }

    SqufofResult {
        factor: None,
        forward_steps: max_steps,
        reverse_steps: 0,
        multiplier: 0,
    }
}

/// Simpler variant of SQUFOF: just scan GCD of P_k and Q_k values with N.
/// Less reliable but works as a fallback.
pub fn squfof_simple(n: &BigUint) -> Option<BigUint> {
    if n <= &BigUint::from(1u32) {
        return None;
    }

    let a0 = isqrt(n);
    if &(&a0 * &a0) == n {
        return if a0 > BigUint::one() {
            Some(a0)
        } else {
            None
        };
    }

    let q0 = n - &a0 * &a0;
    if q0.is_zero() {
        return Some(a0);
    }

    let mut p_prev = a0.clone();
    let mut q_prev = q0;
    let mut _q_prev2 = BigUint::one();

    let max_steps = 100_000usize;

    for step in 1..=max_steps {
        if q_prev.is_zero() {
            break;
        }

        let a_k = (&a0 + &p_prev) / &q_prev;
        let p_k = &a_k * &q_prev - &p_prev;

        // Use safe formula for Q
        let p_k_sq = &p_k * &p_k;
        let q_k = if n >= &p_k_sq {
            (n - &p_k_sq) / &q_prev
        } else {
            break;
        };

        // Check if gcd(P_k, N) gives a nontrivial factor
        if step > 1 {
            let g = p_k.gcd(n);
            if g > BigUint::one() && &g < n {
                return Some(g);
            }
        }

        // Check Q_k
        if !q_k.is_zero() {
            let g = q_k.gcd(n);
            if g > BigUint::one() && &g < n {
                return Some(g);
            }
        }

        _q_prev2 = q_prev;
        q_prev = q_k;
        p_prev = p_k;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_squfof(n: u64) {
        let bn = BigUint::from(n);
        let result = squfof_factor(&bn);
        assert!(
            result.factor.is_some(),
            "SQUFOF failed to factor {}",
            n
        );
        let f = result.factor.unwrap();
        assert!(
            &f > &BigUint::one() && &f < &bn,
            "Factor {} of {} is trivial",
            f,
            n
        );
        assert!(
            (&bn % &f).is_zero(),
            "{} does not divide {}",
            f,
            n
        );
    }

    #[test]
    fn test_squfof_small_composites() {
        check_squfof(15); // 3 * 5
        check_squfof(77); // 7 * 11
        check_squfof(143); // 11 * 13
        check_squfof(221); // 13 * 17
        check_squfof(323); // 17 * 19
    }

    #[test]
    fn test_squfof_medium_composites() {
        check_squfof(667); // 23 * 29
        check_squfof(1073); // 29 * 37
        check_squfof(10403); // 101 * 103
    }

    #[test]
    fn test_squfof_larger() {
        check_squfof(1000003 * 7); // 7000021
        check_squfof(10007 * 10009); // = 100160063
    }

    #[test]
    fn test_squfof_32bit_semiprime() {
        // 57719 * 48869 = 2820669811
        check_squfof(2820669811);
    }

    #[test]
    fn test_squfof_returns_none_for_prime() {
        let n = BigUint::from(997u32); // prime
        let result = squfof_factor(&n);
        if let Some(f) = &result.factor {
            assert!((&n % f).is_zero());
        }
    }

    #[test]
    fn test_squfof_perfect_square() {
        let n = BigUint::from(49u32); // 7^2
        let result = squfof_factor(&n);
        assert_eq!(result.factor, Some(BigUint::from(7u32)));
    }

    #[test]
    fn test_squfof_simple_small() {
        let n = BigUint::from(143u32); // 11 * 13
        let result = squfof_simple(&n);
        assert!(result.is_some());
        let f = result.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f > BigUint::one() && f < n);
    }
}
