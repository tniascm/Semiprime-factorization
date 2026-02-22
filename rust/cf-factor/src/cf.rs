//! Continued fraction expansion of sqrt(N).
//!
//! The continued fraction expansion of sqrt(N) for non-square N is periodic:
//!   sqrt(N) = a_0 + 1/(a_1 + 1/(a_2 + ...))
//! where [a_1, a_2, ..., a_r] repeats with period r.
//!
//! We track convergents p_k/q_k satisfying p_k^2 - N*q_k^2 = (-1)^(k+1) * Q_k,
//! where Q_k are the quadratic form values used in SQUFOF and other algorithms.

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

/// A single step in the continued fraction expansion.
#[derive(Debug, Clone)]
pub struct CfStep {
    /// The partial quotient a_k.
    pub a: BigUint,
    /// P_k value in the recurrence.
    pub p_val: BigUint,
    /// Q_k value — the "form value" at this step.
    pub q_val: BigUint,
    /// Numerator convergent h_k.
    pub h: BigInt,
    /// Denominator convergent k: BigInt.
    pub k: BigInt,
}

/// Result of a continued fraction expansion.
#[derive(Debug, Clone)]
pub struct CfExpansion {
    /// The number N whose sqrt we are expanding.
    pub n: BigUint,
    /// Floor of sqrt(N).
    pub a0: BigUint,
    /// The sequence of partial quotients [a_0, a_1, a_2, ...].
    pub partial_quotients: Vec<BigUint>,
    /// The Q_k values at each step.
    pub q_values: Vec<BigUint>,
    /// The P_k values at each step.
    pub p_values: Vec<BigUint>,
    /// Convergent numerators h_k.
    pub h_values: Vec<BigInt>,
    /// Convergent denominators k_k.
    pub k_values: Vec<BigInt>,
    /// Detected period length (None if not found within max_terms).
    pub period: Option<usize>,
}

/// Compute floor(sqrt(n)) for BigUint using Newton's method.
pub fn isqrt(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    if *n == BigUint::one() {
        return BigUint::one();
    }

    // Initial guess: 2^((bits+1)/2)
    let bits = n.bits();
    let mut x = BigUint::one() << ((bits + 1) / 2);

    loop {
        // x_next = (x + n/x) / 2
        let x_next = (&x + n / &x) >> 1;
        if x_next >= x {
            return x;
        }
        x = x_next;
    }
}

/// Check if n is a perfect square. Returns Some(sqrt) if so, None otherwise.
pub fn is_perfect_square(n: &BigUint) -> Option<BigUint> {
    if n.is_zero() {
        return Some(BigUint::zero());
    }
    let s = isqrt(n);
    if &(&s * &s) == n {
        Some(s)
    } else {
        None
    }
}

/// Expand the continued fraction of sqrt(N) for up to max_terms partial quotients.
///
/// Returns a `CfExpansion` containing all partial quotients, convergents, Q-values,
/// and the detected period (if found).
///
/// The recurrence used:
/// ```text
/// a_0 = floor(sqrt(N))
/// P_0 = a_0, Q_0 = N - a_0^2
/// For k >= 1:
///   a_k = floor((a_0 + P_{k-1}) / Q_{k-1})
///   P_k = a_k * Q_{k-1} - P_{k-1}
///   Q_k = Q_{k-2} + a_k * (P_{k-1} - P_k)
///
/// Convergents:
///   h_{-1} = 1, h_0 = a_0
///   h_k = a_k * h_{k-1} + h_{k-2}
///   k_{-1} = 0, k_0 = 1
///   k_k = a_k * k_{k-1} + k_{k-2}
/// ```
pub fn cf_expand(n: &BigUint, max_terms: usize) -> CfExpansion {
    let a0 = isqrt(n);

    // Check if N is a perfect square — CF expansion is finite.
    if &(&a0 * &a0) == n {
        return CfExpansion {
            n: n.clone(),
            a0: a0.clone(),
            partial_quotients: vec![a0],
            q_values: vec![BigUint::zero()],
            p_values: vec![BigUint::zero()],
            h_values: vec![],
            k_values: vec![],
            period: Some(0),
        };
    }

    let mut partial_quotients = Vec::with_capacity(max_terms);
    let mut q_values = Vec::with_capacity(max_terms);
    let mut p_values = Vec::with_capacity(max_terms);
    let mut h_values = Vec::with_capacity(max_terms);
    let mut k_values = Vec::with_capacity(max_terms);

    // Step 0
    partial_quotients.push(a0.clone());

    let p0 = a0.clone();
    let q0 = n - &a0 * &a0;
    p_values.push(p0.clone());
    q_values.push(q0.clone());

    // Convergents: h_{-1} = 1, h_0 = a_0; k_{-1} = 0, k_0 = 1
    let h0 = BigInt::from(a0.clone());
    let k0 = BigInt::one();
    h_values.push(h0.clone());
    k_values.push(k0.clone());

    let mut h_prev2 = BigInt::one(); // h_{-1}
    let mut h_prev1 = h0;
    let mut k_prev2 = BigInt::zero(); // k_{-1}
    let mut k_prev1 = k0;

    let mut p_prev = p0;
    let mut q_prev = q0;
    let mut q_prev2 = BigUint::one(); // Q_{-1} = 1 conceptually (for the Q recurrence)

    let mut period: Option<usize> = None;

    for i in 1..max_terms {
        if q_prev.is_zero() {
            break;
        }

        // a_k = floor((a0 + P_{k-1}) / Q_{k-1})
        let a_k = (&a0 + &p_prev) / &q_prev;

        // P_k = a_k * Q_{k-1} - P_{k-1}
        let p_k = &a_k * &q_prev - &p_prev;

        // Q_k = Q_{k-2} + a_k * (P_{k-1} - P_k)
        // Note: P_{k-1} >= P_k always in this recurrence for sqrt expansion
        let q_k = if p_prev >= p_k {
            &q_prev2 + &a_k * (&p_prev - &p_k)
        } else {
            // This shouldn't happen for standard sqrt CF, but handle gracefully
            let diff = &p_k - &p_prev;
            if q_prev2 >= a_k.clone() * &diff {
                &q_prev2 - &a_k * diff
            } else {
                // Fallback: use the direct formula Q_k = (N - P_k^2) / Q_{k-1}
                (n - &p_k * &p_k) / &q_prev
            }
        };

        // Convergents
        let a_k_int = BigInt::from(a_k.clone());
        let h_k = &a_k_int * &h_prev1 + &h_prev2;
        let k_k = &a_k_int * &k_prev1 + &k_prev2;

        partial_quotients.push(a_k.clone());
        p_values.push(p_k.clone());
        q_values.push(q_k.clone());
        h_values.push(h_k.clone());
        k_values.push(k_k.clone());

        // Period detection: period ends when a_k = 2*a_0 (first occurrence only)
        if period.is_none() && a_k == BigUint::from(2u32) * &a0 {
            period = Some(i); // period length = i (number of terms in the periodic part)
            // Don't break — caller might want more terms
        }

        // Update state
        q_prev2 = q_prev;
        q_prev = q_k;
        p_prev = p_k;
        h_prev2 = h_prev1;
        h_prev1 = h_k;
        k_prev2 = k_prev1;
        k_prev1 = k_k;
    }

    CfExpansion {
        n: n.clone(),
        a0: a0.clone(),
        partial_quotients,
        q_values,
        p_values,
        h_values,
        k_values,
        period,
    }
}

/// Verify the fundamental relation: h_k^2 - N * k_k^2 = (-1)^(k+1) * Q_k
/// for convergent index k (0-based, where k=0 corresponds to a_0).
///
/// Returns true if the relation holds at step `index`.
pub fn verify_convergent_relation(expansion: &CfExpansion, index: usize) -> bool {
    if index >= expansion.h_values.len() || index >= expansion.q_values.len() {
        return false;
    }

    let n_int = BigInt::from(expansion.n.clone());
    let h = &expansion.h_values[index];
    let k = &expansion.k_values[index];
    let q = BigInt::from(expansion.q_values[index].clone());

    let lhs = h * h - &n_int * k * k;
    // (-1)^(k+1) * Q_k where k is the 0-based index
    // At index 0 (a_0): (-1)^1 * Q_0 = -Q_0
    // At index 1 (a_1): (-1)^2 * Q_1 = Q_1
    // At index 2 (a_2): (-1)^3 * Q_2 = -Q_2
    let sign = if index % 2 == 0 {
        -BigInt::one()
    } else {
        BigInt::one()
    };
    let rhs = sign * &q;

    lhs == rhs
}

/// Iterator-based CF expansion for memory-efficient processing.
/// Yields one `CfStep` at a time without storing the entire expansion.
pub struct CfIterator {
    n: BigUint,
    a0: BigUint,
    p_prev: BigUint,
    q_prev: BigUint,
    q_prev2: BigUint,
    h_prev1: BigInt,
    h_prev2: BigInt,
    k_prev1: BigInt,
    k_prev2: BigInt,
    step: usize,
    finished: bool,
}

impl CfIterator {
    pub fn new(n: &BigUint) -> Self {
        let a0 = isqrt(n);
        let is_square = &(&a0 * &a0) == n;

        CfIterator {
            n: n.clone(),
            a0: a0.clone(),
            p_prev: a0.clone(),
            q_prev: if is_square {
                BigUint::zero()
            } else {
                n - &a0 * &a0
            },
            q_prev2: BigUint::one(),
            h_prev1: BigInt::from(a0.clone()),
            h_prev2: BigInt::one(),
            k_prev1: BigInt::one(),
            k_prev2: BigInt::zero(),
            step: 0,
            finished: is_square,
        }
    }
}

impl Iterator for CfIterator {
    type Item = CfStep;

    fn next(&mut self) -> Option<CfStep> {
        if self.finished {
            return None;
        }

        if self.step == 0 {
            self.step = 1;
            let result = CfStep {
                a: self.a0.clone(),
                p_val: self.p_prev.clone(),
                q_val: self.q_prev.clone(),
                h: self.h_prev1.clone(),
                k: self.k_prev1.clone(),
            };
            return Some(result);
        }

        if self.q_prev.is_zero() {
            self.finished = true;
            return None;
        }

        let a_k = (&self.a0 + &self.p_prev) / &self.q_prev;
        let p_k = &a_k * &self.q_prev - &self.p_prev;

        let q_k = if self.p_prev >= p_k {
            &self.q_prev2 + &a_k * (&self.p_prev - &p_k)
        } else {
            let diff = &p_k - &self.p_prev;
            if self.q_prev2 >= a_k.clone() * &diff {
                &self.q_prev2 - &a_k * diff
            } else {
                (&self.n - &p_k * &p_k) / &self.q_prev
            }
        };

        let a_k_int = BigInt::from(a_k.clone());
        let h_k = &a_k_int * &self.h_prev1 + &self.h_prev2;
        let k_k = &a_k_int * &self.k_prev1 + &self.k_prev2;

        let result = CfStep {
            a: a_k,
            p_val: p_k.clone(),
            q_val: q_k.clone(),
            h: h_k.clone(),
            k: k_k.clone(),
        };

        self.q_prev2 = std::mem::replace(&mut self.q_prev, q_k);
        self.p_prev = p_k;
        self.h_prev2 = std::mem::replace(&mut self.h_prev1, h_k);
        self.k_prev2 = std::mem::replace(&mut self.k_prev1, k_k);
        self.step += 1;

        Some(result)
    }
}

/// Get the periodic part of the CF expansion of sqrt(n).
/// Returns (a0, periodic_part) where sqrt(n) = [a0; periodic_part, periodic_part, ...].
pub fn cf_periodic_part(n: &BigUint, max_terms: usize) -> (BigUint, Vec<BigUint>) {
    let expansion = cf_expand(n, max_terms);
    if let Some(period) = expansion.period {
        let periodic = expansion.partial_quotients[1..=period].to_vec();
        (expansion.a0, periodic)
    } else {
        // Return what we have
        let periodic = expansion.partial_quotients[1..].to_vec();
        (expansion.a0, periodic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(&BigUint::from(0u32)), BigUint::from(0u32));
        assert_eq!(isqrt(&BigUint::from(1u32)), BigUint::from(1u32));
        assert_eq!(isqrt(&BigUint::from(4u32)), BigUint::from(2u32));
        assert_eq!(isqrt(&BigUint::from(7u32)), BigUint::from(2u32));
        assert_eq!(isqrt(&BigUint::from(9u32)), BigUint::from(3u32));
        assert_eq!(isqrt(&BigUint::from(15u32)), BigUint::from(3u32));
        assert_eq!(isqrt(&BigUint::from(16u32)), BigUint::from(4u32));
        assert_eq!(isqrt(&BigUint::from(100u32)), BigUint::from(10u32));
    }

    #[test]
    fn test_is_perfect_square() {
        assert_eq!(is_perfect_square(&BigUint::from(0u32)), Some(BigUint::from(0u32)));
        assert_eq!(is_perfect_square(&BigUint::from(1u32)), Some(BigUint::from(1u32)));
        assert_eq!(is_perfect_square(&BigUint::from(4u32)), Some(BigUint::from(2u32)));
        assert_eq!(is_perfect_square(&BigUint::from(7u32)), None);
        assert_eq!(is_perfect_square(&BigUint::from(9u32)), Some(BigUint::from(3u32)));
        assert_eq!(
            is_perfect_square(&BigUint::from(10000u32)),
            Some(BigUint::from(100u32))
        );
    }

    #[test]
    fn test_cf_sqrt7() {
        // sqrt(7) = [2; 1, 1, 1, 4, 1, 1, 1, 4, ...]
        // Period = 4: [1, 1, 1, 4]
        let n = BigUint::from(7u32);
        let expansion = cf_expand(&n, 20);

        assert_eq!(expansion.a0, BigUint::from(2u32));
        assert_eq!(expansion.partial_quotients[0], BigUint::from(2u32));
        assert_eq!(expansion.partial_quotients[1], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[2], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[3], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[4], BigUint::from(4u32));
        assert_eq!(expansion.partial_quotients[5], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[6], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[7], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[8], BigUint::from(4u32));

        // Period should be 4
        assert_eq!(expansion.period, Some(4));
    }

    #[test]
    fn test_cf_sqrt2() {
        // sqrt(2) = [1; 2, 2, 2, ...]
        let n = BigUint::from(2u32);
        let expansion = cf_expand(&n, 10);
        assert_eq!(expansion.a0, BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[0], BigUint::from(1u32));
        for i in 1..expansion.partial_quotients.len() {
            assert_eq!(expansion.partial_quotients[i], BigUint::from(2u32));
        }
        assert_eq!(expansion.period, Some(1));
    }

    #[test]
    fn test_cf_sqrt13() {
        // sqrt(13) = [3; 1, 1, 1, 1, 6, ...]
        let n = BigUint::from(13u32);
        let expansion = cf_expand(&n, 15);
        assert_eq!(expansion.a0, BigUint::from(3u32));
        assert_eq!(expansion.partial_quotients[0], BigUint::from(3u32));
        assert_eq!(expansion.partial_quotients[1], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[2], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[3], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[4], BigUint::from(1u32));
        assert_eq!(expansion.partial_quotients[5], BigUint::from(6u32));
    }

    #[test]
    fn test_convergent_relation() {
        // Verify h_k^2 - N * k_k^2 = (-1)^(k+1) * Q_k
        let n = BigUint::from(7u32);
        let expansion = cf_expand(&n, 20);

        for i in 0..expansion.h_values.len().min(15) {
            assert!(
                verify_convergent_relation(&expansion, i),
                "Convergent relation failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_convergent_relation_various_n() {
        for n_val in [2u32, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 23, 29, 31, 77, 143] {
            let n = BigUint::from(n_val);
            let expansion = cf_expand(&n, 30);
            for i in 0..expansion.h_values.len().min(20) {
                assert!(
                    verify_convergent_relation(&expansion, i),
                    "Convergent relation failed for N={} at index {}",
                    n_val,
                    i
                );
            }
        }
    }

    #[test]
    fn test_cf_iterator() {
        let n = BigUint::from(7u32);
        let iter = CfIterator::new(&n);
        let steps: Vec<CfStep> = iter.take(9).collect();

        assert_eq!(steps[0].a, BigUint::from(2u32));
        assert_eq!(steps[1].a, BigUint::from(1u32));
        assert_eq!(steps[4].a, BigUint::from(4u32));
        assert_eq!(steps[5].a, BigUint::from(1u32));
        assert_eq!(steps[8].a, BigUint::from(4u32));
    }

    #[test]
    fn test_cf_periodic_part() {
        let n = BigUint::from(7u32);
        let (a0, periodic) = cf_periodic_part(&n, 20);
        assert_eq!(a0, BigUint::from(2u32));
        assert_eq!(
            periodic,
            vec![
                BigUint::from(1u32),
                BigUint::from(1u32),
                BigUint::from(1u32),
                BigUint::from(4u32),
            ]
        );
    }

    #[test]
    fn test_cf_perfect_square() {
        let n = BigUint::from(25u32);
        let expansion = cf_expand(&n, 10);
        assert_eq!(expansion.a0, BigUint::from(5u32));
        assert_eq!(expansion.period, Some(0));
        assert_eq!(expansion.partial_quotients.len(), 1);
    }
}
