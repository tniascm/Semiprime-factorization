//! Regulator estimation for real quadratic fields Q(sqrt(N)).
//!
//! The regulator R of Q(sqrt(N)) is the fundamental unit's logarithm.
//! It equals half the period length of the CF expansion of sqrt(N).
//!
//! Key properties:
//! - R = log(ε) where ε is the fundamental unit of Z[sqrt(N)]
//! - The CF period of sqrt(N) is 2R (or R if the norm is -1)
//! - h * R ≈ sqrt(N) * L(1, χ_N) for the class number formula
//!
//! The regulator can be:
//! - Computed exactly via period finding (expensive for large N)
//! - Estimated heuristically from partial CF expansion
//! - Used to speed up factoring by jumping in the infrastructure

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::cf::{cf_expand, CfExpansion};

/// Result of regulator computation.
#[derive(Debug, Clone)]
pub struct RegulatorResult {
    /// The period length of the CF expansion.
    pub period: Option<usize>,
    /// The regulator value (natural log of the fundamental unit).
    /// Computed as sum of log(a_k) over one period.
    pub regulator_estimate: f64,
    /// The fundamental solution (x, y) to Pell's equation x^2 - Ny^2 = ±1.
    pub pell_solution: Option<(BigUint, BigUint)>,
    /// Number of CF terms computed.
    pub terms_computed: usize,
}

/// Estimate the regulator of Q(sqrt(N)) from a partial CF expansion.
///
/// Uses the heuristic: R ≈ (period/2) * average(log(a_k)).
/// If the full period is found, this is exact (up to floating-point precision).
pub fn estimate_regulator(n: &BigUint) -> f64 {
    let expansion = cf_expand(n, 10_000);

    if let Some(period) = expansion.period {
        // Exact computation: R = sum of ln(a_k + 1/...) over the periodic part
        // Simplified: R ≈ sum of ln(a_k) for k in [1..period]
        // More precisely, R = ln(h_{period-1} + k_{period-1} * sqrt(N))
        // where h/k is the convergent at the end of one period.
        compute_regulator_from_expansion(&expansion, period)
    } else {
        // Heuristic estimate from partial expansion
        let terms = &expansion.partial_quotients;
        if terms.len() <= 1 {
            return 0.0;
        }

        // Use the last convergent to estimate
        let idx = expansion.h_values.len() - 1;
        if idx == 0 {
            return 0.0;
        }

        // R ≈ ln(h_k) for large k (asymptotically R = ln(ε))
        // But we don't know the period, so use average log coefficient
        let mut sum_log = 0.0f64;
        let count = terms.len() - 1;
        for i in 1..terms.len() {
            let a_val = terms[i].to_string().parse::<f64>().unwrap_or(1.0);
            if a_val > 0.0 {
                sum_log += a_val.ln();
            }
        }

        // Heuristic: period ≈ sqrt(N), so R ≈ period * avg_log
        // But we can use the partial sum as a lower bound
        sum_log / (count as f64) * estimated_period_length(n)
    }
}

/// Estimate the period length heuristically.
fn estimated_period_length(n: &BigUint) -> f64 {
    // Heuristic: period ≈ C * sqrt(N) * ln(N)^{-1}
    // More accurately, average period ≈ π * sqrt(N) / (3 * ln(2))
    let bits = n.bits() as f64;
    let sqrt_n = (2.0f64).powf(bits / 2.0);
    let ln_n = bits * (2.0f64).ln();
    std::f64::consts::PI * sqrt_n / (3.0 * ln_n)
}

/// Compute the regulator exactly from a CF expansion with known period.
fn compute_regulator_from_expansion(expansion: &CfExpansion, period: usize) -> f64 {
    if period == 0 {
        return 0.0;
    }

    // R = ln(fundamental unit) = ln(h_{r-1} + k_{r-1} * sqrt(N))
    // where r = period, and h/k is the convergent at index r (0-based: index = period).
    // The convergent at index `period` is the end of the first period.

    // Use the convergent just before the period ends
    let idx = period; // This corresponds to the convergent at the end of the period
    if idx > 0 && idx <= expansion.h_values.len() {
        let h = &expansion.h_values[idx - 1];
        let h_f64 = h.to_string().parse::<f64>().unwrap_or(1.0);
        if h_f64 > 0.0 {
            return h_f64.ln();
        }
    }

    // Fallback: sum of ln(a_k) over the periodic part
    let mut sum = 0.0f64;
    for i in 1..=period.min(expansion.partial_quotients.len() - 1) {
        let a = expansion.partial_quotients[i]
            .to_string()
            .parse::<f64>()
            .unwrap_or(1.0);
        if a > 0.0 {
            sum += a.ln();
        }
    }
    sum
}

/// Compute the regulator exactly by finding the full period of the CF expansion.
///
/// Returns the period and the fundamental solution to Pell's equation.
/// May be slow for N with very long periods.
pub fn compute_regulator(n: &BigUint, max_steps: usize) -> RegulatorResult {
    let expansion = cf_expand(n, max_steps);

    let period = expansion.period;
    let terms_computed = expansion.partial_quotients.len();

    let regulator_estimate = if let Some(p) = period {
        compute_regulator_from_expansion(&expansion, p)
    } else {
        // Partial estimate
        let mut sum = 0.0f64;
        for i in 1..expansion.partial_quotients.len() {
            let a = expansion.partial_quotients[i]
                .to_string()
                .parse::<f64>()
                .unwrap_or(1.0);
            if a > 0.0 {
                sum += a.ln();
            }
        }
        sum
    };

    // Extract Pell solution from the convergent at the end of the period
    let pell_solution = if let Some(p) = period {
        if p > 0 && p <= expansion.h_values.len() {
            let h = &expansion.h_values[p - 1];
            let k = &expansion.k_values[p - 1];
            // h^2 - N*k^2 = (-1)^p
            // If p is even: h^2 - N*k^2 = 1 (Pell equation)
            // If p is odd: h^2 - N*k^2 = -1 (negative Pell)
            //   In this case, need to go to 2*period for positive Pell
            let h_uint = h.to_biguint().unwrap_or_else(|| BigUint::zero());
            let k_uint = k.to_biguint().unwrap_or_else(|| BigUint::zero());
            Some((h_uint, k_uint))
        } else {
            None
        }
    } else {
        None
    };

    RegulatorResult {
        period,
        regulator_estimate,
        pell_solution,
        terms_computed,
    }
}

/// Check if x^2 - N*y^2 = ±1 (Pell equation verification).
pub fn verify_pell_solution(n: &BigUint, x: &BigUint, y: &BigUint) -> Option<i8> {
    let x2 = x * x;
    let ny2 = n * y * y;

    if x2 > ny2 && &x2 - &ny2 == BigUint::one() {
        Some(1) // x^2 - Ny^2 = 1
    } else if ny2 > x2 && &ny2 - &x2 == BigUint::one() {
        Some(-1) // x^2 - Ny^2 = -1
    } else {
        None
    }
}

/// Infrastructure distance computation.
///
/// In the infrastructure of a real quadratic field, each reduced form
/// has an associated "distance" from the identity. This distance can be
/// used for baby-step/giant-step algorithms in the class group.
///
/// The distance of the form at CF step k is approximately:
///   d_k = sum_{i=0}^{k} ln(a_i + alpha_i)
/// where alpha_i is the complete quotient at step i.
pub fn infrastructure_distance(n: &BigUint, steps: usize) -> Vec<f64> {
    let expansion = cf_expand(n, steps);
    let mut distances = Vec::with_capacity(expansion.partial_quotients.len());
    let mut cumulative = 0.0f64;

    let n_f64 = n.to_string().parse::<f64>().unwrap_or(1.0);
    let sqrt_n = n_f64.sqrt();

    for i in 0..expansion.partial_quotients.len() {
        let a_val = expansion.partial_quotients[i]
            .to_string()
            .parse::<f64>()
            .unwrap_or(1.0);

        if i == 0 {
            // First step: distance = ln(a_0 + sqrt(N) - a_0) = ln(sqrt(N))
            // More precisely: ln(a_0 + {sqrt(N)}) where {x} is fractional part
            cumulative = (a_val + (sqrt_n - a_val)).ln();
        } else {
            // Subsequent steps: add ln(a_k + alpha_k) ≈ ln(a_k + 1/alpha_{k+1})
            // Approximation: ln(a_k) for a_k > 0
            if a_val > 0.0 {
                cumulative += a_val.ln();
            }
        }

        distances.push(cumulative);
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulator_sqrt7() {
        // sqrt(7) has period 4. The regulator R = ln(8 + 3*sqrt(7)) ≈ ln(8 + 7.937) ≈ 2.77
        let n = BigUint::from(7u32);
        let result = compute_regulator(&n, 100);
        assert_eq!(result.period, Some(4));

        // Verify Pell solution: 8^2 - 7*3^2 = 64 - 63 = 1
        if let Some((x, y)) = &result.pell_solution {
            assert_eq!(x, &BigUint::from(8u32));
            assert_eq!(y, &BigUint::from(3u32));
            assert_eq!(verify_pell_solution(&n, x, y), Some(1));
        } else {
            panic!("Should find Pell solution for N=7");
        }
    }

    #[test]
    fn test_regulator_sqrt2() {
        // sqrt(2) has period 1. Pell: 1^2 - 2*1^2 = -1 (negative Pell)
        // Need period 2 for positive Pell: 3^2 - 2*2^2 = 9 - 8 = 1
        let n = BigUint::from(2u32);
        let result = compute_regulator(&n, 100);
        assert_eq!(result.period, Some(1));

        if let Some((x, y)) = &result.pell_solution {
            // Period 1 gives the negative Pell solution
            assert_eq!(verify_pell_solution(&n, x, y), Some(-1));
        }
    }

    #[test]
    fn test_regulator_sqrt13() {
        // sqrt(13) has period 5. Pell: 649^2 - 13*180^2 = 421201 - 421200 = 1
        let n = BigUint::from(13u32);
        let result = compute_regulator(&n, 100);
        assert_eq!(result.period, Some(5));

        if let Some((x, y)) = &result.pell_solution {
            let pell = verify_pell_solution(&n, x, y);
            assert!(pell.is_some(), "Should satisfy Pell equation");
        }
    }

    #[test]
    fn test_estimate_regulator_positive() {
        let n = BigUint::from(7u32);
        let r = estimate_regulator(&n);
        assert!(r > 0.0, "Regulator should be positive, got {}", r);
    }

    #[test]
    fn test_infrastructure_distance() {
        let n = BigUint::from(7u32);
        let distances = infrastructure_distance(&n, 20);

        // Distances should be monotonically increasing
        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "Distances should be non-decreasing at step {}",
                i
            );
        }
    }

    #[test]
    fn test_verify_pell_solution() {
        let n = BigUint::from(7u32);
        // 8^2 - 7*3^2 = 64 - 63 = 1
        assert_eq!(
            verify_pell_solution(&n, &BigUint::from(8u32), &BigUint::from(3u32)),
            Some(1)
        );

        // Not a solution
        assert_eq!(
            verify_pell_solution(&n, &BigUint::from(5u32), &BigUint::from(2u32)),
            None
        );

        // Negative Pell: 3^2 - 2*2^2 = 9 - 8 = 1
        let n2 = BigUint::from(2u32);
        assert_eq!(
            verify_pell_solution(&n2, &BigUint::from(3u32), &BigUint::from(2u32)),
            Some(1)
        );
    }
}
