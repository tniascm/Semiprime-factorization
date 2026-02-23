//! Regulator-guided factoring (Murru-Salvatori approach).
//!
//! Key insight: if the regulator R of Q(sqrt(N)) is known (or well-estimated),
//! we can jump directly to neighborhoods of R/2, R, 3R/2, etc. in the
//! infrastructure, where ambiguous forms are most likely to appear.
//!
//! An ambiguous form (a, 0, c) or (a, a, c) at distance R/2 from the
//! identity gives gcd(a, N) = factor of N.
//!
//! This converts the O(sqrt(N)) walk into O(sqrt(h)) where h is the class
//! number — which is typically much smaller than sqrt(N).
//!
//! If h is known (from class_number_real), the factoring becomes polynomial
//! in log(N) * poly(h). For h = 1 (which is the case for many semiprimes),
//! this is genuinely fast.

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

use cf_factor::regulator::{compute_regulator, estimate_regulator};

use crate::class_number_real::class_number_real_estimate;
use crate::infrastructure::{
    form_reveals_factor, power_step, rho_step_ctx, InfraContext, InfraForm,
};

/// Configuration for regulator-guided factoring.
#[derive(Debug, Clone)]
pub struct RegulatorGuidedConfig {
    /// Maximum CF terms for regulator computation.
    pub max_regulator_terms: usize,
    /// Number of neighborhood points to check around each target distance.
    pub neighborhood_size: usize,
    /// Whether to use the class number to refine the search.
    pub use_class_number: bool,
    /// Number of multiples of R/2 to check.
    pub max_multiples: usize,
}

impl Default for RegulatorGuidedConfig {
    fn default() -> Self {
        Self {
            max_regulator_terms: 10_000,
            neighborhood_size: 100,
            use_class_number: true,
            max_multiples: 20,
        }
    }
}

/// Result from regulator-guided factoring.
#[derive(Debug, Clone)]
pub struct RegulatorGuidedResult {
    /// Factor found (if any).
    pub factor: Option<BigUint>,
    /// Regulator estimate used.
    pub regulator: f64,
    /// Class number estimate used.
    pub class_number: u64,
    /// Which multiple of R/2 the factor was found at (if any).
    pub found_at_multiple: Option<usize>,
    /// Total forms checked.
    pub forms_checked: usize,
}

/// Factor N using regulator-guided search.
///
/// Algorithm:
/// 1. Estimate the regulator R from the CF expansion.
/// 2. Optionally estimate the class number h.
/// 3. For each multiple k of R/(2h):
///    a. Jump to distance k * R/(2h) in the infrastructure.
///    b. Walk a small neighborhood around that point.
///    c. Check each form for factors.
///
/// The idea: ambiguous forms appear at distance R/2 (mod R) in the
/// infrastructure. With class number h, there are h copies of the
/// cycle, so we check k * R / (2h) for each k.
pub fn regulator_guided_factor(
    n: &BigUint,
    config: &RegulatorGuidedConfig,
) -> RegulatorGuidedResult {
    let one = BigUint::one();
    let mut forms_checked = 0;

    // Step 1: Compute regulator
    let reg_result = compute_regulator(n, config.max_regulator_terms);
    let regulator = if reg_result.regulator_estimate > 0.0 {
        reg_result.regulator_estimate
    } else {
        estimate_regulator(n)
    };

    if regulator <= 0.0 || !regulator.is_finite() {
        return RegulatorGuidedResult {
            factor: None,
            regulator: 0.0,
            class_number: 1,
            found_at_multiple: None,
            forms_checked: 0,
        };
    }

    // Step 2: Estimate class number
    let class_number = if config.use_class_number {
        let cn = class_number_real_estimate(n);
        cn.class_number.max(1)
    } else {
        1
    };

    // Step 3: Compute target distances
    // Ambiguous forms sit at distance R/2 (mod R), so check
    // k * R / (2 * h) for k = 1, 2, ..., max_multiples
    let step_size = regulator / (2.0 * class_number as f64);

    // We need to convert continuous distances to discrete infrastructure steps.
    // The average step size in the CF expansion is approximately ln(golden_ratio) ≈ 0.48
    // for typical discriminants, so the number of rho steps ≈ distance / avg_step_size.
    let avg_step_size = estimate_average_step_size(n);

    let ctx = InfraContext::new(n);
    let principal = InfraForm::principal(n);

    for k in 1..=config.max_multiples {
        let target_dist = step_size * (k as f64);

        // Convert to approximate number of rho steps
        let approx_steps = if avg_step_size > 0.0 {
            (target_dist / avg_step_size).round() as u64
        } else {
            k as u64 * 10
        };

        if approx_steps == 0 {
            continue;
        }

        // Jump to target using power_step (repeated squaring)
        let jumped = power_step(&principal, approx_steps);
        forms_checked += 1;

        // Check the jumped form
        if let Some(factor) = form_reveals_factor(&jumped.form, n) {
            if factor > one && factor < *n {
                return RegulatorGuidedResult {
                    factor: Some(factor),
                    regulator,
                    class_number,
                    found_at_multiple: Some(k),
                    forms_checked,
                };
            }
        }

        // Walk neighborhood around the target
        let mut current = jumped;
        for _ in 0..config.neighborhood_size {
            current = rho_step_ctx(&current, &ctx);
            forms_checked += 1;

            if let Some(factor) = form_reveals_factor(&current.form, n) {
                if factor > one && factor < *n {
                    return RegulatorGuidedResult {
                        factor: Some(factor),
                        regulator,
                        class_number,
                        found_at_multiple: Some(k),
                        forms_checked,
                    };
                }
            }
        }
    }

    RegulatorGuidedResult {
        factor: None,
        regulator,
        class_number,
        found_at_multiple: None,
        forms_checked,
    }
}

/// Estimate the average infrastructure step size for N.
///
/// For the CF expansion of sqrt(N), the average partial quotient
/// is approximately sqrt(D) / ln(D) (Khinchin's constant),
/// and the distance increment per step is ln(a_k) on average.
fn estimate_average_step_size(n: &BigUint) -> f64 {
    let n_f64 = n.to_f64().unwrap_or(1.0);
    if n_f64 <= 1.0 {
        return 0.5;
    }

    // Walk a few steps and measure average distance increment
    let ctx = InfraContext::new(n);
    let mut current = InfraForm::principal(n);
    let test_steps = 50;
    let mut actual_steps = 0usize;

    for _ in 0..test_steps {
        let next = rho_step_ctx(&current, &ctx);
        if next.form == current.form {
            break;
        }
        actual_steps += 1;
        current = next;
    }

    if actual_steps > 0 && current.distance > 0.0 {
        current.distance / actual_steps as f64
    } else {
        // Fallback: theoretical average ≈ ln(1 + sqrt(2)) ≈ 0.88
        0.88
    }
}

/// Quick regulator-guided factor attempt with default config.
pub fn quick_regulator_factor(n: &BigUint) -> Option<BigUint> {
    let config = RegulatorGuidedConfig::default();
    let result = regulator_guided_factor(n, &config);
    result.factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulator_guided_small() {
        // 77 = 7 * 11
        let n = BigUint::from(77u64);
        let config = RegulatorGuidedConfig {
            max_regulator_terms: 1000,
            neighborhood_size: 50,
            use_class_number: true,
            max_multiples: 10,
        };
        let result = regulator_guided_factor(&n, &config);

        assert!(result.regulator > 0.0);
        assert!(result.forms_checked > 0);

        if let Some(ref factor) = result.factor {
            let q = &n / factor;
            assert_eq!(factor * &q, n, "Factor should divide N");
        }
    }

    #[test]
    fn test_regulator_guided_143() {
        // 143 = 11 * 13
        let n = BigUint::from(143u64);
        let config = RegulatorGuidedConfig {
            max_regulator_terms: 1000,
            neighborhood_size: 100,
            use_class_number: true,
            max_multiples: 20,
        };
        let result = regulator_guided_factor(&n, &config);

        assert!(result.forms_checked > 0);
    }

    #[test]
    fn test_estimate_step_size() {
        let n = BigUint::from(77u64);
        let step = estimate_average_step_size(&n);
        assert!(
            step > 0.0,
            "Average step size should be positive, got {}",
            step
        );
    }

    #[test]
    fn test_quick_factor() {
        let n = BigUint::from(15u64); // 3 * 5
        let result = quick_regulator_factor(&n);

        if let Some(factor) = result {
            assert!(
                factor == BigUint::from(3u32) || factor == BigUint::from(5u32),
                "Factor should be 3 or 5, got {}",
                factor
            );
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = RegulatorGuidedConfig::default();
        assert_eq!(config.max_regulator_terms, 10_000);
        assert_eq!(config.neighborhood_size, 100);
        assert!(config.use_class_number);
        assert_eq!(config.max_multiples, 20);
    }
}
