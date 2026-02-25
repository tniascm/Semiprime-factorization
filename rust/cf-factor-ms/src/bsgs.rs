//! Baby-step/giant-step (BSGS) factoring in the class group infrastructure.
//!
//! Murru-Salvatori approach: search for ambiguous forms (which reveal factors)
//! by walking the infrastructure with baby steps and jumping with giant steps.
//! Complexity: O(D^{1/4}) where D = 4N (the discriminant).
//!
//! The idea: the class group of Q(sqrt(N)) has order h (the class number).
//! The regulator R is the circumference of the infrastructure cycle.
//! We search for a form f such that f is ambiguous (f = f^{-1}), which
//! means f^2 = identity, so the order of f divides 2.
//!
//! BSGS: store baby steps {g^i : i = 0..m} in a hash table, then check
//! giant steps g^{j*m} for collisions. A collision g^i = g^{j*m} means
//! g^{i - j*m} = identity, giving the group order.

use num_bigint::BigUint;
use num_traits::One;

use crate::infrastructure::{
    form_reveals_factor, giant_step, power_step, rho_step_ctx, walk_infrastructure, InfraContext,
    InfraForm, InfraHashTable,
};

/// Configuration for the BSGS search.
#[derive(Debug, Clone)]
pub struct BsgsConfig {
    /// Number of baby steps to take.
    pub baby_steps: usize,
    /// Number of giant steps to take.
    pub giant_steps: usize,
    /// Whether to also check each baby/giant step for ambiguous forms.
    pub check_ambiguous: bool,
}

impl Default for BsgsConfig {
    fn default() -> Self {
        Self {
            baby_steps: 3000,
            giant_steps: 3000,
            check_ambiguous: true,
        }
    }
}

/// Result from BSGS factoring.
#[derive(Debug, Clone)]
pub struct BsgsResult {
    /// Factor found (if any).
    pub factor: Option<BigUint>,
    /// Number of baby steps taken.
    pub baby_steps_taken: usize,
    /// Number of giant steps taken.
    pub giant_steps_taken: usize,
    /// Number of collisions found.
    pub collisions: usize,
    /// Whether the factor came from an ambiguous form.
    pub from_ambiguous: bool,
}

/// Compute BSGS step sizes from N.
///
/// For discriminant D = 4N, the class number h satisfies h*R ~ sqrt(D).
/// The search space is O(h) or O(R), both ~ N^{1/4} for typical semiprimes.
pub fn compute_bsgs_size(n: &BigUint) -> (usize, usize) {
    let bits = n.bits() as f64;
    // sqrt(N^{1/4}) = N^{1/8}, but we want O(N^{1/4}) total
    // Baby steps: m ~ N^{1/8}, Giant steps: ~ N^{1/8}
    // For practical sizes: m = 2^(bits/8) capped at reasonable limits
    let m = (2.0f64.powf(bits / 8.0)).ceil() as usize;
    let m = m.max(100).min(100_000);
    (m, m)
}

/// Baby-step/giant-step factoring via class group search.
///
/// Algorithm:
/// 1. Start from the principal form f0.
/// 2. Baby steps: compute f0, rho(f0), rho^2(f0), ..., rho^m(f0)
///    and store in hash table.
/// 3. Compute giant step size G = rho^m(f0) (m baby steps from identity).
/// 4. Giant steps: compute G, G^2, G^3, ..., G^k
///    and check for collisions with baby step table.
/// 5. A collision means we've found a form whose order divides some value,
///    potentially revealing a factor.
///
/// Additionally, check each form along the way for ambiguous forms
/// that directly reveal factors.
pub fn bsgs_factor(n: &BigUint, config: &BsgsConfig) -> BsgsResult {
    let one = BigUint::one();
    let mut baby_steps_taken = 0;
    let mut giant_steps_taken = 0;
    let mut collisions = 0;

    // Phase 1: Baby steps — walk the infrastructure and store forms
    let ctx = InfraContext::new(n);
    let mut table = InfraHashTable::new();
    let mut current = InfraForm::principal(n);
    table.insert(current.clone());
    baby_steps_taken += 1;

    for _ in 1..config.baby_steps {
        // Check for ambiguous form
        if config.check_ambiguous {
            if let Some(factor) = form_reveals_factor(&current.form, n) {
                if factor > one && factor < *n {
                    return BsgsResult {
                        factor: Some(factor),
                        baby_steps_taken,
                        giant_steps_taken,
                        collisions,
                        from_ambiguous: true,
                    };
                }
            }
        }

        current = rho_step_ctx(&current, &ctx);
        table.insert(current.clone());
        baby_steps_taken += 1;
    }

    // Phase 2: Giant steps — powers of the form at distance m
    // The "giant step" form is the form at distance m (baby_steps worth of rho steps)
    let giant_form = current.clone(); // This is rho^m(principal)

    let mut giant_current = giant_form.clone();

    for _ in 0..config.giant_steps {
        giant_steps_taken += 1;

        // Check for ambiguous form
        if config.check_ambiguous {
            if let Some(factor) = form_reveals_factor(&giant_current.form, n) {
                if factor > one && factor < *n {
                    return BsgsResult {
                        factor: Some(factor),
                        baby_steps_taken,
                        giant_steps_taken,
                        collisions,
                        from_ambiguous: true,
                    };
                }
            }
        }

        // Check for collision with baby step table
        if let Some(matches) = table.lookup(&giant_current.form) {
            collisions += 1;
            // Collision: giant at distance D_g matches baby at distance D_b.
            // The period P = D_g - D_b is a multiple of the regulator R.
            // Ambiguous forms (which reveal factors) sit at distance R/2 (mod R).
            // Navigate to P/2 and check for factors.
            for baby_match in matches {
                let period_dist = giant_current.distance - baby_match.distance;
                if period_dist <= 0.0 {
                    continue;
                }

                // Use the generator (one rho-step form) for power_step jumping
                let gen = rho_step_ctx(&InfraForm::principal(n), &ctx);
                let gen_dist = gen.distance;
                if gen_dist <= 0.0 {
                    continue;
                }

                // Try several divisors of the period: P/2, P/3, P/4, P/6
                // Ambiguous forms appear at R/2, and P might be kR for some k
                for divisor in &[2u64, 3, 4, 6] {
                    let target = period_dist / (*divisor as f64);
                    let target_power = (target / gen_dist).round() as u64;
                    if target_power == 0 {
                        continue;
                    }

                    let jumped = power_step(&gen, target_power);

                    // Check the jumped form and its neighborhood
                    if let Some(factor) = form_reveals_factor(&jumped.form, n) {
                        if factor > one && factor < *n {
                            return BsgsResult {
                                factor: Some(factor),
                                baby_steps_taken,
                                giant_steps_taken,
                                collisions,
                                from_ambiguous: false,
                            };
                        }
                    }

                    // Walk a small neighborhood (distance drift from power_step approximation)
                    let mut walker = jumped;
                    for _ in 0..200 {
                        walker = rho_step_ctx(&walker, &ctx);
                        if let Some(factor) = form_reveals_factor(&walker.form, n) {
                            if factor > one && factor < *n {
                                return BsgsResult {
                                    factor: Some(factor),
                                    baby_steps_taken,
                                    giant_steps_taken,
                                    collisions,
                                    from_ambiguous: false,
                                };
                            }
                        }
                    }
                }
            }
        }

        // Advance giant step: compose with giant_form
        giant_current = giant_step(&giant_current, &giant_form);
    }

    BsgsResult {
        factor: None,
        baby_steps_taken,
        giant_steps_taken,
        collisions,
        from_ambiguous: false,
    }
}

/// Simplified BSGS: just walk the infrastructure checking for factors.
///
/// This is the "baby step only" variant — no giant steps, just linear walk.
/// Useful as a baseline and for small N.
pub fn linear_walk_factor(n: &BigUint, max_steps: usize) -> BsgsResult {
    let one = BigUint::one();
    let forms = walk_infrastructure(n, max_steps);

    for (i, f) in forms.iter().enumerate() {
        if let Some(factor) = form_reveals_factor(&f.form, n) {
            if factor > one && factor < *n {
                return BsgsResult {
                    factor: Some(factor),
                    baby_steps_taken: i + 1,
                    giant_steps_taken: 0,
                    collisions: 0,
                    from_ambiguous: true,
                };
            }
        }
    }

    BsgsResult {
        factor: None,
        baby_steps_taken: forms.len(),
        giant_steps_taken: 0,
        collisions: 0,
        from_ambiguous: false,
    }
}

/// Helper: convert QuadForm to InfraForm with zero distance (for inverse lookup).
impl From<cf_factor::forms::QuadForm> for InfraForm {
    fn from(form: cf_factor::forms::QuadForm) -> Self {
        InfraForm {
            form,
            distance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bsgs_size() {
        let n = BigUint::from(15347u64);
        let (baby, giant) = compute_bsgs_size(&n);
        assert!(baby >= 100);
        assert!(giant >= 100);
    }

    #[test]
    fn test_linear_walk_small() {
        // 77 = 7 * 11
        let n = BigUint::from(77u64);
        let result = linear_walk_factor(&n, 500);

        if let Some(ref factor) = result.factor {
            let q = &n / factor;
            assert_eq!(factor * &q, n, "Factor should divide N");
        }
    }

    #[test]
    fn test_bsgs_small() {
        // 143 = 11 * 13
        let n = BigUint::from(143u64);
        let config = BsgsConfig {
            baby_steps: 200,
            giant_steps: 200,
            check_ambiguous: true,
        };
        let result = bsgs_factor(&n, &config);

        if let Some(ref factor) = result.factor {
            let q = &n / factor;
            assert_eq!(factor * &q, n, "Factor should divide N");
        }
    }

    #[test]
    fn test_bsgs_semiprime() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let config = BsgsConfig {
            baby_steps: 500,
            giant_steps: 500,
            check_ambiguous: true,
        };
        let result = bsgs_factor(&n, &config);

        // May or may not factor in this many steps — just verify it runs
        assert!(result.baby_steps_taken > 0);
    }

    #[test]
    fn test_bsgs_default_config() {
        let config = BsgsConfig::default();
        assert_eq!(config.baby_steps, 3000);
        assert_eq!(config.giant_steps, 3000);
        assert!(config.check_ambiguous);
    }
}
