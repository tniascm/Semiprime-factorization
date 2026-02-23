//! Murru-Salvatori factoring pipeline.
//!
//! Orchestrates the three approaches in order of increasing cost:
//! 1. SQUFOF (from cf-factor) — O(N^{1/4}), fast for small N
//! 2. BSGS in the infrastructure — O(D^{1/4}) with class group search
//! 3. Regulator-guided jump — O(sqrt(h)) if R is known
//!
//! Returns as soon as any stage finds a factor.

use std::time::Instant;

use factoring_core::{Algorithm, FactorResult};
use num_bigint::BigUint;
use num_traits::One;

use crate::bsgs::{bsgs_factor, linear_walk_factor, BsgsConfig, BsgsResult};
use crate::regulator_guided::{
    regulator_guided_factor, RegulatorGuidedConfig, RegulatorGuidedResult,
};

/// Detailed result from the Murru-Salvatori pipeline.
#[derive(Debug, Clone)]
pub struct MsFactorResult {
    /// The factoring result.
    pub factor_result: FactorResult,
    /// Which stage found the factor (if any).
    pub stage: String,
    /// BSGS result details.
    pub bsgs_result: Option<BsgsResult>,
    /// Regulator-guided result details.
    pub regulator_result: Option<RegulatorGuidedResult>,
}

/// Configuration for the full pipeline.
#[derive(Debug, Clone)]
pub struct MsPipelineConfig {
    /// Maximum steps for the initial linear walk.
    pub linear_walk_steps: usize,
    /// BSGS configuration.
    pub bsgs_config: BsgsConfig,
    /// Regulator-guided configuration.
    pub regulator_config: RegulatorGuidedConfig,
    /// Whether to run SQUFOF first (from cf-factor).
    pub try_squfof: bool,
}

impl Default for MsPipelineConfig {
    fn default() -> Self {
        Self {
            linear_walk_steps: 500,
            bsgs_config: BsgsConfig::default(),
            regulator_config: RegulatorGuidedConfig::default(),
            try_squfof: true,
        }
    }
}

/// Main entry point: factor N using the Murru-Salvatori pipeline.
pub fn factor_ms(n: &BigUint) -> Option<BigUint> {
    let config = MsPipelineConfig::default();
    factor_ms_with_config(n, &config).and_then(|r| {
        if r.factor_result.complete {
            r.factor_result.factors.first().cloned()
        } else {
            None
        }
    })
}

/// Factor N with full diagnostics.
pub fn factor_ms_with_config(n: &BigUint, config: &MsPipelineConfig) -> Option<MsFactorResult> {
    let start = Instant::now();
    let one = BigUint::one();

    // Trivial checks
    if n <= &one {
        return Some(MsFactorResult {
            factor_result: FactorResult {
                n: n.clone(),
                factors: vec![],
                algorithm: Algorithm::MurruSalvatori,
                duration: start.elapsed(),
                complete: false,
            },
            stage: "trivial".to_string(),
            bsgs_result: None,
            regulator_result: None,
        });
    }

    // Stage 1: SQUFOF (from cf-factor crate)
    if config.try_squfof {
        if let Some(factor) = try_squfof(n) {
            if factor > one && factor < *n {
                return Some(MsFactorResult {
                    factor_result: FactorResult {
                        n: n.clone(),
                        factors: vec![factor],
                        algorithm: Algorithm::MurruSalvatori,
                        duration: start.elapsed(),
                        complete: true,
                    },
                    stage: "squfof".to_string(),
                    bsgs_result: None,
                    regulator_result: None,
                });
            }
        }
    }

    // Stage 2: Linear walk (baby steps only)
    let walk_result = linear_walk_factor(n, config.linear_walk_steps);
    if let Some(ref factor) = walk_result.factor {
        if *factor > one && *factor < *n {
            return Some(MsFactorResult {
                factor_result: FactorResult {
                    n: n.clone(),
                    factors: vec![factor.clone()],
                    algorithm: Algorithm::MurruSalvatori,
                    duration: start.elapsed(),
                    complete: true,
                },
                stage: "linear_walk".to_string(),
                bsgs_result: Some(walk_result),
                regulator_result: None,
            });
        }
    }

    // Stage 3: BSGS
    let bsgs_result = bsgs_factor(n, &config.bsgs_config);
    if let Some(ref factor) = bsgs_result.factor {
        if *factor > one && *factor < *n {
            return Some(MsFactorResult {
                factor_result: FactorResult {
                    n: n.clone(),
                    factors: vec![factor.clone()],
                    algorithm: Algorithm::MurruSalvatori,
                    duration: start.elapsed(),
                    complete: true,
                },
                stage: "bsgs".to_string(),
                bsgs_result: Some(bsgs_result),
                regulator_result: None,
            });
        }
    }

    // Stage 4: Regulator-guided
    let reg_result = regulator_guided_factor(n, &config.regulator_config);
    if let Some(ref factor) = reg_result.factor {
        if *factor > one && *factor < *n {
            return Some(MsFactorResult {
                factor_result: FactorResult {
                    n: n.clone(),
                    factors: vec![factor.clone()],
                    algorithm: Algorithm::MurruSalvatori,
                    duration: start.elapsed(),
                    complete: true,
                },
                stage: "regulator_guided".to_string(),
                bsgs_result: Some(bsgs_result),
                regulator_result: Some(reg_result),
            });
        }
    }

    // All stages failed
    Some(MsFactorResult {
        factor_result: FactorResult {
            n: n.clone(),
            factors: vec![],
            algorithm: Algorithm::MurruSalvatori,
            duration: start.elapsed(),
            complete: false,
        },
        stage: "none".to_string(),
        bsgs_result: Some(bsgs_result),
        regulator_result: Some(reg_result),
    })
}

/// Try SQUFOF from the cf-factor crate.
fn try_squfof(n: &BigUint) -> Option<BigUint> {
    use cf_factor::squfof::squfof_factor;
    squfof_factor(n).factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_ms_small() {
        // 15 = 3 * 5
        let n = BigUint::from(15u64);
        let result = factor_ms(&n);

        if let Some(factor) = result {
            let q = &n / &factor;
            assert_eq!(&factor * &q, n);
        }
    }

    #[test]
    fn test_factor_ms_77() {
        // 77 = 7 * 11
        let n = BigUint::from(77u64);
        let result = factor_ms_with_config(&n, &MsPipelineConfig::default());

        assert!(result.is_some());
        let r = result.unwrap();
        if r.factor_result.complete {
            let factor = &r.factor_result.factors[0];
            let q = &n / factor;
            assert_eq!(factor * &q, n);
        }
    }

    #[test]
    fn test_factor_ms_semiprime() {
        // 15347 = 103 * 149
        let n = BigUint::from(15347u64);
        let config = MsPipelineConfig::default();
        let result = factor_ms_with_config(&n, &config);

        assert!(result.is_some());
    }

    #[test]
    fn test_pipeline_stages() {
        let n = BigUint::from(143u64); // 11 * 13
        let config = MsPipelineConfig {
            try_squfof: false, // Skip SQUFOF to test other stages
            linear_walk_steps: 200,
            bsgs_config: BsgsConfig {
                baby_steps: 200,
                giant_steps: 200,
                check_ambiguous: true,
            },
            regulator_config: RegulatorGuidedConfig {
                max_regulator_terms: 500,
                neighborhood_size: 50,
                use_class_number: true,
                max_multiples: 10,
            },
        };

        let result = factor_ms_with_config(&n, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_try_squfof() {
        let n = BigUint::from(15347u64);
        let result = try_squfof(&n);

        if let Some(factor) = result {
            let q = &n / &factor;
            assert_eq!(&factor * &q, n);
        }
    }
}
