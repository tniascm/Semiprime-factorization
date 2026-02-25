//! Multi-objective fitness evaluation for CADO-NFS parameter configurations.
//!
//! Fitness is computed from CADO-NFS run results with three objectives:
//! - **Speed** (60%): Inverse of total time — faster = better
//! - **Robustness** (25%): Success rate across multiple test semiprimes
//! - **Efficiency** (15%): Relation yield rate (relations / sieve time)
//!
//! Each individual is evaluated on 3-5 random semiprimes of the target bit size.
//! Results are aggregated into a single scalar fitness score.

use std::time::Duration;

use num_bigint::BigUint;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::cado::{CadoError, CadoInstallation};
use crate::evolution::{FitnessCache, ParamIndividual};
use crate::params::CadoParams;

/// Weights for the multi-objective fitness function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessWeights {
    /// Weight for speed objective (default: 0.60).
    pub speed: f64,
    /// Weight for robustness objective (default: 0.25).
    pub robustness: f64,
    /// Weight for efficiency objective (default: 0.15).
    pub efficiency: f64,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        FitnessWeights {
            speed: 0.60,
            robustness: 0.25,
            efficiency: 0.15,
        }
    }
}

/// Configuration for fitness evaluation.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Number of test semiprimes per evaluation.
    pub num_tests: usize,
    /// Target semiprime bit size.
    pub n_bits: u32,
    /// Timeout per individual CADO-NFS run.
    pub timeout: Duration,
    /// Baseline time for the default configuration (for normalization).
    pub baseline_time_secs: f64,
    /// Fitness weights.
    pub weights: FitnessWeights,
}

impl EvalConfig {
    /// Quick evaluation config (fewer tests, shorter timeout).
    pub fn quick(n_bits: u32) -> Self {
        EvalConfig {
            num_tests: 2,
            n_bits,
            timeout: Duration::from_secs(120), // 2 minutes max per run
            baseline_time_secs: 60.0,          // will be updated after baseline measurement
            weights: FitnessWeights::default(),
        }
    }

    /// Full evaluation config.
    pub fn full(n_bits: u32) -> Self {
        EvalConfig {
            num_tests: 5,
            n_bits,
            timeout: Duration::from_secs(1800), // 30 minutes max
            baseline_time_secs: 60.0,
            weights: FitnessWeights::default(),
        }
    }
}

/// Aggregated evaluation result across multiple test semiprimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Combined fitness score.
    pub fitness: f64,
    /// Speed score component (normalized).
    pub speed_score: f64,
    /// Robustness score component (success rate 0-1).
    pub robustness_score: f64,
    /// Efficiency score component (normalized).
    pub efficiency_score: f64,
    /// Number of successful factorizations.
    pub successes: usize,
    /// Total number of tests run.
    pub total_tests: usize,
    /// Average time per successful factorization.
    pub avg_time_secs: f64,
    /// Median time per successful factorization.
    pub median_time_secs: f64,
    /// Individual run results.
    pub run_results: Vec<RunSummary>,
}

/// Summary of a single CADO-NFS run for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Whether factorization succeeded.
    pub success: bool,
    /// Wall-clock time.
    pub time_secs: f64,
    /// Relations found (if available).
    pub relations: Option<u64>,
    /// Sieve time (if available).
    pub sieve_time_secs: Option<f64>,
}

/// Evaluate a parameter configuration on multiple test semiprimes.
///
/// This is the core fitness function. It runs CADO-NFS multiple times
/// with different random semiprimes and aggregates the results.
pub fn evaluate_params(
    install: &CadoInstallation,
    params: &CadoParams,
    test_semiprimes: &[BigUint],
    config: &EvalConfig,
) -> EvalResult {
    let mut run_results = Vec::new();
    let mut success_times = Vec::new();

    for n in test_semiprimes {
        match install.run_with_kill_timeout(n, params, config.timeout) {
            Ok(result) => {
                let time_secs = result.total_time.as_secs_f64();
                let sieve_time = result
                    .phase_times
                    .get("sieve")
                    .map(|d| d.as_secs_f64());

                run_results.push(RunSummary {
                    success: result.success,
                    time_secs,
                    relations: result.relations_found,
                    sieve_time_secs: sieve_time,
                });

                if result.success {
                    success_times.push(time_secs);
                }
            }
            Err(CadoError::Timeout(_)) => {
                run_results.push(RunSummary {
                    success: false,
                    time_secs: config.timeout.as_secs_f64(),
                    relations: None,
                    sieve_time_secs: None,
                });
            }
            Err(e) => {
                log::warn!("CADO-NFS run failed: {}", e);
                run_results.push(RunSummary {
                    success: false,
                    time_secs: config.timeout.as_secs_f64(),
                    relations: None,
                    sieve_time_secs: None,
                });
            }
        }
    }

    let total_tests = run_results.len();
    let successes = success_times.len();

    // Compute speed score: higher when faster
    let avg_time = if success_times.is_empty() {
        config.timeout.as_secs_f64()
    } else {
        success_times.iter().sum::<f64>() / success_times.len() as f64
    };

    let median_time = if success_times.is_empty() {
        config.timeout.as_secs_f64()
    } else {
        success_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = success_times.len() / 2;
        if success_times.len() % 2 == 0 && success_times.len() >= 2 {
            (success_times[mid - 1] + success_times[mid]) / 2.0
        } else {
            success_times[mid]
        }
    };

    // Speed: ratio of baseline time to actual time (>1 means faster than baseline)
    let speed_score = if avg_time > 0.0 {
        (config.baseline_time_secs / avg_time).min(10.0) // cap at 10x
    } else {
        0.0
    };

    // Robustness: success rate
    let robustness_score = if total_tests > 0 {
        successes as f64 / total_tests as f64
    } else {
        0.0
    };

    // Efficiency: average relations per sieve second
    let efficiency_score = compute_efficiency(&run_results, config.baseline_time_secs);

    // Combined fitness
    let fitness = config.weights.speed * speed_score
        + config.weights.robustness * robustness_score
        + config.weights.efficiency * efficiency_score;

    EvalResult {
        fitness,
        speed_score,
        robustness_score,
        efficiency_score,
        successes,
        total_tests,
        avg_time_secs: avg_time,
        median_time_secs: median_time,
        run_results,
    }
}

/// Compute efficiency score from run results.
///
/// Efficiency = average (relations / sieve_time) normalized against baseline.
fn compute_efficiency(runs: &[RunSummary], baseline_time: f64) -> f64 {
    let mut yields = Vec::new();

    for run in runs {
        if run.success {
            if let (Some(rels), Some(sieve_t)) = (run.relations, run.sieve_time_secs) {
                if sieve_t > 0.0 {
                    yields.push(rels as f64 / sieve_t);
                }
            }
        }
    }

    if yields.is_empty() {
        return 0.0;
    }

    let avg_yield = yields.iter().sum::<f64>() / yields.len() as f64;

    // Normalize: assume baseline yields ~1000 rels/sec (rough estimate)
    // The actual normalization will depend on the baseline measurement
    let baseline_yield = 1000.0 / baseline_time.max(1.0) * 1000.0;
    (avg_yield / baseline_yield.max(1.0)).min(5.0) // cap at 5x
}

/// Evaluate a single individual, using cache if available.
///
/// Returns the fitness score and optionally updates the individual.
pub fn evaluate_individual_cached(
    install: &CadoInstallation,
    individual: &mut ParamIndividual,
    test_semiprimes: &[BigUint],
    config: &EvalConfig,
    cache: &mut FitnessCache,
) -> f64 {
    // Check cache first
    if let Some(cached_fitness) = cache.get(&individual.params) {
        individual.fitness = cached_fitness;
        return cached_fitness;
    }

    // Run evaluation
    let result = evaluate_params(install, &individual.params, test_semiprimes, config);
    individual.fitness = result.fitness;

    // Cache the result
    cache.insert(&individual.params, result.fitness);

    result.fitness
}

/// Generate random test semiprimes of the given bit size.
///
/// Each semiprime is a product of two primes of roughly equal size.
pub fn generate_test_semiprimes(n_bits: u32, count: usize, rng: &mut impl Rng) -> Vec<BigUint> {
    (0..count)
        .map(|_| factoring_core::generate_rsa_target(n_bits, rng).n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_weights_default() {
        let w = FitnessWeights::default();
        assert!((w.speed + w.robustness + w.efficiency - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_config_quick() {
        let config = EvalConfig::quick(100);
        assert_eq!(config.num_tests, 2);
        assert_eq!(config.n_bits, 100);
    }

    #[test]
    fn test_eval_config_full() {
        let config = EvalConfig::full(120);
        assert_eq!(config.num_tests, 5);
        assert_eq!(config.n_bits, 120);
    }

    #[test]
    fn test_generate_test_semiprimes() {
        let mut rng = rand::thread_rng();
        let semiprimes = generate_test_semiprimes(32, 5, &mut rng);
        assert_eq!(semiprimes.len(), 5);
        for n in &semiprimes {
            assert!(n.bits() >= 30); // Should be close to 32 bits
        }
    }

    #[test]
    fn test_compute_efficiency_no_data() {
        let runs = vec![RunSummary {
            success: false,
            time_secs: 10.0,
            relations: None,
            sieve_time_secs: None,
        }];
        let eff = compute_efficiency(&runs, 60.0);
        assert_eq!(eff, 0.0);
    }

    #[test]
    fn test_compute_efficiency_with_data() {
        let runs = vec![RunSummary {
            success: true,
            time_secs: 10.0,
            relations: Some(50000),
            sieve_time_secs: Some(8.0),
        }];
        let eff = compute_efficiency(&runs, 60.0);
        assert!(eff > 0.0);
    }

    #[test]
    fn test_run_summary_serialization() {
        let summary = RunSummary {
            success: true,
            time_secs: 12.5,
            relations: Some(45000),
            sieve_time_secs: Some(8.3),
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_eval_result_fitness_calculation() {
        let result = EvalResult {
            fitness: 0.0,
            speed_score: 2.0,       // 2x faster than baseline
            robustness_score: 1.0,  // 100% success
            efficiency_score: 1.5,  // 1.5x efficient
            successes: 3,
            total_tests: 3,
            avg_time_secs: 30.0,
            median_time_secs: 28.0,
            run_results: vec![],
        };

        let weights = FitnessWeights::default();
        let expected = weights.speed * result.speed_score
            + weights.robustness * result.robustness_score
            + weights.efficiency * result.efficiency_score;

        assert!((expected - (0.6 * 2.0 + 0.25 * 1.0 + 0.15 * 1.5)).abs() < 0.001);
    }
}
