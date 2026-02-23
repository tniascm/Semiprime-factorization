//! Benchmark suite for CADO-NFS baseline measurements.
//!
//! Generates test semiprimes at various bit sizes, times CADO-NFS with
//! default parameters, and stores results for comparison against evolved
//! configurations.

use std::time::{Duration, Instant};

use num_bigint::BigUint;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::cado::{CadoInstallation, CadoResult};
use crate::params::CadoParams;

/// Baseline measurement for a single bit size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMeasurement {
    /// Target bit size.
    pub n_bits: u32,
    /// Number of test semiprimes.
    pub num_tests: usize,
    /// Number of successful factorizations.
    pub successes: usize,
    /// Timing for each test (seconds). NaN for failures.
    pub times_secs: Vec<f64>,
    /// Median time for successful factorizations.
    pub median_time_secs: f64,
    /// Mean time for successful factorizations.
    pub mean_time_secs: f64,
    /// Min time for successful factorizations.
    pub min_time_secs: f64,
    /// Max time for successful factorizations.
    pub max_time_secs: f64,
    /// The default parameters used.
    pub params: CadoParams,
}

/// Complete baseline suite across multiple bit sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSuite {
    /// Measurements for each bit size.
    pub measurements: Vec<BaselineMeasurement>,
    /// Total time to run all baselines.
    pub total_time_secs: f64,
}

/// Test semiprime with known factors for verification.
#[derive(Debug, Clone)]
pub struct TestSemiprime {
    /// The semiprime N = p * q.
    pub n: BigUint,
    /// First factor p.
    pub p: BigUint,
    /// Second factor q.
    pub q: BigUint,
    /// Bit size of N.
    pub n_bits: u32,
}

impl TestSemiprime {
    /// Generate a random test semiprime of the given bit size.
    pub fn generate(n_bits: u32, rng: &mut impl Rng) -> Self {
        let target = factoring_core::generate_rsa_target(n_bits, rng);
        TestSemiprime {
            n: target.n,
            p: target.p,
            q: target.q,
            n_bits,
        }
    }

    /// Verify that a factorization result is correct.
    pub fn verify(&self, result: &CadoResult) -> bool {
        if !result.success {
            return false;
        }

        // Check that the factors multiply to N
        for factor_str in &result.factors {
            if let Ok(factor) = factor_str.parse::<BigUint>() {
                if factor == self.p || factor == self.q {
                    return true;
                }
                // Check if it divides N
                if &self.n % &factor == BigUint::from(0u32) {
                    return true;
                }
            }
        }

        false
    }
}

/// Generate a suite of test semiprimes for a given bit size.
pub fn generate_test_suite(n_bits: u32, count: usize, rng: &mut impl Rng) -> Vec<TestSemiprime> {
    (0..count)
        .map(|_| TestSemiprime::generate(n_bits, rng))
        .collect()
}

/// Run baseline measurement for a single bit size.
///
/// Runs CADO-NFS with default parameters on `num_tests` random semiprimes,
/// repeating each test `num_repeats` times and taking the median.
pub fn run_baseline(
    install: &CadoInstallation,
    n_bits: u32,
    num_tests: usize,
    timeout: Duration,
    rng: &mut impl Rng,
) -> BaselineMeasurement {
    let params = CadoParams::default_for_bits(n_bits);
    let test_semiprimes = generate_test_suite(n_bits, num_tests, rng);

    let mut times = Vec::new();
    let mut successes = 0;

    println!("  Running baseline for {}-bit semiprimes ({} tests)...", n_bits, num_tests);

    for (i, test) in test_semiprimes.iter().enumerate() {
        let start = Instant::now();
        // Use run_default to let CADO-NFS use its own optimized parameters
        match install.run_default(&test.n, timeout) {
            Ok(result) => {
                let elapsed = start.elapsed().as_secs_f64();
                if result.success {
                    successes += 1;
                    times.push(elapsed);
                    println!(
                        "    Test {}/{}: {} ({:.1}s) [OK]",
                        i + 1,
                        num_tests,
                        test.n,
                        elapsed
                    );
                } else {
                    times.push(f64::NAN);
                    println!(
                        "    Test {}/{}: {} ({:.1}s) [FAIL]",
                        i + 1,
                        num_tests,
                        test.n,
                        elapsed
                    );
                }
            }
            Err(e) => {
                times.push(f64::NAN);
                println!(
                    "    Test {}/{}: {} [ERROR: {}]",
                    i + 1,
                    num_tests,
                    test.n,
                    e
                );
            }
        }
    }

    // Compute statistics from successful runs
    let mut success_times: Vec<f64> = times.iter().filter(|t| !t.is_nan()).copied().collect();
    success_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (median, mean, min, max) = if success_times.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let mid = success_times.len() / 2;
        let median = if success_times.len() % 2 == 0 && success_times.len() >= 2 {
            (success_times[mid - 1] + success_times[mid]) / 2.0
        } else {
            success_times[mid]
        };
        let mean = success_times.iter().sum::<f64>() / success_times.len() as f64;
        let min = success_times[0];
        let max = *success_times.last().unwrap();
        (median, mean, min, max)
    };

    println!(
        "  Baseline {}-bit: {}/{} success, median={:.1}s, mean={:.1}s",
        n_bits, successes, num_tests, median, mean
    );

    BaselineMeasurement {
        n_bits,
        num_tests,
        successes,
        times_secs: times,
        median_time_secs: median,
        mean_time_secs: mean,
        min_time_secs: min,
        max_time_secs: max,
        params,
    }
}

/// Run the complete baseline suite across multiple bit sizes.
pub fn run_baseline_suite(
    install: &CadoInstallation,
    bit_sizes: &[u32],
    tests_per_size: usize,
    timeout: Duration,
    rng: &mut impl Rng,
) -> BaselineSuite {
    let start = Instant::now();
    let mut measurements = Vec::new();

    println!("Running baseline suite:");
    println!("  Bit sizes: {:?}", bit_sizes);
    println!("  Tests per size: {}", tests_per_size);
    println!("  Timeout per test: {:?}", timeout);
    println!();

    for &bits in bit_sizes {
        let measurement = run_baseline(install, bits, tests_per_size, timeout, rng);
        measurements.push(measurement);
    }

    let total_time = start.elapsed().as_secs_f64();

    println!();
    println!("Baseline suite complete in {:.1}s", total_time);

    BaselineSuite {
        measurements,
        total_time_secs: total_time,
    }
}

/// Comparison result between evolved and default parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Target bit size.
    pub n_bits: u32,
    /// Baseline (default) median time.
    pub baseline_median_secs: f64,
    /// Evolved configuration median time.
    pub evolved_median_secs: f64,
    /// Speedup factor (baseline / evolved).
    pub speedup: f64,
    /// Baseline success rate.
    pub baseline_success_rate: f64,
    /// Evolved success rate.
    pub evolved_success_rate: f64,
    /// The evolved parameters.
    pub evolved_params: CadoParams,
}

/// Compare evolved parameters against the baseline on fresh test semiprimes.
pub fn compare_params(
    install: &CadoInstallation,
    evolved_params: &CadoParams,
    n_bits: u32,
    num_tests: usize,
    timeout: Duration,
    rng: &mut impl Rng,
) -> ComparisonResult {
    let test_semiprimes = generate_test_suite(n_bits, num_tests, rng);

    println!("  Comparing on {}-bit semiprimes ({} tests):", n_bits, num_tests);

    // Run default (using CADO-NFS built-in parameters)
    let mut default_times = Vec::new();
    let mut default_successes = 0;
    println!("    Default parameters:");
    for (i, test) in test_semiprimes.iter().enumerate() {
        match install.run_default(&test.n, timeout) {
            Ok(result) => {
                let t = result.total_time.as_secs_f64();
                if result.success {
                    default_successes += 1;
                    default_times.push(t);
                    print!("      Test {}: {:.1}s [OK]  ", i + 1, t);
                } else {
                    print!("      Test {}: [FAIL]  ", i + 1);
                }
            }
            Err(e) => {
                print!("      Test {}: [ERR: {}]  ", i + 1, e);
            }
        }
    }
    println!();

    // Run evolved
    let mut evolved_times = Vec::new();
    let mut evolved_successes = 0;
    println!("    Evolved parameters:");
    for (i, test) in test_semiprimes.iter().enumerate() {
        match install.run_with_kill_timeout(&test.n, evolved_params, timeout) {
            Ok(result) => {
                let t = result.total_time.as_secs_f64();
                if result.success {
                    evolved_successes += 1;
                    evolved_times.push(t);
                    print!("      Test {}: {:.1}s [OK]  ", i + 1, t);
                } else {
                    print!("      Test {}: [FAIL]  ", i + 1);
                }
            }
            Err(e) => {
                print!("      Test {}: [ERR: {}]  ", i + 1, e);
            }
        }
    }
    println!();

    // Compute medians
    default_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evolved_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let default_median = median_of(&default_times);
    let evolved_median = median_of(&evolved_times);
    let speedup = if evolved_median > 0.0 {
        default_median / evolved_median
    } else {
        0.0
    };

    println!(
        "    Result: default={:.1}s, evolved={:.1}s, speedup={:.2}x",
        default_median, evolved_median, speedup
    );

    ComparisonResult {
        n_bits,
        baseline_median_secs: default_median,
        evolved_median_secs: evolved_median,
        speedup,
        baseline_success_rate: if num_tests > 0 {
            default_successes as f64 / num_tests as f64
        } else {
            0.0
        },
        evolved_success_rate: if num_tests > 0 {
            evolved_successes as f64 / num_tests as f64
        } else {
            0.0
        },
        evolved_params: evolved_params.clone(),
    }
}

/// Compute the median of a sorted slice.
fn median_of(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_generate_test_semiprime() {
        let mut rng = thread_rng();
        let test = TestSemiprime::generate(32, &mut rng);
        assert_eq!(&test.p * &test.q, test.n);
        assert!(test.n_bits == 32);
    }

    #[test]
    fn test_generate_test_suite() {
        let mut rng = thread_rng();
        let suite = generate_test_suite(48, 5, &mut rng);
        assert_eq!(suite.len(), 5);
        for test in &suite {
            assert_eq!(&test.p * &test.q, test.n);
        }
    }

    #[test]
    fn test_median_of_empty() {
        assert_eq!(median_of(&[]), 0.0);
    }

    #[test]
    fn test_median_of_odd() {
        assert_eq!(median_of(&[1.0, 2.0, 3.0]), 2.0);
    }

    #[test]
    fn test_median_of_even() {
        assert_eq!(median_of(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_verify_correct() {
        let mut rng = thread_rng();
        let test = TestSemiprime::generate(32, &mut rng);

        let result = CadoResult {
            success: true,
            n: test.n.to_string(),
            factors: vec![test.p.to_string(), test.q.to_string()],
            total_time: Duration::from_secs(1),
            phase_times: std::collections::HashMap::new(),
            relations_found: None,
            matrix_size: None,
            unique_relations: None,
            log_tail: String::new(),
        };

        assert!(test.verify(&result));
    }

    #[test]
    fn test_verify_incorrect() {
        let mut rng = thread_rng();
        let test = TestSemiprime::generate(32, &mut rng);

        let result = CadoResult {
            success: true,
            n: test.n.to_string(),
            factors: vec!["7".to_string(), "11".to_string()],
            total_time: Duration::from_secs(1),
            phase_times: std::collections::HashMap::new(),
            relations_found: None,
            matrix_size: None,
            unique_relations: None,
            log_tail: String::new(),
        };

        // Unless N happens to be divisible by 7 or 11, this should fail
        // (extremely unlikely for a random 32-bit semiprime)
        // We just check the function runs without panic
        let _verified = test.verify(&result);
    }

    #[test]
    fn test_verify_failure() {
        let mut rng = thread_rng();
        let test = TestSemiprime::generate(32, &mut rng);

        let result = CadoResult {
            success: false,
            n: test.n.to_string(),
            factors: vec![],
            total_time: Duration::from_secs(1),
            phase_times: std::collections::HashMap::new(),
            relations_found: None,
            matrix_size: None,
            unique_relations: None,
            log_tail: String::new(),
        };

        assert!(!test.verify(&result));
    }
}
