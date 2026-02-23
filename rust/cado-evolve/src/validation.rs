//! Statistical validation for evolved CADO-NFS parameter configurations.
//!
//! Provides rigorous A/B testing between default and evolved parameters:
//! - Paired comparison on identical composites
//! - Welch's t-test for significance
//! - Confidence intervals on speedup
//! - Percentile reporting (p50, p75, p90, p95)
//! - Cross-size transfer testing

use std::time::Duration;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::benchmark::generate_test_suite;
use crate::cado::CadoInstallation;
use crate::params::CadoParams;

/// Result of a single paired trial: same composite, default vs evolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedTrial {
    /// The composite (as string for serialization).
    pub n: String,
    /// Bit size of the composite.
    pub n_bits: u32,
    /// Default config time (seconds), None if failed.
    pub default_time: Option<f64>,
    /// Evolved config time (seconds), None if failed.
    pub evolved_time: Option<f64>,
    /// Per-trial speedup (default/evolved), None if either failed.
    pub speedup: Option<f64>,
}

/// Percentile statistics for a set of times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl Percentiles {
    /// Compute percentiles from a slice of times (need not be sorted).
    pub fn from_times(times: &[f64]) -> Self {
        if times.is_empty() {
            return Percentiles {
                p50: 0.0, p75: 0.0, p90: 0.0, p95: 0.0,
                mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0, count: 0,
            };
        }

        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        Percentiles {
            p50: percentile_of(&sorted, 0.50),
            p75: percentile_of(&sorted, 0.75),
            p90: percentile_of(&sorted, 0.90),
            p95: percentile_of(&sorted, 0.95),
            mean,
            std_dev: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            count: n,
        }
    }
}

/// Compute a given percentile from a sorted slice using linear interpolation.
fn percentile_of(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Result of Welch's t-test comparing two sample means.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    /// t-statistic.
    pub t_statistic: f64,
    /// Degrees of freedom (Welch-Satterthwaite).
    pub degrees_of_freedom: f64,
    /// Two-sided p-value (approximate).
    pub p_value: f64,
    /// Whether the difference is significant at alpha=0.05.
    pub significant_at_05: bool,
    /// 95% confidence interval for (mean_default - mean_evolved).
    pub ci_95_low: f64,
    /// 95% confidence interval for (mean_default - mean_evolved).
    pub ci_95_high: f64,
}

/// Perform Welch's t-test on two independent samples.
///
/// Tests H0: mean(a) == mean(b) vs H1: mean(a) != mean(b).
pub fn welch_t_test(a: &[f64], b: &[f64]) -> TTestResult {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return TTestResult {
            t_statistic: 0.0,
            degrees_of_freedom: 0.0,
            p_value: 1.0,
            significant_at_05: false,
            ci_95_low: f64::NEG_INFINITY,
            ci_95_high: f64::INFINITY,
        };
    }

    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;

    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    if se < 1e-12 {
        // Essentially no variance — can't compute meaningful t
        return TTestResult {
            t_statistic: 0.0,
            degrees_of_freedom: n_a + n_b - 2.0,
            p_value: 1.0,
            significant_at_05: false,
            ci_95_low: mean_a - mean_b,
            ci_95_high: mean_a - mean_b,
        };
    }

    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom > 0.0 { num / denom } else { n_a + n_b - 2.0 };

    // Approximate p-value using the normal distribution for large df,
    // or a crude t-distribution approximation for small df.
    let p_value = approximate_two_sided_p(t.abs(), df);

    // 95% CI for (mean_a - mean_b)
    let t_crit = approximate_t_critical(0.025, df);
    let diff = mean_a - mean_b;

    TTestResult {
        t_statistic: t,
        degrees_of_freedom: df,
        p_value,
        significant_at_05: p_value < 0.05,
        ci_95_low: diff - t_crit * se,
        ci_95_high: diff + t_crit * se,
    }
}

/// Approximate two-sided p-value for t-distribution.
///
/// Uses the normal approximation for df >= 30, and a crude
/// adjustment for smaller df.
fn approximate_two_sided_p(t_abs: f64, df: f64) -> f64 {
    // For large df, t ≈ N(0,1)
    // Use the complementary error function approximation
    let z = if df >= 30.0 {
        t_abs
    } else {
        // Adjust t for small df: t * sqrt(df/(df-2)) -> z
        // This is a rough correction
        t_abs * (1.0 - 1.0 / (4.0 * df)).max(0.5)
    };

    // Approximate 2 * P(Z > z) using the Abramowitz & Stegun formula
    let p = 2.0 * normal_sf(z);
    p.clamp(0.0, 1.0)
}

/// Approximate critical value for t-distribution at significance level alpha.
fn approximate_t_critical(alpha: f64, df: f64) -> f64 {
    // For df >= 30, use z-critical
    // For smaller df, apply an inflation factor
    let z = normal_quantile(1.0 - alpha);
    if df >= 30.0 {
        z
    } else {
        // Rough correction: t_crit ≈ z * (1 + 1/(4*df))
        z * (1.0 + 1.0 / (4.0 * df))
    }
}

/// Standard normal survival function: P(Z > z).
/// Abramowitz & Stegun approximation 26.2.17.
fn normal_sf(z: f64) -> f64 {
    if z < 0.0 {
        return 1.0 - normal_sf(-z);
    }
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937
                    + t * (-1.821255978
                        + t * 1.330274429)))));
    p.clamp(0.0, 1.0)
}

/// Approximate quantile of standard normal (inverse CDF).
/// Rational approximation, accurate to ~4.5e-4.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-10 { return 0.0; }

    if p < 0.5 {
        return -normal_quantile(1.0 - p);
    }

    // Rational approximation for 0.5 < p < 1
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

/// Complete validation result for one bit size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Target bit size.
    pub n_bits: u32,
    /// Number of composites tested.
    pub num_composites: usize,
    /// Individual paired trials.
    pub trials: Vec<PairedTrial>,
    /// Default parameter percentiles.
    pub default_stats: Percentiles,
    /// Evolved parameter percentiles.
    pub evolved_stats: Percentiles,
    /// Welch's t-test on the times.
    pub t_test: TTestResult,
    /// Speedup statistics (from paired ratios).
    pub speedup_stats: Percentiles,
    /// Mean speedup across all paired trials.
    pub mean_speedup: f64,
    /// Default success rate.
    pub default_success_rate: f64,
    /// Evolved success rate.
    pub evolved_success_rate: f64,
    /// Total wall-clock time for this validation.
    pub total_time_secs: f64,
}

/// Run a rigorous paired A/B validation.
///
/// For each of `num_composites` random semiprimes:
///   1. Factor with default CADO-NFS params
///   2. Factor with evolved params
///   3. Record paired times
///
/// Same composite is used for both configs to reduce variance.
pub fn run_validation(
    install: &CadoInstallation,
    evolved_params: &CadoParams,
    n_bits: u32,
    num_composites: usize,
    timeout: Duration,
    rng: &mut impl Rng,
) -> ValidationResult {
    let start = std::time::Instant::now();
    let test_semiprimes = generate_test_suite(n_bits, num_composites, rng);

    println!(
        "  Validation A/B: {}-bit, {} composites, timeout {}s",
        n_bits, num_composites, timeout.as_secs()
    );

    let mut trials = Vec::new();
    let mut default_times = Vec::new();
    let mut evolved_times = Vec::new();
    let mut default_successes = 0usize;
    let mut evolved_successes = 0usize;

    for (i, test) in test_semiprimes.iter().enumerate() {
        print!("    #{:>2}/{}: ", i + 1, num_composites);

        // Run default
        let default_time = match install.run_default(&test.n, timeout) {
            Ok(result) if result.success => {
                let t = result.total_time.as_secs_f64();
                default_successes += 1;
                default_times.push(t);
                print!("default={:.1}s ", t);
                Some(t)
            }
            Ok(_) => { print!("default=FAIL "); None }
            Err(e) => { print!("default=ERR({}) ", e); None }
        };

        // Run evolved
        let evolved_time = match install.run_with_kill_timeout(&test.n, evolved_params, timeout) {
            Ok(result) if result.success => {
                let t = result.total_time.as_secs_f64();
                evolved_successes += 1;
                evolved_times.push(t);
                print!("evolved={:.1}s ", t);
                Some(t)
            }
            Ok(_) => { print!("evolved=FAIL "); None }
            Err(e) => { print!("evolved=ERR({}) ", e); None }
        };

        let speedup = match (default_time, evolved_time) {
            (Some(d), Some(e)) if e > 0.0 => {
                let s = d / e;
                print!("speedup={:.2}x", s);
                Some(s)
            }
            _ => { print!("speedup=N/A"); None }
        };

        println!();

        trials.push(PairedTrial {
            n: test.n.to_string(),
            n_bits,
            default_time,
            evolved_time,
            speedup,
        });
    }

    let total_time = start.elapsed().as_secs_f64();

    // Compute statistics
    let default_stats = Percentiles::from_times(&default_times);
    let evolved_stats = Percentiles::from_times(&evolved_times);
    let t_test = welch_t_test(&default_times, &evolved_times);

    let speedups: Vec<f64> = trials.iter().filter_map(|t| t.speedup).collect();
    let speedup_stats = Percentiles::from_times(&speedups);
    let mean_speedup = if speedups.is_empty() {
        0.0
    } else {
        speedups.iter().sum::<f64>() / speedups.len() as f64
    };

    // Print summary
    println!();
    println!("  --- Validation Summary ({}-bit) ---", n_bits);
    println!("  Composites: {}, Timeout: {}s", num_composites, timeout.as_secs());
    println!();
    println!("  Default:  p50={:.1}s  p90={:.1}s  mean={:.1}s ± {:.1}s  ({}/{})",
        default_stats.p50, default_stats.p90,
        default_stats.mean, default_stats.std_dev,
        default_successes, num_composites);
    println!("  Evolved:  p50={:.1}s  p90={:.1}s  mean={:.1}s ± {:.1}s  ({}/{})",
        evolved_stats.p50, evolved_stats.p90,
        evolved_stats.mean, evolved_stats.std_dev,
        evolved_successes, num_composites);
    println!();
    println!("  Speedup:  mean={:.3}x  p50={:.3}x  p90={:.3}x  range=[{:.2}x, {:.2}x]",
        mean_speedup, speedup_stats.p50, speedup_stats.p90,
        speedup_stats.min, speedup_stats.max);
    println!();
    println!("  t-test:   t={:.3}, df={:.1}, p={:.4} {}",
        t_test.t_statistic, t_test.degrees_of_freedom, t_test.p_value,
        if t_test.significant_at_05 { "*** SIGNIFICANT ***" } else { "(not significant)" });
    println!("  95% CI for time difference: [{:.2}s, {:.2}s]",
        t_test.ci_95_low, t_test.ci_95_high);
    println!("  Total validation time: {:.1}s", total_time);

    ValidationResult {
        n_bits,
        num_composites,
        trials,
        default_stats,
        evolved_stats,
        t_test,
        speedup_stats,
        mean_speedup,
        default_success_rate: default_successes as f64 / num_composites as f64,
        evolved_success_rate: evolved_successes as f64 / num_composites as f64,
        total_time_secs: total_time,
    }
}

/// Cross-size transfer test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferTestResult {
    /// The evolved parameters being tested.
    pub evolved_params: CadoParams,
    /// Original bit size the params were evolved for.
    pub source_bits: u32,
    /// Validation results for each target bit size.
    pub results: Vec<ValidationResult>,
}

/// Test whether evolved params transfer to other bit sizes.
///
/// Runs paired A/B validation at each target bit size.
pub fn run_transfer_test(
    install: &CadoInstallation,
    evolved_params: &CadoParams,
    source_bits: u32,
    target_bit_sizes: &[u32],
    composites_per_size: usize,
    timeout: Duration,
    rng: &mut impl Rng,
) -> TransferTestResult {
    println!("=== Transfer Test ===");
    println!("  Source: {}-bit evolved params", source_bits);
    println!("  Targets: {:?}", target_bit_sizes);
    println!("  Composites per size: {}", composites_per_size);
    println!();

    let mut results = Vec::new();

    for &bits in target_bit_sizes {
        println!("--- {}-bit ---", bits);
        let result = run_validation(
            install,
            evolved_params,
            bits,
            composites_per_size,
            timeout,
            rng,
        );
        results.push(result);
        println!();
    }

    // Print transfer summary
    println!("=== Transfer Summary ===");
    println!("  {:>6} | {:>8} | {:>8} | {:>8} | {:>10} | {:>8}",
        "Bits", "Default", "Evolved", "Speedup", "p-value", "Signif.");
    println!("  {}", "-".repeat(65));

    for r in &results {
        println!("  {:>6} | {:>7.1}s | {:>7.1}s | {:>7.3}x | {:>10.4} | {:>8}",
            r.n_bits,
            r.default_stats.p50,
            r.evolved_stats.p50,
            r.mean_speedup,
            r.t_test.p_value,
            if r.t_test.significant_at_05 { "YES" } else { "no" });
    }
    println!();

    TransferTestResult {
        evolved_params: evolved_params.clone(),
        source_bits,
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentiles_empty() {
        let p = Percentiles::from_times(&[]);
        assert_eq!(p.count, 0);
        assert_eq!(p.p50, 0.0);
    }

    #[test]
    fn test_percentiles_single() {
        let p = Percentiles::from_times(&[5.0]);
        assert_eq!(p.count, 1);
        assert_eq!(p.p50, 5.0);
        assert_eq!(p.p90, 5.0);
        assert_eq!(p.mean, 5.0);
        assert_eq!(p.std_dev, 0.0);
    }

    #[test]
    fn test_percentiles_multiple() {
        let times: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p = Percentiles::from_times(&times);
        assert_eq!(p.count, 100);
        assert!((p.p50 - 50.0).abs() < 1.0);
        assert!((p.p90 - 90.0).abs() < 1.0);
        assert!((p.p95 - 95.0).abs() < 1.0);
        assert_eq!(p.min, 1.0);
        assert_eq!(p.max, 100.0);
    }

    #[test]
    fn test_percentile_of_sorted() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_of(&sorted, 0.5) - 3.0).abs() < 0.01);
        assert!((percentile_of(&sorted, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile_of(&sorted, 1.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_welch_t_test_same_samples() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let t = welch_t_test(&a, &a);
        assert!(t.t_statistic.abs() < 0.01);
        assert!(t.p_value > 0.9);
        assert!(!t.significant_at_05);
    }

    #[test]
    fn test_welch_t_test_different_samples() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8, 10.3, 10.7, 10.4, 10.6, 10.1];
        let b = vec![8.0, 8.5, 8.2, 8.8, 8.1, 8.3, 8.7, 8.4, 8.6, 8.9];
        let t = welch_t_test(&a, &b);
        // Mean diff is about 2.0, should be significant
        assert!(t.t_statistic > 1.0);
        assert!(t.significant_at_05);
        assert!(t.ci_95_low > 0.0); // diff is positive (a > b)
    }

    #[test]
    fn test_welch_t_test_insufficient_data() {
        let a = vec![10.0];
        let b = vec![8.0];
        let t = welch_t_test(&a, &b);
        // Not enough data for meaningful test
        assert_eq!(t.p_value, 1.0);
        assert!(!t.significant_at_05);
    }

    #[test]
    fn test_normal_sf_symmetry() {
        // P(Z > 0) ≈ 0.5
        assert!((normal_sf(0.0) - 0.5).abs() < 0.01);
        // P(Z > 1.96) ≈ 0.025
        assert!((normal_sf(1.96) - 0.025).abs() < 0.005);
        // P(Z > -1.96) ≈ 0.975
        assert!((normal_sf(-1.96) - 0.975).abs() < 0.005);
    }

    #[test]
    fn test_normal_quantile_standard() {
        // Inverse of the above
        assert!((normal_quantile(0.5)).abs() < 0.01);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.05);
        assert!((normal_quantile(0.025) + 1.96).abs() < 0.05);
    }
}
