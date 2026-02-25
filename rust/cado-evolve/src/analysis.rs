//! Post-evolution analysis and reporting.
//!
//! Generates reports on:
//! 1. Parameter sensitivity: which params most affect runtime
//! 2. Interaction effects: which param pairs have synergistic effects
//! 3. Scaling curves: evolved vs defaults at each bit size
//! 4. Configuration recommendations per bit size
//! 5. Comparison tables

use serde::{Deserialize, Serialize};

use crate::benchmark::ComparisonResult;
use crate::evolution::ParamIndividual;
use crate::params::CadoParams;

/// Complete analysis report from an evolution run.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Parameter sensitivity ranking.
    pub sensitivity: Vec<ParamSensitivity>,
    /// Per-parameter statistics across the evolved population.
    pub param_statistics: Vec<ParamStatistic>,
    /// Convergence history (best fitness per generation).
    pub convergence_history: Vec<f64>,
    /// Diversity metrics.
    pub diversity: DiversityMetrics,
    /// Comparison against baselines.
    pub comparisons: Vec<ComparisonResult>,
    /// Recommended configuration.
    pub recommendation: Option<ConfigRecommendation>,
}

/// Sensitivity of a single parameter to fitness.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParamSensitivity {
    /// Parameter name.
    pub name: String,
    /// Correlation between parameter value and fitness.
    pub correlation: f64,
    /// Variance of the parameter across the top-K individuals.
    pub top_k_variance: f64,
    /// Whether this parameter converged (low variance in top individuals).
    pub converged: bool,
}

/// Statistics for a single parameter across the population.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParamStatistic {
    /// Parameter name.
    pub name: String,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Value in the best individual.
    pub best_value: f64,
}

/// Diversity metrics for the population.
#[derive(Debug, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Number of unique parameter configurations (by hash).
    pub unique_configs: usize,
    /// Total configurations evaluated.
    pub total_configs: usize,
    /// Diversity ratio (unique / total).
    pub diversity_ratio: f64,
    /// Number of distinct poly_degree values seen.
    pub degree_diversity: usize,
}

/// Recommended configuration with rationale.
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigRecommendation {
    /// The recommended parameters.
    pub params: CadoParams,
    /// Summary string.
    pub summary: String,
    /// Target bit size this recommendation is for.
    pub n_bits: u32,
    /// Expected speedup over defaults.
    pub expected_speedup: f64,
    /// Confidence level (0-1) based on evaluation count and consistency.
    pub confidence: f64,
}

/// Analyze the evolved population and generate a report.
pub fn generate_report(
    top_individuals: &[ParamIndividual],
    convergence_history: &[f64],
    comparisons: &[ComparisonResult],
    n_bits: u32,
) -> AnalysisReport {
    let sensitivity = analyze_sensitivity(top_individuals);
    let param_statistics = compute_param_statistics(top_individuals);
    let diversity = compute_diversity(top_individuals);
    let recommendation = make_recommendation(top_individuals, comparisons, n_bits);

    AnalysisReport {
        sensitivity,
        param_statistics,
        convergence_history: convergence_history.to_vec(),
        diversity,
        comparisons: comparisons.to_vec(),
        recommendation,
    }
}

/// Analyze parameter sensitivity by correlating each parameter with fitness.
fn analyze_sensitivity(individuals: &[ParamIndividual]) -> Vec<ParamSensitivity> {
    if individuals.is_empty() {
        return Vec::new();
    }

    let fitnesses: Vec<f64> = individuals.iter().map(|i| i.fitness).collect();
    let top_k = std::cmp::min(10, individuals.len());
    let mut sorted = individuals.to_vec();
    sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    let top = &sorted[..top_k];

    let param_extractors: Vec<(&str, Box<dyn Fn(&CadoParams) -> f64>)> = vec![
        ("poly_degree", Box::new(|p: &CadoParams| p.poly_degree as f64)),
        ("poly_admax", Box::new(|p: &CadoParams| p.poly_admax as f64)),
        ("poly_incr", Box::new(|p: &CadoParams| p.poly_incr as f64)),
        ("fb_rational_bound", Box::new(|p: &CadoParams| p.fb_rational_bound as f64)),
        ("fb_algebraic_bound", Box::new(|p: &CadoParams| p.fb_algebraic_bound as f64)),
        ("lp_rational_bits", Box::new(|p: &CadoParams| p.lp_rational_bits as f64)),
        ("lp_algebraic_bits", Box::new(|p: &CadoParams| p.lp_algebraic_bits as f64)),
        ("sieve_mfbr", Box::new(|p: &CadoParams| p.sieve_mfbr as f64)),
        ("sieve_mfba", Box::new(|p: &CadoParams| p.sieve_mfba as f64)),
        ("sieve_qrange", Box::new(|p: &CadoParams| p.sieve_qrange as f64)),
    ];

    let mut sensitivities = Vec::new();

    for (name, extractor) in &param_extractors {
        let values: Vec<f64> = individuals.iter().map(|i| extractor(&i.params)).collect();
        let top_values: Vec<f64> = top.iter().map(|i| extractor(&i.params)).collect();

        let correlation = pearson_correlation(&values, &fitnesses);
        let top_k_variance = variance(&top_values);
        let full_variance = variance(&values);

        // Consider converged if top-K variance is <10% of full population variance
        let converged = full_variance > 0.0 && (top_k_variance / full_variance) < 0.1;

        sensitivities.push(ParamSensitivity {
            name: name.to_string(),
            correlation,
            top_k_variance,
            converged,
        });
    }

    // Sort by absolute correlation (most sensitive first)
    sensitivities.sort_by(|a, b| {
        b.correlation
            .abs()
            .partial_cmp(&a.correlation.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    sensitivities
}

/// Compute descriptive statistics for each parameter.
fn compute_param_statistics(individuals: &[ParamIndividual]) -> Vec<ParamStatistic> {
    if individuals.is_empty() {
        return Vec::new();
    }

    let best = individuals
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let extractors: Vec<(&str, Box<dyn Fn(&CadoParams) -> f64>)> = vec![
        ("poly_degree", Box::new(|p: &CadoParams| p.poly_degree as f64)),
        ("poly_admax", Box::new(|p: &CadoParams| p.poly_admax as f64)),
        ("poly_incr", Box::new(|p: &CadoParams| p.poly_incr as f64)),
        ("fb_rational_bound", Box::new(|p: &CadoParams| p.fb_rational_bound as f64)),
        ("fb_algebraic_bound", Box::new(|p: &CadoParams| p.fb_algebraic_bound as f64)),
        ("lp_rational_bits", Box::new(|p: &CadoParams| p.lp_rational_bits as f64)),
        ("lp_algebraic_bits", Box::new(|p: &CadoParams| p.lp_algebraic_bits as f64)),
        ("sieve_mfbr", Box::new(|p: &CadoParams| p.sieve_mfbr as f64)),
        ("sieve_mfba", Box::new(|p: &CadoParams| p.sieve_mfba as f64)),
        ("sieve_qrange", Box::new(|p: &CadoParams| p.sieve_qrange as f64)),
    ];

    let mut stats = Vec::new();

    for (name, extractor) in &extractors {
        let values: Vec<f64> = individuals.iter().map(|i| extractor(&i.params)).collect();
        let mean = mean_of(&values);
        let std_dev = variance(&values).sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let best_value = extractor(&best.params);

        stats.push(ParamStatistic {
            name: name.to_string(),
            mean,
            std_dev,
            min,
            max,
            best_value,
        });
    }

    stats
}

/// Compute diversity metrics for the population.
fn compute_diversity(individuals: &[ParamIndividual]) -> DiversityMetrics {
    let mut hashes = std::collections::HashSet::new();
    let mut degrees = std::collections::HashSet::new();

    for ind in individuals {
        hashes.insert(ind.params.fitness_hash());
        degrees.insert(ind.params.poly_degree);
    }

    let total = individuals.len();
    let unique = hashes.len();

    DiversityMetrics {
        unique_configs: unique,
        total_configs: total,
        diversity_ratio: if total > 0 {
            unique as f64 / total as f64
        } else {
            0.0
        },
        degree_diversity: degrees.len(),
    }
}

/// Generate a configuration recommendation from the analysis.
fn make_recommendation(
    individuals: &[ParamIndividual],
    comparisons: &[ComparisonResult],
    n_bits: u32,
) -> Option<ConfigRecommendation> {
    // Find the best individual
    let best = individuals
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))?;

    // Compute expected speedup from comparisons
    let avg_speedup = if comparisons.is_empty() {
        1.0
    } else {
        comparisons.iter().map(|c| c.speedup).sum::<f64>() / comparisons.len() as f64
    };

    // Confidence based on consistency of top results
    let top_k = std::cmp::min(5, individuals.len());
    let mut sorted = individuals.to_vec();
    sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    let top = &sorted[..top_k];

    let fitness_values: Vec<f64> = top.iter().map(|i| i.fitness).collect();
    let fitness_cv = if mean_of(&fitness_values) > 0.0 {
        variance(&fitness_values).sqrt() / mean_of(&fitness_values)
    } else {
        1.0
    };

    // Low CV = high confidence
    let confidence = (1.0 - fitness_cv).max(0.0).min(1.0);

    let summary = format!(
        "For {}-bit: {}, expected {:.1}x speedup (confidence: {:.0}%)",
        n_bits,
        best.params.summary(),
        avg_speedup,
        confidence * 100.0
    );

    Some(ConfigRecommendation {
        params: best.params.clone(),
        summary,
        n_bits,
        expected_speedup: avg_speedup,
        confidence,
    })
}

/// Print a formatted report summary to stdout.
pub fn print_report_summary(report: &AnalysisReport) {
    println!("╔══════════════════════════════════════════════╗");
    println!("║        CADO-Evolve Analysis Report           ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    // Parameter sensitivity
    println!("Parameter Sensitivity (by correlation with fitness):");
    println!("  {:>22} | {:>10} | {:>12} | {:>9}",
        "Parameter", "Corr.", "Top-K Var.", "Converged");
    println!("  {}", "-".repeat(62));

    for s in &report.sensitivity {
        println!(
            "  {:>22} | {:>10.4} | {:>12.2} | {:>9}",
            s.name,
            s.correlation,
            s.top_k_variance,
            if s.converged { "YES" } else { "no" }
        );
    }
    println!();

    // Parameter statistics
    println!("Parameter Statistics:");
    println!("  {:>22} | {:>12} | {:>10} | {:>12} | {:>12}",
        "Parameter", "Best Value", "Mean", "Std Dev", "Range");
    println!("  {}", "-".repeat(78));

    for s in &report.param_statistics {
        println!(
            "  {:>22} | {:>12.1} | {:>10.1} | {:>12.1} | {:.0}-{:.0}",
            s.name, s.best_value, s.mean, s.std_dev, s.min, s.max
        );
    }
    println!();

    // Diversity
    println!("Population Diversity:");
    println!("  Unique configs: {}/{} ({:.1}%)",
        report.diversity.unique_configs,
        report.diversity.total_configs,
        report.diversity.diversity_ratio * 100.0
    );
    println!("  Degree diversity: {} distinct values", report.diversity.degree_diversity);
    println!();

    // Convergence
    if !report.convergence_history.is_empty() {
        let first = report.convergence_history.first().unwrap();
        let last = report.convergence_history.last().unwrap();
        let improvement = if *first > 0.0 {
            (last - first) / first * 100.0
        } else {
            0.0
        };
        println!("Convergence:");
        println!("  Initial fitness: {:.4}", first);
        println!("  Final fitness:   {:.4}", last);
        println!("  Improvement:     {:.1}%", improvement);
        println!("  Generations:     {}", report.convergence_history.len());
        println!();
    }

    // Comparisons
    if !report.comparisons.is_empty() {
        println!("Comparison vs Defaults:");
        println!("  {:>6} | {:>12} | {:>12} | {:>8}",
            "Bits", "Default(s)", "Evolved(s)", "Speedup");
        println!("  {}", "-".repeat(48));

        for c in &report.comparisons {
            println!(
                "  {:>6} | {:>12.1} | {:>12.1} | {:>7.2}x",
                c.n_bits, c.baseline_median_secs, c.evolved_median_secs, c.speedup
            );
        }
        println!();
    }

    // Recommendation
    if let Some(ref rec) = report.recommendation {
        println!("Recommendation:");
        println!("  {}", rec.summary);
        println!("  Evolved: {}", rec.params);
        println!("  Default: {}", CadoParams::default_for_bits(rec.n_bits));
        println!();
    }
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Variance of a slice.
fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean_of(values);
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// Mean of a slice.
fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_individual(degree: u32, fitness: f64) -> ParamIndividual {
        let mut params = CadoParams::default_for_bits(100);
        params.poly_degree = degree;
        let mut ind = ParamIndividual::new(params);
        ind.fitness = fitness;
        ind
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 7.0, 1.0, 9.0];
        let corr = pearson_correlation(&x, &y);
        // Not exactly zero but should be close to 0 (random-ish)
        assert!(corr.abs() < 1.0);
    }

    #[test]
    fn test_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&values);
        assert!(v > 0.0);
        assert!((v - 4.571).abs() < 0.01); // sample variance
    }

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean_of(&values), 3.0);
    }

    #[test]
    fn test_compute_diversity() {
        let individuals = vec![
            make_individual(3, 1.0),
            make_individual(4, 2.0),
            make_individual(5, 3.0),
            make_individual(3, 4.0),
        ];
        let diversity = compute_diversity(&individuals);
        assert_eq!(diversity.total_configs, 4);
        assert!(diversity.unique_configs >= 2); // At least 2 different hashes
        assert_eq!(diversity.degree_diversity, 3);
    }

    #[test]
    fn test_analyze_sensitivity_empty() {
        let result = analyze_sensitivity(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_analyze_sensitivity() {
        let individuals: Vec<ParamIndividual> = (0..20)
            .map(|i| make_individual(3 + (i % 3), i as f64 * 10.0))
            .collect();

        let result = analyze_sensitivity(&individuals);
        assert_eq!(result.len(), 10); // 10 parameters
    }

    #[test]
    fn test_compute_param_statistics() {
        let individuals = vec![
            make_individual(3, 100.0),
            make_individual(4, 200.0),
            make_individual(5, 300.0),
        ];

        let stats = compute_param_statistics(&individuals);
        assert_eq!(stats.len(), 10);

        // Check poly_degree stats
        let degree_stat = stats.iter().find(|s| s.name == "poly_degree").unwrap();
        assert_eq!(degree_stat.best_value, 5.0);
        assert_eq!(degree_stat.min, 3.0);
        assert_eq!(degree_stat.max, 5.0);
    }

    #[test]
    fn test_make_recommendation() {
        let individuals = vec![
            make_individual(3, 100.0),
            make_individual(4, 200.0),
        ];

        let rec = make_recommendation(&individuals, &[], 100);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert_eq!(rec.n_bits, 100);
        assert_eq!(rec.expected_speedup, 1.0); // No comparisons
    }

    #[test]
    fn test_generate_report() {
        let individuals = vec![
            make_individual(3, 100.0),
            make_individual(4, 200.0),
            make_individual(5, 300.0),
        ];

        let history = vec![100.0, 150.0, 200.0, 300.0];
        let report = generate_report(&individuals, &history, &[], 100);

        assert_eq!(report.sensitivity.len(), 10);
        assert_eq!(report.param_statistics.len(), 10);
        assert_eq!(report.convergence_history.len(), 4);
        assert!(report.recommendation.is_some());
    }
}
