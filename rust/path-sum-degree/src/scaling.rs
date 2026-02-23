/// Degree scaling analysis across bit sizes.
///
/// For each (channel, n_bits, degree), we record the maximum |correlation|
/// found among randomly sampled degree-d monomials.  Then we fit a power law
/// deg_lower_bound(n) ≈ a · n^b to find the scaling exponent b.
///
/// Interpretation:
///   b ≈ 0       → degree bounded by a constant  (Amy-Stinchcombe regime: poly-time)
///   b ≈ 1/3     → degree grows as n^{1/3}       (matches GNFS exponent)
///   b ≈ 1/2     → degree grows as n^{1/2}       (random function baseline)
///   b ≈ 1       → degree grows linearly with n  (hard: no improvement over brute force)

use crate::degree::{
    run_correlation_scan, run_fit_degree, real_spectrum,
    CorrelationResult, CrtRankResult, FitDegreeResult, RealSpectrum,
};
use eisenstein_hunt::{Channel, CHANNELS};
use rayon::prelude::*;
use serde::Serialize;

/// Configuration for a scaling scan.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Bit sizes to scan.
    pub bit_sizes: Vec<u32>,
    /// Maximum degree to test for each bit size.
    pub max_degree: u32,
    /// Number of balanced semiprimes per (n_bits, channel).
    pub n_semiprimes: usize,
    /// Number of random monomials to sample per (n_bits, degree, channel).
    pub n_monomials: usize,
    /// Whether to also run CRT rank + real spectrum (for n_bits ≤ 24).
    pub run_crt_rank: bool,
    /// Number of permutation simulations for the calibrated null threshold.
    /// Set to 0 to use the old fixed 3/√m threshold.
    pub n_null_sims: usize,
    /// Quantile for the calibrated threshold (e.g. 0.99 → 1% false-positive rate).
    pub null_quantile: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            bit_sizes: vec![14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48],
            max_degree: 6,
            n_semiprimes: 2000,
            n_monomials: 200,
            run_crt_rank: true,
            n_null_sims: 50,
            null_quantile: 0.99,
            seed: 0x4e32_dead_beef_0001,
        }
    }
}

/// Results for a single (channel, n_bits) block.
#[derive(Debug, Serialize)]
pub struct BlockResult {
    pub channel_weight: u32,
    pub channel_ell: u64,
    pub n_bits: u32,
    pub correlations: Vec<CorrelationResult>,
    pub fit_degree: Option<FitDegreeResult>,
    pub crt_rank: Option<CrtRankResult>,
    /// Real-valued spectral analysis (for n_bits ≤ 24 when run_crt_rank is set).
    pub real_spectrum: Option<RealSpectrum>,
}

/// Full scan result across all channels and bit sizes.
#[derive(Debug, Serialize)]
pub struct ScanResult {
    pub config_max_degree: u32,
    pub config_n_semiprimes: usize,
    pub config_n_monomials: usize,
    pub blocks: Vec<BlockResult>,
    pub power_law_fits: Vec<PowerLawFit>,
}

/// Power-law fit: deg_lower_bound(n) ≈ a · n^b for a single channel.
#[derive(Debug, Clone, Serialize)]
pub struct PowerLawFit {
    pub channel_weight: u32,
    pub channel_ell: u64,
    /// Estimated exponent b.
    pub exponent_b: f64,
    /// Estimated coefficient a.
    pub coefficient_a: f64,
    /// R² of the log-log fit.
    pub r_squared: f64,
    /// Interpretation label based on b.
    pub interpretation: String,
}

// ---------------------------------------------------------------------------
// Main scan entry point
// ---------------------------------------------------------------------------

/// Run the full degree scaling scan across all channels and bit sizes.
///
/// Uses Rayon for parallelism across (channel × n_bits) blocks.
pub fn run_scan(config: &ScanConfig) -> ScanResult {
    // Collect all (channel_idx, n_bits) work items
    let work: Vec<(usize, u32)> = (0..CHANNELS.len())
        .flat_map(|ci| config.bit_sizes.iter().map(move |&nb| (ci, nb)))
        .collect();

    let blocks: Vec<BlockResult> = work
        .into_par_iter()
        .map(|(ci, n_bits)| {
            let ch = &CHANNELS[ci];
            let seed = config.seed ^ (ci as u64 * 0x9e37_79b9) ^ (n_bits as u64 * 0x6c62_272e);

            let correlations = run_correlation_scan(
                ch, n_bits, config.max_degree,
                config.n_semiprimes, config.n_monomials, seed,
                config.n_null_sims, config.null_quantile,
            );

            let fit_degree = if n_bits <= 32 {
                let samples = config.n_semiprimes.min(5000);
                Some(run_fit_degree(ch, n_bits, samples, seed ^ 0x1234))
            } else {
                None
            };

            // CRT rank extended to n_bits ≤ 24 (improvement #2).
            let crt_rank = if config.run_crt_rank && n_bits <= 24 {
                Some(crate::degree::crt_rank(n_bits, ch))
            } else {
                None
            };

            // Real-valued spectrum for n_bits ≤ 24 (improvement #3).
            let real_spectrum = if config.run_crt_rank && n_bits <= 24 {
                Some(real_spectrum(n_bits, ch))
            } else {
                None
            };

            BlockResult {
                channel_weight: ch.weight,
                channel_ell: ch.ell,
                n_bits,
                correlations,
                fit_degree,
                crt_rank,
                real_spectrum,
            }
        })
        .collect();

    // Fit power laws per channel
    let power_law_fits: Vec<PowerLawFit> = CHANNELS
        .iter()
        .map(|ch| fit_power_law_for_channel(ch, &blocks, config.max_degree))
        .collect();

    ScanResult {
        config_max_degree: config.max_degree,
        config_n_semiprimes: config.n_semiprimes,
        config_n_monomials: config.n_monomials,
        blocks,
        power_law_fits,
    }
}

// ---------------------------------------------------------------------------
// Power-law fitting
// ---------------------------------------------------------------------------

/// For a given channel, extract the "maximum confirmed degree lower bound" at each
/// n_bits and fit a power law d*(n) ≈ a · n^b via log-log linear regression.
///
/// "Maximum confirmed degree lower bound" = maximum d for which significant
/// correlation was found (max_abs_corr > threshold).
fn fit_power_law_for_channel(ch: &Channel, blocks: &[BlockResult], max_degree: u32) -> PowerLawFit {
    // Collect (n_bits, d*) pairs where d* is the max confirmed degree
    let mut points: Vec<(f64, f64)> = Vec::new();

    for block in blocks.iter().filter(|b| b.channel_weight == ch.weight && b.channel_ell == ch.ell) {
        let d_star = block
            .correlations
            .iter()
            .filter(|r| r.significant)
            .map(|r| r.degree)
            .max()
            .unwrap_or(0);

        if d_star > 0 && block.n_bits > 0 {
            points.push((block.n_bits as f64, d_star as f64));
        }
    }

    if points.len() < 2 {
        return PowerLawFit {
            channel_weight: ch.weight,
            channel_ell: ch.ell,
            exponent_b: f64::NAN,
            coefficient_a: f64::NAN,
            r_squared: f64::NAN,
            interpretation: "insufficient_data".into(),
        };
    }

    // Log-log regression: log(d*) = log(a) + b * log(n)
    let log_pts: Vec<(f64, f64)> = points
        .iter()
        .map(|&(n, d)| (n.ln(), d.ln()))
        .collect();

    let (b, log_a, r2) = linear_regression(&log_pts);
    let a = log_a.exp();

    let interpretation = interpret_exponent(b, max_degree);

    PowerLawFit {
        channel_weight: ch.weight,
        channel_ell: ch.ell,
        exponent_b: b,
        coefficient_a: a,
        r_squared: r2,
        interpretation,
    }
}

/// Simple OLS linear regression on (x, y) pairs.  Returns (slope, intercept, r²).
fn linear_regression(pts: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let _syy: f64 = pts.iter().map(|p| p.1 * p.1).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-10 {
        return (0.0, sy / n, 0.0);
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    let ss_res: f64 = pts.iter().map(|p| (p.1 - (slope * p.0 + intercept)).powi(2)).sum();
    let y_mean = sy / n;
    let ss_tot: f64 = pts.iter().map(|p| (p.1 - y_mean).powi(2)).sum();
    let r2 = if ss_tot < 1e-10 { 1.0 } else { 1.0 - ss_res / ss_tot };

    (slope, intercept, r2)
}

fn interpret_exponent(b: f64, _max_degree: u32) -> String {
    if b.is_nan() {
        return "insufficient_data".into();
    }
    // If all points have d* = max_degree, the true degree might be higher.
    // We flag this as "saturated" (lower bound only).
    if b < 0.05 {
        "CONSTANT: degree bounded — Amy-Stinchcombe poly-time regime".into()
    } else if b < 0.25 {
        "SUBLINEAR: degree o(n^{1/3}) — better than GNFS if confirmed".into()
    } else if b < 0.40 {
        "GNFS_REGIME: degree ≈ n^{1/3} — consistent with GNFS complexity".into()
    } else if b < 0.60 {
        "SQRT_REGIME: degree ≈ n^{1/2} — consistent with random function".into()
    } else {
        "LINEAR: degree ≈ n — full hardness, barrier confirmed".into()
    }
}

// ---------------------------------------------------------------------------
// Summary printing
// ---------------------------------------------------------------------------

/// Print a human-readable summary of the scan result.
pub fn print_summary(result: &ScanResult) {
    println!("\n=== E20: Boolean Polynomial Degree Audit ===");
    println!("Amy-Stinchcombe path sum degree analysis for Eisenstein channels");
    println!("Config: {} semiprimes/block, {} monomials/degree, max_degree={}\n",
        result.config_n_semiprimes, result.config_n_monomials, result.config_max_degree);

    // Per-channel power-law fit summary
    println!("Power-Law Fits: deg_lower_bound(n) ≈ a·n^b");
    println!("{:>8} {:>8} {:>8} {:>8} {:>8}  {}", "weight", "ell", "b", "a", "R²", "interpretation");
    println!("{}", "-".repeat(80));
    for fit in &result.power_law_fits {
        if fit.exponent_b.is_nan() {
            println!("{:>8} {:>8}  (insufficient data)", fit.channel_weight, fit.channel_ell);
        } else {
            println!("{:>8} {:>8} {:>8.3} {:>8.3} {:>8.3}  {}",
                fit.channel_weight, fit.channel_ell,
                fit.exponent_b, fit.coefficient_a, fit.r_squared,
                fit.interpretation);
        }
    }

    // Per-channel, per-n_bits correlation table
    println!("\nCorrelation Lower Bounds (max |corr| over sampled monomials)");
    // Report which threshold was used (calibrated vs fixed).
    let using_calibrated = result.blocks.first()
        .and_then(|b| b.correlations.first())
        .map(|r| r.null_quantile > 0.0)
        .unwrap_or(false);
    if using_calibrated {
        let q = result.blocks.first()
            .and_then(|b| b.correlations.first())
            .map(|r| r.null_quantile)
            .unwrap_or(0.99);
        println!("  '*' = significant (> calibrated {:.0}%-quantile of null distribution)", q * 100.0);
    } else {
        println!("  '*' = significant (> fixed threshold 3/sqrt(m))");
    }
    for ch in eisenstein_hunt::CHANNELS {
        println!("\n  Channel k={}, ℓ={}", ch.weight, ch.ell);
        println!("  {:>8}  {}", "n_bits", (1..=result.config_max_degree).map(|d| format!("  d={d}  ")).collect::<String>());
        let channel_blocks: Vec<&BlockResult> = result.blocks.iter()
            .filter(|b| b.channel_weight == ch.weight && b.channel_ell == ch.ell)
            .collect();
        let mut by_n: Vec<&BlockResult> = channel_blocks;
        by_n.sort_by_key(|b| b.n_bits);
        for block in by_n {
            let row: String = block.correlations.iter()
                .map(|r| {
                    let sig = if r.significant { "*" } else { " " };
                    format!("{:>6.3}{sig}", r.max_abs_corr)
                })
                .collect::<Vec<_>>()
                .join("  ");
            println!("  {:>8}  {}", block.n_bits, row);
        }
    }

    // CRT rank summary
    let has_crt = result.blocks.iter().any(|b| b.crt_rank.is_some());
    if has_crt {
        println!("\nCRT Rank (n_bits ≤ 24)");
        println!("{:>8} {:>8} {:>10} {:>10} {:>12}", "weight", "ell", "n_bits", "n_primes", "rank_fraction");
        println!("{}", "-".repeat(55));
        let mut crt_rows: Vec<_> = result.blocks.iter()
            .filter_map(|b| b.crt_rank.as_ref().map(|r| (b, r)))
            .collect();
        crt_rows.sort_by_key(|(b, r)| (r.n_bits, b.channel_weight, b.channel_ell));
        for (_, r) in crt_rows {
            println!("{:>8} {:>8} {:>10} {:>10} {:>12.3}",
                r.channel_weight, r.channel_ell, r.n_bits, r.n_primes, r.rank_fraction);
        }
    }

    // Real spectrum summary
    let has_spec = result.blocks.iter().any(|b| b.real_spectrum.is_some());
    if has_spec {
        println!("\nReal Spectrum (n_bits ≤ 24)");
        println!(
            "{:>8} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10} {:>13}",
            "weight", "ell", "n_bits", "n_primes", "f2_rank", "real_rank",
            "stable_rk", "frac_stable"
        );
        println!("{}", "-".repeat(80));
        let mut spec_rows: Vec<_> = result.blocks.iter()
            .filter_map(|b| b.real_spectrum.as_ref().map(|s| {
                let f2 = b.crt_rank.as_ref().map(|r| r.rank).unwrap_or(0);
                (b, s, f2)
            }))
            .collect();
        spec_rows.sort_by_key(|(b, s, _)| (s.n_bits, b.channel_weight, b.channel_ell));
        for (_, s, f2) in &spec_rows {
            let frac = if s.n_primes > 0 { s.stable_rank / s.n_primes as f64 } else { 0.0 };
            println!(
                "{:>8} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10.2} {:>13.3}",
                s.channel_weight, s.channel_ell, s.n_bits, s.n_primes,
                f2, if s.real_rank > 0 { s.real_rank.to_string() } else { "—".to_string() },
                s.stable_rank, frac
            );
        }

        // Top-eigenvalue spectrum for the first channel at each n_bits where computed.
        let mut shown: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (_, s, _) in &spec_rows {
            if !s.top_eigenvalues.is_empty() && shown.insert(s.n_bits) {
                let top5: Vec<String> = s.top_eigenvalues.iter().take(5)
                    .map(|&v| format!("{:.2}", v))
                    .collect();
                println!(
                    "  n={} (ch k={},ℓ={}): top-5 |λ| = [{}]  spectral_norm={:.3}",
                    s.n_bits, s.channel_weight, s.channel_ell,
                    top5.join(", "), s.spectral_norm
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_perfect() {
        // y = 2x + 1 → slope=2, intercept=1, R²=1
        let pts = vec![(1.0_f64, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)];
        let (slope, intercept, r2) = linear_regression(&pts);
        assert!((slope - 2.0).abs() < 1e-9);
        assert!((intercept - 1.0).abs() < 1e-9);
        assert!((r2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_regression_constant() {
        // y = 5 → slope=0, R²=1 (or undefined for SS_tot=0)
        let pts = vec![(1.0_f64, 5.0), (2.0, 5.0), (3.0, 5.0)];
        let (slope, _, _) = linear_regression(&pts);
        assert!(slope.abs() < 1e-9);
    }

    #[test]
    fn test_interpret_exponent() {
        assert!(interpret_exponent(0.02, 6).contains("CONSTANT"));
        assert!(interpret_exponent(0.20, 6).contains("SUBLINEAR"));
        assert!(interpret_exponent(0.35, 6).contains("GNFS_REGIME"));
        assert!(interpret_exponent(0.50, 6).contains("SQRT_REGIME"));
        assert!(interpret_exponent(0.90, 6).contains("LINEAR"));
    }

    // Ignored by default: runs the full Rayon-parallel scan which is too slow in
    // unoptimised debug mode. Run with `cargo test --release -- --ignored` or via
    // `cargo run -p path-sum-degree --release -- --mode=quick`.
    #[test]
    #[ignore]
    fn test_scan_quick_smoke() {
        let config = ScanConfig {
            bit_sizes: vec![14, 16],
            max_degree: 2,
            n_semiprimes: 100,
            n_monomials: 20,
            run_crt_rank: true,
            n_null_sims: 10,
            null_quantile: 0.99,
            seed: 42,
        };
        let result = run_scan(&config);
        assert!(!result.blocks.is_empty());
        assert_eq!(result.power_law_fits.len(), eisenstein_hunt::CHANNELS.len());
        // Each block has `max_degree` correlation results
        for block in &result.blocks {
            assert_eq!(block.correlations.len(), config.max_degree as usize);
        }
    }
}
