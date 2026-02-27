//! E1: Group Structure Scaling Analysis
//!
//! Systematic evaluation of group-theoretic factoring methods across bit sizes 16-64.
//! Tests smooth-order factoring, phi-recovery pipeline, and Chebotarev density analysis
//! with timing, success rates, and scaling exponents.

use std::path::Path;
use std::time::Instant;

use factoring_core::generate_rsa_target;
use group_structure::{
    approximate_carmichael, chebotarev, factor_from_phi, factor_via_smooth_orders,
};
use num_bigint::BigUint;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

// ============================================================
// Configuration
// ============================================================

const BIT_SIZES: &[u32] = &[16, 20, 24, 28, 32, 40, 48, 56, 64];
const SEMIPRIMES_PER_SIZE: usize = 20;
const SEED: u64 = 42;

// Method parameters
const SMOOTH_BOUND: u64 = 100;
const SMOOTH_TRIALS: usize = 500;
const CARMICHAEL_SAMPLES: usize = 200;
const MAX_ORDER: u64 = 100_000;
const CHEBOTAREV_MAX_PRIME: u64 = 20;
const CHEBOTAREV_SAMPLES: u64 = 200;
const TIMEOUT_SECS: f64 = 5.0;

// ============================================================
// Result structures
// ============================================================

#[derive(Serialize)]
struct E1Report {
    experiment: String,
    config: Config,
    per_bit_size: Vec<BitSizeResult>,
    scaling: ScalingAnalysis,
    verdict: String,
}

#[derive(Serialize)]
struct Config {
    bit_sizes: Vec<u32>,
    semiprimes_per_size: usize,
    seed: u64,
    smooth_bound: u64,
    smooth_trials: usize,
    carmichael_samples: usize,
    max_order: u64,
    chebotarev_max_prime: u64,
    chebotarev_samples: u64,
    timeout_secs: f64,
}

#[derive(Serialize)]
struct BitSizeResult {
    bit_size: u32,
    n_semiprimes: usize,
    smooth_factor: MethodStats,
    phi_recovery: MethodStats,
    chebotarev: MethodStats,
}

#[derive(Serialize)]
struct MethodStats {
    success_count: usize,
    total_count: usize,
    success_rate: f64,
    mean_time_ms: f64,
    median_time_ms: f64,
    timeout_count: usize,
}

#[derive(Serialize)]
struct ScalingAnalysis {
    smooth_factor_exponent: Option<f64>,
    phi_recovery_exponent: Option<f64>,
    chebotarev_exponent: Option<f64>,
    smooth_factor_success_trend: String,
    phi_recovery_success_trend: String,
    chebotarev_success_trend: String,
}

// ============================================================
// Core logic
// ============================================================

fn run_method_with_timeout<F>(f: F) -> (bool, f64)
where
    F: FnOnce() -> bool,
{
    let start = Instant::now();
    let success = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    (success, elapsed_ms)
}

fn compute_method_stats(results: &[(bool, f64)]) -> MethodStats {
    let total = results.len();
    let success_count = results.iter().filter(|(s, _)| *s).count();
    let timeout_count = results
        .iter()
        .filter(|(_, t)| *t >= TIMEOUT_SECS * 1000.0 * 0.99)
        .count();

    let times: Vec<f64> = results.iter().map(|(_, t)| *t).collect();
    let mean_time = if times.is_empty() {
        0.0
    } else {
        times.iter().sum::<f64>() / times.len() as f64
    };
    let median_time = if times.is_empty() {
        0.0
    } else {
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    MethodStats {
        success_count,
        total_count: total,
        success_rate: if total > 0 {
            success_count as f64 / total as f64
        } else {
            0.0
        },
        mean_time_ms: mean_time,
        median_time_ms: median_time,
        timeout_count,
    }
}

fn run_bit_size(bit_size: u32, count: usize, rng: &mut StdRng) -> BitSizeResult {
    let mut smooth_results = Vec::new();
    let mut phi_results = Vec::new();
    let mut cheb_results = Vec::new();

    for _ in 0..count {
        let target = generate_rsa_target(bit_size, rng);
        let n = &target.n;
        let true_p = &target.p;

        // Method 1: Smooth-order factoring
        let n_clone = n.clone();
        let tp = true_p.clone();
        let (success, time) = run_method_with_timeout(|| {
            let start = Instant::now();
            if let Some(p) = factor_via_smooth_orders(&n_clone, SMOOTH_BOUND, SMOOTH_TRIALS) {
                if start.elapsed().as_secs_f64() > TIMEOUT_SECS {
                    return false;
                }
                p == tp || &n_clone / &p == tp
            } else {
                false
            }
        });
        smooth_results.push((success, time.min(TIMEOUT_SECS * 1000.0)));

        // Method 2: Phi-recovery pipeline (Carmichael approx -> factor_from_phi)
        let n_clone = n.clone();
        let tp = true_p.clone();
        let (success, time) = run_method_with_timeout(|| {
            let start = Instant::now();
            if let Some(lambda) = approximate_carmichael(&n_clone, CARMICHAEL_SAMPLES, MAX_ORDER) {
                if start.elapsed().as_secs_f64() > TIMEOUT_SECS {
                    return false;
                }
                // Try multiples of lambda as phi candidates
                for mult in 1..=100u64 {
                    if start.elapsed().as_secs_f64() > TIMEOUT_SECS {
                        return false;
                    }
                    let phi = BigUint::from(lambda.saturating_mul(mult));
                    if let Some((p, _q)) = factor_from_phi(&n_clone, &phi) {
                        if p == tp || &n_clone / &p == tp {
                            return true;
                        }
                    }
                }
                false
            } else {
                false
            }
        });
        phi_results.push((success, time.min(TIMEOUT_SECS * 1000.0)));

        // Method 3: Chebotarev density analysis
        let n_clone = n.clone();
        let tp = true_p.clone();
        let (success, time) = run_method_with_timeout(|| {
            if let Some(p) =
                chebotarev::factor_via_chebotarev(&n_clone, CHEBOTAREV_MAX_PRIME, CHEBOTAREV_SAMPLES)
            {
                p == tp || &n_clone / &p == tp
            } else {
                false
            }
        });
        cheb_results.push((success, time.min(TIMEOUT_SECS * 1000.0)));
    }

    BitSizeResult {
        bit_size,
        n_semiprimes: count,
        smooth_factor: compute_method_stats(&smooth_results),
        phi_recovery: compute_method_stats(&phi_results),
        chebotarev: compute_method_stats(&cheb_results),
    }
}

/// Fit log-log regression: log(time) = a * log(bits) + b
/// Returns the exponent a, or None if insufficient data.
fn log_log_slope(bit_sizes: &[u32], values: &[f64]) -> Option<f64> {
    let valid: Vec<(f64, f64)> = bit_sizes
        .iter()
        .zip(values.iter())
        .filter(|(_, v)| **v > 0.0 && v.is_finite())
        .map(|(b, v)| ((*b as f64).ln(), v.ln()))
        .collect();

    if valid.len() < 3 {
        return None;
    }

    let n = valid.len() as f64;
    let sum_x: f64 = valid.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = valid.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = valid.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = valid.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return None;
    }

    Some((n * sum_xy - sum_x * sum_y) / denom)
}

fn classify_success_trend(bit_sizes: &[u32], rates: &[f64]) -> String {
    let valid: Vec<(f64, f64)> = bit_sizes
        .iter()
        .zip(rates.iter())
        .filter(|(_, r)| r.is_finite())
        .map(|(b, r)| (*b as f64, *r))
        .collect();

    if valid.len() < 3 {
        return "INSUFFICIENT DATA".to_string();
    }

    let last_third = &valid[valid.len() * 2 / 3..];
    let first_third = &valid[..valid.len() / 3];

    let first_mean: f64 = first_third.iter().map(|(_, r)| r).sum::<f64>() / first_third.len() as f64;
    let last_mean: f64 = last_third.iter().map(|(_, r)| r).sum::<f64>() / last_third.len() as f64;

    if last_mean < 0.01 && first_mean > 0.1 {
        "EXPONENTIAL DECAY -- success drops to near-zero".to_string()
    } else if last_mean < first_mean * 0.5 {
        "SIGNIFICANT DECAY -- success rate halves or worse".to_string()
    } else if last_mean < first_mean * 0.9 {
        "MODERATE DECAY -- gradual decline".to_string()
    } else {
        "STABLE -- no clear decay pattern".to_string()
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("======================================================================");
    println!("E1: Group Structure Scaling Analysis");
    println!("======================================================================");
    println!("Bit sizes: {:?}", BIT_SIZES);
    println!("Semiprimes per size: {}", SEMIPRIMES_PER_SIZE);
    println!("Methods: smooth-order, phi-recovery, Chebotarev density");
    println!("Timeout per method per semiprime: {:.0}s", TIMEOUT_SECS);
    println!();

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut per_bit_size = Vec::new();

    println!(
        "{:>5} | {:>12} {:>8} | {:>12} {:>8} | {:>12} {:>8}",
        "bits", "smooth_rate", "time_ms", "phi_rate", "time_ms", "cheb_rate", "time_ms"
    );
    println!("{}", "-".repeat(85));

    for &bits in BIT_SIZES {
        let result = run_bit_size(bits, SEMIPRIMES_PER_SIZE, &mut rng);
        println!(
            "{:>5} | {:>11.1}% {:>7.1} | {:>11.1}% {:>7.1} | {:>11.1}% {:>7.1}",
            bits,
            result.smooth_factor.success_rate * 100.0,
            result.smooth_factor.median_time_ms,
            result.phi_recovery.success_rate * 100.0,
            result.phi_recovery.median_time_ms,
            result.chebotarev.success_rate * 100.0,
            result.chebotarev.median_time_ms,
        );
        per_bit_size.push(result);
    }

    // Scaling analysis
    println!("\n======================================================================");
    println!("SCALING ANALYSIS");
    println!("======================================================================");

    let bits_vec: Vec<u32> = per_bit_size.iter().map(|r| r.bit_size).collect();
    let smooth_times: Vec<f64> = per_bit_size.iter().map(|r| r.smooth_factor.mean_time_ms).collect();
    let phi_times: Vec<f64> = per_bit_size.iter().map(|r| r.phi_recovery.mean_time_ms).collect();
    let cheb_times: Vec<f64> = per_bit_size.iter().map(|r| r.chebotarev.mean_time_ms).collect();

    let smooth_exp = log_log_slope(&bits_vec, &smooth_times);
    let phi_exp = log_log_slope(&bits_vec, &phi_times);
    let cheb_exp = log_log_slope(&bits_vec, &cheb_times);

    if let Some(exp) = smooth_exp {
        println!("  Smooth-order time exponent: {:.2} (time ~ bits^{:.2})", exp, exp);
    }
    if let Some(exp) = phi_exp {
        println!("  Phi-recovery time exponent: {:.2} (time ~ bits^{:.2})", exp, exp);
    }
    if let Some(exp) = cheb_exp {
        println!("  Chebotarev time exponent:   {:.2} (time ~ bits^{:.2})", exp, exp);
    }

    let smooth_rates: Vec<f64> = per_bit_size.iter().map(|r| r.smooth_factor.success_rate).collect();
    let phi_rates: Vec<f64> = per_bit_size.iter().map(|r| r.phi_recovery.success_rate).collect();
    let cheb_rates: Vec<f64> = per_bit_size.iter().map(|r| r.chebotarev.success_rate).collect();

    let smooth_trend = classify_success_trend(&bits_vec, &smooth_rates);
    let phi_trend = classify_success_trend(&bits_vec, &phi_rates);
    let cheb_trend = classify_success_trend(&bits_vec, &cheb_rates);

    println!("\n  Smooth-order success: {}", smooth_trend);
    println!("  Phi-recovery success: {}", phi_trend);
    println!("  Chebotarev success:   {}", cheb_trend);

    // Verdict
    let any_viable_at_64 = per_bit_size
        .iter()
        .find(|r| r.bit_size == 64)
        .map(|r| {
            r.smooth_factor.success_rate > 0.01
                || r.phi_recovery.success_rate > 0.01
                || r.chebotarev.success_rate > 0.01
        })
        .unwrap_or(false);

    let verdict = if any_viable_at_64 {
        "SUBEXPONENTIAL -- some methods viable at 64-bit (investigate scaling further)"
    } else {
        "EXPONENTIAL BARRIER -- all methods fail at 64-bit, success decays with N"
    };

    println!("\n======================================================================");
    println!("VERDICT: {}", verdict);
    println!("======================================================================");

    // Save JSON
    let scaling = ScalingAnalysis {
        smooth_factor_exponent: smooth_exp,
        phi_recovery_exponent: phi_exp,
        chebotarev_exponent: cheb_exp,
        smooth_factor_success_trend: smooth_trend,
        phi_recovery_success_trend: phi_trend,
        chebotarev_success_trend: cheb_trend,
    };

    let report = E1Report {
        experiment: "E1: Group Structure Scaling Analysis".to_string(),
        config: Config {
            bit_sizes: BIT_SIZES.to_vec(),
            semiprimes_per_size: SEMIPRIMES_PER_SIZE,
            seed: SEED,
            smooth_bound: SMOOTH_BOUND,
            smooth_trials: SMOOTH_TRIALS,
            carmichael_samples: CARMICHAEL_SAMPLES,
            max_order: MAX_ORDER,
            chebotarev_max_prime: CHEBOTAREV_MAX_PRIME,
            chebotarev_samples: CHEBOTAREV_SAMPLES,
            timeout_secs: TIMEOUT_SECS,
        },
        per_bit_size,
        scaling,
        verdict: verdict.to_string(),
    };

    // Save to data directory (two levels up from rust/group-structure/)
    let data_dir = Path::new("../../data");
    std::fs::create_dir_all(data_dir).ok();
    let output_path = data_dir.join("E1_group_structure_results.json");
    match serde_json::to_string_pretty(&report) {
        Ok(json) => match std::fs::write(&output_path, &json) {
            Ok(_) => println!("\nResults saved to {}", output_path.display()),
            Err(e) => {
                eprintln!("Failed to write {}: {}", output_path.display(), e);
                println!("{}", json);
            }
        },
        Err(e) => eprintln!("JSON serialization failed: {}", e),
    }
}
