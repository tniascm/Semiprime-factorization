//! E29 scaling experiment: MCMC smooth rate vs bit size.
//!
//! Extends E29v2 to 256-bit semiprimes using BigUint semiprime generation.
//! Tracks how MCMC advantage (vs uniform and lattice sieve) scales with N.

use std::path::PathBuf;
use std::time::Instant;

use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::json;

use classical_nfs::polynomial::select_polynomial;
use factoring_core::generate_rsa_target;

use novel_relation_sampler::logger::{read_checkpoint, write_checkpoint, EventLogger};
use novel_relation_sampler::sampler::{
    sample_lattice_sieve, sample_mcmc_big, sample_uniform_big,
};
use novel_relation_sampler::{MethodResult, SieveParams};

/// Scaling experiment configuration.
#[derive(Debug, Clone, Serialize)]
struct ScalingConfig {
    bit_sizes: Vec<u32>,
    semiprimes_per_size: usize,
    mcmc_chains: usize,
    mcmc_t_start: f64,
    mcmc_t_end: f64,
    seed: u64,
    lattice_sieve_max_bits: u32,
}

impl ScalingConfig {
    fn candidates_for_bits(&self, bits: u32) -> usize {
        if bits <= 64 {
            10_000
        } else if bits <= 128 {
            5_000
        } else {
            2_000
        }
    }
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            bit_sizes: vec![32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200, 256],
            semiprimes_per_size: 10,
            mcmc_chains: 10,
            mcmc_t_start: 10.0,
            mcmc_t_end: 0.1,
            seed: 42,
            lattice_sieve_max_bits: 80,
        }
    }
}

/// Per-semiprime comparison for scaling.
#[derive(Debug, Clone, Serialize)]
struct ScalingComparison {
    n_hex: String,
    bits: u32,
    poly_degree: usize,
    m_bits: u64,
    candidates_per_method: usize,
    uniform: MethodResult,
    mcmc: MethodResult,
    lattice: Option<MethodResult>,
    mcmc_vs_uniform_ratio: f64,
    mcmc_vs_lattice_rate: Option<f64>,
    mcmc_duplicate_rate: f64,
}

/// Per-bit-size aggregate.
#[derive(Debug, Serialize)]
struct ScalingBitSummary {
    bits: u32,
    count: usize,
    candidates_per_method: usize,
    poly_degree: usize,
    // Smooth rates
    uniform_mean_smooth_rate: f64,
    mcmc_mean_smooth_rate: f64,
    mcmc_mean_unique_smooth_rate: f64,
    lattice_mean_smooth_rate: Option<f64>,
    // Ratios
    mcmc_vs_uniform_ratio: f64,
    mcmc_vs_lattice_ratio: Option<f64>,
    // Throughput
    uniform_mean_rels_per_sec: f64,
    mcmc_mean_rels_per_sec: f64,
    lattice_mean_rels_per_sec: Option<f64>,
    // Timing
    uniform_mean_time_ms: f64,
    mcmc_mean_time_ms: f64,
    lattice_mean_time_ms: Option<f64>,
    // Dedup
    mcmc_mean_duplicate_rate: f64,
    // Energy
    uniform_mean_energy: f64,
    mcmc_mean_energy: f64,
    // Counts
    mcmc_mean_unique_both_smooth: f64,
    uniform_mean_both_smooth: f64,
}

/// Scaling analysis: how metrics change with bit size.
#[derive(Debug, Serialize)]
struct ScalingAnalysis {
    /// log2(mcmc_unique_smooth_rate) vs bits — linear regression slope
    mcmc_rate_decay_slope: f64,
    mcmc_rate_decay_intercept: f64,
    mcmc_rate_decay_r_squared: f64,
    /// MCMC/uniform ratio vs bits — linear regression slope
    advantage_ratio_slope: f64,
    advantage_ratio_intercept: f64,
    /// Predicted bits where MCMC unique smooth rate drops below 1e-6
    predicted_zero_crossing_bits: f64,
}

#[derive(Debug, Serialize)]
struct ScalingReport {
    experiment: String,
    config: ScalingConfig,
    per_bit_size: Vec<ScalingBitSummary>,
    scaling_analysis: Option<ScalingAnalysis>,
    per_semiprime: Vec<ScalingComparison>,
}

/// Simple linear regression: y = slope*x + intercept
fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (0.0, sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    let ss_res: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let pred = slope * x + intercept;
            (y - pred).powi(2)
        })
        .sum();
    let y_mean = sy / n;
    let ss_tot: f64 = ys.iter().map(|y| (y - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, intercept, r_squared)
}

fn main() {
    let config = ScalingConfig::default();
    let output_dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/e29_scaling"
    ));

    std::fs::create_dir_all(&output_dir).ok();
    let events_path = output_dir.join("events.jsonl");
    let mut logger = EventLogger::new(&events_path).unwrap_or_else(|e| {
        eprintln!("Warning: cannot create event log: {}", e);
        EventLogger::noop()
    });

    let resume_from = read_checkpoint(&output_dir).unwrap_or(0);

    logger.log("config", serde_json::to_value(&config).unwrap());

    eprintln!("E29 Scaling: MCMC smooth rate vs bit size");
    eprintln!("==========================================");
    eprintln!(
        "Bit sizes: {:?}, {} semiprimes each",
        config.bit_sizes, config.semiprimes_per_size,
    );
    eprintln!(
        "Lattice sieve cutoff: {} bits",
        config.lattice_sieve_max_bits
    );
    if resume_from > 0 {
        eprintln!("Resuming from checkpoint index {}", resume_from);
    }
    eprintln!();

    let mut all_results: Vec<ScalingComparison> = Vec::new();
    let mut per_bit_summaries: Vec<ScalingBitSummary> = Vec::new();
    let total_semiprimes: usize = config.bit_sizes.len() * config.semiprimes_per_size;
    let mut global_index = 0usize;
    let global_start = Instant::now();

    for &bits in &config.bit_sizes {
        eprintln!("--- {} bits ---", bits);
        let sieve_params = SieveParams::for_bits(bits);
        let candidates = config.candidates_for_bits(bits);
        let degree = sieve_params.degree;
        let mut rng_gen = StdRng::seed_from_u64(config.seed + bits as u64);

        let mut bit_results: Vec<ScalingComparison> = Vec::new();
        let run_lattice = bits <= config.lattice_sieve_max_bits;

        for i in 0..config.semiprimes_per_size {
            // Generate semiprime using factoring-core (BigUint)
            let target = generate_rsa_target(bits, &mut rng_gen);
            let n_big = &target.n;
            let n_hex = format!("{:x}", n_big);

            if global_index < resume_from {
                global_index += 1;
                continue;
            }

            let sp_start = Instant::now();

            // Polynomial selection (works with BigUint)
            let poly = select_polynomial(n_big, degree);
            let m_u64 = poly.m.to_u64().expect(&format!(
                "m does not fit u64 for {}-bit N (m has {} bits)",
                bits,
                poly.m.bits()
            ));
            let m_bits = poly.m.bits();

            logger.log(
                "semiprime",
                json!({
                    "n_hex": &n_hex,
                    "bits": bits,
                    "actual_bits": n_big.bits(),
                    "index": i,
                    "poly_degree": degree,
                    "m_bits": m_bits,
                    "sieve_area": sieve_params.sieve_area,
                    "max_b": sieve_params.max_b,
                    "fb_bound": sieve_params.fb_bound,
                    "candidates": candidates,
                }),
            );

            // 1. Uniform sampling (BigUint energy)
            let mut rng_u =
                StdRng::seed_from_u64(config.seed.wrapping_add(n_big.to_u64().unwrap_or(bits as u64)));
            let uniform_result =
                sample_uniform_big(&poly, m_u64, &sieve_params, candidates, &mut rng_u);

            // 2. MCMC sampling (BigUint energy)
            let mut rng_m = StdRng::seed_from_u64(
                config
                    .seed
                    .wrapping_add(n_big.to_u64().unwrap_or(bits as u64))
                    .wrapping_add(1),
            );
            let mcmc_result = sample_mcmc_big(
                &poly,
                m_u64,
                &sieve_params,
                candidates,
                config.mcmc_chains,
                config.mcmc_t_start,
                config.mcmc_t_end,
                &mut rng_m,
            );

            // 3. Lattice sieve (only for small bit sizes)
            let lattice_result = if run_lattice {
                Some(sample_lattice_sieve(&poly, m_u64, &sieve_params))
            } else {
                None
            };

            // Comparison metrics
            let u_rate = uniform_result.unique_smooth_rate();
            let m_rate = mcmc_result.smooth_rate();
            let m_unique_rate = mcmc_result.unique_smooth_rate();

            let mcmc_vs_uniform = if u_rate > 0.0 {
                m_unique_rate / u_rate
            } else if m_unique_rate > 0.0 {
                f64::INFINITY
            } else {
                1.0
            };

            let mcmc_vs_lattice_rate = lattice_result.as_ref().map(|lr| {
                let l_rate = lr.smooth_rate();
                if l_rate > 0.0 {
                    m_unique_rate / l_rate
                } else if m_unique_rate > 0.0 {
                    f64::INFINITY
                } else {
                    1.0
                }
            });

            let mcmc_dup_rate = if mcmc_result.valid_candidates > 0 {
                1.0 - (mcmc_result.unique_valid as f64 / mcmc_result.valid_candidates as f64)
            } else {
                0.0
            };

            let comparison = ScalingComparison {
                n_hex: n_hex.clone(),
                bits,
                poly_degree: degree,
                m_bits,
                candidates_per_method: candidates,
                uniform: uniform_result,
                mcmc: mcmc_result,
                lattice: lattice_result,
                mcmc_vs_uniform_ratio: mcmc_vs_uniform,
                mcmc_vs_lattice_rate,
                mcmc_duplicate_rate: mcmc_dup_rate,
            };

            logger.log(
                "comparison",
                json!({
                    "n_hex": &n_hex,
                    "bits": bits,
                    "uniform_unique_smooth_rate": u_rate,
                    "mcmc_smooth_rate": m_rate,
                    "mcmc_unique_smooth_rate": m_unique_rate,
                    "mcmc_vs_uniform": mcmc_vs_uniform,
                    "mcmc_vs_lattice_rate": mcmc_vs_lattice_rate,
                    "mcmc_duplicate_rate": mcmc_dup_rate,
                    "mcmc_unique_both_smooth": comparison.mcmc.unique_both_smooth,
                    "uniform_both_smooth": comparison.uniform.both_smooth,
                }),
            );

            let sp_elapsed = sp_start.elapsed().as_secs_f64();
            global_index += 1;

            let done_frac = global_index as f64 / total_semiprimes as f64;
            let elapsed_total = global_start.elapsed().as_secs_f64();
            let eta = if done_frac > 0.0 {
                elapsed_total / done_frac - elapsed_total
            } else {
                0.0
            };

            let lat_str = if let Some(ref lr) = comparison.lattice {
                format!(" lat={:.4}%", lr.smooth_rate() * 100.0)
            } else {
                String::new()
            };

            eprintln!(
                "  [{}/{}] {}b uni={:.4}% mcmc={:.4}%(uniq={:.4}%,dup={:.0}%){} mcmc/uni={:.1}x  {:.1}s ETA {:.0}s",
                global_index,
                total_semiprimes,
                bits,
                u_rate * 100.0,
                m_rate * 100.0,
                m_unique_rate * 100.0,
                mcmc_dup_rate * 100.0,
                lat_str,
                mcmc_vs_uniform,
                sp_elapsed,
                eta,
            );

            bit_results.push(comparison);
            write_checkpoint(&output_dir, global_index).ok();
        }

        // Aggregate for this bit size
        let count = bit_results.len();
        if count > 0 {
            let mean = |f: &dyn Fn(&ScalingComparison) -> f64| -> f64 {
                bit_results.iter().map(f).sum::<f64>() / count as f64
            };

            let u_smooth = mean(&|r| r.uniform.unique_smooth_rate());
            let m_smooth = mean(&|r| r.mcmc.smooth_rate());
            let m_unique = mean(&|r| r.mcmc.unique_smooth_rate());

            let lattice_smooth = if run_lattice {
                Some(mean(&|r| {
                    r.lattice.as_ref().map_or(0.0, |lr| lr.smooth_rate())
                }))
            } else {
                None
            };

            let lattice_rps = if run_lattice {
                Some(mean(&|r| {
                    r.lattice
                        .as_ref()
                        .map_or(0.0, |lr| lr.relations_per_second())
                }))
            } else {
                None
            };

            let lattice_time = if run_lattice {
                Some(mean(&|r| r.lattice.as_ref().map_or(0.0, |lr| lr.time_ms)))
            } else {
                None
            };

            let summary = ScalingBitSummary {
                bits,
                count,
                candidates_per_method: config.candidates_for_bits(bits),
                poly_degree: degree,
                uniform_mean_smooth_rate: u_smooth,
                mcmc_mean_smooth_rate: m_smooth,
                mcmc_mean_unique_smooth_rate: m_unique,
                lattice_mean_smooth_rate: lattice_smooth,
                mcmc_vs_uniform_ratio: if u_smooth > 0.0 {
                    m_unique / u_smooth
                } else {
                    0.0
                },
                mcmc_vs_lattice_ratio: lattice_smooth.map(|ls| {
                    if ls > 0.0 {
                        m_unique / ls
                    } else {
                        0.0
                    }
                }),
                uniform_mean_rels_per_sec: mean(&|r| r.uniform.relations_per_second()),
                mcmc_mean_rels_per_sec: mean(&|r| r.mcmc.relations_per_second()),
                lattice_mean_rels_per_sec: lattice_rps,
                uniform_mean_time_ms: mean(&|r| r.uniform.time_ms),
                mcmc_mean_time_ms: mean(&|r| r.mcmc.time_ms),
                lattice_mean_time_ms: lattice_time,
                mcmc_mean_duplicate_rate: mean(&|r| r.mcmc_duplicate_rate),
                uniform_mean_energy: mean(&|r| r.uniform.mean_energy),
                mcmc_mean_energy: mean(&|r| r.mcmc.mean_energy),
                mcmc_mean_unique_both_smooth: mean(&|r| r.mcmc.unique_both_smooth as f64),
                uniform_mean_both_smooth: mean(&|r| r.uniform.both_smooth as f64),
            };

            eprintln!(
                "  Smooth rates: uni={:.6}% mcmc_uniq={:.6}%{}",
                summary.uniform_mean_smooth_rate * 100.0,
                summary.mcmc_mean_unique_smooth_rate * 100.0,
                if let Some(ls) = summary.lattice_mean_smooth_rate {
                    format!(" lat={:.6}%", ls * 100.0)
                } else {
                    String::new()
                },
            );
            eprintln!(
                "  MCMC dup={:.1}% mcmc/uni={:.1}x  energy: uni={:.1} mcmc={:.1}",
                summary.mcmc_mean_duplicate_rate * 100.0,
                summary.mcmc_vs_uniform_ratio,
                summary.uniform_mean_energy,
                summary.mcmc_mean_energy,
            );
            eprintln!(
                "  Counts: uni_smooth={:.1} mcmc_uniq_smooth={:.1}",
                summary.uniform_mean_both_smooth,
                summary.mcmc_mean_unique_both_smooth,
            );
            eprintln!();

            per_bit_summaries.push(summary);
        }

        all_results.extend(bit_results);
    }

    // Scaling analysis
    let analysis = compute_scaling_analysis(&per_bit_summaries);

    if let Some(ref a) = analysis {
        eprintln!("=== Scaling Analysis ===");
        eprintln!(
            "MCMC rate decay: ln(rate) = {:.4} * bits + {:.2}  (R²={:.3})",
            a.mcmc_rate_decay_slope, a.mcmc_rate_decay_intercept, a.mcmc_rate_decay_r_squared,
        );
        eprintln!(
            "Advantage ratio trend: ratio = {:.4} * bits + {:.2}",
            a.advantage_ratio_slope, a.advantage_ratio_intercept,
        );
        eprintln!(
            "Predicted zero-crossing: {} bits",
            a.predicted_zero_crossing_bits as u32,
        );
        eprintln!();
    }

    // Write report
    let report = ScalingReport {
        experiment: "E29 Scaling: MCMC smooth rate vs bit size".to_string(),
        config,
        per_bit_size: per_bit_summaries,
        scaling_analysis: analysis,
        per_semiprime: all_results,
    };

    let summary_path = output_dir.join("summary.json");
    match serde_json::to_string_pretty(&report) {
        Ok(json_str) => match std::fs::write(&summary_path, &json_str) {
            Ok(()) => eprintln!("Results written to {}", summary_path.display()),
            Err(e) => eprintln!("Error writing summary: {}", e),
        },
        Err(e) => eprintln!("JSON serialization error: {}", e),
    }

    let _ = std::fs::remove_file(output_dir.join("checkpoint.txt"));
    eprintln!("\nTotal time: {:.1}s", global_start.elapsed().as_secs_f64());
}

fn compute_scaling_analysis(summaries: &[ScalingBitSummary]) -> Option<ScalingAnalysis> {
    // Filter to bit sizes where MCMC found at least some smooth pairs
    let data: Vec<(f64, f64, f64)> = summaries
        .iter()
        .filter(|s| s.mcmc_mean_unique_smooth_rate > 0.0)
        .map(|s| {
            (
                s.bits as f64,
                s.mcmc_mean_unique_smooth_rate.ln(),
                s.mcmc_vs_uniform_ratio,
            )
        })
        .collect();

    if data.len() < 3 {
        return None;
    }

    let bits: Vec<f64> = data.iter().map(|d| d.0).collect();
    let ln_rates: Vec<f64> = data.iter().map(|d| d.1).collect();
    let ratios: Vec<f64> = data.iter().map(|d| d.2).collect();

    let (rate_slope, rate_intercept, rate_r2) = linear_regression(&bits, &ln_rates);
    let (ratio_slope, ratio_intercept, _) = linear_regression(&bits, &ratios);

    // Predict where ln(rate) = ln(1e-6) ≈ -13.8
    let target_ln_rate = (-13.8f64).min(ln_rates.last().copied().unwrap_or(-13.8));
    let zero_crossing = if rate_slope.abs() > 1e-10 {
        (target_ln_rate - rate_intercept) / rate_slope
    } else {
        f64::INFINITY
    };

    Some(ScalingAnalysis {
        mcmc_rate_decay_slope: rate_slope,
        mcmc_rate_decay_intercept: rate_intercept,
        mcmc_rate_decay_r_squared: rate_r2,
        advantage_ratio_slope: ratio_slope,
        advantage_ratio_intercept: ratio_intercept,
        predicted_zero_crossing_bits: zero_crossing,
    })
}
