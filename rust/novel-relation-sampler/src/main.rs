//! E29 benchmark driver: novel relation collection via joint norm biasing.
//!
//! Compares MCMC sampling (biased toward small log-norm product) against
//! uniform random sampling within the same NFS sieve range.

use std::path::PathBuf;
use std::time::Instant;

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;

use classical_nfs::polynomial::select_polynomial;

use novel_relation_sampler::logger::{read_checkpoint, write_checkpoint, EventLogger};
use novel_relation_sampler::sampler::{sample_mcmc, sample_uniform};
use novel_relation_sampler::{ComparisonResult, E29Config, SieveParams};

// --- Semiprime generation (from nfs-tn-hybrid) ---

fn generate_balanced_semiprime(bits: u32, rng: &mut StdRng) -> u64 {
    let half = bits / 2;
    let lo = 1u64 << (half - 1);
    let hi = 1u64 << half;
    loop {
        let p = random_prime(lo, hi, rng);
        let q = random_prime(lo, hi, rng);
        if p == q {
            continue;
        }
        if let Some(n) = p.checked_mul(q) {
            let n_bits = 64 - n.leading_zeros();
            if n_bits == bits || n_bits == bits - 1 {
                let (min_f, max_f) = if p < q { (p, q) } else { (q, p) };
                if (min_f as f64) / (max_f as f64) >= 0.3 {
                    return n;
                }
            }
        }
    }
}

fn random_prime(lo: u64, hi: u64, rng: &mut StdRng) -> u64 {
    loop {
        let mut c = rng.gen_range(lo..hi);
        if c % 2 == 0 {
            c += 1;
        }
        if is_probably_prime(c) {
            return c;
        }
    }
}

fn is_probably_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 { d /= 2; r += 1; }
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in &witnesses {
        if a >= n { continue; }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 { continue; }
        let mut found = false;
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 { found = true; break; }
        }
        if !found { return false; }
    }
    true
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let m = modulus as u128;
    base %= modulus;
    let mut b = base as u128;
    while exp > 0 {
        if exp % 2 == 1 { result = result * b % m; }
        exp /= 2;
        b = b * b % m;
    }
    result as u64
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// --- Aggregate statistics ---

#[derive(Debug, Serialize)]
struct BitSizeSummary {
    bits: u32,
    count: usize,
    uniform_mean_smooth_rate: f64,
    mcmc_mean_smooth_rate: f64,
    ratio: f64,
    uniform_mean_time_ms: f64,
    mcmc_mean_time_ms: f64,
    uniform_mean_rat_log2: f64,
    mcmc_mean_rat_log2: f64,
    uniform_mean_alg_log2: f64,
    mcmc_mean_alg_log2: f64,
    uniform_mean_energy: f64,
    mcmc_mean_energy: f64,
}

#[derive(Debug, Serialize)]
struct E29Report {
    experiment: String,
    config: E29Config,
    per_bit_size: Vec<BitSizeSummary>,
    per_semiprime: Vec<ComparisonResult>,
}

fn main() {
    let config = E29Config::default();
    let output_dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/e29_novel_relation"
    ));

    std::fs::create_dir_all(&output_dir).ok();
    let events_path = output_dir.join("events.jsonl");
    let mut logger = EventLogger::new(&events_path).unwrap_or_else(|e| {
        eprintln!("Warning: cannot create event log: {}", e);
        EventLogger::noop()
    });

    // Check for checkpoint (resume support)
    let resume_from = read_checkpoint(&output_dir).unwrap_or(0);

    logger.log("config", serde_json::to_value(&config).unwrap());

    eprintln!("E29: Novel Relation Collection via Joint Norm Biasing");
    eprintln!("=====================================================");
    eprintln!(
        "Bit sizes: {:?}, {} semiprimes each, {} candidates/method",
        config.bit_sizes, config.semiprimes_per_size, config.candidates_per_method
    );
    if resume_from > 0 {
        eprintln!("Resuming from checkpoint index {}", resume_from);
    }
    eprintln!();

    let mut all_results: Vec<ComparisonResult> = Vec::new();
    let mut per_bit_summaries: Vec<BitSizeSummary> = Vec::new();
    let total_semiprimes: usize = config.bit_sizes.len() * config.semiprimes_per_size;
    let mut global_index = 0usize;
    let global_start = Instant::now();

    for &bits in &config.bit_sizes {
        eprintln!("--- {} bits ---", bits);
        let sieve_params = SieveParams::for_bits(bits);
        let mut rng_gen = StdRng::seed_from_u64(config.seed + bits as u64);

        let mut bit_results: Vec<ComparisonResult> = Vec::new();

        for i in 0..config.semiprimes_per_size {
            let n = generate_balanced_semiprime(bits, &mut rng_gen);

            // Skip already-completed semiprimes (resume)
            if global_index < resume_from {
                global_index += 1;
                continue;
            }

            let sp_start = Instant::now();

            // Polynomial selection
            let n_big = BigUint::from(n);
            let poly = select_polynomial(&n_big, sieve_params.degree);
            let m_u64 = poly.m.to_u64().unwrap_or(0);

            logger.log(
                "semiprime",
                json!({
                    "n": n, "bits": bits, "index": i,
                    "poly_degree": sieve_params.degree, "m": m_u64,
                    "sieve_area": sieve_params.sieve_area,
                    "max_b": sieve_params.max_b,
                    "fb_bound": sieve_params.fb_bound,
                }),
            );

            // Uniform sampling
            let mut rng_u = StdRng::seed_from_u64(config.seed + n);
            let uniform_result = sample_uniform(
                &poly,
                m_u64,
                &sieve_params,
                config.candidates_per_method,
                &mut rng_u,
            );

            logger.log(
                "method_result",
                serde_json::to_value(&uniform_result).unwrap(),
            );

            // MCMC sampling
            let mut rng_m = StdRng::seed_from_u64(config.seed + n + 1);
            let mcmc_result = sample_mcmc(
                &poly,
                m_u64,
                &sieve_params,
                config.candidates_per_method,
                config.mcmc_chains,
                config.mcmc_t_start,
                config.mcmc_t_end,
                &mut rng_m,
            );

            logger.log(
                "method_result",
                serde_json::to_value(&mcmc_result).unwrap(),
            );

            // Comparison
            let u_rate = uniform_result.smooth_rate();
            let m_rate = mcmc_result.smooth_rate();
            let ratio = if u_rate > 0.0 { m_rate / u_rate } else if m_rate > 0.0 { f64::INFINITY } else { 1.0 };

            let comparison = ComparisonResult {
                n,
                bits,
                poly_degree: sieve_params.degree,
                m: m_u64,
                uniform: uniform_result,
                mcmc: mcmc_result,
                smooth_rate_ratio: ratio,
            };

            logger.log(
                "comparison",
                json!({
                    "n": n, "bits": bits,
                    "uniform_smooth_rate": u_rate,
                    "mcmc_smooth_rate": m_rate,
                    "ratio": ratio,
                }),
            );

            let sp_elapsed = sp_start.elapsed().as_secs_f64();
            global_index += 1;

            // Progress
            let done_frac = global_index as f64 / total_semiprimes as f64;
            let elapsed_total = global_start.elapsed().as_secs_f64();
            let eta = if done_frac > 0.0 {
                elapsed_total / done_frac - elapsed_total
            } else {
                0.0
            };

            eprintln!(
                "  [{}/{}] N={} uniform={:.3}% mcmc={:.3}% ratio={:.2}x  {:.1}s  ETA {:.0}s",
                global_index,
                total_semiprimes,
                n,
                u_rate * 100.0,
                m_rate * 100.0,
                ratio,
                sp_elapsed,
                eta,
            );

            // Sanity check
            if comparison.uniform.valid_candidates == 0 || comparison.mcmc.valid_candidates == 0 {
                logger.log(
                    "error",
                    json!({
                        "message": "zero valid candidates",
                        "n": n, "bits": bits,
                        "uniform_valid": comparison.uniform.valid_candidates,
                        "mcmc_valid": comparison.mcmc.valid_candidates,
                    }),
                );
            }

            bit_results.push(comparison);

            // Checkpoint
            write_checkpoint(&output_dir, global_index).ok();
        }

        // Aggregate for this bit size
        let count = bit_results.len();
        if count > 0 {
            let mean = |f: &dyn Fn(&ComparisonResult) -> f64| -> f64 {
                bit_results.iter().map(f).sum::<f64>() / count as f64
            };

            let u_smooth = mean(&|r| r.uniform.smooth_rate());
            let m_smooth = mean(&|r| r.mcmc.smooth_rate());

            let summary = BitSizeSummary {
                bits,
                count,
                uniform_mean_smooth_rate: u_smooth,
                mcmc_mean_smooth_rate: m_smooth,
                ratio: if u_smooth > 0.0 { m_smooth / u_smooth } else { 0.0 },
                uniform_mean_time_ms: mean(&|r| r.uniform.time_ms),
                mcmc_mean_time_ms: mean(&|r| r.mcmc.time_ms),
                uniform_mean_rat_log2: mean(&|r| r.uniform.mean_rat_norm_log2),
                mcmc_mean_rat_log2: mean(&|r| r.mcmc.mean_rat_norm_log2),
                uniform_mean_alg_log2: mean(&|r| r.uniform.mean_alg_norm_log2),
                mcmc_mean_alg_log2: mean(&|r| r.mcmc.mean_alg_norm_log2),
                uniform_mean_energy: mean(&|r| r.uniform.mean_energy),
                mcmc_mean_energy: mean(&|r| r.mcmc.mean_energy),
            };

            eprintln!(
                "  Summary: uniform={:.4}% mcmc={:.4}% ratio={:.2}x",
                summary.uniform_mean_smooth_rate * 100.0,
                summary.mcmc_mean_smooth_rate * 100.0,
                summary.ratio,
            );
            eprintln!(
                "  Energy: uniform={:.1} mcmc={:.1} (lower is better)",
                summary.uniform_mean_energy, summary.mcmc_mean_energy,
            );
            eprintln!();

            per_bit_summaries.push(summary);
        }

        all_results.extend(bit_results);
    }

    // Write summary JSON
    let report = E29Report {
        experiment: "E29: Novel Relation Collection via Joint Norm Biasing".to_string(),
        config,
        per_bit_size: per_bit_summaries,
        per_semiprime: all_results,
    };

    let summary_path = output_dir.join("summary.json");
    match serde_json::to_string_pretty(&report) {
        Ok(json_str) => {
            match std::fs::write(&summary_path, &json_str) {
                Ok(()) => eprintln!("Results written to {}", summary_path.display()),
                Err(e) => eprintln!("Error writing summary: {}", e),
            }
        }
        Err(e) => eprintln!("JSON serialization error: {}", e),
    }

    // Clean up checkpoint on successful completion
    let _ = std::fs::remove_file(output_dir.join("checkpoint.txt"));

    eprintln!("\nTotal time: {:.1}s", global_start.elapsed().as_secs_f64());
}
