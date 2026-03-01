//! E29 Production: Sieve-scored MCMC benchmark.
//!
//! Runs the production sieve-MCMC pipeline on semiprimes at various bit sizes
//! and outputs timing/relation statistics for head-to-head comparison with CADO-NFS.

use std::fs;
use std::path::Path;
use std::time::Instant;

use classical_nfs::polynomial::select_polynomial;
use factoring_core::generate_rsa_target;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use novel_relation_sampler::sieve_mcmc::{collect_relations, SieveMcmcConfig};

#[derive(Debug, Serialize)]
struct ProductionResult {
    n_hex: String,
    n_decimal: String,
    bits: u32,
    poly_degree: usize,
    m_bits: u64,
    total_relations: usize,
    full_relations: usize,
    partial_relations: usize,
    candidates_found: usize,
    rows_processed: usize,
    setup_ms: f64,
    sieve_ms: f64,
    mcmc_ms: f64,
    cofactor_ms: f64,
    total_ms: f64,
    rels_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct BitSummary {
    bits: u32,
    semiprimes_tested: usize,
    mean_relations: f64,
    mean_full: f64,
    mean_partial: f64,
    mean_candidates: f64,
    mean_total_ms: f64,
    mean_rels_per_sec: f64,
    mean_sieve_ms: f64,
    mean_mcmc_ms: f64,
    mean_cofactor_ms: f64,
}

#[derive(Debug, Serialize)]
struct ProductionOutput {
    experiment: String,
    threads: usize,
    per_bit_size: Vec<BitSummary>,
    per_semiprime: Vec<ProductionResult>,
}

fn main() {
    let bit_sizes: Vec<u32> = vec![32, 48, 64, 80, 96, 112, 128];
    let semiprimes_per_size = 5;
    let seed_base = 42u64;
    let threads = rayon::current_num_threads();

    eprintln!("=== E29 Production Sieve-MCMC Benchmark ===");
    eprintln!("Threads: {}", threads);
    eprintln!("Bit sizes: {:?}", bit_sizes);
    eprintln!("Semiprimes per size: {}", semiprimes_per_size);
    eprintln!();

    let mut all_results: Vec<ProductionResult> = Vec::new();
    let mut per_bit_summaries: Vec<BitSummary> = Vec::new();
    let global_start = Instant::now();

    for &bits in &bit_sizes {
        eprintln!("--- {} bits ---", bits);
        let config = SieveMcmcConfig::for_bits(bits);
        let degree = match bits {
            0..=96 => 3,
            97..=128 => 4,
            _ => 5,
        };

        let mut rng_gen = StdRng::seed_from_u64(seed_base + bits as u64);
        let mut bit_results: Vec<ProductionResult> = Vec::new();

        for i in 0..semiprimes_per_size {
            let target = generate_rsa_target(bits, &mut rng_gen);
            let n_big = &target.n;
            let n_hex = format!("{:x}", n_big);

            let poly = select_polynomial(n_big, degree);
            let m_u64 = poly.m.to_u64().expect("m must fit in u64");
            let m_bits = poly.m.bits();

            eprintln!("  [{}/{}] N={} (m={} bits, degree={})",
                      i + 1, semiprimes_per_size, &n_hex[..n_hex.len().min(16)], m_bits, degree);

            let (relations, timings) = collect_relations(&poly, m_u64, &config);

            let rels_per_sec = if timings.total_ms > 0.0 {
                relations.len() as f64 / (timings.total_ms / 1000.0)
            } else {
                0.0
            };

            eprintln!("    {} relations ({} full, {} partial) in {:.0}ms = {:.0} rels/sec",
                      relations.len(), timings.full_relations, timings.partial_relations,
                      timings.total_ms, rels_per_sec);
            eprintln!("    breakdown: sieve {:.0}ms, mcmc {:.0}ms, cofactor {:.0}ms, setup {:.0}ms",
                      timings.sieve_ms, timings.mcmc_ms, timings.cofactor_ms, timings.setup_ms);

            let result = ProductionResult {
                n_hex: n_hex.clone(),
                n_decimal: target.n.to_string(),
                bits,
                poly_degree: degree,
                m_bits,
                total_relations: relations.len(),
                full_relations: timings.full_relations,
                partial_relations: timings.partial_relations,
                candidates_found: timings.candidates_found,
                rows_processed: timings.rows_processed,
                setup_ms: timings.setup_ms,
                sieve_ms: timings.sieve_ms,
                mcmc_ms: timings.mcmc_ms,
                cofactor_ms: timings.cofactor_ms,
                total_ms: timings.total_ms,
                rels_per_sec,
            };
            bit_results.push(result);
        }

        // Aggregate
        let n = bit_results.len() as f64;
        let summary = BitSummary {
            bits,
            semiprimes_tested: bit_results.len(),
            mean_relations: bit_results.iter().map(|r| r.total_relations as f64).sum::<f64>() / n,
            mean_full: bit_results.iter().map(|r| r.full_relations as f64).sum::<f64>() / n,
            mean_partial: bit_results.iter().map(|r| r.partial_relations as f64).sum::<f64>() / n,
            mean_candidates: bit_results.iter().map(|r| r.candidates_found as f64).sum::<f64>() / n,
            mean_total_ms: bit_results.iter().map(|r| r.total_ms).sum::<f64>() / n,
            mean_rels_per_sec: bit_results.iter().map(|r| r.rels_per_sec).sum::<f64>() / n,
            mean_sieve_ms: bit_results.iter().map(|r| r.sieve_ms).sum::<f64>() / n,
            mean_mcmc_ms: bit_results.iter().map(|r| r.mcmc_ms).sum::<f64>() / n,
            mean_cofactor_ms: bit_results.iter().map(|r| r.cofactor_ms).sum::<f64>() / n,
        };

        eprintln!("  Summary: {:.0} rels, {:.0} rels/sec, {:.0}ms total",
                  summary.mean_relations, summary.mean_rels_per_sec, summary.mean_total_ms);
        per_bit_summaries.push(summary);
        all_results.extend(bit_results);
    }

    let global_elapsed = global_start.elapsed().as_secs_f64();
    eprintln!("\nTotal elapsed: {:.1}s", global_elapsed);

    // Write output
    let output = ProductionOutput {
        experiment: "E29 Production Sieve-MCMC".to_string(),
        threads,
        per_bit_size: per_bit_summaries,
        per_semiprime: all_results,
    };

    let output_dir = Path::new("data/e29_scaling");
    fs::create_dir_all(output_dir).ok();
    let output_path = output_dir.join("production_results.json");
    let json = serde_json::to_string_pretty(&output).expect("serialize");
    fs::write(&output_path, &json).expect("write output");
    eprintln!("Results written to {}", output_path.display());
}
