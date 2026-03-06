//! E29 Production: Flat sieve vs Special-q lattice sieve benchmark.
//!
//! Runs both pipelines on semiprimes at various bit sizes and outputs
//! timing/relation statistics for head-to-head comparison.

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
use novel_relation_sampler::specialq::{collect_relations_specialq, SpecialQConfig};

#[derive(Debug, Serialize)]
struct ProductionResult {
    method: String,
    n_hex: String,
    bits: u32,
    poly_degree: usize,
    total_relations: usize,
    full_relations: usize,
    partial_relations: usize,
    candidates_found: usize,
    setup_ms: f64,
    sieve_ms: f64,
    scan_ms: f64,
    cofactor_ms: f64,
    total_ms: f64,
    rels_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct BitSummary {
    bits: u32,
    method: String,
    semiprimes_tested: usize,
    mean_relations: f64,
    mean_full: f64,
    mean_partial: f64,
    mean_total_ms: f64,
    mean_rels_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct ProductionOutput {
    experiment: String,
    threads: usize,
    summaries: Vec<BitSummary>,
    per_semiprime: Vec<ProductionResult>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let min_bits: u32 = args
        .iter()
        .position(|a| a == "--min-bits")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let semiprimes_per_size: usize = args
        .iter()
        .position(|a| a == "--semiprimes")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let all_sizes: Vec<u32> = vec![32, 48, 64, 80, 96, 112, 128];
    let bit_sizes: Vec<u32> = all_sizes.into_iter().filter(|&b| b >= min_bits).collect();
    let seed_base = 42u64;
    let threads = rayon::current_num_threads();

    eprintln!("=== E29 Flat vs Special-Q Lattice Sieve Benchmark ===");
    eprintln!("Threads: {}", threads);
    eprintln!("Bit sizes: {:?}", bit_sizes);
    eprintln!("Semiprimes per size: {}", semiprimes_per_size);
    eprintln!();

    let mut all_results: Vec<ProductionResult> = Vec::new();
    let mut summaries: Vec<BitSummary> = Vec::new();
    let global_start = Instant::now();

    for &bits in &bit_sizes {
        let degree = match bits {
            0..=96 => 3,
            97..=128 => 4,
            _ => 5,
        };

        let flat_config = SieveMcmcConfig::for_bits(bits);
        let sq_config = SpecialQConfig::for_bits(bits);

        eprintln!("=== {} bits (degree {}) ===", bits, degree);

        let mut rng_gen = StdRng::seed_from_u64(seed_base + bits as u64);

        let mut flat_results: Vec<ProductionResult> = Vec::new();
        let mut sq_results: Vec<ProductionResult> = Vec::new();

        for i in 0..semiprimes_per_size {
            let target = generate_rsa_target(bits, &mut rng_gen);
            let n_big = &target.n;
            let n_hex = format!("{:x}", n_big);

            let poly = select_polynomial(n_big, degree);
            let m_u64 = poly.m.to_u64().expect("m must fit in u64");

            eprintln!(
                "  [{}/{}] N={}...",
                i + 1,
                semiprimes_per_size,
                &n_hex[..n_hex.len().min(16)]
            );

            // --- Flat sieve ---
            let (flat_rels, flat_t) = collect_relations(&poly, m_u64, &flat_config);
            let flat_rps = if flat_t.total_ms > 0.0 {
                flat_rels.len() as f64 / (flat_t.total_ms / 1000.0)
            } else {
                0.0
            };
            eprintln!(
                "    Flat:  {} rels ({} full) in {:.0}ms = {:.0} rels/sec",
                flat_rels.len(),
                flat_t.full_relations,
                flat_t.total_ms,
                flat_rps
            );

            flat_results.push(ProductionResult {
                method: "flat".into(),
                n_hex: n_hex.clone(),
                bits,
                poly_degree: degree,
                total_relations: flat_rels.len(),
                full_relations: flat_t.full_relations,
                partial_relations: flat_t.partial_relations,
                candidates_found: flat_t.candidates_found,
                setup_ms: flat_t.setup_ms,
                sieve_ms: flat_t.sieve_ms,
                scan_ms: flat_t.mcmc_ms,
                cofactor_ms: flat_t.cofactor_ms,
                total_ms: flat_t.total_ms,
                rels_per_sec: flat_rps,
            });

            // --- Special-q lattice sieve ---
            let (sq_rels, sq_t) = collect_relations_specialq(&poly, m_u64, &sq_config);
            let sq_rps = if sq_t.total_ms > 0.0 {
                sq_rels.len() as f64 / (sq_t.total_ms / 1000.0)
            } else {
                0.0
            };
            eprintln!(
                "    SQ:    {} rels ({} full) in {:.0}ms = {:.0} rels/sec",
                sq_rels.len(),
                sq_t.full_relations,
                sq_t.total_ms,
                sq_rps
            );

            sq_results.push(ProductionResult {
                method: "specialq".into(),
                n_hex: n_hex.clone(),
                bits,
                poly_degree: degree,
                total_relations: sq_rels.len(),
                full_relations: sq_t.full_relations,
                partial_relations: sq_t.partial_relations,
                candidates_found: sq_t.candidates_found,
                setup_ms: sq_t.setup_ms,
                sieve_ms: sq_t.sieve_ms,
                scan_ms: sq_t.mcmc_ms,
                cofactor_ms: sq_t.cofactor_ms,
                total_ms: sq_t.total_ms,
                rels_per_sec: sq_rps,
            });
        }

        // Aggregate summaries
        for (method, results) in [("flat", &flat_results), ("specialq", &sq_results)] {
            let n = results.len() as f64;
            summaries.push(BitSummary {
                bits,
                method: method.into(),
                semiprimes_tested: results.len(),
                mean_relations: results.iter().map(|r| r.total_relations as f64).sum::<f64>() / n,
                mean_full: results.iter().map(|r| r.full_relations as f64).sum::<f64>() / n,
                mean_partial: results.iter().map(|r| r.partial_relations as f64).sum::<f64>() / n,
                mean_total_ms: results.iter().map(|r| r.total_ms).sum::<f64>() / n,
                mean_rels_per_sec: results.iter().map(|r| r.rels_per_sec).sum::<f64>() / n,
            });
        }

        // Print comparison
        let flat_mean_rps: f64 =
            flat_results.iter().map(|r| r.rels_per_sec).sum::<f64>() / flat_results.len() as f64;
        let sq_mean_rps: f64 =
            sq_results.iter().map(|r| r.rels_per_sec).sum::<f64>() / sq_results.len() as f64;
        let speedup = if flat_mean_rps > 0.0 {
            sq_mean_rps / flat_mean_rps
        } else if sq_mean_rps > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        eprintln!(
            "  >> {}-bit: flat={:.0} rels/sec, SQ={:.0} rels/sec, speedup={:.1}x",
            bits, flat_mean_rps, sq_mean_rps, speedup
        );
        eprintln!();

        all_results.extend(flat_results);
        all_results.extend(sq_results);
    }

    let global_elapsed = global_start.elapsed().as_secs_f64();
    eprintln!("Total elapsed: {:.1}s", global_elapsed);

    // Write output
    let output = ProductionOutput {
        experiment: "E29 Flat vs Special-Q Lattice Sieve".to_string(),
        threads,
        summaries,
        per_semiprime: all_results,
    };

    let output_dir = Path::new("data/e29_scaling");
    fs::create_dir_all(output_dir).ok();
    let output_path = output_dir.join("specialq_results.json");
    let json = serde_json::to_string_pretty(&output).expect("serialize");
    fs::write(&output_path, &json).expect("write output");
    eprintln!("Results written to {}", output_path.display());
}
