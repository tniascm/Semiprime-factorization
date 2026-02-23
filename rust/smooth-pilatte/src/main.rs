//! Smooth-Pilatte experiment runner.
//!
//! Evaluates the Pilatte lattice-geometric smooth relation approach across
//! multiple bit sizes and configurations, outputting JSON results.

use std::time::Instant;

use factoring_core::generate_rsa_target;
use rand::thread_rng;
use serde::Serialize;

use smooth_pilatte::extract::{factor_smooth_pilatte_with_config, PilatteConfig};
use smooth_pilatte::lattice::{
    build_pilatte_lattice, extract_exponent_vectors, pilatte_dimension,
};

#[derive(Serialize)]
struct ExperimentResult {
    bit_size: u32,
    n: String,
    factor_found: bool,
    factor: Option<String>,
    dimension: usize,
    vectors_tested: usize,
    smooth_relations: usize,
    smooth_rate: f64,
    dependencies_found: usize,
    gcd_attempts: usize,
    used_weighted: bool,
    time_ms: f64,
}

#[derive(Serialize)]
struct LatticeQualityResult {
    bit_size: u32,
    dimension: usize,
    hermite_factor: f64,
    orthogonality_defect: f64,
    shortest_vector_norm: f64,
    exponent_vectors_extracted: usize,
    smooth_probability: f64,
}

#[derive(Serialize)]
struct FullReport {
    experiment: String,
    description: String,
    lattice_quality: Vec<LatticeQualityResult>,
    factoring_results: Vec<ExperimentResult>,
}

fn main() {
    println!("=== Smooth-Pilatte: Lattice-Geometric Smooth Relation Finder ===");
    println!("Prong 2 of the six-prong factoring plan");
    println!();

    let mut rng = thread_rng();
    let mut lattice_quality_results = Vec::new();
    let mut factoring_results = Vec::new();

    // Section 1: Lattice quality analysis
    println!("--- Section 1: Lattice Quality Analysis ---");
    for &bits in &[16u32, 24, 32, 40, 48, 56, 64] {
        let target = generate_rsa_target(bits, &mut rng);
        let dim = pilatte_dimension(bits);
        let result = build_pilatte_lattice(&target.n, dim);
        let vectors = extract_exponent_vectors(&result);

        let quality = LatticeQualityResult {
            bit_size: bits,
            dimension: dim,
            hermite_factor: result.quality.hermite_factor,
            orthogonality_defect: result.quality.orthogonality_defect,
            shortest_vector_norm: result.quality.shortest_vector_norm,
            exponent_vectors_extracted: vectors.len(),
            smooth_probability: result.params.smooth_probability,
        };

        println!(
            "  {:>3}-bit: dim={:>2}, hermite={:.4}, ortho_defect={:.4}, svn={:.2}, vecs={}, smooth_prob={:.2e}",
            bits, dim, quality.hermite_factor, quality.orthogonality_defect,
            quality.shortest_vector_norm, quality.exponent_vectors_extracted,
            quality.smooth_probability
        );

        lattice_quality_results.push(quality);
    }

    // Section 2: Factoring benchmarks
    println!();
    println!("--- Section 2: Factoring Benchmarks ---");

    let config = PilatteConfig::default();

    for &bits in &[16u32, 20, 24, 28, 32, 36, 40, 48] {
        let target = generate_rsa_target(bits, &mut rng);
        let start = Instant::now();

        let result = factor_smooth_pilatte_with_config(&target.n, &config);
        let elapsed = start.elapsed();

        let (factor_found, factor_str, detail) = match result {
            Some(ref r) => (
                r.factor_result.complete,
                r.factor_result.factors.first().map(|f| f.to_string()),
                r.clone(),
            ),
            None => {
                println!("  {:>3}-bit: ERROR - no result returned", bits);
                continue;
            }
        };

        let verified = if factor_found {
            if let Some(ref f) = detail.factor_result.factors.first() {
                let q = &target.n / *f;
                *f * &q == target.n
            } else {
                false
            }
        } else {
            false
        };

        println!(
            "  {:>3}-bit: {} | dim={:>2} | vecs={:>5} | smooth={:>3} (rate={:.4}) | deps={} | gcd={} | {:.1}ms{}",
            bits,
            if factor_found { "FACTORED" } else { "FAILED  " },
            detail.dimension,
            detail.vectors_tested,
            detail.smooth_relations,
            detail.smooth_rate,
            detail.dependencies_found,
            detail.gcd_attempts,
            elapsed.as_secs_f64() * 1000.0,
            if verified { " [verified]" } else { "" }
        );

        factoring_results.push(ExperimentResult {
            bit_size: bits,
            n: target.n.to_string(),
            factor_found,
            factor: factor_str,
            dimension: detail.dimension,
            vectors_tested: detail.vectors_tested,
            smooth_relations: detail.smooth_relations,
            smooth_rate: detail.smooth_rate,
            dependencies_found: detail.dependencies_found,
            gcd_attempts: detail.gcd_attempts,
            used_weighted: detail.used_weighted,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // Section 3: Dimension scaling comparison
    println!();
    println!("--- Section 3: Dimension Scaling ---");
    let target_32 = generate_rsa_target(32, &mut rng);

    for dim_mult in [1.0f64, 1.5, 2.0, 3.0] {
        let base_dim = pilatte_dimension(32);
        let dim = (base_dim as f64 * dim_mult).ceil() as usize;
        let config = PilatteConfig {
            dimension_override: dim,
            max_enum_vectors: 3000,
            ..PilatteConfig::default()
        };

        let start = Instant::now();
        let result = factor_smooth_pilatte_with_config(&target_32.n, &config);
        let elapsed = start.elapsed();

        let found = result
            .as_ref()
            .map(|r| r.factor_result.complete)
            .unwrap_or(false);

        println!(
            "  dim={:>3} ({:.1}x): {} in {:.1}ms",
            dim,
            dim_mult,
            if found { "FACTORED" } else { "FAILED  " },
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // Output JSON report
    let report = FullReport {
        experiment: "smooth-pilatte".to_string(),
        description: "Pilatte lattice-geometric smooth relation finder (Prong 2)".to_string(),
        lattice_quality: lattice_quality_results,
        factoring_results,
    };

    println!();
    println!("--- JSON Report ---");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
