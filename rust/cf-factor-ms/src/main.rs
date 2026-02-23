//! Murru-Salvatori experiment runner.
//!
//! Evaluates the CF-based BSGS and regulator-guided factoring approaches
//! across multiple bit sizes, outputting JSON results.

use std::time::Instant;

use factoring_core::generate_rsa_target;
use rand::thread_rng;
use serde::Serialize;

use cf_factor_ms::bsgs::{bsgs_factor, linear_walk_factor, BsgsConfig};
use cf_factor_ms::class_number_real::class_number_real_estimate;
use cf_factor_ms::infrastructure::walk_infrastructure;
use cf_factor_ms::pipeline::{factor_ms_with_config, MsPipelineConfig};
use cf_factor_ms::regulator_guided::{regulator_guided_factor, RegulatorGuidedConfig};

#[derive(Serialize)]
struct ExperimentResult {
    bit_size: u32,
    n: String,
    factor_found: bool,
    factor: Option<String>,
    stage: String,
    time_ms: f64,
}

#[derive(Serialize)]
struct InfrastructureResult {
    bit_size: u32,
    n: String,
    forms_walked: usize,
    final_distance: f64,
    class_number_estimate: u64,
    regulator_estimate: f64,
}

#[derive(Serialize)]
struct StageComparison {
    bit_size: u32,
    n: String,
    linear_walk_found: bool,
    linear_walk_steps: usize,
    linear_walk_ms: f64,
    bsgs_found: bool,
    bsgs_baby_steps: usize,
    bsgs_giant_steps: usize,
    bsgs_ms: f64,
    regulator_found: bool,
    regulator_forms: usize,
    regulator_ms: f64,
}

#[derive(Serialize)]
struct FullReport {
    experiment: String,
    description: String,
    infrastructure: Vec<InfrastructureResult>,
    pipeline_results: Vec<ExperimentResult>,
    stage_comparison: Vec<StageComparison>,
}

fn main() {
    println!("=== CF-Factor-MS: Murru-Salvatori Continued Fraction Factoring ===");
    println!("Prong 4 of the six-prong factoring plan");
    println!();

    let mut rng = thread_rng();
    let mut infra_results = Vec::new();
    let mut pipeline_results = Vec::new();
    let mut stage_comparisons = Vec::new();

    // Section 1: Infrastructure analysis
    println!("--- Section 1: Infrastructure Analysis ---");
    for &bits in &[16u32, 24, 32, 40, 48] {
        let target = generate_rsa_target(bits, &mut rng);
        let forms = walk_infrastructure(&target.n, 500);
        let cn = class_number_real_estimate(&target.n);
        let final_dist = forms.last().map(|f| f.distance).unwrap_or(0.0);

        println!(
            "  {:>3}-bit: {} forms, dist={:.2}, h~{}, R~{:.2}",
            bits,
            forms.len(),
            final_dist,
            cn.class_number,
            cn.regulator,
        );

        infra_results.push(InfrastructureResult {
            bit_size: bits,
            n: target.n.to_string(),
            forms_walked: forms.len(),
            final_distance: final_dist,
            class_number_estimate: cn.class_number,
            regulator_estimate: cn.regulator,
        });
    }

    // Section 2: Full pipeline benchmarks
    println!();
    println!("--- Section 2: Pipeline Benchmarks ---");
    let config = MsPipelineConfig::default();

    for &bits in &[16u32, 20, 24, 28, 32, 36, 40, 48] {
        let target = generate_rsa_target(bits, &mut rng);
        let start = Instant::now();

        let result = factor_ms_with_config(&target.n, &config);
        let elapsed = start.elapsed();

        let (factor_found, factor_str, stage) = match result {
            Some(ref r) => (
                r.factor_result.complete,
                r.factor_result.factors.first().map(|f| f.to_string()),
                r.stage.clone(),
            ),
            None => (false, None, "error".to_string()),
        };

        let verified = if factor_found {
            if let Some(ref r) = result {
                if let Some(ref f) = r.factor_result.factors.first() {
                    let q = &target.n / *f;
                    *f * &q == target.n
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        println!(
            "  {:>3}-bit: {} via {} | {:.1}ms{}",
            bits,
            if factor_found { "FACTORED" } else { "FAILED  " },
            stage,
            elapsed.as_secs_f64() * 1000.0,
            if verified { " [verified]" } else { "" }
        );

        pipeline_results.push(ExperimentResult {
            bit_size: bits,
            n: target.n.to_string(),
            factor_found,
            factor: factor_str,
            stage,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // Section 3: Stage comparison
    println!();
    println!("--- Section 3: Stage Comparison (32-bit) ---");
    for _ in 0..5 {
        let target = generate_rsa_target(32, &mut rng);

        // Linear walk
        let start = Instant::now();
        let lw = linear_walk_factor(&target.n, 1000);
        let lw_ms = start.elapsed().as_secs_f64() * 1000.0;

        // BSGS
        let start = Instant::now();
        let bs = bsgs_factor(
            &target.n,
            &BsgsConfig {
                baby_steps: 500,
                giant_steps: 500,
                check_ambiguous: true,
            },
        );
        let bs_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Regulator-guided
        let start = Instant::now();
        let rg = regulator_guided_factor(
            &target.n,
            &RegulatorGuidedConfig {
                max_regulator_terms: 5000,
                neighborhood_size: 100,
                use_class_number: true,
                max_multiples: 10,
            },
        );
        let rg_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "  N={}: walk={}/{:.1}ms bsgs={}/{:.1}ms reg={}/{:.1}ms",
            &target.n.to_string()[..8.min(target.n.to_string().len())],
            if lw.factor.is_some() { "Y" } else { "N" },
            lw_ms,
            if bs.factor.is_some() { "Y" } else { "N" },
            bs_ms,
            if rg.factor.is_some() { "Y" } else { "N" },
            rg_ms,
        );

        stage_comparisons.push(StageComparison {
            bit_size: 32,
            n: target.n.to_string(),
            linear_walk_found: lw.factor.is_some(),
            linear_walk_steps: lw.baby_steps_taken,
            linear_walk_ms: lw_ms,
            bsgs_found: bs.factor.is_some(),
            bsgs_baby_steps: bs.baby_steps_taken,
            bsgs_giant_steps: bs.giant_steps_taken,
            bsgs_ms: bs_ms,
            regulator_found: rg.factor.is_some(),
            regulator_forms: rg.forms_checked,
            regulator_ms: rg_ms,
        });
    }

    // Output JSON report
    let report = FullReport {
        experiment: "cf-factor-ms".to_string(),
        description: "Murru-Salvatori CF + BSGS + regulator-guided factoring (Prong 4)"
            .to_string(),
        infrastructure: infra_results,
        pipeline_results,
        stage_comparison: stage_comparisons,
    };

    println!();
    println!("--- JSON Report ---");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
