//! cado-evolve CLI: Evolutionary GNFS parameter optimization.
//!
//! Modes:
//!   --mode=baseline --bits=120        Run baseline measurements
//!   --mode=evolve --bits=120 --quick  Evolutionary parameter search (quick)
//!   --mode=evolve --bits=120          Evolutionary parameter search (full)
//!   --mode=compare --bits=120         Compare evolved params vs defaults
//!
//! Options:
//!   --cado-dir=<path>   Path to CADO-NFS installation (default: ~/cado-nfs)
//!   --bits=<N>          Target semiprime bit size (default: 100)
//!   --quick             Quick mode (fewer islands, shorter runs)
//!   --generations=<N>   Override number of generations

use std::time::{Duration, Instant};

use rand::thread_rng;
use serde::Serialize;

use cado_evolve::analysis;
use cado_evolve::benchmark;
use cado_evolve::cado::CadoInstallation;
use cado_evolve::evolution::{FitnessCache, IslandConfig, ParamIslandModel};
use cado_evolve::fitness::{self, EvalConfig};
use cado_evolve::params::CadoParams;

/// CLI configuration parsed from command-line arguments.
struct CliConfig {
    mode: Mode,
    cado_dir: String,
    n_bits: u32,
    quick: bool,
    num_generations: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Baseline,
    Evolve,
    Compare,
}

/// Checkpoint data for evolution runs.
#[derive(Serialize)]
struct EvolutionCheckpoint {
    generation: u32,
    elapsed_secs: f64,
    global_best_fitness: f64,
    global_best_params: String,
    cache_entries: usize,
    island_fitnesses: Vec<(f64, f64)>,
}

fn parse_args() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();

    let mode = if args.iter().any(|a| a.contains("baseline")) {
        Mode::Baseline
    } else if args.iter().any(|a| a.contains("compare")) {
        Mode::Compare
    } else {
        Mode::Evolve
    };

    let cado_dir = args
        .iter()
        .find(|a| a.starts_with("--cado-dir="))
        .map(|a| a.strip_prefix("--cado-dir=").unwrap().to_string())
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{}/cado-nfs", home)
        });

    let n_bits = args
        .iter()
        .find(|a| a.starts_with("--bits="))
        .and_then(|a| a.strip_prefix("--bits=")?.parse::<u32>().ok())
        .unwrap_or(100);

    let quick = args.iter().any(|a| a == "--quick" || a == "-q");

    let num_generations = args
        .iter()
        .find(|a| a.starts_with("--generations="))
        .and_then(|a| a.strip_prefix("--generations=")?.parse::<u32>().ok());

    CliConfig {
        mode,
        cado_dir,
        n_bits,
        quick,
        num_generations,
    }
}

fn main() {
    env_logger::init();

    let config = parse_args();

    println!("========================================");
    println!("  cado-evolve: GNFS Parameter Evolution");
    println!("========================================");
    println!();

    // Validate CADO-NFS installation
    println!("Checking CADO-NFS installation at: {}", config.cado_dir);
    let install = match CadoInstallation::validate(&config.cado_dir) {
        Ok(install) => {
            println!("  CADO-NFS found: {}", install.cado_nfs_py.display());
            println!("  Python: {}", install.python);
            install
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            eprintln!("CADO-NFS is required. Install with:");
            eprintln!("  git clone https://gitlab.inria.fr/cado-nfs/cado-nfs.git ~/cado-nfs");
            eprintln!("  cd ~/cado-nfs && make");
            eprintln!();
            eprintln!("Or specify a custom path: --cado-dir=/path/to/cado-nfs");
            std::process::exit(1);
        }
    };
    println!();

    match config.mode {
        Mode::Baseline => run_baseline_mode(&install, &config),
        Mode::Evolve => run_evolve_mode(&install, &config),
        Mode::Compare => run_compare_mode(&install, &config),
    }

    println!();
    println!("========================================");
    println!("  Done.");
    println!("========================================");
}

/// Run baseline measurement mode.
fn run_baseline_mode(install: &CadoInstallation, config: &CliConfig) {
    println!("--- Baseline Mode ---");
    println!("  Target: {}-bit semiprimes", config.n_bits);
    println!();

    let mut rng = thread_rng();

    let bit_sizes: Vec<u32> = if config.quick {
        vec![config.n_bits]
    } else {
        // Test range around the target
        let start = config.n_bits.saturating_sub(20);
        let end = config.n_bits + 20;
        (start..=end).step_by(10).collect()
    };

    let tests_per_size = if config.quick { 3 } else { 10 };
    let timeout = if config.quick {
        Duration::from_secs(300)
    } else {
        Duration::from_secs(1800)
    };

    let suite = benchmark::run_baseline_suite(install, &bit_sizes, tests_per_size, timeout, &mut rng);

    // Save results
    save_json("cado_baselines.json", &suite);

    // Print summary
    println!();
    println!("Baseline Summary:");
    println!("  {:>6} | {:>8} | {:>10} | {:>10} | {:>10}",
        "Bits", "Success", "Median(s)", "Mean(s)", "Range(s)");
    println!("  {}", "-".repeat(60));

    for m in &suite.measurements {
        println!(
            "  {:>6} | {:>5}/{:<2} | {:>10.1} | {:>10.1} | {:.1}-{:.1}",
            m.n_bits,
            m.successes,
            m.num_tests,
            m.median_time_secs,
            m.mean_time_secs,
            m.min_time_secs,
            m.max_time_secs,
        );
    }
}

/// Run evolutionary parameter search mode.
fn run_evolve_mode(install: &CadoInstallation, config: &CliConfig) {
    println!("--- Evolution Mode ---");
    println!("  Target: {}-bit semiprimes", config.n_bits);
    println!("  Mode: {}", if config.quick { "QUICK" } else { "FULL" });
    println!();

    let mut rng = thread_rng();

    // Step 1: Quick baseline to get normalization factor
    println!("Step 1: Baseline measurement (3 tests)...");
    let baseline = benchmark::run_baseline(
        install,
        config.n_bits,
        3,
        Duration::from_secs(600),
        &mut rng,
    );

    let baseline_time = if baseline.median_time_secs > 0.0 {
        baseline.median_time_secs
    } else {
        60.0 // fallback
    };

    println!("  Baseline median: {:.1}s", baseline_time);
    println!();

    // Step 2: Initialize island model
    let island_config = if config.quick {
        IslandConfig::quick(config.n_bits)
    } else {
        IslandConfig::full(config.n_bits)
    };

    let num_generations = config.num_generations.unwrap_or(if config.quick { 10 } else { 50 });

    let mut model = ParamIslandModel::new(&island_config, &mut rng);
    let mut cache = FitnessCache::new(if config.quick { 1_000 } else { 10_000 });

    let eval_config = if config.quick {
        let mut ec = EvalConfig::quick(config.n_bits);
        ec.baseline_time_secs = baseline_time;
        ec
    } else {
        let mut ec = EvalConfig::full(config.n_bits);
        ec.baseline_time_secs = baseline_time;
        ec
    };

    println!(
        "Step 2: Evolution ({} islands × {} individuals, {} generations)",
        island_config.num_islands, island_config.island_size, num_generations
    );
    println!("  Migration: every {} gens", island_config.migration_interval);
    println!("  Culling: every {} gens", island_config.culling_interval);
    println!("  Mutation rate: {:.0}%", island_config.mutation_rate * 100.0);
    println!("  Timeout per CADO run: {:?}", eval_config.timeout);
    println!();

    let start = Instant::now();
    let report_interval = if config.quick { 1 } else { 5 };
    let checkpoint_interval = if config.quick { 5 } else { 10 };
    let mut convergence_history = Vec::new();

    for gen in 0..num_generations {
        // Generate fresh test semiprimes for this generation
        let test_semiprimes = fitness::generate_test_semiprimes(
            config.n_bits,
            eval_config.num_tests,
            &mut rng,
        );

        // Evaluate all individuals across all islands
        for island in &mut model.islands {
            for individual in &mut island.individuals {
                fitness::evaluate_individual_cached(
                    install,
                    individual,
                    &test_semiprimes,
                    &eval_config,
                    &mut cache,
                );
            }
        }

        // Evolve
        model.evolve_generation(&mut rng);

        // Track convergence
        if let Some(best) = model.global_best() {
            convergence_history.push(best.fitness);
        }

        // Report
        if (gen + 1) % report_interval == 0 || gen == 0 {
            let elapsed = start.elapsed();
            if let Some(best) = model.global_best() {
                let island_bests: Vec<String> = model
                    .island_fitness_summary()
                    .iter()
                    .map(|(best_f, _)| format!("{:.2}", best_f))
                    .collect();

                println!(
                    "  Gen {:>3}/{} | best: {:>8.3} | islands: [{}] | cache: {} | {:.1}s",
                    gen + 1,
                    num_generations,
                    best.fitness,
                    island_bests.join(", "),
                    cache.len(),
                    elapsed.as_secs_f64(),
                );
            }
        }

        // Checkpoint
        if (gen + 1) % checkpoint_interval == 0 {
            let checkpoint = EvolutionCheckpoint {
                generation: gen + 1,
                elapsed_secs: start.elapsed().as_secs_f64(),
                global_best_fitness: model
                    .global_best()
                    .map(|b| b.fitness)
                    .unwrap_or(0.0),
                global_best_params: model
                    .global_best()
                    .map(|b| b.params.summary())
                    .unwrap_or_default(),
                cache_entries: cache.len(),
                island_fitnesses: model.island_fitness_summary(),
            };

            let path = format!("cado_evolve_gen{:04}.json", gen + 1);
            save_json(&path, &checkpoint);
            println!("  [Checkpoint: {}]", path);
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("Evolution complete in {:.1}s", elapsed.as_secs_f64());
    println!("  Cache: {} entries", cache.len());
    println!();

    // Step 3: Report best configuration
    if let Some(best) = model.global_best() {
        println!("--- Best Evolved Configuration ---");
        println!("  Fitness: {:.4}", best.fitness);
        println!("  Parameters: {}", best.params);
        println!("  Summary: {}", best.params.summary());
        println!();

        // Save best params
        let params_path = format!("cado_evolved_{}bit.params", config.n_bits);
        if let Err(e) = best.params.to_param_file(std::path::Path::new(&params_path)) {
            eprintln!("  Warning: failed to save params: {}", e);
        } else {
            println!("  Parameters saved: {}", params_path);
        }

        // Step 4: Compare against baseline on fresh semiprimes
        println!();
        println!("Step 3: Comparison on fresh semiprimes...");
        let comparison = benchmark::compare_params(
            install,
            &best.params,
            config.n_bits,
            if config.quick { 3 } else { 5 },
            eval_config.timeout,
            &mut rng,
        );

        println!();
        println!("--- Comparison Results ---");
        println!(
            "  Default:  {:.1}s median, {:.0}% success",
            comparison.baseline_median_secs,
            comparison.baseline_success_rate * 100.0
        );
        println!(
            "  Evolved:  {:.1}s median, {:.0}% success",
            comparison.evolved_median_secs,
            comparison.evolved_success_rate * 100.0
        );
        println!("  Speedup:  {:.2}x", comparison.speedup);

        save_json(
            &format!("cado_comparison_{}bit.json", config.n_bits),
            &comparison,
        );

        // Step 4: Post-evolution analysis
        println!();
        println!("Step 4: Post-evolution analysis...");
        println!();

        // Collect all individuals across islands
        let all_individuals: Vec<cado_evolve::evolution::ParamIndividual> = model
            .islands
            .iter()
            .flat_map(|island| island.individuals.iter().cloned())
            .collect();

        let report = analysis::generate_report(
            &all_individuals,
            &convergence_history,
            &[comparison],
            config.n_bits,
        );

        analysis::print_report_summary(&report);

        save_json(
            &format!("cado_analysis_{}bit.json", config.n_bits),
            &report,
        );
        println!("  Analysis report saved: cado_analysis_{}bit.json", config.n_bits);
    }
}

/// Run comparison mode.
fn run_compare_mode(install: &CadoInstallation, config: &CliConfig) {
    println!("--- Compare Mode ---");
    println!("  Target: {}-bit semiprimes", config.n_bits);
    println!();

    // Look for saved evolved parameters
    let params_path = format!("cado_evolved_{}bit.params", config.n_bits);
    if !std::path::Path::new(&params_path).exists() {
        eprintln!("Error: No evolved parameters found at {}", params_path);
        eprintln!("Run --mode=evolve first to generate parameters.");
        std::process::exit(1);
    }

    // Parse the parameter file
    let content = match std::fs::read_to_string(&params_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", params_path, e);
            std::process::exit(1);
        }
    };

    let evolved_params = match parse_params_file(&content, config.n_bits) {
        Some(p) => p,
        None => {
            eprintln!("Error: Failed to parse parameter file {}", params_path);
            std::process::exit(1);
        }
    };

    println!("  Loaded evolved parameters from: {}", params_path);
    println!("  {}", evolved_params.summary());
    println!();

    let mut rng = thread_rng();
    let num_tests = if config.quick { 5 } else { 20 };
    let timeout = if config.quick {
        Duration::from_secs(300)
    } else {
        Duration::from_secs(1800)
    };

    let comparison = benchmark::compare_params(
        install,
        &evolved_params,
        config.n_bits,
        num_tests,
        timeout,
        &mut rng,
    );

    println!();
    println!("--- Final Comparison ---");
    println!(
        "  Default:  {:.1}s median, {:.0}% success rate",
        comparison.baseline_median_secs,
        comparison.baseline_success_rate * 100.0
    );
    println!(
        "  Evolved:  {:.1}s median, {:.0}% success rate",
        comparison.evolved_median_secs,
        comparison.evolved_success_rate * 100.0
    );
    println!("  Speedup:  {:.2}x", comparison.speedup);
    println!();

    if comparison.speedup > 1.0 {
        println!("  Evolved configuration is FASTER by {:.1}%",
            (comparison.speedup - 1.0) * 100.0);
    } else if comparison.speedup < 1.0 {
        println!("  Evolved configuration is SLOWER by {:.1}%",
            (1.0 - comparison.speedup) * 100.0);
    } else {
        println!("  No significant difference.");
    }

    save_json(
        &format!("cado_compare_{}bit.json", config.n_bits),
        &comparison,
    );
}

/// Parse a CADO-NFS parameter file back into CadoParams.
fn parse_params_file(content: &str, n_bits: u32) -> Option<CadoParams> {
    let mut params = CadoParams::default_for_bits(n_bits);

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            continue;
        }

        let key = parts[0].trim();
        let value = parts[1].trim();

        match key {
            "tasks.polyselect.degree" => {
                if let Ok(v) = value.parse() { params.poly_degree = v; }
            }
            "tasks.polyselect.admax" => {
                if let Ok(v) = value.parse() { params.poly_admax = v; }
            }
            "tasks.polyselect.incr" => {
                if let Ok(v) = value.parse() { params.poly_incr = v; }
            }
            "tasks.lim0" => {
                if let Ok(v) = value.parse() { params.fb_rational_bound = v; }
            }
            "tasks.lim1" => {
                if let Ok(v) = value.parse() { params.fb_algebraic_bound = v; }
            }
            "tasks.lpb0" => {
                if let Ok(v) = value.parse() { params.lp_rational_bits = v; }
            }
            "tasks.lpb1" => {
                if let Ok(v) = value.parse() { params.lp_algebraic_bits = v; }
            }
            "tasks.sieve.mfb0" => {
                if let Ok(v) = value.parse() { params.sieve_mfbr = v; }
            }
            "tasks.sieve.mfb1" => {
                if let Ok(v) = value.parse() { params.sieve_mfba = v; }
            }
            "tasks.sieve.qrange" => {
                if let Ok(v) = value.parse() { params.sieve_qrange = v; }
            }
            _ => {} // Ignore unknown parameters
        }
    }

    Some(params)
}

/// Save a serializable value as JSON.
fn save_json<T: Serialize>(path: &str, data: &T) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, &json) {
                eprintln!("  Warning: failed to write {}: {}", path, e);
            }
        }
        Err(e) => {
            eprintln!("  Warning: failed to serialize {}: {}", path, e);
        }
    }
}
