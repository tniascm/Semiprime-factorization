//! Alpha-Evolve v4: M4-tuned island-model evolutionary search for factoring algorithms.
//!
//! Enterprise-grade research infrastructure with:
//! - Deterministic seeding (ChaCha8Rng) for reproducible runs
//! - Staged evaluation protocol (warmup → exploration → full)
//! - Explicit Rayon ThreadPool (default 4 threads for M4 P-cores)
//! - Per-step UTC timestamps and structured logging
//! - Run directory management with config persistence

use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::cmp::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use alpha_evolve::analysis;
use alpha_evolve::evolution::{FitnessCache, IslandModel};
use alpha_evolve::fitness::{
    evaluate_fitness_cascaded, evaluate_fitness_multiobjective, evaluate_fitness_on_suite,
    semiprime_ladder, FitnessResult,
};
use alpha_evolve::novelty::NoveltyArchive;
use alpha_evolve::{
    seed_cf_regulator_jump, seed_dixon_smooth, seed_ecm_cf_hybrid, seed_fermat_like,
    seed_hart_like, seed_lattice_gcd, seed_lehman_like, seed_pm1_rho_hybrid,
    seed_pollard_rho, seed_trial_like, Program,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "alpha-evolve", about = "Evolutionary search for factoring algorithms")]
struct Cli {
    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of Rayon threads (default: 4 for M4 P-cores)
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Output directory
    #[arg(long, default_value = "runs")]
    output_dir: String,

    /// Stage: 0=all, 1=warmup, 2=exploration, 3=full
    #[arg(long, default_value_t = 0)]
    stage: u8,

    /// Quick mode (alias for --stage 1)
    #[arg(short, long)]
    quick: bool,

    /// Verbose per-generation output
    #[arg(short, long)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// Stage configuration
// ---------------------------------------------------------------------------

struct StageConfig {
    name: &'static str,
    num_islands: usize,
    island_size: usize,
    num_generations: u32,
    bit_sizes: Vec<u32>,
    count_per_size: usize,
    timeout_ms: u128,
    report_interval: u32,
    checkpoint_interval: u32,
    cache_size: usize,
    novelty_k: usize,
    novelty_max: usize,
}

fn stage_configs() -> Vec<StageConfig> {
    vec![
        StageConfig {
            name: "warmup",
            num_islands: 5,
            island_size: 20,
            num_generations: 50,
            bit_sizes: vec![8, 12, 16],
            count_per_size: 5,
            timeout_ms: 100,
            report_interval: 10,
            checkpoint_interval: 50,
            cache_size: 10_000,
            novelty_k: 5,
            novelty_max: 1_000,
        },
        StageConfig {
            name: "exploration",
            num_islands: 10,
            island_size: 50,
            num_generations: 200,
            bit_sizes: vec![16, 20, 24, 28, 32],
            count_per_size: 3,
            timeout_ms: 200,
            report_interval: 20,
            checkpoint_interval: 50,
            cache_size: 30_000,
            novelty_k: 5,
            novelty_max: 5_000,
        },
        StageConfig {
            name: "full",
            num_islands: 20,
            island_size: 100,
            num_generations: 1000,
            bit_sizes: vec![16, 20, 24, 28, 32, 36, 40, 48, 56, 64],
            count_per_size: 2,
            timeout_ms: 500,
            report_interval: 50,
            checkpoint_interval: 100,
            cache_size: 50_000,
            novelty_k: 5,
            novelty_max: 10_000,
        },
    ]
}

// ---------------------------------------------------------------------------
// Checkpoint / serialization
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Checkpoint {
    stage: String,
    generation: u32,
    elapsed_secs: f64,
    cache_entries: usize,
    novelty_archive_size: usize,
    unique_behaviors: usize,
    global_best_fitness: f64,
    global_best_program: String,
    global_best_nodes: usize,
    top_programs: Vec<ProgramEntry>,
}

#[derive(Serialize)]
struct ProgramEntry {
    fitness: f64,
    nodes: usize,
    program: String,
}

#[derive(Serialize)]
struct RunConfig {
    seed: u64,
    threads: usize,
    stages: Vec<String>,
    rustc_version: String,
}

// ---------------------------------------------------------------------------
// Timestamped logging
// ---------------------------------------------------------------------------

fn log_ts(stage: &str, msg: &str) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    println!("[{:02}:{:02}:{:02} UTC] [{}] {}", h, m, s, stage, msg);
}

// ---------------------------------------------------------------------------
// Run directory setup
// ---------------------------------------------------------------------------

fn setup_run_dir(base: &str, seed: u64) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYYMMDD_HHMMSS using epoch arithmetic
    // Simple: just use epoch seconds for uniqueness
    let dir = format!("{}/seed{}_{}", base, seed, secs);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("Warning: failed to create run directory {}: {}", dir, e);
    }
    dir
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn print_fitness(name: &str, result: &FitnessResult) {
    log_ts(
        "baseline",
        &format!(
            "{:<15} | score: {:>10.2} | factored: {}/{} | max bits: {} | time: {}ms",
            name,
            result.score,
            result.success_count,
            result.total_attempts,
            result.max_bits_factored,
            result.total_time_ms
        ),
    );
}

fn create_checkpoint(
    stage_name: &str,
    model: &IslandModel,
    cache: &FitnessCache,
    novelty_archive: &NoveltyArchive,
    start: &Instant,
) -> Checkpoint {
    let mut all: Vec<(&Program, f64)> = model
        .islands
        .iter()
        .flat_map(|island| {
            island
                .individuals
                .iter()
                .map(|i| (&i.program, i.fitness))
        })
        .collect();
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let top_programs: Vec<ProgramEntry> = all
        .into_iter()
        .take(20)
        .map(|(prog, fit)| ProgramEntry {
            fitness: fit,
            nodes: prog.root.node_count(),
            program: format!("{}", prog),
        })
        .collect();

    let (best_fitness, best_program, best_nodes) = if let Some(best) = model.global_best() {
        (
            best.fitness,
            format!("{}", best.program),
            best.program.root.node_count(),
        )
    } else {
        (0.0, "none".to_string(), 0)
    };

    Checkpoint {
        stage: stage_name.to_string(),
        generation: model.generation,
        elapsed_secs: start.elapsed().as_secs_f64(),
        cache_entries: cache.len(),
        novelty_archive_size: novelty_archive.size(),
        unique_behaviors: novelty_archive.unique_count(),
        global_best_fitness: best_fitness,
        global_best_program: best_program,
        global_best_nodes: best_nodes,
        top_programs,
    }
}

/// Collect the top N programs from an island model, deduplicated by string representation.
fn collect_top_programs(model: &IslandModel, n: usize) -> Vec<(Program, f64)> {
    let mut all: Vec<(Program, f64)> = model
        .islands
        .iter()
        .flat_map(|island| {
            island
                .individuals
                .iter()
                .map(|i| (i.program.clone(), i.fitness))
        })
        .collect();
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut seen = std::collections::HashSet::new();
    all.retain(|(prog, _)| {
        let key = format!("{}", prog);
        seen.insert(key)
    });
    all.truncate(n);
    all
}

// ---------------------------------------------------------------------------
// Stage runner
// ---------------------------------------------------------------------------

fn run_stage(
    stage_idx: usize,
    config: &StageConfig,
    rng: &mut ChaCha8Rng,
    pool: &rayon::ThreadPool,
    run_dir: &str,
    seed_programs: &[(Program, f64)],
    verbose: bool,
) -> Vec<(Program, f64)> {
    let stage_label = format!("stage {}: {}", stage_idx + 1, config.name);

    log_ts(
        &stage_label,
        &format!(
            "START — {} islands x {} pop, {} gens, bits {:?}",
            config.num_islands, config.island_size, config.num_generations, config.bit_sizes
        ),
    );

    // Pre-generate deterministic test suite for this stage
    let test_suite = semiprime_ladder(&config.bit_sizes, config.count_per_size, rng);
    let timeout_ms = config.timeout_ms;

    log_ts(
        &stage_label,
        &format!(
            "Test suite: {} semiprimes ({} sizes x {} each)",
            test_suite.len(),
            config.bit_sizes.len(),
            config.count_per_size
        ),
    );

    // Fitness function uses pre-generated suite (deterministic, no internal RNG)
    let fitness_fn = move |program: &Program| -> f64 {
        evaluate_fitness_on_suite(program, &test_suite, timeout_ms).score
    };

    // Initialize island model
    let mut model = IslandModel::new(config.num_islands, config.island_size, rng);

    // Inject seed programs from previous stage (distributed across islands)
    if !seed_programs.is_empty() {
        for (i, (prog, _)) in seed_programs.iter().enumerate() {
            let island_idx = i % model.islands.len();
            if let Some(slot) = model.islands[island_idx].individuals.first_mut() {
                slot.program = prog.clone();
                slot.fitness = 0.0;
            }
        }
        log_ts(
            &stage_label,
            &format!(
                "Injected {} programs from previous stage",
                seed_programs.len()
            ),
        );
    }

    let mut cache = FitnessCache::new(config.cache_size);
    let mut novelty_archive = NoveltyArchive::new(config.novelty_k, config.novelty_max);

    let stage_start = Instant::now();
    let mut total_evals = 0u64;

    for gen in 0..config.num_generations {
        model.evolve_generation_parallel(rng, &fitness_fn, &mut cache, pool);

        let island_total: usize = model.islands.iter().map(|i| i.individuals.len()).sum();
        total_evals += island_total as u64;

        // Update novelty archive
        if gen % 10 == 0 {
            // Use a separate sub-RNG for novelty evaluation to avoid disturbing main RNG sequence
            let mut novelty_rng = ChaCha8Rng::seed_from_u64(rng.gen::<u64>());
            for island in &model.islands {
                if let Some(best) = island.best() {
                    let mo_result =
                        evaluate_fitness_multiobjective(&best.program, &mut novelty_rng);
                    novelty_archive.add(&mo_result.fingerprint);
                }
            }
        }

        // Progress report
        if verbose || (gen + 1) % config.report_interval == 0 || gen == 0 {
            let elapsed = stage_start.elapsed();
            if let Some(best) = model.global_best() {
                let evals_per_sec =
                    total_evals as f64 / elapsed.as_secs_f64().max(0.001);
                log_ts(
                    &stage_label,
                    &format!(
                        "gen {}/{} | best={:.1} | evals={}K ({:.0}/s) | cache={} | novelty={} | {:.1}s",
                        gen + 1,
                        config.num_generations,
                        best.fitness,
                        total_evals / 1000,
                        evals_per_sec,
                        cache.len(),
                        novelty_archive.size(),
                        elapsed.as_secs_f64(),
                    ),
                );
            }
        }

        // Checkpoint
        if (gen + 1) % config.checkpoint_interval == 0 {
            let checkpoint =
                create_checkpoint(config.name, &model, &cache, &novelty_archive, &stage_start);
            if let Ok(json) = serde_json::to_string_pretty(&checkpoint) {
                let path = format!(
                    "{}/stage{}_checkpoint_gen{:04}.json",
                    run_dir,
                    stage_idx + 1,
                    gen + 1
                );
                if let Err(e) = std::fs::write(&path, &json) {
                    eprintln!("  Warning: failed to write checkpoint: {}", e);
                }
            }
        }
    }

    let elapsed = stage_start.elapsed();
    log_ts(
        &stage_label,
        &format!(
            "DONE — {:.1}s, {}K evals, cache={}, novelty={}",
            elapsed.as_secs_f64(),
            total_evals / 1000,
            cache.len(),
            novelty_archive.size(),
        ),
    );

    // Final checkpoint
    let checkpoint =
        create_checkpoint(config.name, &model, &cache, &novelty_archive, &stage_start);
    if let Ok(json) = serde_json::to_string_pretty(&checkpoint) {
        let path = format!("{}/stage{}_checkpoint_final.json", run_dir, stage_idx + 1);
        let _ = std::fs::write(&path, &json);
    }

    // Print top 5
    let top = collect_top_programs(&model, 20);
    log_ts(&stage_label, "Top 5 programs:");
    for (i, (prog, fit)) in top.iter().take(5).enumerate() {
        log_ts(
            &stage_label,
            &format!(
                "  #{}: fitness={:.2}, nodes={}, {}",
                i + 1,
                fit,
                prog.root.node_count(),
                prog
            ),
        );
    }

    // Return top 20 for seeding next stage
    top
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // Resolve stage selection
    let stage = if cli.quick { 1 } else { cli.stage };

    // Build Rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .thread_name(|idx| format!("ae-worker-{}", idx))
        .build()
        .expect("Failed to build Rayon thread pool");

    // Setup run directory
    let run_dir = setup_run_dir(&cli.output_dir, cli.seed);

    log_ts("init", "========================================");
    log_ts("init", "  Alpha-Evolve v4: M4-Tuned Evolution");
    log_ts("init", "========================================");
    log_ts(
        "init",
        &format!("  Seed: {}", cli.seed),
    );
    log_ts(
        "init",
        &format!("  Threads: {} (Rayon pool)", cli.threads),
    );
    log_ts(
        "init",
        &format!(
            "  Stage: {}",
            match stage {
                0 => "all (1→2→3)",
                1 => "warmup only",
                2 => "exploration only",
                3 => "full only",
                _ => "unknown",
            }
        ),
    );
    log_ts(
        "init",
        &format!("  Run dir: {}", run_dir),
    );
    log_ts("init", "  Primitives: 25 ops + 6 macro blocks");

    // Save run config
    let rustc_version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let all_stages = stage_configs();
    let run_config = RunConfig {
        seed: cli.seed,
        threads: cli.threads,
        stages: all_stages.iter().map(|s| s.name.to_string()).collect(),
        rustc_version,
    };
    if let Ok(json) = serde_json::to_string_pretty(&run_config) {
        let _ = std::fs::write(format!("{}/run_config.json", run_dir), &json);
    }

    // Master RNG
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);

    // Baseline evaluation
    log_ts("baseline", "Evaluating seed programs...");
    let seeds: Vec<(&str, Program)> = vec![
        ("Pollard Rho", seed_pollard_rho()),
        ("Trial-Like", seed_trial_like()),
        ("Fermat-Like", seed_fermat_like()),
        ("Lehman-Like", seed_lehman_like()),
        ("Hart-Like", seed_hart_like()),
        ("Dixon Smooth", seed_dixon_smooth()),
        ("CF Regulator", seed_cf_regulator_jump()),
        ("Lattice GCD", seed_lattice_gcd()),
        ("ECM-CF Hybrid", seed_ecm_cf_hybrid()),
        ("Pm1+Rho Hybrid", seed_pm1_rho_hybrid()),
    ];
    for (name, program) in &seeds {
        let result = evaluate_fitness_cascaded(program, &mut rng);
        print_fitness(name, &result);
    }

    // Determine which stages to run
    let stages_to_run: Vec<usize> = match stage {
        0 => vec![0, 1, 2],
        1 => vec![0],
        2 => vec![1],
        3 => vec![2],
        _ => vec![0],
    };

    let all_stages = stage_configs();
    let mut carry_programs: Vec<(Program, f64)> = Vec::new();

    let global_start = Instant::now();

    for &stage_idx in &stages_to_run {
        let config = &all_stages[stage_idx];
        carry_programs = run_stage(
            stage_idx,
            config,
            &mut rng,
            &pool,
            &run_dir,
            &carry_programs,
            cli.verbose,
        );
    }

    let total_elapsed = global_start.elapsed();

    // --- Final report ---
    if !carry_programs.is_empty() {
        log_ts("report", "Running post-evolution analysis...");

        let baselines: Vec<(&str, Program)> = vec![
            ("Pollard Rho", seed_pollard_rho()),
            ("Fermat", seed_fermat_like()),
            ("Hart", seed_hart_like()),
            ("Lehman", seed_lehman_like()),
        ];

        let novelty_archive = NoveltyArchive::new(5, 100);
        let report =
            analysis::generate_report(&carry_programs, &baselines, &novelty_archive, &mut rng);
        analysis::print_report_summary(&report);

        if let Ok(json) = serde_json::to_string_pretty(&report) {
            let path = format!("{}/analysis.json", run_dir);
            if let Err(e) = std::fs::write(&path, &json) {
                eprintln!("  Warning: failed to write analysis: {}", e);
            } else {
                log_ts("report", &format!("Analysis saved: {}", path));
            }
        }

        // Test best on various bit sizes
        if let Some((best_prog, best_fit)) = carry_programs.first() {
            log_ts(
                "report",
                &format!(
                    "Best program: fitness={:.2}, nodes={}",
                    best_fit,
                    best_prog.root.node_count()
                ),
            );
            log_ts("report", &format!("  {}", best_prog));

            log_ts("report", "Testing best on semiprimes:");
            for bits in [16, 20, 24, 28, 32, 36, 40, 48] {
                let target = factoring_core::generate_rsa_target(bits, &mut rng);
                let start = Instant::now();
                let result = best_prog.evaluate(&target.n);
                let elapsed = start.elapsed();

                match result {
                    Some(factor) => {
                        let cofactor = &target.n / &factor;
                        log_ts(
                            "report",
                            &format!(
                                "  {:>3}-bit: {} = {} x {} ({:?})",
                                bits, target.n, factor, cofactor, elapsed
                            ),
                        );
                    }
                    None => {
                        log_ts(
                            "report",
                            &format!(
                                "  {:>3}-bit: {} -- no factor ({:?})",
                                bits, target.n, elapsed
                            ),
                        );
                    }
                }
            }
        }
    }

    log_ts("done", "========================================");
    log_ts(
        "done",
        &format!("  Total time: {:.1}s", total_elapsed.as_secs_f64()),
    );
    log_ts(
        "done",
        &format!("  Run dir: {}", run_dir),
    );
    log_ts("done", "========================================");
}
