//! Alpha-Evolve v3: Scaled island-model evolutionary search for factoring algorithms.
//!
//! Uses IslandModel with Rayon parallel fitness evaluation, multi-objective
//! fitness (scaling + novelty + efficiency), fitness caching, and checkpointing.
//! Scales to 20 islands × 100 individuals × 1000 generations (~500K evaluations).

use rand::thread_rng;
use serde::Serialize;
use std::cmp::Ordering;
use std::time::Instant;

use alpha_evolve::evolution::{FitnessCache, IslandModel};
use alpha_evolve::fitness::{
    evaluate_fitness_cascaded, evaluate_fitness_multiobjective, FitnessResult,
};
use alpha_evolve::novelty::NoveltyArchive;
use alpha_evolve::{
    seed_cf_regulator_jump, seed_dixon_smooth, seed_ecm_cf_hybrid, seed_fermat_like,
    seed_hart_like, seed_lattice_gcd, seed_lehman_like, seed_pm1_rho_hybrid,
    seed_pollard_rho, seed_trial_like, Program,
};

/// Checkpoint data serialized to JSON periodically.
#[derive(Serialize)]
struct Checkpoint {
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

/// Run configuration.
struct Config {
    num_islands: usize,
    island_size: usize,
    num_generations: u32,
    report_interval: u32,
    checkpoint_interval: u32,
    cache_size: usize,
    novelty_k: usize,
    novelty_max: usize,
}

impl Config {
    fn default_full() -> Self {
        Config {
            num_islands: 20,
            island_size: 100,
            num_generations: 1000,
            report_interval: 50,
            checkpoint_interval: 100,
            cache_size: 50_000,
            novelty_k: 5,
            novelty_max: 10_000,
        }
    }

    fn default_quick() -> Self {
        Config {
            num_islands: 5,
            island_size: 30,
            num_generations: 50,
            report_interval: 10,
            checkpoint_interval: 25,
            cache_size: 10_000,
            novelty_k: 5,
            novelty_max: 1_000,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick" || a == "-q");
    let config = if quick_mode {
        Config::default_quick()
    } else {
        Config::default_full()
    };

    println!("========================================");
    println!("  Alpha-Evolve v3: Scaled Island Evolution");
    println!("========================================");
    println!();

    if quick_mode {
        println!("  Mode: QUICK ({}×{}, {} gens)", config.num_islands, config.island_size, config.num_generations);
    } else {
        println!("  Mode: FULL ({}×{}, {} gens)", config.num_islands, config.island_size, config.num_generations);
    }
    println!("  Primitives: 25 ops + 6 macro blocks");
    println!("  Fitness: multi-objective (scaling + novelty + efficiency)");
    println!("  Cache: {} entries", config.cache_size);
    println!("  Novelty: k={}, archive max={}", config.novelty_k, config.novelty_max);
    println!();

    let mut rng = thread_rng();

    // --- Baseline seed programs ---
    println!("--- Baseline seed programs ---");
    println!();

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
        let result = evaluate_fitness_cascaded(program);
        print_fitness(name, &result);
    }
    println!();

    // --- Initialize evolution ---
    let mut model = IslandModel::new(config.num_islands, config.island_size, &mut rng);
    let mut cache = FitnessCache::new(config.cache_size);
    let mut novelty_archive = NoveltyArchive::new(config.novelty_k, config.novelty_max);

    // Use multi-objective fitness with novelty integration
    let fitness_fn = |program: &Program| -> f64 {
        let result = evaluate_fitness_multiobjective(program);
        result.combined_score
    };

    println!(
        "--- Island Evolution: {} islands × {} individuals, {} generations ---",
        config.num_islands, config.island_size, config.num_generations
    );
    println!("  Migration: every {} gens, ring topology", model.migration_interval);
    println!("  Culling: worst island reset every 50 gens");
    println!("  Mutation rate: 95%");
    println!("  Bloat control: covariant parsimony pressure");
    println!("  Parallel: Rayon thread pool");
    println!();

    let start = Instant::now();
    let mut total_evals = 0u64;

    for gen in 0..config.num_generations {
        model.evolve_generation_parallel(&mut rng, &fitness_fn, &mut cache);

        // Count evaluations
        let island_total: usize = model.islands.iter().map(|i| i.individuals.len()).sum();
        total_evals += island_total as u64;

        // Update novelty archive with best individuals' fingerprints
        if gen % 10 == 0 {
            for island in &model.islands {
                if let Some(best) = island.best() {
                    let mo_result = evaluate_fitness_multiobjective(&best.program);
                    novelty_archive.add(&mo_result.fingerprint);
                }
            }
        }

        // Print progress
        if (gen + 1) % config.report_interval == 0 || gen == 0 {
            let elapsed = start.elapsed();
            if let Some(best) = model.global_best() {
                let island_bests: Vec<String> = model
                    .islands
                    .iter()
                    .map(|island| {
                        format!("{:.0}", island.best().map(|b| b.fitness).unwrap_or(0.0))
                    })
                    .collect();
                let evals_per_sec = total_evals as f64 / elapsed.as_secs_f64().max(0.001);
                println!(
                    "  Gen {:>4}/{} | best: {:>10.1} | islands: [{}] | evals: {}K ({:.0}/s) | cache: {} | novelty: {} | {:.1}s",
                    gen + 1,
                    config.num_generations,
                    best.fitness,
                    island_bests.join(", "),
                    total_evals / 1000,
                    evals_per_sec,
                    cache.len(),
                    novelty_archive.size(),
                    elapsed.as_secs_f64(),
                );
            }
        }

        // Checkpoint
        if (gen + 1) % config.checkpoint_interval == 0 {
            let checkpoint = create_checkpoint(&model, &cache, &novelty_archive, &start);
            if let Ok(json) = serde_json::to_string_pretty(&checkpoint) {
                let checkpoint_path = format!("checkpoint_gen{:04}.json", gen + 1);
                if let Err(e) = std::fs::write(&checkpoint_path, &json) {
                    eprintln!("  Warning: failed to write checkpoint: {}", e);
                } else {
                    println!("  [Checkpoint saved: {}]", checkpoint_path);
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Evolution took {:?} ({:.0} evals/sec)", elapsed, total_evals as f64 / elapsed.as_secs_f64().max(0.001));
    println!("  Total evaluations: {}K", total_evals / 1000);
    println!("  Cache entries: {}", cache.len());
    println!("  Novelty archive: {} behaviors ({} unique)", novelty_archive.size(), novelty_archive.unique_count());
    println!();

    // --- Final report ---
    println!("--- Final Results ---");
    println!();

    // Top 5 programs
    let mut all_individuals: Vec<&alpha_evolve::Individual> = model
        .islands
        .iter()
        .flat_map(|island| island.individuals.iter())
        .collect();
    all_individuals.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(Ordering::Equal)
    });

    println!("Top 5 evolved programs:");
    for (i, individual) in all_individuals.iter().take(5).enumerate() {
        println!(
            "  #{}: fitness={:.2}, nodes={}, program={}",
            i + 1,
            individual.fitness,
            individual.program.root.node_count(),
            individual.program
        );
    }
    println!();

    // Test best program on various bit sizes
    if let Some(best) = model.global_best() {
        println!("Best evolved program:");
        println!("  {}", best.program);
        println!("  Fitness: {:.2}", best.fitness);
        println!("  Nodes: {}", best.program.root.node_count());
        println!();

        println!("Testing best program on semiprimes:");
        let mut rng2 = thread_rng();
        for bits in [16, 20, 24, 28, 32, 36, 40, 48] {
            let target = factoring_core::generate_rsa_target(bits, &mut rng2);
            let start = Instant::now();
            let result = best.program.evaluate(&target.n);
            let elapsed = start.elapsed();

            match result {
                Some(factor) => {
                    let cofactor = &target.n / &factor;
                    println!(
                        "  {:>3}-bit: {} = {} × {} ({:?})",
                        bits, target.n, factor, cofactor, elapsed
                    );
                }
                None => {
                    println!(
                        "  {:>3}-bit: {} — no factor found ({:?})",
                        bits, target.n, elapsed
                    );
                }
            }
        }
    }

    // Write final JSON report
    let checkpoint = create_checkpoint(&model, &cache, &novelty_archive, &start);
    if let Ok(json) = serde_json::to_string_pretty(&checkpoint) {
        let report_path = "alpha_evolve_final_report.json";
        if let Err(e) = std::fs::write(report_path, &json) {
            eprintln!("  Warning: failed to write final report: {}", e);
        } else {
            println!();
            println!("  Final report saved: {}", report_path);
        }
    }

    println!();
    println!("========================================");
    println!("  Evolution complete.");
    println!("========================================");
}

/// Pretty-print a fitness result for a named program.
fn print_fitness(name: &str, result: &FitnessResult) {
    println!(
        "  {:<15} | score: {:>10.2} | factored: {}/{} | max bits: {} | time: {}ms",
        name,
        result.score,
        result.success_count,
        result.total_attempts,
        result.max_bits_factored,
        result.total_time_ms
    );
}

/// Create a checkpoint from the current state.
fn create_checkpoint(
    model: &IslandModel,
    cache: &FitnessCache,
    novelty_archive: &NoveltyArchive,
    start: &Instant,
) -> Checkpoint {
    // Collect top 20 programs across all islands
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

    let (best_fitness, best_program, best_nodes) =
        if let Some(best) = model.global_best() {
            (
                best.fitness,
                format!("{}", best.program),
                best.program.root.node_count(),
            )
        } else {
            (0.0, "none".to_string(), 0)
        };

    Checkpoint {
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
