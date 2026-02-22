//! Alpha-Evolve v2 demo: island-model evolutionary search for factoring algorithms.
//!
//! Uses IslandModel with multiple populations evolving independently, periodic
//! migration in a ring topology, and FunSearch-style culling. Fitness is evaluated
//! using cascaded difficulty levels that skip expensive tests for weak programs.

use rand::thread_rng;
use std::time::Instant;

use alpha_evolve::evolution::IslandModel;
use alpha_evolve::fitness::{evaluate_fitness_cascaded, FitnessResult};
use alpha_evolve::{
    seed_fermat_like, seed_hart_like, seed_lehman_like, seed_pollard_rho, seed_trial_like,
    Program,
};

fn main() {
    println!("========================================");
    println!("  Alpha-Evolve v2: Island Evolution");
    println!("========================================");
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
    ];

    for (name, program) in &seeds {
        let result = evaluate_fitness_cascaded(program);
        print_fitness(name, &result);
    }
    println!();

    // --- Island model evolution ---
    let num_islands = 5;
    let island_size = 30;
    let num_generations = 50;

    println!(
        "--- Island Evolution: {} islands x {} individuals, {} generations ---",
        num_islands, island_size, num_generations
    );
    println!("  Migration: every 10 gens, ring topology");
    println!("  Culling: worst island reset every 50 gens");
    println!("  Mutation rate: 95%");
    println!("  Bloat control: covariant parsimony pressure");
    println!();

    let mut model = IslandModel::new(num_islands, island_size, &mut rng);
    let start = Instant::now();

    for gen in 0..num_generations {
        model.evolve_generation(&mut rng, &|program: &Program| {
            evaluate_fitness_cascaded(program).score
        });

        // Print progress every 5 generations
        if (gen + 1) % 5 == 0 || gen == 0 {
            if let Some(best) = model.global_best() {
                let island_bests: Vec<String> = model
                    .islands
                    .iter()
                    .map(|island| {
                        format!("{:.0}", island.best().map(|b| b.fitness).unwrap_or(0.0))
                    })
                    .collect();
                println!(
                    "  Gen {:>3}  |  global best: {:>10.2}  |  islands: [{}]  |  program: {}",
                    gen + 1,
                    best.fitness,
                    island_bests.join(", "),
                    best.program
                );
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Evolution took {:?}", elapsed);
    println!();

    // --- Final report ---
    println!("--- Final Results ---");
    println!();

    if let Some(best) = model.global_best() {
        println!("Best evolved program:");
        println!("  {}", best.program);
        println!("  Fitness: {:.2}", best.fitness);
        println!("  Nodes: {}", best.program.root.node_count());
        println!();

        // Test on sample semiprimes
        println!("Testing best program on semiprimes:");
        let mut rng2 = thread_rng();
        for bits in [16, 20, 24, 28, 32, 36] {
            let target = factoring_core::generate_rsa_target(bits, &mut rng2);
            let start = Instant::now();
            let result = best.program.evaluate(&target.n);
            let elapsed = start.elapsed();

            match result {
                Some(factor) => {
                    let cofactor = &target.n / &factor;
                    println!(
                        "  {}-bit: {} = {} x {} ({:?})",
                        bits, target.n, factor, cofactor, elapsed
                    );
                }
                None => {
                    println!(
                        "  {}-bit: {} -- no factor found ({:?})",
                        bits, target.n, elapsed
                    );
                }
            }
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
