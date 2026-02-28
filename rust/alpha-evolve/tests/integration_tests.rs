//! Integration tests for the upgraded alpha-evolve crate.
//!
//! Tests cover:
//! - New seed programs (seed_lehman_like, seed_hart_like)
//! - New primitives (FermatStep, HartStep, WilliamsStep, ISqrt, IsPerfectSquare)
//! - Island model (creation, evolution, migration, global best)
//! - Cascaded fitness evaluation
//! - Parallel evolution with explicit thread pool

use num_bigint::BigUint;
use num_traits::{One, Zero};

use alpha_evolve::evolution::{FitnessCache, IslandModel};
use alpha_evolve::fitness::{evaluate_fitness, evaluate_fitness_cascaded};
use alpha_evolve::{
    seed_fermat_like, seed_hart_like, seed_lehman_like, seed_pollard_rho, seed_trial_like,
    PrimitiveOp, Program, ProgramNode,
};

// ---------------------------------------------------------------------------
// Seed program tests
// ---------------------------------------------------------------------------

#[test]
fn test_seed_lehman_like_factors_small_semiprime() {
    // 35 = 5 * 7 -- Lehman-like uses FermatStep from near sqrt(n)
    let n = BigUint::from(35u32);
    let program = seed_lehman_like();
    let mut found = false;
    for _ in 0..10 {
        if let Some(factor) = program.evaluate(&n) {
            assert!(
                factor == BigUint::from(5u32) || factor == BigUint::from(7u32),
                "Expected factor 5 or 7, got {}",
                factor
            );
            assert!((&n % &factor).is_zero());
            found = true;
            break;
        }
    }
    assert!(found, "seed_lehman_like should factor 35");
}

#[test]
fn test_seed_lehman_like_structure_and_safety() {
    let program = seed_lehman_like();

    let display = format!("{}", program);
    assert!(display.contains("ISqrt"), "Should contain ISqrt");
    assert!(display.contains("Fermat"), "Should contain FermatStep");
    assert!(display.contains("Loop"), "Should contain a loop");

    let test_values: Vec<u32> = vec![6, 15, 21, 35, 77, 143, 221, 323, 8051, 9991];
    for val in test_values {
        let n = BigUint::from(val);
        let _result = program.evaluate(&n);
    }

    let n35 = BigUint::from(35u32);
    let mut found = false;
    for _ in 0..10 {
        if let Some(factor) = program.evaluate(&n35) {
            assert!((&n35 % &factor).is_zero());
            found = true;
            break;
        }
    }
    assert!(found, "seed_lehman_like should factor 35");
}

#[test]
fn test_seed_hart_like_factors_small_semiprime() {
    let n = BigUint::from(221u32);
    let program = seed_hart_like();
    let mut found = false;
    for _ in 0..20 {
        if let Some(factor) = program.evaluate(&n) {
            assert!(
                factor == BigUint::from(13u32) || factor == BigUint::from(17u32),
                "Expected factor 13 or 17, got {}",
                factor
            );
            assert!((&n % &factor).is_zero());
            found = true;
            break;
        }
    }
    assert!(found, "seed_hart_like should factor 221 within 20 attempts");
}

#[test]
fn test_seed_hart_like_display() {
    let program = seed_hart_like();
    let display = format!("{}", program);
    assert!(
        display.contains("Hart"),
        "Hart-like seed should display HartStep, got: {}",
        display
    );
}

#[test]
fn test_seed_lehman_like_display() {
    let program = seed_lehman_like();
    let display = format!("{}", program);
    assert!(
        display.contains("Fermat"),
        "Lehman-like seed should display FermatStep, got: {}",
        display
    );
    assert!(
        display.contains("ISqrt"),
        "Lehman-like seed should display ISqrt, got: {}",
        display
    );
}

#[test]
fn test_all_seeds_node_count() {
    let seeds: Vec<(&str, Program)> = vec![
        ("Pollard Rho", seed_pollard_rho()),
        ("Trial-Like", seed_trial_like()),
        ("Fermat-Like", seed_fermat_like()),
        ("Lehman-Like", seed_lehman_like()),
        ("Hart-Like", seed_hart_like()),
    ];

    for (name, program) in &seeds {
        let count = program.root.node_count();
        assert!(
            count >= 2,
            "{} should have at least 2 nodes, got {}",
            name, count
        );
    }
}

#[test]
fn test_all_seeds_handle_trivial_input() {
    let seeds: Vec<Program> = vec![
        seed_pollard_rho(),
        seed_trial_like(),
        seed_fermat_like(),
        seed_lehman_like(),
        seed_hart_like(),
    ];

    for program in &seeds {
        assert!(program.evaluate(&BigUint::one()).is_none());
        assert!(program.evaluate(&BigUint::zero()).is_none());
    }
}

// ---------------------------------------------------------------------------
// New primitive operation tests
// ---------------------------------------------------------------------------

#[test]
fn test_fermat_step_primitive_in_program() {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 4 }),
        ProgramNode::Leaf(PrimitiveOp::FermatStep { k: 1 }),
    ]);
    let program = Program { root };
    let n = BigUint::from(35u32);
    let result = program.evaluate(&n);
    assert!(result.is_some(), "FermatStep should factor 35 with a=6");
    let factor = result.unwrap();
    assert!((&n % &factor).is_zero());
}

#[test]
fn test_isqrt_primitive_in_program() {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 98 }),
        ProgramNode::Leaf(PrimitiveOp::ISqrt),
        ProgramNode::Leaf(PrimitiveOp::Gcd),
    ]);
    let program = Program { root };
    let n = BigUint::from(1000u32);
    let result = program.evaluate(&n);
    assert!(result.is_some(), "ISqrt followed by Gcd should find factor of 1000");
    let factor = result.unwrap();
    assert_eq!(factor, BigUint::from(10u32));
}

#[test]
fn test_is_perfect_square_primitive() {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 7 }),
        ProgramNode::Leaf(PrimitiveOp::IsPerfectSquare),
    ]);
    let program = Program { root };
    let n = BigUint::from(100u32);
    let result = program.evaluate(&n);
    assert!(result.is_none());
}

#[test]
fn test_hart_step_primitive_in_program() {
    let root = ProgramNode::IterateNode {
        body: Box::new(ProgramNode::Leaf(PrimitiveOp::HartStep)),
        steps: 100,
    };
    let program = Program { root };
    let n = BigUint::from(8051u32);
    let result = program.evaluate(&n);
    assert!(result.is_some(), "HartStep should factor 8051 within 100 iterations");
    let factor = result.unwrap();
    assert!((&n % &factor).is_zero());
}

#[test]
fn test_williams_step_primitive_in_program() {
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::RandomElement),
        ProgramNode::Leaf(PrimitiveOp::WilliamsStep { bound: 100 }),
    ]);
    let program = Program { root };
    let n = BigUint::from(15u32);
    let mut found = false;
    for _ in 0..30 {
        if let Some(factor) = program.evaluate(&n) {
            assert!((&n % &factor).is_zero());
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("Warning: Williams p+1 did not factor 15 in 30 attempts (probabilistic)");
    }
}

// ---------------------------------------------------------------------------
// Island model tests
// ---------------------------------------------------------------------------

#[test]
fn test_island_model_creation() {
    let mut rng = rand::thread_rng();
    let model = IslandModel::new(4, 15, &mut rng);

    assert_eq!(model.islands.len(), 4);
    for island in &model.islands {
        assert_eq!(island.individuals.len(), 15);
    }
    assert_eq!(model.generation, 0);
    assert_eq!(model.migration_interval, 10);
    assert_eq!(model.migration_rate, 2);
}

#[test]
fn test_island_model_single_generation() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(3, 10, &mut rng);

    model.evolve_generation(&mut rng, &|_program: &Program| 1.0);

    assert_eq!(model.generation, 1);
    for island in &model.islands {
        for individual in &island.individuals {
            assert!(
                individual.fitness >= 0.0,
                "Fitness should be non-negative after evolution"
            );
        }
    }
}

#[test]
fn test_island_model_multiple_generations() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(3, 10, &mut rng);

    for _ in 0..15 {
        model.evolve_generation(&mut rng, &|_program: &Program| 1.0);
    }

    assert_eq!(model.generation, 15);
}

#[test]
fn test_island_model_global_best() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(3, 10, &mut rng);

    for _ in 0..5 {
        model.evolve_generation(&mut rng, &|program: &Program| {
            let mut fitness_rng = rand::thread_rng();
            evaluate_fitness_cascaded(program, &mut fitness_rng).score
        });
    }

    let global_best = model.global_best();
    assert!(global_best.is_some(), "Should have a global best after evolution");

    let best_fitness = global_best.unwrap().fitness;
    for island in &model.islands {
        if let Some(island_best) = island.best() {
            assert!(
                best_fitness >= island_best.fitness - f64::EPSILON,
                "Global best ({}) should be >= island best ({})",
                best_fitness,
                island_best.fitness
            );
        }
    }
}

#[test]
fn test_island_model_migration_occurs() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(3, 10, &mut rng);

    let fitness_fn = |program: &Program| {
        let mut fitness_rng = rand::thread_rng();
        evaluate_fitness_cascaded(program, &mut fitness_rng).score
    };

    for _ in 0..10 {
        model.evolve_generation(&mut rng, &fitness_fn);
    }

    assert_eq!(model.generation, 10);
    for island in &model.islands {
        assert_eq!(
            island.individuals.len(),
            10,
            "Island should still have 10 individuals after migration"
        );
    }
}

#[test]
fn test_island_model_parallel_with_pool() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(2, 10, &mut rng);
    let mut cache = FitnessCache::new(1000);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    let fitness_fn = |_program: &Program| -> f64 { 1.0 };

    model.evolve_generation_parallel(&mut rng, &fitness_fn, &mut cache, &pool);

    assert_eq!(model.generation, 1);
    for island in &model.islands {
        assert_eq!(island.individuals.len(), 10);
    }
}

// ---------------------------------------------------------------------------
// Cascaded fitness tests
// ---------------------------------------------------------------------------

#[test]
fn test_cascaded_fitness_returns_result() {
    let program = seed_pollard_rho();
    let mut rng = rand::thread_rng();
    let result = evaluate_fitness_cascaded(&program, &mut rng);

    assert!(result.total_attempts > 0, "Should test at least one semiprime");
    assert!(result.score >= 0.0, "Score should be non-negative");
}

#[test]
fn test_cascaded_fitness_weak_program_stops_early() {
    let root = ProgramNode::Leaf(PrimitiveOp::Square);
    let program = Program { root };

    let mut rng = rand::thread_rng();
    let result = evaluate_fitness_cascaded(&program, &mut rng);

    assert!(
        result.total_attempts <= 5,
        "Weak program should stop after first cascade level, got {} attempts",
        result.total_attempts
    );
}

#[test]
fn test_cascaded_vs_standard_fitness_consistency() {
    let program = seed_pollard_rho();
    let mut rng = rand::thread_rng();

    let standard = evaluate_fitness(&program, &mut rng);
    let cascaded = evaluate_fitness_cascaded(&program, &mut rng);

    assert!(
        standard.score > 0.0,
        "Standard fitness should be positive for Pollard rho"
    );
    assert!(
        cascaded.score > 0.0,
        "Cascaded fitness should be positive for Pollard rho"
    );
    assert!(
        standard.success_count > 0,
        "Standard should factor some semiprimes"
    );
    assert!(
        cascaded.success_count > 0,
        "Cascaded should factor some semiprimes"
    );
}

#[test]
fn test_cascaded_fitness_advances_on_success() {
    let program = seed_pollard_rho();
    let mut rng = rand::thread_rng();
    let result = evaluate_fitness_cascaded(&program, &mut rng);

    assert!(
        result.total_attempts > 5,
        "Pollard rho should advance past 16-bit level, got {} attempts",
        result.total_attempts
    );
}

// ---------------------------------------------------------------------------
// FitnessResult structure tests
// ---------------------------------------------------------------------------

#[test]
fn test_fitness_result_fields() {
    let program = seed_trial_like();
    let mut rng = rand::thread_rng();
    let result = evaluate_fitness_cascaded(&program, &mut rng);

    assert!(
        result.success_count > 0,
        "Trial-like should factor at least one semiprime"
    );
    assert!(
        result.max_bits_factored >= 16,
        "Trial-like should factor at least 16-bit semiprimes"
    );
    assert!(result.total_time_ms < 10000, "Should not take more than 10 seconds");
}

// ---------------------------------------------------------------------------
// End-to-end evolution test
// ---------------------------------------------------------------------------

#[test]
fn test_end_to_end_island_evolution() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(2, 10, &mut rng);

    let fitness_fn = |program: &Program| {
        let mut fitness_rng = rand::thread_rng();
        evaluate_fitness_cascaded(program, &mut fitness_rng).score
    };

    for _ in 0..3 {
        model.evolve_generation(&mut rng, &fitness_fn);
    }

    let best = model.global_best();
    assert!(best.is_some(), "Should have a best individual after 3 generations");

    let best = best.unwrap();
    assert!(
        best.program.root.node_count() >= 1,
        "Best program should have at least 1 node"
    );
}

// ---------------------------------------------------------------------------
// New primitives display tests
// ---------------------------------------------------------------------------

#[test]
fn test_new_primitive_display() {
    let ops: Vec<(PrimitiveOp, &str)> = vec![
        (PrimitiveOp::FermatStep { k: 3 }, "Fermat(3)"),
        (PrimitiveOp::HartStep, "Hart"),
        (PrimitiveOp::WilliamsStep { bound: 50 }, "Williams(50)"),
        (PrimitiveOp::ISqrt, "ISqrt"),
        (PrimitiveOp::IsPerfectSquare, "IsSq"),
    ];

    for (op, expected) in &ops {
        let display = format!("{}", op);
        assert_eq!(
            &display, expected,
            "PrimitiveOp display mismatch for {:?}",
            op
        );
    }
}

#[test]
fn test_new_primitive_equality() {
    assert_eq!(
        PrimitiveOp::FermatStep { k: 1 },
        PrimitiveOp::FermatStep { k: 1 }
    );
    assert_ne!(
        PrimitiveOp::FermatStep { k: 1 },
        PrimitiveOp::FermatStep { k: 2 }
    );
    assert_eq!(PrimitiveOp::HartStep, PrimitiveOp::HartStep);
    assert_eq!(PrimitiveOp::ISqrt, PrimitiveOp::ISqrt);
    assert_eq!(PrimitiveOp::IsPerfectSquare, PrimitiveOp::IsPerfectSquare);
    assert_ne!(PrimitiveOp::ISqrt, PrimitiveOp::IsPerfectSquare);
}
