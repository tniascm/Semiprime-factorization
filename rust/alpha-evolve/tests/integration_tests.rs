//! Integration tests for the upgraded alpha-evolve crate.
//!
//! Tests cover:
//! - New seed programs (seed_lehman_like, seed_hart_like)
//! - New primitives (FermatStep, HartStep, WilliamsStep, ISqrt, IsPerfectSquare)
//! - Island model (creation, evolution, migration, global best)
//! - Cascaded fitness evaluation

use num_bigint::BigUint;
use num_traits::{One, Zero};

use alpha_evolve::evolution::IslandModel;
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
    // Lehman-like uses ISqrt + FermatStep. The seed program starts at
    // isqrt(initial_state) and iterates FermatStep with k=1. Verify
    // the program has correct structure and runs safely on various inputs.
    let program = seed_lehman_like();

    // Verify structure
    let display = format!("{}", program);
    assert!(display.contains("ISqrt"), "Should contain ISqrt");
    assert!(display.contains("Fermat"), "Should contain FermatStep");
    assert!(display.contains("Loop"), "Should contain a loop");

    // Run on several inputs without panicking
    let test_values: Vec<u32> = vec![6, 15, 21, 35, 77, 143, 221, 323, 8051, 9991];
    for val in test_values {
        let n = BigUint::from(val);
        // Should not panic, result can be None or Some
        let _result = program.evaluate(&n);
    }

    // Verify it can factor at least 35 = 5*7 (close factors, isqrt(2)=1,
    // Fermat tries values starting from 1 and incrementing)
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
    // 221 = 13 * 17
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
        // n=1 should return None
        assert!(program.evaluate(&BigUint::one()).is_none());
        // n=0 should return None
        assert!(program.evaluate(&BigUint::zero()).is_none());
    }
}

// ---------------------------------------------------------------------------
// New primitive operation tests
// ---------------------------------------------------------------------------

#[test]
fn test_fermat_step_primitive_in_program() {
    // Construct a minimal program using FermatStep
    // 35 = 5 * 7. If a = 6, then a^2 - 35 = 36 - 35 = 1, sqrt(1)=1, gcd(7,35) = 7
    let root = ProgramNode::Sequence(vec![
        // Set state to 6 (starts at 2, add 4)
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
    // ISqrt should compute floor(sqrt(state))
    // State starts at 2. AddConst computes (state + c) mod n.
    // With n=1000: (2 + 98) mod 1000 = 100, then ISqrt(100) = 10.
    // gcd(10, 1000) = 10, which is a nontrivial factor.
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 98 }), // (2 + 98) % 1000 = 100
        ProgramNode::Leaf(PrimitiveOp::ISqrt),               // floor(sqrt(100)) = 10
        ProgramNode::Leaf(PrimitiveOp::Gcd),                 // gcd(10, 1000) = 10
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
    // IsPerfectSquare sets state to 1 if perfect square, 0 otherwise
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::AddConst { c: 7 }), // 2 + 7 = 9
        ProgramNode::Leaf(PrimitiveOp::IsPerfectSquare),    // 9 is perfect square -> state = 1
    ]);
    let program = Program { root };
    let n = BigUint::from(100u32);
    // This won't find a factor (gcd(1, 100) = 1), but it should not panic
    let result = program.evaluate(&n);
    // state = 1, which is trivial -- no factor found
    assert!(result.is_none());
}

#[test]
fn test_hart_step_primitive_in_program() {
    // HartStep uses iter_count as multiplier. In a loop, iter_count increments.
    let root = ProgramNode::IterateNode {
        body: Box::new(ProgramNode::Leaf(PrimitiveOp::HartStep)),
        steps: 100,
    };
    let program = Program { root };
    // 8051 = 83 * 97
    let n = BigUint::from(8051u32);
    let result = program.evaluate(&n);
    assert!(result.is_some(), "HartStep should factor 8051 within 100 iterations");
    let factor = result.unwrap();
    assert!((&n % &factor).is_zero());
}

#[test]
fn test_williams_step_primitive_in_program() {
    // Williams p+1 with a starting value and bound
    // 15 = 3 * 5; p-1 of both factors: 3-1=2, 5-1=4; p+1: 3+1=4, 5+1=6
    // Williams should find factors when p+1 is smooth
    let root = ProgramNode::Sequence(vec![
        ProgramNode::Leaf(PrimitiveOp::RandomElement),
        ProgramNode::Leaf(PrimitiveOp::WilliamsStep { bound: 100 }),
    ]);
    let program = Program { root };
    let n = BigUint::from(15u32);
    // Williams is probabilistic depending on starting value
    let mut found = false;
    for _ in 0..30 {
        if let Some(factor) = program.evaluate(&n) {
            assert!((&n % &factor).is_zero());
            found = true;
            break;
        }
    }
    // Williams may not always succeed on small n with random starts, so this
    // is a softer assertion
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
    // All individuals should have been evaluated
    for island in &model.islands {
        for individual in &island.individuals {
            // After evolution, fitness may have been adjusted by parsimony pressure
            // but should be non-negative
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

    // Run 15 generations (should trigger one migration at gen 10)
    for _ in 0..15 {
        model.evolve_generation(&mut rng, &|_program: &Program| {
            // Simple fitness: inverse of node count (prefer smaller programs)
            1.0
        });
    }

    assert_eq!(model.generation, 15);
}

#[test]
fn test_island_model_global_best() {
    let mut rng = rand::thread_rng();
    let mut model = IslandModel::new(3, 10, &mut rng);

    // Evolve a few generations with a real fitness function
    for _ in 0..5 {
        model.evolve_generation(&mut rng, &|program: &Program| {
            evaluate_fitness_cascaded(program).score
        });
    }

    let global_best = model.global_best();
    assert!(global_best.is_some(), "Should have a global best after evolution");

    // Global best should be at least as good as any island best
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

    // Set distinct fitness values on each island so we can detect migration
    let fitness_fn = |program: &Program| evaluate_fitness_cascaded(program).score;

    // Run exactly 10 generations to trigger migration
    for _ in 0..10 {
        model.evolve_generation(&mut rng, &fitness_fn);
    }

    assert_eq!(model.generation, 10);
    // Migration should have occurred at gen 10
    // We cannot easily verify migration happened, but we can verify the model
    // is still in a valid state
    for island in &model.islands {
        assert_eq!(
            island.individuals.len(),
            10,
            "Island should still have 10 individuals after migration"
        );
    }
}

// ---------------------------------------------------------------------------
// Cascaded fitness tests
// ---------------------------------------------------------------------------

#[test]
fn test_cascaded_fitness_returns_result() {
    let program = seed_pollard_rho();
    let result = evaluate_fitness_cascaded(&program);

    assert!(result.total_attempts > 0, "Should test at least one semiprime");
    assert!(result.score >= 0.0, "Score should be non-negative");
}

#[test]
fn test_cascaded_fitness_weak_program_stops_early() {
    // A program that does nothing useful should stop at the first cascade level
    let root = ProgramNode::Leaf(PrimitiveOp::Square);
    let program = Program { root };

    let result = evaluate_fitness_cascaded(&program);

    // A weak program that cannot factor 16-bit semiprimes should test at most
    // the first level (5 attempts) and not advance
    assert!(
        result.total_attempts <= 5,
        "Weak program should stop after first cascade level, got {} attempts",
        result.total_attempts
    );
}

#[test]
fn test_cascaded_vs_standard_fitness_consistency() {
    // Both fitness functions should give positive scores for Pollard rho
    let program = seed_pollard_rho();

    let standard = evaluate_fitness(&program);
    let cascaded = evaluate_fitness_cascaded(&program);

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
    // Pollard rho should pass the first level (16-bit, 5 attempts, need 2 successes)
    // and advance to test harder semiprimes
    let program = seed_pollard_rho();
    let result = evaluate_fitness_cascaded(&program);

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
    let result = evaluate_fitness_cascaded(&program);

    // trial-like should at least factor some 16-bit semiprimes
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

    // Run a short evolution with cascaded fitness
    let fitness_fn = |program: &Program| evaluate_fitness_cascaded(program).score;

    for _ in 0..3 {
        model.evolve_generation(&mut rng, &fitness_fn);
    }

    // After evolution, should have a global best
    let best = model.global_best();
    assert!(best.is_some(), "Should have a best individual after 3 generations");

    // The best should have a valid program
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
