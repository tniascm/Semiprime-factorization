//! Genetic programming operations: random generation, mutation, crossover,
//! tournament selection, and population management.
//!
//! Includes an island model for parallel population evolution with periodic
//! migration and FunSearch-style island culling.

use std::cmp::Ordering;

use rand::Rng;

use crate::{
    seed_fermat_like, seed_hart_like, seed_lehman_like, seed_pollard_rho, seed_trial_like,
    Individual, PrimitiveOp, Program, ProgramNode,
};

// ---------------------------------------------------------------------------
// Random program generation
// ---------------------------------------------------------------------------

/// Generate a random primitive operation.
///
/// Covers all 25 primitive variants: 14 original + 11 new crate-wrapping operations.
fn random_op(rng: &mut impl Rng) -> PrimitiveOp {
    match rng.gen_range(0..25) {
        0 => PrimitiveOp::ModPow,
        1 => PrimitiveOp::Gcd,
        2 => PrimitiveOp::RandomElement,
        3 => PrimitiveOp::Iterate {
            steps: rng.gen_range(1..=50),
        },
        4 => PrimitiveOp::AccumulateGcd {
            batch_size: rng.gen_range(1..=20),
        },
        5 => PrimitiveOp::SubtractGcd,
        6 => PrimitiveOp::Square,
        7 => PrimitiveOp::AddConst {
            c: rng.gen_range(1..=100),
        },
        8 => PrimitiveOp::MultiplyMod,
        9 => PrimitiveOp::FermatStep {
            k: rng.gen_range(1..=10),
        },
        10 => PrimitiveOp::HartStep,
        11 => PrimitiveOp::WilliamsStep {
            bound: rng.gen_range(100..=10000),
        },
        12 => PrimitiveOp::ISqrt,
        13 => PrimitiveOp::IsPerfectSquare,
        // --- New crate-wrapping primitives ---
        14 => PrimitiveOp::CfConvergent {
            k: rng.gen_range(1..=50),
        },
        15 => PrimitiveOp::SqufofStep,
        16 => PrimitiveOp::RhoFormStep,
        17 => PrimitiveOp::EcmCurve {
            b1: rng.gen_range(100..=10000),
        },
        18 => PrimitiveOp::LllShortVector,
        19 => PrimitiveOp::SmoothTest {
            bound: rng.gen_range(50..=5000),
        },
        20 => PrimitiveOp::PilatteVector,
        21 => PrimitiveOp::QuadraticResidue,
        22 => PrimitiveOp::PollardPm1 {
            bound: rng.gen_range(50..=5000),
        },
        23 => PrimitiveOp::DixonAccumulate,
        _ => PrimitiveOp::DixonCombine,
    }
}

/// Generate a random program node with bounded depth.
///
/// Includes all 6 node types: Leaf, Sequence, IterateNode, GcdCheck,
/// ConditionalGt, and MemoryOp.
fn random_node(rng: &mut impl Rng, max_depth: u32) -> ProgramNode {
    if max_depth <= 1 {
        // At max depth, only generate terminal nodes (Leaf or MemoryOp)
        if rng.gen_bool(0.9) {
            return ProgramNode::Leaf(random_op(rng));
        } else {
            return ProgramNode::MemoryOp {
                store: rng.gen_bool(0.5),
                slot: rng.gen_range(0..4),
            };
        }
    }

    match rng.gen_range(0..8) {
        0 | 1 => ProgramNode::Leaf(random_op(rng)),
        2 => {
            let len = rng.gen_range(2..=4);
            let children: Vec<ProgramNode> =
                (0..len).map(|_| random_node(rng, max_depth - 1)).collect();
            ProgramNode::Sequence(children)
        }
        3 => ProgramNode::IterateNode {
            body: Box::new(random_node(rng, max_depth - 1)),
            steps: rng.gen_range(5..=100),
        },
        4 => ProgramNode::GcdCheck {
            setup: Box::new(random_node(rng, max_depth - 1)),
        },
        5 => ProgramNode::ConditionalGt {
            threshold: rng.gen_range(1..=1000),
            if_true: Box::new(random_node(rng, max_depth - 1)),
            if_false: Box::new(random_node(rng, max_depth - 1)),
        },
        6 => ProgramNode::MemoryOp {
            store: true,
            slot: rng.gen_range(0..4),
        },
        _ => ProgramNode::MemoryOp {
            store: false,
            slot: rng.gen_range(0..4),
        },
    }
}

/// Generate a random program tree with the given maximum depth.
pub fn random_program(rng: &mut impl Rng, max_depth: u32) -> Program {
    let root = random_node(rng, max_depth);
    Program { root }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

/// Randomly modify one node in the program tree.
/// May change an operation, a constant, or a step count.
pub fn mutate(program: &Program, rng: &mut impl Rng) -> Program {
    let count = program.root.node_count();
    if count == 0 {
        return program.clone();
    }

    let target_index = rng.gen_range(0..count);

    // Get the node at target_index and create a mutated replacement
    let replacement = if let Some(node) = program.root.get_node(target_index) {
        mutate_node(node, rng)
    } else {
        // Fallback: just generate a new random leaf
        ProgramNode::Leaf(random_op(rng))
    };

    let new_root = program.root.replace_node(target_index, &replacement);
    Program { root: new_root }
}

/// Create a mutated version of a single node.
fn mutate_node(node: &ProgramNode, rng: &mut impl Rng) -> ProgramNode {
    match node {
        ProgramNode::Leaf(op) => {
            // Either change the operation entirely, or tweak parameters
            if rng.gen_bool(0.5) {
                ProgramNode::Leaf(random_op(rng))
            } else {
                match op {
                    PrimitiveOp::AddConst { c } => {
                        let delta: i64 = rng.gen_range(-10..=10);
                        let new_c = (*c as i64 + delta).max(0) as u64;
                        ProgramNode::Leaf(PrimitiveOp::AddConst { c: new_c })
                    }
                    PrimitiveOp::Iterate { steps } => {
                        let delta: i32 = rng.gen_range(-20..=20);
                        let new_steps = (*steps as i32 + delta).max(1) as u32;
                        ProgramNode::Leaf(PrimitiveOp::Iterate { steps: new_steps })
                    }
                    PrimitiveOp::AccumulateGcd { batch_size } => {
                        let delta: i32 = rng.gen_range(-5..=5);
                        let new_size = (*batch_size as i32 + delta).max(1) as u32;
                        ProgramNode::Leaf(PrimitiveOp::AccumulateGcd {
                            batch_size: new_size,
                        })
                    }
                    PrimitiveOp::FermatStep { k } => {
                        let delta: i64 = rng.gen_range(-3..=3);
                        let new_k = (*k as i64 + delta).max(1) as u64;
                        ProgramNode::Leaf(PrimitiveOp::FermatStep { k: new_k })
                    }
                    PrimitiveOp::WilliamsStep { bound } => {
                        let delta: i64 = rng.gen_range(-500..=500);
                        let new_bound = (*bound as i64 + delta).max(10) as u64;
                        ProgramNode::Leaf(PrimitiveOp::WilliamsStep { bound: new_bound })
                    }
                    // --- New parameterized primitives ---
                    PrimitiveOp::CfConvergent { k } => {
                        let delta: i32 = rng.gen_range(-10..=10);
                        let new_k = (*k as i32 + delta).max(1) as u32;
                        ProgramNode::Leaf(PrimitiveOp::CfConvergent { k: new_k })
                    }
                    PrimitiveOp::EcmCurve { b1 } => {
                        let delta: i64 = rng.gen_range(-2000..=2000);
                        let new_b1 = (*b1 as i64 + delta).max(50) as u64;
                        ProgramNode::Leaf(PrimitiveOp::EcmCurve { b1: new_b1 })
                    }
                    PrimitiveOp::SmoothTest { bound } => {
                        let delta: i64 = rng.gen_range(-500..=500);
                        let new_bound = (*bound as i64 + delta).max(10) as u64;
                        ProgramNode::Leaf(PrimitiveOp::SmoothTest { bound: new_bound })
                    }
                    PrimitiveOp::PollardPm1 { bound } => {
                        let delta: i64 = rng.gen_range(-500..=500);
                        let new_bound = (*bound as i64 + delta).max(10) as u64;
                        ProgramNode::Leaf(PrimitiveOp::PollardPm1 { bound: new_bound })
                    }
                    _ => ProgramNode::Leaf(random_op(rng)),
                }
            }
        }
        ProgramNode::IterateNode { body, steps } => {
            // Tweak the step count
            let delta: i32 = rng.gen_range(-20..=20);
            let new_steps = (*steps as i32 + delta).max(1) as u32;
            ProgramNode::IterateNode {
                body: body.clone(),
                steps: new_steps,
            }
        }
        ProgramNode::Sequence(children) => {
            if children.is_empty() {
                return ProgramNode::Leaf(random_op(rng));
            }
            // Mutate by replacing one child with a random node
            let mut new_children = children.clone();
            let child_idx = rng.gen_range(0..new_children.len());
            new_children[child_idx] = random_node(rng, 2);
            ProgramNode::Sequence(new_children)
        }
        ProgramNode::GcdCheck { setup: _ } => {
            // Replace setup with a new random node
            ProgramNode::GcdCheck {
                setup: Box::new(random_node(rng, 2)),
            }
        }
        ProgramNode::ConditionalGt {
            threshold,
            if_true,
            if_false,
        } => {
            // Either tweak the threshold, or replace one branch
            match rng.gen_range(0..3) {
                0 => {
                    // Tweak threshold
                    let delta: i64 = rng.gen_range(-100..=100);
                    let new_threshold = (*threshold as i64 + delta).max(1) as u64;
                    ProgramNode::ConditionalGt {
                        threshold: new_threshold,
                        if_true: if_true.clone(),
                        if_false: if_false.clone(),
                    }
                }
                1 => {
                    // Replace if_true branch
                    ProgramNode::ConditionalGt {
                        threshold: *threshold,
                        if_true: Box::new(random_node(rng, 2)),
                        if_false: if_false.clone(),
                    }
                }
                _ => {
                    // Replace if_false branch
                    ProgramNode::ConditionalGt {
                        threshold: *threshold,
                        if_true: if_true.clone(),
                        if_false: Box::new(random_node(rng, 2)),
                    }
                }
            }
        }
        ProgramNode::MemoryOp { store, slot } => {
            // Either toggle store/load, or change slot
            if rng.gen_bool(0.5) {
                ProgramNode::MemoryOp {
                    store: !store,
                    slot: *slot,
                }
            } else {
                ProgramNode::MemoryOp {
                    store: *store,
                    slot: rng.gen_range(0..4),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Crossover
// ---------------------------------------------------------------------------

/// Swap a random subtree from `b` into a copy of `a`.
pub fn crossover(a: &Program, b: &Program, rng: &mut impl Rng) -> Program {
    let a_count = a.root.node_count();
    let b_count = b.root.node_count();

    if a_count == 0 || b_count == 0 {
        return a.clone();
    }

    // Pick a random insertion point in a and a random subtree from b
    let a_index = rng.gen_range(0..a_count);
    let b_index = rng.gen_range(0..b_count);

    let donor_subtree = b.root.get_node(b_index).cloned().unwrap_or_else(|| {
        ProgramNode::Leaf(random_op(rng))
    });

    let new_root = a.root.replace_node(a_index, &donor_subtree);
    Program { root: new_root }
}

// ---------------------------------------------------------------------------
// Tournament selection
// ---------------------------------------------------------------------------

/// Pick `tournament_size` individuals at random and return the one with the
/// highest fitness.
pub fn tournament_select<'a>(
    population: &'a [Individual],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Individual {
    assert!(!population.is_empty(), "Population must not be empty");

    let effective_size = tournament_size.min(population.len());
    let mut best_idx = rng.gen_range(0..population.len());

    for _ in 1..effective_size {
        let candidate_idx = rng.gen_range(0..population.len());
        if population[candidate_idx].fitness > population[best_idx].fitness {
            best_idx = candidate_idx;
        }
    }

    &population[best_idx]
}

// ---------------------------------------------------------------------------
// Parsimony pressure
// ---------------------------------------------------------------------------

/// Compute covariant parsimony coefficient: c = Cov(fitness, size) / Var(size).
///
/// Returns the coefficient to subtract from fitness: adjusted_fitness = fitness - c * size.
/// This penalizes bloat by making fitness depend on program size in proportion to
/// how much size and fitness are correlated in the current population.
fn parsimony_coefficient(population: &[Individual]) -> f64 {
    let n = population.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sizes: Vec<f64> = population
        .iter()
        .map(|i| i.program.root.node_count() as f64)
        .collect();
    let fitnesses: Vec<f64> = population.iter().map(|i| i.fitness).collect();

    let mean_size = sizes.iter().sum::<f64>() / n;
    let mean_fitness = fitnesses.iter().sum::<f64>() / n;

    let var_size: f64 = sizes.iter().map(|s| (s - mean_size).powi(2)).sum::<f64>() / n;
    if var_size < 1e-10 {
        return 0.0;
    }

    let cov: f64 = sizes
        .iter()
        .zip(fitnesses.iter())
        .map(|(s, f)| (s - mean_size) * (f - mean_fitness))
        .sum::<f64>()
        / n;

    cov / var_size
}

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------

/// A population of program individuals undergoing evolution.
pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: u32,
}

impl Population {
    /// Create an initial population with random programs plus the three seed programs.
    pub fn new(size: usize, rng: &mut impl Rng) -> Self {
        let mut individuals = Vec::with_capacity(size);

        // Add seed programs first
        let seeds = vec![
            seed_pollard_rho(),
            seed_trial_like(),
            seed_fermat_like(),
            seed_lehman_like(),
            seed_hart_like(),
        ];
        for seed in seeds {
            individuals.push(Individual {
                program: seed,
                fitness: 0.0,
            });
        }

        // Fill the rest with random programs (depth 3-5)
        while individuals.len() < size {
            let depth = rng.gen_range(3..=5);
            let program = random_program(rng, depth);
            individuals.push(Individual {
                program,
                fitness: 0.0,
            });
        }

        Population {
            individuals,
            generation: 0,
        }
    }

    /// Run one generation of evolution:
    /// 1. Evaluate fitness for all individuals.
    /// 2. Apply covariant parsimony pressure to penalize bloat.
    /// 3. Select parents via tournament selection.
    /// 4. Create offspring via crossover and mutation.
    /// 5. Replace the worst individuals with offspring.
    pub fn evolve_generation(
        &mut self,
        rng: &mut impl Rng,
        fitness_fn: &dyn Fn(&Program) -> f64,
    ) {
        let pop_size = self.individuals.len();
        if pop_size == 0 {
            return;
        }

        // Step 1: Evaluate fitness
        for individual in &mut self.individuals {
            individual.fitness = fitness_fn(&individual.program);
        }

        // Step 2: Apply covariant parsimony pressure
        let c = parsimony_coefficient(&self.individuals);
        if c > 0.0 {
            for individual in &mut self.individuals {
                let size = individual.program.root.node_count() as f64;
                individual.fitness = (individual.fitness - c * size).max(0.0);
            }
        }

        // Step 3 & 4: Create offspring
        let num_offspring = pop_size / 4; // Replace bottom 25%
        let tournament_size = 3;
        let mut offspring: Vec<Individual> = Vec::with_capacity(num_offspring);

        for _ in 0..num_offspring {
            let parent_a = tournament_select(&self.individuals, tournament_size, rng);
            let parent_b = tournament_select(&self.individuals, tournament_size, rng);

            let mut child_program = crossover(&parent_a.program, &parent_b.program, rng);

            // Mutate with 95% probability (high mutation rate for exploration)
            if rng.gen_bool(0.95) {
                child_program = mutate(&child_program, rng);
            }

            // Hard size limit: reject programs with more than 200 nodes
            if child_program.root.node_count() > 200 {
                child_program = random_program(rng, 3);
            }

            let child_fitness = fitness_fn(&child_program);
            offspring.push(Individual {
                program: child_program,
                fitness: child_fitness,
            });
        }

        // Step 5: Sort population by fitness (ascending) and replace the worst
        self.individuals
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));

        // Replace the worst individuals (first in the sorted-ascending list)
        let replace_count = offspring.len().min(pop_size);
        for (i, child) in offspring.into_iter().enumerate() {
            if i < replace_count {
                self.individuals[i] = child;
            }
        }

        self.generation += 1;
    }

    /// Return a reference to the individual with the highest fitness.
    pub fn best(&self) -> Option<&Individual> {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ---------------------------------------------------------------------------
// Island model
// ---------------------------------------------------------------------------

/// Island model: multiple populations evolving independently with periodic migration.
///
/// Each island is an independent `Population` that undergoes its own selection,
/// crossover, and mutation. At regular intervals, the best individuals from each
/// island migrate to the next island in a ring topology. Every 50 generations,
/// the worst-performing island is culled and reseeded from the best island
/// (FunSearch-style).
pub struct IslandModel {
    /// The independent island populations.
    pub islands: Vec<Population>,
    /// Current global generation counter.
    pub generation: u32,
    /// How often (in generations) migration occurs.
    pub migration_interval: u32,
    /// Number of individuals to migrate between islands each migration event.
    pub migration_rate: usize,
}

impl IslandModel {
    /// Create an island model with `num_islands` islands, each of `island_size` individuals.
    pub fn new(num_islands: usize, island_size: usize, rng: &mut impl Rng) -> Self {
        let islands: Vec<Population> = (0..num_islands)
            .map(|_| Population::new(island_size, rng))
            .collect();

        IslandModel {
            islands,
            generation: 0,
            migration_interval: 10,
            migration_rate: 2,
        }
    }

    /// Evolve all islands for one generation, with migration at intervals.
    ///
    /// After every `migration_interval` generations, the best individuals from
    /// each island are sent to the next island in a ring topology. Every 50
    /// generations, the worst island is culled and reseeded from the best.
    pub fn evolve_generation(
        &mut self,
        rng: &mut impl Rng,
        fitness_fn: &dyn Fn(&Program) -> f64,
    ) {
        // Evolve each island independently
        for island in &mut self.islands {
            island.evolve_generation(rng, fitness_fn);
        }

        self.generation += 1;

        // Migration: ring topology, send best individuals to next island
        if self.generation % self.migration_interval == 0 && self.islands.len() > 1 {
            self.migrate();
        }

        // FunSearch-style culling: every 50 generations, kill worst island and reseed
        if self.generation % 50 == 0 && self.islands.len() > 2 {
            self.cull_worst_island(rng);
        }
    }

    /// Ring topology migration: best individuals from island i go to island (i+1) % n.
    fn migrate(&mut self) {
        let n = self.islands.len();
        if n < 2 {
            return;
        }

        // Collect migrants (best from each island)
        let mut migrants: Vec<Vec<Individual>> = Vec::with_capacity(n);
        for island in &self.islands {
            let mut sorted: Vec<Individual> = island.individuals.clone();
            sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
            let top: Vec<Individual> = sorted.into_iter().take(self.migration_rate).collect();
            migrants.push(top);
        }

        // Send migrants to next island, replacing worst individuals
        for i in 0..n {
            let dest = (i + 1) % n;
            let incoming = migrants[i].clone();

            // Sort destination island ascending by fitness
            self.islands[dest].individuals.sort_by(|a, b| {
                a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal)
            });

            // Replace worst with incoming migrants
            for (j, migrant) in incoming.into_iter().enumerate() {
                if j < self.islands[dest].individuals.len() {
                    self.islands[dest].individuals[j] = migrant;
                }
            }
        }
    }

    /// Kill the worst-performing island and reseed from the best.
    fn cull_worst_island(&mut self, rng: &mut impl Rng) {
        if self.islands.len() < 3 {
            return;
        }

        // Find best and worst islands by their best individual's fitness
        let island_bests: Vec<f64> = self
            .islands
            .iter()
            .map(|island| island.best().map(|b| b.fitness).unwrap_or(0.0))
            .collect();

        let worst_idx = island_bests
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_idx = island_bests
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if worst_idx == best_idx {
            return;
        }

        // Get the best program from the best island to seed the new one
        let seed_program = self.islands[best_idx].best().map(|b| b.program.clone());

        // Create a fresh island with seed
        let island_size = self.islands[worst_idx].individuals.len();
        let mut new_island = Population::new(island_size, rng);

        // Replace first individual with the seed from the best island
        if let Some(seed) = seed_program {
            if !new_island.individuals.is_empty() {
                new_island.individuals[0] = Individual {
                    program: seed,
                    fitness: 0.0,
                };
            }
        }

        self.islands[worst_idx] = new_island;
    }

    /// Get the global best individual across all islands.
    pub fn global_best(&self) -> Option<&Individual> {
        self.islands
            .iter()
            .filter_map(|island| island.best())
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_program_valid() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let prog = random_program(&mut rng, 4);
            assert!(prog.root.node_count() >= 1, "Random program should have at least 1 node");
        }
    }

    #[test]
    fn test_mutation_produces_different() {
        let mut rng = rand::thread_rng();
        let original = seed_pollard_rho();
        let original_str = format!("{}", original);

        // Mutation is probabilistic -- try many times and check that at least
        // one mutation produces a different program.
        let mut any_different = false;
        for _ in 0..50 {
            let mutated = mutate(&original, &mut rng);
            let mutated_str = format!("{}", mutated);
            if mutated_str != original_str {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "At least one mutation out of 50 should produce a different program"
        );
    }

    #[test]
    fn test_crossover_produces_program() {
        let mut rng = rand::thread_rng();
        let a = seed_pollard_rho();
        let b = seed_trial_like();
        let child = crossover(&a, &b, &mut rng);
        assert!(
            child.root.node_count() >= 1,
            "Crossover should produce a valid program"
        );
    }

    #[test]
    fn test_tournament_select() {
        let mut rng = rand::thread_rng();
        let population = vec![
            Individual {
                program: seed_pollard_rho(),
                fitness: 1.0,
            },
            Individual {
                program: seed_trial_like(),
                fitness: 10.0,
            },
            Individual {
                program: seed_fermat_like(),
                fitness: 5.0,
            },
        ];
        // With tournament size equal to population, the best individual should
        // be selected most of the time (sampling is with replacement, so
        // occasionally a suboptimal pick is possible). Over 100 trials the
        // best (fitness=10.0) should win the majority.
        let mut best_count = 0;
        for _ in 0..100 {
            let selected = tournament_select(&population, 3, &mut rng);
            if (selected.fitness - 10.0).abs() < f64::EPSILON {
                best_count += 1;
            }
        }
        assert!(
            best_count >= 50,
            "Tournament should select the best individual most of the time, but only got {}/100",
            best_count
        );
    }

    #[test]
    fn test_population_new() {
        let mut rng = rand::thread_rng();
        let pop = Population::new(20, &mut rng);
        assert_eq!(pop.individuals.len(), 20);
        assert_eq!(pop.generation, 0);
    }

    #[test]
    fn test_island_model_creation() {
        let mut rng = rand::thread_rng();
        let model = IslandModel::new(3, 10, &mut rng);
        assert_eq!(model.islands.len(), 3);
        assert_eq!(model.islands[0].individuals.len(), 10);
        assert_eq!(model.generation, 0);
    }

    #[test]
    fn test_parsimony_coefficient() {
        // With uniform size, coefficient should be near 0
        let pop = vec![
            Individual {
                program: seed_pollard_rho(),
                fitness: 10.0,
            },
            Individual {
                program: seed_pollard_rho(),
                fitness: 20.0,
            },
            Individual {
                program: seed_pollard_rho(),
                fitness: 30.0,
            },
        ];
        let c = parsimony_coefficient(&pop);
        assert!(
            c.abs() < 1e-6,
            "Uniform-size population should have near-zero parsimony coefficient, got {}",
            c
        );
    }
}
