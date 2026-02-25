//! Island-model evolutionary search for CADO-NFS parameter optimization.
//!
//! Reuses the proven island model architecture from Alpha-Evolve, but with
//! `CadoParams` as the individual type instead of `Program`.
//!
//! Features:
//! - Tournament selection with configurable pressure
//! - Island model with ring-topology migration
//! - Worst-island culling (diversity reset)
//! - Elitism (preserve best individual per island)
//! - Fitness caching by parameter hash

use std::collections::HashMap;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::cado::CadoResult;
use crate::params::CadoParams;

/// An individual in the population: a CADO-NFS parameter configuration
/// with its evaluated fitness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamIndividual {
    /// The parameter configuration.
    pub params: CadoParams,
    /// Aggregated fitness score (higher = better).
    pub fitness: f64,
    /// Cached evaluation result from last CADO-NFS run.
    pub result: Option<CadoResult>,
}

impl ParamIndividual {
    /// Create a new individual with zero fitness.
    pub fn new(params: CadoParams) -> Self {
        ParamIndividual {
            params,
            fitness: 0.0,
            result: None,
        }
    }
}

/// A population (island) of parameter configurations.
#[derive(Debug, Clone)]
pub struct ParamPopulation {
    /// Individuals in this island.
    pub individuals: Vec<ParamIndividual>,
    /// Generation counter for this island.
    pub generation: u32,
    /// Best fitness ever seen on this island.
    pub best_ever_fitness: f64,
    /// Generations since fitness improved.
    pub stagnation_count: u32,
}

impl ParamPopulation {
    /// Create a new population with random individuals.
    pub fn new(size: usize, n_bits: u32, rng: &mut impl Rng) -> Self {
        let individuals: Vec<ParamIndividual> = (0..size)
            .map(|_| ParamIndividual::new(CadoParams::random(rng, n_bits)))
            .collect();

        ParamPopulation {
            individuals,
            generation: 0,
            best_ever_fitness: 0.0,
            stagnation_count: 0,
        }
    }

    /// Create a population seeded from specific parameters.
    ///
    /// First individual is the exact seed, next several are mutations,
    /// remainder are random for diversity.
    pub fn new_from_params(size: usize, n_bits: u32, seed: &CadoParams, rng: &mut impl Rng) -> Self {
        let mut individuals = Vec::with_capacity(size);

        // First individual: exact seed params
        individuals.push(ParamIndividual::new(seed.clone()));

        // Next half: mutations of seed (local neighborhood search)
        let seed_mutations = std::cmp::min(size / 2, size - 1);
        for _ in 0..seed_mutations {
            individuals.push(ParamIndividual::new(seed.mutate(rng, n_bits)));
        }

        // Rest: random (for diversity)
        while individuals.len() < size {
            individuals.push(ParamIndividual::new(CadoParams::random(rng, n_bits)));
        }

        ParamPopulation {
            individuals,
            generation: 0,
            best_ever_fitness: 0.0,
            stagnation_count: 0,
        }
    }

    /// Create a population seeded with default + nearby configurations.
    pub fn new_seeded(size: usize, n_bits: u32, rng: &mut impl Rng) -> Self {
        let mut individuals = Vec::with_capacity(size);

        // First individual: default parameters
        individuals.push(ParamIndividual::new(CadoParams::default_for_bits(n_bits)));

        // Next few: slight mutations of default
        let default = CadoParams::default_for_bits(n_bits);
        for _ in 0..std::cmp::min(4, size - 1) {
            individuals.push(ParamIndividual::new(default.mutate(rng, n_bits)));
        }

        // Rest: random
        while individuals.len() < size {
            individuals.push(ParamIndividual::new(CadoParams::random(rng, n_bits)));
        }

        ParamPopulation {
            individuals,
            generation: 0,
            best_ever_fitness: 0.0,
            stagnation_count: 0,
        }
    }

    /// Get the best individual in this population.
    pub fn best(&self) -> Option<&ParamIndividual> {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the worst individual's fitness.
    pub fn worst_fitness(&self) -> f64 {
        self.individuals
            .iter()
            .map(|i| i.fitness)
            .fold(f64::INFINITY, f64::min)
    }

    /// Average fitness of the population.
    pub fn avg_fitness(&self) -> f64 {
        if self.individuals.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.individuals.iter().map(|i| i.fitness).sum();
        sum / self.individuals.len() as f64
    }

    /// Tournament selection: pick `tournament_size` random individuals, return the best.
    pub fn tournament_select(&self, rng: &mut impl Rng, tournament_size: usize) -> &ParamIndividual {
        let n = self.individuals.len();
        let mut best_idx = rng.gen_range(0..n);
        let mut best_fitness = self.individuals[best_idx].fitness;

        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..n);
            if self.individuals[idx].fitness > best_fitness {
                best_idx = idx;
                best_fitness = self.individuals[idx].fitness;
            }
        }

        &self.individuals[best_idx]
    }

    /// Evolve one generation using tournament selection, crossover, and mutation.
    ///
    /// - `mutation_rate`: probability of mutation (vs crossover) [0.0, 1.0]
    /// - `tournament_size`: number of individuals in each tournament
    /// - `n_bits`: target semiprime bit size (for valid parameter ranges)
    pub fn evolve_generation(
        &mut self,
        rng: &mut impl Rng,
        mutation_rate: f64,
        tournament_size: usize,
        n_bits: u32,
    ) {
        let pop_size = self.individuals.len();
        let mut new_individuals = Vec::with_capacity(pop_size);

        // Elitism: keep the best individual
        if let Some(best) = self.best() {
            new_individuals.push(best.clone());
        }

        // Fill rest via selection + variation
        while new_individuals.len() < pop_size {
            let child_params = if rng.gen_bool(mutation_rate) {
                // Mutation
                let parent = self.tournament_select(rng, tournament_size);
                parent.params.mutate(rng, n_bits)
            } else {
                // Crossover
                let p1 = self.tournament_select(rng, tournament_size);
                let p2 = self.tournament_select(rng, tournament_size);
                let child = p1.params.crossover(&p2.params, rng);
                // Also mutate crossover offspring with 50% probability
                if rng.gen_bool(0.5) {
                    child.mutate(rng, n_bits)
                } else {
                    child
                }
            };

            new_individuals.push(ParamIndividual::new(child_params));
        }

        // Track stagnation
        let old_best = self.best_ever_fitness;
        self.individuals = new_individuals;
        self.generation += 1;

        if let Some(best) = self.best() {
            if best.fitness > old_best {
                self.best_ever_fitness = best.fitness;
                self.stagnation_count = 0;
            } else {
                self.stagnation_count += 1;
            }
        }
    }
}

/// Island model: multiple populations with periodic migration and culling.
#[derive(Debug, Clone)]
pub struct ParamIslandModel {
    /// The islands (populations).
    pub islands: Vec<ParamPopulation>,
    /// Global generation counter.
    pub generation: u32,
    /// Target semiprime bit size.
    pub n_bits: u32,
    /// Migration interval (generations between migrations).
    pub migration_interval: u32,
    /// Culling interval (generations between worst-island resets).
    pub culling_interval: u32,
    /// Tournament size for selection.
    pub tournament_size: usize,
    /// Mutation rate [0.0, 1.0].
    pub mutation_rate: f64,
}

/// Configuration for the island model.
pub struct IslandConfig {
    pub num_islands: usize,
    pub island_size: usize,
    pub n_bits: u32,
    pub migration_interval: u32,
    pub culling_interval: u32,
    pub tournament_size: usize,
    pub mutation_rate: f64,
}

impl IslandConfig {
    /// Quick configuration for testing.
    pub fn quick(n_bits: u32) -> Self {
        IslandConfig {
            num_islands: 3,
            island_size: 8,
            n_bits,
            migration_interval: 3,
            culling_interval: 10,
            tournament_size: 3,
            mutation_rate: 0.7,
        }
    }

    /// Full configuration for production runs.
    pub fn full(n_bits: u32) -> Self {
        IslandConfig {
            num_islands: 10,
            island_size: 50,
            n_bits,
            migration_interval: 10,
            culling_interval: 30,
            tournament_size: 4,
            mutation_rate: 0.7,
        }
    }
}

impl ParamIslandModel {
    /// Create a new island model with the given configuration.
    pub fn new(config: &IslandConfig, rng: &mut impl Rng) -> Self {
        let islands: Vec<ParamPopulation> = (0..config.num_islands)
            .map(|i| {
                if i == 0 {
                    // First island: seeded with defaults
                    ParamPopulation::new_seeded(config.island_size, config.n_bits, rng)
                } else {
                    ParamPopulation::new(config.island_size, config.n_bits, rng)
                }
            })
            .collect();

        ParamIslandModel {
            islands,
            generation: 0,
            n_bits: config.n_bits,
            migration_interval: config.migration_interval,
            culling_interval: config.culling_interval,
            tournament_size: config.tournament_size,
            mutation_rate: config.mutation_rate,
        }
    }

    /// Create a new island model seeded from specific parameters.
    ///
    /// Island 0: seeded with the provided params + mutations around it.
    /// Island 1: seeded with default params for the target bit size.
    /// Remaining islands: fully random for diversity.
    pub fn new_seeded_from(config: &IslandConfig, seed: &CadoParams, rng: &mut impl Rng) -> Self {
        let islands: Vec<ParamPopulation> = (0..config.num_islands)
            .map(|i| {
                if i == 0 {
                    // First island: seeded from provided params
                    ParamPopulation::new_from_params(
                        config.island_size, config.n_bits, seed, rng,
                    )
                } else if i == 1 {
                    // Second island: seeded from defaults
                    ParamPopulation::new_seeded(config.island_size, config.n_bits, rng)
                } else {
                    // Rest: fully random for diversity
                    ParamPopulation::new(config.island_size, config.n_bits, rng)
                }
            })
            .collect();

        ParamIslandModel {
            islands,
            generation: 0,
            n_bits: config.n_bits,
            migration_interval: config.migration_interval,
            culling_interval: config.culling_interval,
            tournament_size: config.tournament_size,
            mutation_rate: config.mutation_rate,
        }
    }

    /// Evolve one generation across all islands.
    ///
    /// After selection/variation, handles migration and culling if due.
    pub fn evolve_generation(&mut self, rng: &mut impl Rng) {
        // Evolve each island
        for island in &mut self.islands {
            island.evolve_generation(rng, self.mutation_rate, self.tournament_size, self.n_bits);
        }

        self.generation += 1;

        // Migration: ring topology
        if self.generation % self.migration_interval == 0 && self.islands.len() > 1 {
            self.migrate(rng);
        }

        // Culling: reset worst island
        if self.generation % self.culling_interval == 0 && self.islands.len() > 2 {
            self.cull_worst(rng);
        }
    }

    /// Migrate best individuals between islands (ring topology).
    ///
    /// Each island sends its best individual to the next island,
    /// replacing the worst individual there.
    fn migrate(&mut self, _rng: &mut impl Rng) {
        let n = self.islands.len();
        if n < 2 {
            return;
        }

        // Collect best individuals from each island
        let migrants: Vec<Option<ParamIndividual>> = self
            .islands
            .iter()
            .map(|island| island.best().cloned())
            .collect();

        // Send each best to the next island (ring), replacing worst
        for i in 0..n {
            let next = (i + 1) % n;
            if let Some(ref migrant) = migrants[i] {
                // Find worst in destination
                if let Some(worst_idx) = self.islands[next]
                    .individuals
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.fitness
                            .partial_cmp(&b.fitness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                {
                    // Only replace if migrant is better
                    if migrant.fitness > self.islands[next].individuals[worst_idx].fitness {
                        self.islands[next].individuals[worst_idx] = migrant.clone();
                    }
                }
            }
        }

        log::debug!("Migration at gen {}: ring topology across {} islands", self.generation, n);
    }

    /// Cull the worst-performing island by resetting it with fresh random individuals.
    fn cull_worst(&mut self, rng: &mut impl Rng) {
        if self.islands.len() <= 2 {
            return;
        }

        // Find island with lowest best fitness
        let worst_idx = self
            .islands
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_best = a.best().map(|i| i.fitness).unwrap_or(0.0);
                let b_best = b.best().map(|i| i.fitness).unwrap_or(0.0);
                a_best
                    .partial_cmp(&b_best)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let island_size = self.islands[worst_idx].individuals.len();

        log::info!(
            "Culling island {} (best fitness: {:.2})",
            worst_idx,
            self.islands[worst_idx]
                .best()
                .map(|i| i.fitness)
                .unwrap_or(0.0)
        );

        // Replace with fresh random individuals, keeping one copy of global best
        let mut new_pop = ParamPopulation::new(island_size, self.n_bits, rng);

        // Seed with global best if available
        if let Some(global_best) = self.global_best() {
            if !new_pop.individuals.is_empty() {
                new_pop.individuals[0] = global_best.clone();
            }
        }

        self.islands[worst_idx] = new_pop;
    }

    /// Get the best individual across all islands.
    pub fn global_best(&self) -> Option<&ParamIndividual> {
        self.islands
            .iter()
            .filter_map(|island| island.best())
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get fitness summary for all islands.
    pub fn island_fitness_summary(&self) -> Vec<(f64, f64)> {
        self.islands
            .iter()
            .map(|island| {
                let best = island.best().map(|i| i.fitness).unwrap_or(0.0);
                let avg = island.avg_fitness();
                (best, avg)
            })
            .collect()
    }

    /// Total number of individuals across all islands.
    pub fn total_individuals(&self) -> usize {
        self.islands.iter().map(|i| i.individuals.len()).sum()
    }
}

/// Cache for fitness evaluations keyed by parameter hash.
///
/// Avoids re-running CADO-NFS for identical parameter configurations.
pub struct FitnessCache {
    cache: HashMap<u64, f64>,
    max_size: usize,
}

impl FitnessCache {
    /// Create a new cache with given max capacity.
    pub fn new(max_size: usize) -> Self {
        FitnessCache {
            cache: HashMap::new(),
            max_size,
        }
    }

    /// Look up cached fitness for a parameter configuration.
    pub fn get(&self, params: &CadoParams) -> Option<f64> {
        self.cache.get(&params.fitness_hash()).copied()
    }

    /// Store fitness for a parameter configuration.
    pub fn insert(&mut self, params: &CadoParams, fitness: f64) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: clear half the cache
            let keys: Vec<u64> = self.cache.keys().take(self.max_size / 2).copied().collect();
            for k in keys {
                self.cache.remove(&k);
            }
        }
        self.cache.insert(params.fitness_hash(), fitness);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Hit rate statistics (for reporting).
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_individual_new() {
        let params = CadoParams::default_for_bits(100);
        let ind = ParamIndividual::new(params);
        assert_eq!(ind.fitness, 0.0);
        assert!(ind.result.is_none());
    }

    #[test]
    fn test_population_new() {
        let mut rng = thread_rng();
        let pop = ParamPopulation::new(10, 100, &mut rng);
        assert_eq!(pop.individuals.len(), 10);
        assert_eq!(pop.generation, 0);
    }

    #[test]
    fn test_population_seeded() {
        let mut rng = thread_rng();
        let pop = ParamPopulation::new_seeded(10, 100, &mut rng);
        assert_eq!(pop.individuals.len(), 10);
        // First individual should have default params
        let default = CadoParams::default_for_bits(100);
        assert_eq!(
            pop.individuals[0].params.poly_degree,
            default.poly_degree
        );
    }

    #[test]
    fn test_tournament_select() {
        let mut rng = thread_rng();
        let mut pop = ParamPopulation::new(10, 100, &mut rng);
        // Set one individual to have high fitness
        pop.individuals[3].fitness = 100.0;
        // Run tournament 50 times — with tournament size 5, the best individual
        // should be selected most of the time
        let mut found_best = false;
        for _ in 0..50 {
            let selected = pop.tournament_select(&mut rng, 5);
            if selected.fitness == 100.0 {
                found_best = true;
                break;
            }
        }
        assert!(found_best, "Tournament should find the best individual over 50 trials");
    }

    #[test]
    fn test_evolve_generation() {
        let mut rng = thread_rng();
        let mut pop = ParamPopulation::new(20, 100, &mut rng);
        // Set some fitness values
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f64;
        }
        pop.evolve_generation(&mut rng, 0.7, 3, 100);
        assert_eq!(pop.individuals.len(), 20);
        assert_eq!(pop.generation, 1);
    }

    #[test]
    fn test_island_model_creation() {
        let mut rng = thread_rng();
        let config = IslandConfig::quick(100);
        let model = ParamIslandModel::new(&config, &mut rng);
        assert_eq!(model.islands.len(), 3);
        assert_eq!(model.total_individuals(), 24); // 3 * 8
    }

    #[test]
    fn test_island_model_evolve() {
        let mut rng = thread_rng();
        let config = IslandConfig::quick(100);
        let mut model = ParamIslandModel::new(&config, &mut rng);

        // Set some fitness values so selection works
        for island in &mut model.islands {
            for (i, ind) in island.individuals.iter_mut().enumerate() {
                ind.fitness = i as f64;
            }
        }

        // Evolve 20 generations (should trigger migration and culling)
        for _ in 0..20 {
            model.evolve_generation(&mut rng);
        }

        assert_eq!(model.generation, 20);
        assert_eq!(model.islands.len(), 3);
    }

    #[test]
    fn test_global_best() {
        let mut rng = thread_rng();
        let config = IslandConfig::quick(100);
        let mut model = ParamIslandModel::new(&config, &mut rng);
        model.islands[2].individuals[0].fitness = 999.0;

        let best = model.global_best().unwrap();
        assert_eq!(best.fitness, 999.0);
    }

    #[test]
    fn test_fitness_cache() {
        let mut cache = FitnessCache::new(100);
        let params = CadoParams::default_for_bits(100);

        assert!(cache.get(&params).is_none());
        cache.insert(&params, 42.0);
        assert_eq!(cache.get(&params), Some(42.0));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_fitness_cache_eviction() {
        let mut rng = thread_rng();
        let mut cache = FitnessCache::new(10);

        // Fill beyond capacity
        for _ in 0..20 {
            let params = CadoParams::random(&mut rng, 100);
            cache.insert(&params, 1.0);
        }

        // Should have evicted some entries
        assert!(cache.len() <= 15); // After eviction, shouldn't be at max
    }

    #[test]
    fn test_island_fitness_summary() {
        let mut rng = thread_rng();
        let config = IslandConfig::quick(100);
        let model = ParamIslandModel::new(&config, &mut rng);
        let summary = model.island_fitness_summary();
        assert_eq!(summary.len(), 3);
    }
}
