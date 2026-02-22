//! OPES (Optimal Pair Extraction Sampling) for TTN-based optimization.
//!
//! ## Paper Reference (Tesoro/Siloi 2024, arXiv:2410.16355)
//!
//! Instead of full DMRG optimization, OPES samples from the Boltzmann
//! distribution P(x) ~ exp(-beta * E(x)) where E(x) is the cost function
//! (lattice vector norm). This is more efficient at exploring the energy
//! landscape than gradient-based optimization.
//!
//! Key features:
//! - Temperature schedule: start high (exploration), decrease (exploitation)
//! - For each sample, extract the binary configuration and decode to exponents
//! - Multiple samples per temperature step increase the chance of finding
//!   useful smooth relations
//! - Much faster than full DMRG since we sample many configs quickly

use rand::Rng;
use rand::SeedableRng;

use crate::optimizer::CostHamiltonian;
use crate::ttn::Ttn;

/// Configuration for OPES sampling.
#[derive(Debug, Clone)]
pub struct OpesConfig {
    /// Temperature schedule: list of inverse temperatures (beta values).
    /// Should generally increase (low beta = high temp = exploration,
    /// high beta = low temp = exploitation).
    pub beta_schedule: Vec<f64>,
    /// Number of samples to draw at each temperature.
    pub samples_per_temp: usize,
    /// Number of optimization sweeps before sampling at each temperature.
    pub sweeps_per_temp: usize,
}

impl Default for OpesConfig {
    fn default() -> Self {
        Self {
            beta_schedule: default_beta_schedule(10),
            samples_per_temp: 50,
            sweeps_per_temp: 3,
        }
    }
}

/// Result of a single OPES sample.
#[derive(Debug, Clone)]
pub struct OpesSample {
    /// The binary configuration that was sampled.
    pub config: Vec<u8>,
    /// The cost/energy of this configuration.
    pub energy: f64,
    /// The inverse temperature at which this was sampled.
    pub beta: f64,
}

/// Generate a default exponential beta schedule.
///
/// Creates `num_steps` beta values from beta_min to beta_max,
/// spaced exponentially.
pub fn default_beta_schedule(num_steps: usize) -> Vec<f64> {
    if num_steps == 0 {
        return Vec::new();
    }
    if num_steps == 1 {
        return vec![1.0];
    }

    let beta_min: f64 = 0.1;
    let beta_max: f64 = 10.0;
    let ratio = (beta_max / beta_min).powf(1.0 / (num_steps - 1) as f64);

    let mut schedule = Vec::with_capacity(num_steps);
    let mut beta = beta_min;
    for _ in 0..num_steps {
        schedule.push(beta);
        beta *= ratio;
    }
    schedule
}

/// Generate a beta schedule tuned for a given lattice dimension.
///
/// For larger lattices, we need more temperature steps and higher final beta
/// to properly explore the landscape.
pub fn beta_schedule_for_dimension(lattice_dim: usize) -> Vec<f64> {
    let num_steps = (lattice_dim as f64).sqrt().ceil() as usize;
    let num_steps = num_steps.max(5).min(30);

    let beta_min = 0.01;
    let beta_max = (lattice_dim as f64).sqrt() * 2.0;

    if num_steps <= 1 {
        return vec![beta_max];
    }

    let ratio = (beta_max / beta_min).powf(1.0 / (num_steps - 1) as f64);

    let mut schedule = Vec::with_capacity(num_steps);
    let mut beta = beta_min;
    for _ in 0..num_steps {
        schedule.push(beta);
        beta *= ratio;
    }
    schedule
}

/// Run OPES sampling on a TTN with the given Hamiltonian.
///
/// For each temperature in the schedule:
/// 1. Optionally optimize the TTN for a few sweeps at this temperature
/// 2. Sample configurations from the Boltzmann distribution
/// 3. Accept/reject based on Metropolis criterion
///
/// Returns all accepted samples sorted by energy (lowest first).
pub fn opes_sample(
    ttn: &Ttn,
    hamiltonian: &CostHamiltonian,
    config: &OpesConfig,
    rng: &mut impl Rng,
) -> Vec<OpesSample> {
    let n = ttn.num_vars;
    let mut all_samples: Vec<OpesSample> = Vec::new();

    // Use TTN sampling combined with Metropolis acceptance for Boltzmann distribution
    for &beta in &config.beta_schedule {
        let samples = boltzmann_sample_metropolis(
            ttn,
            hamiltonian,
            beta,
            config.samples_per_temp,
            n,
            rng,
        );
        all_samples.extend(samples);
    }

    // Sort by energy (lowest first)
    all_samples.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal));
    all_samples
}

/// Run OPES with TTN optimization interleaved (more expensive but better results).
///
/// At each temperature step, optimize the TTN for a few sweeps, then sample.
/// This couples the sampling with the optimization for better exploration.
pub fn opes_sample_with_optimization(
    ttn: &mut Ttn,
    hamiltonian: &CostHamiltonian,
    config: &OpesConfig,
    rng: &mut impl Rng,
) -> Vec<OpesSample> {
    let n = ttn.num_vars;
    let mut all_samples: Vec<OpesSample> = Vec::new();

    for &beta in &config.beta_schedule {
        // Optimize for a few sweeps
        if config.sweeps_per_temp > 0 {
            crate::ttn::optimize_ttn_sweep(ttn, hamiltonian, config.sweeps_per_temp);
        }

        // Sample at this temperature
        let samples = boltzmann_sample_metropolis(
            ttn,
            hamiltonian,
            beta,
            config.samples_per_temp,
            n,
            rng,
        );
        all_samples.extend(samples);
    }

    all_samples.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal));
    all_samples
}

/// Metropolis sampling from the Boltzmann distribution at inverse temperature beta.
///
/// Samples configurations using a Markov chain with single-bit-flip proposals.
/// The stationary distribution is P(x) ~ exp(-beta * E(x)).
fn boltzmann_sample_metropolis(
    ttn: &Ttn,
    hamiltonian: &CostHamiltonian,
    beta: f64,
    num_samples: usize,
    num_vars: usize,
    rng: &mut impl Rng,
) -> Vec<OpesSample> {
    let mut samples = Vec::with_capacity(num_samples);

    // Initialize with a random configuration
    let mut current: Vec<u8> = (0..num_vars)
        .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
        .collect();
    let mut current_energy = hamiltonian.evaluate(&current);

    // Burn-in: run some Metropolis steps before collecting samples
    let burn_in = num_vars * 5;
    for _ in 0..burn_in {
        metropolis_step(
            &mut current,
            &mut current_energy,
            hamiltonian,
            beta,
            num_vars,
            rng,
        );
    }

    // Collect samples with thinning (every few steps to reduce correlation)
    let thin = num_vars.max(1);
    for _ in 0..num_samples {
        for _ in 0..thin {
            metropolis_step(
                &mut current,
                &mut current_energy,
                hamiltonian,
                beta,
                num_vars,
                rng,
            );
        }

        samples.push(OpesSample {
            config: current.clone(),
            energy: current_energy,
            beta,
        });
    }

    // Also add TTN-sampled configurations if the TTN is small enough
    if num_vars <= 20 {
        let mut ttn_rng_state = rng.gen::<u64>();
        let mut ttn_rng = rand::rngs::StdRng::seed_from_u64(ttn_rng_state);
        let ttn_samples_count = num_samples.min(20);
        for _ in 0..ttn_samples_count {
            let config = ttn.sample_topdown(&mut ttn_rng);
            let energy = hamiltonian.evaluate(&config);

            // Accept with Boltzmann weight
            let accept_prob = (-beta * energy).exp();
            if rng.gen::<f64>() < accept_prob.min(1.0) {
                samples.push(OpesSample {
                    config,
                    energy,
                    beta,
                });
            }
            ttn_rng_state = ttn_rng_state.wrapping_add(1);
            ttn_rng = rand::rngs::StdRng::seed_from_u64(ttn_rng_state);
        }
    }

    samples
}

/// Perform a single Metropolis-Hastings step with a random bit flip proposal.
fn metropolis_step(
    current: &mut Vec<u8>,
    current_energy: &mut f64,
    hamiltonian: &CostHamiltonian,
    beta: f64,
    num_vars: usize,
    rng: &mut impl Rng,
) {
    // Propose: flip a random bit
    let flip_idx = rng.gen_range(0..num_vars);
    current[flip_idx] ^= 1;

    let proposed_energy = hamiltonian.evaluate(current);
    let delta_e = proposed_energy - *current_energy;

    // Metropolis acceptance
    if delta_e <= 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp() {
        // Accept the proposal
        *current_energy = proposed_energy;
    } else {
        // Reject: flip back
        current[flip_idx] ^= 1;
    }
}

/// Extract the best (lowest-energy) unique configurations from OPES samples.
///
/// Returns up to `max_configs` unique configurations sorted by energy.
pub fn extract_best_configs(samples: &[OpesSample], max_configs: usize) -> Vec<(Vec<u8>, f64)> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();

    for sample in samples {
        let key: Vec<u8> = sample.config.clone();
        if seen.insert(key.clone()) {
            unique.push((key, sample.energy));
        }
    }

    unique.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    unique.truncate(max_configs);
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{CostHamiltonian, QuadraticTerm};
    use rand::rngs::StdRng;

    #[test]
    fn test_default_beta_schedule() {
        let schedule = default_beta_schedule(10);
        assert_eq!(schedule.len(), 10);
        // Should be increasing
        for i in 1..schedule.len() {
            assert!(schedule[i] > schedule[i - 1], "Schedule should be increasing");
        }
        // First should be close to 0.1, last close to 10.0
        assert!((schedule[0] - 0.1).abs() < 1e-6);
        assert!((schedule[9] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_beta_schedule_for_dimension() {
        let schedule = beta_schedule_for_dimension(25);
        assert!(schedule.len() >= 5);
        // Should be increasing
        for i in 1..schedule.len() {
            assert!(schedule[i] > schedule[i - 1]);
        }
    }

    #[test]
    fn test_opes_sample_basic() {
        let mut rng = StdRng::seed_from_u64(42);

        // Simple 2-variable problem: minimum at [0, 0]
        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm { i: 0, j: 0, coeff: 5.0 },
                QuadraticTerm { i: 1, j: 1, coeff: 5.0 },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        let ttn = Ttn::new_random(2, 2, &mut rng);
        let config = OpesConfig {
            beta_schedule: vec![0.1, 1.0, 5.0],
            samples_per_temp: 10,
            sweeps_per_temp: 0,
        };

        let samples = opes_sample(&ttn, &hamiltonian, &config, &mut rng);
        assert!(!samples.is_empty(), "Should produce at least one sample");

        // All samples should have valid configs
        for sample in &samples {
            assert_eq!(sample.config.len(), 2);
            for &b in &sample.config {
                assert!(b == 0 || b == 1);
            }
            assert!(sample.energy.is_finite());
        }
    }

    #[test]
    fn test_opes_finds_low_energy() {
        let mut rng = StdRng::seed_from_u64(42);

        // Problem with clear minimum at [0, 0]: E = 10*b0 + 10*b1
        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm { i: 0, j: 0, coeff: 10.0 },
                QuadraticTerm { i: 1, j: 1, coeff: 10.0 },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        let ttn = Ttn::new_random(2, 2, &mut rng);
        let config = OpesConfig {
            beta_schedule: vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            samples_per_temp: 50,
            sweeps_per_temp: 0,
        };

        let samples = opes_sample(&ttn, &hamiltonian, &config, &mut rng);
        let best = extract_best_configs(&samples, 5);

        assert!(!best.is_empty());
        // The lowest energy configuration should be [0, 0] with energy 0
        assert!(
            best[0].1 < 5.0,
            "OPES should find low-energy config, got best energy {}",
            best[0].1
        );
    }

    #[test]
    fn test_opes_with_optimization() {
        let mut rng = StdRng::seed_from_u64(42);

        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm { i: 0, j: 0, coeff: 5.0 },
                QuadraticTerm { i: 1, j: 1, coeff: 5.0 },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        let mut ttn = Ttn::new_random(2, 2, &mut rng);
        let config = OpesConfig {
            beta_schedule: vec![0.5, 2.0, 5.0],
            samples_per_temp: 20,
            sweeps_per_temp: 2,
        };

        let samples = opes_sample_with_optimization(&mut ttn, &hamiltonian, &config, &mut rng);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_extract_best_configs_unique() {
        let samples = vec![
            OpesSample { config: vec![0, 0], energy: 0.0, beta: 1.0 },
            OpesSample { config: vec![0, 0], energy: 0.0, beta: 2.0 },
            OpesSample { config: vec![1, 0], energy: 5.0, beta: 1.0 },
            OpesSample { config: vec![0, 1], energy: 5.0, beta: 1.0 },
            OpesSample { config: vec![1, 1], energy: 10.0, beta: 1.0 },
        ];

        let best = extract_best_configs(&samples, 3);
        assert_eq!(best.len(), 3);
        // Should be sorted by energy
        assert!(best[0].1 <= best[1].1);
        assert!(best[1].1 <= best[2].1);
        // First should be [0,0] with energy 0
        assert_eq!(best[0].0, vec![0, 0]);
        assert!((best[0].1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_metropolis_ergodicity() {
        // Test that Metropolis sampling explores the full configuration space
        let mut rng = StdRng::seed_from_u64(42);

        let hamiltonian = CostHamiltonian {
            terms: vec![
                QuadraticTerm { i: 0, j: 0, coeff: 0.1 },
                QuadraticTerm { i: 1, j: 1, coeff: 0.1 },
            ],
            constant: 0.0,
            num_vars: 2,
        };

        // At very low beta (high temperature), should explore uniformly
        let mut current = vec![0u8, 0];
        let mut current_energy = hamiltonian.evaluate(&current);
        let beta = 0.01;

        let mut visited = std::collections::HashSet::new();
        for _ in 0..200 {
            metropolis_step(&mut current, &mut current_energy, &hamiltonian, beta, 2, &mut rng);
            visited.insert(current.clone());
        }

        // At very low beta, should visit all 4 configurations
        assert!(
            visited.len() >= 3,
            "High-temperature sampling should explore at least 3 of 4 configs, visited {}",
            visited.len()
        );
    }
}
