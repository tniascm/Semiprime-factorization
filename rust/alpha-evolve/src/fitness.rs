//! Fitness evaluation for evolved programs.
//!
//! Programs are tested against a ladder of semiprimes of increasing bit size.
//! The fitness score rewards both correctness (finding factors), speed,
//! scaling behavior, and behavioral novelty.
//!
//! Three fitness functions are available:
//! - `evaluate_fitness`: Simple flat evaluation across bit sizes.
//! - `evaluate_fitness_cascaded`: Cascaded difficulty levels (skip hard if easy fails).
//! - `evaluate_fitness_multiobjective`: Multi-objective with scaling + novelty + efficiency.

use num_bigint::BigUint;
use num_traits::Zero;
use std::time::Instant;

use crate::novelty::BehaviorFingerprint;
use crate::Program;
use factoring_core::generate_rsa_target;

/// Result of evaluating a program's fitness across a suite of test semiprimes.
#[derive(Debug, Clone)]
pub struct FitnessResult {
    /// Number of semiprimes successfully factored.
    pub success_count: u32,
    /// Total number of semiprimes tested.
    pub total_attempts: u32,
    /// Largest semiprime bit size that was successfully factored.
    pub max_bits_factored: u32,
    /// Total wall-clock time in milliseconds across all attempts.
    pub total_time_ms: u64,
    /// Composite fitness score.
    pub score: f64,
}

/// Extended fitness result with multi-objective components.
#[derive(Debug, Clone)]
pub struct MultiObjectiveFitness {
    /// Base factoring result.
    pub base: FitnessResult,
    /// Scaling reward: quadratic in bits for programs that handle larger N.
    pub scaling_score: f64,
    /// Novelty bonus: distance from known behaviors.
    pub novelty_score: f64,
    /// Efficiency score: successes per primitive executed.
    pub efficiency_score: f64,
    /// Behavioral fingerprint for novelty archive.
    pub fingerprint: BehaviorFingerprint,
    /// Combined weighted score.
    pub combined_score: f64,
}

/// Generate a ladder of test semiprimes.
///
/// For each bit size in `bit_sizes`, generates `count_per_size` semiprimes
/// using `factoring_core::generate_rsa_target`.
pub fn semiprime_ladder(bit_sizes: &[u32], count_per_size: usize) -> Vec<(BigUint, u32)> {
    let mut rng = rand::thread_rng();
    let mut ladder: Vec<(BigUint, u32)> = Vec::with_capacity(bit_sizes.len() * count_per_size);

    for &bits in bit_sizes {
        for _ in 0..count_per_size {
            let target = generate_rsa_target(bits, &mut rng);
            ladder.push((target.n, bits));
        }
    }

    ladder
}

/// Evaluate a program's fitness against the standard semiprime ladder.
///
/// Tests at bit sizes [16, 20, 24, 28, 32] with 3 semiprimes per bit size.
/// Each attempt has a 100ms timeout. Panics are caught via `catch_unwind`.
///
/// Score = sum over successes of (bits * (100.0 / time_ms)).
/// Faster solutions at larger bit sizes score higher.
///
/// NOTE: The cascaded version (`evaluate_fitness_cascaded`) is preferred for
/// production use as it avoids wasting compute on programs that cannot factor
/// even small semiprimes.
pub fn evaluate_fitness(program: &Program) -> FitnessResult {
    let bit_sizes: Vec<u32> = vec![16, 20, 24, 28, 32];
    let count_per_size: usize = 3;
    let timeout_ms: u128 = 100;

    let ladder = semiprime_ladder(&bit_sizes, count_per_size);

    let mut success_count: u32 = 0;
    let mut total_attempts: u32 = 0;
    let mut max_bits_factored: u32 = 0;
    let mut total_time_ms: u64 = 0;
    let mut score: f64 = 0.0;

    for (n, bits) in &ladder {
        total_attempts += 1;

        let program_clone = program.clone();
        let n_clone = n.clone();
        let bits_val = *bits;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let start = Instant::now();
            let factor = program_clone.evaluate(&n_clone);
            let elapsed = start.elapsed();
            (factor, elapsed)
        }));

        match result {
            Ok((maybe_factor, elapsed)) => {
                let elapsed_ms = elapsed.as_millis();
                total_time_ms += elapsed_ms as u64;

                if elapsed_ms > timeout_ms {
                    continue;
                }

                if let Some(ref factor) = maybe_factor {
                    if !factor.is_zero()
                        && *factor > BigUint::from(1u32)
                        && *factor < n_clone
                        && (&n_clone % factor).is_zero()
                    {
                        success_count += 1;
                        if bits_val > max_bits_factored {
                            max_bits_factored = bits_val;
                        }
                        let time_factor = 100.0 / (elapsed_ms.max(1) as f64);
                        score += bits_val as f64 * time_factor;
                    }
                }
            }
            Err(_) => {
                continue;
            }
        }
    }

    FitnessResult {
        success_count,
        total_attempts,
        max_bits_factored,
        total_time_ms,
        score,
    }
}

/// A single difficulty level in the evaluation cascade.
struct CascadeLevel {
    /// Bit size of semiprimes at this level.
    bit_size: u32,
    /// Number of semiprimes to test at this level.
    count: usize,
    /// Minimum successes required to advance to the next level.
    min_successes_to_advance: u32,
}

/// Evaluate fitness with cascaded difficulty levels.
///
/// Programs must pass easier levels before being tested on harder ones.
/// This avoids wasting compute on programs that cannot factor small numbers.
/// Levels proceed from 16-bit up to 36-bit semiprimes.
pub fn evaluate_fitness_cascaded(program: &Program) -> FitnessResult {
    let levels: Vec<CascadeLevel> = vec![
        CascadeLevel { bit_size: 16, count: 5, min_successes_to_advance: 2 },
        CascadeLevel { bit_size: 20, count: 4, min_successes_to_advance: 1 },
        CascadeLevel { bit_size: 24, count: 3, min_successes_to_advance: 1 },
        CascadeLevel { bit_size: 28, count: 3, min_successes_to_advance: 1 },
        CascadeLevel { bit_size: 32, count: 3, min_successes_to_advance: 1 },
        CascadeLevel { bit_size: 36, count: 2, min_successes_to_advance: 1 },
    ];
    let timeout_ms: u128 = 100;

    let mut success_count: u32 = 0;
    let mut total_attempts: u32 = 0;
    let mut max_bits_factored: u32 = 0;
    let mut total_time_ms: u64 = 0;
    let mut score: f64 = 0.0;

    let mut rng = rand::thread_rng();

    for level in &levels {
        let mut level_successes = 0u32;

        for _ in 0..level.count {
            total_attempts += 1;
            let target = generate_rsa_target(level.bit_size, &mut rng);

            let program_clone = program.clone();
            let n_clone = target.n.clone();

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let start = Instant::now();
                let factor = program_clone.evaluate(&n_clone);
                let elapsed = start.elapsed();
                (factor, elapsed)
            }));

            match result {
                Ok((maybe_factor, elapsed)) => {
                    let elapsed_ms = elapsed.as_millis();
                    total_time_ms += elapsed_ms as u64;

                    if elapsed_ms > timeout_ms {
                        continue;
                    }

                    if let Some(ref factor) = maybe_factor {
                        if !factor.is_zero()
                            && *factor > BigUint::from(1u32)
                            && *factor < n_clone
                            && (&n_clone % factor).is_zero()
                        {
                            success_count += 1;
                            level_successes += 1;
                            if level.bit_size > max_bits_factored {
                                max_bits_factored = level.bit_size;
                            }
                            let time_factor = 100.0 / (elapsed_ms.max(1) as f64);
                            score += level.bit_size as f64 * time_factor;
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Don't advance to harder levels if we failed this one
        if level_successes < level.min_successes_to_advance {
            break;
        }
    }

    FitnessResult {
        success_count,
        total_attempts,
        max_bits_factored,
        total_time_ms,
        score,
    }
}

// ---------------------------------------------------------------------------
// Multi-objective fitness with novelty search
// ---------------------------------------------------------------------------

/// Extended cascade levels for multi-objective evaluation.
/// Extends to 64-bit semiprimes to test scaling behavior.
const MULTI_OBJECTIVE_LEVELS: &[(u32, usize, u32)] = &[
    // (bit_size, count, min_successes_to_advance)
    (16, 5, 3),
    (20, 4, 2),
    (24, 3, 2),
    (28, 3, 1),
    (32, 3, 1),
    (36, 3, 1),
    (40, 2, 1),
    (48, 2, 1),
    (56, 1, 1),
    (64, 1, 1),
];

/// Per-sample timeout for multi-objective evaluation (ms).
const MULTI_TIMEOUT_MS: u128 = 500;

/// Evaluate a program on a single semiprime, returning (success, time_ms).
fn evaluate_single(program: &Program, n: &BigUint) -> (bool, u128) {
    let program_clone = program.clone();
    let n_clone = n.clone();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let start = Instant::now();
        let factor = program_clone.evaluate(&n_clone);
        let elapsed = start.elapsed();
        (factor, elapsed)
    }));

    match result {
        Ok((maybe_factor, elapsed)) => {
            let elapsed_ms = elapsed.as_millis();
            if elapsed_ms > MULTI_TIMEOUT_MS {
                return (false, elapsed_ms);
            }
            let success = if let Some(ref factor) = maybe_factor {
                !factor.is_zero()
                    && *factor > BigUint::from(1u32)
                    && *factor < *n
                    && (n % factor).is_zero()
            } else {
                false
            };
            (success, elapsed_ms)
        }
        Err(_) => (false, MULTI_TIMEOUT_MS),
    }
}

/// Multi-objective fitness evaluation.
///
/// Combines three fitness components:
/// 1. **Scaling reward** (50%): `Σ(bits² × 100/time_ms)` — quadratic in bits
///    so programs that scale to larger N dominate.
/// 2. **Novelty bonus** (30%): Included via the returned fingerprint, to be
///    scored against the novelty archive by the caller.
/// 3. **Efficiency bonus** (20%): `success_count / program_size` — rewards
///    concise programs that factor more.
///
/// Uses cascaded evaluation with 10 levels (16-bit through 64-bit).
/// Returns a `MultiObjectiveFitness` with all component scores and a
/// behavioral fingerprint for novelty archive integration.
pub fn evaluate_fitness_multiobjective(program: &Program) -> MultiObjectiveFitness {
    let mut rng = rand::thread_rng();

    // Total test cases across all levels for fingerprint
    let total_tests: usize = MULTI_OBJECTIVE_LEVELS.iter().map(|(_, c, _)| *c as usize).sum();
    let mut fingerprint = BehaviorFingerprint::new(total_tests);
    let mut test_idx = 0usize;

    let mut success_count: u32 = 0;
    let mut total_attempts: u32 = 0;
    let mut max_bits_factored: u32 = 0;
    let mut total_time_ms: u64 = 0;
    let mut scaling_score: f64 = 0.0;

    for &(bit_size, count, min_to_advance) in MULTI_OBJECTIVE_LEVELS {
        let mut level_successes = 0u32;

        for _ in 0..count {
            total_attempts += 1;
            let target = generate_rsa_target(bit_size, &mut rng);
            let (success, elapsed_ms) = evaluate_single(program, &target.n);

            total_time_ms += elapsed_ms as u64;

            if success {
                success_count += 1;
                level_successes += 1;
                fingerprint.set_pass(test_idx);

                if bit_size > max_bits_factored {
                    max_bits_factored = bit_size;
                }

                // Scaling reward: quadratic in bits
                let time_factor = 100.0 / (elapsed_ms.max(1) as f64);
                scaling_score += (bit_size as f64) * (bit_size as f64) * time_factor;
            }

            test_idx += 1;
        }

        // Early termination: don't test harder levels if this one failed
        if level_successes < min_to_advance {
            break;
        }
    }

    // Efficiency score: successes per node in the program
    let program_size = program.root.node_count().max(1) as f64;
    let efficiency_score = success_count as f64 / program_size;

    // Base score (for backward compatibility)
    let base_score = scaling_score;

    // Combined weighted score (novelty is added by the caller from the archive)
    // Weights: 50% scaling, 20% efficiency, 30% reserved for novelty
    let combined_score = 0.7 * scaling_score + 0.3 * (efficiency_score * 1000.0);

    MultiObjectiveFitness {
        base: FitnessResult {
            success_count,
            total_attempts,
            max_bits_factored,
            total_time_ms,
            score: base_score,
        },
        scaling_score,
        novelty_score: 0.0, // Set by caller from novelty archive
        efficiency_score,
        fingerprint,
        combined_score,
    }
}

/// Compute the final score incorporating novelty from the archive.
///
/// Called after `evaluate_fitness_multiobjective` with the novelty score
/// from the archive.
pub fn finalize_score(fitness: &mut MultiObjectiveFitness, novelty_score: f64) {
    fitness.novelty_score = novelty_score;
    // Final weights: 50% scaling, 30% novelty, 20% efficiency
    fitness.combined_score = 0.5 * fitness.scaling_score
        + 0.3 * (novelty_score * 1000.0)
        + 0.2 * (fitness.efficiency_score * 1000.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_pollard_rho;

    #[test]
    fn test_semiprime_ladder_sizes() {
        let ladder = semiprime_ladder(&[16, 20], 2);
        assert_eq!(ladder.len(), 4);
        // Check that each entry has the expected bit size label
        assert_eq!(ladder[0].1, 16);
        assert_eq!(ladder[1].1, 16);
        assert_eq!(ladder[2].1, 20);
        assert_eq!(ladder[3].1, 20);
    }

    #[test]
    fn test_fitness_evaluation() {
        let program = seed_pollard_rho();
        let result = evaluate_fitness(&program);
        // The rho seed should manage to factor at least some 16-bit semiprimes
        assert!(
            result.score > 0.0,
            "Pollard rho seed should have positive fitness, got score={}",
            result.score
        );
        assert!(
            result.success_count > 0,
            "Pollard rho seed should factor at least one semiprime"
        );
    }

    #[test]
    fn test_cascaded_fitness_evaluation() {
        let program = seed_pollard_rho();
        let result = evaluate_fitness_cascaded(&program);
        assert!(
            result.score > 0.0,
            "Pollard rho seed should have positive cascaded fitness, got score={}",
            result.score
        );
        assert!(
            result.success_count > 0,
            "Pollard rho seed should factor at least one semiprime in cascade"
        );
        assert!(
            result.total_attempts >= 5,
            "Pollard rho should advance past the first cascade level (16-bit, 5 attempts), got {} attempts",
            result.total_attempts
        );
    }

    #[test]
    fn test_multiobjective_fitness() {
        let program = seed_pollard_rho();
        let result = evaluate_fitness_multiobjective(&program);

        assert!(
            result.base.success_count > 0,
            "Pollard rho should factor at least one semiprime"
        );
        assert!(
            result.scaling_score > 0.0,
            "Scaling score should be positive"
        );
        assert!(
            result.efficiency_score > 0.0,
            "Efficiency score should be positive"
        );
        assert!(
            result.combined_score > 0.0,
            "Combined score should be positive"
        );
        assert_eq!(
            result.fingerprint.pass_count(),
            result.base.success_count as usize,
            "Fingerprint pass count should match success count"
        );
    }

    #[test]
    fn test_multiobjective_fingerprint_size() {
        let total_tests: usize = MULTI_OBJECTIVE_LEVELS
            .iter()
            .map(|(_, c, _)| *c as usize)
            .sum();
        let program = seed_pollard_rho();
        let result = evaluate_fitness_multiobjective(&program);
        assert_eq!(
            result.fingerprint.bits.len(),
            total_tests,
            "Fingerprint should cover all test cases"
        );
    }

    #[test]
    fn test_finalize_score() {
        let program = seed_pollard_rho();
        let mut result = evaluate_fitness_multiobjective(&program);
        finalize_score(&mut result, 0.5);
        assert!(
            result.novelty_score > 0.0,
            "Novelty score should be set after finalize"
        );
        assert!(
            result.combined_score > 0.0,
            "Combined score should be positive after finalize"
        );
    }
}
