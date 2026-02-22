//! Fitness evaluation for evolved programs.
//!
//! Programs are tested against a ladder of semiprimes of increasing bit size.
//! The fitness score rewards both correctness (finding factors) and speed.

use num_bigint::BigUint;
use num_traits::Zero;
use std::time::Instant;

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

        // Use catch_unwind to handle panics gracefully
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

                // Check timeout
                if elapsed_ms > timeout_ms {
                    continue;
                }

                // Check if the factor is valid
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
                        // Score: bits * (100 / time_ms)
                        // Use at least 1ms to avoid division by zero
                        let time_factor = 100.0 / (elapsed_ms.max(1) as f64);
                        score += bits_val as f64 * time_factor;
                    }
                }
            }
            Err(_) => {
                // Program panicked; score 0 for this attempt
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
        // The rho seed should pass at least the first cascade level (16-bit)
        assert!(
            result.score > 0.0,
            "Pollard rho seed should have positive cascaded fitness, got score={}",
            result.score
        );
        assert!(
            result.success_count > 0,
            "Pollard rho seed should factor at least one semiprime in cascade"
        );
        // Cascaded evaluation should test fewer total attempts if early levels fail,
        // but for Pollard rho it should advance past level 1.
        assert!(
            result.total_attempts >= 5,
            "Pollard rho should advance past the first cascade level (16-bit, 5 attempts), got {} attempts",
            result.total_attempts
        );
    }
}
