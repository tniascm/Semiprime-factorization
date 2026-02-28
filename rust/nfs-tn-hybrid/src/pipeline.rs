//! End-to-end NFS-TN hybrid pipeline.
//!
//! 1. Select NFS polynomial (reuse classical-nfs)
//! 2. Build joint QUBO Hamiltonian from NFS norms
//! 3. Optimize TTN and sample candidate (a,b) pairs
//! 4. Check smoothness of rational and algebraic norms
//! 5. Collect relations, run linear algebra, extract factors

use std::time::Instant;

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use classical_nfs::polynomial::select_polynomial;
use classical_nfs::sieve::{eval_homogeneous_abs, sieve_primes};

use tnss_factoring::optimizer::CostHamiltonian;

use crate::hamiltonian::{build_joint_hamiltonian, decode_ab};
use crate::NfsTnConfig;

/// Result of a single NFS-TN hybrid factoring attempt.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NfsTnResult {
    /// The semiprime N.
    pub n: u64,
    /// Whether factoring succeeded.
    pub success: bool,
    /// Factor found (0 if none).
    pub factor: u64,
    /// Total wall-clock time in milliseconds.
    pub time_ms: f64,
    /// NFS polynomial degree used.
    pub poly_degree: usize,
    /// NFS base m.
    pub m: u64,
    /// Number of binary variables in the QUBO.
    pub num_vars: usize,
    /// TTN bond dimension used.
    pub bond_dim: usize,
    /// Total (a,b) candidates sampled from TTN.
    pub candidates_sampled: usize,
    /// Number of candidates that were coprime (gcd(a,b)=1, b>0).
    pub valid_candidates: usize,
    /// Number of candidates with smooth rational norm.
    pub rational_smooth: usize,
    /// Number of candidates with smooth algebraic norm.
    pub algebraic_smooth: usize,
    /// Number of fully smooth relations (both sides).
    pub full_relations: usize,
    /// Mean rational norm of sampled candidates (log2).
    pub mean_rational_norm_log2: f64,
    /// Mean algebraic norm of sampled candidates (log2).
    pub mean_algebraic_norm_log2: f64,
}

/// Run the NFS-TN hybrid factoring pipeline on a single semiprime.
pub fn factor_nfs_tn(n: u64, config: &NfsTnConfig) -> NfsTnResult {
    let start = Instant::now();
    let n_big = BigUint::from(n);

    // Step 1: Polynomial selection
    let degree = config.poly_degree;
    let poly = select_polynomial(&n_big, degree);
    let m_u64 = poly.m.to_u64().unwrap_or(0);

    // Get polynomial coefficients as f64
    let poly_coeffs: Vec<f64> = poly
        .coefficients
        .iter()
        .map(|c| c.to_u64().unwrap_or(0) as f64)
        .collect();

    // Step 2: Build joint Hamiltonian
    let a_offset = 1i64 << (config.a_bits - 1); // center the a range around 0
    let (hamiltonian, total_vars) = build_joint_hamiltonian(
        m_u64,
        &poly_coeffs,
        config.a_bits,
        config.b_bits,
        a_offset,
        config.alpha_rational,
        config.alpha_algebraic,
        config.coprimality_penalty,
        10.0, // quadratization penalty
    );

    // Step 3: Sample candidates using simulated annealing MCMC on the QUBO
    // This is faster than TTN optimization for small variable counts and
    // directly tests whether the NFS norm landscape yields smooth pairs.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let bond_dim = config.bond_dim.max(2);

    let t_mcmc = Instant::now();
    let best_configs = mcmc_sample_qubo(
        &hamiltonian,
        total_vars,
        config.num_samples,
        config.num_sweeps,
        &mut rng,
    );
    eprintln!("    MCMC sample: {:.1}ms (vars={}, {} configs)",
        t_mcmc.elapsed().as_secs_f64() * 1000.0, total_vars, best_configs.len());

    // Step 5: Decode candidates and check smoothness
    let rational_fb = sieve_primes(config.rational_smooth_bound);
    let algebraic_fb = sieve_primes(config.algebraic_smooth_bound);

    let mut candidates_sampled = 0usize;
    let mut valid_candidates = 0usize;
    let mut rational_smooth_count = 0usize;
    let mut algebraic_smooth_count = 0usize;
    let mut full_relations = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;

    // Collect smooth relations
    let mut smooth_ab_pairs: Vec<(i64, i64)> = Vec::new();
    let mut rational_exps_list: Vec<Vec<u32>> = Vec::new();
    let mut algebraic_exps_list: Vec<Vec<u32>> = Vec::new();

    for (binary_config, _energy) in &best_configs {
        candidates_sampled += 1;
        let (a, b) = decode_ab(binary_config, config.a_bits, config.b_bits, a_offset);

        // Skip invalid: b must be > 0, gcd(a,b) = 1
        if b <= 0 {
            continue;
        }
        let g = gcd_i64(a.unsigned_abs(), b as u64);
        if g > 1 {
            continue;
        }
        valid_candidates += 1;

        // Compute rational norm: |a - b*m|
        let rat_norm = (a as i128 - (b as i128) * (m_u64 as i128)).unsigned_abs() as u64;
        if rat_norm == 0 {
            continue;
        }

        // Compute algebraic norm: |F(a, b)| using homogeneous evaluation
        let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);
        let alg_norm = alg_norm_big.to_u64().unwrap_or(u64::MAX);

        sum_rat_log2 += (rat_norm as f64).log2();
        sum_alg_log2 += if alg_norm > 0 {
            (alg_norm as f64).log2()
        } else {
            0.0
        };

        // Check rational smoothness
        let (rat_exps, rat_cofactor) = trial_divide(rat_norm, &rational_fb);
        let rat_smooth = rat_cofactor == 1;
        if rat_smooth {
            rational_smooth_count += 1;
        }

        // Check algebraic smoothness
        let (alg_exps, alg_cofactor) = trial_divide(alg_norm, &algebraic_fb);
        let alg_smooth = alg_cofactor == 1;
        if alg_smooth {
            algebraic_smooth_count += 1;
        }

        if rat_smooth && alg_smooth {
            full_relations += 1;
            smooth_ab_pairs.push((a, b));
            rational_exps_list.push(rat_exps);
            algebraic_exps_list.push(alg_exps);
        }
    }

    // Step 6: If we have enough relations, attempt factoring via linear algebra
    let target_relations = rational_fb.len() + 1;
    let mut factor_found = 0u64;

    if full_relations >= target_relations.min(3) {
        // Build exponent matrix over GF(2)
        let num_cols = rational_fb.len() + algebraic_fb.len() + 1; // +1 for sign
        let mut matrix: Vec<Vec<u8>> = Vec::new();

        for i in 0..full_relations {
            let mut row = vec![0u8; num_cols];
            let (a, b) = smooth_ab_pairs[i];

            // Sign bit
            if a - b * (m_u64 as i64) < 0 {
                row[0] = 1;
            }

            // Rational exponents mod 2
            for (j, &exp) in rational_exps_list[i].iter().enumerate() {
                row[1 + j] = (exp & 1) as u8;
            }

            // Algebraic exponents mod 2
            let offset = 1 + rational_fb.len();
            for (j, &exp) in algebraic_exps_list[i].iter().enumerate() {
                if offset + j < num_cols {
                    row[offset + j] = (exp & 1) as u8;
                }
            }

            matrix.push(row);
        }

        // Find GF(2) dependencies
        let deps = classical_nfs::linalg::find_dependencies(&matrix, num_cols);

        // Try each dependency for factor extraction
        for dep in &deps {
            if dep.is_empty() || dep.len() < 2 {
                continue;
            }

            // Compute X² = product of (a + bm) for rational side
            let mut x_product = 1u128;
            for &idx in dep {
                if idx < smooth_ab_pairs.len() {
                    let (a, b) = smooth_ab_pairs[idx];
                    let val = a as i128 + (b as i128) * (m_u64 as i128);
                    x_product = x_product.wrapping_mul(val.unsigned_abs());
                }
            }

            // Compute Y = product of p^(e/2) from half-exponents
            let mut combined_exp = vec![0u64; rational_fb.len()];
            for &idx in dep {
                if idx < rational_exps_list.len() {
                    for (j, &e) in rational_exps_list[idx].iter().enumerate() {
                        combined_exp[j] += e as u64;
                    }
                }
            }

            let mut y_product = 1u128;
            for (j, &e) in combined_exp.iter().enumerate() {
                if e % 2 != 0 {
                    continue; // Not a perfect square — skip this dependency
                }
                for _ in 0..(e / 2) {
                    y_product = y_product.wrapping_mul(rational_fb[j] as u128);
                }
            }

            // gcd(X - Y, N) and gcd(X + Y, N)
            let n_128 = n as u128;
            let x_mod = x_product % n_128;
            let y_mod = y_product % n_128;

            for diff in [
                (x_mod + n_128 - y_mod) % n_128,
                (x_mod + y_mod) % n_128,
            ] {
                let g = gcd_u128(diff, n_128);
                if g > 1 && g < n_128 {
                    factor_found = g as u64;
                    break;
                }
            }

            if factor_found > 0 {
                break;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let mean_rat_log2 = if valid_candidates > 0 {
        sum_rat_log2 / valid_candidates as f64
    } else {
        0.0
    };
    let mean_alg_log2 = if valid_candidates > 0 {
        sum_alg_log2 / valid_candidates as f64
    } else {
        0.0
    };

    NfsTnResult {
        n,
        success: factor_found > 0,
        factor: factor_found,
        time_ms: elapsed,
        poly_degree: degree,
        m: m_u64,
        num_vars: total_vars,
        bond_dim,
        candidates_sampled,
        valid_candidates,
        rational_smooth: rational_smooth_count,
        algebraic_smooth: algebraic_smooth_count,
        full_relations,
        mean_rational_norm_log2: mean_rat_log2,
        mean_algebraic_norm_log2: mean_alg_log2,
    }
}

/// Trial-divide val by the factor base, returning (exponents, cofactor).
fn trial_divide(mut val: u64, factor_base: &[u64]) -> (Vec<u32>, u64) {
    let mut exponents = vec![0u32; factor_base.len()];
    for (i, &p) in factor_base.iter().enumerate() {
        while val % p == 0 {
            val /= p;
            exponents[i] += 1;
        }
    }
    (exponents, val)
}

fn gcd_i64(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd_i64(b, a % b)
    }
}

fn gcd_u128(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd_u128(b, a % b)
    }
}

/// Simulated annealing MCMC sampler on the QUBO Hamiltonian.
///
/// Runs multiple independent chains at decreasing temperatures to find
/// low-energy configurations (small NFS norms). Returns unique (config, energy)
/// pairs sorted by energy.
fn mcmc_sample_qubo(
    hamiltonian: &CostHamiltonian,
    num_vars: usize,
    num_samples: usize,
    num_rounds: usize,
    rng: &mut StdRng,
) -> Vec<(Vec<u8>, f64)> {
    let mut all_configs: Vec<(Vec<u8>, f64)> = Vec::new();

    // Temperature schedule for simulated annealing
    let temps = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1];

    for _round in 0..num_rounds {
        // Initialize random configuration
        let mut current: Vec<u8> = (0..num_vars)
            .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
            .collect();
        let mut current_energy = hamiltonian.evaluate(&current);

        for &temp in &temps {
            let beta = 1.0 / temp;
            let steps_per_temp = num_samples / (temps.len() * num_rounds).max(1);

            for _ in 0..steps_per_temp.max(10) {
                // Single-bit flip proposal
                let flip_idx = rng.gen_range(0..num_vars);
                current[flip_idx] ^= 1;
                let new_energy = hamiltonian.evaluate(&current);

                let delta_e = new_energy - current_energy;
                let accept = if delta_e <= 0.0 {
                    true
                } else {
                    rng.gen::<f64>() < (-beta * delta_e).exp()
                };

                if accept {
                    current_energy = new_energy;
                    all_configs.push((current.clone(), current_energy));
                } else {
                    current[flip_idx] ^= 1; // revert
                }
            }
        }
    }

    // Sort by energy and deduplicate
    all_configs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all_configs.dedup_by(|a, b| a.0 == b.0);
    all_configs.truncate(num_samples);
    all_configs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_divide() {
        let fb = vec![2, 3, 5, 7];
        let (exps, cofactor) = trial_divide(360, &fb);
        // 360 = 2^3 * 3^2 * 5
        assert_eq!(exps, vec![3, 2, 1, 0]);
        assert_eq!(cofactor, 1);
    }

    #[test]
    fn test_trial_divide_cofactor() {
        let fb = vec![2, 3, 5];
        let (exps, cofactor) = trial_divide(77, &fb);
        // 77 = 7 * 11
        assert_eq!(exps, vec![0, 0, 0]);
        assert_eq!(cofactor, 77);
    }

    #[test]
    fn test_factor_nfs_tn_small() {
        // Very small test: 15 = 3 * 5
        let config = NfsTnConfig {
            a_bits: 4,
            b_bits: 3,
            poly_degree: 3,
            alpha_rational: 1.0,
            alpha_algebraic: 0.5,
            coprimality_penalty: 10.0,
            bond_dim: 2,
            num_sweeps: 3,
            num_samples: 50,
            factor_base_bound: 30,
            rational_smooth_bound: 30,
            algebraic_smooth_bound: 30,
            seed: 42,
        };

        let result = factor_nfs_tn(15, &config);
        // We don't necessarily expect success at this tiny scale,
        // but it should run without crashing
        assert!(result.candidates_sampled > 0);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_factor_nfs_tn_medium() {
        // 143 = 11 * 13
        let config = NfsTnConfig {
            a_bits: 5,
            b_bits: 3,
            poly_degree: 3,
            alpha_rational: 1.0,
            alpha_algebraic: 0.5,
            coprimality_penalty: 10.0,
            bond_dim: 2,
            num_sweeps: 3,
            num_samples: 50,
            factor_base_bound: 50,
            rational_smooth_bound: 50,
            algebraic_smooth_bound: 50,
            seed: 42,
        };

        let result = factor_nfs_tn(143, &config);
        assert!(result.candidates_sampled > 0);
        eprintln!(
            "N=143: success={}, factor={}, valid={}, rat_smooth={}, alg_smooth={}, full_rel={}",
            result.success,
            result.factor,
            result.valid_candidates,
            result.rational_smooth,
            result.algebraic_smooth,
            result.full_relations,
        );
    }
}
