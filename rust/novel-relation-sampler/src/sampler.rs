//! Uniform, MCMC, and lattice sieve samplers for NFS (a,b) pair generation.

use std::collections::HashSet;
use std::time::Instant;

use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::Rng;

use classical_nfs::polynomial::NfsPolynomial;
use classical_nfs::sieve::{
    compute_polynomial_roots, eval_homogeneous_abs, line_sieve, sieve_primes, SieveConfig,
};

use crate::smoothness::{gcd, is_smooth};
use crate::{MethodResult, SieveParams};

/// Compute the log-norm product energy.
/// H(a,b) = log(max(|a - b*m|, 1)) + log(max(|F(a,b)|, 1))
fn compute_energy(a: i64, b: i64, m: u64, poly: &NfsPolynomial) -> (u64, u64, f64) {
    let rat_norm = (a as i128 - (b as i128) * (m as i128)).unsigned_abs() as u64;
    let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);
    let alg_norm = alg_norm_big.to_u64().unwrap_or(u64::MAX);

    let energy = (rat_norm.max(1) as f64).ln() + (alg_norm.max(1) as f64).ln();
    (rat_norm, alg_norm, energy)
}

const LN_2: f64 = std::f64::consts::LN_2;

/// Compute energy with accurate BigUint handling for large N.
///
/// Returns (rat_norm, alg_norm_option, energy):
/// - alg_norm_option is None when the algebraic norm exceeds u64, meaning
///   it's almost certainly not B-smooth for any reasonable factor base.
/// - Energy is computed accurately even for huge norms using bits()*ln(2).
pub fn compute_energy_big(a: i64, b: i64, m: u64, poly: &NfsPolynomial) -> (u64, Option<u64>, f64) {
    let rat_norm = (a as i128 - (b as i128) * (m as i128)).unsigned_abs() as u64;
    let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);

    let rat_energy = (rat_norm.max(1) as f64).ln();
    let (alg_norm_opt, alg_energy) = match alg_norm_big.to_u64() {
        Some(alg) => (Some(alg), (alg.max(1) as f64).ln()),
        None => (None, alg_norm_big.bits() as f64 * LN_2),
    };

    (rat_norm, alg_norm_opt, rat_energy + alg_energy)
}

/// Run uniform random sampling over the sieve range.
pub fn sample_uniform(
    poly: &NfsPolynomial,
    m: u64,
    params: &SieveParams,
    num_candidates: usize,
    rng: &mut StdRng,
) -> MethodResult {
    let start = Instant::now();
    let factor_base = sieve_primes(params.fb_bound);
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    let mut tested = 0usize;
    let mut valid = 0usize;
    let mut rat_smooth_count = 0usize;
    let mut alg_smooth_count = 0usize;
    let mut both_smooth_count = 0usize;
    let mut unique_valid = 0usize;
    let mut unique_both_smooth = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;
    let mut sum_energy = 0.0f64;

    while tested < num_candidates {
        let a = rng.gen_range(-params.sieve_area..=params.sieve_area);
        let b = rng.gen_range(1..=params.max_b);
        tested += 1;

        if gcd(a.unsigned_abs(), b as u64) > 1 {
            continue;
        }

        let (rat_norm, alg_norm, energy) = compute_energy(a, b, m, poly);
        if rat_norm == 0 || alg_norm == 0 {
            continue;
        }

        valid += 1;
        let is_new = seen.insert((a, b));
        if is_new {
            unique_valid += 1;
        }

        sum_rat_log2 += (rat_norm as f64).log2();
        sum_alg_log2 += (alg_norm as f64).log2();
        sum_energy += energy;

        let rs = is_smooth(rat_norm, &factor_base);
        let als = is_smooth(alg_norm, &factor_base);
        if rs {
            rat_smooth_count += 1;
        }
        if als {
            alg_smooth_count += 1;
        }
        if rs && als {
            both_smooth_count += 1;
            if is_new {
                unique_both_smooth += 1;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    MethodResult {
        method: "uniform".to_string(),
        candidates_tested: tested,
        valid_candidates: valid,
        rational_smooth: rat_smooth_count,
        algebraic_smooth: alg_smooth_count,
        both_smooth: both_smooth_count,
        unique_valid,
        unique_both_smooth,
        mean_rat_norm_log2: if valid > 0 { sum_rat_log2 / valid as f64 } else { 0.0 },
        mean_alg_norm_log2: if valid > 0 { sum_alg_log2 / valid as f64 } else { 0.0 },
        mean_energy: if valid > 0 { sum_energy / valid as f64 } else { 0.0 },
        time_ms: elapsed,
    }
}

/// Run MCMC sampling with log-norm product energy.
///
/// Uses simulated annealing with multi-scale proposals and geometric
/// temperature cooling. Multiple independent chains for coverage.
pub fn sample_mcmc(
    poly: &NfsPolynomial,
    m: u64,
    params: &SieveParams,
    num_candidates: usize,
    num_chains: usize,
    t_start: f64,
    t_end: f64,
    rng: &mut StdRng,
) -> MethodResult {
    let start = Instant::now();
    let factor_base = sieve_primes(params.fb_bound);
    let steps_per_chain = num_candidates / num_chains.max(1);
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    let mut tested = 0usize;
    let mut valid = 0usize;
    let mut rat_smooth_count = 0usize;
    let mut alg_smooth_count = 0usize;
    let mut both_smooth_count = 0usize;
    let mut unique_valid = 0usize;
    let mut unique_both_smooth = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;
    let mut sum_energy = 0.0f64;

    let log_ratio = (t_end / t_start).ln();

    for _chain in 0..num_chains {
        // Initialize chain at random position in sieve range
        let mut cur_a = rng.gen_range(-params.sieve_area..=params.sieve_area);
        let mut cur_b = rng.gen_range(1..=params.max_b);
        let (mut cur_rat, mut cur_alg, mut cur_energy) = compute_energy(cur_a, cur_b, m, poly);

        for step in 0..steps_per_chain {
            // Geometric temperature schedule
            let frac = step as f64 / (steps_per_chain.max(1) - 1).max(1) as f64;
            let temp = t_start * (frac * log_ratio).exp();
            let beta = 1.0 / temp;

            // Multi-scale proposal
            let (prop_a, prop_b) = propose(cur_a, cur_b, params, rng);

            let (prop_rat, prop_alg, prop_energy) = compute_energy(prop_a, prop_b, m, poly);
            let delta_e = prop_energy - cur_energy;

            let accept = delta_e <= 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp();

            if accept {
                cur_a = prop_a;
                cur_b = prop_b;
                cur_rat = prop_rat;
                cur_alg = prop_alg;
                cur_energy = prop_energy;
            }

            // Record the current state as a candidate
            tested += 1;
            if gcd(cur_a.unsigned_abs(), cur_b as u64) > 1 || cur_rat == 0 || cur_alg == 0 {
                continue;
            }

            valid += 1;
            let is_new = seen.insert((cur_a, cur_b));
            if is_new {
                unique_valid += 1;
            }

            sum_rat_log2 += (cur_rat as f64).log2();
            sum_alg_log2 += (cur_alg as f64).log2();
            sum_energy += cur_energy;

            let rs = is_smooth(cur_rat, &factor_base);
            let als = is_smooth(cur_alg, &factor_base);
            if rs {
                rat_smooth_count += 1;
            }
            if als {
                alg_smooth_count += 1;
            }
            if rs && als {
                both_smooth_count += 1;
                if is_new {
                    unique_both_smooth += 1;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    MethodResult {
        method: "mcmc".to_string(),
        candidates_tested: tested,
        valid_candidates: valid,
        rational_smooth: rat_smooth_count,
        algebraic_smooth: alg_smooth_count,
        both_smooth: both_smooth_count,
        unique_valid,
        unique_both_smooth,
        mean_rat_norm_log2: if valid > 0 { sum_rat_log2 / valid as f64 } else { 0.0 },
        mean_alg_norm_log2: if valid > 0 { sum_alg_log2 / valid as f64 } else { 0.0 },
        mean_energy: if valid > 0 { sum_energy / valid as f64 } else { 0.0 },
        time_ms: elapsed,
    }
}

/// Run lattice sieve (the standard NFS baseline) over the sieve range.
///
/// Calls `line_sieve()` from classical-nfs to enumerate all (a,b) in the grid,
/// accumulate log(p) for factor base primes dividing norms, then trial-divides
/// sieve hits to confirm smoothness.
pub fn sample_lattice_sieve(
    poly: &NfsPolynomial,
    m: u64,
    params: &SieveParams,
) -> MethodResult {
    let start = Instant::now();
    let factor_base = sieve_primes(params.fb_bound);

    // Build SieveConfig from our SieveParams
    let config = SieveConfig {
        rational_bound: params.fb_bound,
        algebraic_bound: params.fb_bound,
        sieve_area: params.sieve_area,
        max_b: params.max_b,
        large_prime_multiplier: 1,
        sieve_threshold: 0.0, // line_sieve computes dynamic thresholds
    };

    // Compute algebraic roots for sieving
    let alg_roots = compute_polynomial_roots(poly, &factor_base);

    // Run the line sieve
    let hits = line_sieve(poly, &factor_base, &alg_roots, &config);

    // Total grid size
    let grid_size = (2 * params.sieve_area + 1) as usize * params.max_b as usize;

    // Trial-divide each hit to confirm smoothness
    let mut valid = 0usize;
    let mut rat_smooth_count = 0usize;
    let mut alg_smooth_count = 0usize;
    let mut both_smooth_count = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;
    let mut sum_energy = 0.0f64;

    for hit in &hits {
        let rat_norm = (hit.a as i128 - (hit.b as i128) * (m as i128)).unsigned_abs() as u64;
        let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, hit.a, hit.b);
        let alg_norm = alg_norm_big.to_u64().unwrap_or(u64::MAX);

        if rat_norm == 0 || alg_norm == 0 {
            continue;
        }

        valid += 1;
        sum_rat_log2 += (rat_norm as f64).log2();
        sum_alg_log2 += (alg_norm as f64).log2();
        let energy = (rat_norm.max(1) as f64).ln() + (alg_norm.max(1) as f64).ln();
        sum_energy += energy;

        let rs = is_smooth(rat_norm, &factor_base);
        let als = is_smooth(alg_norm, &factor_base);
        if rs {
            rat_smooth_count += 1;
        }
        if als {
            alg_smooth_count += 1;
        }
        if rs && als {
            both_smooth_count += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    MethodResult {
        method: "lattice_sieve".to_string(),
        candidates_tested: grid_size,
        valid_candidates: valid,
        rational_smooth: rat_smooth_count,
        algebraic_smooth: alg_smooth_count,
        both_smooth: both_smooth_count,
        // Lattice sieve is exhaustive — no duplicates
        unique_valid: valid,
        unique_both_smooth: both_smooth_count,
        mean_rat_norm_log2: if valid > 0 { sum_rat_log2 / valid as f64 } else { 0.0 },
        mean_alg_norm_log2: if valid > 0 { sum_alg_log2 / valid as f64 } else { 0.0 },
        mean_energy: if valid > 0 { sum_energy / valid as f64 } else { 0.0 },
        time_ms: elapsed,
    }
}

/// Run uniform random sampling with accurate BigUint energy.
pub fn sample_uniform_big(
    poly: &NfsPolynomial,
    m: u64,
    params: &SieveParams,
    num_candidates: usize,
    rng: &mut StdRng,
) -> MethodResult {
    let start = Instant::now();
    let factor_base = sieve_primes(params.fb_bound);
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    let mut tested = 0usize;
    let mut valid = 0usize;
    let mut rat_smooth_count = 0usize;
    let mut alg_smooth_count = 0usize;
    let mut both_smooth_count = 0usize;
    let mut unique_valid = 0usize;
    let mut unique_both_smooth = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;
    let mut sum_energy = 0.0f64;

    while tested < num_candidates {
        let a = rng.gen_range(-params.sieve_area..=params.sieve_area);
        let b = rng.gen_range(1..=params.max_b);
        tested += 1;

        if gcd(a.unsigned_abs(), b as u64) > 1 {
            continue;
        }

        let (rat_norm, alg_norm_opt, energy) = compute_energy_big(a, b, m, poly);
        if rat_norm == 0 {
            continue;
        }
        match alg_norm_opt {
            Some(0) | None => continue,
            _ => {}
        }
        let alg_norm = alg_norm_opt.unwrap();

        valid += 1;
        let is_new = seen.insert((a, b));
        if is_new {
            unique_valid += 1;
        }

        sum_rat_log2 += (rat_norm as f64).log2();
        sum_alg_log2 += (alg_norm as f64).log2();
        sum_energy += energy;

        let rs = is_smooth(rat_norm, &factor_base);
        let als = is_smooth(alg_norm, &factor_base);
        if rs {
            rat_smooth_count += 1;
        }
        if als {
            alg_smooth_count += 1;
        }
        if rs && als {
            both_smooth_count += 1;
            if is_new {
                unique_both_smooth += 1;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    MethodResult {
        method: "uniform".to_string(),
        candidates_tested: tested,
        valid_candidates: valid,
        rational_smooth: rat_smooth_count,
        algebraic_smooth: alg_smooth_count,
        both_smooth: both_smooth_count,
        unique_valid,
        unique_both_smooth,
        mean_rat_norm_log2: if valid > 0 { sum_rat_log2 / valid as f64 } else { 0.0 },
        mean_alg_norm_log2: if valid > 0 { sum_alg_log2 / valid as f64 } else { 0.0 },
        mean_energy: if valid > 0 { sum_energy / valid as f64 } else { 0.0 },
        time_ms: elapsed,
    }
}

/// Run MCMC sampling with accurate BigUint energy for large N.
pub fn sample_mcmc_big(
    poly: &NfsPolynomial,
    m: u64,
    params: &SieveParams,
    num_candidates: usize,
    num_chains: usize,
    t_start: f64,
    t_end: f64,
    rng: &mut StdRng,
) -> MethodResult {
    let start = Instant::now();
    let factor_base = sieve_primes(params.fb_bound);
    let steps_per_chain = num_candidates / num_chains.max(1);
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    let mut tested = 0usize;
    let mut valid = 0usize;
    let mut rat_smooth_count = 0usize;
    let mut alg_smooth_count = 0usize;
    let mut both_smooth_count = 0usize;
    let mut unique_valid = 0usize;
    let mut unique_both_smooth = 0usize;
    let mut sum_rat_log2 = 0.0f64;
    let mut sum_alg_log2 = 0.0f64;
    let mut sum_energy = 0.0f64;

    let log_ratio = (t_end / t_start).ln();

    for _chain in 0..num_chains {
        let mut cur_a = rng.gen_range(-params.sieve_area..=params.sieve_area);
        let mut cur_b = rng.gen_range(1..=params.max_b);
        let (mut cur_rat, mut cur_alg_opt, mut cur_energy) =
            compute_energy_big(cur_a, cur_b, m, poly);

        for step in 0..steps_per_chain {
            let frac = step as f64 / (steps_per_chain.max(1) - 1).max(1) as f64;
            let temp = t_start * (frac * log_ratio).exp();
            let beta = 1.0 / temp;

            let (prop_a, prop_b) = propose(cur_a, cur_b, params, rng);
            let (prop_rat, prop_alg_opt, prop_energy) =
                compute_energy_big(prop_a, prop_b, m, poly);
            let delta_e = prop_energy - cur_energy;

            let accept = delta_e <= 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp();

            if accept {
                cur_a = prop_a;
                cur_b = prop_b;
                cur_rat = prop_rat;
                cur_alg_opt = prop_alg_opt;
                cur_energy = prop_energy;
            }

            tested += 1;
            if gcd(cur_a.unsigned_abs(), cur_b as u64) > 1 || cur_rat == 0 {
                continue;
            }

            let cur_alg = match cur_alg_opt {
                Some(0) | None => continue,
                Some(v) => v,
            };

            valid += 1;
            let is_new = seen.insert((cur_a, cur_b));
            if is_new {
                unique_valid += 1;
            }

            sum_rat_log2 += (cur_rat as f64).log2();
            sum_alg_log2 += (cur_alg as f64).log2();
            sum_energy += cur_energy;

            let rs = is_smooth(cur_rat, &factor_base);
            let als = is_smooth(cur_alg, &factor_base);
            if rs {
                rat_smooth_count += 1;
            }
            if als {
                alg_smooth_count += 1;
            }
            if rs && als {
                both_smooth_count += 1;
                if is_new {
                    unique_both_smooth += 1;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    MethodResult {
        method: "mcmc".to_string(),
        candidates_tested: tested,
        valid_candidates: valid,
        rational_smooth: rat_smooth_count,
        algebraic_smooth: alg_smooth_count,
        both_smooth: both_smooth_count,
        unique_valid,
        unique_both_smooth,
        mean_rat_norm_log2: if valid > 0 { sum_rat_log2 / valid as f64 } else { 0.0 },
        mean_alg_norm_log2: if valid > 0 { sum_alg_log2 / valid as f64 } else { 0.0 },
        mean_energy: if valid > 0 { sum_energy / valid as f64 } else { 0.0 },
        time_ms: elapsed,
    }
}

/// Multi-scale random walk proposal.
/// 60% small (|δ| ≤ 4), 30% medium (|δ| ≤ area/8), 10% large (|δ| ≤ area).
fn propose(cur_a: i64, cur_b: i64, params: &SieveParams, rng: &mut StdRng) -> (i64, i64) {
    let r: f64 = rng.gen();
    let max_delta = if r < 0.6 {
        4i64
    } else if r < 0.9 {
        (params.sieve_area / 8).max(4)
    } else {
        params.sieve_area
    };

    // Perturb a or b with equal probability
    if rng.gen::<bool>() {
        let delta = rng.gen_range(-max_delta..=max_delta);
        let new_a = (cur_a + delta).clamp(-params.sieve_area, params.sieve_area);
        (new_a, cur_b)
    } else {
        let b_delta = max_delta.min(params.max_b);
        let delta = rng.gen_range(-b_delta..=b_delta);
        let new_b = (cur_b + delta).clamp(1, params.max_b);
        (cur_a, new_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use rand::SeedableRng;

    fn test_poly() -> NfsPolynomial {
        // Simple polynomial for 15 = 3 * 5, degree 3, m = 2
        // f(x) = x^3 + x^2 + x + 1 (just for testing, not real NFS poly)
        NfsPolynomial {
            coefficients: vec![
                BigUint::from(1u64),
                BigUint::from(1u64),
                BigUint::from(1u64),
                BigUint::from(1u64),
            ],
            m: BigUint::from(2u64),
            degree: 3,
        }
    }

    #[test]
    fn test_compute_energy() {
        let poly = test_poly();
        let (rat, alg, energy) = compute_energy(3, 1, 2, &poly);
        assert_eq!(rat, 1); // |3 - 1*2| = 1
        assert!(alg > 0);
        assert!(energy >= 0.0);
    }

    #[test]
    fn test_sample_uniform_runs() {
        let poly = test_poly();
        let params = SieveParams { sieve_area: 64, max_b: 16, fb_bound: 30, degree: 3 };
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_uniform(&poly, 2, &params, 100, &mut rng);
        assert_eq!(result.candidates_tested, 100);
        assert!(result.valid_candidates > 0);
        assert!(result.unique_valid <= result.valid_candidates);
        assert!(result.unique_both_smooth <= result.both_smooth);
    }

    #[test]
    fn test_sample_mcmc_runs() {
        let poly = test_poly();
        let params = SieveParams { sieve_area: 64, max_b: 16, fb_bound: 30, degree: 3 };
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample_mcmc(&poly, 2, &params, 100, 5, 10.0, 0.1, &mut rng);
        assert!(result.candidates_tested > 0);
        assert!(result.unique_valid <= result.valid_candidates);
        assert!(result.unique_both_smooth <= result.both_smooth);
    }

    #[test]
    fn test_sample_lattice_sieve_runs() {
        let poly = test_poly();
        let params = SieveParams { sieve_area: 64, max_b: 16, fb_bound: 30, degree: 3 };
        let result = sample_lattice_sieve(&poly, 2, &params);
        assert!(result.candidates_tested > 0);
        // Lattice sieve is exhaustive — unique counts equal total counts
        assert_eq!(result.unique_valid, result.valid_candidates);
        assert_eq!(result.unique_both_smooth, result.both_smooth);
    }
}
