//! # TNSS: Tensor Network Schnorr Sieving
//!
//! An implementation of the Tensor Network Schnorr Sieving algorithm for
//! integer factorization, based on the approach described in arXiv:2410.16355.
//!
//! ## Algorithm Overview (v1 - original)
//!
//! 1. **Schnorr Lattice Construction**: Build a lattice where short vectors
//!    correspond to smooth relations over a factor base of small primes.
//! 2. **Binary Encoding**: Encode the lattice exponent search as a binary
//!    optimization problem (QUBO).
//! 3. **MPS Optimization**: Use a Matrix Product State (MPS) tensor network
//!    with DMRG-style sweeps to find low-energy (short vector) solutions.
//! 4. **Relation Collection**: Extract smooth relations from optimized MPS
//!    configurations.
//! 5. **Factor Extraction**: Use Gaussian elimination over GF(2) to combine
//!    relations into a congruence of squares, then compute gcd to find factors.
//!
//! ## Algorithm Overview (v2 - Tesoro/Siloi 2024)
//!
//! 1. **Schnorr Lattice with precision parameter c**: B_{f,c} with
//!    diagonal f(j) and last row = ceil(10^c * ln(p_j)).
//! 2. **LLL Preprocessing**: Basis is LLL-reduced before optimization.
//! 3. **TTN (Tree Tensor Network)**: Binary tree topology, NOT 1D MPS.
//!    Bond dimension m = ceil(n^{2/5}).
//! 4. **OPES Sampling**: Samples from Boltzmann distribution at multiple
//!    temperatures instead of full DMRG optimization.
//! 5. **Parameter scaling**: Factor base ~ ell^1.5, bond dim ~ n^{2/5}.
//! 6. **Factor extraction**: Collect smooth relations, GF(2) Gaussian
//!    elimination, gcd(X+Y, N).

pub mod lattice;
pub mod opes;
pub mod optimizer;
pub mod relations;
pub mod sr_pairs;
pub mod tensor;
pub mod ttn;

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;

use lattice::{
    build_schnorr_lattice, first_n_primes, lattice_vector_to_relation, sieve_primes,
    SmoothRelation,
};
use opes::{beta_schedule_for_dimension, extract_best_configs, opes_sample, OpesConfig};
use optimizer::{binary_to_exponents, build_cost_from_lattice, extract_solution, optimize_sweep};
use relations::{extract_factor, gaussian_elimination_gf2, validate_relation};
use tensor::Mps;
use ttn::{extract_ttn_solution, optimize_ttn_sweep, Ttn};

// ============================================================
// Configuration Types
// ============================================================

/// Configuration for the TNSS factoring algorithm (v1 - original).
#[derive(Debug, Clone)]
pub struct TnssConfig {
    /// Number of primes in the factor base.
    pub factor_base_size: usize,
    /// Scaling constant C for the Schnorr lattice.
    pub scaling: f64,
    /// Number of bits to encode each exponent.
    pub bits_per_exponent: usize,
    /// Exponent range: each e_i in [-exponent_bound, exponent_bound].
    pub exponent_bound: i64,
    /// MPS bond dimension (higher = more expressive but slower).
    pub bond_dim: usize,
    /// Number of DMRG sweeps per relation search.
    pub num_sweeps: usize,
    /// Number of relation-finding attempts.
    pub num_attempts: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for TnssConfig {
    fn default() -> Self {
        Self {
            factor_base_size: 10,
            scaling: 100.0,
            bits_per_exponent: 4,
            exponent_bound: 7,
            bond_dim: 8,
            num_sweeps: 20,
            num_attempts: 50,
            seed: 42,
        }
    }
}

/// Configuration for the TNSS v2 algorithm (Tesoro/Siloi 2024 paper).
#[derive(Debug, Clone)]
pub struct TnssConfigV2 {
    /// Number of primes in the factor base.
    pub factor_base_size: usize,
    /// Precision parameter c for the Schnorr lattice.
    /// The scaling is 10^c.
    pub precision_c: f64,
    /// Number of bits to encode each exponent.
    pub bits_per_exponent: usize,
    /// Exponent range: each e_i in [-exponent_bound, exponent_bound].
    pub exponent_bound: i64,
    /// TTN bond dimension (paper: ceil(n^{2/5})).
    pub bond_dim: usize,
    /// Temperature schedule for OPES sampling (inverse temperatures).
    pub temperature_schedule: Vec<f64>,
    /// Number of samples per temperature in OPES.
    pub samples_per_temp: usize,
    /// Number of TTN optimization sweeps per OPES temperature step.
    pub sweeps_per_temp: usize,
    /// Number of different precision c values to try (CVP instances).
    pub num_cvp_instances: usize,
    /// Range of precision c values to try: [c_min, c_max].
    pub c_range: (f64, f64),
    /// Number of relation-finding attempts per CVP instance.
    pub num_attempts_per_instance: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for TnssConfigV2 {
    fn default() -> Self {
        Self {
            factor_base_size: 10,
            precision_c: 2.0,
            bits_per_exponent: 4,
            exponent_bound: 7,
            bond_dim: 4,
            temperature_schedule: opes::default_beta_schedule(8),
            samples_per_temp: 30,
            sweeps_per_temp: 2,
            num_cvp_instances: 5,
            c_range: (1.0, 4.0),
            num_attempts_per_instance: 20,
            seed: 42,
        }
    }
}

/// Result of the TNSS factoring algorithm.
#[derive(Debug, Clone)]
pub struct TnssResult {
    /// The number that was factored.
    pub n: BigUint,
    /// Found factor (non-trivial), or None if factoring failed.
    pub factor: Option<BigUint>,
    /// Number of smooth relations found.
    pub relations_found: usize,
    /// Total relation-finding attempts made.
    pub attempts_made: usize,
}

// ============================================================
// V1 API (backward compatible)
// ============================================================

/// Run the full TNSS factoring pipeline on the given composite number (v1).
///
/// Returns a `TnssResult` containing the found factor (if any).
pub fn factor_tnss(n: &BigUint, config: &TnssConfig) -> TnssResult {
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Step 1: Build factor base
    let primes = sieve_primes(200);
    let factor_base: Vec<u64> = primes
        .into_iter()
        .take(config.factor_base_size)
        .collect();

    // Quick trial division check
    for &p in &factor_base {
        let p_big = BigUint::from(p);
        if n > &p_big && n.is_multiple_of(&p_big) {
            return TnssResult {
                n: n.clone(),
                factor: Some(p_big),
                relations_found: 0,
                attempts_made: 0,
            };
        }
    }

    // Step 2: Build Schnorr lattice
    let lattice = build_schnorr_lattice(n, &factor_base, config.scaling);
    let num_vars = config.factor_base_size * config.bits_per_exponent;

    // Step 3: Collect smooth relations via MPS optimization
    let mut relations: Vec<SmoothRelation> = Vec::new();
    let mut attempts = 0;

    // We need at least factor_base_size + 1 relations for GF(2) elimination
    let target_relations = config.factor_base_size + 1;

    // The target vector for CVP: we want Lx ~ 0 (short vectors)
    let dim = lattice.len();
    let target = vec![0i64; dim];

    let hamiltonian = build_cost_from_lattice(
        &lattice,
        &target,
        config.factor_base_size,
        config.bits_per_exponent,
        config.exponent_bound,
    );

    for _ in 0..config.num_attempts {
        if relations.len() >= target_relations {
            break;
        }
        attempts += 1;

        // Create a new random MPS for each attempt
        let mut mps = Mps::new_random(num_vars, config.bond_dim, &mut rng);

        // Optimize with DMRG sweeps
        optimize_sweep(&mut mps, &hamiltonian, config.num_sweeps);

        // Extract the best configuration
        let exponents = extract_solution(
            &mps,
            config.factor_base_size,
            config.bits_per_exponent,
            config.exponent_bound,
        );

        // Check if this gives a smooth relation
        if let Some(rel) = lattice_vector_to_relation(&exponents, &factor_base, n) {
            if validate_relation(&rel, n) {
                relations.push(rel);
            }
        }
    }

    // Step 4: Gaussian elimination over GF(2) to find factor
    let mut factor = None;
    if relations.len() >= 2 {
        let null_vecs = gaussian_elimination_gf2(&relations, config.factor_base_size);
        for null_vec in &null_vecs {
            if let Some(f) = extract_factor(null_vec, &relations, n) {
                if f > BigUint::one() && &f < n {
                    factor = Some(f);
                    break;
                }
            }
        }
    }

    TnssResult {
        n: n.clone(),
        factor,
        relations_found: relations.len(),
        attempts_made: attempts,
    }
}

/// Run the TTN factoring pipeline (v1).
///
/// Uses a Tree Tensor Network instead of MPS for the variational optimization.
pub fn factor_ttn(n: &BigUint, config: &TnssConfig) -> TnssResult {
    let mut rng = StdRng::seed_from_u64(config.seed);

    let primes = sieve_primes(200);
    let factor_base: Vec<u64> = primes.into_iter().take(config.factor_base_size).collect();

    // Quick trial division
    for &p in &factor_base {
        let p_big = BigUint::from(p);
        if n > &p_big && n.is_multiple_of(&p_big) {
            return TnssResult {
                n: n.clone(),
                factor: Some(p_big),
                relations_found: 0,
                attempts_made: 0,
            };
        }
    }

    let lattice = build_schnorr_lattice(n, &factor_base, config.scaling);
    let num_vars = config.factor_base_size * config.bits_per_exponent;
    let dim = lattice.len();
    let target = vec![0i64; dim];

    let hamiltonian = build_cost_from_lattice(
        &lattice,
        &target,
        config.factor_base_size,
        config.bits_per_exponent,
        config.exponent_bound,
    );

    let mut relations: Vec<SmoothRelation> = Vec::new();
    let target_relations = config.factor_base_size + 1;
    let mut attempts = 0;

    for _ in 0..config.num_attempts {
        if relations.len() >= target_relations {
            break;
        }
        attempts += 1;

        let mut ttn_state = Ttn::new_random(num_vars, config.bond_dim, &mut rng);
        optimize_ttn_sweep(&mut ttn_state, &hamiltonian, config.num_sweeps);

        let exponents = extract_ttn_solution(
            &ttn_state,
            config.factor_base_size,
            config.bits_per_exponent,
            config.exponent_bound,
        );

        if let Some(rel) = lattice_vector_to_relation(&exponents, &factor_base, n) {
            if validate_relation(&rel, n) {
                relations.push(rel);
            }
        }
    }

    let mut factor = None;
    if relations.len() >= 2 {
        let null_vecs = gaussian_elimination_gf2(&relations, config.factor_base_size);
        for null_vec in &null_vecs {
            if let Some(f) = extract_factor(null_vec, &relations, n) {
                if f > BigUint::one() && &f < n {
                    factor = Some(f);
                    break;
                }
            }
        }
    }

    TnssResult {
        n: n.clone(),
        factor,
        relations_found: relations.len(),
        attempts_made: attempts,
    }
}

// ============================================================
// V2 API (Tesoro/Siloi 2024 paper approach)
// ============================================================

/// Run the TNSS v2 factoring pipeline (Tesoro/Siloi 2024 paper approach).
///
/// Key differences from v1:
/// 1. Uses LLL-reduced Schnorr lattice with precision parameter c
/// 2. Uses TTN instead of MPS
/// 3. Uses OPES sampling instead of pure DMRG optimization
/// 4. Tries multiple precision c values (CVP instances)
/// 5. Uses paper-based parameter scaling
/// 6. Only accepts exact modular relations or smooth-residue relations
pub fn factor_tnss_v2(n: &BigUint, config: &TnssConfigV2) -> TnssResult {
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Step 1: Build factor base
    let factor_base = first_n_primes(config.factor_base_size);

    // Quick trial division check
    for &p in &factor_base {
        let p_big = BigUint::from(p);
        if n > &p_big && n.is_multiple_of(&p_big) {
            return TnssResult {
                n: n.clone(),
                factor: Some(p_big),
                relations_found: 0,
                attempts_made: 0,
            };
        }
    }

    let mut all_relations: Vec<SmoothRelation> = Vec::new();
    let target_relations = config.factor_base_size + 1;
    let mut total_attempts = 0;
    let mut exact_count: usize = 0;
    let mut residue_count: usize = 0;
    let mut candidates_tested: usize = 0;

    // Step 2: Try different precision c values (CVP instances)
    let c_step = if config.num_cvp_instances > 1 {
        (config.c_range.1 - config.c_range.0) / (config.num_cvp_instances - 1) as f64
    } else {
        0.0
    };

    for cvp_idx in 0..config.num_cvp_instances {
        if all_relations.len() >= target_relations {
            break;
        }

        let c = config.c_range.0 + c_step * cvp_idx as f64;

        // Step 2a: Build and LLL-reduce the lattice
        let reduced_basis = lattice::build_reduced_lattice_v2(n, &factor_base, c);

        // Step 2b: Also try to extract relations directly from the short lattice vectors
        let short_vectors = lattice::extract_short_vectors(
            &reduced_basis,
            config.factor_base_size,
            config.factor_base_size + 2,
        );

        for sv in &short_vectors {
            candidates_tested += 1;
            if let Some(rel) = lattice_vector_to_relation(sv, &factor_base, n) {
                if validate_relation(&rel, n) {
                    classify_relation(&rel, n, &mut exact_count, &mut residue_count);
                    all_relations.push(rel);
                }
            }
        }

        // Step 2c: Build QUBO Hamiltonian from the reduced lattice
        // Convert reduced basis to integer lattice for Hamiltonian construction
        let int_lattice: Vec<Vec<i64>> = reduced_basis
            .iter()
            .map(|row| row.iter().map(|&v| v.round() as i64).collect())
            .collect();
        let dim = int_lattice.len();
        let target = vec![0i64; dim];

        let hamiltonian = build_cost_from_lattice(
            &int_lattice,
            &target,
            config.factor_base_size,
            config.bits_per_exponent,
            config.exponent_bound,
        );

        let num_vars = config.factor_base_size * config.bits_per_exponent;

        // Step 2d: For each attempt, create TTN + OPES sample
        for _ in 0..config.num_attempts_per_instance {
            if all_relations.len() >= target_relations {
                break;
            }
            total_attempts += 1;

            let mut ttn_state = Ttn::new_random(num_vars, config.bond_dim, &mut rng);

            // OPES sampling
            let opes_config = OpesConfig {
                beta_schedule: config.temperature_schedule.clone(),
                samples_per_temp: config.samples_per_temp,
                sweeps_per_temp: config.sweeps_per_temp,
            };

            let samples = opes_sample(&ttn_state, &hamiltonian, &opes_config, &mut rng);
            let best_configs = extract_best_configs(&samples, 10);

            // Extract relations from the best configurations
            for (binary_config, _energy) in &best_configs {
                let exponents = binary_to_exponents(
                    binary_config,
                    config.factor_base_size,
                    config.bits_per_exponent,
                    config.exponent_bound,
                );

                candidates_tested += 1;
                if let Some(rel) = lattice_vector_to_relation(&exponents, &factor_base, n) {
                    if validate_relation(&rel, n) {
                        classify_relation(&rel, n, &mut exact_count, &mut residue_count);
                        all_relations.push(rel);
                    }
                }
            }

            // Also try the TTN's own max-config (if small enough)
            if num_vars <= 20 {
                // Do a few optimization sweeps
                optimize_ttn_sweep(&mut ttn_state, &hamiltonian, config.sweeps_per_temp);

                let exponents = extract_ttn_solution(
                    &ttn_state,
                    config.factor_base_size,
                    config.bits_per_exponent,
                    config.exponent_bound,
                );

                candidates_tested += 1;
                if let Some(rel) = lattice_vector_to_relation(&exponents, &factor_base, n) {
                    if validate_relation(&rel, n) {
                        classify_relation(&rel, n, &mut exact_count, &mut residue_count);
                        all_relations.push(rel);
                    }
                }
            }
        }
    }

    eprintln!(
        "TNSS v2 diagnostics: candidates_tested={}, exact_relations={}, residue_factored={}, total_valid={}",
        candidates_tested, exact_count, residue_count, all_relations.len()
    );

    // Step 3: Gaussian elimination over GF(2) to find factor
    let mut factor = None;
    if all_relations.len() >= 2 {
        let null_vecs = gaussian_elimination_gf2(&all_relations, config.factor_base_size);
        eprintln!(
            "TNSS v2: GF(2) elimination found {} null vectors from {} relations",
            null_vecs.len(),
            all_relations.len()
        );
        for null_vec in &null_vecs {
            if let Some(f) = extract_factor(null_vec, &all_relations, n) {
                if f > BigUint::one() && &f < n {
                    factor = Some(f);
                    break;
                }
            }
        }
    }

    TnssResult {
        n: n.clone(),
        factor,
        relations_found: all_relations.len(),
        attempts_made: total_attempts,
    }
}

// ============================================================
// V3 API (Paper-correct SR-pair pipeline)
// ============================================================

/// Configuration for the TNSS v3 pipeline (SR-pair based).
#[derive(Debug, Clone)]
pub struct TnssConfigV3 {
    /// Number of precision c values to try (each gives a different lattice geometry).
    pub num_c_values: usize,
    /// Range of precision c values.
    pub c_range: (f64, f64),
    /// Number of random f(j) permutation seeds to try per c value.
    pub num_seeds_per_c: usize,
    /// Perturbation radius: try ±k on each component of short vectors.
    pub perturbation_radius: i64,
    /// Maximum number of candidate vectors to test per (c, seed) pair.
    pub max_candidates_per_instance: usize,
    /// Base random seed.
    pub seed: u64,
}

impl Default for TnssConfigV3 {
    fn default() -> Self {
        Self {
            num_c_values: 10,
            c_range: (1.0, 5.0),
            num_seeds_per_c: 5,
            perturbation_radius: 2,
            max_candidates_per_instance: 500,
            seed: 42,
        }
    }
}

/// Choose v3 configuration based on bit size.
pub fn config_for_bits_v3(bits: usize) -> TnssConfigV3 {
    if bits <= 20 {
        TnssConfigV3 {
            num_c_values: 8,
            c_range: (1.0, 4.0),
            num_seeds_per_c: 4,
            perturbation_radius: 2,
            max_candidates_per_instance: 300,
            seed: 42,
        }
    } else if bits <= 40 {
        TnssConfigV3 {
            num_c_values: 20,
            c_range: (1.0, 6.0),
            num_seeds_per_c: 12,
            perturbation_radius: 3,
            max_candidates_per_instance: 3000,
            seed: 42,
        }
    } else if bits <= 64 {
        TnssConfigV3 {
            num_c_values: 30,
            c_range: (1.0, 8.0),
            num_seeds_per_c: 15,
            perturbation_radius: 4,
            max_candidates_per_instance: 5000,
            seed: 42,
        }
    } else {
        TnssConfigV3 {
            num_c_values: 40,
            c_range: (1.0, 10.0),
            num_seeds_per_c: 20,
            perturbation_radius: 5,
            max_candidates_per_instance: 10000,
            seed: 42,
        }
    }
}

/// Run the TNSS v3 factoring pipeline using the paper's exact SR-pair algorithm.
///
/// Key differences from v2:
/// 1. Uses TWO factor bases: P₁ (lattice, small) and P₂ (smoothness, large)
/// 2. Uses proper SR-pair validation: BOTH u AND |u-vN| must be B₂-smooth
/// 3. Uses LLL + diverse lattice perturbation instead of TTN sampling
/// 4. Uses GF(2) elimination on SR-pair exponent vectors over P₂
pub fn factor_tnss_v3(n: &BigUint, config: &TnssConfigV3) -> TnssResult {
    let bits = n.bits() as usize;

    // Step 1: Build factor base configuration
    let sr_config = sr_pairs::SrPairConfig::for_bits(bits);
    let p1 = &sr_config.p1_primes;
    let pi1 = p1.len();
    let pi2 = sr_config.p2_primes.len();

    // Quick trial division over P₂ (larger base = more chances)
    for &p in sr_config.p2_primes.iter().take(200) {
        let p_big = BigUint::from(p);
        if n > &p_big && n.is_multiple_of(&p_big) {
            return TnssResult {
                n: n.clone(),
                factor: Some(p_big),
                relations_found: 0,
                attempts_made: 0,
            };
        }
    }

    // The GF(2) matrix has 2*pi2 columns (u-exponents and rem-exponents separately).
    // We need > 2*pi2 SR-pairs, with extra margin for non-trivial null vectors.
    // More surplus = more congruences = higher chance of non-trivial gcd.
    let target_sr_pairs = (2.5 * pi2 as f64).ceil() as usize + 10;
    let mut all_sr_pairs: Vec<sr_pairs::SrPair> = Vec::new();
    let mut total_candidates = 0usize;
    let mut total_smooth = 0usize;

    // Step 2: Try multiple (c, seed) combinations to get diverse vectors
    let c_step = if config.num_c_values > 1 {
        (config.c_range.1 - config.c_range.0) / (config.num_c_values - 1) as f64
    } else {
        0.0
    };

    'outer: for c_idx in 0..config.num_c_values {
        if all_sr_pairs.len() >= target_sr_pairs {
            break;
        }
        let c = config.c_range.0 + c_step * c_idx as f64;

        for seed_idx in 0..config.num_seeds_per_c {
            if all_sr_pairs.len() >= target_sr_pairs {
                break 'outer;
            }
            let seed = config.seed.wrapping_add(seed_idx as u64 * 1000 + c_idx as u64);

            // Build Schnorr lattice B_{f,c} with this seed's f(j) permutation
            let (basis, target) = sr_pairs::build_schnorr_lattice(n, p1, c, seed);

            // LLL reduce the basis
            let mut lll_basis = basis.clone();
            lattice::lll_reduce(&mut lll_basis, &lattice::LllParams { delta: 0.99 });

            // Extract candidate exponent vectors
            let mut candidates: Vec<Vec<i64>> = Vec::new();

            // a) Short vectors directly from LLL basis rows
            for row in &lll_basis {
                let exps: Vec<i64> = row.iter().take(pi1).map(|&v| v.round() as i64).collect();
                if exps.iter().any(|&e| e != 0) {
                    candidates.push(exps);
                }
            }

            // b) Babai nearest plane CVP
            let babai_coeffs = sr_pairs::babai_nearest_plane(&lll_basis, &target);
            // Reconstruct the exponent vector from Babai coefficients
            let mut babai_vec = vec![0.0f64; lll_basis.get(0).map_or(0, |r| r.len())];
            for (i, &coeff) in babai_coeffs.iter().enumerate() {
                if i < lll_basis.len() {
                    for (k, &val) in lll_basis[i].iter().enumerate() {
                        babai_vec[k] += coeff * val;
                    }
                }
            }
            let babai_exps: Vec<i64> = babai_vec.iter().take(pi1).map(|&v| v.round() as i64).collect();
            if babai_exps.iter().any(|&e| e != 0) {
                candidates.push(babai_exps);
            }

            // c) Perturbations of each candidate
            let base_candidates = candidates.clone();
            for base in &base_candidates {
                if candidates.len() >= config.max_candidates_per_instance {
                    break;
                }
                for dim in 0..pi1 {
                    for delta in 1..=config.perturbation_radius {
                        let mut plus = base.clone();
                        plus[dim] += delta;
                        candidates.push(plus);

                        let mut minus = base.clone();
                        minus[dim] -= delta;
                        candidates.push(minus);
                    }
                }
            }

            // d) Pairwise sums/differences of basis short vectors
            let num_basis = lll_basis.len().min(pi1);
            for i in 0..num_basis {
                for j in (i + 1)..num_basis {
                    if candidates.len() >= config.max_candidates_per_instance {
                        break;
                    }
                    let sum_vec: Vec<i64> = (0..pi1)
                        .map(|k| {
                            let a = lll_basis[i].get(k).copied().unwrap_or(0.0).round() as i64;
                            let b = lll_basis[j].get(k).copied().unwrap_or(0.0).round() as i64;
                            a + b
                        })
                        .collect();
                    let diff_vec: Vec<i64> = (0..pi1)
                        .map(|k| {
                            let a = lll_basis[i].get(k).copied().unwrap_or(0.0).round() as i64;
                            let b = lll_basis[j].get(k).copied().unwrap_or(0.0).round() as i64;
                            a - b
                        })
                        .collect();
                    if sum_vec.iter().any(|&e| e != 0) {
                        candidates.push(sum_vec);
                    }
                    if diff_vec.iter().any(|&e| e != 0) {
                        candidates.push(diff_vec);
                    }
                }
            }

            // Step 3: Test each candidate as an SR-pair
            for cand in &candidates {
                if all_sr_pairs.len() >= target_sr_pairs {
                    break;
                }
                total_candidates += 1;
                if let Some(pair) = sr_pairs::construct_sr_pair(cand, &sr_config, n) {
                    total_smooth += 1;
                    all_sr_pairs.push(pair);
                }
            }
        }
    }

    eprintln!(
        "TNSS v3: tested {} candidates, found {} sr-pairs (target: {}, P1={}, P2={})",
        total_candidates,
        all_sr_pairs.len(),
        target_sr_pairs,
        pi1,
        pi2
    );

    // Step 4: GF(2) Gaussian elimination on SR-pair exponent vectors
    let mut factor = None;
    if all_sr_pairs.len() >= 2 {
        let congruences = sr_pairs::find_congruences(&all_sr_pairs, pi2);
        eprintln!(
            "TNSS v3: found {} congruences from {} sr-pairs",
            congruences.len(),
            all_sr_pairs.len()
        );

        for cong in &congruences {
            if let Some(f) = sr_pairs::extract_factors_from_congruence(
                cong,
                &all_sr_pairs,
                n,
                &sr_config.p2_primes,
            ) {
                if f > BigUint::one() && &f < n {
                    factor = Some(f);
                    break;
                }
            }
        }
        if factor.is_none() {
            eprintln!(
                "TNSS v3: tried all {} congruences, no factor found",
                congruences.len()
            );
        }
    }

    TnssResult {
        n: n.clone(),
        factor,
        relations_found: all_sr_pairs.len(),
        attempts_made: total_candidates,
    }
}

/// Convenience function to factor a u64 number using v3 pipeline.
pub fn factor_u64_v3(n: u64) -> TnssResult {
    let n_big = BigUint::from(n);
    let bits = n_big
        .to_f64()
        .map(|f| f.log2().ceil() as usize)
        .unwrap_or(64);
    let config = config_for_bits_v3(bits);
    factor_tnss_v3(&n_big, &config)
}

/// Classify a relation as exact or residue-factored for diagnostic tracking.
fn classify_relation(
    rel: &SmoothRelation,
    n: &BigUint,
    exact_count: &mut usize,
    residue_count: &mut usize,
) {
    let lhs_mod = &rel.lhs % n;
    let rhs_mod = &rel.rhs % n;
    if lhs_mod == rhs_mod || lhs_mod.is_zero() || rhs_mod.is_zero() {
        *exact_count += 1;
    } else {
        // Must be a residue-factored or sign-negated relation
        *residue_count += 1;
    }
}

// ============================================================
// Configuration helpers
// ============================================================

/// Convenience function to factor a u64 number (uses v1).
pub fn factor_u64(n: u64) -> TnssResult {
    let n_big = BigUint::from(n);
    let config = config_for_bits(
        n_big
            .to_f64()
            .map(|f| f.log2().ceil() as usize)
            .unwrap_or(64),
    );
    factor_tnss(&n_big, &config)
}

/// Convenience function to factor a u64 number using v2 pipeline.
pub fn factor_u64_v2(n: u64) -> TnssResult {
    let n_big = BigUint::from(n);
    let bits = n_big
        .to_f64()
        .map(|f| f.log2().ceil() as usize)
        .unwrap_or(64);
    let config = config_for_bits_v2(bits);
    factor_tnss_v2(&n_big, &config)
}

/// Choose a reasonable configuration based on the bit size of n (v1).
pub fn config_for_bits(bits: usize) -> TnssConfig {
    if bits <= 32 {
        TnssConfig {
            factor_base_size: 8,
            scaling: 50.0,
            bits_per_exponent: 3,
            exponent_bound: 3,
            bond_dim: 4,
            num_sweeps: 15,
            num_attempts: 100,
            seed: 42,
        }
    } else if bits <= 48 {
        TnssConfig {
            factor_base_size: 12,
            scaling: 100.0,
            bits_per_exponent: 4,
            exponent_bound: 7,
            bond_dim: 8,
            num_sweeps: 20,
            num_attempts: 200,
            seed: 42,
        }
    } else {
        TnssConfig {
            factor_base_size: 20,
            scaling: 200.0,
            bits_per_exponent: 5,
            exponent_bound: 15,
            bond_dim: 16,
            num_sweeps: 30,
            num_attempts: 500,
            seed: 42,
        }
    }
}

/// Choose configuration based on bit size using the paper's scaling (v2).
///
/// Paper-based parameter scaling:
/// - Factor base size pi ~ ell^1.5 (where ell = bit length of semiprime)
/// - Lattice rank n = pi + 1
/// - Bond dimension m = ceil(n^{2/5})
/// - Precision c varies across CVP instances
pub fn config_for_bits_v2(bits: usize) -> TnssConfigV2 {
    let ell = bits as f64;

    // Factor base size: pi ~ ell^1.5, but capped reasonably
    let factor_base_size = (ell.powf(1.5)).ceil() as usize;
    let factor_base_size = factor_base_size.max(6).min(200);

    // Lattice rank
    let n = factor_base_size + 1;

    // Bond dimension: m = ceil(n^{2/5})
    let bond_dim = ttn::compute_paper_bond_dim(n);

    // Bits per exponent: enough to cover the expected range
    let bits_per_exponent = if bits <= 20 { 3 } else if bits <= 48 { 4 } else { 5 };
    let exponent_bound = (1i64 << (bits_per_exponent - 1)) - 1;

    // Temperature schedule based on lattice dimension
    let temperature_schedule = beta_schedule_for_dimension(n);

    // Precision c range: wider for larger numbers
    let c_min = 1.0;
    let c_max = if bits <= 20 { 3.0 } else if bits <= 48 { 4.0 } else { 6.0 };

    // Number of CVP instances: more for larger numbers
    let num_cvp_instances = if bits <= 20 { 3 } else if bits <= 48 { 5 } else { 8 };

    // Samples and attempts scale with difficulty
    let samples_per_temp = if bits <= 20 { 20 } else if bits <= 48 { 50 } else { 100 };
    let sweeps_per_temp = if bits <= 20 { 2 } else { 3 };
    let num_attempts_per_instance = if bits <= 20 { 10 } else if bits <= 48 { 20 } else { 50 };

    TnssConfigV2 {
        factor_base_size,
        precision_c: (c_min + c_max) / 2.0,
        bits_per_exponent,
        exponent_bound,
        bond_dim,
        temperature_schedule,
        samples_per_temp,
        sweeps_per_temp,
        num_cvp_instances,
        c_range: (c_min, c_max),
        num_attempts_per_instance,
        seed: 42,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // V1 Tests (backward compatibility)
    // ============================================================

    #[test]
    fn test_factor_small_via_trial_division() {
        // These should all succeed via trial division
        for &n in &[15u64, 21, 35, 77, 143, 323] {
            let result = factor_u64(n);
            assert!(
                result.factor.is_some(),
                "Should factor {} via trial division",
                n
            );
        }
    }

    #[test]
    fn test_config_for_bits() {
        let c16 = config_for_bits(16);
        assert_eq!(c16.factor_base_size, 8);

        let c48 = config_for_bits(48);
        assert_eq!(c48.factor_base_size, 12);

        let c64 = config_for_bits(64);
        assert_eq!(c64.factor_base_size, 20);
    }

    // ============================================================
    // V2 Tests
    // ============================================================

    #[test]
    fn test_config_for_bits_v2_scaling() {
        let c16 = config_for_bits_v2(16);
        assert!(c16.factor_base_size >= 6);
        assert!(c16.bond_dim >= 2);
        assert!(!c16.temperature_schedule.is_empty());
        assert!(c16.num_cvp_instances >= 1);

        let c32 = config_for_bits_v2(32);
        assert!(c32.factor_base_size > c16.factor_base_size);

        let c64 = config_for_bits_v2(64);
        assert!(c64.factor_base_size > c32.factor_base_size);
    }

    #[test]
    fn test_config_for_bits_v2_bond_dim() {
        // Bond dim should follow n^{2/5} scaling
        let c16 = config_for_bits_v2(16);
        let expected_n = c16.factor_base_size + 1;
        let expected_bond = ttn::compute_paper_bond_dim(expected_n);
        assert_eq!(c16.bond_dim, expected_bond);
    }

    #[test]
    fn test_factor_tnss_v2_small_trial_division() {
        // Should succeed via trial division
        let n = BigUint::from(15u64);
        let config = config_for_bits_v2(4);
        let result = factor_tnss_v2(&n, &config);
        assert!(result.factor.is_some(), "Should factor 15");
        let f = result.factor.unwrap();
        assert!(
            f == BigUint::from(3u64) || f == BigUint::from(5u64),
            "Factor of 15 should be 3 or 5, got {}",
            f
        );
    }

    #[test]
    fn test_factor_tnss_v2_77() {
        let n = BigUint::from(77u64);
        let config = config_for_bits_v2(7);
        let result = factor_tnss_v2(&n, &config);
        assert!(result.factor.is_some(), "Should factor 77 (7 x 11)");
    }

    #[test]
    fn test_factor_tnss_v2_143() {
        let n = BigUint::from(143u64);
        let config = config_for_bits_v2(8);
        let result = factor_tnss_v2(&n, &config);
        assert!(result.factor.is_some(), "Should factor 143 (11 x 13)");
    }

    #[test]
    fn test_factor_tnss_v2_323() {
        let n = BigUint::from(323u64);
        let config = config_for_bits_v2(9);
        let result = factor_tnss_v2(&n, &config);
        assert!(result.factor.is_some(), "Should factor 323 (17 x 19)");
    }

    #[test]
    fn test_factor_u64_v2_small() {
        let result = factor_u64_v2(15);
        assert!(result.factor.is_some(), "Should factor 15 via v2");
    }

    #[test]
    fn test_factor_u64_v2_medium() {
        let result = factor_u64_v2(221); // 13 * 17
        assert!(result.factor.is_some(), "Should factor 221 via v2");
    }

    // ============================================================
    // Beyond-trial-division tests: factors NOT in factor base
    // These test the actual TNSS algorithm, not just trial division.
    // ============================================================

    #[test]
    fn test_factor_v2_beyond_trial_16bit() {
        // 16-bit: 631 * 641 = 404471 (both primes > factor base)
        let n = BigUint::from(404471u64);
        let config = config_for_bits_v2(19);
        let result = factor_tnss_v2(&n, &config);
        eprintln!(
            "TNSS v2 16-bit: relations={}, attempts={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
        // Don't assert success — we're measuring if the algorithm works at all
    }

    #[test]
    fn test_factor_v2_beyond_trial_20bit() {
        // 20-bit: 1009 * 1013 = 1022117
        let n = BigUint::from(1022117u64);
        let config = config_for_bits_v2(20);
        let result = factor_tnss_v2(&n, &config);
        eprintln!(
            "TNSS v2 20-bit: relations={}, attempts={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }

    // ============================================================
    // V3 Tests (SR-pair pipeline)
    // ============================================================

    #[test]
    fn test_config_for_bits_v3() {
        let c20 = config_for_bits_v3(20);
        assert!(c20.num_c_values >= 5);
        let c40 = config_for_bits_v3(40);
        assert!(c40.num_c_values > c20.num_c_values);
    }

    #[test]
    fn test_factor_v3_small_trial() {
        // Should succeed via trial division (factors in P₂)
        for &n in &[15u64, 21, 35, 77, 143, 323] {
            let result = factor_u64_v3(n);
            assert!(
                result.factor.is_some(),
                "v3 should factor {} via trial division",
                n
            );
        }
    }

    #[test]
    fn test_factor_v3_77_sr_pairs() {
        // 77 = 7 * 11 — factors ARE in P₂ (trial division),
        // but test the pipeline path too
        let n = BigUint::from(77u64);
        let config = config_for_bits_v3(7);
        let result = factor_tnss_v3(&n, &config);
        assert!(result.factor.is_some(), "v3 should factor 77");
    }

    #[test]
    fn test_factor_v3_beyond_trial_19bit() {
        // 404471 = 631 * 641 (both factors > 200, so NOT caught by trial division)
        let n = BigUint::from(404471u64);
        let config = config_for_bits_v3(19);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 404471: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
        // Diagnostic — don't hard-assert until algorithm is validated
    }

    #[test]
    fn test_factor_v3_beyond_trial_20bit() {
        // 1022117 = 1009 * 1013
        let n = BigUint::from(1022117u64);
        let config = config_for_bits_v3(20);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 1022117: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }

    #[test]
    #[ignore] // Uses significant memory for GF(2) elimination
    fn test_factor_v3_real_sr_pair_pipeline() {
        // 4088459 = 2017 * 2027
        // 2017 is the 306th prime — beyond trial division (first 200)
        // but within P₂ (~704 primes for 22-bit), so smoothness check works
        let n = BigUint::from(4088459u64);
        let config = config_for_bits_v3(22);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 4088459=2017*2027: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }

    #[test]
    #[ignore] // Heavy test: ~3.5s + memory, run separately with --ignored
    fn test_factor_v3_32bit() {
        // 2502200483 = 50021 * 50023 (32-bit semiprime)
        // Both factors far beyond P₂ — tests true SR-pair smooth relation pipeline
        let n = BigUint::from(2502200483u64);
        let config = config_for_bits_v3(32);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 32-bit 2502200483=50021*50023: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }

    #[test]
    #[ignore] // Heavy test: run separately with --ignored
    fn test_factor_v3_36bit() {
        // 17188783211 = 131101 * 131111 (35-bit)
        let n = BigUint::from(17188783211u64);
        let config = config_for_bits_v3(35);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 36-bit: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }

    #[test]
    #[ignore] // Heavy test: 23s, run separately with --ignored
    fn test_factor_v3_40bit() {
        // 274916705369 = 524309 * 524341 (39-bit)
        let n = BigUint::from(274916705369u64);
        let config = config_for_bits_v3(39);
        let result = factor_tnss_v3(&n, &config);
        eprintln!(
            "TNSS v3 40-bit: sr_pairs={}, candidates={}, factor={:?}",
            result.relations_found, result.attempts_made, result.factor
        );
    }
}
