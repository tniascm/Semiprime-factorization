//! Integration tests for the TNSS factoring crate.

use num_bigint::BigUint;
use num_traits::One;
use rand::rngs::StdRng;
use rand::SeedableRng;

use tnss_factoring::lattice::{
    build_reduced_lattice_v2, build_schnorr_lattice, extract_short_vectors, first_n_primes,
    gram_schmidt, sieve_primes, vector_norm,
};
use tnss_factoring::opes::{
    default_beta_schedule, extract_best_configs, opes_sample, OpesConfig,
};
use tnss_factoring::optimizer::{
    binary_to_exponents, build_cost_from_lattice, exponents_to_binary, optimize_sweep,
    CostHamiltonian, QuadraticTerm,
};
use tnss_factoring::relations::gaussian_elimination_gf2;
use tnss_factoring::tensor::Mps;
use tnss_factoring::ttn::{compute_paper_bond_dim, Ttn};
use tnss_factoring::{
    config_for_bits_v2, factor_tnss, factor_tnss_v2, factor_u64, factor_u64_v2, TnssConfig,
    TnssConfigV2,
};

// ============================================================
// Lattice Construction Tests
// ============================================================

#[test]
fn test_sieve_primes_correctness() {
    let primes = sieve_primes(100);
    let expected = vec![
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97,
    ];
    assert_eq!(primes, expected);
}

#[test]
fn test_first_n_primes_correctness() {
    let primes = first_n_primes(25);
    assert_eq!(primes.len(), 25);
    assert_eq!(primes[0], 2);
    assert_eq!(primes[24], 97);
}

#[test]
fn test_lattice_structure() {
    let n = BigUint::from(143u64); // 11 * 13
    let factor_base = vec![2, 3, 5, 7, 11];
    let lattice = build_schnorr_lattice(&n, &factor_base, 100.0);

    // Dimension check
    assert_eq!(lattice.len(), 6);
    for row in &lattice {
        assert_eq!(row.len(), 6);
    }

    // Identity block
    for i in 0..5 {
        for j in 0..5 {
            if i == j {
                assert_eq!(lattice[i][j], 1, "Diagonal should be 1 at ({}, {})", i, j);
            } else {
                assert_eq!(
                    lattice[i][j], 0,
                    "Off-diagonal should be 0 at ({}, {})",
                    i, j
                );
            }
        }
    }

    // Last row: only last element non-zero
    for j in 0..5 {
        assert_eq!(lattice[5][j], 0, "Last row should be 0 at column {}", j);
    }
    assert!(
        lattice[5][5] > 0,
        "Bottom-right should be positive (C*ln(N))"
    );

    // Last column should have positive values (C*ln(p_i) > 0 for all primes)
    for i in 0..5 {
        assert!(
            lattice[i][5] > 0,
            "Last column should be positive at row {} (C*ln({}))",
            i,
            factor_base[i]
        );
    }
}

#[test]
fn test_lattice_logarithm_ordering() {
    let n = BigUint::from(1000u64);
    let factor_base = vec![2, 3, 5, 7, 11, 13];
    let lattice = build_schnorr_lattice(&n, &factor_base, 100.0);

    // The last column values should be in increasing order (since ln is monotone)
    let k = factor_base.len();
    for i in 1..k {
        assert!(
            lattice[i][k] > lattice[i - 1][k],
            "C*ln(p_{}) = {} should be > C*ln(p_{}) = {}",
            i,
            lattice[i][k],
            i - 1,
            lattice[i - 1][k]
        );
    }
}

// ============================================================
// LLL Reduction Tests
// ============================================================

#[test]
fn test_lll_reduction_produces_shorter_vectors() {
    let n = BigUint::from(143u64);
    let factor_base = vec![2, 3, 5, 7, 11];

    // Build and reduce
    let unreduced = tnss_factoring::lattice::build_schnorr_lattice_v2(&n, &factor_base, 2.0, None);
    let reduced = build_reduced_lattice_v2(&n, &factor_base, 2.0);

    let norm_unreduced_min: f64 = unreduced
        .iter()
        .map(|r| vector_norm(r))
        .fold(f64::MAX, f64::min);
    let norm_reduced_first = vector_norm(&reduced[0]);

    assert!(
        norm_reduced_first <= norm_unreduced_min + 1e-6,
        "LLL should produce shorter first vector: reduced={} vs unreduced_min={}",
        norm_reduced_first,
        norm_unreduced_min
    );
}

#[test]
fn test_gram_schmidt_produces_orthogonal_vectors() {
    let basis = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let (ortho, _mu) = gram_schmidt(&basis);

    // Check orthogonality
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dot: f64 = (0..3).map(|k| ortho[i][k] * ortho[j][k]).sum();
            assert!(
                dot.abs() < 1e-8,
                "Vectors {} and {} should be orthogonal, dot={}",
                i,
                j,
                dot
            );
        }
    }
}

// ============================================================
// TTN Construction and Contraction Tests
// ============================================================

#[test]
fn test_ttn_contraction_gives_valid_probabilities() {
    let mut rng = StdRng::seed_from_u64(42);
    let ttn = Ttn::new_random(4, 2, &mut rng);

    // All amplitudes should be finite
    for bits in 0..16u64 {
        let config: Vec<u8> = (0..4).map(|j| ((bits >> j) & 1) as u8).collect();
        let amp = ttn.evaluate(&config);
        assert!(amp.is_finite(), "Amplitude should be finite for config {:?}", config);
    }

    // Norm squared should be positive
    let norm_sq = ttn.contract_norm_squared();
    assert!(norm_sq > 0.0, "Norm squared should be positive, got {}", norm_sq);
}

#[test]
fn test_ttn_paper_bond_dim_scaling() {
    // Verify the bond dimension formula
    assert_eq!(compute_paper_bond_dim(4), 2);   // 4^0.4 = 1.74 -> ceil = 2
    assert_eq!(compute_paper_bond_dim(10), 3);  // 10^0.4 = 2.51 -> ceil = 3
    assert_eq!(compute_paper_bond_dim(32), 4);  // 32^0.4 = 4.0 -> ceil = 4
    assert_eq!(compute_paper_bond_dim(100), 7); // 100^0.4 = 6.31 -> ceil = 7
}

// ============================================================
// OPES Sampling Tests
// ============================================================

#[test]
fn test_opes_sampling_produces_valid_configs() {
    let mut rng = StdRng::seed_from_u64(42);

    let hamiltonian = CostHamiltonian {
        terms: vec![
            QuadraticTerm { i: 0, j: 0, coeff: 1.0 },
            QuadraticTerm { i: 1, j: 1, coeff: 1.0 },
            QuadraticTerm { i: 2, j: 2, coeff: 1.0 },
        ],
        constant: 0.0,
        num_vars: 3,
    };

    let ttn = Ttn::new_random(3, 2, &mut rng);
    let config = OpesConfig {
        beta_schedule: vec![0.1, 1.0, 5.0],
        samples_per_temp: 10,
        sweeps_per_temp: 0,
    };

    let samples = opes_sample(&ttn, &hamiltonian, &config, &mut rng);
    assert!(!samples.is_empty());

    for sample in &samples {
        assert_eq!(sample.config.len(), 3);
        for &b in &sample.config {
            assert!(b == 0 || b == 1);
        }
        assert!(sample.energy.is_finite());
    }
}

#[test]
fn test_opes_extract_best_returns_sorted() {
    let mut rng = StdRng::seed_from_u64(42);

    let hamiltonian = CostHamiltonian {
        terms: vec![
            QuadraticTerm { i: 0, j: 0, coeff: 5.0 },
            QuadraticTerm { i: 1, j: 1, coeff: 3.0 },
        ],
        constant: 0.0,
        num_vars: 2,
    };

    let ttn = Ttn::new_random(2, 2, &mut rng);
    let config = OpesConfig {
        beta_schedule: default_beta_schedule(5),
        samples_per_temp: 20,
        sweeps_per_temp: 0,
    };

    let samples = opes_sample(&ttn, &hamiltonian, &config, &mut rng);
    let best = extract_best_configs(&samples, 5);

    // Should be sorted by energy
    for i in 1..best.len() {
        assert!(
            best[i].1 >= best[i - 1].1,
            "Configs should be sorted by energy: {} >= {}",
            best[i].1,
            best[i - 1].1
        );
    }
}

// ============================================================
// Factor Extraction Tests
// ============================================================

#[test]
fn test_extract_short_vectors_from_reduced_lattice() {
    let n = BigUint::from(143u64);
    let factor_base = vec![2, 3, 5, 7, 11];
    let reduced = build_reduced_lattice_v2(&n, &factor_base, 2.0);

    let vectors = extract_short_vectors(&reduced, factor_base.len(), 10);
    assert!(!vectors.is_empty(), "Should extract at least one short vector");

    for v in &vectors {
        assert_eq!(v.len(), factor_base.len());
        assert!(v.iter().any(|&e| e != 0), "Vector should be non-trivial");
    }
}

// ============================================================
// MPS / Tensor Tests (backward compatibility)
// ============================================================

#[test]
fn test_mps_deterministic_evaluation() {
    let mut rng = StdRng::seed_from_u64(42);
    let mps = Mps::new_random(6, 4, &mut rng);

    let config = vec![0u8, 1, 0, 1, 1, 0];
    let amp1 = mps.evaluate(&config);
    let amp2 = mps.evaluate(&config);
    assert_eq!(amp1, amp2, "Evaluation should be deterministic");
}

#[test]
fn test_mps_norm_consistency() {
    let mut rng = StdRng::seed_from_u64(42);
    let mps = Mps::new_random(4, 3, &mut rng);

    // Compute norm squared two ways
    let norm_sq_contract = mps.contract_norm_squared();

    // Compute by explicit enumeration
    let n = mps.num_vars;
    let mut norm_sq_enum = 0.0;
    for bits in 0..(1u64 << n) {
        let config: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
        let amp = mps.evaluate(&config);
        norm_sq_enum += amp * amp;
    }

    assert!(
        (norm_sq_contract - norm_sq_enum).abs() < 1e-10,
        "Norm from contraction ({}) should match enumeration ({})",
        norm_sq_contract,
        norm_sq_enum
    );
}

#[test]
fn test_mps_sampling_valid() {
    let mut rng = StdRng::seed_from_u64(42);
    let mps = Mps::new_random(8, 4, &mut rng);

    let mut rng2 = StdRng::seed_from_u64(123);
    for _ in 0..50 {
        let config = mps.sample(&mut rng2);
        assert_eq!(config.len(), 8);
        for &b in &config {
            assert!(b == 0 || b == 1, "Sample should be binary, got {}", b);
        }
    }
}

#[test]
fn test_mps_find_max_small() {
    let mut rng = StdRng::seed_from_u64(42);
    let mps = Mps::new_random(5, 2, &mut rng);

    let (best_config, best_amp_sq) = mps.find_max_config();

    // Verify it's actually the maximum
    let n = mps.num_vars;
    for bits in 0..(1u64 << n) {
        let config: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
        let amp = mps.evaluate(&config);
        assert!(
            amp * amp <= best_amp_sq + 1e-12,
            "Config {:?} has amp^2 {} > claimed max {}",
            config,
            amp * amp,
            best_amp_sq
        );
    }

    // Verify the best config gives the claimed amplitude
    let actual_amp = mps.evaluate(&best_config);
    assert!(
        (actual_amp * actual_amp - best_amp_sq).abs() < 1e-12,
        "Best config amplitude doesn't match: {} vs {}",
        actual_amp * actual_amp,
        best_amp_sq
    );
}

// ============================================================
// Optimizer Tests
// ============================================================

#[test]
fn test_binary_encoding_roundtrip_various() {
    let test_cases: Vec<(Vec<i64>, usize, i64)> = vec![
        (vec![0, 0, 0], 3, 3),
        (vec![1, -1, 2], 4, 7),
        (vec![-3, 3, 0, -2], 3, 3),
        (vec![7, -7], 4, 7),
    ];

    for (exponents, bits, bound) in test_cases {
        let binary = exponents_to_binary(&exponents, bits, bound);
        let recovered = binary_to_exponents(&binary, exponents.len(), bits, bound);
        assert_eq!(
            exponents, recovered,
            "Roundtrip failed for {:?} with bits={}, bound={}",
            exponents, bits, bound
        );
    }
}

#[test]
fn test_cost_hamiltonian_minimum_at_zero() {
    // Cost = x0^2 + x1^2 where x_i = b_i (no offset)
    // Minimum at b_0 = b_1 = 0 with cost 0
    let h = CostHamiltonian {
        terms: vec![
            QuadraticTerm {
                i: 0,
                j: 0,
                coeff: 1.0,
            },
            QuadraticTerm {
                i: 1,
                j: 1,
                coeff: 1.0,
            },
        ],
        constant: 0.0,
        num_vars: 2,
    };

    assert!((h.evaluate(&[0, 0]) - 0.0).abs() < 1e-10);
    assert!((h.evaluate(&[1, 0]) - 1.0).abs() < 1e-10);
    assert!((h.evaluate(&[0, 1]) - 1.0).abs() < 1e-10);
    assert!((h.evaluate(&[1, 1]) - 2.0).abs() < 1e-10);
}

#[test]
fn test_build_cost_from_lattice_consistency() {
    let n = BigUint::from(143u64);
    let factor_base = vec![2, 3, 5];
    let lattice = build_schnorr_lattice(&n, &factor_base, 50.0);
    let target = vec![0i64; lattice.len()];

    let h = build_cost_from_lattice(&lattice, &target, 3, 2, 1);

    // The cost should be non-negative for all configurations
    let num_vars = 3 * 2; // 3 exponents, 2 bits each
    for bits in 0..(1u64 << num_vars) {
        let config: Vec<u8> = (0..num_vars).map(|j| ((bits >> j) & 1) as u8).collect();
        let cost = h.evaluate(&config);
        assert!(
            cost >= -1e-6, // Allow small numerical error
            "Cost should be non-negative, got {} for config {:?}",
            cost,
            config
        );
    }
}

#[test]
fn test_optimizer_simple_problem() {
    let mut rng = StdRng::seed_from_u64(42);

    // Minimize cost = b_0 + b_1 (minimum at [0, 0] with cost 0)
    let h = CostHamiltonian {
        terms: vec![
            QuadraticTerm {
                i: 0,
                j: 0,
                coeff: 5.0,
            },
            QuadraticTerm {
                i: 1,
                j: 1,
                coeff: 5.0,
            },
        ],
        constant: 0.0,
        num_vars: 2,
    };

    let mut mps = Mps::new_random(2, 2, &mut rng);
    let energy = optimize_sweep(&mut mps, &h, 10);

    // The optimizer should find an energy close to 0
    // (the minimum is 0 at config [0,0])
    assert!(
        energy < 5.0,
        "Optimizer should find energy < 5.0, got {}",
        energy
    );
}

// ============================================================
// Relations / GF(2) Tests
// ============================================================

#[test]
fn test_gf2_elimination_finds_null_space() {
    use tnss_factoring::lattice::SmoothRelation;

    // Create relations where combining pairs gives even exponents
    let relations = vec![
        SmoothRelation {
            lhs: BigUint::from(2u64),
            rhs: BigUint::one(),
            exponents: vec![1, 0, 0],
        },
        SmoothRelation {
            lhs: BigUint::from(6u64),
            rhs: BigUint::one(),
            exponents: vec![1, 1, 0],
        },
        SmoothRelation {
            lhs: BigUint::from(3u64),
            rhs: BigUint::one(),
            exponents: vec![0, 1, 0],
        },
    ];

    let null_vecs = gaussian_elimination_gf2(&relations, 3);

    // Let's check that any found null vector actually gives all-even exponents
    for null_vec in &null_vecs {
        let mut combined = vec![0i32; 3];
        for &idx in null_vec {
            for (j, &e) in relations[idx].exponents.iter().enumerate() {
                combined[j] += e;
            }
        }
        for &e in &combined {
            assert!(
                e % 2 == 0,
                "Null vector {:?} gives odd exponent in {:?}",
                null_vec,
                combined
            );
        }
    }
}

// ============================================================
// Full Pipeline Tests (V1)
// ============================================================

#[test]
fn test_factor_small_semiprime_15() {
    // N = 15 = 3 * 5
    let result = factor_u64(15);
    assert!(result.factor.is_some(), "Should factor 15 (3 x 5)");
    let f = result.factor.unwrap();
    assert!(
        f == BigUint::from(3u64) || f == BigUint::from(5u64),
        "Factor of 15 should be 3 or 5, got {}",
        f
    );
}

#[test]
fn test_factor_small_semiprime_21() {
    let result = factor_u64(21);
    assert!(result.factor.is_some(), "Should factor 21 (3 x 7)");
    let f = result.factor.unwrap();
    assert!(
        f == BigUint::from(3u64) || f == BigUint::from(7u64),
        "Factor of 21 should be 3 or 7, got {}",
        f
    );
}

#[test]
fn test_factor_small_semiprime_77() {
    let result = factor_u64(77);
    assert!(result.factor.is_some(), "Should factor 77 (7 x 11)");
    let f = result.factor.unwrap();
    assert!(
        f == BigUint::from(7u64) || f == BigUint::from(11u64),
        "Factor of 77 should be 7 or 11, got {}",
        f
    );
}

#[test]
fn test_factor_143() {
    // N = 143 = 11 * 13
    let result = factor_u64(143);
    assert!(result.factor.is_some(), "Should factor 143 (11 x 13)");
    let f = result.factor.unwrap();
    let n = BigUint::from(143u64);
    let other = &n / &f;
    assert!(
        f > BigUint::one() && other > BigUint::one(),
        "Factors should be non-trivial: {} x {}",
        f,
        other
    );
    assert_eq!(&f * &other, n, "Factors should multiply to N");
}

#[test]
fn test_factor_323() {
    // N = 323 = 17 * 19
    let result = factor_u64(323);
    assert!(result.factor.is_some(), "Should factor 323 (17 x 19)");
    let f = result.factor.unwrap();
    let n = BigUint::from(323u64);
    let other = &n / &f;
    assert_eq!(&f * &other, n, "Factors should multiply to N");
}

#[test]
fn test_factor_with_custom_config() {
    let n = BigUint::from(221u64); // 13 * 17
    let config = TnssConfig {
        factor_base_size: 10,
        scaling: 80.0,
        bits_per_exponent: 3,
        exponent_bound: 3,
        bond_dim: 4,
        num_sweeps: 10,
        num_attempts: 50,
        seed: 123,
    };

    let result = factor_tnss(&n, &config);
    assert!(result.factor.is_some(), "Should factor 221 (13 x 17)");
}

#[test]
fn test_result_structure() {
    let result = factor_u64(35);
    assert_eq!(result.n, BigUint::from(35u64));
    // Should have found a factor via trial division
    assert!(result.factor.is_some());
}

// ============================================================
// Full Pipeline Tests (V2)
// ============================================================

#[test]
fn test_factor_v2_small_semiprimes() {
    // These should all succeed via trial division in the v2 factor base
    for &n in &[15u64, 21, 35, 77, 143, 221, 323] {
        let result = factor_u64_v2(n);
        assert!(
            result.factor.is_some(),
            "V2 should factor {} via trial division",
            n
        );
    }
}

#[test]
fn test_factor_v2_custom_config() {
    let n = BigUint::from(143u64);
    let config = TnssConfigV2 {
        factor_base_size: 10,
        precision_c: 2.0,
        bits_per_exponent: 3,
        exponent_bound: 3,
        bond_dim: 3,
        temperature_schedule: vec![0.5, 1.0, 3.0],
        samples_per_temp: 15,
        sweeps_per_temp: 1,
        num_cvp_instances: 2,
        c_range: (1.0, 3.0),
        num_attempts_per_instance: 5,
        seed: 42,
    };

    let result = factor_tnss_v2(&n, &config);
    assert!(
        result.factor.is_some(),
        "V2 should factor 143 with custom config"
    );
}

#[test]
fn test_config_for_bits_v2_produces_valid_config() {
    for bits in &[8, 16, 24, 32, 48, 64] {
        let config = config_for_bits_v2(*bits);
        assert!(config.factor_base_size >= 6);
        assert!(config.bond_dim >= 2);
        assert!(!config.temperature_schedule.is_empty());
        assert!(config.num_cvp_instances >= 1);
        assert!(config.bits_per_exponent >= 3);
        assert!(config.exponent_bound >= 3);
        assert!(config.c_range.0 < config.c_range.1);
    }
}
