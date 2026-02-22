use spectral_factoring::dimension::{
    dim_s2, dim_s2_new, dim_s2_old, legendre_symbol, count_cusps,
    count_elliptic_2, count_elliptic_3, psi_index,
};
use spectral_factoring::modular_symbols::{enumerate_p1, modular_symbol_space};
use spectral_factoring::hecke::hecke_matrix;
use spectral_factoring::spectral::{factor_from_spectral, factor_from_dimensions};

#[test]
fn test_dimension_formula_primes() {
    // Verify dim S_2(Gamma_0(p)) for all primes up to 100
    let expected: Vec<(u64, u64)> = vec![
        (2, 0), (3, 0), (5, 0), (7, 0), (11, 1), (13, 0),
        (17, 1), (19, 1), (23, 2), (29, 2), (31, 2), (37, 2),
        (41, 3), (43, 3), (47, 4), (53, 4), (59, 5), (61, 4),
        (67, 5), (71, 6), (73, 5), (79, 6), (83, 7), (89, 7),
        (97, 7),
    ];

    for (p, expected_dim) in expected {
        let computed = dim_s2(p);
        assert_eq!(computed, expected_dim,
            "dim S_2(Gamma_0({})) = {} but expected {}", p, computed, expected_dim);
    }
}

#[test]
fn test_modular_symbols_p1_size() {
    // |P^1(Z/NZ)| = psi(N) = N * product(1 + 1/p for p | N)
    let test_levels: Vec<u64> = vec![7, 11, 13, 23, 77, 143];
    for n in test_levels {
        let p1 = enumerate_p1(n);
        let expected = psi_index(n);
        assert_eq!(p1.len() as u64, expected,
            "P^1(Z/{}Z) has {} elements but expected {}", n, p1.len(), expected);
    }
}

#[test]
fn test_hecke_operator_level_11() {
    // Level 11: dim S_2 = 1
    // The unique newform has a_2 = -2, a_3 = -1, a_5 = 1, a_7 = -2
    let space = modular_symbol_space(11);
    assert!(space.dimension >= 1, "Level 11 should have positive quotient dimension");

    // Hecke traces should be computable
    if space.dimension > 0 {
        let t2 = hecke_matrix(&space, 2);
        let t3 = hecke_matrix(&space, 3);
        // These should be well-defined matrices
        assert_eq!(t2.len(), space.dimension);
        assert_eq!(t3.len(), space.dimension);
    }
}

#[test]
fn test_spectral_factor_semiprimes() {
    let test_cases = vec![
        (77u64, 7u64, 11u64),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (437, 19, 23),
        (667, 23, 29),
    ];

    for (n, p, q) in test_cases {
        let result = factor_from_spectral(n);
        assert_eq!(result, Some((p, q)),
            "factor_from_spectral({}) should return ({}, {})", n, p, q);
    }
}

#[test]
fn test_old_new_dimension_consistency() {
    // For N = pq (two distinct primes):
    // dim_old(N) = 2*dim(p) + 2*dim(q)
    // dim_total(N) = dim_old(N) + dim_new(N)
    let test_cases = vec![
        (77u64, 7u64, 11u64),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
    ];

    for (n, p, q) in test_cases {
        let total = dim_s2(n);
        let new = dim_s2_new(n);
        let old = dim_s2_old(n);

        assert_eq!(total, new + old,
            "dim_total({}) = {} != dim_new + dim_old = {} + {}", n, total, new, old);

        let expected_old = 2 * dim_s2(p) + 2 * dim_s2(q);
        assert_eq!(old, expected_old,
            "dim_old({}) = {} != 2*dim({}) + 2*dim({}) = 2*{} + 2*{} = {}",
            n, old, p, q, dim_s2(p), dim_s2(q), expected_old);
    }
}

#[test]
fn test_dimension_based_factoring() {
    // The dimension-based approach should work for all these semiprimes
    let semiprimes = vec![
        77u64, 143, 221, 323, 437, 667, 899, 1517, 2021,
    ];

    for n in semiprimes {
        let result = factor_from_dimensions(n);
        assert!(result.is_some(), "Should factor {} via dimensions", n);
        let (p, q) = result.unwrap();
        assert_eq!(p * q, n, "{} != {} * {}", n, p, q);
        assert!(p > 1 && q > 1, "Factors should be non-trivial: {} = {} * {}", n, p, q);
    }
}

#[test]
fn test_cusps_count() {
    // For prime p: 2 cusps
    assert_eq!(count_cusps(5), 2);
    assert_eq!(count_cusps(7), 2);
    assert_eq!(count_cusps(11), 2);

    // For p*q: 4 cusps (squarefree with 2 prime factors)
    assert_eq!(count_cusps(15), 4); // 3*5
    assert_eq!(count_cusps(77), 4); // 7*11
}

#[test]
fn test_elliptic_points() {
    // nu_2(p) = 1 + (-1/p): nonzero iff p = 1 mod 4
    // p=5: (-1/5)=1 (since 5=1mod4), so nu_2(5) = 2
    assert_eq!(count_elliptic_2(5), 2);
    // p=7: (-1/7)=-1, so nu_2(7) = 0
    assert_eq!(count_elliptic_2(7), 0);
    // p=13: (-1/13)=1 (13=1mod4), so nu_2(13) = 2
    assert_eq!(count_elliptic_2(13), 2);

    // nu_3(p) = 1 + (-3/p): nonzero iff p = 1 mod 3
    // p=7: (-3/7): -3 mod 7 = 4, (4/7)=4^3=64=1mod7 => 1. nu_3(7) = 2
    assert_eq!(count_elliptic_3(7), 2);
    // p=5: (-3/5): -3 mod 5 = 2, (2/5)=-1. nu_3(5) = 0
    assert_eq!(count_elliptic_3(5), 0);
}

#[test]
fn test_legendre_symbol_quadratic_reciprocity() {
    // Quadratic reciprocity: for odd primes p != q:
    // (p/q)(q/p) = (-1)^((p-1)/2 * (q-1)/2)
    let primes = vec![3u64, 5, 7, 11, 13, 17, 19, 23];
    for i in 0..primes.len() {
        for j in (i + 1)..primes.len() {
            let p = primes[i];
            let q = primes[j];
            let lhs = legendre_symbol(p as i64, q) * legendre_symbol(q as i64, p);
            let rhs = if ((p - 1) / 2) * ((q - 1) / 2) % 2 == 0 { 1 } else { -1 };
            assert_eq!(lhs, rhs,
                "Quadratic reciprocity failed for ({}/{}) * ({}/{}) = {} != {}",
                p, q, q, p, lhs, rhs);
        }
    }
}
