//! Integration tests for the L-function factoring crate.

use l_function_factoring::characters;
use l_function_factoring::class_number;
use l_function_factoring::complex;
use l_function_factoring::factor;
use l_function_factoring::gauss_sums;
use l_function_factoring::l_function;
use std::f64::consts::PI;

#[test]
fn test_kronecker_symbol_basic() {
    // (-4 | 1) = 1
    assert_eq!(characters::kronecker_symbol(-4, 1), 1);

    // Test periodicity: (-4 | n) has period 4
    // (-4 | 1) = 1, (-4 | 2) = 0, (-4 | 3) = -1
    assert_eq!(characters::kronecker_symbol(-4, 1), 1);
    assert_eq!(characters::kronecker_symbol(-4, 3), -1);

    // Quadratic residues mod 5: 1, 4 are QR, 2, 3 are QNR
    assert_eq!(characters::kronecker_symbol(1, 5), 1);
    assert_eq!(characters::kronecker_symbol(4, 5), 1);
    assert_eq!(characters::kronecker_symbol(2, 5), -1);
    assert_eq!(characters::kronecker_symbol(3, 5), -1);
}

#[test]
fn test_gauss_sum_primitive_chars() {
    // For primitive characters mod p (prime), |tau(chi)|^2 = p
    for &p in &[5u64, 7, 11, 13] {
        let chars = characters::enumerate_characters(p);
        for chi in &chars {
            if chi.is_principal {
                continue;
            }
            let mag_sq = gauss_sums::gauss_sum_magnitude_sq(chi);
            assert!(
                (mag_sq - p as f64).abs() < 1.0,
                "For primitive char mod {}: |tau|^2 = {}, expected {}",
                p,
                mag_sq,
                p
            );
        }
    }
}

#[test]
fn test_conductor_detection_composite() {
    // For N = 15 = 3 * 5:
    // - Characters with conductor 3 (trivial on 5-part): should exist
    // - Characters with conductor 5 (trivial on 3-part): should exist
    // - Characters with conductor 15 (primitive): should exist
    let chars = characters::enumerate_characters(15);
    let conductors: Vec<u64> = chars.iter().map(|c| gauss_sums::detect_conductor(c)).collect();

    assert!(conductors.contains(&1), "Should have principal character (conductor 1)");
    assert!(conductors.contains(&3), "Should have characters with conductor 3");
    assert!(conductors.contains(&5), "Should have characters with conductor 5");
    assert!(conductors.contains(&15), "Should have primitive characters");
}

#[test]
fn test_l_function_values_chi_minus4() {
    // L(1, chi_{-4}) = pi/4
    let chi = characters::kronecker_character(-4, 4);
    let l1 = l_function::l_function_at_1(&chi, 10000);
    assert!(
        (l1 - PI / 4.0).abs() < 0.01,
        "L(1, chi_{{-4}}) = {}, expected {}",
        l1,
        PI / 4.0
    );

    // L(2, chi_{-4}) = Catalan's constant G ~ 0.9159655941
    let l2 = l_function::l_function_partial(&chi, 2.0, 10000);
    let catalan = 0.9159655941;
    assert!(
        (l2.0 - catalan).abs() < 0.01,
        "L(2, chi_{{-4}}) = {}, expected {}",
        l2.0,
        catalan
    );
}

#[test]
fn test_class_number_small_discriminants() {
    // Verify class numbers for well-known discriminants
    let known: Vec<(i64, u64)> = vec![
        (-3, 1),
        (-4, 1),
        (-7, 1),
        (-8, 1),
        (-11, 1),
        (-15, 2),
        (-20, 2),
        (-23, 3),
        (-24, 2),
    ];

    for (d, expected_h) in known {
        let h = class_number::class_number_direct(d);
        assert_eq!(h, expected_h, "h({}) should be {}, got {}", d, expected_h, h);
    }
}

#[test]
fn test_class_number_formula_vs_direct() {
    for &d in &[-3i64, -4, -7, -8, -11, -15, -20, -23] {
        let h_direct = class_number::class_number_direct(d);
        let h_formula = class_number::class_number_via_kronecker(d, 50000);
        assert!(
            (h_formula - h_direct as f64).abs() < 0.5,
            "h({}) formula = {:.3}, direct = {}",
            d,
            h_formula,
            h_direct
        );
    }
}

#[test]
fn test_factor_by_conductor_semiprimes() {
    let semiprimes: Vec<(u64, u64, u64)> = vec![
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
    ];

    for (n, p, q) in semiprimes {
        let result = factor::factor_by_conductor(n);
        assert!(
            result.factors.is_some(),
            "Should factor {} = {} * {}",
            n,
            p,
            q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, n, "Factors of {} should multiply correctly", n);
        let (small, big) = if a < b { (a, b) } else { (b, a) };
        assert_eq!(small, p, "Smaller factor of {} should be {}", n, p);
        assert_eq!(big, q, "Larger factor of {} should be {}", n, q);
    }
}

#[test]
fn test_factor_small_semiprimes_unified() {
    // Use the unified factor() function
    for &(n, p, q) in &[(15u64, 3, 5), (21, 3, 7), (35, 5, 7), (77, 7, 11)] {
        let result = factor::factor(n);
        assert!(
            result.factors.is_some(),
            "Unified factor should handle {} = {} * {}",
            n,
            p,
            q
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, n);
    }
}

#[test]
fn test_character_orthogonality() {
    // Orthogonality relation: sum over a of chi(a) = 0 for non-principal chi
    let chars = characters::enumerate_characters(7);
    for chi in &chars {
        let sum: complex::Complex = (1..7u64)
            .map(|a| chi.eval(a))
            .fold(complex::ZERO, |acc, v| complex::cadd(acc, v));

        if chi.is_principal {
            // Sum of principal character over units = phi(N)
            assert!(
                (sum.0 - 6.0).abs() < 1e-8,
                "Principal char sum should be phi(7) = 6"
            );
        } else {
            // Sum of non-principal character = 0
            assert!(
                complex::cabs(sum) < 1e-8,
                "Non-principal char sum should be 0, got |sum| = {}",
                complex::cabs(sum)
            );
        }
    }
}

#[test]
fn test_euler_product_convergence() {
    // Compare Euler product with partial sums for L(2, chi)
    let chi = characters::kronecker_character(-4, 4);
    let primes = l_function::sieve_primes(500);

    let l_euler = l_function::l_function_euler_product(&chi, 2.0, &primes);
    let l_partial = l_function::l_function_partial(&chi, 2.0, 5000);

    assert!(
        (l_euler.0 - l_partial.0).abs() < 0.05,
        "Euler product L(2) = {:.6}, partial sum = {:.6}",
        l_euler.0,
        l_partial.0
    );
}
