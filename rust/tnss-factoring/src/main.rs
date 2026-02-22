//! TNSS: Tensor Network Schnorr Sieving -- Demo
//!
//! Demonstrates both v1 (MPS) and v2 (TTN + OPES, Tesoro/Siloi 2024) factoring.

use num_bigint::BigUint;
use tnss_factoring::{
    config_for_bits, config_for_bits_v2, factor_tnss, factor_ttn, factor_u64, factor_u64_v2,
};

fn main() {
    println!("========================================");
    println!("  TNSS: Tensor Network Schnorr Sieving");
    println!("========================================");
    println!();
    println!("V1: Schnorr lattice + MPS tensor network optimization");
    println!("V2: LLL-reduced lattice + TTN + OPES sampling (arXiv:2410.16355)");
    println!();

    // Test on small semiprimes
    let test_cases: Vec<(u64, &str)> = vec![
        (15, "3 x 5"),
        (21, "3 x 7"),
        (35, "5 x 7"),
        (77, "7 x 11"),
        (143, "11 x 13"),
        (323, "17 x 19"),
        (1007, "19 x 53"),
        (3599, "59 x 61"),
        (10403, "101 x 103"),
    ];

    println!("--- Small Semiprimes (trial division range) ---");
    println!();

    for (n, expected) in &test_cases {
        let result = factor_u64(*n);
        let status = if result.factor.is_some() {
            "FACTORED"
        } else {
            "FAILED"
        };

        print!("  N = {:>8} ({:>12}): [{}]", n, expected, status);
        if let Some(ref f) = result.factor {
            let other = BigUint::from(*n) / f;
            print!(" => {} x {}", f, other);
        }
        println!(
            "  (relations: {}, attempts: {})",
            result.relations_found, result.attempts_made
        );
    }

    println!();
    println!("--- V2 Pipeline (TTN + OPES + LLL) ---");
    println!();

    for (n, expected) in &test_cases {
        let result = factor_u64_v2(*n);
        let status = if result.factor.is_some() {
            "FACTORED"
        } else {
            "FAILED"
        };

        print!("  N = {:>8} ({:>12}): [{}]", n, expected, status);
        if let Some(ref f) = result.factor {
            let other = BigUint::from(*n) / f;
            print!(" => {} x {}", f, other);
        }
        println!(
            "  (relations: {}, attempts: {})",
            result.relations_found, result.attempts_made
        );
    }

    println!();
    println!("--- V2 Parameter Scaling ---");
    println!();

    for bits in &[8, 16, 24, 32, 48] {
        let config = config_for_bits_v2(*bits);
        println!(
            "  {}-bit: factor_base={}, bond_dim={}, cvp_instances={}, c_range=({:.1}, {:.1})",
            bits,
            config.factor_base_size,
            config.bond_dim,
            config.num_cvp_instances,
            config.c_range.0,
            config.c_range.1
        );
    }

    println!();
    println!("--- TTN vs MPS Comparison ---");
    println!();

    let comparison_cases: Vec<(u64, &str)> = vec![
        (15, "3 x 5"),
        (21, "3 x 7"),
        (35, "5 x 7"),
        (77, "7 x 11"),
        (143, "11 x 13"),
        (323, "17 x 19"),
    ];

    for (n, expected) in &comparison_cases {
        let n_big = BigUint::from(*n);
        let config = config_for_bits((*n as f64).log2().ceil() as usize);

        // MPS
        let mps_result = factor_tnss(&n_big, &config);
        let mps_status = if mps_result.factor.is_some() {
            "OK"
        } else {
            "FAIL"
        };

        // TTN
        let ttn_result = factor_ttn(&n_big, &config);
        let ttn_status = if ttn_result.factor.is_some() {
            "OK"
        } else {
            "FAIL"
        };

        // V2
        let v2_result = factor_u64_v2(*n);
        let v2_status = if v2_result.factor.is_some() {
            "OK"
        } else {
            "FAIL"
        };

        println!(
            "  N = {:>5} ({}): MPS [{}] rel={}, TTN [{}] rel={}, V2 [{}] rel={}",
            n,
            expected,
            mps_status,
            mps_result.relations_found,
            ttn_status,
            ttn_result.relations_found,
            v2_status,
            v2_result.relations_found
        );
    }

    println!();
    println!("========================================");
    println!("  Demo complete.");
    println!("========================================");
}
