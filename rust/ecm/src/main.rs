use std::time::Instant;

use ecm::{ecm_factor, ecm_factor_with_result, ecm_parallel_multi_curve, EcmParams};
use factoring_core::generate_rsa_target;
use num_bigint::BigUint;
use num_traits::Zero;

fn main() {
    println!("=== Elliptic Curve Method (ECM) Factorization ===\n");

    section_1_basic_ecm();
    section_2_stage1_vs_stage2();
    section_3_multi_curve_scaling();
    section_4_factor_result();
    section_5_random_rsa_targets();
}

// -------------------------------------------------------------------------
// Section 1 — Basic ECM
// -------------------------------------------------------------------------

fn section_1_basic_ecm() {
    println!("--- Section 1: Basic ECM on Known Semiprimes ---\n");

    struct TestCase {
        n: u64,
        description: &'static str,
    }

    let test_cases = vec![
        TestCase {
            n: 8051,
            description: "83 x 97",
        },
        TestCase {
            n: 15347,
            description: "103 x 149",
        },
        TestCase {
            n: 1_000_003 * 1_000_033,
            description: "1000003 x 1000033",
        },
        TestCase {
            n: 104729 * 104743,
            description: "104729 x 104743",
        },
    ];

    let params = EcmParams::default();
    println!(
        "  Params: B1={}, B2={}, curves={}\n",
        params.b1, params.b2, params.num_curves
    );

    for tc in &test_cases {
        let n = BigUint::from(tc.n);
        let result = ecm_factor(&n, &params);
        if result.complete {
            println!(
                "  n={:<20} ({}): {} x {} in {:?}",
                tc.n, tc.description, result.factors[0], result.factors[1], result.duration
            );
        } else {
            println!(
                "  n={:<20} ({}): FAILED in {:?}",
                tc.n, tc.description, result.duration
            );
        }
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 2 — Stage 1 vs Stage 2 Comparison
// -------------------------------------------------------------------------

fn section_2_stage1_vs_stage2() {
    println!("--- Section 2: Stage 1 vs Stage 2 Comparison ---\n");

    // Use a number where Stage 2 matters: both factors have (p-1) with
    // a large prime factor, so Stage 1 alone with small B1 is unlikely
    // to succeed but Stage 2 with large B2 should catch it.
    //
    // n = 1000003 * 1000033 = 1_000_036_000_099
    // 1000003 - 1 = 1000002 = 2 * 3 * 166667  (166667 is prime)
    // 1000033 - 1 = 1000032 = 2^5 * 3 * 10417 + ... needs Stage 2
    let n = BigUint::from(1_000_036_000_099u64);
    println!("  Target: n = 1000003 x 1000033 = {}", n);
    println!("    (p-1) = 1000002 has large prime factor 166667");
    println!();

    struct StageConfig {
        label: &'static str,
        b1: u64,
        b2: u64,
    }

    let configs = vec![
        StageConfig {
            label: "Stage 1 only (B1=1000, B2=B1)",
            b1: 1_000,
            b2: 1_000,
        },
        StageConfig {
            label: "Stage 1 only (B1=10000, B2=B1)",
            b1: 10_000,
            b2: 10_000,
        },
        StageConfig {
            label: "Stage 2 enabled (B1=1000, B2=200000)",
            b1: 1_000,
            b2: 200_000,
        },
        StageConfig {
            label: "Stage 2 enabled (B1=10000, B2=1000000)",
            b1: 10_000,
            b2: 1_000_000,
        },
    ];

    let num_curves = 64;

    for cfg in &configs {
        let params = EcmParams {
            b1: cfg.b1,
            b2: cfg.b2,
            num_curves,
        };

        let start = Instant::now();
        let result = ecm_factor(&n, &params);
        let elapsed = start.elapsed();

        let status = if result.complete {
            format!(
                "SUCCESS: {} x {} ({:?})",
                result.factors[0], result.factors[1], elapsed
            )
        } else {
            format!("FAILED ({:?})", elapsed)
        };

        println!("  {}: {}", cfg.label, status);
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 3 — Multi-Curve Scaling
// -------------------------------------------------------------------------

fn section_3_multi_curve_scaling() {
    println!("--- Section 3: Multi-Curve Scaling ---\n");

    // Use a moderately hard target where more curves = higher success probability.
    // 48-bit semiprime: 224737 * 350377 = 78_737_654_849
    let n = BigUint::from(224_737u64) * BigUint::from(350_377u64);
    let b1 = 5_000u64;
    let b2 = 500_000u64;

    println!("  Target: n = {} ({} bits)", n, n.bits());
    println!("  Bounds: B1={}, B2={}", b1, b2);
    println!();

    let curve_counts = [1, 4, 16, 64];
    let trials_per_count = 5;

    for &num_curves in &curve_counts {
        let mut successes = 0;
        let mut total_time = std::time::Duration::ZERO;

        for _ in 0..trials_per_count {
            let start = Instant::now();
            let factor = ecm_parallel_multi_curve(&n, b1, b2, num_curves);
            total_time += start.elapsed();
            if let Some(f) = factor {
                if !f.is_zero() && f != n {
                    successes += 1;
                }
            }
        }

        let avg_ms = total_time.as_millis() as f64 / trials_per_count as f64;
        println!(
            "  curves={:<3}: {}/{} succeeded, avg time={:.1}ms",
            num_curves, successes, trials_per_count, avg_ms
        );
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 4 — ECM with FactorResult
// -------------------------------------------------------------------------

fn section_4_factor_result() {
    println!("--- Section 4: ECM with FactorResult ---\n");

    let test_cases: &[(u64, &str)] = &[
        (8051, "83 x 97"),
        (104729 * 104743, "104729 x 104743"),
    ];

    for &(n_val, desc) in test_cases {
        let n = BigUint::from(n_val);
        let params = EcmParams {
            b1: 50_000,
            b2: 2_000_000,
            num_curves: 64,
        };

        let result = ecm_factor_with_result(&n, &params);

        println!("  FactorResult for n = {} ({}):", n_val, desc);
        println!("    algorithm:  {}", result.algorithm);
        println!("    n:          {}", result.n);
        println!("    complete:   {}", result.complete);
        println!("    duration:   {:?}", result.duration);
        if result.complete {
            println!(
                "    factors:    {} x {}",
                result.factors[0], result.factors[1]
            );
            let product: BigUint = result.factors.iter().product();
            println!("    verified:   {}", product == result.n);
        } else {
            println!("    factors:    (none found)");
        }
        println!();
    }
}

// -------------------------------------------------------------------------
// Section 5 — Random RSA Targets
// -------------------------------------------------------------------------

fn section_5_random_rsa_targets() {
    println!("--- Section 5: Random RSA Targets ---\n");

    let mut rng = rand::thread_rng();

    let bit_sizes = [32, 48, 64];

    for &bits in &bit_sizes {
        println!("  {}-bit semiprimes:", bits);

        // Scale params with bit size
        let params = match bits {
            32 => EcmParams {
                b1: 5_000,
                b2: 100_000,
                num_curves: 32,
            },
            48 => EcmParams {
                b1: 50_000,
                b2: 2_000_000,
                num_curves: 64,
            },
            _ => EcmParams {
                b1: 100_000,
                b2: 10_000_000,
                num_curves: 128,
            },
        };

        let trials = 3;
        let mut successes = 0;

        for i in 0..trials {
            let target = generate_rsa_target(bits, &mut rng);
            let result = ecm_factor(&target.n, &params);

            if result.complete {
                let verified = target.verify(&result);
                successes += 1;
                println!(
                    "    [{}] n={} => {} x {} in {:?} (verified: {})",
                    i + 1,
                    target.n,
                    result.factors[0],
                    result.factors[1],
                    result.duration,
                    verified
                );
            } else {
                println!(
                    "    [{}] n={} ({}x{}) => FAILED in {:?}",
                    i + 1,
                    target.n,
                    target.p,
                    target.q,
                    result.duration
                );
            }
        }

        println!(
            "    Success rate: {}/{}\n",
            successes, trials
        );
    }
}
