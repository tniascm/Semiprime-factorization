use std::collections::HashMap;

use factoring_core::generate_rsa_target;
use group_structure::{
    approximate_carmichael, baby_step_giant_step, chebotarev, element_order, euler_totient,
    factor_from_phi, factor_order, factor_via_smooth_orders, find_smooth_order_element,
    phi_from_orders, pohlig_hellman_order, sample_orders,
};
use num_bigint::BigUint;

fn main() {
    println!("=== Group Structure Analysis for Factorization ===\n");

    section_1_element_orders();
    section_2_pohlig_hellman();
    section_3_baby_step_giant_step();
    section_4_carmichael();
    section_5_smooth_order_factoring();
    section_6_phi_recovery();
    section_7_chebotarev();
}

// -------------------------------------------------------------------------
// Section 1 — Element Orders
// -------------------------------------------------------------------------

fn section_1_element_orders() {
    println!("--- Section 1: Element Orders in (Z/nZ)* ---\n");

    let test_cases: &[(u64, u64, u64)] = &[
        // (n, p, q) — known semiprimes
        (35, 5, 7),
        (77, 7, 11),
        (221, 13, 17),
        (899, 29, 31),
    ];

    for &(n_val, p, q) in test_cases {
        let n = BigUint::from(n_val);
        let phi = (p - 1) * (q - 1);
        println!("  n = {} = {} x {}, phi(n) = {}", n_val, p, q, phi);

        let orders = sample_orders(&n, 100, 10_000);
        if orders.is_empty() {
            println!("    No orders sampled (all elements trivial?)\n");
            continue;
        }

        let max_ord = orders.iter().max().unwrap();
        let min_ord = orders.iter().min().unwrap();
        let avg_ord: f64 = orders.iter().map(|&o| o as f64).sum::<f64>() / orders.len() as f64;

        // Build distribution of orders
        let mut freq: HashMap<u64, usize> = HashMap::new();
        for &o in &orders {
            *freq.entry(o).or_insert(0) += 1;
        }
        let mut dist: Vec<(u64, usize)> = freq.into_iter().collect();
        dist.sort();

        println!(
            "    Sampled {} orders: min={}, max={}, avg={:.1}",
            orders.len(),
            min_ord,
            max_ord,
            avg_ord
        );
        println!(
            "    Unique orders: {} — distribution: {:?}",
            dist.len(),
            dist
        );

        // Verify max order divides phi(n)
        println!(
            "    max_order {} divides phi(n) {}? {}",
            max_ord,
            phi,
            phi % max_ord == 0
        );
        println!();
    }
}

// -------------------------------------------------------------------------
// Section 2 — Pohlig-Hellman Decomposition
// -------------------------------------------------------------------------

fn section_2_pohlig_hellman() {
    println!("--- Section 2: Pohlig-Hellman Order Computation ---\n");

    // Demonstrate factor_order on known composite orders
    let orders_to_factor: &[u64] = &[12, 24, 60, 120, 360, 840, 2520];
    println!("  Order factorizations:");
    for &ord in orders_to_factor {
        let factors = factor_order(ord);
        let factor_str: Vec<String> = factors
            .iter()
            .map(|(p, e)| {
                if *e == 1 {
                    format!("{}", p)
                } else {
                    format!("{}^{}", p, e)
                }
            })
            .collect();
        println!("    {} = {}", ord, factor_str.join(" * "));
    }
    println!();

    // Compare pohlig_hellman_order vs element_order on a moderate-sized n
    let n = BigUint::from(1073u64); // 29 * 37
    let phi = 28u64 * 36; // = 1008
    println!(
        "  Comparing Pohlig-Hellman vs sequential order computation for n = 1073 (29 x 37):"
    );
    println!("    phi(n) = {}, lambda(n) = lcm(28,36) = {}", phi, lcm(28, 36));

    // Test several specific elements
    let test_elements: &[u64] = &[2, 3, 5, 7, 10, 11, 13];
    for &a_val in test_elements {
        let a = BigUint::from(a_val);
        let seq = element_order(&a, &n, 10_000);
        // Use phi as the order bound for Pohlig-Hellman
        let ph = pohlig_hellman_order(&a, &n, phi);
        let match_str = if seq == ph { "MATCH" } else { "MISMATCH" };
        println!(
            "    ord({}) : sequential={:?}, Pohlig-Hellman={:?}  [{}]",
            a_val, seq, ph, match_str
        );
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 3 — Baby-Step Giant-Step
// -------------------------------------------------------------------------

fn section_3_baby_step_giant_step() {
    println!("--- Section 3: Baby-Step Giant-Step Discrete Logarithm ---\n");

    struct DlogTest {
        base: u64,
        target: u64,
        modulus: u64,
        order: u64,
        expected: Option<u64>,
    }

    let tests = vec![
        DlogTest {
            base: 2,
            target: 8,
            modulus: 35,
            order: 12,
            expected: Some(3),
        },
        DlogTest {
            base: 2,
            target: 1,
            modulus: 35,
            order: 12,
            expected: Some(0),
        },
        DlogTest {
            base: 2,
            target: 4,
            modulus: 15,
            order: 4,
            expected: Some(2),
        },
        DlogTest {
            base: 3,
            target: 9,
            modulus: 77,
            order: 30,
            expected: Some(2),
        },
        // 3^x = 27 mod 77 => x = 3
        DlogTest {
            base: 3,
            target: 27,
            modulus: 77,
            order: 30,
            expected: Some(3),
        },
        // No solution: 5 is not a power of 2 mod 35
        DlogTest {
            base: 2,
            target: 5,
            modulus: 35,
            order: 12,
            expected: None,
        },
    ];

    for test in &tests {
        let b = BigUint::from(test.base);
        let t = BigUint::from(test.target);
        let n = BigUint::from(test.modulus);
        let result = baby_step_giant_step(&b, &t, &n, test.order);

        let status = match (&result, &test.expected) {
            (Some(x), Some(e)) if x == e => "OK".to_string(),
            (None, None) => "OK (no solution)".to_string(),
            _ => format!("UNEXPECTED (got {:?}, expected {:?})", result, test.expected),
        };

        println!(
            "  {}^x = {} (mod {}), order={}: x={:?}  [{}]",
            test.base, test.target, test.modulus, test.order, result, status
        );
    }
    println!();

    // Larger example: find discrete log in Z/221Z (13 * 17)
    let n221 = BigUint::from(221u64);
    let base = BigUint::from(2u64);
    let order_of_2 = element_order(&base, &n221, 10_000);
    println!("  Larger example: n=221 (13 x 17), base=2");
    if let Some(ord) = order_of_2 {
        println!("    Order of 2 mod 221 = {}", ord);
        // Pick a known power: 2^10 mod 221
        let target_val = BigUint::from(2u64).modpow(&BigUint::from(10u64), &n221);
        let dlog = baby_step_giant_step(&base, &target_val, &n221, ord);
        println!(
            "    2^10 mod 221 = {} => BSGS finds x = {:?} (expected 10)",
            target_val, dlog
        );
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 4 — Carmichael Function
// -------------------------------------------------------------------------

fn section_4_carmichael() {
    println!("--- Section 4: Carmichael Function Approximation ---\n");

    struct CarmichaelTest {
        n: u64,
        p: u64,
        q: u64,
        true_lambda: u64,
    }

    let tests = vec![
        CarmichaelTest {
            n: 15,
            p: 3,
            q: 5,
            true_lambda: lcm(2, 4),
        }, // lcm(2,4) = 4
        CarmichaelTest {
            n: 35,
            p: 5,
            q: 7,
            true_lambda: lcm(4, 6),
        }, // lcm(4,6) = 12
        CarmichaelTest {
            n: 77,
            p: 7,
            q: 11,
            true_lambda: lcm(6, 10),
        }, // lcm(6,10) = 30
        CarmichaelTest {
            n: 221,
            p: 13,
            q: 17,
            true_lambda: lcm(12, 16),
        }, // lcm(12,16) = 48
        CarmichaelTest {
            n: 323,
            p: 17,
            q: 19,
            true_lambda: lcm(16, 18),
        }, // lcm(16,18) = 144
        CarmichaelTest {
            n: 899,
            p: 29,
            q: 31,
            true_lambda: lcm(28, 30),
        }, // lcm(28,30) = 420
        CarmichaelTest {
            n: 10403,
            p: 101,
            q: 103,
            true_lambda: lcm(100, 102),
        }, // lcm(100,102) = 5100
    ];

    for tc in &tests {
        let n = BigUint::from(tc.n);
        let approx = approximate_carmichael(&n, 500, 100_000);

        let phi = (tc.p - 1) * (tc.q - 1);

        let status = match approx {
            Some(val) if val == tc.true_lambda => "EXACT MATCH".to_string(),
            Some(val) if val % tc.true_lambda == 0 => {
                format!("multiple ({}x)", val / tc.true_lambda)
            }
            Some(val) if tc.true_lambda % val == 0 => {
                format!("divisor (1/{})", tc.true_lambda / val)
            }
            Some(val) => format!("approx={}", val),
            None => "FAILED".to_string(),
        };

        println!(
            "  n={:<6} ({} x {}): true lambda={:<5}, phi={:<6}, approx={:<10?}  [{}]",
            tc.n, tc.p, tc.q, tc.true_lambda, phi, approx, status
        );
    }
    println!();
}

// -------------------------------------------------------------------------
// Section 5 — Smooth Order Factoring
// -------------------------------------------------------------------------

fn section_5_smooth_order_factoring() {
    println!("--- Section 5: Smooth Order Factoring ---\n");

    let mut rng = rand::thread_rng();

    // 5a. Small known semiprimes
    println!("  5a. Small known semiprimes:");
    let small_cases: &[(u64, &str)] = &[
        (35, "5 x 7"),
        (77, "7 x 11"),
        (221, "13 x 17"),
        (323, "17 x 19"),
        (899, "29 x 31"),
        (10403, "101 x 103"),
    ];

    for &(n_val, desc) in small_cases {
        let n = BigUint::from(n_val);

        // First try to find a smooth-order element
        let smooth_elem = find_smooth_order_element(&n, 50, 500);
        let smooth_info = match &smooth_elem {
            Some((elem, ord)) => format!("a={}, ord={}", elem, ord),
            None => "none found".to_string(),
        };

        // Then try full factoring
        let factor = factor_via_smooth_orders(&n, 50, 500);
        let result_str = match &factor {
            Some(p) => {
                let q = &n / p;
                format!("SUCCESS: {} = {} x {}", n_val, p, q)
            }
            None => format!("FAILED for {}", n_val),
        };

        println!(
            "    n={:<6} ({}): smooth_elem=[{}] => {}",
            n_val, desc, smooth_info, result_str
        );
    }
    println!();

    // 5b. Random 16-bit semiprimes
    println!("  5b. Random 16-bit semiprimes:");
    let mut success_16 = 0;
    let total_16 = 5;
    for i in 0..total_16 {
        let target = generate_rsa_target(16, &mut rng);
        let factor = factor_via_smooth_orders(&target.n, 100, 1000);
        let status = match &factor {
            Some(p) => {
                let q = &target.n / p;
                if p * &q == target.n {
                    success_16 += 1;
                    format!("FACTORED: {} x {}", p, q)
                } else {
                    "WRONG FACTORS".to_string()
                }
            }
            None => "failed".to_string(),
        };
        println!(
            "    [{}] n={} ({}x{}): {}",
            i + 1,
            target.n,
            target.p,
            target.q,
            status
        );
    }
    println!("    16-bit success rate: {}/{}\n", success_16, total_16);

    // 5c. Random 24-bit semiprimes
    println!("  5c. Random 24-bit semiprimes:");
    let mut success_24 = 0;
    let total_24 = 5;
    for i in 0..total_24 {
        let target = generate_rsa_target(24, &mut rng);
        let factor = factor_via_smooth_orders(&target.n, 200, 2000);
        let status = match &factor {
            Some(p) => {
                let q = &target.n / p;
                if p * &q == target.n {
                    success_24 += 1;
                    format!("FACTORED: {} x {}", p, q)
                } else {
                    "WRONG FACTORS".to_string()
                }
            }
            None => "failed".to_string(),
        };
        println!(
            "    [{}] n={} ({}x{}): {}",
            i + 1,
            target.n,
            target.p,
            target.q,
            status
        );
    }
    println!("    24-bit success rate: {}/{}\n", success_24, total_24);
}

// -------------------------------------------------------------------------
// Section 6 — phi(n) Recovery Pipeline
// -------------------------------------------------------------------------

fn section_6_phi_recovery() {
    println!("--- Section 6: phi(n) Recovery Pipeline ---\n");

    let mut rng = rand::thread_rng();

    for bits in [16, 24, 32] {
        let target = generate_rsa_target(bits, &mut rng);
        let true_phi = euler_totient(&target.p, &target.q);
        println!(
            "  {}-bit RSA: N = {} (p={}, q={})",
            bits, target.n, target.p, target.q
        );
        println!("    True phi(N) = {}", true_phi);

        // Sample element orders
        let orders = sample_orders(&target.n, 500, 100_000);
        println!("    Sampled {} element orders", orders.len());

        if !orders.is_empty() {
            let max_ord = orders.iter().max().unwrap();
            let min_ord = orders.iter().min().unwrap();
            let avg_ord: f64 =
                orders.iter().map(|&o| o as f64).sum::<f64>() / orders.len() as f64;
            let unique_count = {
                let mut u = orders.clone();
                u.sort();
                u.dedup();
                u.len()
            };
            println!(
                "    Order stats: min={}, max={}, avg={:.1}, unique={}",
                min_ord, max_ord, avg_ord, unique_count
            );

            // Try to recover phi(n) from orders
            match phi_from_orders(&target.n, &orders) {
                Some(phi) => {
                    let phi_match = if phi == true_phi {
                        "exact match"
                    } else {
                        "candidate (may be multiple)"
                    };
                    println!("    Recovered phi(N) = {} [{}]", phi, phi_match);

                    match factor_from_phi(&target.n, &phi) {
                        Some((p, q)) => {
                            let verified = &p * &q == target.n;
                            println!(
                                "    FACTORED: {} = {} x {} (verified: {})",
                                target.n, p, q, verified
                            );
                        }
                        None => println!("    phi(N) candidate didn't yield factors"),
                    }
                }
                None => println!("    Could not recover phi(N) from sampled orders"),
            }
        }
        println!();
    }
}

// -------------------------------------------------------------------------
// Section 7 — Chebotarev Density Analysis (Experiment 5)
// -------------------------------------------------------------------------

fn section_7_chebotarev() {
    println!("--- Experiment 5: Chebotarev Density Analysis ---\n");

    let cheb_cases: &[(BigUint, &str)] = &[
        (BigUint::from(77u32), "7x11"),
        (BigUint::from(143u32), "11x13"),
        (BigUint::from(323u32), "17x19"),
        (BigUint::from(1007u32), "19x53"),
    ];

    for (n, label) in cheb_cases {
        println!("N = {} ({})", n, label);
        let measurements = chebotarev::chebotarev_scan(n, 20, 200);
        for m in &measurements {
            println!(
                "  r={:>2}: density={:.3} ({}/{}), class={:?}",
                m.r, m.empirical_density, m.num_divisible, m.num_tested, m.classification
            );
        }

        let constraints = chebotarev::extract_constraints(&measurements);
        println!("  Divides both: {:?}", constraints.divides_both);
        println!("  Divides one:  {:?}", constraints.divides_one);
        println!("  Divides neither: {:?}", constraints.divides_neither);

        if let Some(f) = chebotarev::factor_via_chebotarev(n, 20, 200) {
            let other = n / &f;
            println!("  FACTORED: {} x {}", f, other);
        } else {
            println!("  Not factored via Chebotarev alone");
        }
        println!();
    }
}

// -------------------------------------------------------------------------
// Utility
// -------------------------------------------------------------------------

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd_u64(a, b) * b
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
