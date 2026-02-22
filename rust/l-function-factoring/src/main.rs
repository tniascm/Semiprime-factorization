//! Demo program for L-function factoring methods.

use l_function_factoring::class_number;
use l_function_factoring::factor;
use l_function_factoring::gauss_sums;
use l_function_factoring::sampling;

fn main() {
    println!("========================================");
    println!("  L-function Factoring");
    println!("========================================");
    println!();

    let test_numbers: Vec<(u64, u64, u64)> = vec![
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (1003, 17, 59),
    ];

    // --- Method 1: Conductor Detection ---
    println!("--- Method 1: Conductor Detection ---");
    println!("For N = pq, find characters with conductor < N");
    println!();

    for &(n, expected_p, expected_q) in &test_numbers {
        print!("  Testing N = {} ({} x {}): ", n, expected_p, expected_q);

        let result = factor::factor_by_conductor(n);
        match result.factors {
            Some((a, b)) => {
                let (p, q) = if a < b { (a, b) } else { (b, a) };
                println!("Found factors {} x {} [OK]", p, q);
            }
            None => println!("[FAILED]"),
        }
    }
    println!();

    // --- Method 2: Class Number Analysis ---
    println!("--- Method 2: Class Number Analysis ---");
    println!("h(-4N) via L(1, chi_{{-4N}})");
    println!();

    for &(n, expected_p, expected_q) in &test_numbers[..3].iter().copied().collect::<Vec<_>>() {
        let d = -(4 * n as i64);
        let h = class_number::class_number_via_kronecker(d, 50000);
        println!(
            "  N = {} ({} x {}): h(-4*{}) = {:.2} (rounded: {})",
            n,
            expected_p,
            expected_q,
            n,
            h,
            h.round() as u64
        );

        let result = factor::factor_by_class_number(n);
        match result.factors {
            Some((a, b)) => {
                let (p, q) = if a < b { (a, b) } else { (b, a) };
                println!("    -> Factors: {} x {} [OK]", p, q);
            }
            None => println!("    -> [FAILED]"),
        }
    }
    println!();

    // --- Method 3: L-function Decomposition ---
    println!("--- Method 3: L-function Decomposition ---");
    println!("Detect multiplicative factorization of L-functions");
    println!();

    for &(n, expected_p, expected_q) in &test_numbers[..3].iter().copied().collect::<Vec<_>>() {
        print!("  Testing N = {} ({} x {}): ", n, expected_p, expected_q);

        let result = factor::factor_by_l_function_decomposition(n);
        match result.factors {
            Some((a, b)) => {
                let (p, q) = if a < b { (a, b) } else { (b, a) };
                println!("Found factors {} x {} [OK]", p, q);
            }
            None => println!("[FAILED]"),
        }
    }
    println!();

    // --- Conductor Analysis Detail ---
    println!("--- Conductor Analysis for N = 77 ---");
    let analyses = gauss_sums::analyze_conductors(77);
    let mut conductor_counts: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::new();
    for a in &analyses {
        *conductor_counts.entry(a.conductor_direct).or_insert(0) += 1;
    }
    let mut sorted: Vec<_> = conductor_counts.iter().collect();
    sorted.sort_by_key(|&(c, _)| *c);
    for (cond, count) in sorted {
        println!("  Conductor {:>3}: {} characters", cond, count);
    }
    println!();

    // --- Class Number Verification ---
    println!("--- Class Number Verification ---");
    println!("Comparing formula vs direct computation for small |D|:");
    for &d in &[-3i64, -4, -7, -8, -11, -15, -20, -23, -24, -35] {
        let h_direct = class_number::class_number_direct(d);
        let h_formula = class_number::class_number_via_kronecker(d, 50000);
        println!(
            "  h({:>4}) = {:>2} (direct)  {:.3} (formula)  {}",
            d,
            h_direct,
            h_formula,
            if (h_formula - h_direct as f64).abs() < 0.5 {
                "OK"
            } else {
                "MISMATCH"
            }
        );
    }
    println!();

    // --- Full Factoring Pipeline ---
    println!("--- Full Factoring Pipeline ---");
    for &(n, _, _) in &test_numbers {
        let result = factor::factor(n);
        match result.factors {
            Some((a, b)) => {
                let (p, q) = if a < b { (a, b) } else { (b, a) };
                println!(
                    "  {} = {} x {} [via {:?}]",
                    n, p, q, result.method
                );
            }
            None => println!("  {} = ??? [FAILED]", n),
        }
    }
    println!();

    // --- Experiment 2: Sublinear Conductor Detection ---
    println!("--- Experiment 2: Sublinear Conductor Detection ---");
    println!("Testing sampling methods to find conductor-revealing characters");
    println!();

    let test_semiprimes: Vec<(u64, &str)> = vec![
        (77, "7x11"),
        (143, "11x13"),
        (323, "17x19"),
        (1007, "19x53"),
        (3599, "59x61"),
        (10403, "101x103"),
    ];

    for &(n, label) in &test_semiprimes {
        println!("N = {} ({})", n, label);
        let results = sampling::run_all_methods(n, 10000);
        for r in &results {
            let status = if r.factor_found.is_some() {
                " OK "
            } else {
                "FAIL"
            };
            println!(
                "  {:20}: [{}] {} samples, {}us",
                r.method, status, r.samples_tested, r.time_us
            );
        }
        println!();
    }
}
