//! Demo binary for the Class Number Oracle.
//!
//! Benchmarks H(D) algorithms, runs the Eichler-Selberg trace formula,
//! and analyzes discriminant structure for semiprimes N = pq.

use class_number_oracle::{class_number, discriminant_analysis, eichler_selberg};

fn main() {
    println!("=== Class Number Oracle ===\n");

    // 1. Benchmark H(D) methods on increasing |D|
    println!("--- H(D) Algorithm Comparison ---");
    let test_discriminants: Vec<i64> = vec![
        -3, -4, -7, -8, -11, -15, -20, -23, -24, -35, -40, -67, -100, -163, -1000, -5000, -10000,
    ];
    for &d in &test_discriminants {
        let bench = class_number::bench_class_number(d);
        println!(
            "  D = {:>6}: H(D) = {} (exact: {}us, shanks: {}us, analytic: {:.1} in {}us)",
            d,
            bench.h_exact,
            bench.time_exact_us,
            bench.time_shanks_us,
            bench.h_analytic,
            bench.time_analytic_us
        );
    }

    // 2. Hurwitz class numbers
    println!("\n--- Hurwitz Class Numbers ---");
    for &d in &[-3i64, -4, -7, -8, -11, -12, -15, -16, -20] {
        let h = class_number::hurwitz_class_number(d);
        let h_plain = class_number::class_number_exact(d);
        println!(
            "  D = {:>4}: h(D) = {}, H(D) = {:.4}",
            d, h_plain, h
        );
    }

    // 3. Eichler-Selberg trace formula
    println!("\n--- Eichler-Selberg Trace Formula ---");
    let test_cases: Vec<(u64, u64, u64)> = vec![
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
    ];
    for &(n, _p, _q) in &test_cases {
        for &l in &[2u64, 3, 5, 7] {
            if n % l != 0 {
                let result = eichler_selberg::eichler_selberg_trace(n, l);
                println!(
                    "  N={}, l={}: tr(T_{}) = {:.2} ({} terms, {}us)",
                    n, l, l, result.trace, result.num_terms, result.time_us
                );
            }
        }
    }

    // 4. Complexity analysis
    println!("\n--- Trace Formula Complexity ---");
    for &(n, _p, _q) in &test_cases {
        for &l in &[2u64, 5, 11, 23, 47, 97] {
            if n % l != 0 {
                let complexity = eichler_selberg::analyze_complexity(n, l);
                println!(
                    "  N={}, l={}: {} terms, max|D|={}, naive_ops~{:.0}, shanks_ops~{:.0}",
                    n,
                    l,
                    complexity.num_main_terms,
                    complexity.max_abs_d,
                    complexity.naive_total_ops_estimate,
                    complexity.shanks_total_ops_estimate
                );
            }
        }
    }

    // 5. Discriminant analysis
    println!("\n--- Discriminant Structure Analysis ---");
    for &(n, p, q) in &test_cases {
        let analysis = discriminant_analysis::analyze_discriminants(n, p, q, 2);
        let sp = &analysis.splitting_pattern;
        println!(
            "  N={} ({}x{}), l=2: {} discriminants",
            n, p, q, sp.total
        );
        println!(
            "    Split both: {}, p-only: {}, q-only: {}, neither: {}",
            sp.split_both, sp.split_p_only, sp.split_q_only, sp.split_neither
        );
        println!(
            "    H sums: both={:.1}, p={:.1}, q={:.1}, neither={:.1}",
            sp.h_sum_split_both,
            sp.h_sum_split_p,
            sp.h_sum_split_q,
            sp.h_sum_split_neither
        );
    }

    // 6. Kronecker distribution
    println!("\n--- Kronecker Symbol Distribution ---");
    for &(n, p, q) in &test_cases {
        for &l in &[2u64, 5, 11] {
            if n % l != 0 {
                let dist = discriminant_analysis::kronecker_distribution(n, p, q, l);
                println!(
                    "  N={}, l={}: (D|{})=[+{} -{} 0:{}], (D|{})=[+{} -{} 0:{}], corr={:.3}",
                    n,
                    l,
                    p,
                    dist.count_split_p,
                    dist.count_inert_p,
                    dist.count_ramified_p,
                    q,
                    dist.count_split_q,
                    dist.count_inert_q,
                    dist.count_ramified_q,
                    dist.kronecker_correlation
                );
            }
        }
    }

    // 7. Multiplicativity check
    println!("\n--- Class Number Multiplicativity Test ---");
    for &(n, p, q) in &test_cases {
        let result = discriminant_analysis::check_class_number_multiplicativity(n, p, q);
        println!(
            "  N={} ({}x{}): {} data points, avg_err={:.4}, max_err={:.4}, multiplicative={}",
            n,
            p,
            q,
            result.data_points.len(),
            result.avg_relative_error,
            result.max_relative_error,
            result.suggests_multiplicativity
        );
    }
}
