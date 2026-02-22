use num_bigint::BigUint;
use trace_lattice::{factor, trace};

fn main() {
    println!("=== Trace Formula Lattice Attack ===\n");

    let test_cases = [
        (77u64, 7u64, 11u64),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (437, 19, 23),
        (667, 23, 29),
        (899, 29, 31),
        (1517, 37, 41),
        (3599, 59, 61),
        (10403, 101, 103),
    ];

    println!("--- Dimension-Based Factoring (u64) ---\n");

    for &(n, p, q) in &test_cases {
        let result = factor::factor_trace_lattice(n);
        let status = if result.factors.is_some() {
            "OK"
        } else {
            "FAIL"
        };
        print!("  N = {:>6} ({}x{}): [{}]", n, p, q, status);
        if let Some((a, b)) = result.factors {
            print!(" => {} x {}", a, b);
        }
        println!(
            "  (dim_total={}, dim_new={}, dim_old={})",
            result
                .trace_constraints
                .first()
                .map(|c| c.dim_total)
                .unwrap_or(0),
            result
                .trace_constraints
                .first()
                .map(|c| c.dim_new)
                .unwrap_or(0),
            result
                .trace_constraints
                .first()
                .map(|c| c.dim_old)
                .unwrap_or(0)
        );
    }

    println!("\n--- BigUint Dimension-Based Factoring ---\n");

    for &(n, p, q) in &test_cases {
        let big_n = BigUint::from(n);
        let result = factor::factor_trace_lattice_big(&big_n);
        let status = if result.factors.is_some() {
            "OK"
        } else {
            "FAIL"
        };
        print!("  N = {:>6} ({}x{}): [{}]", n, p, q, status);
        if let Some((a, b)) = &result.factors {
            print!(" => {} x {}", a, b);
        }
        println!(" [{}]", result.method);
    }

    println!("\n--- BigUint Factoring: Larger Semiprimes ---\n");

    // Balanced semiprimes of increasing size
    let big_cases: Vec<(BigUint, BigUint, &str)> = vec![
        (
            BigUint::from(16777259u64),
            BigUint::from(16777289u64),
            "~48-bit",
        ),
        (
            BigUint::from(67108879u64),
            BigUint::from(67108913u64),
            "~52-bit",
        ),
        (
            BigUint::from(268435459u64),
            BigUint::from(268435463u64),
            "~56-bit",
        ),
        (
            BigUint::from(1073741827u64),
            BigUint::from(1073741831u64),
            "~60-bit",
        ),
        (
            BigUint::from(4294967311u64),
            BigUint::from(4294967357u64),
            "~64-bit",
        ),
    ];

    for (p, q, label) in &big_cases {
        let n = p * q;
        let start = std::time::Instant::now();
        let result = factor::factor_trace_lattice_big(&n);
        let elapsed = start.elapsed();

        let status = if result.factors.is_some() {
            "OK"
        } else {
            "FAIL"
        };
        print!("  {} N = {} [{}]", label, n, status);
        if let Some((a, b)) = &result.factors {
            print!(" => {} x {}", a, b);
        }
        println!(" ({:.2?}) [{}]", elapsed, result.method);
    }

    println!("\n--- Trace Structure Analysis ---\n");

    for &(_n, p, q) in &test_cases[..4] {
        let lines = factor::analyze_trace_structure(p, q);
        for line in &lines {
            println!("  {}", line);
        }
        println!();
    }

    // Verify known dimension values
    println!("--- Dimension Formula Verification ---\n");

    println!("  Comparing u64 vs BigUint dimension formulas:");
    let known_dims = [
        (11, 1),
        (23, 2),
        (29, 2),
        (37, 2),
        (41, 3),
        (43, 3),
        (53, 4),
        (61, 4),
        (67, 5),
        (71, 6),
        (77, 7),
    ];
    for &(n, expected) in &known_dims {
        let u64_dim = trace::dim_s2(n);
        let big_dim = trace::dim_s2_big(&BigUint::from(n));
        let u64_ok = u64_dim == expected;
        let big_ok = big_dim == BigUint::from(expected);
        let status = if u64_ok && big_ok {
            "OK"
        } else {
            "MISMATCH"
        };
        println!(
            "  dim S_2(Gamma_0({})) = {} (u64) / {} (big) expected {} [{}]",
            n, u64_dim, big_dim, expected, status
        );
    }

    println!("\n--- Spectral Decomposition Verification ---\n");

    for &(n, p, q) in &test_cases[..4] {
        let big_n = BigUint::from(n);
        let big_p = BigUint::from(p);
        let big_q = BigUint::from(q);
        let valid = factor::verify_spectral_decomposition(&big_n, &big_p, &big_q);
        let lines = factor::analyze_trace_structure_big(&big_p, &big_q);
        for line in &lines {
            println!("  {}", line);
        }
        println!("  Spectral verification: {}", if valid { "PASS" } else { "FAIL" });
        println!();
    }
}
