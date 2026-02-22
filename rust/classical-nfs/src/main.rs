use classical_nfs::{nfs_factor, quadratic_sieve, select_polynomial, NfsParams};
use num_bigint::BigUint;
use num_traits::Zero;
use std::time::Instant;

/// A semiprime test target with known factorization.
struct Semiprime {
    n: u64,
    p: u64,
    q: u64,
}

fn main() {
    println!("=== Classical Factoring: Quadratic Sieve vs Number Field Sieve ===\n");

    let targets = vec![
        Semiprime { n: 8051, p: 83, q: 97 },
        Semiprime { n: 15347, p: 103, q: 149 },
        Semiprime { n: 67591, p: 257, q: 263 },
        Semiprime { n: 1042961, p: 1009, q: 1033 },
    ];

    // ── Section 1: Polynomial Selection ──────────────────────────────────────
    println!("=== 1. NFS Polynomial Selection (base-m, degree 3) ===\n");

    for target in &targets {
        let n = BigUint::from(target.n);
        let poly = select_polynomial(&n, 3);
        println!(
            "  n = {:>10} = {} x {}",
            target.n, target.p, target.q
        );
        println!(
            "    m = {}, degree = {}, coeffs = {:?}",
            poly.m, poly.degree, poly.coefficients
        );

        // Verify f(m) = n
        let m_i64 = poly.m.to_string().parse::<i64>().unwrap_or(0);
        let eval_at_m = classical_nfs::eval_polynomial(&poly, m_i64);
        let valid = eval_at_m == n;
        println!(
            "    f(m) = {} {}",
            eval_at_m,
            if valid { "[OK: f(m) = n]" } else { "[ERROR: f(m) != n]" }
        );
        println!();
    }

    // ── Section 2: Head-to-Head QS vs NFS ────────────────────────────────────
    println!("=== 2. Head-to-Head: QS vs NFS ===\n");

    println!(
        "  {:>10}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}",
        "N", "Expected", "QS Time", "QS Result", "NFS Time", "NFS Result"
    );
    println!("  {}", "-".repeat(72));

    for target in &targets {
        let n = BigUint::from(target.n);
        let expected = format!("{}x{}", target.p, target.q);

        // Quadratic Sieve
        let start = Instant::now();
        let qs_result = quadratic_sieve(&n, 200);
        let qs_time = start.elapsed();

        let qs_str = match &qs_result {
            Some(f) => {
                let cofactor = &n / f;
                if *f != BigUint::from(1u64) && cofactor != BigUint::from(1u64) && (&n % f) == BigUint::zero() {
                    format!("{}x{}", f, cofactor)
                } else {
                    "trivial".to_string()
                }
            }
            None => "FAIL".to_string(),
        };

        // Number Field Sieve
        let params = NfsParams {
            factor_base_bound: 200,
            sieve_range: 1000,
            num_threads: 4,
        };
        let start = Instant::now();
        let nfs_result = nfs_factor(&n, &params);
        let nfs_time = start.elapsed();

        let nfs_str = match &nfs_result {
            Some(f) => {
                let cofactor = &n / f;
                if *f != BigUint::from(1u64) && cofactor != BigUint::from(1u64) && (&n % f) == BigUint::zero() {
                    format!("{}x{}", f, cofactor)
                } else {
                    "trivial".to_string()
                }
            }
            None => "FAIL".to_string(),
        };

        println!(
            "  {:>10}  {:>14}  {:>7.3}ms  {:>10}  {:>7.3}ms  {:>10}",
            target.n,
            expected,
            qs_time.as_secs_f64() * 1000.0,
            qs_str,
            nfs_time.as_secs_f64() * 1000.0,
            nfs_str,
        );
    }

    // ── Section 3: Verification ──────────────────────────────────────────────
    println!("\n=== 3. Verification ===\n");

    let mut pass = 0;
    let mut fail = 0;

    for target in &targets {
        let n = BigUint::from(target.n);
        let params = NfsParams {
            factor_base_bound: 200,
            sieve_range: 1000,
            num_threads: 4,
        };
        let result = nfs_factor(&n, &params);
        match result {
            Some(ref f) => {
                let cofactor = &n / f;
                let valid = (f * &cofactor) == n
                    && *f > BigUint::from(1u64)
                    && cofactor > BigUint::from(1u64);
                if valid {
                    println!("  [PASS] {} = {} x {}", target.n, f, cofactor);
                    pass += 1;
                } else {
                    println!("  [FAIL] {} -- trivial factorization", target.n);
                    fail += 1;
                }
            }
            None => {
                println!("  [FAIL] {} -- no factor found", target.n);
                fail += 1;
            }
        }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    println!("=== Done ===");
}
