//! Demo binary for cf-factor: integer factoring via continued fractions.

use num_bigint::BigUint;
use std::time::Instant;

fn main() {
    println!("=== cf-factor: Integer Factoring via Continued Fractions ===\n");

    // Demo 1: Continued fraction expansion
    println!("--- Continued Fraction Expansion of sqrt(7) ---");
    let n = BigUint::from(7u32);
    let (a0, periodic) = cf_factor::cf::cf_periodic_part(&n, 20);
    let periodic_strs: Vec<String> = periodic.iter().map(|x| x.to_string()).collect();
    println!("sqrt(7) = [{};  {}]", a0, periodic_strs.join(", "));
    println!();

    // Demo 2: Regulator computation
    println!("--- Regulator of Q(sqrt(7)) ---");
    let reg = cf_factor::regulator::compute_regulator(&n, 100);
    println!("Period: {:?}", reg.period);
    println!("Regulator estimate: {:.6}", reg.regulator_estimate);
    if let Some((x, y)) = &reg.pell_solution {
        println!("Pell solution: {}^2 - 7*{}^2 = {}", x, y, {
            let x2: BigUint = x * x;
            let ny2: BigUint = &n * y * y;
            if x2 >= ny2 {
                format!("{}", x2 - ny2)
            } else {
                format!("-{}", ny2 - x2)
            }
        });
    }
    println!();

    // Demo 3: SQUFOF factoring
    println!("--- SQUFOF Factoring ---");
    let test_numbers: Vec<u64> = vec![
        15, 77, 143, 221, 323, 667, 1073, 10403, 2820669811,
    ];

    for n_val in &test_numbers {
        let n = BigUint::from(*n_val);
        let start = Instant::now();
        let result = cf_factor::factor::factor_squfof(&n);
        let elapsed = start.elapsed();

        match &result.factor {
            Some(f) => {
                let cofactor = &n / f;
                println!(
                    "  {} = {} x {} (method: {}, cf_terms: {}, time: {:?})",
                    n_val, f, cofactor, result.method, result.cf_terms, elapsed
                );
            }
            None => {
                println!(
                    "  {} = FAILED (method: {}, time: {:?})",
                    n_val, result.method, elapsed
                );
            }
        }
    }
    println!();

    // Demo 4: Full pipeline
    println!("--- Full Factoring Pipeline ---");
    let pipeline_numbers: Vec<u64> = vec![
        60,              // 2^2 * 3 * 5
        10403,           // 101 * 103
        2820669811,      // 57719 * 48869
    ];

    for n_val in &pipeline_numbers {
        let n = BigUint::from(*n_val);
        let start = Instant::now();
        let result = cf_factor::factor::factor_cf(&n);
        let elapsed = start.elapsed();

        match &result.factor {
            Some(f) => {
                let cofactor = &n / f;
                println!(
                    "  {} = {} x {} (method: {}, cf_terms: {}, forms: {}, time: {:?})",
                    n_val, f, cofactor, result.method, result.cf_terms, result.forms_explored, elapsed
                );
            }
            None => {
                println!(
                    "  {} = FAILED (method: {}, time: {:?})",
                    n_val, result.method, elapsed
                );
            }
        }
    }
    println!();

    // Demo 5: Full factorization
    println!("--- Full Factorization ---");
    let n = BigUint::from(2u32 * 2 * 3 * 5 * 7 * 11 * 13);
    let start = Instant::now();
    let factors = cf_factor::factor::full_factorization(&n);
    let elapsed = start.elapsed();
    let factor_strs: Vec<String> = factors.iter().map(|f| f.to_string()).collect();
    println!(
        "  {} = {} (time: {:?})",
        n,
        factor_strs.join(" x "),
        elapsed
    );

    println!("\n=== Done ===");
}
