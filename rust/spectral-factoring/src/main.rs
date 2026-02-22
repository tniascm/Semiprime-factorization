use spectral_factoring::dimension::{dim_s2, dim_s2_new, dim_s2_old, factor_u64};
use spectral_factoring::spectral::{detailed_spectral_analysis, factor_from_spectral};

fn main() {
    println!("========================================");
    println!("  Spectral Factoring via Hecke Eigenvalues");
    println!("========================================");
    println!();
    println!("This implements a factoring method based on the Langlands-inspired insight");
    println!("that the old/new decomposition of modular forms for Gamma_0(N) reveals");
    println!("the factorization of N.");
    println!();
    println!("For N = pq (product of two distinct primes):");
    println!("  S_2(Gamma_0(pq)) = S_2^old + S_2^new");
    println!("  S_2^old = S_2(Gamma_0(p))^2 + S_2(Gamma_0(q))^2");
    println!("  This decomposition reveals p and q!");
    println!();

    // Test semiprimes
    let test_cases: Vec<(u64, u64, u64)> = vec![
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (437, 19, 23),
        (667, 23, 29),
    ];

    // Print dimension table
    println!("--- Dimension Table ---");
    println!("{:>6} {:>6} {:>8} {:>8} {:>8} {:>12}",
        "N", "p*q", "dim(N)", "new(N)", "old(N)", "2d(p)+2d(q)");
    println!("{:-<6} {:-<6} {:-<8} {:-<8} {:-<8} {:-<12}", "", "", "", "", "", "");

    for &(n, p, q) in &test_cases {
        let total = dim_s2(n);
        let new = dim_s2_new(n);
        let old = dim_s2_old(n);
        let expected_old = 2 * dim_s2(p) + 2 * dim_s2(q);
        let check = if old == expected_old { "OK" } else { "MISMATCH" };
        println!("{:>6} {:>2}*{:<3} {:>8} {:>8} {:>8} {:>12} {}",
            n, p, q, total, new, old, expected_old, check);
    }
    println!();

    // Detailed analysis for each
    println!("--- Detailed Spectral Analysis ---");
    println!();

    for &(n, _p, _q) in &test_cases {
        let result = detailed_spectral_analysis(n);

        println!("N = {} (factorization: {:?})", n, factor_u64(n));
        println!("  Result: {:?}", result.factors);
        for detail in &result.details {
            println!("  {}", detail);
        }
        println!();
    }

    // Summary
    println!("--- Factoring Summary ---");
    println!();

    let mut all_correct = true;
    for &(n, p, q) in &test_cases {
        let result = factor_from_spectral(n);
        let status = match result {
            Some((a, b)) if (a == p && b == q) || (a == q && b == p) => "CORRECT",
            Some((a, b)) => {
                all_correct = false;
                &format!("WRONG (got {} x {})", a, b).leak()
            }
            None => {
                all_correct = false;
                "FAILED"
            }
        };
        println!("  {} = {} x {} ... {}", n, p, q, status);
    }

    println!();
    if all_correct {
        println!("All factorizations correct!");
    } else {
        println!("Some factorizations failed.");
    }

    // Larger examples
    println!();
    println!("--- Larger Semiprimes ---");
    let larger_cases: Vec<u64> = vec![
        899,   // 29 * 31
        1517,  // 37 * 41
        2021,  // 43 * 47
        3127,  // 53 * 59
        4087,  // 61 * 67
    ];

    for n in &larger_cases {
        let total = dim_s2(*n);
        let new = dim_s2_new(*n);
        let old = dim_s2_old(*n);
        let result = factor_from_spectral(*n);
        match result {
            Some((p, q)) => {
                let expected_old = 2 * dim_s2(p) + 2 * dim_s2(q);
                let check = if old == expected_old { "dims match" } else { "dims differ" };
                println!("  {} = {} x {} [dim={}, new={}, old={}, expected_old={}, {}]",
                    n, p, q, total, new, old, expected_old, check);
            }
            None => {
                println!("  {} = ??? [dim={}, new={}, old={}]", n, total, new, old);
            }
        }
    }
}
