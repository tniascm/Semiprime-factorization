use lattice_reduction::{
    basis_quality, ducas_verification, lll_reduce, schnorr_factoring_lattice, LllParams,
};
use num_bigint::BigUint;

fn main() {
    println!("=== Lattice Reduction for Factorization ===\n");

    // ─── Section 1: LLL reduction with basis quality metrics ────────────────────
    println!("=== 1. LLL Reduction with Basis Quality Metrics ===\n");

    let mut basis = vec![
        vec![1.0, 1.0, 1.0],
        vec![-1.0, 0.0, 2.0],
        vec![3.0, 5.0, 6.0],
    ];

    println!("  Original basis:");
    for (i, v) in basis.iter().enumerate() {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("    b{} = {:?}  (norm: {:.4})", i, v, norm);
    }

    let quality_before = basis_quality(&basis);
    println!("\n  Quality BEFORE reduction:");
    println!(
        "    Hermite factor:       {:.6}  (ideal: 1.0)",
        quality_before.hermite_factor
    );
    println!(
        "    Orthogonality defect: {:.6}  (ideal: 1.0)",
        quality_before.orthogonality_defect
    );
    println!(
        "    Shortest vector norm: {:.6}",
        quality_before.shortest_vector_norm
    );

    let params = LllParams::default();
    lll_reduce(&mut basis, &params);

    println!("\n  LLL-reduced basis (delta={:.2}):", params.delta);
    for (i, v) in basis.iter().enumerate() {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("    b{} = {:?}  (norm: {:.4})", i, v, norm);
    }

    let quality_after = basis_quality(&basis);
    println!("\n  Quality AFTER reduction:");
    println!(
        "    Hermite factor:       {:.6}  (ideal: 1.0)",
        quality_after.hermite_factor
    );
    println!(
        "    Orthogonality defect: {:.6}  (ideal: 1.0)",
        quality_after.orthogonality_defect
    );
    println!(
        "    Shortest vector norm: {:.6}",
        quality_after.shortest_vector_norm
    );

    let norm_improvement = if quality_before.shortest_vector_norm > 0.0 {
        (1.0 - quality_after.shortest_vector_norm / quality_before.shortest_vector_norm) * 100.0
    } else {
        0.0
    };
    let defect_improvement = if quality_before.orthogonality_defect > 0.0 {
        (1.0 - quality_after.orthogonality_defect / quality_before.orthogonality_defect) * 100.0
    } else {
        0.0
    };
    println!("\n  Improvement:");
    println!(
        "    Shortest vector: {:.1}% shorter",
        norm_improvement
    );
    println!(
        "    Orthogonality defect: {:.1}% closer to ideal",
        defect_improvement
    );

    // ─── Section 2: Schnorr factoring lattice construction ──────────────────────
    println!("\n=== 2. Schnorr Factoring Lattice Construction ===\n");

    let semiprimes: Vec<(u64, &str)> = vec![
        (1001, "7 x 11 x 13"),
        (15347, "103 x 149"),
        (8051, "83 x 97"),
    ];

    for (n_val, label) in &semiprimes {
        let n = BigUint::from(*n_val);
        let primes: Vec<u64> = vec![2, 3, 5, 7, 11, 13, 17, 19];
        let dimension = primes.len() + 1;

        let basis = schnorr_factoring_lattice(&n, dimension, &primes);

        println!("  N = {} ({})", n_val, label);
        println!(
            "    Factor base: {:?} ({} primes)",
            primes,
            primes.len()
        );
        println!(
            "    Lattice dimension: {} x {}",
            basis.len(),
            basis[0].len()
        );

        // Show norms of basis vectors
        let norms: Vec<f64> = basis
            .iter()
            .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        let min_norm = norms.iter().cloned().fold(f64::MAX, f64::min);
        let max_norm = norms.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "    Basis vector norms: min={:.2}, max={:.2}",
            min_norm, max_norm
        );

        // LLL-reduce and show improvement
        let mut reduced = basis.clone();
        lll_reduce(&mut reduced, &params);

        let reduced_norms: Vec<f64> = reduced
            .iter()
            .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        let reduced_min = reduced_norms.iter().cloned().fold(f64::MAX, f64::min);
        let reduced_max = reduced_norms.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "    After LLL: min_norm={:.2}, max_norm={:.2}",
            reduced_min, reduced_max
        );

        let quality = basis_quality(&reduced);
        println!(
            "    Hermite factor: {:.4}, Orthogonality defect: {:.4}",
            quality.hermite_factor, quality.orthogonality_defect
        );
        println!();
    }

    // ─── Section 3: Ducas verification experiment ───────────────────────────────
    println!("=== 3. Ducas Verification (Schnorr's Method Effectiveness) ===\n");
    println!(
        "  Ducas showed that Schnorr's lattice-based factoring approach yields"
    );
    println!("  essentially 0 out of 1000 useful relations. We replicate this.\n");

    let ducas_cases: Vec<(u64, &str, usize, usize)> = vec![
        (15347, "103 x 149", 10, 20),
        (67591, "257 x 263", 12, 20),
        (8051, "83 x 97", 10, 20),
    ];

    println!(
        "  {:>8}  {:>14}  {:>5}  {:>10}  {:>9}  {:>12}",
        "N", "Factors", "Dim", "Trials", "Successes", "Avg SVP Norm"
    );
    println!("  {}", "-".repeat(68));

    for (n_val, label, dim, trials) in &ducas_cases {
        let n = BigUint::from(*n_val);
        let result = ducas_verification(&n, *dim, *trials);

        println!(
            "  {:>8}  {:>14}  {:>5}  {:>10}  {:>6} ({:>4.1}%)  {:>12.2}",
            n_val,
            label,
            dim,
            result.num_trials,
            result.num_successes,
            result.success_rate * 100.0,
            result.avg_shortest_vector_norm
        );
    }

    println!();
    println!(
        "  Observation: Low success rates confirm Ducas' finding that Schnorr's"
    );
    println!(
        "  lattice approach struggles to produce useful factoring relations,"
    );
    println!(
        "  even for small semiprimes. The short vectors found after LLL reduction"
    );
    println!("  rarely correspond to valid smooth relations over the factor base.");
}
