use num_bigint::BigUint;
use quantum_inspired::{
    cayley_graph_walk, continued_fraction, convergents, grover_speedup_estimate, period_factor,
    SampleQueryOracle,
};
use rand::thread_rng;

fn main() {
    println!("=== Quantum-Inspired Classical Factoring ===\n");

    // ─── Section 1: Period-based factoring (Shor-style, classical simulation) ───
    println!("=== 1. Period-Based Factoring (Classical Shor Simulation) ===\n");

    let test_cases: Vec<(u64, &str)> = vec![
        (15, "3 x 5"),
        (21, "3 x 7"),
        (35, "5 x 7"),
        (8051, "83 x 97"),
        (1_000_003 * 17, "17 x 1000003"),
    ];

    for (n, expected) in &test_cases {
        let n_big = BigUint::from(*n);
        print!("  Factoring {:>12} (expected: {:>16}) => ", n, expected);
        match period_factor(&n_big, 100) {
            Some(f) => {
                let cofactor = &n_big / &f;
                println!("{} x {}", f, cofactor);
            }
            None => println!("Failed (period too large for classical search)"),
        }
    }

    // ─── Section 2: Continued fractions and convergents ─────────────────────────
    println!("\n=== 2. Continued Fractions & Convergents ===\n");

    let sample_fractions: Vec<(u64, u64, &str)> = vec![
        (31, 13, "31/13"),
        (355, 113, "355/113 (approx pi)"),
        (89, 55, "89/55 (Fibonacci ratio, near golden ratio)"),
        (649, 200, "649/200"),
    ];

    for (num, den, label) in &sample_fractions {
        let cf = continued_fraction(*num, *den, 20);
        let convs = convergents(&cf);

        println!("  {} = {}/{}", label, num, den);
        println!("    Continued fraction: {:?}", cf);
        println!("    Convergents (best rational approximations):");
        for (i, (p, q)) in convs.iter().enumerate() {
            let approx = *p as f64 / *q as f64;
            let exact = *num as f64 / *den as f64;
            let error = (approx - exact).abs();
            println!(
                "      k={}: {}/{} = {:.10}  (error: {:.2e})",
                i, p, q, approx, error
            );
        }
        println!();
    }

    // ─── Section 3: Grover speedup estimates ────────────────────────────────────
    println!("=== 3. Grover Speedup Estimates ===\n");
    println!(
        "  {:>6}  {:>18}  {:>18}  {:>10}",
        "Bits", "Classical Evals", "Quantum Evals", "Speedup"
    );
    println!("  {}", "-".repeat(58));

    for bits in [10, 20, 30, 40, 50, 64] {
        // Success probability = 1/2^bits (one needle in the haystack)
        let p = 1.0 / (2.0_f64.powi(bits));
        let est = grover_speedup_estimate(bits as u32, p);
        println!(
            "  {:>6}  {:>18.2e}  {:>18.2e}  {:>10.1}x",
            bits, est.classical_evals, est.quantum_evals, est.speedup_factor
        );
    }
    println!();
    println!("  Grover provides a quadratic speedup: O(sqrt(N)) vs O(N).");
    println!("  Speedup factor = (4/pi) * sqrt(1/p) for success probability p.");

    // ─── Section 4: Cayley graph walk on (Z/15Z)* ──────────────────────────────
    println!("\n=== 4. Cayley Graph Walk on (Z/15Z)* ===\n");

    let n = BigUint::from(15u32);
    let generators = vec![BigUint::from(2u32), BigUint::from(4u32)];
    let steps = 10_000;

    println!("  Group: (Z/15Z)* = {{1, 2, 4, 7, 8, 11, 13, 14}}  (order 8)");
    println!("  Generators: {{2, 4}}");
    println!("  Random walk steps: {}\n", steps);

    let result = cayley_graph_walk(&n, &generators, steps);

    println!(
        "  {:>6}  {:>8}  {:>10}",
        "Node", "Visits", "Frequency"
    );
    println!("  {}", "-".repeat(28));

    let total_visits: u64 = result.iter().map(|(_, c)| c).sum();
    for (node, count) in &result {
        let freq = *count as f64 / total_visits as f64;
        println!("  {:>6}  {:>8}  {:>9.2}%", node, count, freq * 100.0);
    }
    println!();
    println!(
        "  Uniform distribution expected: {:.2}% per node",
        100.0 / 8.0
    );
    println!("  (Z/15Z)* has 8 elements coprime to 15.");
    println!("  Quantum walks on Cayley graphs can reveal hidden subgroup");
    println!("  structure exponentially faster than classical random walks.");

    // ─── Section 5: Sample-and-Query Oracle (Tang dequantization) ───────────────
    println!("\n=== 5. Sample-and-Query Oracle (Tang Dequantization) ===\n");

    let matrix = vec![
        vec![3.0, 1.0, 0.0, 0.5],
        vec![0.0, 4.0, 1.0, 0.0],
        vec![1.0, 0.0, 2.0, 0.0],
        vec![0.0, 0.0, 0.5, 1.0],
    ];

    let oracle = SampleQueryOracle::new(matrix.clone());

    println!("  Input matrix (4x4):");
    for (i, row) in matrix.iter().enumerate() {
        println!("    Row {}: {:?}", i, row);
    }

    // Show row norms and sampling probabilities
    println!("\n  Row norms and sampling probabilities:");
    let row_norms_sq: Vec<f64> = matrix
        .iter()
        .map(|row| row.iter().map(|v| v * v).sum())
        .collect();
    let total: f64 = row_norms_sq.iter().sum();
    for (i, &ns) in row_norms_sq.iter().enumerate() {
        println!(
            "    Row {}: ||row||^2 = {:.2}, sampling prob = {:.2}%",
            i,
            ns,
            ns / total * 100.0
        );
    }
    println!("    Total Frobenius norm^2 = {:.2}", total);

    // Demonstrate sampling
    let mut rng = thread_rng();
    let num_samples = 10_000;
    let mut counts = vec![0u32; matrix.len()];
    for _ in 0..num_samples {
        let idx = oracle.sample_row(&mut rng);
        counts[idx] += 1;
    }
    println!("\n  Empirical sampling over {} draws:", num_samples);
    for (i, &c) in counts.iter().enumerate() {
        println!(
            "    Row {}: {} times ({:.1}%)",
            i,
            c,
            c as f64 / num_samples as f64 * 100.0
        );
    }

    // Demonstrate entry query
    println!("\n  Entry queries:");
    println!("    M[0][0] = {}", oracle.query_entry(0, 0));
    println!("    M[1][1] = {}", oracle.query_entry(1, 1));
    println!("    M[2][2] = {}", oracle.query_entry(2, 2));
    println!("    M[0][3] = {}", oracle.query_entry(0, 3));

    // Low-rank approximation
    println!("\n  Low-rank approximation (rank=2, 200 samples):");
    let approx = oracle.low_rank_approx(2, 200, &mut rng);
    let mut total_error = 0.0;
    for (i, (orig_row, approx_row)) in matrix.iter().zip(approx.iter()).enumerate() {
        let row_error: f64 = orig_row
            .iter()
            .zip(approx_row.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        total_error += row_error * row_error;
        let approx_strs: Vec<String> = approx_row.iter().map(|v| format!("{:>6.2}", v)).collect();
        println!("    Row {}: [{}]  (row error: {:.4})", i, approx_strs.join(", "), row_error);
    }
    println!(
        "    Total Frobenius error: {:.4}",
        total_error.sqrt()
    );
    println!();
    println!("  Tang's dequantization: if classical algorithms have sample-and-query");
    println!("  access to input data, many quantum ML speedups can be matched classically.");
}
