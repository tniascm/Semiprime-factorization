use ising_factoring::{
    encode_factoring_qubo, evaluate_qubo, extract_factors, parallel_tempering,
};

fn main() {
    println!("=== Ising/QUBO Factorization ===\n");

    // ─── Section 1: Small numbers (4-bit factors) ───────────────────────────────
    println!("=== 1. Small Semiprimes (4-bit factors) ===\n");

    let small_numbers: Vec<(u64, &str)> = vec![
        (15, "3 x 5"),
        (21, "3 x 7"),
        (35, "5 x 7"),
        (77, "7 x 11"),
    ];

    let p_bits_small = 4;
    let q_bits_small = 4;

    println!(
        "  {:>4}  {:>10}  {:>10}  {:>6}  {:>6}  {:>8}  {:>10}  {:>7}",
        "N", "Expected", "Found", "p", "q", "Energy", "QUBO Size", "Status"
    );
    println!("  {}", "-".repeat(72));

    for (n, expected) in &small_numbers {
        let qubo = encode_factoring_qubo(*n, p_bits_small, q_bits_small);
        let qubo_size = qubo.size;

        // Try multiple attempts for reliability
        let mut best_p = 0u64;
        let mut best_q = 0u64;
        let mut best_energy = f64::MAX;
        let mut success = false;

        for _ in 0..5 {
            let (solution, energy) = parallel_tempering(&qubo, 8, 50_000);
            let (p, q) = extract_factors(&solution, p_bits_small, q_bits_small);
            if energy < best_energy {
                best_energy = energy;
                best_p = p;
                best_q = q;
            }
            if p > 1 && q > 1 && p * q == *n {
                best_p = p;
                best_q = q;
                best_energy = energy;
                success = true;
                break;
            }
        }

        let found_str = if success {
            format!("{} x {}", best_p, best_q)
        } else {
            format!("{} x {}", best_p, best_q)
        };

        println!(
            "  {:>4}  {:>10}  {:>10}  {:>6}  {:>6}  {:>8.1}  {:>10}  {:>7}",
            n,
            expected,
            found_str,
            best_p,
            best_q,
            best_energy,
            qubo_size,
            if success { "OK" } else { "MISS" }
        );
    }

    // ─── Section 2: QUBO matrix size analysis ───────────────────────────────────
    println!("\n=== 2. QUBO Matrix Size Analysis ===\n");
    println!(
        "  Variables = p_bits + q_bits + z_vars (p*q product bits) + carry_vars\n"
    );

    println!(
        "  {:>6}  {:>6}  {:>12}  {:>18}",
        "p_bits", "q_bits", "Total Vars", "Matrix Entries"
    );
    println!("  {}", "-".repeat(48));

    for (pb, qb) in [(3, 3), (4, 4), (5, 5), (6, 6), (8, 8)] {
        // Use a representative number for each bit width
        let n_rep = (1u64 << (pb - 1)) * (1u64 << (qb - 1)) + 1;
        let qubo = encode_factoring_qubo(n_rep, pb, qb);
        let total = qubo.size;
        let matrix_entries = total * total;
        println!(
            "  {:>6}  {:>6}  {:>12}  {:>18}",
            pb, qb, total, matrix_entries
        );
    }

    println!();
    println!("  The QUBO matrix grows quadratically with the number of bits.");
    println!("  Auxiliary variables (z, carry) dominate for larger bit widths.");

    // ─── Section 3: 8-bit range semiprimes with more replicas ───────────────────
    println!("\n=== 3. 8-Bit Range Semiprimes (Harder Problems) ===\n");

    let hard_numbers: Vec<(u64, &str, usize, usize)> = vec![
        (143, "11 x 13", 5, 5),
        (221, "13 x 17", 5, 5),
        (323, "17 x 19", 5, 5),
    ];

    let replicas = 16;
    let steps = 200_000;

    println!(
        "  Using parallel tempering: {} replicas, {} steps/replica\n",
        replicas, steps
    );

    println!(
        "  {:>4}  {:>10}  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}  {:>7}",
        "N", "Expected", "p_bits", "q_bits", "QUBO Size", "Energy", "Found", "Status"
    );
    println!("  {}", "-".repeat(72));

    for (n, expected, p_bits, q_bits) in &hard_numbers {
        let qubo = encode_factoring_qubo(*n, *p_bits, *q_bits);
        let qubo_size = qubo.size;

        let mut best_p = 0u64;
        let mut best_q = 0u64;
        let mut best_energy = f64::MAX;
        let mut success = false;

        // Multiple restarts for harder problems
        for _ in 0..10 {
            let (solution, energy) = parallel_tempering(&qubo, replicas, steps);
            let (p, q) = extract_factors(&solution, *p_bits, *q_bits);

            if energy < best_energy {
                best_energy = energy;
                best_p = p;
                best_q = q;
            }

            if p > 1 && q > 1 && p * q == *n {
                best_p = p;
                best_q = q;
                best_energy = energy;
                success = true;
                break;
            }
        }

        let found_str = if best_p > 1 && best_q > 1 {
            format!("{} x {}", best_p, best_q)
        } else {
            "---".to_string()
        };

        println!(
            "  {:>4}  {:>10}  {:>6}  {:>6}  {:>10}  {:>10.1}  {:>10}  {:>7}",
            n, expected, p_bits, q_bits, qubo_size, best_energy, found_str,
            if success { "OK" } else { "MISS" }
        );
    }

    // ─── Section 4: Energy landscape analysis ───────────────────────────────────
    println!("\n=== 4. Energy Landscape: Correct vs Random Assignments ===\n");

    let n: u64 = 15;
    let p_bits = 4;
    let q_bits = 4;
    let qubo = encode_factoring_qubo(n, p_bits, q_bits);

    // Evaluate energy for known correct factorization
    // p=3 (0011), q=5 (0101) => set bits correctly
    // Note: we can only set p and q bits; z and carry bits need correct values too.
    // We'll use the solver for a "correct" solution and compare with random.
    println!("  N = {} (3 x 5), {} total QUBO variables\n", n, qubo.size);

    // Run solver to find best energy
    let (best_sol, best_energy) = parallel_tempering(&qubo, 16, 100_000);
    let (bp, bq) = extract_factors(&best_sol, p_bits, q_bits);
    println!(
        "  Best found:  p={}, q={}, p*q={}, energy={:.2}",
        bp,
        bq,
        bp * bq,
        best_energy
    );

    // Evaluate some random assignments for comparison
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let mut random_energies: Vec<f64> = Vec::new();
    for _ in 0..1000 {
        let random_assignment: Vec<bool> = (0..qubo.size).map(|_| rng.gen()).collect();
        let energy = evaluate_qubo(&qubo, &random_assignment);
        random_energies.push(energy);
    }
    random_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_random: f64 = random_energies.iter().sum::<f64>() / random_energies.len() as f64;
    let min_random = random_energies[0];
    let max_random = random_energies[random_energies.len() - 1];

    println!("  Random assignments (1000 samples):");
    println!("    Min energy:  {:.2}", min_random);
    println!("    Avg energy:  {:.2}", avg_random);
    println!("    Max energy:  {:.2}", max_random);
    println!(
        "    Best solver energy is {:.1}x lower than random average",
        if best_energy.abs() > 1e-6 {
            avg_random / best_energy
        } else {
            f64::INFINITY
        }
    );

    println!();
    println!("  The QUBO formulation creates an energy landscape where the");
    println!("  correct factorization corresponds to the global minimum (energy ~0).");
    println!("  Simulated annealing with parallel tempering explores this landscape");
    println!("  to find the ground state, encoding factoring as optimization.");
}
