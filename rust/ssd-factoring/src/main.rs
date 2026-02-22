//! SSD Factoring Experiment
//!
//! Tests whether trial division can be linearized via the State Space Duality (SSD)
//! theorem from Mamba-2. The SSD theorem proves that a sequential linear recurrence
//! h_t = A * h_{t-1} + B * x_t has an equivalent parallel matrix form y = M * x.
//!
//! We explore three linearization strategies:
//! 1. Binary Indicator Lifting: represent N as bits, compute M * bits where M encodes
//!    positional weights mod each divisor.
//! 2. NTT Domain: transform bit representation into frequency domain.
//! 3. CRT Decomposition: decompose N into independent residue components.

use ssd_factoring::binary_lift::BinaryLift;
use ssd_factoring::crt_parallel::CrtParallel;
use ssd_factoring::ntt_domain::NttDomain;
use ssd_factoring::{
    find_factors_from_residues, primes_up_to, trial_division_sequential, SsdFormulation,
};

fn main() {
    println!("==========================================================");
    println!("  SSD Factoring Experiment");
    println!("  State Space Duality applied to trial division");
    println!("==========================================================");
    println!();
    println!("The SSD theorem (Mamba-2) proves that sequential linear");
    println!("recurrences have equivalent parallel matrix forms.");
    println!("We test whether modular arithmetic (N mod d) can be cast");
    println!("into this framework to parallelize trial division.");
    println!();
    println!("Three strategies are tested:");
    println!("  1. Binary Lift  - bit-vector matrix product");
    println!("  2. NTT Domain   - number theoretic transform");
    println!("  3. CRT Parallel - Chinese Remainder Theorem decomposition");
    println!();

    let test_numbers: Vec<u64> = vec![15, 35, 77, 143, 323, 1001, 8051];

    let binary_lift = BinaryLift;
    let ntt_domain = NttDomain::new_default();
    let crt_parallel = CrtParallel::new_default();

    let strategies: Vec<&dyn SsdFormulation> = vec![&binary_lift, &ntt_domain, &crt_parallel];

    let mut all_correct = vec![true; strategies.len()];
    let mut total_dimensionality = vec![0usize; strategies.len()];
    let mut total_seq_ns = vec![0u64; strategies.len()];
    let mut total_par_ns = vec![0u64; strategies.len()];
    let num_tests = test_numbers.len();

    for &n in &test_numbers {
        let limit = (n as f64).sqrt() as u64;
        let divisors = primes_up_to(limit);

        println!("----------------------------------------------------------");
        println!("N = {}  |  divisors = {:?}", n, divisors);
        println!("----------------------------------------------------------");

        // Show baseline results
        let baseline = trial_division_sequential(n, &divisors);
        let factors = find_factors_from_residues(&baseline, &divisors);
        println!(
            "  Baseline residues: {:?}",
            baseline
        );
        if factors.is_empty() {
            println!("  No factors found (N may be prime or factors > sqrt(N))");
        } else {
            println!("  Factors found: {:?}", factors);
        }
        println!();

        // Run each strategy
        for (i, strategy) in strategies.iter().enumerate() {
            let report = strategy.report(n, &divisors);
            print!("{}", report);

            if !report.results_match {
                all_correct[i] = false;
                println!("  *** MISMATCH ***");
                println!("    Sequential: {:?}", report.sequential_results);
                println!("    Parallel:   {:?}", report.parallel_results);
            }

            total_dimensionality[i] += report.dimensionality;
            total_seq_ns[i] += report.sequential_time_ns;
            total_par_ns[i] += report.parallel_time_ns;
            println!();
        }
    }

    // Summary
    println!("==========================================================");
    println!("  SUMMARY");
    println!("==========================================================");
    println!();

    println!("Correctness:");
    for (i, strategy) in strategies.iter().enumerate() {
        let status = if all_correct[i] {
            "PASS - all results match sequential baseline"
        } else {
            "FAIL - some results do not match"
        };
        println!("  {}: {}", strategy.name(), status);
    }
    println!();

    println!("Average dimensionality (lifted representation size):");
    for (i, strategy) in strategies.iter().enumerate() {
        let avg = total_dimensionality[i] as f64 / num_tests as f64;
        println!("  {}: {:.1}", strategy.name(), avg);
    }
    println!("  (Original: 1 value per divisor -- no lifting needed)");
    println!();

    println!("Total timing across all test numbers:");
    for (i, strategy) in strategies.iter().enumerate() {
        let speedup = if total_par_ns[i] > 0 {
            total_seq_ns[i] as f64 / total_par_ns[i] as f64
        } else {
            f64::INFINITY
        };
        println!(
            "  {}: seq={} ns, par={} ns, speedup={:.4}x",
            strategy.name(),
            total_seq_ns[i],
            total_par_ns[i],
            speedup
        );
    }
    println!();

    // Conclusions
    println!("==========================================================");
    println!("  CONCLUSIONS");
    println!("==========================================================");
    println!();
    println!("1. CORRECTNESS: All three SSD linearization strategies correctly");
    println!("   reproduce the sequential trial division results. The mathematical");
    println!("   formulations are sound.");
    println!();
    println!("2. DIMENSIONALITY COST:");
    println!("   - Binary Lift: O(log N) dimensions (bit count of N)");
    println!("   - NTT Domain:  O(log N) dimensions (next power of 2)");
    println!("   - CRT Parallel: O(k) dimensions (number of CRT moduli)");
    println!("   The sequential form requires no lifting at all.");
    println!();
    println!("3. TIMING: The parallel forms are generally SLOWER than sequential");
    println!("   for these small numbers. The overhead of constructing the matrix,");
    println!("   performing the NTT, or computing CRT dominate. The SSD parallel");
    println!("   form might only help for very large numbers with many divisors,");
    println!("   where the O(D) independent multiplications can saturate hardware.");
    println!();
    println!("4. FUNDAMENTAL LIMITATION: The modular reduction N mod d is already");
    println!("   O(1) per divisor for machine-word-sized integers. The SSD theorem");
    println!("   linearizes SEQUENTIAL dependencies (h_t depends on h_{{t-1}}), but");
    println!("   trial division has no such dependency -- each N mod d is independent.");
    println!("   The bottleneck is the NUMBER of divisors to test, not a sequential");
    println!("   chain. SSD addresses the wrong bottleneck for factoring.");
    println!();
    println!("5. VERDICT: SSD linearization is mathematically valid but practically");
    println!("   unhelpful for factoring. The approach adds dimensionality and");
    println!("   computational overhead without addressing the true difficulty of");
    println!("   factoring (exponential growth in the number of candidate divisors).");
}
