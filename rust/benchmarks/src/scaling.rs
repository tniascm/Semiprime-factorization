//! Scaling benchmarks for all 5 experiments.
//!
//! Tests each factoring method on semiprimes of increasing bit size,
//! measuring time and success rate to determine empirical complexity exponents.

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use factoring_core::{generate_rsa_target, RsaTarget};

/// Maximum time per single factoring attempt (seconds).
const TIMEOUT_SECS: u64 = 30;

/// Run a closure with a timeout. Returns None if timed out.
fn with_timeout<T: Send + 'static>(
    timeout: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Option<(T, Duration)> {
    let (tx, rx) = mpsc::channel();
    let start = Instant::now();
    std::thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(result) => Some((result, start.elapsed())),
        Err(_) => None,
    }
}

fn main() {
    println!("================================================================");
    println!("  SCALING BENCHMARKS: Empirical Complexity of 5 Experiments");
    println!("================================================================\n");

    let mut rng = StdRng::seed_from_u64(12345);

    // Generate test semiprimes at each bit size
    // For small sizes, use fixed known semiprimes for reproducibility
    let bit_sizes: Vec<u32> = vec![16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64];

    let mut targets: Vec<RsaTarget> = Vec::new();
    for &bits in &bit_sizes {
        let target = generate_rsa_target(bits, &mut rng);
        targets.push(target);
    }

    println!("Test semiprimes:");
    for t in &targets {
        println!(
            "  {:>2}-bit: N = {} = {} x {}",
            t.bit_size, t.n, t.p, t.q
        );
    }
    println!();

    // Run each experiment
    bench_experiment_1(&targets);
    bench_experiment_2(&targets);
    bench_experiment_3(&targets);
    bench_experiment_4(&targets);
    bench_experiment_5(&targets);

    // Summary comparison
    println!("\n================================================================");
    println!("  COMPARISON: All Methods");
    println!("================================================================\n");
    bench_all_comparison(&targets);
}

// ============================================================================
// Experiment 1: Class Number Oracle
// ============================================================================

fn bench_experiment_1(targets: &[RsaTarget]) {
    println!("================================================================");
    println!("  Experiment 1: Class Number Oracle (Eichler-Selberg)");
    println!("================================================================\n");
    println!("  Tests whether H(D) computation and discriminant analysis scale.");
    println!("  This is an analysis tool, not a direct factoring method.\n");
    println!(
        "  {:>5} {:>12} {:>12} {:>8} {:>8}",
        "bits", "N", "time_us", "#disc", "status"
    );
    println!("  {}", "-".repeat(52));

    for t in targets {
        let n = to_u64(&t.n);
        let p = to_u64(&t.p);
        let q = to_u64(&t.q);

        if n == 0 || p == 0 || q == 0 {
            println!("  {:>5} {:>12} {:>12} {:>8} {}", t.bit_size, "too large", "-", "-", "SKIP");
            continue;
        }

        let start = Instant::now();

        // Run discriminant analysis for l=2,3,5
        let l_values = vec![2u64, 3, 5];
        let result = class_number_oracle::discriminant_analysis::analyze_discriminants_multi_l(
            n, p, q, &l_values,
        );
        let elapsed = start.elapsed();

        let total_disc: usize = result.iter().map(|r| r.discriminants.len()).sum();

        println!(
            "  {:>5} {:>12} {:>12} {:>8} {}",
            t.bit_size,
            n,
            elapsed.as_micros(),
            total_disc,
            "OK"
        );
    }
    println!();

    // Also benchmark H(D) computation scaling
    println!("  H(D) computation scaling (Shanks BSGS vs exact):");
    println!(
        "  {:>10} {:>12} {:>12} {:>10}",
        "|D|", "exact_us", "shanks_us", "speedup"
    );
    println!("  {}", "-".repeat(50));

    let test_discriminants: Vec<i64> = vec![
        -23, -67, -163, -1000, -5000, -10000, -50000, -100000, -500000,
    ];

    for &d in &test_discriminants {
        let start_exact = Instant::now();
        let _h_exact = class_number_oracle::class_number::class_number_exact(d);
        let exact_us = start_exact.elapsed().as_micros();

        let start_shanks = Instant::now();
        let _h_shanks = class_number_oracle::class_number::class_number_shanks(d);
        let shanks_us = start_shanks.elapsed().as_micros();

        let speedup = if shanks_us > 0 {
            exact_us as f64 / shanks_us as f64
        } else {
            f64::INFINITY
        };

        println!(
            "  {:>10} {:>12} {:>12} {:>10.1}x",
            d.abs(),
            exact_us,
            shanks_us,
            speedup
        );
    }
    println!();
}

// ============================================================================
// Experiment 2: Sublinear Conductor Detection
// ============================================================================

fn bench_experiment_2(targets: &[RsaTarget]) {
    println!("================================================================");
    println!("  Experiment 2: Sublinear Conductor Detection");
    println!("================================================================\n");
    println!("  Tests sampling strategies on semiprimes of increasing size.");
    println!("  (partial_gauss skipped for >32-bit: O(N) per sample)\n");

    println!(
        "  {:>5} {:>15} {:>15} {:>15} {:>15} {:>10}",
        "bits", "random", "subgroup", "cross_corr", "cond_witness", "best_us"
    );
    println!("  {}", "-".repeat(80));

    for t in targets {
        let n = to_u64(&t.n);
        if n == 0 {
            println!(
                "  {:>5} {:>15} {:>15} {:>15} {:>15} {:>10}",
                t.bit_size, "SKIP", "SKIP", "SKIP", "SKIP", "-"
            );
            continue;
        }

        print!("  {:>5}", t.bit_size);

        let max_samples = if t.bit_size <= 32 { 1000 } else { 5000 };
        let max_order = if t.bit_size <= 24 { 600 } else { 1500 };
        let mut best_time = u128::MAX;

        // Method 1: random_sampling
        let r1 = l_function_factoring::sampling::random_sampling(n, max_samples);
        print_sampling_result(&r1, &mut best_time);

        // Method 2: subgroup_chain
        let r2 = l_function_factoring::sampling::subgroup_chain_sampling(n, max_order);
        print_sampling_result(&r2, &mut best_time);

        // Method 3: cross_correlation
        let r3 = l_function_factoring::sampling::cross_correlation(n, max_samples);
        print_sampling_result(&r3, &mut best_time);

        // Method 4: conductor_witness
        let r4 = l_function_factoring::sampling::conductor_witness(n, max_samples);
        print_sampling_result(&r4, &mut best_time);

        if best_time < u128::MAX {
            println!(" {:>10}", best_time);
        } else {
            println!(" {:>10}", "NONE");
        }
    }
    println!();
}

fn print_sampling_result(
    r: &l_function_factoring::sampling::SamplingResult,
    best_time: &mut u128,
) {
    if r.factor_found.is_some() {
        if (r.time_us as u128) < *best_time {
            *best_time = r.time_us as u128;
        }
        print!(" {:>5}s/{:>6}us", r.samples_tested, r.time_us);
    } else {
        print!("  FAIL/{:>8}", r.samples_tested);
    }
}

// ============================================================================
// Experiment 3: TTN vs MPS Tensor Networks
// ============================================================================

fn bench_experiment_3(targets: &[RsaTarget]) {
    println!("================================================================");
    println!("  Experiment 3: TTN vs MPS Tensor Networks");
    println!("================================================================\n");
    println!("  Compares MPS and TTN on semiprimes (10s timeout per method).\n");
    println!(
        "  {:>5} {:>12} {:>10} {:>6} {:>10} {:>6} {:>8}",
        "bits", "N", "MPS_us", "MPS", "TTN_us", "TTN", "winner"
    );
    println!("  {}", "-".repeat(64));

    let timeout = Duration::from_secs(10);

    for t in targets {
        if t.bit_size > 40 {
            println!(
                "  {:>5} {:>12} {:>10} {:>6} {:>10} {:>6} {:>8}",
                t.bit_size, "too large", "-", "SKIP", "-", "SKIP", "-"
            );
            continue;
        }

        let config = tnss_factoring::config_for_bits(t.bit_size as usize);

        // MPS with timeout
        let n_clone = t.n.clone();
        let cfg = config.clone();
        let mps_result = with_timeout(timeout, move || {
            tnss_factoring::factor_tnss(&n_clone, &cfg)
        });

        let (mps_us, mps_ok) = match &mps_result {
            Some((r, d)) => (d.as_micros(), r.factor.is_some()),
            None => (0, false),
        };
        let mps_timed_out = mps_result.is_none();

        // TTN with timeout
        let n_clone = t.n.clone();
        let cfg = config.clone();
        let ttn_result = with_timeout(timeout, move || {
            tnss_factoring::factor_ttn(&n_clone, &cfg)
        });

        let (ttn_us, ttn_ok) = match &ttn_result {
            Some((r, d)) => (d.as_micros(), r.factor.is_some()),
            None => (0, false),
        };
        let ttn_timed_out = ttn_result.is_none();

        let winner = match (mps_ok, ttn_ok) {
            (true, true) => {
                if mps_us < ttn_us {
                    "MPS"
                } else {
                    "TTN"
                }
            }
            (true, false) => "MPS",
            (false, true) => "TTN",
            (false, false) => "NONE",
        };

        let mps_status = if mps_timed_out {
            "T/O"
        } else if mps_ok {
            "OK"
        } else {
            "FAIL"
        };
        let ttn_status = if ttn_timed_out {
            "T/O"
        } else if ttn_ok {
            "OK"
        } else {
            "FAIL"
        };

        println!(
            "  {:>5} {:>12} {:>10} {:>6} {:>10} {:>6} {:>8}",
            t.bit_size,
            t.n,
            if mps_timed_out {
                "TIMEOUT".to_string()
            } else {
                format!("{}", mps_us)
            },
            mps_status,
            if ttn_timed_out {
                "TIMEOUT".to_string()
            } else {
                format!("{}", ttn_us)
            },
            ttn_status,
            winner
        );
    }
    println!();
}

// ============================================================================
// Experiment 4: Trace Formula Lattice Attack
// ============================================================================

fn bench_experiment_4(targets: &[RsaTarget]) {
    println!("================================================================");
    println!("  Experiment 4: Trace Formula Lattice Attack");
    println!("================================================================\n");
    println!("  Uses dimension matching + LLL to factor via spectral data.\n");
    println!(
        "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "bits", "N", "time_us", "dim_tot", "dim_new", "dim_old", "status"
    );
    println!("  {}", "-".repeat(64));

    for t in targets {
        let n = to_u64(&t.n);
        if n == 0 {
            println!(
                "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
                t.bit_size, "too large", "-", "-", "-", "-", "SKIP"
            );
            continue;
        }

        let n_val = n;
        let trace_result = with_timeout(Duration::from_secs(TIMEOUT_SECS), move || {
            trace_lattice::factor::factor_trace_lattice(n_val)
        });

        match trace_result {
            None => {
                println!(
                    "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
                    t.bit_size, n, "TIMEOUT", "-", "-", "-", "TIMEOUT"
                );
            }
            Some((result, elapsed)) => {
                let dim_total = trace_lattice::trace::dim_s2(n);
                let dim_new = trace_lattice::trace::dim_s2_new(n);
                let dim_old = dim_total - dim_new;

                let status = if result.factors.is_some() {
                    "OK"
                } else {
                    "FAIL"
                };

                println!(
                    "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
                    t.bit_size,
                    n,
                    elapsed.as_micros(),
                    dim_total,
                    dim_new,
                    dim_old,
                    status
                );
            }
        }
    }
    println!();

    // Also show dimension growth rate
    println!("  Dimension growth: dim S_2(Gamma_0(N)) vs N");
    println!("  {:>12} {:>10} {:>10} {:>12}", "N", "dim", "dim/N*12", "log2(dim)");
    println!("  {}", "-".repeat(48));
    for t in targets {
        let n = to_u64(&t.n);
        if n == 0 || n > 1_000_000_000_000 {
            continue;
        }
        let dim = trace_lattice::trace::dim_s2(n);
        let ratio = (dim as f64) / (n as f64) * 12.0;
        let log2_dim = if dim > 0 {
            (dim as f64).log2()
        } else {
            0.0
        };
        println!(
            "  {:>12} {:>10} {:>10.4} {:>12.2}",
            n, dim, ratio, log2_dim
        );
    }
    println!();
}

// ============================================================================
// Experiment 5: Chebotarev Density Discovery
// ============================================================================

fn bench_experiment_5(targets: &[RsaTarget]) {
    println!("================================================================");
    println!("  Experiment 5: Chebotarev Density Discovery");
    println!("================================================================\n");
    println!("  Measures element order densities and attempts Pollard p-1 style extraction.\n");
    println!(
        "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "bits", "N", "scan_us", "div_both", "div_one", "div_none", "factor?"
    );
    println!("  {}", "-".repeat(64));

    for t in targets {
        // Chebotarev requires element_order which uses BigUint
        // Practical up to ~40 bits with reasonable samples
        if t.bit_size > 44 {
            println!(
                "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
                t.bit_size, "too slow", "-", "-", "-", "-", "SKIP"
            );
            continue;
        }

        let max_prime = if t.bit_size <= 24 { 30 } else { 20 };
        let samples = if t.bit_size <= 24 {
            200
        } else if t.bit_size <= 32 {
            100
        } else {
            50
        };

        let start_scan = Instant::now();
        let measurements =
            group_structure::chebotarev::chebotarev_scan(&t.n, max_prime, samples);
        let scan_us = start_scan.elapsed().as_micros();

        let constraints = group_structure::chebotarev::extract_constraints(&measurements);

        let start_factor = Instant::now();
        let factor_result =
            group_structure::chebotarev::factor_via_chebotarev(&t.n, max_prime, samples);
        let _factor_us = start_factor.elapsed().as_micros();

        let total_us = scan_us + start_factor.elapsed().as_micros();

        let factored = if factor_result.is_some() {
            "YES"
        } else {
            "NO"
        };

        println!(
            "  {:>5} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}",
            t.bit_size,
            t.n,
            total_us,
            constraints.divides_both.len(),
            constraints.divides_one.len(),
            constraints.divides_neither.len(),
            factored
        );
    }
    println!();
}

// ============================================================================
// Cross-method comparison
// ============================================================================

fn bench_all_comparison(targets: &[RsaTarget]) {
    println!(
        "  {:>5} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "bits", "Exp2_Sampled", "Exp3_MPS", "Exp3_TTN", "Exp4_Trace", "Exp5_Cheb"
    );
    println!("  {}", "-".repeat(77));

    for t in targets {
        print!("  {:>5}", t.bit_size);

        // Exp 2: Best of random + conductor_witness (fast methods only)
        let n64 = to_u64(&t.n);
        if n64 > 0 {
            let max_s = if t.bit_size <= 32 { 1000 } else { 5000 };
            let start = Instant::now();
            let r1 = l_function_factoring::sampling::random_sampling(n64, max_s);
            let r2 = l_function_factoring::sampling::conductor_witness(n64, max_s);
            let elapsed = start.elapsed();
            let any_success = r1.factor_found.is_some() || r2.factor_found.is_some();
            if any_success {
                print!(" {:>12}us+", elapsed.as_micros());
            } else {
                print!(" {:>12}us-", elapsed.as_micros());
            }
        } else {
            print!(" {:>14}", "SKIP");
        }

        // Exp 3: MPS (10s timeout)
        if t.bit_size <= 40 {
            let config = tnss_factoring::config_for_bits(t.bit_size as usize);
            let n_clone = t.n.clone();
            match with_timeout(Duration::from_secs(10), move || {
                tnss_factoring::factor_tnss(&n_clone, &config)
            }) {
                Some((result, elapsed)) => {
                    if result.factor.is_some() {
                        print!(" {:>12}us+", elapsed.as_micros());
                    } else {
                        print!(" {:>12}us-", elapsed.as_micros());
                    }
                }
                None => print!(" {:>14}", "TIMEOUT"),
            }
        } else {
            print!(" {:>14}", "SKIP");
        }

        // Exp 3: TTN (10s timeout)
        if t.bit_size <= 40 {
            let config = tnss_factoring::config_for_bits(t.bit_size as usize);
            let n_clone = t.n.clone();
            match with_timeout(Duration::from_secs(10), move || {
                tnss_factoring::factor_ttn(&n_clone, &config)
            }) {
                Some((result, elapsed)) => {
                    if result.factor.is_some() {
                        print!(" {:>12}us+", elapsed.as_micros());
                    } else {
                        print!(" {:>12}us-", elapsed.as_micros());
                    }
                }
                None => print!(" {:>14}", "TIMEOUT"),
            }
        } else {
            print!(" {:>14}", "SKIP");
        }

        // Exp 4: Trace lattice (30s timeout)
        if n64 > 0 {
            let n_val = n64;
            match with_timeout(Duration::from_secs(TIMEOUT_SECS), move || {
                trace_lattice::factor::factor_trace_lattice(n_val)
            }) {
                Some((result, elapsed)) => {
                    if result.factors.is_some() {
                        print!(" {:>12}us+", elapsed.as_micros());
                    } else {
                        print!(" {:>12}us-", elapsed.as_micros());
                    }
                }
                None => print!(" {:>14}", "TIMEOUT"),
            }
        } else {
            print!(" {:>14}", "SKIP");
        }

        // Exp 5: Chebotarev (30s timeout)
        if t.bit_size <= 44 {
            let max_prime = if t.bit_size <= 24 { 30 } else { 20 };
            let samples = if t.bit_size <= 24 {
                200
            } else if t.bit_size <= 32 {
                100
            } else {
                50
            };
            let n_clone = t.n.clone();
            match with_timeout(Duration::from_secs(TIMEOUT_SECS), move || {
                group_structure::chebotarev::factor_via_chebotarev(&n_clone, max_prime, samples)
            }) {
                Some((result, elapsed)) => {
                    if result.is_some() {
                        print!(" {:>12}us+", elapsed.as_micros());
                    } else {
                        print!(" {:>12}us-", elapsed.as_micros());
                    }
                }
                None => print!(" {:>14}", "TIMEOUT"),
            }
        } else {
            print!(" {:>14}", "SKIP");
        }

        println!();
    }

    println!();
    println!("  Legend: NNNus+ = succeeded in NNN microseconds");
    println!("         NNNus- = failed after NNN microseconds");
    println!("         SKIP   = method not applicable at this size");
    println!("         TIMEOUT = exceeded {}s limit", TIMEOUT_SECS);
}

// ============================================================================
// Utilities
// ============================================================================

fn to_u64(n: &BigUint) -> u64 {
    n.to_u64().unwrap_or(0)
}
