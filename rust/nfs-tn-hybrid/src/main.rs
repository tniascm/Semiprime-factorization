//! Benchmark driver: NFS-TN hybrid vs classical NFS vs standard TNSS.
//!
//! Tests on balanced semiprimes at various bit sizes and outputs JSON results.

use std::time::Instant;

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use classical_nfs::pipeline::{factor_nfs_with_stats, NfsPipelineParams};
use nfs_tn_hybrid::pipeline::{factor_nfs_tn, NfsTnResult};
use nfs_tn_hybrid::NfsTnConfig;

/// Generate a balanced semiprime with the given bit size.
fn generate_balanced_semiprime(bits: u32, rng: &mut StdRng) -> u64 {
    let half = bits / 2;
    let lo = 1u64 << (half - 1);
    let hi = 1u64 << half;

    loop {
        let p = random_prime(lo, hi, rng);
        let q = random_prime(lo, hi, rng);
        if p == q {
            continue;
        }
        let n = p.checked_mul(q);
        if let Some(n) = n {
            let n_bits = 64 - n.leading_zeros();
            if n_bits == bits || n_bits == bits - 1 {
                // Check balance: min(p,q)/max(p,q) >= 0.3
                let (min_f, max_f) = if p < q { (p, q) } else { (q, p) };
                if (min_f as f64) / (max_f as f64) >= 0.3 {
                    return n;
                }
            }
        }
    }
}

/// Simple probabilistic prime test (Miller-Rabin with a few rounds).
fn is_probably_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    // Deterministic witnesses for n < 2^64
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in &witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut found = false;
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let m = modulus as u128;
    base %= modulus;
    let mut b = base as u128;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * b % m;
        }
        exp /= 2;
        b = b * b % m;
    }
    result as u64
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn random_prime(lo: u64, hi: u64, rng: &mut StdRng) -> u64 {
    loop {
        let mut candidate = rng.gen_range(lo..hi);
        if candidate % 2 == 0 {
            candidate += 1;
        }
        if is_probably_prime(candidate) {
            return candidate;
        }
    }
}

/// Benchmark result for one method on one semiprime.
#[derive(Debug, Serialize)]
struct MethodResult {
    success: bool,
    factor: u64,
    time_ms: f64,
}

/// Benchmark result for one semiprime across all methods.
#[derive(Debug, Serialize)]
struct SemiprimeResult {
    n: u64,
    bits: u32,
    nfs_tn: NfsTnResult,
    classical_nfs: MethodResult,
}

/// Aggregate results for a bit size.
#[derive(Debug, Serialize)]
struct BitSizeResult {
    bits: u32,
    count: usize,
    nfs_tn: AggregateStats,
    classical_nfs: AggregateStats,
}

#[derive(Debug, Serialize)]
struct AggregateStats {
    success_rate: f64,
    mean_time_ms: f64,
    median_time_ms: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    experiment: String,
    bit_sizes: Vec<u32>,
    count_per_size: usize,
    seed: u64,
    per_bit_size: Vec<BitSizeResult>,
    per_semiprime: Vec<SemiprimeResult>,
}

fn main() {
    let seed = 42u64;
    let bit_sizes: Vec<u32> = vec![32, 40, 48, 56, 64];
    let count_per_size = 10;

    println!("NFS-TN Hybrid Benchmark");
    println!("=======================");
    println!(
        "Bit sizes: {:?}, {} semiprimes each",
        bit_sizes, count_per_size
    );
    println!();

    let mut all_results: Vec<SemiprimeResult> = Vec::new();
    let mut per_bit_results: Vec<BitSizeResult> = Vec::new();

    for &bits in &bit_sizes {
        println!("--- {} bits ---", bits);
        let mut rng = StdRng::seed_from_u64(seed + bits as u64);

        let mut nfs_tn_times = Vec::new();
        let mut nfs_tn_successes = 0;
        let mut classical_times = Vec::new();
        let mut classical_successes = 0;

        let mut bit_results = Vec::new();

        for i in 0..count_per_size {
            let n = generate_balanced_semiprime(bits, &mut rng);
            println!("  [{}/{}] N = {} ({} bits)", i + 1, count_per_size, n, bits);

            // NFS-TN hybrid
            let nfs_tn_config = NfsTnConfig::for_bits(bits);
            let nfs_tn_result = factor_nfs_tn(n, &nfs_tn_config);
            println!(
                "    NFS-TN: success={}, factor={}, vars={}, bond={}, valid={}, smooth={}/{}, rel={}, time={:.1}ms",
                nfs_tn_result.success,
                nfs_tn_result.factor,
                nfs_tn_result.num_vars,
                nfs_tn_result.bond_dim,
                nfs_tn_result.valid_candidates,
                nfs_tn_result.rational_smooth,
                nfs_tn_result.algebraic_smooth,
                nfs_tn_result.full_relations,
                nfs_tn_result.time_ms,
            );
            nfs_tn_times.push(nfs_tn_result.time_ms);
            if nfs_tn_result.success {
                nfs_tn_successes += 1;
            }

            // Classical NFS
            let classical_start = Instant::now();
            let nfs_params = NfsPipelineParams::for_bits(bits as u64);
            let n_big = BigUint::from(n);
            let (classical_factor, _stats) = factor_nfs_with_stats(&n_big, &nfs_params);
            let classical_time_ms = classical_start.elapsed().as_secs_f64() * 1000.0;
            let classical_success = classical_factor.is_some();
            let classical_factor_u64 = classical_factor
                .as_ref()
                .and_then(|f| f.to_u64())
                .unwrap_or(0);
            println!(
                "    Classical NFS: success={}, factor={}, time={:.1}ms",
                classical_success, classical_factor_u64, classical_time_ms,
            );
            classical_times.push(classical_time_ms);
            if classical_success {
                classical_successes += 1;
            }

            bit_results.push(SemiprimeResult {
                n,
                bits,
                nfs_tn: nfs_tn_result,
                classical_nfs: MethodResult {
                    success: classical_success,
                    factor: classical_factor_u64,
                    time_ms: classical_time_ms,
                },
            });
        }

        // Aggregate stats
        nfs_tn_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classical_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let nfs_tn_agg = AggregateStats {
            success_rate: nfs_tn_successes as f64 / count_per_size as f64,
            mean_time_ms: nfs_tn_times.iter().sum::<f64>() / count_per_size as f64,
            median_time_ms: nfs_tn_times[count_per_size / 2],
        };
        let classical_agg = AggregateStats {
            success_rate: classical_successes as f64 / count_per_size as f64,
            mean_time_ms: classical_times.iter().sum::<f64>() / count_per_size as f64,
            median_time_ms: classical_times[count_per_size / 2],
        };

        println!(
            "  Summary: NFS-TN {:.0}% / {:.1}ms, Classical {:.0}% / {:.1}ms",
            nfs_tn_agg.success_rate * 100.0,
            nfs_tn_agg.mean_time_ms,
            classical_agg.success_rate * 100.0,
            classical_agg.mean_time_ms,
        );
        println!();

        per_bit_results.push(BitSizeResult {
            bits,
            count: count_per_size,
            nfs_tn: nfs_tn_agg,
            classical_nfs: classical_agg,
        });

        all_results.extend(bit_results);
    }

    // Write JSON
    let report = BenchmarkReport {
        experiment: "NFS-TN Hybrid vs Classical NFS".to_string(),
        bit_sizes: bit_sizes.clone(),
        count_per_size,
        seed,
        per_bit_size: per_bit_results,
        per_semiprime: all_results,
    };

    let json_path = "../../data/nfs_tn_hybrid_benchmark.json";
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            if let Err(e) = std::fs::write(json_path, &json) {
                eprintln!("Warning: could not write {}: {}", json_path, e);
                // Try current directory
                let alt_path = "nfs_tn_hybrid_benchmark.json";
                std::fs::write(alt_path, &json).ok();
                println!("Results written to {}", alt_path);
            } else {
                println!("Results written to {}", json_path);
            }
        }
        Err(e) => eprintln!("JSON serialization error: {}", e),
    }
}
