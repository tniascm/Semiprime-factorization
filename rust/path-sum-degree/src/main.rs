/// E20: Boolean Polynomial Degree Audit — CLI
///
/// Usage:
///   path-sum-degree --mode=audit   [--bits=14,16,18,20] [--seed=N]
///   path-sum-degree --mode=scale   [--bits=14,16,...,48] [--samples=2000] [--monomials=200] [--max-degree=6] [--seed=N]
///   path-sum-degree --mode=quick   (fast smoke test: 2 bit sizes, low sample count)
///
/// Modes:
///   audit  — full analysis at small n (n ≤ 20): CRT rank + min fitting degree + correlation
///   scale  — degree scaling scan: correlation lower bounds across all bit sizes
///   quick  — fast smoke test for CI

use path_sum_degree::scaling::{run_scan, ScanConfig, print_summary};
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_args(&args);

    let mode = opts.get("mode").map(|s| s.as_str()).unwrap_or("scale");

    match mode {
        "audit" => run_audit(&opts),
        "scale" => run_scale(&opts),
        "quick" => run_quick(),
        other => {
            eprintln!("Unknown mode: {other}. Use --mode=audit|scale|quick");
            std::process::exit(1);
        }
    }
}

fn run_audit(opts: &HashMap<String, String>) {
    let seed = parse_u64(opts, "seed", 0x4e32_dead_beef_0001);
    let bit_sizes = parse_bit_sizes(opts, &[14, 16, 18, 20]);

    println!("E20 AUDIT mode: CRT rank + min fitting degree + correlation");
    println!("Bit sizes: {:?}", bit_sizes);
    println!("Seed: 0x{seed:016x}\n");

    let config = ScanConfig {
        bit_sizes,
        max_degree: 4,
        n_semiprimes: 3000,
        n_monomials: 300,
        run_crt_rank: true,
        seed,
    };

    let result = run_scan(&config);
    print_summary(&result);
    write_json(&result, "data/E20_audit_results.json");
}

fn run_scale(opts: &HashMap<String, String>) {
    let seed = parse_u64(opts, "seed", 0x4e32_dead_beef_0001);
    let samples = parse_usize(opts, "samples", 2000);
    let monomials = parse_usize(opts, "monomials", 200);
    let max_degree = parse_u32(opts, "max-degree", 6);
    let bit_sizes = parse_bit_sizes(opts, &[14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48]);

    println!("E20 SCALE mode: degree scaling law across bit sizes");
    println!("Bit sizes: {:?}", bit_sizes);
    println!("Samples per block: {samples}, monomials per degree: {monomials}, max degree: {max_degree}");
    println!("Seed: 0x{seed:016x}\n");

    let config = ScanConfig {
        bit_sizes,
        max_degree,
        n_semiprimes: samples,
        n_monomials: monomials,
        run_crt_rank: true,
        seed,
    };

    let result = run_scan(&config);
    print_summary(&result);
    write_json(&result, "data/E20_scaling_results.json");
}

fn run_quick() {
    println!("E20 QUICK mode: smoke test (small sample)");
    let config = ScanConfig {
        bit_sizes: vec![14, 16],
        max_degree: 3,
        n_semiprimes: 300,
        n_monomials: 50,
        run_crt_rank: true,
        seed: 42,
    };
    let result = run_scan(&config);
    print_summary(&result);
    // Quick mode: just print, no file write
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for arg in args {
        if let Some(kv) = arg.strip_prefix("--") {
            if let Some((k, v)) = kv.split_once('=') {
                map.insert(k.to_string(), v.to_string());
            } else {
                map.insert(kv.to_string(), "true".to_string());
            }
        }
    }
    map
}

fn parse_u64(opts: &HashMap<String, String>, key: &str, default: u64) -> u64 {
    opts.get(key)
        .and_then(|v| {
            if let Some(hex) = v.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).ok()
            } else {
                v.parse().ok()
            }
        })
        .unwrap_or(default)
}

fn parse_usize(opts: &HashMap<String, String>, key: &str, default: usize) -> usize {
    opts.get(key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_u32(opts: &HashMap<String, String>, key: &str, default: u32) -> u32 {
    opts.get(key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_bit_sizes(opts: &HashMap<String, String>, default: &[u32]) -> Vec<u32> {
    opts.get("bits")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn write_json<T: serde::Serialize>(value: &T, path: &str) {
    // Create parent directory if needed
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Warning: could not create directory {parent:?}: {e}");
                return;
            }
        }
    }
    match serde_json::to_string_pretty(value) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("Warning: could not write {path}: {e}");
            } else {
                println!("\nResults written to {path}");
            }
        }
        Err(e) => eprintln!("Warning: could not serialize results: {e}"),
    }
}
