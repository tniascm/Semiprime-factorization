/// E19: Eisenstein Congruence Hunt — Driver
///
/// Generates balanced semiprimes, then searches all 7 Eisenstein congruence
/// channels for poly(log N)-computable functions matching σ_{k-1}(N) mod ℓ.

use eisenstein_hunt::{
    generate_semiprimes, ground_truth, search_channel, SearchResult, CHANNELS,
};
use std::time::Instant;

fn main() {
    println!("=== E19: Eisenstein Congruence Hunt ===");
    println!("Testing poly(log N) candidates against sigma_{{k-1}}(N) mod ell\n");

    let start = Instant::now();

    // Generate semiprimes: scale with ℓ² for birthday collision coverage
    // Need sqrt(ℓ²) = ℓ semiprimes for mod-ℓ² birthday collisions
    let semiprimes_1k = generate_semiprimes(1000, 16, 32, 42);
    let semiprimes_5k = generate_semiprimes(5000, 16, 32, 137);
    let semiprimes_20k = generate_semiprimes(20000, 16, 32, 271);
    // For ℓ=43867: ℓ²≈1.924B. Birthday bound for ~10 genuine mod-ℓ² collisions
    // needs sqrt(20 * 1.924B) ≈ 196K samples. Use 200K with wider bit range
    // (24-48 bit) to avoid small-prime pool exhaustion.
    let semiprimes_200k = generate_semiprimes(200_000, 24, 48, 389);
    println!(
        "Generated {} + {} + {} + {} balanced semiprimes\n",
        semiprimes_1k.len(),
        semiprimes_5k.len(),
        semiprimes_20k.len(),
        semiprimes_200k.len()
    );

    // Header
    println!(
        "{:>4} {:>6} {:>8} {:>8} {:>12} {:>12} {:>6}  {}",
        "k", "ell", "col_l", "col_l2", "candidates", "surv_1st", "surv_all", "status"
    );
    println!("{}", "-".repeat(82));

    let mut results = Vec::new();
    let mut total_tested = 0usize;

    for ch in CHANNELS {
        // Scale semiprimes with ℓ: need ℓ² collisions for mod-ℓ² check
        // Birthday bound: need ~sqrt(2·ℓ²·k) semiprimes for k mod-ℓ² collisions
        let semiprimes = if ch.ell > 10000 {
            &semiprimes_200k // ℓ=43867: ℓ²≈1.924B, 200K gives ~10 genuine collisions
        } else if ch.ell > 3000 {
            &semiprimes_20k
        } else if ch.ell > 500 {
            &semiprimes_5k
        } else {
            &semiprimes_1k
        };
        let targets: Vec<u64> = semiprimes.iter().map(|sp| ground_truth(sp, ch)).collect();

        let channel_result = search_channel(ch, semiprimes, &targets);
        total_tested += channel_result.total_candidates;

        let status = if channel_result.survived_all > 0 {
            "*** BREAKTHROUGH ***"
        } else {
            "CLOSED"
        };

        let col_str = if channel_result.collision_consistent {
            format!("y({})", channel_result.collision_tests)
        } else {
            format!("n({})", channel_result.collision_tests)
        };
        let col_sq_str = if channel_result.collision_sq_consistent {
            format!("y({})", channel_result.collision_sq_tests)
        } else {
            format!("n({})", channel_result.collision_sq_tests)
        };

        println!(
            "{:>4} {:>6} {:>8} {:>8} {:>12} {:>12} {:>6}  {}",
            channel_result.weight,
            channel_result.ell,
            col_str,
            col_sq_str,
            channel_result.total_candidates,
            channel_result.survived_first,
            channel_result.survived_all,
            status
        );

        if !channel_result.survivors.is_empty() {
            for s in &channel_result.survivors {
                println!("  SURVIVOR: {}", s);
            }
        }

        // Print auxiliary collision check results
        for aux in &channel_result.aux_collisions {
            let yn = if aux.consistent {
                format!("y({})", aux.collisions_tested)
            } else {
                format!("n({})", aux.collisions_tested)
            };
            println!("      aux {:>35}: {}", aux.label, yn);
        }

        results.push(channel_result);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let breakthrough = results.iter().any(|r| r.survived_all > 0);
    let num_sp = semiprimes_200k.len();

    println!("\n{}", "=".repeat(72));
    println!("Total candidates tested: {}", total_tested);
    println!("Wall time:               {:.2}s", elapsed);
    println!(
        "Throughput:              {:.0} candidates/s",
        total_tested as f64 / elapsed
    );
    println!("Breakthrough:            {}", if breakthrough { "YES" } else { "NO" });

    let search_result = SearchResult {
        channels: results,
        total_candidates_tested: total_tested,
        total_wall_seconds: elapsed,
        candidates_per_second: total_tested as f64 / elapsed,
        num_semiprimes: num_sp,
        breakthrough,
    };

    let json = serde_json::to_string_pretty(&search_result).unwrap();

    // Save to project data directory (two levels up from crate)
    let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data");
    std::fs::create_dir_all(&data_dir).ok();
    let output_path = data_dir.join("E19_eisenstein_hunt_results.json");
    std::fs::write(&output_path, &json).unwrap();
    println!("\nResults saved to {}", output_path.display());

    if breakthrough {
        println!("\n!!! POTENTIAL BREAKTHROUGH DETECTED !!!");
        println!("A poly(log N)-computable function matches sigma_{{k-1}}(N) mod ell");
        println!("for all tested semiprimes. Verify with larger N before celebrating.");
    } else {
        println!("\nAll candidates eliminated. Eisenstein congruence gate empirically closed.");
    }
}
