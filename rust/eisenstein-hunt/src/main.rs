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

    // Generate semiprimes: 1000 for small ℓ, 5000 for large ℓ (birthday collision coverage)
    let semiprimes_1k = generate_semiprimes(1000, 16, 32);
    let semiprimes_5k = generate_semiprimes(5000, 16, 32);
    println!(
        "Generated {} + {} balanced semiprimes (16-32 bit)\n",
        semiprimes_1k.len(),
        semiprimes_5k.len()
    );

    // Header
    println!(
        "{:>4} {:>6} {:>8} {:>12} {:>12} {:>6}  {}",
        "k", "ell", "collsn", "candidates", "surv_1st", "surv_all", "status"
    );
    println!("{}", "-".repeat(72));

    let mut results = Vec::new();
    let mut total_tested = 0usize;

    for ch in CHANNELS {
        let semiprimes = if ch.ell > 10000 { &semiprimes_5k } else { &semiprimes_1k };
        let targets: Vec<u64> = semiprimes.iter().map(|sp| ground_truth(sp, ch)).collect();

        let channel_result = search_channel(ch, semiprimes, &targets);
        total_tested += channel_result.total_candidates;

        let status = if channel_result.survived_all > 0 {
            "*** BREAKTHROUGH ***"
        } else {
            "CLOSED"
        };

        let collision_str = if channel_result.collision_consistent {
            format!("yes({})", channel_result.collision_tests)
        } else {
            format!("no({})", channel_result.collision_tests)
        };

        println!(
            "{:>4} {:>6} {:>8} {:>12} {:>12} {:>6}  {}",
            channel_result.weight,
            channel_result.ell,
            collision_str,
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

        results.push(channel_result);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let breakthrough = results.iter().any(|r| r.survived_all > 0);
    let num_sp = semiprimes_1k.len().max(semiprimes_5k.len());

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
