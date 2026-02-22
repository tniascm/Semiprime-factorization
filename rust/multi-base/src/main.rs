use factoring_core::generate_rsa_target;
use multi_base::{analyze, compare_semiprime_vs_random};
use num_bigint::BigUint;
use rand::Rng;

fn main() {
    println!("=== Multi-Base Representation Analysis ===\n");

    let mut rng = rand::thread_rng();

    for bits in [32, 64, 128] {
        let target = generate_rsa_target(bits, &mut rng);
        println!("--- {}-bit semiprime: {} ---", bits, target.n);
        println!("  p = {}", target.p);
        println!("  q = {}", target.q);

        let analysis = analyze(&target.n);

        println!("\n  Base representations:");
        for rep in &analysis.representations {
            println!(
                "    {:20} | {:4} digits | entropy: {:.4} | autocorr: {:.4}",
                rep.base_name, rep.num_digits, rep.entropy, rep.autocorrelation_lag1
            );
        }

        println!("\n  RNS representation (first 10 primes):");
        for (i, residue) in analysis.rns.iter().enumerate().take(10) {
            let marker = if *residue == 0 { " ← DIVISIBLE" } else { "" };
            print!("    {} mod {} = {}{}", target.n, multi_base::RNS_MODULI[i], residue, marker);
            println!();
        }

        println!("\n  Cross-base features:");
        println!(
            "    Entropy variance: {:.6}",
            analysis.cross_base_features.entropy_variance
        );
        println!(
            "    Min entropy base: {}",
            analysis.cross_base_features.min_entropy_base
        );
        println!(
            "    Max entropy base: {}",
            analysis.cross_base_features.max_entropy_base
        );
        println!(
            "    RNS zeros: {}",
            analysis.cross_base_features.rns_zero_count
        );

        // Compare with random number of same bit size
        let random_n = loop {
            let mut bytes = vec![0u8; (bits as usize + 7) / 8];
            rng.fill(&mut bytes[..]);
            bytes[0] |= 0x80;
            let n = BigUint::from_bytes_be(&bytes);
            // Make sure it's odd and not a perfect square
            if n.bit(0) {
                break n;
            }
        };
        let random_analysis = analyze(&random_n);
        let comparison = compare_semiprime_vs_random(&analysis, &random_analysis);

        println!("\n  Entropy comparison (semiprime vs random):");
        for (base_name, semi_ent, rand_ent) in &comparison {
            let diff = semi_ent - rand_ent;
            let marker = if diff.abs() > 0.1 { " ***" } else { "" };
            println!(
                "    {:20} | semi: {:.4} | rand: {:.4} | diff: {:+.4}{}",
                base_name, semi_ent, rand_ent, diff, marker
            );
        }

        println!();
    }
}
