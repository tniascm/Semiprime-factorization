use compression_factor::batch_comparison;
use factoring_core::generate_rsa_target;
use num_bigint::BigUint;
use rand::Rng;

fn main() {
    println!("=== Compression-Based Structure Detection ===\n");

    let mut rng = rand::thread_rng();
    let sample_size = 100;

    for bits in [32, 64, 128] {
        println!("--- {}-bit numbers ({} samples each) ---", bits, sample_size);

        // Generate semiprimes
        let semiprimes: Vec<BigUint> = (0..sample_size)
            .map(|_| {
                let target = generate_rsa_target(bits, &mut rng);
                target.n
            })
            .collect();

        // Generate random odd numbers of same size
        let randoms: Vec<BigUint> = (0..sample_size)
            .map(|_| {
                let mut bytes = vec![0u8; (bits as usize + 7) / 8];
                rng.fill(&mut bytes[..]);
                bytes[0] |= 0x80;
                if let Some(last) = bytes.last_mut() {
                    *last |= 0x01;
                }
                BigUint::from_bytes_be(&bytes)
            })
            .collect();

        let (semi_analyses, rand_analyses) = batch_comparison(&semiprimes, &randoms);

        // Aggregate statistics
        let semi_avg_rle: f64 =
            semi_analyses.iter().map(|a| a.rle_ratio).sum::<f64>() / sample_size as f64;
        let rand_avg_rle: f64 =
            rand_analyses.iter().map(|a| a.rle_ratio).sum::<f64>() / sample_size as f64;

        let semi_avg_entropy: f64 =
            semi_analyses.iter().map(|a| a.byte_entropy).sum::<f64>() / sample_size as f64;
        let rand_avg_entropy: f64 =
            rand_analyses.iter().map(|a| a.byte_entropy).sum::<f64>() / sample_size as f64;

        let semi_avg_transitions: f64 =
            semi_analyses.iter().map(|a| a.bit_transitions as f64).sum::<f64>() / sample_size as f64;
        let rand_avg_transitions: f64 =
            rand_analyses.iter().map(|a| a.bit_transitions as f64).sum::<f64>() / sample_size as f64;

        println!("  Metric              | Semiprimes | Random    | Diff");
        println!("  --------------------|------------|-----------|--------");
        println!(
            "  RLE ratio           | {:.4}     | {:.4}    | {:+.4}",
            semi_avg_rle,
            rand_avg_rle,
            semi_avg_rle - rand_avg_rle
        );
        println!(
            "  Byte entropy        | {:.4}     | {:.4}    | {:+.4}",
            semi_avg_entropy,
            rand_avg_entropy,
            semi_avg_entropy - rand_avg_entropy
        );
        println!(
            "  Bit transitions     | {:.1}      | {:.1}     | {:+.1}",
            semi_avg_transitions,
            rand_avg_transitions,
            semi_avg_transitions - rand_avg_transitions
        );
        println!();
    }
}
