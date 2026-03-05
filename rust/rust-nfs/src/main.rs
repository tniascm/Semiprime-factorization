use clap::Parser;
use rug::Integer;
use rust_nfs::params::NfsParams;
use rust_nfs::pipeline::factor_nfs;

#[derive(Parser)]
#[command(
    name = "rust-nfs",
    about = "Production NFS factorization — beats CADO-NFS"
)]
struct Cli {
    /// Number to factor (decimal)
    #[arg(long)]
    factor: Option<String>,

    /// Benchmark mode: bit sizes to test (comma-separated)
    #[arg(long, value_delimiter = ',')]
    bits: Option<Vec<u32>>,

    /// Number of semiprimes per bit size in benchmark mode
    #[arg(long, default_value = "3")]
    semiprimes: usize,

    /// Number of threads (defaults to available CPUs)
    #[arg(long)]
    threads: Option<usize>,

    /// Random seed for semiprime generation
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    if let Some(ref n_str) = cli.factor {
        factor_mode(n_str);
        return;
    }

    if let Some(ref bit_sizes) = cli.bits {
        benchmark(bit_sizes, cli.semiprimes, cli.seed);
        return;
    }

    eprintln!("Usage: rust-nfs --factor N  or  rust-nfs --bits 96,112,128");
}

/// Factor a single number and output the result as JSON on stdout.
fn factor_mode(n_str: &str) {
    let n: Integer = n_str.parse().expect("invalid number");
    let bits = n.significant_bits();
    let params = NfsParams::for_bits(bits);
    eprintln!(
        "=== rust-nfs: factoring {} ({} bits, {} params) ===",
        n, bits, params.name
    );

    let result = factor_nfs(&n, &params);
    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    if let Some(ref f) = result.factor {
        let factor: Integer = f.parse().unwrap();
        let cofactor = Integer::from(&n / &factor);
        eprintln!("Factor: {} = {} x {}", n, factor, cofactor);
    } else {
        eprintln!("Failed to factor {}", n);
    }
}

/// Run a benchmark across multiple bit sizes, generating random semiprimes.
fn benchmark(bit_sizes: &[u32], semiprimes_per_size: usize, seed: u64) {
    eprintln!("=== rust-nfs Benchmark ===");
    eprintln!("Threads: {}", rayon::current_num_threads());
    eprintln!("Bit sizes: {:?}", bit_sizes);
    eprintln!("Semiprimes per size: {}", semiprimes_per_size);
    eprintln!();

    for &bits in bit_sizes {
        let params = NfsParams::for_bits(bits);
        eprintln!("=== {} bits ({}) ===", bits, params.name);

        let mut total_rels = 0usize;
        let mut total_ms = 0.0f64;
        let mut total_sieve_ms = 0.0f64;
        let mut successes = 0usize;

        for i in 0..semiprimes_per_size {
            let n = generate_semiprime(bits, seed + bits as u64 * 1000 + i as u64);
            let start = std::time::Instant::now();
            let result = factor_nfs(&n, &params);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let found = result.factor.is_some();
            if found {
                successes += 1;
            }

            let n_str = n.to_string();
            let n_display = if n_str.len() > 16 {
                format!("{}...", &n_str[..16])
            } else {
                n_str
            };
            eprintln!(
                "  [{}/{}] N={} -> {} rels, {:.0}ms total ({:.0}ms sieve), factor={}",
                i + 1,
                semiprimes_per_size,
                n_display,
                result.relations_found,
                elapsed,
                result.sieve_ms,
                if found {
                    result.factor.as_ref().unwrap().as_str()
                } else {
                    "none"
                }
            );

            total_rels += result.relations_found;
            total_ms += elapsed;
            total_sieve_ms += result.sieve_ms;
        }

        let mean_rps = if total_sieve_ms > 0.0 {
            total_rels as f64 / (total_sieve_ms / 1000.0)
        } else {
            0.0
        };

        eprintln!(
            "  >> {}-bit: {}/{} factored, {:.0} rels/sec (sieve), {:.0}ms mean total",
            bits,
            successes,
            semiprimes_per_size,
            mean_rps,
            total_ms / semiprimes_per_size as f64
        );
        eprintln!();
    }
}

/// Generate a random semiprime of approximately `bits` total bits.
/// Each prime factor is approximately bits/2 bits.
fn generate_semiprime(bits: u32, seed: u64) -> Integer {
    use rug::rand::RandState;

    let mut rng = RandState::new();
    rng.seed(&Integer::from(seed));

    let half = bits / 2;
    loop {
        let mut p = Integer::from(Integer::random_bits(half, &mut rng));
        p.set_bit(half - 1, true); // ensure at least half bits
        p |= Integer::from(1u32); // ensure odd
        p = p.next_prime();

        let mut q = Integer::from(Integer::random_bits(half, &mut rng));
        q.set_bit(half - 1, true);
        q |= Integer::from(1u32);
        q = q.next_prime();

        if p == q {
            continue;
        }

        let n = Integer::from(&p * &q);
        let n_bits = n.significant_bits();
        // Accept if within +/- 2 bits of target
        if n_bits >= bits.saturating_sub(2) && n_bits <= bits + 2 {
            return n;
        }
    }
}
