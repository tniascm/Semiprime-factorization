use clap::Parser;

#[derive(Parser)]
#[command(name = "rust-nfs", about = "Production NFS factorization")]
struct Cli {
    #[arg(long)]
    factor: Option<String>,

    #[arg(long, value_delimiter = ',')]
    bits: Option<Vec<u32>>,

    #[arg(long, default_value = "3")]
    semiprimes: usize,

    #[arg(long)]
    threads: Option<usize>,
}

fn main() {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    eprintln!("rust-nfs: not yet implemented");
}
