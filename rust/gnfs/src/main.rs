use clap::{Parser, Subcommand};
use rug::Integer;

use gnfs::log::{setup_run_dir, utc_timestamp};
use gnfs::params::GnfsParams;
use gnfs::pipeline::factor_gnfs;

#[derive(Parser)]
#[command(name = "gnfs", about = "Production GNFS factorization")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Factor a number
    Factor {
        /// The number to factor (decimal)
        number: String,
        /// Parameter preset (e.g., "c60", "c80", "test")
        #[arg(long, default_value = "auto")]
        params: String,
        /// Output directory for run artifacts
        #[arg(long, default_value = "runs")]
        output_dir: String,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Factor {
            number,
            params: params_name,
            output_dir,
            seed,
            verbose: _verbose,
        } => {
            let n: Integer = match number.parse() {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("Invalid number: {}", e);
                    std::process::exit(1);
                }
            };

            let digits = n.to_string().len() as u32;
            let bits = n.significant_bits();

            println!("[{}] [init] ========================================", utc_timestamp());
            println!("[{}] [init]   GNFS v0.1 — Production Factorization", utc_timestamp());
            println!("[{}] [init] ========================================", utc_timestamp());
            println!("[{}] [init]   N = {} ({} digits, {} bits)", utc_timestamp(), &n, digits, bits);
            println!("[{}] [init]   Seed: {}", utc_timestamp(), seed);

            let params = match params_name.as_str() {
                "c20" => GnfsParams::c20(),
                "c30" => GnfsParams::c30(),
                "c60" => GnfsParams::c60(),
                "c80" => GnfsParams::c80(),
                "test" => GnfsParams::test_small(),
                "auto" => GnfsParams::for_bits(bits as u64),
                _ => {
                    eprintln!("Unknown params preset: {}", params_name);
                    std::process::exit(1);
                }
            };

            println!("[{}] [init]   Params: {} (degree={}, lim={}, lpb={})",
                utc_timestamp(), params.name, params.degree, params.lim0, params.lpb0);

            let run_dir = setup_run_dir(&output_dir, digits, seed);
            println!("[{}] [init]   Run dir: {}", utc_timestamp(), run_dir.display());

            // Write run config
            let config = serde_json::json!({
                "n": n.to_string(),
                "digits": digits,
                "bits": bits,
                "seed": seed,
                "params": &params,
            });
            std::fs::write(
                run_dir.join("run_config.json"),
                serde_json::to_string_pretty(&config).unwrap(),
            ).ok();

            // Run pipeline
            let result = factor_gnfs(&n, &params, Some(&run_dir));

            // Summary
            println!("[{}] [done] ========================================", utc_timestamp());
            if let Some(ref f) = result.factor {
                let factor: Integer = f.parse().unwrap();
                let cofactor = Integer::from(&n / &factor);
                println!("[{}] [done]   {} = {} × {}", utc_timestamp(), &n, f, cofactor);
            } else {
                println!("[{}] [done]   No factor found", utc_timestamp());
            }
            println!("[{}] [done]   Relations: {}", utc_timestamp(), result.relations_found);
            println!("[{}] [done]   Matrix: {} × {}", utc_timestamp(), result.matrix_rows, result.matrix_cols);
            println!("[{}] [done]   Dependencies: {} found, {} tried",
                utc_timestamp(), result.dependencies_found, result.dependencies_tried);
            println!("[{}] [done]   Time: {:.1}s", utc_timestamp(), result.total_secs);
            println!("[{}] [done] ========================================", utc_timestamp());

            // Write summary
            std::fs::write(
                run_dir.join("summary.json"),
                serde_json::to_string_pretty(&result).unwrap(),
            ).ok();
        }
    }
}
