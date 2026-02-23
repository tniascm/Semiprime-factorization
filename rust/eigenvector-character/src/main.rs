/// E21: Eigenvector Multiplicative Character Audit — CLI
///
/// Usage:
///   eigenvector-character [--mode=audit|control|scaling] [--bits=14,16,18,20]
///                         [--channels=0,1,...,6] [--eigenvectors=3] [--seed=N]
///
/// Modes:
///   audit   — (default) run character audit over prime-restricted CRT matrix
///   control — pipeline validation: full-group H[a][b]=parity(ab mod ℓ) matrix,
///             where characters ARE exact eigenvectors → should find amp ≈ 1
///   scaling — Fourier scaling: DFT of centered parity h̃=2h−1 on (ℤ/ℓℤ)*
///             for many primes ℓ; measures A_max(ℓ) and fits power law
///
/// Outputs:
///   - Human-readable summary to stdout
///   - JSON to data/E21_*.json

use eigenvector_character::{
    run_character_audit, run_full_group_control, scaling_primes,
    CharacterAuditResult, FullGroupControlResult, FourierScalingAnalysis,
};
use eisenstein_hunt::CHANNELS;
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_args(&args);

    let mode = opts.get("mode").map(|s| s.as_str()).unwrap_or("audit");

    match mode {
        "audit"   => run_audit_mode(&opts),
        "control" => run_control_mode(&opts),
        "scaling" => run_scaling_mode(&opts),
        other => {
            eprintln!("Unknown mode: {other}. Use --mode=audit|control|scaling");
            std::process::exit(1);
        }
    }
}

fn run_audit_mode(opts: &HashMap<String, String>) {
    let seed         = parse_u64(opts, "seed", 0x4e32_e21_cafe_0001);
    let n_eigenvecs  = parse_usize(opts, "eigenvectors", 3);
    let bit_sizes    = parse_bit_sizes(opts, &[14, 16, 18, 20]);
    let channel_ids  = parse_channel_ids(opts, 7);

    println!("E21 CHARACTER AUDIT: eigenvector multiplicative character structure");
    println!("Bit sizes  : {:?}", bit_sizes);
    println!("Channels   : {:?}", channel_ids);
    println!("Eigenvectors per block: {n_eigenvecs}");
    println!("Seed       : 0x{seed:016x}");
    println!("{}", "─".repeat(72));

    let mut all_results: Vec<CharacterAuditResult> = Vec::new();

    for &n_bits in &bit_sizes {
        for &ch_idx in &channel_ids {
            let ch = &CHANNELS[ch_idx];
            eprint!("  n={n_bits:2}, k={:2}, ℓ={:6} … ", ch.weight, ch.ell);

            let result = run_character_audit(
                n_bits,
                ch,
                n_eigenvecs,
                seed ^ (n_bits as u64 * 0x1000 + ch_idx as u64),
            );
            eprint!("done  ({} primes, {} pairs)\n", result.n_primes, result.n_valid_pairs);
            all_results.push(result);
        }
    }

    println!();
    print_summary(&all_results);
    write_json(&all_results, "data/E21_character_audit.json");
}

fn run_control_mode(opts: &HashMap<String, String>) {
    let seed        = parse_u64(opts, "seed", 0x4e32_e210_c001);
    let n_eigenvecs = parse_usize(opts, "eigenvectors", 5);

    // Use channel primes ℓ ≤ 1000 for the control (larger ℓ makes O(ℓ²) matrices infeasible).
    let mut ells: Vec<u64> = CHANNELS.iter().map(|c| c.ell)
        .filter(|&e| e <= 1000)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter().collect();
    ells.sort();

    println!("E21 CONTROL: full-group H[a][b] = parity(ab mod ℓ), characters are exact eigenvectors");
    println!("Primes ℓ   : {:?}", ells);
    println!("Eigenvectors: {n_eigenvecs}");
    println!("Seed       : 0x{seed:016x}");
    println!("{}", "─".repeat(72));

    let mut all: Vec<FullGroupControlResult> = Vec::new();

    for &ell in &ells {
        eprint!("  ℓ={ell:6} (group order {}) … ", ell - 1);
        let result = run_full_group_control(ell, n_eigenvecs, seed ^ ell);
        eprintln!("done");
        all.push(result);
    }

    println!();
    print_control_summary(&all);
    write_json(&all, "data/E21_control_results.json");
}

fn run_scaling_mode(opts: &HashMap<String, String>) {
    use eigenvector_character::{centered_parity_spectrum, FourierScalingResult};

    let lo      = parse_u64(opts, "lo", 101);
    let hi      = parse_u64(opts, "hi", 50_000);
    let n_pts   = parse_usize(opts, "points", 30);
    let top_k   = parse_usize(opts, "topk", 10);

    let primes = scaling_primes(lo, hi, n_pts);

    println!("E21 FOURIER SCALING: centered parity h̃(a) = 2(a mod 2) − 1 on (ℤ/ℓℤ)*");
    println!("Primes     : {} values from {} to {}", primes.len(), primes.first().unwrap(), primes.last().unwrap());
    println!("Top-k chars: {top_k}");
    println!("{}", "─".repeat(72));

    // Compute per-prime with progress output.
    let mut results: Vec<FourierScalingResult> = Vec::with_capacity(primes.len());
    for &ell in &primes {
        eprint!("  ℓ={ell:6} (order {:6}) … ", ell - 1);
        let result = centered_parity_spectrum(ell, top_k);
        eprintln!("A_max={:.5}  A_max·√ℓ={:.3}", result.a_max, result.a_max_scaled);
        results.push(result);
    }

    // Build analysis with log-log fit.
    let analysis = eigenvector_character::build_fourier_scaling_analysis(results);

    println!();
    print_scaling_summary(&analysis);
    write_json(&analysis, "data/E21_fourier_scaling.json");
}

// ---------------------------------------------------------------------------
// Summary printer
// ---------------------------------------------------------------------------

fn print_summary(results: &[CharacterAuditResult]) {
    println!(
        "┌─────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ E21 RESULTS: Dominant eigenvector vs. multiplicative character χ_{{r*}}   │"
    );
    println!(
        "└─────────────────────────────────────────────────────────────────────────┘"
    );
    println!();

    // -----------------------------------------------------------------------
    // Table 1: top eigenvector character fit (corr_re_best and best_char_amp)
    // -----------------------------------------------------------------------
    println!("Table 1: Eigenvector-character fit (top eigenvector, u₁)");
    println!(
        "{:>4}  {:>3}  {:>6}  {:>6}  {:>6}  {:>6}  {:>5}  {:>5}  {:>6}",
        "bits", "k", "ℓ", "r*", "corr_re", "corr_im", "amp", "λ₁", "#pairs"
    );
    println!("{}", "─".repeat(63));

    for r in results {
        let Some(ev) = r.eigenvectors.first() else { continue };
        println!(
            "{:>4}  {:>3}  {:>6}  {:>6}  {:>+6.3}  {:>+6.3}  {:>5.3}  {:>+5.1}  {:>6}",
            r.n_bits,
            r.channel_weight,
            r.channel_ell,
            ev.best_char_r,
            ev.corr_re_best,
            ev.corr_im_best,
            ev.best_char_amp,
            ev.eigenvalue,
            ev.n_pairs_used,
        );
    }

    println!();

    // -----------------------------------------------------------------------
    // Table 2: product tests
    // -----------------------------------------------------------------------
    println!("Table 2: Product test  u1(p)*u1(q)  vs  Re(chi_{{r*}}(.))");
    println!(
        "  corr_sigma = correlation with Re(chi_{{r*}}(sigma_{{k-1}}(N) mod ell))  [needs p,q]"
    );
    println!(
        "  corr_Nk    = correlation with Re(chi_{{r*}}(N^{{k-1}} mod ell))         [from N only]"
    );
    println!(
        "  If |corr_Nk| < 0.1: barrier intact.  If |corr_Nk| > 0.5: corridor open."
    );
    println!();
    println!(
        "{:>4}  {:>3}  {:>6}  {:>8}  {:>8}  verdict",
        "bits", "k", "ℓ", "corr_σ", "corr_Nk"
    );
    println!("{}", "─".repeat(52));

    for r in results {
        let Some(ev) = r.eigenvectors.first() else { continue };
        let verdict = if ev.product_corr_nk.abs() > 0.5 {
            "⚠ CORRIDOR?"
        } else if ev.product_corr_nk.abs() > 0.2 {
            "~ weak signal"
        } else {
            "✓ barrier"
        };
        println!(
            "{:>4}  {:>3}  {:>6}  {:>+8.4}  {:>+8.4}  {}",
            r.n_bits,
            r.channel_weight,
            r.channel_ell,
            ev.product_corr_sigma,
            ev.product_corr_nk,
            verdict,
        );
    }

    println!();

    // -----------------------------------------------------------------------
    // Key findings summary
    // -----------------------------------------------------------------------
    let n_high_corr = results
        .iter()
        .filter_map(|r| r.eigenvectors.first())
        .filter(|ev| ev.corr_re_best.abs() > 0.8)
        .count();
    let n_barrier = results
        .iter()
        .filter_map(|r| r.eigenvectors.first())
        .filter(|ev| ev.product_corr_nk.abs() < 0.2)
        .count();
    let n_total = results
        .iter()
        .filter(|r| !r.eigenvectors.is_empty())
        .count();

    println!("Key findings:");
    println!(
        "  Eigenvector ~ character (|corr| > 0.8): {n_high_corr} / {n_total} blocks"
    );
    println!(
        "  Barrier intact (|corr_Nk| < 0.2):       {n_barrier} / {n_total} blocks"
    );
    if n_barrier < n_total {
        let n_open = n_total - n_barrier;
        println!("  ⚠ WARNING: {n_open} block(s) show elevated corr_Nk — inspect JSON.");
    }
}

fn print_control_summary(results: &[FullGroupControlResult]) {
    println!("┌─────────────────────────────────────────────────┐");
    println!("│ E21 CONTROL: Full-group eigenvector scan        │");
    println!("│ Characters are exact eigenvectors over the full │");
    println!("│ group — amp should be ~1.0 for non-trivial ones │");
    println!("└─────────────────────────────────────────────────┘");
    println!();
    println!(
        "{:>6}  {:>5}  {:>4}  {:>6}  {:>6}  {:>6}  {:>5}  check",
        "ℓ", "order", "idx", "λ", "r*", "amp", "corr"
    );
    println!("{}", "─".repeat(60));

    for ctrl in results {
        for ev in &ctrl.eigenvectors {
            let corr_mag = (ev.corr_re * ev.corr_re + ev.corr_im * ev.corr_im).sqrt();
            let check = if ev.best_char_amp > 0.90 {
                "✓ character"
            } else if ev.best_char_amp > 0.50 {
                "~ partial"
            } else {
                "✗ NOT char"
            };
            println!(
                "{:>6}  {:>5}  {:>4}  {:>+6.1}  {:>6}  {:>6.3}  {:>5.3}  {}",
                ctrl.ell,
                ctrl.order,
                ev.eigenvector_idx,
                ev.eigenvalue,
                ev.best_char_r,
                ev.best_char_amp,
                corr_mag,
                check,
            );
        }
    }

    println!();
    // Summarize: how many non-trivial eigenvectors (idx >= 1) are characters?
    let n_nontrivial: usize = results.iter().map(|r| r.eigenvectors.iter().skip(1).count()).sum();
    let n_char: usize = results.iter()
        .flat_map(|r| r.eigenvectors.iter().skip(1))
        .filter(|ev| ev.best_char_amp > 0.90)
        .count();
    println!("Pipeline validation: {n_char}/{n_nontrivial} non-trivial eigenvectors");
    println!("   have character amplitude > 0.90 over the full group.");
    if n_char == n_nontrivial {
        println!("   ✓ Pipeline correctly identifies characters when they exist.");
        println!("   → The drop to noise in the prime-restricted audit is REAL,");
        println!("     not a pipeline artifact.");
    } else {
        println!("   ⚠ Some eigenvectors not matched — may indicate convergence issues.");
    }
}

fn print_scaling_summary(analysis: &FourierScalingAnalysis) {
    println!("┌───────────────────────────────────────────────────────────────────────┐");
    println!("│ E21 FOURIER SCALING: centered parity h̃ = 2h − 1 on (ℤ/ℓℤ)*         │");
    println!("│ A_max(ℓ) = max_{{r≥1}} |ĥ̃(r)|  — should scale as C · ℓ^{{−1/2}}      │");
    println!("└───────────────────────────────────────────────────────────────────────┘");
    println!();

    // Main table with corrected ratio column.
    println!(
        "{:>7}  {:>7}  {:>9}  {:>9}  {:>11}  top-3 (r, amp)",
        "ℓ", "order", "A_max", "A_max·√ℓ", "A·√ℓ/√logℓ"
    );
    println!("{}", "─".repeat(82));

    for r in &analysis.results {
        let ell_f = r.ell as f64;
        let corrected = r.a_max * ell_f.sqrt() / ell_f.ln().sqrt();
        let top3_str: String = r.top_amplitudes.iter()
            .take(3)
            .map(|(r_idx, amp)| format!("({r_idx},{amp:.4})"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{:>7}  {:>7}  {:>9.6}  {:>9.3}  {:>11.3}  {}",
            r.ell,
            r.order,
            r.a_max,
            r.a_max_scaled,
            corrected,
            top3_str,
        );
    }

    println!();
    println!("Uncorrected power-law fit: A_max ≈ {:.4} · ℓ^({:.4})", analysis.fitted_prefactor, analysis.fitted_slope);
    println!("  Expected slope : −0.5000  (pure Gauss sum bound)");
    println!("  Observed slope : {:+.4}", analysis.fitted_slope);
    println!("  R²             : {:.6}", analysis.r_squared);
    println!();
    println!("Corrected ratio: A_max · √ℓ / √(log ℓ)");
    println!("  Mean  : {:.3}", analysis.corrected_ratio_mean);
    println!("  Std   : {:.3}", analysis.corrected_ratio_std);
    println!("  CV    : {:.1}%", 100.0 * analysis.corrected_ratio_std / analysis.corrected_ratio_mean);
    println!();

    // Verdict: is the deviation from -0.5 explained by √(log ℓ)?
    let cv = analysis.corrected_ratio_std / analysis.corrected_ratio_mean;
    let slope_dev = analysis.fitted_slope + 0.5; // positive = slower than ℓ^{-1/2}

    if cv < 0.10 && slope_dev.abs() < 0.10 {
        println!("  ✓ A_max = Θ(√(log ℓ) / √ℓ) — confirmed.");
        println!("    Slope deviation from −1/2 is fully explained by √(log ℓ) correction");
        println!("    (extremal-value statistics over ~ℓ/2 characters).");
        println!("    The stable rank head is an intrinsic harmonic artifact,");
        println!("    not an exploitable arithmetic leak.");
    } else if slope_dev > 0.15 {
        println!("  ⚠ Slope SLOWER than ℓ^{{−1/2}} by {slope_dev:.3} — unexpected.");
        println!("    Possible multiplicative bias; warrants investigation.");
    } else if slope_dev < -0.15 {
        println!("  ⚠ Slope FASTER than ℓ^{{−1/2}} by {:.3} — unexpected.", slope_dev.abs());
    } else {
        println!("  ~ Slope near −1/2 (deviation {slope_dev:+.4}) with √(log ℓ) correction (CV={cv:.1}%).");
        if cv < 0.15 {
            println!("    Consistent with A_max = Θ(√(log ℓ) / √ℓ).");
            println!("    No exploitable structure detected.");
        }
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
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

fn parse_bit_sizes(opts: &HashMap<String, String>, default: &[u32]) -> Vec<u32> {
    opts.get("bits")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_channel_ids(opts: &HashMap<String, String>, n_channels: usize) -> Vec<usize> {
    opts.get("channels")
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|&id| id < n_channels)
                .collect()
        })
        .unwrap_or_else(|| (0..n_channels).collect())
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

fn write_json<T: serde::Serialize>(value: &T, path: &str) {
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
            if let Err(e) = std::fs::write(path, &json) {
                eprintln!("Warning: could not write {path}: {e}");
            } else {
                println!("\nResults written to {path}");
            }
        }
        Err(e) => eprintln!("Warning: could not serialize results: {e}"),
    }
}
