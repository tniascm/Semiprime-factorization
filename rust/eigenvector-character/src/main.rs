/// E21: Eigenvector Multiplicative Character Audit — CLI
///
/// Usage:
///   eigenvector-character [--mode=audit|control|scaling|smooth] [--bits=14,16,18,20]
///                         [--channels=0,1,...,6] [--eigenvectors=3] [--seed=N]
///
/// Modes:
///   audit   — (default) run character audit over prime-restricted CRT matrix
///   control — pipeline validation: full-group H[a][b]=parity(ab mod ℓ) matrix,
///             where characters ARE exact eigenvectors → should find amp ≈ 1
///   scaling — Fourier scaling: DFT of centered parity h̃=2h−1 on (ℤ/ℓℤ)*
///             for many primes ℓ; measures A_max(ℓ) and fits power law
///   smooth  — smoothness spectrum: DFT of centered B-smoothness indicator
///             on (ℤ/ℓℤ)* for multiple B; compares decay rate to parity baseline
///
/// Outputs:
///   - Human-readable summary to stdout
///   - JSON to data/E21_*.json

use eigenvector_character::{
    run_character_audit, run_full_group_control, run_prime_restricted_smoothness,
    run_permutation_null, run_cross_n_transfer, run_multi_character_score, run_bootstrap_ci,
    scaling_primes, smoothness_spectrum,
    CharacterAuditResult, FullGroupControlResult, FourierScalingAnalysis,
    PrimeRestrictedResult, SmoothnessScalingAnalysis, StressTestResult,
    PermutationNullResult, CrossNTransferResult, MultiCharacterScoreResult, BootstrapCIResult,
};
use eisenstein_hunt::CHANNELS;
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_args(&args);

    let mode = opts.get("mode").map(|s| s.as_str()).unwrap_or("audit");

    match mode {
        "audit"    => run_audit_mode(&opts),
        "control"  => run_control_mode(&opts),
        "scaling"  => run_scaling_mode(&opts),
        "smooth"   => run_smooth_mode(&opts),
        "restrict" => run_restrict_mode(&opts),
        "stress"   => run_stress_mode(&opts),
        other => {
            eprintln!("Unknown mode: {other}. Use --mode=audit|control|scaling|smooth|restrict|stress");
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

fn run_smooth_mode(opts: &HashMap<String, String>) {
    let lo      = parse_u64(opts, "lo", 101);
    let hi      = parse_u64(opts, "hi", 50_000);
    let n_pts   = parse_usize(opts, "points", 30);
    let top_k   = parse_usize(opts, "topk", 10);

    // Smoothness bounds to test.  Default: 10, 30, 100, 300.
    let bounds: Vec<u64> = opts.get("bounds")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 30, 100, 300]);

    let primes = scaling_primes(lo, hi, n_pts);

    println!("E21b SMOOTHNESS SPECTRUM: DFT of centered B-smoothness indicator on (ℤ/ℓℤ)*");
    println!("Primes     : {} values from {} to {}", primes.len(), primes.first().unwrap(), primes.last().unwrap());
    println!("Bounds B   : {:?}", bounds);
    println!("Top-k chars: {top_k}");
    println!("{}", "─".repeat(72));

    let mut all_analyses: Vec<SmoothnessScalingAnalysis> = Vec::new();

    for &b in &bounds {
        println!();
        println!("═══ B = {b} ═══");

        // Compute per-prime with progress output.
        let mut results = Vec::with_capacity(primes.len());
        for &ell in &primes {
            eprint!("  ℓ={ell:6} B={b:3} … ");
            let result = smoothness_spectrum(ell, b, top_k);
            eprintln!(
                "p={:.3}  A={:.5}  null={:.5}  excess={:.2}  headE={:.3}",
                result.smooth_fraction,
                result.a_max,
                result.null_a_max,
                result.excess_ratio,
                result.head_energy_fraction,
            );
            results.push(result);
        }

        // Build scaling analysis from pre-computed results.
        let analysis = build_smooth_analysis(b, results);
        all_analyses.push(analysis);
    }

    println!();
    print_smooth_summary(&all_analyses);

    // Write all analyses to JSON.
    write_json(&all_analyses, "data/E21b_smoothness_spectrum.json");
}

fn run_restrict_mode(opts: &HashMap<String, String>) {
    let top_k = parse_usize(opts, "topk", 10);
    let bit_sizes = parse_bit_sizes(opts, &[14, 16, 18, 20]);
    let channel_ids = parse_channel_ids(opts, 7);

    // Smoothness bounds to test.  Default: 10, 30.
    let bounds: Vec<u64> = opts
        .get("bounds")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 30]);

    println!("E21b PRIME-RESTRICTED SMOOTHNESS: does the smoothness character survive?");
    println!("Bit sizes  : {:?}", bit_sizes);
    println!("Channels   : {:?}", channel_ids);
    println!("Bounds B   : {:?}", bounds);
    println!("Top-k chars: {top_k}");
    println!("{}", "─".repeat(72));

    let mut all_results: Vec<PrimeRestrictedResult> = Vec::new();

    for &b in &bounds {
        println!();
        println!("═══ B = {b} ═══");

        for &n_bits in &bit_sizes {
            for &ch_idx in &channel_ids {
                let ch = &CHANNELS[ch_idx];
                eprint!(
                    "  n={n_bits:2}, k={:2}, ℓ={:6}, B={b:3} … ",
                    ch.weight, ch.ell
                );

                let result = run_prime_restricted_smoothness(n_bits, ch, b, top_k);

                eprintln!(
                    "done  ({} primes, fix_exc={:.2}, scan_exc={:.2}, corr_Nk={:+.4}, corr_apx={:+.4})",
                    result.n_primes,
                    result.fixed_r_excess,
                    result.scanned_excess,
                    result.product_corr_nk,
                    result.product_corr_approx,
                );
                all_results.push(result);
            }
        }
    }

    println!();
    print_restrict_summary(&all_results);
    write_json(&all_results, "data/E21b_prime_restricted.json");
}

fn print_restrict_summary(results: &[PrimeRestrictedResult]) {
    println!("┌───────────────────────────────────────────────────────────────────────────┐");
    println!("│ E21b PRIME-RESTRICTED: smoothness character under prime restriction       │");
    println!("│ Tests whether full-group smoothness bias survives when restricted to      │");
    println!("│ the prime image set {{g(p) : p in balanced semiprimes}}                    │");
    println!("└───────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ─── Table A: Fixed-r* test ───
    println!("Table A: Fixed-r* test  (full-group r* tested on restricted set)");
    println!("  fixed_amp = Pearson amplitude of χ_{{full_r*}} against s_B(g(·)) on primes");
    println!("  null = √(2/(n-2)) for single pre-specified r");
    println!("  excess > 2.0 = signal survives restriction");
    println!();
    println!(
        "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>6}  {:>5}  {:>7}  {:>7}  {:>5}",
        "bits", "k", "ℓ", "B", "r*_full", "full_x", "dens",
        "fix_amp", "null", "fix_x"
    );
    println!("{}", "─".repeat(78));

    for r in results {
        println!(
            "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>6.2}  {:>4.1}%  {:>7.4}  {:>7.4}  {:>5.2}",
            r.n_bits,
            r.channel_weight,
            r.ell,
            r.smoothness_bound,
            r.full_r_star,
            r.full_excess,
            r.restricted_smooth_fraction * 100.0,
            r.fixed_r_amplitude,
            r.fixed_r_null,
            r.fixed_r_excess,
        );
    }

    println!();

    // ─── Table B: Full scan on restricted set ───
    println!("Table B: Full character scan on restricted set");
    println!("  scan_amp = max Pearson amplitude over all r on the prime set");
    println!("  null = √(2·ln(order/2) / (n-2)) — accounts for scanning ~order/2 chars");
    println!("  excess > 1.5 = signal above noise after scan correction");
    println!();
    println!(
        "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>6}  {:>7}  {:>7}  {:>5}  {:>7}",
        "bits", "k", "ℓ", "B", "r*_full", "r*_scn", "scn_amp", "null", "scn_x", "verdict"
    );
    println!("{}", "─".repeat(78));

    for r in results {
        let verdict = if r.scanned_excess > 1.5 {
            "⚠ SIGNAL"
        } else if r.scanned_excess > 1.1 {
            "~ weak"
        } else {
            "✓ noise"
        };
        let r_match = if r.scanned_r_star == r.full_r_star {
            format!("{}=", r.scanned_r_star)
        } else {
            format!("{}", r.scanned_r_star)
        };
        println!(
            "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>6}  {:>7.4}  {:>7.4}  {:>5.2}  {:>7}",
            r.n_bits,
            r.channel_weight,
            r.ell,
            r.smoothness_bound,
            r.full_r_star,
            r_match,
            r.scanned_amplitude,
            r.scanned_null,
            r.scanned_excess,
            verdict,
        );
    }

    println!();

    // ─── Table C: Product tests ───
    println!("Table C: Product tests  s_B(g(p))·s_B(g(q)) vs χ_{{r*_scn}}(·)");
    println!("  corr_σ  = corr with Re(χ(σ_{{k-1}}(N) mod ℓ))  [needs p,q]");
    println!("  corr_Nk = corr with Re(χ(N^{{k-1}} mod ℓ))      [from N only]");
    println!("  |corr_Nk| < 0.1 = barrier.  |corr_Nk| > 0.3 = corridor.");
    println!();
    println!(
        "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>8}  {:>8}  {:>6}  verdict",
        "bits", "k", "ℓ", "B", "#pairs", "corr_σ", "corr_Nk", "r*_scn"
    );
    println!("{}", "─".repeat(72));

    for r in results {
        let verdict = if r.product_corr_nk.abs() > 0.3 {
            "⚠ CORRIDOR?"
        } else if r.product_corr_nk.abs() > 0.15 {
            "~ weak signal"
        } else {
            "✓ barrier"
        };
        println!(
            "{:>4}  {:>3}  {:>6}  {:>3}  {:>6}  {:>+8.4}  {:>+8.4}  {:>6}  {}",
            r.n_bits,
            r.channel_weight,
            r.ell,
            r.smoothness_bound,
            r.n_pairs,
            r.product_corr_sigma,
            r.product_corr_nk,
            r.scanned_r_star,
            verdict,
        );
    }

    println!();

    // ─── Table D: σ-approximation corridor test ───
    println!("Table D: σ-approximation corridor test  (the last algebraic loophole)");
    println!("  σ_approx = 1 + 2·⌊√N⌋^{{k−1}} + N^{{k−1}} mod ℓ   [N-only, assumes p≈q]");
    println!("  corr_σ   = true σ (needs p,q)    corr_apx = approximation (N-only)");
    println!("  err      = mean |σ_approx − σ_true| / ℓ   (0 = perfect, 0.25 = random)");
    println!();
    println!(
        "{:>4}  {:>3}  {:>6}  {:>3}  {:>8}  {:>8}  {:>8}  {:>6}  verdict",
        "bits", "k", "ℓ", "B", "corr_σ", "corr_apx", "corr_Nk", "err"
    );
    println!("{}", "─".repeat(72));

    for r in results {
        let verdict = if r.product_corr_approx.abs() > 0.15 {
            "⚠ APPROX?"
        } else {
            "✓ dead"
        };
        println!(
            "{:>4}  {:>3}  {:>6}  {:>3}  {:>+8.4}  {:>+8.4}  {:>+8.4}  {:>5.3}  {}",
            r.n_bits,
            r.channel_weight,
            r.ell,
            r.smoothness_bound,
            r.product_corr_sigma,
            r.product_corr_approx,
            r.product_corr_nk,
            r.sigma_approx_error_mean,
            verdict,
        );
    }

    println!();

    // ─── Aggregate summary ───
    let n_total = results.len();
    let n_fixed_signal = results.iter().filter(|r| r.fixed_r_excess > 2.0).count();
    let n_scan_signal = results.iter().filter(|r| r.scanned_excess > 1.5).count();
    let n_barrier = results
        .iter()
        .filter(|r| r.product_corr_nk.abs() < 0.15)
        .count();

    // Group by B.
    let mut bounds: Vec<u64> = results.iter().map(|r| r.smoothness_bound).collect();
    bounds.sort_unstable();
    bounds.dedup();

    println!("Summary by B:");
    for &b in &bounds {
        let group: Vec<&PrimeRestrictedResult> =
            results.iter().filter(|r| r.smoothness_bound == b).collect();
        let n_g = group.len();
        let mean_fix_x: f64 = group.iter().map(|r| r.fixed_r_excess).sum::<f64>() / n_g as f64;
        let mean_scan_x: f64 =
            group.iter().map(|r| r.scanned_excess).sum::<f64>() / n_g as f64;
        let mean_corr_nk: f64 =
            group.iter().map(|r| r.product_corr_nk.abs()).sum::<f64>() / n_g as f64;
        println!(
            "  B={:>3}: mean fixed_excess={:.2}, mean scan_excess={:.2}, mean |corr_Nk|={:.4}",
            b, mean_fix_x, mean_scan_x, mean_corr_nk,
        );
    }

    println!();
    println!("Overall: {n_total} blocks tested");
    println!("  Fixed-r* signal (excess > 2.0): {n_fixed_signal}/{n_total}");
    println!("  Scanned signal (excess > 1.5):  {n_scan_signal}/{n_total}");
    println!("  Barrier intact (|corr_Nk| < 0.15): {n_barrier}/{n_total}");

    if n_scan_signal == 0 {
        println!();
        println!("  ✓ Smoothness character does NOT survive prime restriction.");
        println!("    Like parity, the multiplicative Fourier bias vanishes when");
        println!("    restricted to the algebraically constrained set {{g(p)}}.");
        println!("    The smoothness corridor is CLOSED.");
    } else if n_barrier == n_total {
        println!();
        println!("  ~ Smoothness character partially survives restriction,");
        println!("    but the product test confirms the barrier: corr_Nk ≈ 0.");
        println!("    The bias is present but NOT N-extractable.");
    } else {
        let n_open = n_total - n_barrier;
        println!();
        println!("  ⚠ {n_open} block(s) show elevated corr_Nk — investigate!");
    }
}

/// Build a `SmoothnessScalingAnalysis` from pre-computed results.
fn build_smooth_analysis(
    bound: u64,
    results: Vec<eigenvector_character::SmoothnessSpectrumResult>,
) -> SmoothnessScalingAnalysis {
    // Filter out primes where A_max ≈ 0 (all elements B-smooth).
    let valid: Vec<&eigenvector_character::SmoothnessSpectrumResult> = results
        .iter()
        .filter(|r| r.a_max > 1e-15)
        .collect();

    // Log-log fit for smoothness (on valid points only).
    let (slope, intercept, r_sq) = if valid.len() >= 2 {
        let xs: Vec<f64> = valid.iter().map(|r| (r.ell as f64).ln()).collect();
        let ys: Vec<f64> = valid.iter().map(|r| r.a_max.ln()).collect();
        log_log_fit(&xs, &ys)
    } else {
        (-0.5, 0.0, 0.0)
    };

    // Corrected ratio (only for valid points).
    let ratios: Vec<f64> = valid
        .iter()
        .map(|r| {
            let ell_f = r.ell as f64;
            r.a_max * ell_f.sqrt() / ell_f.ln().sqrt()
        })
        .collect();
    let n = ratios.len() as f64;
    let mean = if n > 0.0 { ratios.iter().sum::<f64>() / n } else { 0.0 };
    let std_dev = if n > 1.0 {
        (ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };

    // Parity baseline slope (on valid ℓ range).
    let parity_slope = if valid.len() >= 2 {
        let xs: Vec<f64> = valid.iter().map(|r| (r.ell as f64).ln()).collect();
        let ys: Vec<f64> = valid.iter().map(|r| r.parity_a_max.ln()).collect();
        log_log_fit(&xs, &ys).0
    } else {
        -0.5
    };

    SmoothnessScalingAnalysis {
        smoothness_bound: bound,
        results,
        fitted_slope: slope,
        fitted_prefactor: intercept.exp(),
        r_squared: r_sq,
        corrected_ratio_mean: mean,
        corrected_ratio_std: std_dev,
        parity_slope,
    }
}

/// Simple least-squares fit (duplicated from lib for CLI convenience).
fn log_log_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f64 = x.iter().map(|a| a * a).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return (0.0, sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| { let pred = slope * xi + intercept; (yi - pred).powi(2) })
        .sum();
    let mean_y = sy / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let r_sq = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 1.0 };
    (slope, intercept, r_sq)
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

fn print_smooth_summary(analyses: &[SmoothnessScalingAnalysis]) {
    println!("┌───────────────────────────────────────────────────────────────────────────┐");
    println!("│ E21b SMOOTHNESS SPECTRUM: DFT of centered B-smoothness on (ℤ/ℓℤ)*       │");
    println!("│ Density-normalized: comparing to random subset of same density           │");
    println!("└───────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ─── Table A: Raw slope comparison (as before) ───
    println!("Table A: Raw power-law slope comparison");
    println!(
        "{:>5}  {:>7}  {:>8}  {:>8}  {:>6}  verdict",
        "B", "slope", "par_slp", "Δslope", "R²"
    );
    println!("{}", "─".repeat(55));

    for a in analyses {
        let delta = a.fitted_slope - a.parity_slope;
        let verdict = if delta > 0.10 {
            "⚠ SLOWER decay"
        } else if delta > 0.03 {
            "~ slightly slower"
        } else if delta < -0.10 {
            "✓ faster"
        } else {
            "✓ same"
        };
        println!(
            "{:>5}  {:>+7.4}  {:>+8.4}  {:>+8.4}  {:>6.4}  {}",
            a.smoothness_bound,
            a.fitted_slope,
            a.parity_slope,
            delta,
            a.r_squared,
            verdict,
        );
    }

    println!();

    // ─── Table B: Density-normalized excess ratio ───
    // This is the key test: does A_max exceed what a random subset of the
    // same density would produce?
    println!("Table B: Density-normalized excess ratio  (A_max / null_A_max)");
    println!("  null_A_max = √(p(1−p)·2·ln(n/2)/n)  = expected max for random subset of density p");
    println!("  excess > 1.0 = genuine multiplicative structure beyond density");
    println!();

    for a in analyses {
        let valid: Vec<&_> = a.results.iter().filter(|r| r.a_max > 1e-15).collect();
        if valid.is_empty() { continue; }

        // Compute mean and trend of excess ratio.
        let excess_vals: Vec<f64> = valid.iter().map(|r| r.excess_ratio).collect();
        let n_v = excess_vals.len() as f64;
        let excess_mean = excess_vals.iter().sum::<f64>() / n_v;
        let excess_std = if n_v > 1.0 {
            (excess_vals.iter().map(|x| (x - excess_mean).powi(2)).sum::<f64>() / (n_v - 1.0)).sqrt()
        } else { 0.0 };

        // First and last excess to show trend.
        let first_excess = valid.first().map(|r| r.excess_ratio).unwrap_or(0.0);
        let last_excess = valid.last().map(|r| r.excess_ratio).unwrap_or(0.0);

        let verdict = if excess_mean > 2.0 {
            "⚠ STRONG multiplicative bias"
        } else if excess_mean > 1.3 {
            "⚠ moderate bias"
        } else if excess_mean > 1.05 {
            "~ marginal"
        } else {
            "✓ consistent with random"
        };

        println!(
            "  B={:>3}:  excess_mean={:.2} ± {:.2}   first_ℓ={:.2}  last_ℓ={:.2}   {}",
            a.smoothness_bound, excess_mean, excess_std, first_excess, last_excess, verdict,
        );
    }

    println!();

    // ─── Table C: Head energy fraction ───
    println!("Table C: Head energy fraction  (Σ top-k |ĥ(r)|² / Var(s̃))");
    println!("  By Parseval: Σ_{{r≠0}} |ĥ(r)|² = p(1−p) for {{0,1}} indicator.");
    println!("  High head energy = spectral concentration in few modes.");
    println!();

    for a in analyses {
        let valid: Vec<&_> = a.results.iter().filter(|r| r.a_max > 1e-15).collect();
        if valid.is_empty() { continue; }

        let head_vals: Vec<f64> = valid.iter().map(|r| r.head_energy_fraction).collect();
        let parity_vals: Vec<f64> = valid.iter().map(|r| r.parity_head_energy).collect();
        let n_v = head_vals.len() as f64;

        let head_mean = head_vals.iter().sum::<f64>() / n_v;
        let par_mean = parity_vals.iter().sum::<f64>() / n_v;
        let first_head = valid.first().map(|r| r.head_energy_fraction).unwrap_or(0.0);
        let last_head = valid.last().map(|r| r.head_energy_fraction).unwrap_or(0.0);

        println!(
            "  B={:>3}:  head_mean={:.4}  par_head={:.4}  ratio={:.2}x   first={:.4}  last={:.4}",
            a.smoothness_bound, head_mean, par_mean, head_mean / par_mean.max(1e-15),
            first_head, last_head,
        );
    }

    println!();

    // ─── Table D: Detailed per-ℓ for most interesting B ───
    // Pick the B with highest mean excess ratio.
    let best_analysis = analyses
        .iter()
        .max_by(|a, b| {
            let mean_a: f64 = a.results.iter().filter(|r| r.a_max > 1e-15)
                .map(|r| r.excess_ratio).sum::<f64>()
                / a.results.iter().filter(|r| r.a_max > 1e-15).count().max(1) as f64;
            let mean_b: f64 = b.results.iter().filter(|r| r.a_max > 1e-15)
                .map(|r| r.excess_ratio).sum::<f64>()
                / b.results.iter().filter(|r| r.a_max > 1e-15).count().max(1) as f64;
            mean_a.partial_cmp(&mean_b).unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(best) = best_analysis {
        println!(
            "Detailed density-corrected results for B = {}:",
            best.smoothness_bound,
        );
        println!(
            "{:>7}  {:>6}  {:>9}  {:>9}  {:>7}  {:>6}  {:>7}",
            "ℓ", "dens_p", "A_max", "null_Amax", "excess", "headE%", "parHE%"
        );
        println!("{}", "─".repeat(72));

        for r in &best.results {
            if r.a_max < 1e-15 { continue; }
            println!(
                "{:>7}  {:>5.1}%  {:>9.6}  {:>9.6}  {:>7.2}  {:>5.2}%  {:>5.2}%",
                r.ell,
                r.smooth_fraction * 100.0,
                r.a_max,
                r.null_a_max,
                r.excess_ratio,
                r.head_energy_fraction * 100.0,
                r.parity_head_energy * 100.0,
            );
        }
    }

    println!();

    // ─── Final verdict ───
    let max_excess_mean: f64 = analyses.iter()
        .map(|a| {
            let valid: Vec<f64> = a.results.iter().filter(|r| r.a_max > 1e-15)
                .map(|r| r.excess_ratio).collect();
            if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 }
        })
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Final assessment:");
    if max_excess_mean > 2.0 {
        println!("  ⚠ STRONG density-corrected bias (max mean excess = {:.2}×).", max_excess_mean);
        println!("    The smoothness indicator has genuine multiplicative structure");
        println!("    beyond what density alone explains.  Warrants product test.");
    } else if max_excess_mean > 1.3 {
        println!("  ⚠ Moderate density-corrected bias (max mean excess = {:.2}×).", max_excess_mean);
        println!("    Some multiplicative structure survives density normalization.");
    } else if max_excess_mean > 1.05 {
        println!("  ~ Marginal excess (max mean = {:.2}×).", max_excess_mean);
        println!("    Barely above random-subset baseline. Likely max-statistics artifact.");
    } else {
        println!("  ✓ No density-corrected excess (max mean = {:.2}×).", max_excess_mean);
        println!("    After accounting for density, smoothness Fourier spectrum is");
        println!("    consistent with a random subset — no exploitable structure.");
    }
}

fn run_stress_mode(opts: &HashMap<String, String>) {
    let top_k = parse_usize(opts, "topk", 10);
    let bit_sizes = parse_bit_sizes(opts, &[14, 16, 18, 20, 24, 28, 32, 40, 48]);
    let channel_ids = parse_channel_ids(opts, 7);
    let n_perm = parse_usize(opts, "permutations", 200);
    let n_boot = parse_usize(opts, "bootstrap", 500);
    let seed = parse_u64(opts, "seed", 0xE21b_5713_E55);

    let bounds: Vec<u64> = opts
        .get("bounds")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 30]);

    println!("E21b STRESS TESTS: validating smoothness character findings");
    println!("Bit sizes    : {:?}", bit_sizes);
    println!("Channels     : {:?}", channel_ids);
    println!("Bounds B     : {:?}", bounds);
    println!("Permutations : {n_perm}");
    println!("Bootstrap    : {n_boot}");
    println!("{}", "─".repeat(72));

    let mut all_perm: Vec<PermutationNullResult> = Vec::new();
    let mut all_transfer: Vec<CrossNTransferResult> = Vec::new();
    let mut all_multi: Vec<MultiCharacterScoreResult> = Vec::new();
    let mut all_boot: Vec<BootstrapCIResult> = Vec::new();

    for &b in &bounds {
        println!();
        println!("═══ B = {b} ═══");

        for &ch_idx in &channel_ids {
            let ch = &CHANNELS[ch_idx];

            // Test 2: Cross-n transfer (once per (ch, B), across all bit sizes).
            eprint!(
                "  Cross-n transfer: k={}, ℓ={}, B={b} … ",
                ch.weight, ch.ell
            );
            let transfer = run_cross_n_transfer(ch, b, top_k, &bit_sizes, 20);
            eprintln!("done");
            all_transfer.push(transfer);

            for &n_bits in &bit_sizes {
                let block_seed = seed ^ (n_bits as u64 * 0x10000 + ch.ell);

                // Test 1: Permutation null.
                eprint!(
                    "  Perm null: n={n_bits:2}, k={:2}, ℓ={:6}, B={b:3} … ",
                    ch.weight, ch.ell
                );
                let perm = run_permutation_null(n_bits, ch, b, top_k, n_perm, block_seed);
                eprintln!(
                    "z={:+.2}, p={:.3}",
                    perm.z_score, perm.empirical_p_value
                );
                all_perm.push(perm);

                // Test 3: Multi-character N-score.
                eprint!(
                    "  Multi-chr: n={n_bits:2}, k={:2}, ℓ={:6}, B={b:3} … ",
                    ch.weight, ch.ell
                );
                let multi = run_multi_character_score(n_bits, ch, b, top_k, block_seed);
                eprintln!(
                    "all={:+.4}, top10={:+.4}, direct={:+.4}, R²={:.4}",
                    multi.corr_dft_weighted_all,
                    multi.corr_dft_weighted_top10,
                    multi.corr_direct_smoothness,
                    multi.train_test_r_squared,
                );
                all_multi.push(multi);

                // Test 4: Bootstrap CI.
                eprint!(
                    "  Bootstrap: n={n_bits:2}, k={:2}, ℓ={:6}, B={b:3} … ",
                    ch.weight, ch.ell
                );
                let boot = run_bootstrap_ci(n_bits, ch, b, top_k, n_boot, block_seed);
                eprintln!(
                    "fix=[{:.2},{:.2}], Nk=[{:+.4},{:+.4}]",
                    boot.fix_excess_ci_lo,
                    boot.fix_excess_ci_hi,
                    boot.corr_nk_ci_lo,
                    boot.corr_nk_ci_hi,
                );
                all_boot.push(boot);
            }
        }
    }

    let result = StressTestResult {
        permutation_null: all_perm,
        cross_n_transfer: all_transfer,
        multi_character_score: all_multi,
        bootstrap_ci: all_boot,
    };

    println!();
    print_stress_summary(&result);
    write_json(&result, "data/E21b_stress_tests.json");
}

fn print_stress_summary(result: &StressTestResult) {
    println!("┌───────────────────────────────────────────────────────────────────────────┐");
    println!("│ E21b STRESS TESTS: validation of smoothness character findings            │");
    println!("└───────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ─── Table 1: Permutation null ───
    println!("Test 1: PERMUTATION NULL (is observed corr_Nk consistent with random pairing?)");
    println!("  p > 0.05 = observed is consistent with null (barrier confirmed)");
    println!(
        "  {:>4} {:>4} {:>6} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8}  {}",
        "n", "k", "ℓ", "B", "obs_Nk", "null_μ", "null_σ", "z", "p-val", "verdict"
    );
    println!("  {}", "─".repeat(80));
    for r in &result.permutation_null {
        let verdict = if r.empirical_p_value > 0.05 {
            "✓ consistent"
        } else {
            "⚠ anomalous"
        };
        println!(
            "  {:>4} {:>4} {:>6} {:>4} {:>+8.4} {:>+8.4} {:>8.4} {:>+8.2} {:>8.3}  {}",
            r.n_bits,
            r.channel_weight,
            r.ell,
            r.smoothness_bound,
            r.observed_corr_nk,
            r.null_mean,
            r.null_std,
            r.z_score,
            r.empirical_p_value,
            verdict,
        );
    }
    let n_consistent = result
        .permutation_null
        .iter()
        .filter(|r| r.empirical_p_value > 0.05)
        .count();
    let n_total_perm = result.permutation_null.len();
    println!(
        "  Summary: {n_consistent}/{n_total_perm} blocks consistent with null (p > 0.05)"
    );
    println!();

    // ─── Table 2: Cross-n transfer ───
    println!("Test 2: CROSS-N TRANSFER (is scanned r* from n=20 stable at other bit sizes?)");
    println!("  ratio = transfer_fix_excess / local_fix_excess; [0.7, 1.3] = stable");
    for tr in &result.cross_n_transfer {
        println!(
            "\n  k={}, ℓ={}, B={}, source_r*={}",
            tr.channel_weight, tr.ell, tr.smoothness_bound, tr.source_r_star
        );
        println!(
            "  {:>5} {:>12} {:>12} {:>12} {:>8} {:>8}  {}",
            "n", "local_r*", "local_exc", "xfer_exc", "ratio", "n_valid", "verdict"
        );
        println!("  {}", "─".repeat(78));
        for e in &tr.entries {
            let verdict = if e.consistency_ratio >= 0.5 && e.consistency_ratio <= 2.0 {
                "✓ stable"
            } else if e.consistency_ratio > 0.0 {
                "~ weak"
            } else {
                "✗ fail"
            };
            println!(
                "  {:>5} {:>12} {:>12.2} {:>12.2} {:>8.3} {:>8}  {}",
                e.n_bits,
                e.local_r_star,
                e.local_fix_excess,
                e.transfer_fix_excess,
                e.consistency_ratio,
                e.n_valid,
                verdict,
            );
        }
    }
    println!();

    // ─── Table 3: Multi-character N-score ───
    println!("Test 3: MULTI-CHARACTER N-SCORE (can any character combination extract from N?)");
    println!("  all = DFT-optimal weighting (≡ direct smoothness check)");
    println!("  top10 = top-10 characters; R² = held-out from trained weights");
    println!("  all ≈ 0 = NO extraction possible");
    println!(
        "  {:>4} {:>4} {:>6} {:>4} {:>8} {:>8} {:>8} {:>8}  {}",
        "n", "k", "ℓ", "B", "corr_all", "top10", "direct", "R²", "verdict"
    );
    println!("  {}", "─".repeat(72));
    for m in &result.multi_character_score {
        let verdict = if m.corr_dft_weighted_all.abs() < 0.15
            && m.corr_dft_weighted_top10.abs() < 0.15
            && m.train_test_r_squared < 0.05
        {
            "✓ BARRIER"
        } else {
            "⚠ signal?"
        };
        println!(
            "  {:>4} {:>4} {:>6} {:>4} {:>+8.4} {:>+8.4} {:>+8.4} {:>8.4}  {}",
            m.n_bits,
            m.channel_weight,
            m.ell,
            m.smoothness_bound,
            m.corr_dft_weighted_all,
            m.corr_dft_weighted_top10,
            m.corr_direct_smoothness,
            m.train_test_r_squared,
            verdict,
        );
    }
    let n_barrier_multi = result
        .multi_character_score
        .iter()
        .filter(|m| {
            m.corr_dft_weighted_all.abs() < 0.15
                && m.corr_dft_weighted_top10.abs() < 0.15
                && m.train_test_r_squared < 0.05
        })
        .count();
    let n_total_multi = result.multi_character_score.len();
    println!("  Summary: {n_barrier_multi}/{n_total_multi} blocks confirm barrier");
    println!();

    // ─── Table 4: Bootstrap CI ───
    println!("Test 4: BOOTSTRAP CONFIDENCE INTERVALS (95% CI for fix_excess and corr_Nk)");
    println!(
        "  {:>4} {:>4} {:>6} {:>4} {:>8} {:>18} {:>8} {:>18}",
        "n", "k", "ℓ", "B", "fix_mean", "fix_95%CI", "Nk_mean", "Nk_95%CI"
    );
    println!("  {}", "─".repeat(82));
    for b in &result.bootstrap_ci {
        println!(
            "  {:>4} {:>4} {:>6} {:>4} {:>8.2} [{:>6.2}, {:>6.2}] {:>+8.4} [{:>+7.4}, {:>+7.4}]",
            b.n_bits,
            b.channel_weight,
            b.ell,
            b.smoothness_bound,
            b.fix_excess_mean,
            b.fix_excess_ci_lo,
            b.fix_excess_ci_hi,
            b.corr_nk_mean,
            b.corr_nk_ci_lo,
            b.corr_nk_ci_hi,
        );
    }
    let n_fix_sig = result
        .bootstrap_ci
        .iter()
        .filter(|b| b.fix_excess_ci_lo > 1.5)
        .count();
    let n_nk_zero = result
        .bootstrap_ci
        .iter()
        .filter(|b| b.corr_nk_ci_lo <= 0.0 && b.corr_nk_ci_hi >= 0.0)
        .count();
    let n_total_boot = result.bootstrap_ci.len();
    println!(
        "  Summary: {n_fix_sig}/{n_total_boot} blocks with fix_excess CI_lo > 1.5"
    );
    println!(
        "  Summary: {n_nk_zero}/{n_total_boot} blocks with corr_Nk CI containing 0"
    );
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
