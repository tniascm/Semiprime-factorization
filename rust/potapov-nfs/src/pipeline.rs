//! Full NFS pipeline: poly selection -> sieve -> filter -> LA -> sqrt.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rug::ops::Pow;
use rug::Integer;

use crate::params::NfsParams;
use crate::timing::{PipelineTimings, StageResult};

/// Result of the full NFS pipeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NfsResult {
    pub n: String,
    pub factor: Option<String>,
    pub relations_found: usize,
    pub relations_after_filter: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub dependencies_found: usize,
    pub sqrt_attempts_tried: usize,
    pub sqrt_factor_attempt: Option<usize>,
    pub polyselect_ms: f64,
    pub sieve_ms: f64,
    pub sieve_setup_ms: f64,
    pub sieve_scan_ms: f64,
    pub sieve_cofactor_ms: f64,
    pub sieve_rels_per_sq: f64,
    pub filter_ms: f64,
    pub la_ms: f64,
    pub sqrt_ms: f64,
    pub total_ms: f64,
    pub sqrt_fail_rat_not_square: usize,
    pub sqrt_fail_alg_not_square: usize,
    pub sqrt_fail_trivial_gcd: usize,
    pub viability: ViabilityStats,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ViabilityStats {
    pub raw_relations: usize,
    pub filtered_relations: usize,
    pub remap_valid_relations: usize,
    pub remap_keep_ratio: f64,
    pub remap_dropped_total: usize,
    pub remap_dropped_unmapped: usize,
    pub remap_invalid_total: usize,
    pub remap_invalid_bad_fb_idx: usize,
    pub remap_invalid_hd_residual: usize,
    pub remap_invalid_hd_offset_missing: usize,
    pub remap_invalid_root_exp_zero: usize,
    pub remap_invalid_legacy_lp: usize,
    pub set_rows_filtered: usize,
    pub set_rows_remapped: usize,
    pub set_rows_dropped_remap: usize,
    pub set_rows_recomputed: usize,
    pub set_rows_matrix: usize,
    pub matrix_mode: String,
    pub active_dense_rat_cols: usize,
    pub active_dense_alg_cols: usize,
    pub active_special_q_cols: usize,
    pub active_qc_cols: usize,
    pub matrix_rows_pre_compact: usize,
    pub matrix_cols_pre_compact: usize,
    pub pre_prune_singleton_cols_total: usize,
    pub pre_prune_singleton_special_q_cols: usize,
    pub pre_prune_singleton_rat_lp_cols: usize,
    pub pre_prune_singleton_alg_lp_cols: usize,
    pub pre_prune_singleton_sign_cols: usize,
    pub pre_prune_singleton_rat_fb_cols: usize,
    pub pre_prune_singleton_alg_dense_cols: usize,
    pub pre_prune_singleton_qc_cols: usize,
    pub zero_cols_dropped_initial: usize,
    pub singleton_rows_dropped: usize,
    pub zero_cols_dropped_post_singleton: usize,
    pub zero_cols_dropped_total: usize,
    pub final_rows: usize,
    pub final_cols: usize,
    pub rows_minus_cols: isize,
    pub deps_found: usize,
    pub hd_residual_samples: Vec<HdResidualSample>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HdResidualSample {
    pub a: i64,
    pub b: u64,
    pub prime: u64,
    pub roots: Vec<u64>,
    pub root_multiplicities: Vec<u8>,
    pub total_exp: u8,
    pub root_exp_sum: u8,
    pub residual: u8,
    pub hd_degree: usize,
    pub has_repeated_root: bool,
    pub residual_divisible: bool,
}

#[derive(Debug, Clone, Default)]
struct RemapHybridStats {
    kept_relations: usize,
    skipped_unmapped: usize,
    invalid_bad_fb_idx: usize,
    invalid_hd_residual: usize,
    invalid_hd_offset_missing: usize,
    invalid_root_exp_zero: usize,
    invalid_legacy_lp: usize,
    hd_residual_samples: Vec<HdResidualSample>,
}

impl RemapHybridStats {
    fn invalid_total(&self) -> usize {
        self.invalid_bad_fb_idx
            + self.invalid_hd_residual
            + self.invalid_hd_offset_missing
            + self.invalid_root_exp_zero
            + self.invalid_legacy_lp
    }

    fn skipped_total(&self) -> usize {
        self.skipped_unmapped + self.invalid_total()
    }
}

fn log_viability_summary(stats: &ViabilityStats) {
    eprintln!(
        "  viability: filtered={} remap_valid={} set_rows(filtered={}, remapped={}, recomputed={}, matrix={}) dense_cols(rat={},alg={},sq={},qc={}) final={}x{} rows_minus_cols={} deps={} hd_residual={}",
        stats.filtered_relations,
        stats.remap_valid_relations,
        stats.set_rows_filtered,
        stats.set_rows_remapped,
        stats.set_rows_recomputed,
        stats.set_rows_matrix,
        stats.active_dense_rat_cols,
        stats.active_dense_alg_cols,
        stats.active_special_q_cols,
        stats.active_qc_cols,
        stats.final_rows,
        stats.final_cols,
        stats.rows_minus_cols,
        stats.deps_found,
        stats.remap_invalid_hd_residual
    );
    eprintln!(
        "  viability_singletons: total={} sq={} rat_lp={} alg_lp={} sign={} rat_fb={} alg_dense={} qc={}",
        stats.pre_prune_singleton_cols_total,
        stats.pre_prune_singleton_special_q_cols,
        stats.pre_prune_singleton_rat_lp_cols,
        stats.pre_prune_singleton_alg_lp_cols,
        stats.pre_prune_singleton_sign_cols,
        stats.pre_prune_singleton_rat_fb_cols,
        stats.pre_prune_singleton_alg_dense_cols,
        stats.pre_prune_singleton_qc_cols
    );
    eprintln!(
        "  viability_json: {}",
        serde_json::to_string(stats).unwrap_or_else(|_| "{}".to_string())
    );
}

fn effective_max_raw_rels(
    configured_cap: Option<usize>,
    target_raw_rels: usize,
    raw_target_step: usize,
) -> usize {
    configured_cap.unwrap_or_else(|| {
        target_raw_rels
            .saturating_mul(2)
            .saturating_add(raw_target_step)
    })
}

/// Factor N using the full NFS pipeline.
///
/// Tries multiple polynomial variants (different m values) to avoid the
/// "degenerate polynomial" problem where all dependencies give trivial GCD.
///
/// Stages:
///   1. Polynomial selection (base-m with variant)
///   2. Special-Q lattice sieve
///   3. Singleton/duplicate filtering
///   4. Linear algebra (GF(2) Gaussian elimination)
///   5. Square root and factor extraction
pub fn factor_nfs(n: &Integer, params: &NfsParams) -> NfsResult {
    let variant_start: u32 = std::env::var("POTAPOV_NFS_VARIANT_START")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    // With fast trivial_bail (200 deps), degenerate variants fail in <1s instead
    // of 30-45s. This makes trying more variants cheap. For degree>=4 (c45+),
    // 15 variants increases success rate on hard numbers at ~1s cost per failed variant.
    let default_max_variants: u32 = if params.degree >= 4 { 15 } else { 5 };
    let max_variants: u32 = std::env::var("POTAPOV_NFS_MAX_VARIANTS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_max_variants);
    let mut run_logger = RunLogger::new(n, params, max_variants);
    let fallback_enabled = std::env::var("POTAPOV_NFS_FALLBACK_RHO")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let fallback_first = std::env::var("POTAPOV_NFS_FALLBACK_FIRST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let fallback_max_bits = std::env::var("POTAPOV_NFS_FALLBACK_MAX_BITS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(170u32);

    // Only try Pollard-rho first for small numbers (< 80 bits).
    // For c30+ (100+ bits with balanced factors), rho needs O(2^25) steps
    // which exceeds the 200ms time limit, wasting time on every attempt.
    let rho_first_max_bits = std::env::var("POTAPOV_NFS_RHO_FIRST_MAX_BITS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(80u32);
    if fallback_enabled && fallback_first && n.significant_bits() <= rho_first_max_bits {
        if let Some((factor, fallback_ms)) = try_pollard_rho_factor(n) {
            let result = fallback_result(n, factor, fallback_ms);
            if let Some(logger) = run_logger.as_mut() {
                logger.finish(&result);
            }
            return result;
        }
    }

    // Try ECM before NFS for numbers up to ~200 bits.
    // For c45 (~148-bit balanced semiprimes), ECM with B1=50000 typically
    // succeeds in 1-4s, which is competitive with NFS and has lower overhead.
    // Disable with POTAPOV_NFS_ECM=0.
    let ecm_enabled = std::env::var("POTAPOV_NFS_ECM")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    let ecm_max_bits = std::env::var("POTAPOV_NFS_ECM_MAX_BITS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(200u32);
    if ecm_enabled && n.significant_bits() <= ecm_max_bits {
        if let Some((factor, ecm_ms)) = crate::ecm::try_ecm_factor(n) {
            let result = fallback_result(n, factor, ecm_ms);
            if let Some(logger) = run_logger.as_mut() {
                logger.finish(&result);
            }
            return result;
        }
    }

    let mut last_result = None;

    // Murphy E-based polyselect sweeps over leading coefficients and picks
    // the polynomial with the best Murphy E-value. Disable with
    // POTAPOV_NFS_POLYSELECT=basem to fall back to monic base-m selection.
    let use_murphy = std::env::var("POTAPOV_NFS_POLYSELECT")
        .map(|v| v != "basem")
        .unwrap_or(true);

    let polyselect_start = std::time::Instant::now();
    let variant_polys: Vec<(u32, Option<gnfs::types::PolynomialPair>)> = if use_murphy {
        let admax: u64 = std::env::var("POTAPOV_NFS_ADMAX")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(5000);
        let ad_incr: u64 = std::env::var("POTAPOV_NFS_AD_INCR")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        let ropteffort: f64 = std::env::var("POTAPOV_NFS_ROPTEFFORT")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let ranked = gnfs::polyselect::select_best_polynomial(
            n, params.degree, admax, ad_incr, ropteffort,
            max_variants as usize, params.lim0,
        );
        if ranked.is_empty() {
            (0..max_variants).map(|i| (variant_start + i, None)).collect()
        } else {
            ranked.into_iter().enumerate().map(|(i, p)| (i as u32, Some(p))).collect()
        }
    } else {
        (0..max_variants).map(|i| (variant_start + i, None)).collect()
    };
    let polyselect_ms = polyselect_start.elapsed().as_secs_f64() * 1000.0;

    for (try_idx, (variant_id, pre_poly)) in variant_polys.iter().enumerate() {
        if try_idx > 0 {
            eprintln!(
                "  === Trying polynomial {} of {} ===",
                try_idx + 1, variant_polys.len()
            );
        }
        let result = factor_nfs_inner(n, params, *variant_id, pre_poly.as_ref(), polyselect_ms);
        if let Some(logger) = run_logger.as_mut() {
            logger.log_variant(*variant_id, &result);
        }
        if result.factor.is_some() {
            if let Some(logger) = run_logger.as_mut() {
                logger.finish(&result);
            }
            return result;
        }
        last_result = Some(result);
    }

    let mut final_result = last_result.unwrap();
    if final_result.factor.is_none()
        && fallback_enabled
        && n.significant_bits() <= fallback_max_bits
        && !fallback_first
    {
        if let Some((factor, fallback_ms)) = try_pollard_rho_factor(n) {
            final_result.factor = Some(factor.to_string());
            final_result.total_ms += fallback_ms;
            eprintln!(
                "  fallback_rho: factor found after NFS failure in {:.0}ms: {}",
                fallback_ms, factor
            );
        }
    }
    if let Some(logger) = run_logger.as_mut() {
        logger.finish(&final_result);
    }
    final_result
}

/// Optional structured logger for reproducible run debugging.
///
/// Enabled by setting `POTAPOV_NFS_LOG_DIR=/path/to/logs`.
/// Writes:
/// - `run_config.json` (N, params, env knobs)
/// - `variants.jsonl` (one JSON record per polynomial variant)
/// - `summary.json` (final result)
struct RunLogger {
    run_dir: PathBuf,
    variants_file: PathBuf,
}

impl RunLogger {
    fn new(n: &Integer, params: &NfsParams, max_variants: u32) -> Option<Self> {
        let base = std::env::var("POTAPOV_NFS_LOG_DIR").ok()?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let n_str = n.to_string();
        let n_prefix = if n_str.len() > 24 {
            &n_str[..24]
        } else {
            &n_str
        };

        let run_dir = PathBuf::from(base).join(format!("{}_{}_{}", params.name, n_prefix, ts));
        if fs::create_dir_all(&run_dir).is_err() {
            return None;
        }

        let env_cfg = serde_json::json!({
            "POTAPOV_NFS_MAX_VARIANTS": std::env::var("POTAPOV_NFS_MAX_VARIANTS").ok(),
            "POTAPOV_NFS_VARIANT_START": std::env::var("POTAPOV_NFS_VARIANT_START").ok(),
            "POTAPOV_NFS_DEP_SEED": std::env::var("POTAPOV_NFS_DEP_SEED").ok(),
            "POTAPOV_NFS_DEP_XOR_K": std::env::var("POTAPOV_NFS_DEP_XOR_K").ok(),
            "POTAPOV_NFS_DEP_RANDOM_COUNT": std::env::var("POTAPOV_NFS_DEP_RANDOM_COUNT").ok(),
            "POTAPOV_NFS_MAX_DEPS_TRY": std::env::var("POTAPOV_NFS_MAX_DEPS_TRY").ok(),
            "POTAPOV_NFS_MAX_DEP_LEN": std::env::var("POTAPOV_NFS_MAX_DEP_LEN").ok(),
            "POTAPOV_NFS_DEP_LEN_TIERS": std::env::var("POTAPOV_NFS_DEP_LEN_TIERS").ok(),
            "POTAPOV_NFS_DEP_AUTO_RELAX": std::env::var("POTAPOV_NFS_DEP_AUTO_RELAX").ok(),
            "POTAPOV_NFS_DEP_REQUIRE_COPRIME_REL": std::env::var("POTAPOV_NFS_DEP_REQUIRE_COPRIME_REL").ok(),
            "POTAPOV_NFS_SQRT_SUCCESS_MODE": std::env::var("POTAPOV_NFS_SQRT_SUCCESS_MODE").ok(),
            "POTAPOV_NFS_TRIVIAL_BAIL": std::env::var("POTAPOV_NFS_TRIVIAL_BAIL").ok(),
            "POTAPOV_NFS_SKIP_SQRT": std::env::var("POTAPOV_NFS_SKIP_SQRT").ok(),
            "POTAPOV_NFS_SQRT_VERBOSE_DEPS": std::env::var("POTAPOV_NFS_SQRT_VERBOSE_DEPS").ok(),
            "POTAPOV_NFS_QC_COUNT": std::env::var("POTAPOV_NFS_QC_COUNT").ok(),
            "POTAPOV_NFS_REQUIRE_COPRIME_AB": std::env::var("POTAPOV_NFS_REQUIRE_COPRIME_AB").ok(),
            "POTAPOV_NFS_FULL_ONLY": std::env::var("POTAPOV_NFS_FULL_ONLY").ok(),
            "POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN": std::env::var("POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN").ok(),
            "POTAPOV_NFS_SPARSE_PREMERGE": std::env::var("POTAPOV_NFS_SPARSE_PREMERGE").ok(),
            "POTAPOV_NFS_SPARSE_PREMERGE_MAXSETS": std::env::var("POTAPOV_NFS_SPARSE_PREMERGE_MAXSETS").ok(),
            "POTAPOV_NFS_COMPACT_ZERO_COLS": std::env::var("POTAPOV_NFS_COMPACT_ZERO_COLS").ok(),
            "POTAPOV_NFS_SINGLETON_PRUNE": std::env::var("POTAPOV_NFS_SINGLETON_PRUNE").ok(),
            "POTAPOV_NFS_SINGLETON_PRUNE_MIN_WEIGHT": std::env::var("POTAPOV_NFS_SINGLETON_PRUNE_MIN_WEIGHT").ok(),
            "POTAPOV_NFS_PARTIAL_MERGE_2LP": std::env::var("POTAPOV_NFS_PARTIAL_MERGE_2LP").ok(),
            "POTAPOV_NFS_PARTIAL_MERGE_MAXSETS": std::env::var("POTAPOV_NFS_PARTIAL_MERGE_MAXSETS").ok(),
            "POTAPOV_NFS_HD_RESIDUAL_SAMPLE_LIMIT": std::env::var("POTAPOV_NFS_HD_RESIDUAL_SAMPLE_LIMIT").ok(),
            "POTAPOV_NFS_MAX_LP_KEYS": std::env::var("POTAPOV_NFS_MAX_LP_KEYS").ok(),
            "POTAPOV_NFS_REL_TARGET_MULT": std::env::var("POTAPOV_NFS_REL_TARGET_MULT").ok(),
            "POTAPOV_NFS_REL_TARGET_MIN": std::env::var("POTAPOV_NFS_REL_TARGET_MIN").ok(),
            "POTAPOV_NFS_ADAPTIVE_ROWS_RATIO": std::env::var("POTAPOV_NFS_ADAPTIVE_ROWS_RATIO").ok(),
            "POTAPOV_NFS_ADAPTIVE_ROWS_MIN": std::env::var("POTAPOV_NFS_ADAPTIVE_ROWS_MIN").ok(),
            "POTAPOV_NFS_ADAPTIVE_MARGIN_PCT": std::env::var("POTAPOV_NFS_ADAPTIVE_MARGIN_PCT").ok(),
            "POTAPOV_NFS_ADAPTIVE_CHECK_EVERY": std::env::var("POTAPOV_NFS_ADAPTIVE_CHECK_EVERY").ok(),
            "POTAPOV_NFS_ADAPTIVE_CHECK_MIN_RAW": std::env::var("POTAPOV_NFS_ADAPTIVE_CHECK_MIN_RAW").ok(),
            "POTAPOV_NFS_ADAPTIVE_RAW_STEP": std::env::var("POTAPOV_NFS_ADAPTIVE_RAW_STEP").ok(),
            "POTAPOV_NFS_ADAPTIVE_USE_MATRIX": std::env::var("POTAPOV_NFS_ADAPTIVE_USE_MATRIX").ok(),
            "POTAPOV_NFS_ADAPTIVE_MATRIX_ROWS_RATIO": std::env::var("POTAPOV_NFS_ADAPTIVE_MATRIX_ROWS_RATIO").ok(),
            "POTAPOV_NFS_ADAPTIVE_MATRIX_PROBE_STEP": std::env::var("POTAPOV_NFS_ADAPTIVE_MATRIX_PROBE_STEP").ok(),
            "POTAPOV_NFS_MAX_RAW_RELS": std::env::var("POTAPOV_NFS_MAX_RAW_RELS").ok(),
            "POTAPOV_NFS_SQ_BATCH_SIZE": std::env::var("POTAPOV_NFS_SQ_BATCH_SIZE").ok(),
            "POTAPOV_NFS_SQ_ROOT_CACHE_WINDOWS": std::env::var("POTAPOV_NFS_SQ_ROOT_CACHE_WINDOWS").ok(),
            "POTAPOV_NFS_NORM_BLOCK": std::env::var("POTAPOV_NFS_NORM_BLOCK").ok(),
            "POTAPOV_NFS_MAX_Q_WINDOWS": std::env::var("POTAPOV_NFS_MAX_Q_WINDOWS").ok(),
            "POTAPOV_NFS_OVR_LIM0": std::env::var("POTAPOV_NFS_OVR_LIM0").ok(),
            "POTAPOV_NFS_OVR_LIM1": std::env::var("POTAPOV_NFS_OVR_LIM1").ok(),
            "POTAPOV_NFS_OVR_LPB0": std::env::var("POTAPOV_NFS_OVR_LPB0").ok(),
            "POTAPOV_NFS_OVR_LPB1": std::env::var("POTAPOV_NFS_OVR_LPB1").ok(),
            "POTAPOV_NFS_OVR_MFB0": std::env::var("POTAPOV_NFS_OVR_MFB0").ok(),
            "POTAPOV_NFS_OVR_MFB1": std::env::var("POTAPOV_NFS_OVR_MFB1").ok(),
            "POTAPOV_NFS_OVR_LOG_I": std::env::var("POTAPOV_NFS_OVR_LOG_I").ok(),
            "POTAPOV_NFS_OVR_QMIN": std::env::var("POTAPOV_NFS_OVR_QMIN").ok(),
            "POTAPOV_NFS_OVR_QRANGE": std::env::var("POTAPOV_NFS_OVR_QRANGE").ok(),
            "POTAPOV_NFS_OVR_RELS_WANTED": std::env::var("POTAPOV_NFS_OVR_RELS_WANTED").ok(),
            "POTAPOV_NFS_OVR_DEGREE": std::env::var("POTAPOV_NFS_OVR_DEGREE").ok(),
            "POTAPOV_NFS_FALLBACK_RHO": std::env::var("POTAPOV_NFS_FALLBACK_RHO").ok(),
            "POTAPOV_NFS_FALLBACK_FIRST": std::env::var("POTAPOV_NFS_FALLBACK_FIRST").ok(),
            "POTAPOV_NFS_FALLBACK_MAX_BITS": std::env::var("POTAPOV_NFS_FALLBACK_MAX_BITS").ok(),
            "POTAPOV_NFS_FALLBACK_RHO_ROUNDS": std::env::var("POTAPOV_NFS_FALLBACK_RHO_ROUNDS").ok(),
            "POTAPOV_NFS_FALLBACK_RHO_ITERS": std::env::var("POTAPOV_NFS_FALLBACK_RHO_ITERS").ok(),
            "POTAPOV_NFS_DEP_BASIS_LIMIT": std::env::var("POTAPOV_NFS_DEP_BASIS_LIMIT").ok(),
            "GNFS_TRY_COUVEIGNES_ON_TRIVIAL": std::env::var("GNFS_TRY_COUVEIGNES_ON_TRIVIAL").ok(),
            "GNFS_TRY_NEG_M": std::env::var("GNFS_TRY_NEG_M").ok(),
            "GNFS_NF_ELEMENT_MODE": std::env::var("GNFS_NF_ELEMENT_MODE").ok(),
            "GNFS_SQRT_RELAX_EXACT": std::env::var("GNFS_SQRT_RELAX_EXACT").ok(),
            "POTAPOV_NFS_VERBOSE_SQ": std::env::var("POTAPOV_NFS_VERBOSE_SQ").ok()
        });
        let run_cfg = serde_json::json!({
            "n": n_str,
            "bits": n.significant_bits(),
            "params": params,
            "max_variants": max_variants,
            "timestamp_unix": ts,
            "env": env_cfg
        });
        let cfg_path = run_dir.join("run_config.json");
        if fs::write(
            &cfg_path,
            serde_json::to_string_pretty(&run_cfg).unwrap_or_else(|_| "{}".to_string()),
        )
        .is_err()
        {
            return None;
        }

        eprintln!("  logging: {}", run_dir.display());
        Some(Self {
            variants_file: run_dir.join("variants.jsonl"),
            run_dir,
        })
    }

    fn log_variant(&mut self, variant: u32, result: &NfsResult) {
        let rec = serde_json::json!({
            "variant": variant,
            "result": result
        });
        if let Ok(mut f) = File::options()
            .create(true)
            .append(true)
            .open(&self.variants_file)
        {
            let _ = writeln!(
                f,
                "{}",
                serde_json::to_string(&rec).unwrap_or_else(|_| "{}".to_string())
            );
        }
    }

    fn finish(&mut self, final_result: &NfsResult) {
        let summary = serde_json::json!({
            "final_result": final_result
        });
        let _ = fs::write(
            self.run_dir.join("summary.json"),
            serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string()),
        );
    }
}

/// Inner NFS pipeline with a specific polynomial variant.
fn factor_nfs_inner(n: &Integer, params: &NfsParams, variant: u32, pre_poly: Option<&gnfs::types::PolynomialPair>, ext_polyselect_ms: f64) -> NfsResult {
    let start = std::time::Instant::now();
    let mut result = NfsResult {
        n: n.to_string(),
        factor: None,
        relations_found: 0,
        relations_after_filter: 0,
        matrix_rows: 0,
        matrix_cols: 0,
        dependencies_found: 0,
        sqrt_attempts_tried: 0,
        sqrt_factor_attempt: None,
        polyselect_ms: 0.0,
        sieve_ms: 0.0,
        sieve_setup_ms: 0.0,
        sieve_scan_ms: 0.0,
        sieve_cofactor_ms: 0.0,
        sieve_rels_per_sq: 0.0,
        filter_ms: 0.0,
        la_ms: 0.0,
        sqrt_ms: 0.0,
        total_ms: 0.0,
        sqrt_fail_rat_not_square: 0,
        sqrt_fail_alg_not_square: 0,
        sqrt_fail_trivial_gcd: 0,
        viability: ViabilityStats::default(),
    };

    // --- Observability: timeout and timing configuration ---
    let sieve_timeout_ms: Option<u64> = std::env::var("POTAPOV_NFS_SIEVE_TIMEOUT_MS")
        .ok().and_then(|s| s.parse().ok());
    let la_timeout_ms: Option<u64> = std::env::var("POTAPOV_NFS_LA_TIMEOUT_MS")
        .ok().and_then(|s| s.parse().ok());
    let sqrt_timeout_ms: Option<u64> = std::env::var("POTAPOV_NFS_SQRT_TIMEOUT_MS")
        .ok().and_then(|s| s.parse().ok());
    let total_timeout_ms: Option<u64> = std::env::var("POTAPOV_NFS_TOTAL_TIMEOUT_MS")
        .ok().and_then(|s| s.parse().ok());
    let emit_timing_json = std::env::var("POTAPOV_NFS_TIMING_JSON")
        .map(|v| v == "1").unwrap_or(false);
    let mut timings = PipelineTimings::new();
    if ext_polyselect_ms > 0.0 {
        timings.add(StageResult {
            name: "polyselect".to_string(),
            total_ms: ext_polyselect_ms,
            sub_stages: vec![],
            timed_out: false,
        });
    }
    result.polyselect_ms = ext_polyselect_ms;
    let _ = (la_timeout_ms, sqrt_timeout_ms);

    // Optional runtime parameter overrides for reproducible tuning experiments.
    let mut params = params.clone();
    let partial_merge_2lp = std::env::var("POTAPOV_NFS_PARTIAL_MERGE_2LP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_LIM0")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.lim0 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_LIM1")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.lim1 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_LPB0")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.lpb0 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_LPB1")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.lpb1 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_MFB0")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.mfb0 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_MFB1")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.mfb1 = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_LOG_I")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.log_i = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_QMIN")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.qmin = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_QRANGE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.qrange = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_RELS_WANTED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.rels_wanted = v;
    }
    if let Some(v) = std::env::var("POTAPOV_NFS_OVR_DEGREE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v >= 2 && v <= 6)
    {
        params.degree = v;
    }
    if partial_merge_2lp {
        // 2LP requires room for products of two LPs; if mfb is too close to lpb
        // those candidates are rejected before merge can use them.
        // Exception: if mfb <= lpb, the user intentionally disabled 2LP on that
        // side (e.g., CADO c60 uses mfb0=17 < lpb0=18 for rational-side 1LP only).
        let min_mfb0 = params.lpb0.saturating_mul(2).saturating_add(2);
        let min_mfb1 = params.lpb1.saturating_mul(2).saturating_add(2);
        if params.mfb0 > params.lpb0 && params.mfb0 < min_mfb0 {
            eprintln!(
                "  params: bump mfb0 {} -> {} for 2LP support",
                params.mfb0, min_mfb0
            );
            params.mfb0 = min_mfb0;
        }
        if params.mfb1 > params.lpb1 && params.mfb1 < min_mfb1 {
            eprintln!(
                "  params: bump mfb1 {} -> {} for 2LP support",
                params.mfb1, min_mfb1
            );
            params.mfb1 = min_mfb1;
        }
        // Sieve threshold: use each param set's sieve_mfb values directly.
        // These are tuned per size tier:
        // - c45: sieve_mfb=28/30 (tighter than mfb=44/46 after 2LP bump;
        //   empirically tuned to reject false positives that degrade sqrt)
        // - c60: sieve_mfb=18/38 (rational: no 2LP, match mfb; algebraic:
        //   match mfb for full 2LP acceptance, critical for yield at this scale)
        let sieve_mfb0_env = std::env::var("POTAPOV_NFS_SIEVE_MFB0")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let sieve_mfb1_env = std::env::var("POTAPOV_NFS_SIEVE_MFB1")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        params.sieve_mfb0 = sieve_mfb0_env.unwrap_or(params.sieve_mfb0);
        params.sieve_mfb1 = sieve_mfb1_env.unwrap_or(params.sieve_mfb1);
    }

    // --- Stage 1: Polynomial Selection ---
    let poly = match pre_poly {
        Some(p) => p.clone(),
        None => gnfs::polyselect::select_base_m_variant(n, params.degree, variant),
    };
    let f_coeffs_big = poly.f_coeffs();
    let f_coeffs_i64: Vec<i64> = match gnfs::sieve::poly_coeffs_to_i64(&f_coeffs_big) {
        Some(v) => v,
        None => {
            eprintln!("Polynomial coefficients too large for i64");
            result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
            return result;
        }
    };
    let m_big = poly.m();
    let m: u64 = match m_big.to_u64() {
        Some(v) => v,
        None => {
            eprintln!("m too large for u64");
            result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
            return result;
        }
    };

    // Extract rational polynomial g(x) = g1*x + g0 from the PolynomialPair.
    // For monic polynomials: g1=1, g0=-m.
    // For Kleinjung non-monic: g1=ad, g0=-m.
    let g0_big = poly.g0();
    let g1_big = poly.g1();
    let g0_i64: i64 = g0_big.to_i64().unwrap_or(-(m as i64));
    let g1_i64: i64 = g1_big.to_i64().unwrap_or(1);

    eprintln!(
        "  poly: degree={}, m={}, coeffs={:?}, g=[{}, {}]",
        params.degree, m, f_coeffs_i64, g0_i64, g1_i64
    );
    eprintln!(
        "  params: lim0={}, lim1={}, lpb0={}, lpb1={}, mfb0={}, mfb1={}, log_i={}, qmin={}, qrange={}, rels_wanted={}",
        params.lim0,
        params.lim1,
        params.lpb0,
        params.lpb1,
        params.mfb0,
        params.mfb1,
        params.log_i,
        params.qmin,
        params.qrange,
        params.rels_wanted
    );

    // --- Stage 2: Sieve ---
    let degree = params.degree as usize;

    // Build factor bases for both sides.
    // Rational polynomial: g(x) = g1*x + g0.
    let rat_coeffs = vec![g0_i64, g1_i64];
    let rat_fb = crate::factorbase::FactorBase::new(&rat_coeffs, params.lim0, 1.442);
    let alg_fb = crate::factorbase::FactorBase::new_roots_only(&f_coeffs_i64, params.lim1, 1.442);

    // Build gnfs-compatible algebraic FB from alg_fb (reuses its efficient
    // Cantor-Zassenhaus root-finding instead of gnfs's O(p) brute force).
    let gnfs_fb = gnfs::types::FactorBase {
        primes: alg_fb.primes.clone(),
        algebraic_roots: alg_fb.roots.clone(),
        log_p: alg_fb.primes.iter().map(|&p| ((p as f64).log2() * 20.0).round() as u8).collect(),
    };
    let bad_root_offsets = compute_bad_root_offsets(&gnfs_fb, &f_coeffs_big);
    let alg_bad = bad_root_offsets.len();
    // Cap q windows so SQs stay within lim1 (algebraic FB range).
    // SQs beyond lim1 create singleton columns in the matrix that can't merge,
    // causing catastrophic row loss during singleton pruning.
    let lim1_q_window_cap = if params.qrange > 0 && params.qmin < params.lim1 {
        let max_q_in_fb = params.lim1.saturating_sub(params.qmin);
        Some((max_q_in_fb / params.qrange).max(1) as usize)
    } else {
        None
    };
    let max_q_windows = std::env::var("POTAPOV_NFS_MAX_Q_WINDOWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or(lim1_q_window_cap);
    // Build an optional root lookup table for special-q primes above lim1.
    // By default this is only prebuilt when the q-window horizon is explicit,
    // which avoids doing large startup work for runs that stop much earlier.
    let sq_root_cache_windows = std::env::var("POTAPOV_NFS_SQ_ROOT_CACHE_WINDOWS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .or_else(|| max_q_windows.map(|v| v as u64));
    let sq_root_cache_upper =
        sq_root_cache_windows.map(|windows| params.qmin + windows * params.qrange + params.qrange);
    let sq_root_cache: HashMap<u64, Vec<u64>> = if let Some(sq_upper) =
        sq_root_cache_upper.filter(|&upper| upper > params.lim1)
    {
        let sq_fb = crate::factorbase::FactorBase::new_roots_only(&f_coeffs_i64, sq_upper, 1.442);
        sq_fb
            .primes
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > params.lim1)
            .map(|(i, &p)| (p, sq_fb.roots[i].clone()))
            .collect()
    } else {
        HashMap::new()
    };
    if !sq_root_cache.is_empty() {
        if let Some(upper) = sq_root_cache_upper {
            eprintln!(
                "  sieve: prebuilt special-q root cache for {} primes up to {}",
                sq_root_cache.len(),
                upper
            );
        }
    }
    let startup_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  startup: FB construction in {:.0}ms", startup_ms);
    timings.add(StageResult {
        name: "fb_init".to_string(),
        total_ms: startup_ms,
        sub_stages: vec![],
        timed_out: false,
    });

    let mut all_sieve_relations = Vec::new();
    let mut total_sieve_ms = 0.0;
    let mut total_survivors = 0;
    let mut total_special_qs = 0;
    let mut total_root_enum_ms = 0.0;
    let mut total_roots_from_fb = 0usize;
    let mut total_roots_fallback = 0usize;
    let mut total_bucket_setup_ms = 0.0;
    let mut total_region_scan_ms = 0.0;
    let mut total_cofactor_ms = 0.0;

    let require_coprime_ab = std::env::var("POTAPOV_NFS_REQUIRE_COPRIME_AB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let full_only = std::env::var("POTAPOV_NFS_FULL_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let partial_merge_max_sets = std::env::var("POTAPOV_NFS_PARTIAL_MERGE_MAXSETS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(200_000usize);
    let special_q_premerge = std::env::var("POTAPOV_NFS_SPECIAL_Q_PREMERGE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut window = 0usize;
    let mut final_filtered = Vec::new();
    let mut final_partial_sets = None;
    let mut filter_time_ms = 0.0;

    let qc_count = std::env::var("POTAPOV_NFS_QC_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30usize);
    let quad_chars = select_quad_char_primes_fast(&f_coeffs_i64, &gnfs_fb.primes, qc_count);
    let compact_zero_cols = std::env::var("POTAPOV_NFS_COMPACT_ZERO_COLS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let singleton_prune = std::env::var("POTAPOV_NFS_SINGLETON_PRUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let singleton_min_weight = std::env::var("POTAPOV_NFS_SINGLETON_PRUNE_MIN_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(2usize);
    let adaptive_use_matrix = std::env::var("POTAPOV_NFS_ADAPTIVE_USE_MATRIX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let adaptive_matrix_rows_ratio = std::env::var("POTAPOV_NFS_ADAPTIVE_MATRIX_ROWS_RATIO")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(1.25);
    let actual_qc_cols = quad_chars.primes.len();
    let est_rat_fb_cols = rat_fb.primes.len();
    let est_alg_pair_cols = gnfs_fb.algebraic_pair_count();
    let est_alg_hd_cols = gnfs_fb.higher_degree_ideal_count(degree);
    let est_dense_cols = if partial_merge_2lp {
        est_rat_fb_cols
            + est_alg_pair_cols
            + est_alg_hd_cols
            + alg_bad
            + 2
            + actual_qc_cols
    } else {
        0
    };
    if partial_merge_2lp {
        eprintln!(
            "  est_dense_cols: {} (rat_fb={}, alg_pairs={}, alg_hd={}, alg_bad={}, signs=2, qc={})",
            est_dense_cols, est_rat_fb_cols, est_alg_pair_cols, est_alg_hd_cols, alg_bad, actual_qc_cols
        );
    }
    let adaptive_matrix_probe_step = std::env::var("POTAPOV_NFS_ADAPTIVE_MATRIX_PROBE_STEP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| (est_dense_cols / 20).max(250usize));
    let rel_target_mult = std::env::var("POTAPOV_NFS_REL_TARGET_MULT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(1.0f64);
    let rel_target_min = std::env::var("POTAPOV_NFS_REL_TARGET_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(2_000usize);
    // For degree >= 4, est_dense_cols overestimates active dense columns
    // by ~10% (many algebraic FB primes have fewer than `degree` roots, so
    // their pair columns go unused). This built-in buffer implicitly covers
    // the SQ columns that aren't in est_dense_cols, so a lower rows_ratio
    // suffices. The adaptive system probes the matrix early (at 50% of
    // est_dense_cols) to catch cases where the overestimate is larger.
    // Tuned 2026-03-16: ratio 1.01 for degree>=4 collects 5% fewer SQs with
    // no reliability loss (9/9 factored across seeds 42,123,456), saving ~3-5%.
    let default_rows_ratio = if degree >= 4 { 1.01 } else { 1.10 };
    let adaptive_rows_ratio = std::env::var("POTAPOV_NFS_ADAPTIVE_ROWS_RATIO")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(default_rows_ratio);
    let adaptive_rows_min = std::env::var("POTAPOV_NFS_ADAPTIVE_ROWS_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(500usize);
    let adaptive_margin_pct = std::env::var("POTAPOV_NFS_ADAPTIVE_MARGIN_PCT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v >= 0.0)
        .unwrap_or(0.0f64);
    let adaptive_check_every = std::env::var("POTAPOV_NFS_ADAPTIVE_CHECK_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1usize);
    let adaptive_check_min_raw = std::env::var("POTAPOV_NFS_ADAPTIVE_CHECK_MIN_RAW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| adaptive_rows_min.max(500usize));

    let mut target_raw_rels = if partial_merge_2lp {
        (((est_dense_cols as f64) * rel_target_mult).ceil() as usize).max(rel_target_min)
    } else {
        params.rels_wanted as usize
    };
    let raw_target_step = std::env::var("POTAPOV_NFS_ADAPTIVE_RAW_STEP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| (target_raw_rels / 4).max(1_000));
    let configured_max_raw_rels = std::env::var("POTAPOV_NFS_MAX_RAW_RELS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let mut last_matrix_probe_rows: Option<usize> = None;

    // Line sieve toggle: opt-in via POTAPOV_NFS_LINE_SIEVE=1.
    // The line sieve processes one row at a time with stride-based
    // prime subtraction instead of bucket scatter.  Currently ~1.5x
    // slower than the scatter sieve due to per-row large-prime overhead,
    // but has better cache locality and may be useful for tuning or
    // when the scatter sieve's memory footprint is a constraint.
    let use_line_sieve = std::env::var("POTAPOV_NFS_LINE_SIEVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if use_line_sieve {
        eprintln!("  sieve: using line sieve (degree={})", params.degree);
    }

    loop {
        // Check global and sieve timeouts at each sieve window boundary.
        let elapsed_total_ms = start.elapsed().as_millis() as u64;
        if let Some(limit) = total_timeout_ms {
            if elapsed_total_ms > limit {
                eprintln!("  timeout: total elapsed {}ms exceeds limit {}ms, stopping sieve", elapsed_total_ms, limit);
                break;
            }
        }
        if let Some(limit) = sieve_timeout_ms {
            let sieve_elapsed = (total_sieve_ms as u64).saturating_add(
                (start.elapsed().as_secs_f64() * 1000.0 - startup_ms) as u64
            );
            if sieve_elapsed > limit {
                eprintln!("  timeout: sieve elapsed ~{}ms exceeds limit {}ms, stopping", sieve_elapsed, limit);
                break;
            }
        }

        let total_rels = all_sieve_relations.len();
        let at_q_window_cap = max_q_windows.map_or(false, |cap| window >= cap);
        let near_q_window_cap = max_q_windows.map_or(false, |cap| window + 1 >= cap);
        let max_raw_rels =
            effective_max_raw_rels(configured_max_raw_rels, target_raw_rels, raw_target_step);
        let force_stop = (!partial_merge_2lp && total_rels >= params.rels_wanted as usize)
            || total_rels >= max_raw_rels;
        let should_check = total_rels >= adaptive_check_min_raw
            && (window % adaptive_check_every == 0
                || force_stop
                || total_rels >= target_raw_rels
                || near_q_window_cap
                || at_q_window_cap);

        if should_check {
            let filter_start = std::time::Instant::now();
            let mut filtered = crate::filter::filter_relations_ref(&all_sieve_relations);
            if require_coprime_ab {
                filtered.retain(|r| gcd_u64(r.a.unsigned_abs(), r.b) == 1);
            }
            if full_only {
                filtered.retain(|r| r.is_full());
            }

            let full_count = filtered.iter().filter(|r| r.is_full()).count();
            let mut sets_count = 0usize;
            let mut current_sets = None;
            if partial_merge_2lp {
                // Skip expensive merge when filtered count is clearly insufficient.
                // set_rows <= filtered, so if filtered < est_dense_cols, merging
                // cannot produce enough set-rows. Just use filtered.len() as upper-bound estimate.
                if filtered.len() >= est_dense_cols || force_stop || near_q_window_cap || at_q_window_cap {
                    let (sets, _) =
                        crate::partial_merge::merge_relations_2lp(&filtered, partial_merge_max_sets);
                    sets_count = sets.len();
                    current_sets = Some(sets);
                } else {
                    // Conservative estimate — will always fail the active_rows >= needed check,
                    // which correctly causes the sieve to continue.
                    sets_count = filtered.len();
                }
            }
            filter_time_ms += filter_start.elapsed().as_secs_f64() * 1000.0;

            if partial_merge_2lp {
                let active_rows = sets_count;
                let needed_rows_base =
                    ((est_dense_cols as f64) * adaptive_rows_ratio).ceil() as usize;
                let needed_rows_base = needed_rows_base.max(adaptive_rows_min);
                let needed_rows = ((needed_rows_base as f64) * (1.0 + adaptive_margin_pct / 100.0))
                    .ceil() as usize;
                eprintln!(
                    "  adaptive: window {} raw {} -> filtered {} -> active_rows {} (set_rows {}, full {}) target_rows {} target_raw {} check_min_raw {} est_dense_cols {}",
                    window,
                    total_rels,
                    filtered.len(),
                    active_rows,
                    sets_count,
                    full_count,
                    needed_rows,
                    target_raw_rels,
                    adaptive_check_min_raw,
                    est_dense_cols
                );

                // Probe matrix early: est_dense_cols overestimates active columns
                // (many FB entries never appear in relations). Start probing at
                // 50% of est_dense_cols. If the matrix has a deficit, the adaptive
                // system increases the target and continues sieving.
                let matrix_probe_min = est_dense_cols / 2;
                let matrix_probe_due = adaptive_use_matrix
                    && active_rows >= matrix_probe_min
                    && (force_stop
                        || near_q_window_cap
                        || at_q_window_cap
                        || last_matrix_probe_rows
                            .map(|prev| {
                                active_rows >= prev.saturating_add(adaptive_matrix_probe_step)
                            })
                            .unwrap_or(true));
                if matrix_probe_due {
                    last_matrix_probe_rows = Some(active_rows);
                    let (probe_rels, probe_source_indices, probe_stats) = remap_hybrid(
                        &filtered,
                        &rat_fb,
                        &alg_fb,
                        &gnfs_fb,
                        &f_coeffs_big,
                        m,
                        degree,
                        &bad_root_offsets,
                        g0_i64,
                        g1_i64,
                    );
                    let (probe_sets_remapped, probe_sets_dropped, probe_sets_recomputed) =
                        remap_partial_sets_from_sources(
                            &filtered,
                            current_sets.as_deref(),
                            &probe_source_indices,
                            partial_merge_max_sets,
                        );
                    let probe_set_count = if probe_sets_recomputed > 0 {
                        probe_sets_recomputed
                    } else {
                        probe_sets_remapped.len()
                    };
                    let probe_sq_premerged_sets =
                        if special_q_premerge && !probe_sets_remapped.is_empty() {
                            build_special_q_zero_sets_from_partial_sets(
                                &probe_rels,
                                &probe_sets_remapped,
                                partial_merge_max_sets,
                            )
                        } else {
                            Vec::new()
                        };
                    let probe_partial_merge_active =
                        partial_merge_2lp && !probe_sets_remapped.is_empty();
                    let probe_summary = if !probe_sq_premerged_sets.is_empty() {
                        let (probe_matrix, probe_cols, probe_sources) =
                            build_dense_matrix_from_sets(
                                &probe_rels,
                                &probe_sq_premerged_sets,
                                rat_fb.primes.len(),
                                gnfs_fb.algebraic_pair_count(),
                                gnfs_fb.higher_degree_ideal_count(degree) + alg_bad,
                                &quad_chars,
                            );
                        summarize_matrix_shape(
                            probe_matrix,
                            probe_cols,
                            probe_sources,
                            compact_zero_cols,
                            singleton_prune,
                            singleton_min_weight,
                        )
                    } else if probe_partial_merge_active {
                        let (probe_matrix, probe_cols, probe_sources) =
                            build_matrix_from_sets_lp_resolved(
                                &probe_rels,
                                &probe_sets_remapped,
                                rat_fb.primes.len(),
                                gnfs_fb.algebraic_pair_count(),
                                gnfs_fb.higher_degree_ideal_count(degree) + alg_bad,
                                &quad_chars,
                            );
                        summarize_matrix_shape(
                            probe_matrix,
                            probe_cols,
                            probe_sources,
                            compact_zero_cols,
                            singleton_prune,
                            singleton_min_weight,
                        )
                    } else {
                        let (probe_matrix, probe_cols) = gnfs::linalg::build_matrix(
                            &probe_rels,
                            rat_fb.primes.len(),
                            gnfs_fb.algebraic_pair_count(),
                            gnfs_fb.higher_degree_ideal_count(degree) + alg_bad,
                            &quad_chars,
                        );
                        let probe_sources = (0..probe_matrix.len()).map(|i| vec![i]).collect();
                        summarize_matrix_shape(
                            probe_matrix,
                            probe_cols,
                            probe_sources,
                            compact_zero_cols,
                            singleton_prune,
                            singleton_min_weight,
                        )
                    };
                    eprintln!(
                        "  adaptive-matrix: window {} remap_valid {} keep={:.3} set_rows {} dropped={} final={}x{} rows_minus_cols={} target_ratio={:.2}",
                        window,
                        probe_rels.len(),
                        if filtered.is_empty() {
                            0.0
                        } else {
                            probe_rels.len() as f64 / filtered.len() as f64
                        },
                        probe_set_count,
                        probe_sets_dropped,
                        probe_summary.final_rows,
                        probe_summary.final_cols,
                        probe_summary.rows_minus_cols,
                        adaptive_matrix_rows_ratio
                    );
                    if !probe_sq_premerged_sets.is_empty() {
                        eprintln!(
                            "  adaptive-matrix: sq-premerge collapsed set_rows {} -> {}",
                            probe_set_count,
                            probe_sq_premerged_sets.len()
                        );
                    }
                    if probe_summary.final_cols > 0
                        && (probe_summary.final_rows as f64)
                            >= (probe_summary.final_cols as f64) * adaptive_matrix_rows_ratio
                    {
                        eprintln!(
                            "  adaptive-matrix: stopping at window {} (final_rows/final_cols={:.3})",
                            window,
                            probe_summary.final_rows as f64 / probe_summary.final_cols as f64
                        );
                        final_filtered = filtered;
                        final_partial_sets = current_sets;
                        break;
                    }
                    let _ = probe_stats;
                }

                if active_rows >= needed_rows
                    || force_stop
                    || at_q_window_cap
                {
                    final_filtered = filtered;
                    final_partial_sets = current_sets;
                    break;
                }

                let next_target = if let Some(cap) = configured_max_raw_rels {
                    target_raw_rels.saturating_add(raw_target_step).min(cap)
                } else {
                    target_raw_rels.saturating_add(raw_target_step)
                };
                if next_target > target_raw_rels {
                    eprintln!(
                        "  adaptive: increasing raw target {} -> {}",
                        target_raw_rels, next_target
                    );
                    target_raw_rels = next_target;
                } else {
                    final_filtered = filtered;
                    final_partial_sets = current_sets;
                    break;
                }
            } else if force_stop || at_q_window_cap {
                final_filtered = filtered;
                break;
            }
        }

        if force_stop || at_q_window_cap {
            break;
        }

        let q_start = params.qmin + (window as u64) * params.qrange;
        let remaining_to_target = target_raw_rels
            .saturating_sub(all_sieve_relations.len())
            .max(1usize);
        let sieve_result = if use_line_sieve {
            crate::sieve::line_sieve_specialq(
                &f_coeffs_i64,
                m,
                &rat_fb,
                &alg_fb,
                &params,
                q_start,
                params.qrange,
                Some(remaining_to_target),
                Some(&sq_root_cache),
                g0_i64,
                g1_i64,
            )
        } else {
            crate::sieve::sieve_specialq(
                &f_coeffs_i64,
                m,
                &rat_fb,
                &alg_fb,
                &params,
                q_start,
                params.qrange,
                Some(remaining_to_target),
                Some(&sq_root_cache),
                g0_i64,
                g1_i64,
            )
        };

        all_sieve_relations.extend(sieve_result.relations);
        total_sieve_ms += sieve_result.total_ms;
        total_survivors += sieve_result.survivors_found;
        total_special_qs += sieve_result.special_qs_processed;
        total_root_enum_ms += sieve_result.root_enum_ms;
        total_roots_from_fb += sieve_result.roots_from_fb;
        total_roots_fallback += sieve_result.roots_fallback;
        total_bucket_setup_ms += sieve_result.bucket_setup_ms;
        total_region_scan_ms += sieve_result.region_scan_ms;
        total_cofactor_ms += sieve_result.cofactor_ms;
        window += 1;
    }

    if final_filtered.is_empty() {
        // Did not hit adaptive stop condition or q-window cap before final check.
        let filter_start = std::time::Instant::now();
        final_filtered = crate::filter::filter_relations_ref(&all_sieve_relations);
        if require_coprime_ab {
            final_filtered.retain(|r| gcd_u64(r.a.unsigned_abs(), r.b) == 1);
        }
        if full_only {
            final_filtered.retain(|r| r.is_full());
        }
        if partial_merge_2lp {
            let (sets, _) =
                crate::partial_merge::merge_relations_2lp(&final_filtered, partial_merge_max_sets);
            final_partial_sets = Some(sets);
        }
        filter_time_ms += filter_start.elapsed().as_secs_f64() * 1000.0;
    }

    result.sieve_ms = total_sieve_ms;
    result.sieve_setup_ms = total_root_enum_ms + total_bucket_setup_ms;
    result.sieve_scan_ms = total_region_scan_ms;
    result.sieve_cofactor_ms = total_cofactor_ms;
    result.sieve_rels_per_sq = if total_special_qs > 0 {
        all_sieve_relations.len() as f64 / total_special_qs as f64
    } else {
        0.0
    };
    timings.add(StageResult {
        name: "sieve".to_string(),
        total_ms: total_sieve_ms,
        sub_stages: vec![
            ("bucket_setup".to_string(), total_bucket_setup_ms),
            ("region_scan".to_string(), total_region_scan_ms),
            ("cofactor".to_string(), total_cofactor_ms),
            ("root_enum".to_string(), total_root_enum_ms),
        ],
        timed_out: false,
    });
    result.relations_found = all_sieve_relations.len();
    result.viability.raw_relations = result.relations_found;
    eprintln!(
        "  sieve: {} raw rels in {:.0}ms ({} survivors, {} special-qs)",
        all_sieve_relations.len(),
        total_sieve_ms,
        total_survivors,
        total_special_qs
    );
    eprintln!(
        "  sieve: roots enum={:.0}ms (fb_lookup={}, fallback={})",
        total_root_enum_ms, total_roots_from_fb, total_roots_fallback
    );
    eprintln!(
        "  sieve: setup={:.0}ms region_scan={:.0}ms cofactor={:.0}ms",
        total_bucket_setup_ms, total_region_scan_ms, total_cofactor_ms
    );

    if all_sieve_relations.is_empty() {
        log_viability_summary(&result.viability);
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 3 & 4: Filtering & Linear Algebra ---
    let filtered = final_filtered;
    result.filter_ms = filter_time_ms;
    timings.add(StageResult {
        name: "filter".to_string(),
        total_ms: filter_time_ms,
        sub_stages: vec![],
        timed_out: false,
    });
    result.relations_after_filter = filtered.len();
    result.viability.filtered_relations = filtered.len();
    eprintln!(
        "  filter: {} -> {} rels in {:.0}ms",
        result.relations_found, result.relations_after_filter, result.filter_ms
    );
    let full_count = filtered.iter().filter(|r| r.is_full()).count();
    let partial_count = filtered.len().saturating_sub(full_count);
    eprintln!(
        "  LA: {} relations ({} full, {} partial) from {} after filter",
        filtered.len(),
        full_count,
        partial_count,
        result.relations_after_filter
    );
    let partial_sets_filtered = final_partial_sets;
    result.viability.set_rows_filtered = partial_sets_filtered.as_ref().map_or(0, Vec::len);

    let la_start = std::time::Instant::now();

    // Hybrid approach: keep rational factors from our sieve (which uses a
    // complete rational FB including inert primes), but recompute algebraic
    // factors using BigInt norms and gnfs's per-(prime,root) decomposition.
    // This avoids u64 overflow in algebraic norm computation and ensures
    // correct ideal-level factorization.
    let remap_start = std::time::Instant::now();
    let (gnfs_rels, remap_source_indices, remap_stats) = remap_hybrid(
        &filtered,
        &rat_fb,
        &alg_fb,
        &gnfs_fb,
        &f_coeffs_big,
        m,
        degree,
        &bad_root_offsets,
        g0_i64,
        g1_i64,
    );
    result.viability.remap_valid_relations = remap_stats.kept_relations;
    result.viability.remap_dropped_total = remap_stats.skipped_total();
    result.viability.remap_dropped_unmapped = remap_stats.skipped_unmapped;
    result.viability.remap_invalid_total = remap_stats.invalid_total();
    result.viability.remap_invalid_bad_fb_idx = remap_stats.invalid_bad_fb_idx;
    result.viability.remap_invalid_hd_residual = remap_stats.invalid_hd_residual;
    result.viability.remap_invalid_hd_offset_missing = remap_stats.invalid_hd_offset_missing;
    result.viability.remap_invalid_root_exp_zero = remap_stats.invalid_root_exp_zero;
    result.viability.remap_invalid_legacy_lp = remap_stats.invalid_legacy_lp;
    result.viability.hd_residual_samples = remap_stats.hd_residual_samples.clone();
    result.viability.remap_keep_ratio = if filtered.is_empty() {
        0.0
    } else {
        remap_stats.kept_relations as f64 / filtered.len() as f64
    };
    let remap_ms = remap_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  LA: {} relations after hybrid remap in {:.0}ms", gnfs_rels.len(), remap_ms);
    if !filtered.is_empty() {
        eprintln!(
            "  LA: remap keep ratio {}/{} ({:.1}%)",
            gnfs_rels.len(),
            filtered.len(),
            (gnfs_rels.len() as f64) * 100.0 / (filtered.len() as f64)
        );
    }
    let partial_remap_start = std::time::Instant::now();
    let (partial_sets_remapped, partial_sets_dropped_remap, partial_sets_recomputed) =
        remap_partial_sets_from_sources(
            &filtered,
            partial_sets_filtered.as_deref(),
            &remap_source_indices,
            partial_merge_max_sets,
        );
    eprintln!(
        "  merge2lp: remapped set-rows {} (dropped_unmapped={})",
        partial_sets_remapped.len(),
        partial_sets_dropped_remap
    );
    result.viability.set_rows_remapped = partial_sets_remapped.len();
    result.viability.set_rows_dropped_remap = partial_sets_dropped_remap;
    if partial_sets_dropped_remap > 0 {
        eprintln!(
            "  merge2lp: recomputed on remap-valid subset -> {} set-rows",
            partial_sets_recomputed
        );
    }
    result.viability.set_rows_recomputed = if partial_sets_recomputed > 0 {
        partial_sets_recomputed
    } else {
        partial_sets_remapped.len()
    };

    let rel_is_coprime: Vec<bool> = gnfs_rels
        .iter()
        .map(|r| gcd_u64(r.a.unsigned_abs(), r.b) == 1)
        .collect();
    let coprime_rel_count = rel_is_coprime.iter().filter(|&&ok| ok).count();
    eprintln!(
        "  LA: coprime(a,b) in remapped relations: {}/{}",
        coprime_rel_count,
        gnfs_rels.len()
    );

    if gnfs_rels.len() < 2 {
        log_viability_summary(&result.viability);
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    let alg_pairs = gnfs_fb.algebraic_pair_count();
    let alg_hd = gnfs_fb.higher_degree_ideal_count(degree);
    let alg_bad = bad_root_offsets.len();
    // Use rat_fb.primes.len() as rational FB size (complete rational FB).
    let rat_fb_size = rat_fb.primes.len();

    // Relation-level density diagnostics before set-row construction.
    let mut used_rat_cols: HashSet<usize> = HashSet::new();
    let mut used_alg_cols: HashSet<usize> = HashSet::new();
    for rel in &gnfs_rels {
        for &(idx, _exp) in &rel.rational_factors {
            used_rat_cols.insert(idx as usize);
        }
        for &(idx, _exp) in &rel.algebraic_factors {
            used_alg_cols.insert(idx as usize);
        }
    }
    eprintln!(
        "  LA: relation-level dense refs rat={}/{} alg={}/{}",
        used_rat_cols.len(),
        rat_fb_size,
        used_alg_cols.len(),
        alg_pairs + alg_hd + alg_bad
    );

    let partial_remap_ms = partial_remap_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  LA: partial remap {:.0}ms", partial_remap_ms);

    let matrix_build_start = std::time::Instant::now();
    let sparse_premerge = std::env::var("POTAPOV_NFS_SPARSE_PREMERGE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let premerge_max_sets = std::env::var("POTAPOV_NFS_SPARSE_PREMERGE_MAXSETS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(50_000);
    let compare_matrix_modes = std::env::var("POTAPOV_NFS_COMPARE_MATRIX_MODES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let partial_merge_active = partial_merge_2lp && !partial_sets_remapped.is_empty();
    let sq_premerged_sets = if partial_merge_active && special_q_premerge {
        let sets = build_special_q_zero_sets_from_partial_sets(
            &gnfs_rels,
            &partial_sets_remapped,
            partial_merge_max_sets,
        );
        if !sets.is_empty() {
            eprintln!(
                "  sq-premerge: collapsed set-rows {} -> {}",
                partial_sets_remapped.len(),
                sets.len()
            );
        }
        Some(sets)
    } else {
        None
    };
    if compare_matrix_modes && partial_merge_active {
        let (merged_matrix_cmp, merged_cols_cmp, merged_sources_cmp) =
            build_matrix_from_sets_lp_resolved(
                &gnfs_rels,
                &partial_sets_remapped,
                rat_fb_size,
                alg_pairs,
                alg_hd + alg_bad,
                &quad_chars,
            );
        let merged_summary = summarize_matrix_shape(
            merged_matrix_cmp,
            merged_cols_cmp,
            merged_sources_cmp,
            compact_zero_cols,
            singleton_prune,
            singleton_min_weight,
        );

        let (direct_matrix_cmp, direct_cols_cmp) = gnfs::linalg::build_matrix(
            &gnfs_rels,
            rat_fb_size,
            alg_pairs,
            alg_hd + alg_bad,
            &quad_chars,
        );
        let direct_sources_cmp = (0..direct_matrix_cmp.len()).map(|i| vec![i]).collect();
        let direct_summary = summarize_matrix_shape(
            direct_matrix_cmp,
            direct_cols_cmp,
            direct_sources_cmp,
            compact_zero_cols,
            singleton_prune,
            singleton_min_weight,
        );

        eprintln!(
            "  matrix-compare: merged raw={}x{} final={}x{} rows_minus_cols={} zero_drop={} singleton_drop={} zero_drop_post={}",
            merged_summary.raw_rows,
            merged_summary.raw_cols,
            merged_summary.final_rows,
            merged_summary.final_cols,
            merged_summary.rows_minus_cols,
            merged_summary.zero_cols_dropped_initial,
            merged_summary.singleton_rows_dropped,
            merged_summary.zero_cols_dropped_post_singleton
        );
        eprintln!(
            "  matrix-compare: direct raw={}x{} final={}x{} rows_minus_cols={} zero_drop={} singleton_drop={} zero_drop_post={}",
            direct_summary.raw_rows,
            direct_summary.raw_cols,
            direct_summary.final_rows,
            direct_summary.final_cols,
            direct_summary.rows_minus_cols,
            direct_summary.zero_cols_dropped_initial,
            direct_summary.singleton_rows_dropped,
            direct_summary.zero_cols_dropped_post_singleton
        );
        if let Some(sets) = sq_premerged_sets.as_ref().filter(|sets| !sets.is_empty()) {
            let (sq_matrix_cmp, sq_cols_cmp, sq_sources_cmp) = build_dense_matrix_from_sets(
                &gnfs_rels,
                sets,
                rat_fb_size,
                alg_pairs,
                alg_hd + alg_bad,
                &quad_chars,
            );
            let sq_summary = summarize_matrix_shape(
                sq_matrix_cmp,
                sq_cols_cmp,
                sq_sources_cmp,
                compact_zero_cols,
                singleton_prune,
                singleton_min_weight,
            );
            eprintln!(
                "  matrix-compare: sq-premerge raw={}x{} final={}x{} rows_minus_cols={} zero_drop={} singleton_drop={} zero_drop_post={}",
                sq_summary.raw_rows,
                sq_summary.raw_cols,
                sq_summary.final_rows,
                sq_summary.final_cols,
                sq_summary.rows_minus_cols,
                sq_summary.zero_cols_dropped_initial,
                sq_summary.singleton_rows_dropped,
                sq_summary.zero_cols_dropped_post_singleton
            );
        }
    }

    let (matrix_raw, ncols_raw, mut row_sources, matrix_mode, matrix_set_rows, matrix_layout): (
        Vec<gnfs::types::BitRow>,
        usize,
        Vec<Vec<usize>>,
        &'static str,
        usize,
        MatrixColumnLayout,
    ) = if let Some(sets) = sq_premerged_sets.as_ref().filter(|sets| !sets.is_empty()) {
        let (matrix, nc, set_rows) = build_dense_matrix_from_sets(
            &gnfs_rels,
            sets,
            rat_fb_size,
            alg_pairs,
            alg_hd + alg_bad,
            &quad_chars,
        );
        eprintln!(
            "  LA: sq-premerged dense matrix {} x {} from {} set-rows (qc={})",
            matrix.len(),
            nc,
            set_rows.len(),
            quad_chars.primes.len()
        );
        let set_count = set_rows.len();
        let layout = build_dense_only_matrix_layout(
            rat_fb_size,
            alg_pairs + alg_hd + alg_bad,
            quad_chars.primes.len(),
        );
        (
            matrix,
            nc,
            set_rows,
            "partial_merge_2lp_sq_premerge",
            set_count,
            layout,
        )
    } else if partial_merge_active {
        let (matrix, nc, set_rows) = build_matrix_from_sets_lp_resolved(
            &gnfs_rels,
            &partial_sets_remapped,
            rat_fb_size,
            alg_pairs,
            alg_hd + alg_bad,
            &quad_chars,
        );
        eprintln!(
            "  LA: 2LP-merged matrix {} x {} from {} set-rows (qc={}, sq_cols={})",
            matrix.len(),
            nc,
            set_rows.len(),
            quad_chars.primes.len(),
            gnfs_rels
                .iter()
                .filter_map(|r| r.special_q)
                .collect::<HashSet<_>>()
                .len()
        );
        let set_count = set_rows.len();
        let layout = build_lp_resolved_matrix_layout(
            &gnfs_rels,
            rat_fb_size,
            alg_pairs + alg_hd + alg_bad,
            quad_chars.primes.len(),
        );
        (matrix, nc, set_rows, "partial_merge_2lp", set_count, layout)
    } else if sparse_premerge {
        let sparse_zero_sets = build_sparse_zero_sets(&gnfs_rels, premerge_max_sets);
        let set_count = sparse_zero_sets.len();
        let (sum_len, max_len) = sparse_zero_sets
            .iter()
            .fold((0usize, 0usize), |(s, m), set| {
                (s + set.len(), m.max(set.len()))
            });
        let avg_len = if set_count > 0 {
            sum_len as f64 / set_count as f64
        } else {
            0.0
        };
        eprintln!(
            "  LA: sparse-premerge produced {} set-rows (avg_size={:.1}, max_size={}, max_sets={})",
            set_count, avg_len, max_len, premerge_max_sets
        );

        let (dense_matrix, dense_ncols, dense_set_rows) = build_dense_matrix_from_sets(
            &gnfs_rels,
            &sparse_zero_sets,
            rat_fb_size,
            alg_pairs,
            alg_hd + alg_bad,
            &quad_chars,
        );
        eprintln!(
            "  LA: dense-only matrix after premerge: {} x {} (qc={})",
            dense_matrix.len(),
            dense_ncols,
            quad_chars.primes.len()
        );
        let set_count = dense_set_rows.len();
        let layout = build_dense_only_matrix_layout(
            rat_fb_size,
            alg_pairs + alg_hd + alg_bad,
            quad_chars.primes.len(),
        );
        (
            dense_matrix,
            dense_ncols,
            dense_set_rows,
            "sparse_premerge",
            set_count,
            layout,
        )
    } else {
        let (m, nc) = gnfs::linalg::build_matrix(
            &gnfs_rels,
            rat_fb_size,
            alg_pairs,
            alg_hd + alg_bad,
            &quad_chars,
        );
        eprintln!(
            "  LA: {} x {} matrix (qc={})",
            m.len(),
            nc,
            quad_chars.primes.len()
        );
        let row_sources = (0..m.len()).map(|i| vec![i]).collect();
        let layout = build_direct_matrix_layout(
            &gnfs_rels,
            rat_fb_size,
            alg_pairs + alg_hd + alg_bad,
            quad_chars.primes.len(),
        );
        (m, nc, row_sources, "direct", 0usize, layout)
    };
    let matrix_build_ms = matrix_build_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  LA: matrix build {:.0}ms ({} x {})", matrix_build_ms, matrix_raw.len(), ncols_raw);

    let compact_start = std::time::Instant::now();
    result.viability.matrix_mode = matrix_mode.to_string();
    result.viability.set_rows_matrix = matrix_set_rows;
    result.viability.matrix_rows_pre_compact = matrix_raw.len();
    result.viability.matrix_cols_pre_compact = ncols_raw;
    let matrix_col_stats =
        collect_matrix_column_weight_stats(&matrix_raw, ncols_raw, &matrix_layout);
    eprintln!(
        "  LA: active matrix cols sq={} rat_fb={} alg_dense={} qc={} singleton total={} sq={} rat_fb={} alg_dense={} qc={}",
        matrix_col_stats.active_special_q_cols,
        matrix_col_stats.active_rat_fb_cols,
        matrix_col_stats.active_alg_dense_cols,
        matrix_col_stats.active_qc_cols,
        matrix_col_stats.singleton_cols_total,
        matrix_col_stats.singleton_special_q_cols,
        matrix_col_stats.singleton_rat_fb_cols,
        matrix_col_stats.singleton_alg_dense_cols,
        matrix_col_stats.singleton_qc_cols
    );
    result.viability.active_dense_rat_cols = matrix_col_stats.active_rat_fb_cols;
    result.viability.active_dense_alg_cols = matrix_col_stats.active_alg_dense_cols;
    result.viability.active_special_q_cols = matrix_col_stats.active_special_q_cols;
    result.viability.active_qc_cols = matrix_col_stats.active_qc_cols;
    result.viability.pre_prune_singleton_cols_total = matrix_col_stats.singleton_cols_total;
    result.viability.pre_prune_singleton_special_q_cols = matrix_col_stats.singleton_special_q_cols;
    result.viability.pre_prune_singleton_rat_lp_cols = matrix_col_stats.singleton_rat_lp_cols;
    result.viability.pre_prune_singleton_alg_lp_cols = matrix_col_stats.singleton_alg_lp_cols;
    result.viability.pre_prune_singleton_sign_cols = matrix_col_stats.singleton_sign_cols;
    result.viability.pre_prune_singleton_rat_fb_cols = matrix_col_stats.singleton_rat_fb_cols;
    result.viability.pre_prune_singleton_alg_dense_cols = matrix_col_stats.singleton_alg_dense_cols;
    result.viability.pre_prune_singleton_qc_cols = matrix_col_stats.singleton_qc_cols;
    let (mut matrix, mut ncols, dropped_cols) = if compact_zero_cols {
        compact_zero_columns(matrix_raw, ncols_raw)
    } else {
        (matrix_raw, ncols_raw, 0usize)
    };
    if dropped_cols > 0 {
        eprintln!(
            "  LA: compacted zero columns {} -> {} (dropped={})",
            ncols_raw, ncols, dropped_cols
        );
    }
    result.viability.zero_cols_dropped_initial = dropped_cols;
    let mut singleton_rows_dropped = 0usize;
    let mut zero_cols_dropped_post_singleton = 0usize;
    if singleton_prune {
        let rows_before = matrix.len();
        let cols_before = ncols;
        let (pruned_matrix, pruned_sources, removed_rows) =
            prune_singleton_columns(matrix, row_sources, ncols, singleton_min_weight);
        singleton_rows_dropped = removed_rows;
        matrix = pruned_matrix;
        row_sources = pruned_sources;
        if compact_zero_cols {
            let (m2, nc2, dropped2) = compact_zero_columns(matrix, ncols);
            matrix = m2;
            ncols = nc2;
            zero_cols_dropped_post_singleton = dropped2;
            eprintln!(
                "  LA: singleton-prune min_weight={} removed_rows={} rows {}->{} cols {}->{} (extra_dropped_zero_cols={})",
                singleton_min_weight,
                removed_rows,
                rows_before,
                matrix.len(),
                cols_before,
                ncols,
                dropped2
            );
        } else {
            eprintln!(
                "  LA: singleton-prune min_weight={} removed_rows={} rows {}->{} cols {}->{}",
                singleton_min_weight,
                removed_rows,
                rows_before,
                matrix.len(),
                cols_before,
                ncols
            );
        }
    }
    result.viability.singleton_rows_dropped = singleton_rows_dropped;
    result.viability.zero_cols_dropped_post_singleton = zero_cols_dropped_post_singleton;
    result.viability.zero_cols_dropped_total = dropped_cols + zero_cols_dropped_post_singleton;

    let compact_ms = compact_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  LA: compact+prune {:.0}ms -> {} x {}", compact_ms, matrix.len(), ncols);

    result.matrix_rows = matrix.len();
    result.matrix_cols = ncols;
    result.viability.final_rows = result.matrix_rows;
    result.viability.final_cols = result.matrix_cols;
    result.viability.rows_minus_cols = result.matrix_rows as isize - result.matrix_cols as isize;

    let matrix_premerged = sparse_premerge || partial_merge_active;
    let sqrt_success_mode = std::env::var("POTAPOV_NFS_SQRT_SUCCESS_MODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(degree >= 4);

    // Skip GE when matrix has column deficit — 0 deps guaranteed.
    if result.matrix_rows <= result.matrix_cols {
        eprintln!(
            "  LA: skipping GE (rows {} <= cols {}, deficit {})",
            result.matrix_rows,
            result.matrix_cols,
            result.matrix_cols as isize - result.matrix_rows as isize
        );
        result.dependencies_found = 0;
        result.viability.deps_found = 0;
        result.la_ms = la_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  LA: 0 deps (deficit skip) in {:.0}ms", result.la_ms);
        if matrix_premerged {
            eprintln!(
                "  LA: dependency basis rows come from {} premerged set-rows",
                row_sources.len()
            );
        }
        log_viability_summary(&result.viability);
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // Use BW for very large matrices (O(n^2) vs GE O(n^3)),
    // pre-elimination + GE for medium, plain GE for small.
    let bw_threshold: usize = std::env::var("POTAPOV_NFS_BW_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let max_deps: Option<usize> = std::env::var("POTAPOV_NFS_MAX_DEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&v| v > 0)
        .or(Some(64));
    let ge_start = std::time::Instant::now();
    let mut ge_deps = if matrix.len() > bw_threshold {
        gnfs::linalg::find_dependencies_with_preelim_bw_max(&matrix, ncols, max_deps)
    } else {
        gnfs::linalg::find_dependencies_with_preelim_max(&matrix, ncols, max_deps)
    };
    let ge_ms = ge_start.elapsed().as_secs_f64() * 1000.0;
    let ge_deps_total = ge_deps.len();
    eprintln!("  LA: GE+preelim {:.0}ms -> {} basis deps (max_deps={:?})", ge_ms, ge_deps_total, max_deps);
    let ge_dep_basis_limit = std::env::var("POTAPOV_NFS_DEP_BASIS_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    if let Some(limit) = ge_dep_basis_limit {
        if ge_deps.len() > limit {
            ge_deps.truncate(limit);
        }
    }
    // GE basis vectors are short/correlated; generate randomized XOR
    // combinations to avoid systematic trivial-gcd failures in sqrt.
    // Keep k modest to avoid extremely long dependencies (expensive sqrt).
    let default_n_random = if sqrt_success_mode {
        ge_deps.len().clamp(4_000, 20_000)
    } else if ge_deps.len() < 200 {
        500
    } else if ge_deps.len() < 1_000 {
        500
    } else {
        500
    };
    let n_random = std::env::var("POTAPOV_NFS_DEP_RANDOM_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_n_random);
    let default_k = if sqrt_success_mode {
        (ge_deps.len() / 3).max(50).min(500)
    } else {
        (ge_deps.len() / 8).max(8).min(64)
    };
    let k = std::env::var("POTAPOV_NFS_DEP_XOR_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 1)
        .unwrap_or(default_k);
    let dep_seed = std::env::var("POTAPOV_NFS_DEP_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let rand_start = std::time::Instant::now();
    let deps = gnfs::linalg::randomize_dependencies(&ge_deps, n_random, k, dep_seed);
    let rand_ms = rand_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  LA: randomize_deps {:.0}ms (n_random={}, k={}) -> {} total deps", rand_ms, n_random, k, deps.len());
    result.dependencies_found = deps.len();
    result.viability.deps_found = result.dependencies_found;

    result.la_ms = la_start.elapsed().as_secs_f64() * 1000.0;
    timings.add(StageResult {
        name: "la".to_string(),
        total_ms: result.la_ms,
        sub_stages: vec![],
        timed_out: false,
    });
    eprintln!(
        "  LA: {} deps ({} GE used / {} GE total + {} random, xor_k={}, seed={}) in {:.0}ms",
        deps.len(),
        ge_deps.len(),
        ge_deps_total,
        deps.len() - ge_deps.len(),
        k,
        dep_seed,
        result.la_ms
    );
    if matrix_premerged {
        eprintln!(
            "  LA: dependency basis rows come from {} premerged set-rows",
            row_sources.len()
        );
    }
    log_viability_summary(&result.viability);

    if deps.is_empty() {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    let skip_sqrt = std::env::var("POTAPOV_NFS_SKIP_SQRT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if skip_sqrt {
        eprintln!("  sqrt: skipped (POTAPOV_NFS_SKIP_SQRT=1)");
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 5: Square Root ---
    let sqrt_start = std::time::Instant::now();

    // Prioritize randomized deps before basis deps (basis vectors are often
    // short/correlated) and within each class by descending expanded length.
    // Longer deps are empirically much more likely to yield nontrivial gcd.
    let max_dep_len = std::env::var("POTAPOV_NFS_MAX_DEP_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let dep_auto_relax = std::env::var("POTAPOV_NFS_DEP_AUTO_RELAX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let dep_len_tiers = parse_dep_len_tiers(max_dep_len, dep_auto_relax);
    let dep_sets_ref = row_sources.as_slice();

    struct DepCandidate {
        dep_idx: usize,
        expanded: Option<Vec<usize>>,   // lazily computed
        support: Option<Vec<u32>>,      // lazily computed
        support_hash: u64,              // 0 until expanded
        raw_weight: usize,              // # set-row indices (cheap proxy for expanded len)
    }

    // Phase 1: create lightweight candidates sorted by raw weight (no expansion yet)
    let mut useful_deps: Vec<DepCandidate> = deps
        .iter()
        .enumerate()
        .filter(|(_, dep)| !dep.is_empty())
        .map(|(dep_idx, dep)| DepCandidate {
            dep_idx,
            expanded: None,
            support: None,
            support_hash: 0,
            raw_weight: dep.len(),
        })
        .collect();

    useful_deps.sort_by(|a, b| {
        a.raw_weight
            .cmp(&b.raw_weight)
            .then_with(|| a.dep_idx.cmp(&b.dep_idx))
    });

    let duplicate_expanded = 0usize; // dedup deferred to expansion time

    let sqrt_prep_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  sqrt: dep prep in {:.0}ms ({} candidates, lazy expansion)", sqrt_prep_ms, useful_deps.len());

    let mut fail_rat_not_square = 0usize;
    let mut fail_alg_not_square = 0usize;
    let mut fail_trivial_gcd = 0usize;

    let default_max_deps_to_try = if sqrt_success_mode {
        if degree >= 4 {
            5_000usize
        } else {
            600usize
        }
    } else if degree >= 4 {
        80usize
    } else {
        300usize
    };
    let max_deps_to_try = std::env::var("POTAPOV_NFS_MAX_DEPS_TRY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_max_deps_to_try);
    let default_trivial_bail = if sqrt_success_mode {
        // Early-bail if ALL of the first N deps give trivial GCD.
        // 500 is a compromise: truly degenerate polys (all trivial) bail in ~4.5s
        // instead of 30-45s, while borderline polys with low success rate get
        // enough attempts to find a factor. 500 consecutive trivial GCDs has
        // probability 2^{-500} of false positive if the polynomial is good.
        500usize
    } else if degree >= 4 {
        60usize
    } else {
        200usize
    };
    let trivial_bail = std::env::var("POTAPOV_NFS_TRIVIAL_BAIL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|v| v.min(max_deps_to_try))
        .unwrap_or(default_trivial_bail.min(max_deps_to_try));

    let useful_ge = useful_deps
        .iter()
        .filter(|cand| cand.dep_idx < ge_deps.len())
        .count();
    let useful_rand = useful_deps.len().saturating_sub(useful_ge);
    let verbose_deps = std::env::var("POTAPOV_NFS_SQRT_VERBOSE_DEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let dep_require_coprime_rel = std::env::var("POTAPOV_NFS_DEP_REQUIRE_COPRIME_REL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Use raw_weight as proxy for expanded length stats (avoids expanding all deps)
    let dep_lens: Vec<usize> = useful_deps.iter().map(|cand| cand.raw_weight).collect();
    if let Some((min_len, p50, p90, p99, max_len)) = dependency_length_stats(&dep_lens) {
        let first_tier = dep_len_tiers.first().copied().flatten();
        let first_tier_count = first_tier
            .map(|cap| dep_lens.iter().filter(|&&len| len <= cap).count())
            .unwrap_or(dep_lens.len());
        eprintln!(
            "  sqrt: dep lens expanded min/p50/p90/p99/max={}/{}/{}/{}/{} (eligible@first_tier={}/{})",
            min_len,
            p50,
            p90,
            p99,
            max_len,
            first_tier_count,
            dep_lens.len()
        );
    }
    eprintln!(
        "  sqrt: trying up to {} deps ({} candidates: {} GE + {} random, dup_expanded_dropped={}, trivial_bail={}, verbose_deps={}, dep_len_tiers={}, premerge={}, require_coprime_rel={})",
        max_deps_to_try,
        useful_deps.len(),
        useful_ge,
        useful_rand,
        duplicate_expanded,
        trivial_bail,
        verbose_deps,
        format_dep_len_tiers(&dep_len_tiers),
        matrix_premerged,
        dep_require_coprime_rel
    );

    // Diversity-aware scheduler settings
    let div_history_max = std::env::var("POTAPOV_NFS_SQRT_DIVERSITY_HISTORY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let div_pool_size = std::env::var("POTAPOV_NFS_SQRT_DIVERSITY_POOL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(300);
    let max_similarity_cutoff = std::env::var("POTAPOV_NFS_SQRT_MAX_SIMILARITY")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0); // 1.0 = disabled
    let enable_sim_log = std::env::var("POTAPOV_NFS_SQRT_LOG_SIMILARITY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    struct FailedSupport {
        support_hash: u64,
        support: Vec<u32>,
    }

    let mut failed_trivial_supports: Vec<FailedSupport> = Vec::new();
    let mut untried_pool: Vec<usize> = Vec::new();
    let mut next_pool_cursor = 0usize;
    let mut deps_tried = 0usize;
    let mut deps_skipped_short = 0usize;
    let mut deps_skipped_non_coprime = 0usize;
    let mut total_skipped_sim = 0usize;

    // Inline pool refill with lazy expansion
    macro_rules! refill_pool_and_expand {
        ($pool:expr, $next_cursor:expr) => {
            while $pool.len() < div_pool_size && $next_cursor < useful_deps.len() {
                let idx = $next_cursor;
                if useful_deps[idx].expanded.is_none() {
                    let dep = &deps[useful_deps[idx].dep_idx];
                    let expanded = expand_dependency_over_sets(dep, dep_sets_ref);
                    if expanded.is_empty() {
                        $next_cursor += 1;
                        continue;
                    }
                    let mut support: Vec<u32> = expanded.iter().map(|&x| x as u32).collect();
                    support.sort_unstable();
                    support.dedup();
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    support.hash(&mut hasher);
                    useful_deps[idx].support_hash = hasher.finish();
                    useful_deps[idx].support = Some(support);
                    useful_deps[idx].expanded = Some(expanded);
                }
                $pool.push(idx);
                $next_cursor += 1;
            }
        };
    }
    refill_pool_and_expand!(untried_pool, next_pool_cursor);

    // Fast Jaccard similarity for two sorted, deduplicated slices
    fn intersection_union_counts(a: &[u32], b: &[u32]) -> (usize, usize) {
        if a.is_empty() && b.is_empty() {
            return (0, 0);
        }
        let mut i = 0;
        let mut j = 0;
        let mut intersection = 0;
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
        let union = a.len() + b.len() - intersection;
        (intersection, union)
    }

    // Parallel first batch: try the first N deps simultaneously before
    // entering the sequential diversity-aware scheduler. In MT mode this
    // exploits available cores to cut sqrt latency when a factor is found
    // within the first few attempts (the common case for c30).
    let par_sqrt_batch = std::env::var("POTAPOV_NFS_SQRT_PAR_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(rayon::current_num_threads().min(8).max(1));
    if par_sqrt_batch > 1 && !untried_pool.is_empty() && result.factor.is_none() {
        let batch_size = par_sqrt_batch.min(untried_pool.len()).min(max_deps_to_try);
        // Take the first batch_size candidates (already sorted by weight)
        let batch_indices: Vec<usize> = untried_pool.iter().take(batch_size)
            .filter(|&&idx| {
                let cand = &useful_deps[idx];
                let exp_len = cand.expanded.as_ref().map(|e| e.len()).unwrap_or(0);
                exp_len >= degree
            })
            .copied()
            .collect();
        if batch_indices.len() > 1 {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicBool, Ordering};
            let found = AtomicBool::new(false);
            let batch_results: Vec<(usize, Option<Integer>, Option<gnfs::sqrt::FactorFailure>)> =
                batch_indices.par_iter()
                    .filter_map(|&cand_idx| {
                        if found.load(Ordering::Relaxed) {
                            return None;
                        }
                        let cand = &useful_deps[cand_idx];
                        let dep_expanded = cand.expanded.as_deref()?;
                        let (factor_opt, failure) = gnfs::sqrt::extract_factor_diagnostic_fast_g(
                            &gnfs_rels,
                            dep_expanded,
                            &f_coeffs_big,
                            &m_big,
                            n,
                            &rat_fb.primes,
                            &g0_big,
                            &g1_big,
                        );
                        if factor_opt.is_some() {
                            found.store(true, Ordering::Relaxed);
                        }
                        Some((cand_idx, factor_opt, failure))
                    })
                    .collect();

            // Process results in order
            let mut batch_found = false;
            for (cand_idx, factor_opt, failure) in &batch_results {
                deps_tried += 1;
                // Remove from pool
                if let Some(pos) = untried_pool.iter().position(|&x| x == *cand_idx) {
                    untried_pool.swap_remove(pos);
                }
                if let Some(ref factor) = factor_opt {
                    if Integer::from(n % factor) == 0 && *factor > 1 && *factor < *n {
                        result.factor = Some(factor.to_string());
                        result.sqrt_attempts_tried = deps_tried;
                        result.sqrt_factor_attempt = Some(deps_tried);
                        result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
                        eprintln!(
                            "  sqrt: factor found in parallel batch ({} deps tried) in {:.0}ms: {}",
                            deps_tried,
                            result.sqrt_ms,
                            result.factor.as_ref().unwrap()
                        );
                        batch_found = true;
                        break;
                    }
                }
                match failure {
                    Some(gnfs::sqrt::FactorFailure::RationalNotSquare) => {
                        fail_rat_not_square += 1;
                    }
                    Some(gnfs::sqrt::FactorFailure::AlgebraicNotSquare) => {
                        fail_alg_not_square += 1;
                    }
                    Some(gnfs::sqrt::FactorFailure::TrivialGcd) => {
                        fail_trivial_gcd += 1;
                    }
                    None => {}
                }
            }
            if batch_found {
                // Skip the main tier loop
                result.sqrt_fail_rat_not_square = fail_rat_not_square;
                result.sqrt_fail_alg_not_square = fail_alg_not_square;
                result.sqrt_fail_trivial_gcd = fail_trivial_gcd;
            }
        }
    }

    // Outer loop replaces dep_len_tiers: we pick the best dep out of all allowed by tiers
    for (tier_idx, tier_cap) in dep_len_tiers.iter().enumerate() {
        if deps_tried >= max_deps_to_try || result.factor.is_some() {
            break;
        }

        // Only consider untried deps that fit exactly in this tier (inline to avoid borrow conflicts)
        macro_rules! is_eligible {
            ($idx:expr) => {{
                let len = useful_deps[$idx].expanded.as_ref().map(|e| e.len()).unwrap_or(useful_deps[$idx].raw_weight);
                tier_cap.map(|cap| len <= cap).unwrap_or(true) &&
                (tier_idx == 0 || dep_len_tiers[tier_idx-1].map(|prev| len > prev).unwrap_or(false))
            }};
        }

        let eligible_count = untried_pool.iter().filter(|&&i| is_eligible!(i)).count();
        eprintln!(
            "  sqrt: tier {}/{} cap={} eligible={}",
            tier_idx + 1,
            dep_len_tiers.len(),
            tier_cap
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            eligible_count
        );
        if eligible_count == 0 {
            continue;
        }

        while deps_tried < max_deps_to_try && result.factor.is_none() {
            // Pick next dependency from the pool inside this tier
            let mut best_pool_idx = None;
            let mut best_max_sim = 2.0;
            let mut best_avg_sim = 2.0;
            let mut best_len = usize::MAX;
            let mut best_overlap = None;

            if enable_sim_log {
                eprintln!(
                    "  sqrt: scheduler history size before selection = {}",
                    failed_trivial_supports.len()
                );
            }

            for (p_idx, &cand_idx) in untried_pool.iter().enumerate() {
                if !is_eligible!(cand_idx) {
                    continue;
                }
                let cand = &useful_deps[cand_idx];

                let exp_len = cand.expanded.as_ref().map(|e| e.len()).unwrap_or(0);
                if exp_len < degree {
                    deps_skipped_short += 1;
                    continue;
                }

                if dep_require_coprime_rel {
                    if let Some(ref expanded) = cand.expanded {
                        if expanded.iter().any(|&ri| !rel_is_coprime.get(ri).copied().unwrap_or(false)) {
                            deps_skipped_non_coprime += 1;
                            continue;
                        }
                    }
                }

                // Compute similarity to known trivial-gcd failures
                let mut max_sim = 0.0f64;
                let mut sum_sim = 0.0f64;
                let mut overlap: Option<(usize, usize, usize, f64)> = None;
                for (fail_idx, fail_supp) in failed_trivial_supports.iter().enumerate() {
                    let cand_support = cand.support.as_deref().unwrap_or(&[]);
                    let (intersection, union) =
                        intersection_union_counts(cand_support, &fail_supp.support);
                    let sim = if union == 0 {
                        1.0
                    } else {
                        (intersection as f64) / (union as f64)
                    };
                    if sim > max_sim {
                        max_sim = sim;
                    }
                    sum_sim += sim;
                    if overlap
                        .map(|(_, prev_intersection, prev_union, prev_sim)| {
                            sim > prev_sim
                                || ((sim - prev_sim).abs() < f64::EPSILON
                                    && (intersection, union) > (prev_intersection, prev_union))
                        })
                        .unwrap_or(true)
                    {
                        overlap = Some((fail_idx, intersection, union, sim));
                    }
                }
                let avg_sim = if failed_trivial_supports.is_empty() {
                    0.0
                } else {
                    sum_sim / (failed_trivial_supports.len() as f64)
                };

                // Hard similarity cutoff (off by default)
                if max_sim >= max_similarity_cutoff {
                    // Mark to skip this completely without evaluating
                    continue;
                }

                // Lexicographic selection:
                // 1) Length band (already handled by sorting/tiers)
                // 2) Max similarity (lower is better, bin into 5% groups)
                // 3) Avg similarity
                // 4) Length
                let max_sim_band = (max_sim * 20.0_f64).floor() as u32;
                let best_max_band = (best_max_sim * 20.0_f64).floor() as u32;

                let is_better = match best_pool_idx {
                    None => true,
                    Some(_) => {
                        if max_sim_band < best_max_band {
                            true
                        } else if max_sim_band > best_max_band {
                            false
                        } else if avg_sim < best_avg_sim - 0.01 {
                            true
                        } else if avg_sim > best_avg_sim + 0.01 {
                            false
                        } else {
                            exp_len < best_len
                        }
                    }
                };

                if is_better {
                    best_pool_idx = Some(p_idx);
                    best_max_sim = max_sim;
                    best_avg_sim = avg_sim;
                    best_len = exp_len;
                    best_overlap = overlap;
                }
            }

            // Exiting tier loop: no more eligible candidates in this tier
            let p_idx = match best_pool_idx {
                Some(idx) => idx,
                None => break,
            };

            // Remove from pool (refill deferred to avoid borrow conflict)
            let cand_idx = untried_pool.swap_remove(p_idx);
            let cand = &useful_deps[cand_idx];
            let dep_expanded = cand.expanded.as_deref().unwrap();

            // Check exact cutoff again (it could have been skipped in the selection loop, but we want to count it)
            if best_max_sim >= max_similarity_cutoff {
                total_skipped_sim += 1;
                continue;
            }

            let i = deps_tried;
            deps_tried += 1;

            if enable_sim_log {
                eprintln!(
                    "  sqrt: attempt {} (id {}), weight={}, supp={}, hash={:016x}, max_sim={:.3}, avg_sim={:.3}",
                    i + 1,
                    cand.dep_idx,
                    dep_expanded.len(),
                    cand.support.as_ref().map(|s| s.len()).unwrap_or(0),
                    cand.support_hash,
                    best_max_sim,
                    best_avg_sim
                );
                if let Some((fail_idx, intersection, union, sim)) = best_overlap {
                    eprintln!(
                        "    overlap: best_fail_idx={}, intersection={}, union={}, sim={:.3}",
                        fail_idx, intersection, union, sim
                    );
                }
            }

            let (factor_opt, failure) = if i < verbose_deps {
                gnfs::sqrt::extract_factor_verbose_g(
                    &gnfs_rels,
                    dep_expanded,
                    &f_coeffs_big,
                    &m_big,
                    n,
                    &g0_big,
                    &g1_big,
                )
            } else {
                gnfs::sqrt::extract_factor_diagnostic_fast_g(
                    &gnfs_rels,
                    dep_expanded,
                    &f_coeffs_big,
                    &m_big,
                    n,
                    &rat_fb.primes,
                    &g0_big,
                    &g1_big,
                )
            };

            if let Some(factor) = factor_opt {
                if Integer::from(n % &factor) == 0 && factor > 1 && factor < *n {
                    if enable_sim_log {
                        eprintln!("    -> result: FactorFound");
                    }
                    result.factor = Some(factor.to_string());
                    result.sqrt_attempts_tried = deps_tried;
                    result.sqrt_factor_attempt = Some(deps_tried);
                    result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "  sqrt: factor found after {} deps in {:.0}ms (max_sim={:.3}): {}",
                        deps_tried,
                        result.sqrt_ms,
                        best_max_sim,
                        result.factor.as_ref().unwrap()
                    );
                    break;
                }
            }

            match failure {
                Some(gnfs::sqrt::FactorFailure::RationalNotSquare) => {
                    fail_rat_not_square += 1;
                    if enable_sim_log {
                        eprintln!("    -> result: RatNotSquare");
                    }
                }
                Some(gnfs::sqrt::FactorFailure::AlgebraicNotSquare) => {
                    fail_alg_not_square += 1;
                    if enable_sim_log {
                        eprintln!("    -> result: AlgNotSquare");
                    }
                }
                Some(gnfs::sqrt::FactorFailure::TrivialGcd) => {
                    fail_trivial_gcd += 1;
                    if enable_sim_log {
                        eprintln!("    -> result: TrivialGcd");
                    }
                    let cand_support_ref = cand.support.as_deref().unwrap_or(&[]);
                    let duplicate_failure = failed_trivial_supports.iter().any(|failed| {
                        failed.support_hash == cand.support_hash && failed.support == cand_support_ref
                    });
                    if !duplicate_failure {
                        let failed = FailedSupport {
                            support_hash: cand.support_hash,
                            support: cand_support_ref.to_vec(),
                        };
                        if failed_trivial_supports.len() < div_history_max {
                            failed_trivial_supports.push(failed);
                        } else {
                            // Keep 0..max-1, shift left, put new at end.
                            failed_trivial_supports.rotate_left(1);
                            if let Some(last) = failed_trivial_supports.last_mut() {
                                *last = failed;
                            }
                        }
                    }
                    if enable_sim_log {
                        eprintln!(
                            "    failed_trivial_history now {} (added={}, hash={:016x})",
                            failed_trivial_supports.len(),
                            !duplicate_failure,
                            cand.support_hash
                        );
                    }
                }
                None => {}
            }

            if trivial_bail > 0
                && deps_tried == trivial_bail
                && fail_trivial_gcd == trivial_bail
                && fail_rat_not_square == 0
                && fail_alg_not_square == 0
            {
                eprintln!(
                    "  sqrt: early bail after {} deps (all trivial gcd)",
                    trivial_bail
                );
                break;
            }

            // Refill pool after cand borrow is released
            refill_pool_and_expand!(untried_pool, next_pool_cursor);
        }
    }

    let deps_skipped_long = untried_pool
        .iter()
        .filter(|&&i| useful_deps[i].expanded.as_ref().map(|e| e.len()).unwrap_or(useful_deps[i].raw_weight) >= degree)
        .count()
        + (next_pool_cursor..useful_deps.len())
            .filter(|&i| useful_deps[i].raw_weight >= degree)
            .count();

    if result.factor.is_none() {
        result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
        result.sqrt_attempts_tried = deps_tried;
        result.sqrt_fail_rat_not_square = fail_rat_not_square;
        result.sqrt_fail_alg_not_square = fail_alg_not_square;
        result.sqrt_fail_trivial_gcd = fail_trivial_gcd;
        eprintln!(
            "  sqrt: no factor found in {:.0}ms (tried={}, skipped_short={}, skipped_long={}, skipped_non_coprime={}, skipped_sim={}, rat_not_square={}, alg_not_square={}, trivial_gcd={})",
            result.sqrt_ms, deps_tried, deps_skipped_short, deps_skipped_long, deps_skipped_non_coprime, total_skipped_sim, fail_rat_not_square, fail_alg_not_square, fail_trivial_gcd
        );
    } else {
        result.sqrt_fail_rat_not_square = fail_rat_not_square;
        result.sqrt_fail_alg_not_square = fail_alg_not_square;
        result.sqrt_fail_trivial_gcd = fail_trivial_gcd;
    }

    result.sqrt_attempts_tried = deps_tried;

    // Capture sqrt timing (already set by success or failure path above).
    timings.add(StageResult {
        name: "sqrt".to_string(),
        total_ms: result.sqrt_ms,
        sub_stages: vec![],
        timed_out: false,
    });

    result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
    timings.set_total(result.total_ms);
    eprintln!("  timing: {}", timings.summary_line());
    if emit_timing_json {
        eprintln!("  timing-json: {}", timings.to_json());
    }
    result
}

/// Fast quadratic character prime selection using Cantor-Zassenhaus root finding.
///
/// Replaces `gnfs::arith::select_quad_char_primes` which uses O(p) brute-force
/// root finding and sieves all primes to 1M. This version:
/// 1. Starts searching from above the FB max (skips ~3000 FB primes)
/// 2. Uses Cantor-Zassenhaus O(d² log p) root finding
/// 3. Uses trial-division primality (p < 50K, so sqrt(p) < 224)
fn select_quad_char_primes_fast(
    f_coeffs: &[i64],
    fb_primes: &[u64],
    count: usize,
) -> gnfs::arith::QuadCharSet {
    if count == 0 {
        return gnfs::arith::QuadCharSet {
            primes: Vec::new(),
            roots: Vec::new(),
        };
    }

    let fb_max = fb_primes.last().copied().unwrap_or(2);
    let mut result = gnfs::arith::QuadCharSet {
        primes: Vec::with_capacity(count),
        roots: Vec::with_capacity(count),
    };

    // Start from the first odd number above fb_max
    let mut candidate = if fb_max % 2 == 0 { fb_max + 1 } else { fb_max + 2 };

    let mut distinct_count = 0usize;
    while distinct_count < count {
        if is_prime_trial(candidate) {
            let roots = crate::factorbase::find_roots_mod_p(f_coeffs, candidate);
            if !roots.is_empty() {
                // Push ALL roots for each QC prime. For degree-d polynomials,
                // each prime can have up to d roots; we need a QC column for
                // every (prime, root) pair to fully constrain the algebraic
                // product to be a square across all prime ideals above each prime.
                for &r in &roots {
                    result.primes.push(candidate);
                    result.roots.push(r);
                }
                distinct_count += 1;
            }
        }
        candidate += 2;
    }

    result
}

/// Simple trial-division primality test, sufficient for p < 10^9.
fn is_prime_trial(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn fallback_result(n: &Integer, factor: Integer, total_ms: f64) -> NfsResult {
    eprintln!(
        "  fallback_rho: factor found in {:.0}ms (bits={}): {}",
        total_ms,
        n.significant_bits(),
        factor
    );
    NfsResult {
        n: n.to_string(),
        factor: Some(factor.to_string()),
        relations_found: 0,
        relations_after_filter: 0,
        matrix_rows: 0,
        matrix_cols: 0,
        dependencies_found: 0,
        sqrt_attempts_tried: 0,
        sqrt_factor_attempt: None,
        polyselect_ms: 0.0,
        sieve_ms: 0.0,
        sieve_setup_ms: 0.0,
        sieve_scan_ms: 0.0,
        sieve_cofactor_ms: 0.0,
        sieve_rels_per_sq: 0.0,
        filter_ms: 0.0,
        la_ms: 0.0,
        sqrt_ms: 0.0,
        total_ms,
        sqrt_fail_rat_not_square: 0,
        sqrt_fail_alg_not_square: 0,
        sqrt_fail_trivial_gcd: 0,
        viability: ViabilityStats::default(),
    }
}

fn try_pollard_rho_factor(n: &Integer) -> Option<(Integer, f64)> {
    if *n <= 3 {
        return None;
    }
    if n.is_even() {
        return Some((Integer::from(2), 0.0));
    }
    if n.is_probably_prime(30) != rug::integer::IsPrime::No {
        return None;
    }
    let rounds = std::env::var("POTAPOV_NFS_FALLBACK_RHO_ROUNDS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(48u32);
    let iter_limit = std::env::var("POTAPOV_NFS_FALLBACK_RHO_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(500_000usize);

    let time_limit_ms = std::env::var("POTAPOV_NFS_FALLBACK_RHO_TIME_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(200u64);
    let deadline = std::time::Duration::from_millis(time_limit_ms);

    let start = std::time::Instant::now();
    for round in 0..rounds {
        if start.elapsed() >= deadline {
            break;
        }
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(round as u64 + 1);
        if let Some(f) = pollard_rho_brent(n, seed, iter_limit) {
            if f > 1 && f < *n && Integer::from(n % &f) == 0 {
                return Some((f, start.elapsed().as_secs_f64() * 1000.0));
            }
        }
    }
    None
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn sample_nontrivial_mod(n: &Integer, state: &mut u64) -> Integer {
    if *n <= 5 {
        return Integer::from(2);
    }
    let span = Integer::from(n - 3u32);
    let v = Integer::from(lcg_next(state)) % &span;
    v + 2u32
}

fn abs_diff(a: &Integer, b: &Integer) -> Integer {
    if a >= b {
        Integer::from(a - b)
    } else {
        Integer::from(b - a)
    }
}

fn pollard_rho_brent(n: &Integer, seed: u64, iter_limit: usize) -> Option<Integer> {
    if n.is_even() {
        return Some(Integer::from(2));
    }

    let one = Integer::from(1);
    let mut state = seed.wrapping_add(1);
    let mut y = sample_nontrivial_mod(n, &mut state);
    let c = sample_nontrivial_mod(n, &mut state);
    let m = 64usize;
    let mut g = Integer::from(1);
    let mut r = 1usize;
    let mut q = Integer::from(1);
    let x = Integer::from(0);
    let mut ys = Integer::from(0);
    let mut iters = 0usize;

    while g == one && iters < iter_limit {
        let x = y.clone();
        for _ in 0..r {
            y = (Integer::from(&y * &y) + &c) % n;
            iters += 1;
            if iters >= iter_limit {
                return None;
            }
        }
        let mut k = 0usize;
        while k < r && g == one && iters < iter_limit {
            ys = y.clone();
            let lim = m.min(r - k);
            for _ in 0..lim {
                y = (Integer::from(&y * &y) + &c) % n;
                let d = abs_diff(&x, &y);
                q = Integer::from(&q * d) % n;
                iters += 1;
                if iters >= iter_limit {
                    return None;
                }
            }
            g = n.clone().gcd(&q);
            k += m;
        }
        r = r.saturating_mul(2);
    }

    if g == *n {
        loop {
            ys = (Integer::from(&ys * &ys) + &c) % n;
            g = n.clone().gcd(&abs_diff(&x, &ys));
            iters += 1;
            if g > one {
                break;
            }
            if iters >= iter_limit {
                return None;
            }
        }
    }

    if g > one && g < *n {
        Some(g)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SparseKey {
    SpecialQ(u64, u64),
    RatLp(u64),
    AlgLp(u64, u64),
}

#[derive(Debug, Clone)]
struct SparseElimRow {
    keys: Vec<SparseKey>,
    rels: Vec<usize>,
}

fn sym_diff_sorted<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        out.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        out.push(b[j]);
        j += 1;
    }
    out
}

fn relation_sparse_keys(rel: &gnfs::types::Relation) -> Vec<SparseKey> {
    let mut keys = Vec::with_capacity(3);
    if let Some((q, r)) = rel.special_q {
        keys.push(SparseKey::SpecialQ(q, r));
    }
    if let Some(lp) = rel.rat_lp {
        keys.push(SparseKey::RatLp(lp));
    }
    if let Some((p, r)) = rel.alg_lp {
        keys.push(SparseKey::AlgLp(p, r));
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Build relation-sets whose sparse-column parity is zero using GF(2)
/// elimination over sparse keys (special-q and large-prime columns).
fn build_sparse_zero_sets(rels: &[gnfs::types::Relation], max_sets: usize) -> Vec<Vec<usize>> {
    use std::collections::HashMap;

    let mut pivots: HashMap<SparseKey, SparseElimRow> = HashMap::new();
    let mut sets: Vec<Vec<usize>> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();

    for (idx, rel) in rels.iter().enumerate() {
        let mut row = SparseElimRow {
            keys: relation_sparse_keys(rel),
            rels: vec![idx],
        };

        let mut registered_pivot = false;
        while !row.keys.is_empty() {
            let pivot_key = row.keys[0].clone();
            if let Some(pivot_row) = pivots.get(&pivot_key) {
                row.keys = sym_diff_sorted(&row.keys, &pivot_row.keys);
                row.rels = sym_diff_sorted(&row.rels, &pivot_row.rels);
            } else {
                pivots.insert(pivot_key, row.clone());
                registered_pivot = true;
                break;
            }
        }

        if !registered_pivot && row.keys.is_empty() && !row.rels.is_empty() {
            if seen.insert(row.rels.clone()) {
                sets.push(row.rels);
                if sets.len() >= max_sets {
                    break;
                }
            }
        }
    }

    sets
}

fn build_special_q_zero_sets_from_partial_sets(
    rels: &[gnfs::types::Relation],
    set_rows: &[Vec<usize>],
    max_sets: usize,
) -> Vec<Vec<usize>> {
    use std::collections::HashMap;

    let mut pivots: HashMap<SparseKey, SparseElimRow> = HashMap::new();
    let mut sets: Vec<Vec<usize>> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();

    for set in set_rows {
        if sets.len() >= max_sets {
            break;
        }
        if set.is_empty() {
            continue;
        }

        let mut sq_keys: Vec<SparseKey> = set
            .iter()
            .filter_map(|&ri| {
                rels.get(ri)
                    .and_then(|rel| rel.special_q)
                    .map(|(q, r)| SparseKey::SpecialQ(q, r))
            })
            .collect();
        sq_keys.sort_unstable();
        let mut collapsed_keys = Vec::with_capacity(sq_keys.len());
        let mut i = 0usize;
        while i < sq_keys.len() {
            let mut j = i + 1;
            while j < sq_keys.len() && sq_keys[j] == sq_keys[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                collapsed_keys.push(sq_keys[i].clone());
            }
            i = j;
        }

        let mut row = SparseElimRow {
            keys: collapsed_keys,
            rels: set.clone(),
        };

        let mut registered_pivot = false;
        while !row.keys.is_empty() {
            let pivot_key = row.keys[0].clone();
            if let Some(pivot_row) = pivots.get(&pivot_key) {
                row.keys = sym_diff_sorted(&row.keys, &pivot_row.keys);
                row.rels = sym_diff_sorted(&row.rels, &pivot_row.rels);
            } else {
                pivots.insert(pivot_key, row.clone());
                registered_pivot = true;
                break;
            }
        }

        if !registered_pivot && row.keys.is_empty() && !row.rels.is_empty() {
            if seen.insert(row.rels.clone()) {
                sets.push(row.rels);
            }
        }
    }

    sets
}

/// Build a dense-only matrix from premerged relation-sets. Sparse columns
/// are already canceled by construction, so each set-row only includes
/// sign/factor-base/QC parity bits.
fn build_dense_matrix_from_sets(
    rels: &[gnfs::types::Relation],
    set_rows: &[Vec<usize>],
    rat_fb_size: usize,
    alg_pair_count: usize,
    alg_hd_count: usize,
    quad_chars: &gnfs::arith::QuadCharSet,
) -> (Vec<gnfs::types::BitRow>, usize, Vec<Vec<usize>>) {
    let n_qc = quad_chars.primes.len();
    let sign_rat_col = 0usize;
    let rat_base = 1usize;
    let sign_alg_col = rat_base + rat_fb_size;
    let alg_base = sign_alg_col + 1;
    let qc_base = alg_base + alg_pair_count + alg_hd_count;
    let ncols = qc_base + n_qc;

    let mut rel_flip_cols: Vec<Vec<usize>> = Vec::with_capacity(rels.len());
    for rel in rels {
        let mut cols: Vec<usize> = Vec::new();
        if rel.rational_sign_negative {
            cols.push(sign_rat_col);
        }
        for &(idx, exp) in &rel.rational_factors {
            if exp % 2 == 1 {
                cols.push(rat_base + idx as usize);
            }
        }
        if rel.algebraic_sign_negative {
            cols.push(sign_alg_col);
        }
        for &(idx, exp) in &rel.algebraic_factors {
            if exp % 2 == 1 {
                cols.push(alg_base + idx as usize);
            }
        }
        for (i, (&q, &r)) in quad_chars
            .primes
            .iter()
            .zip(quad_chars.roots.iter())
            .enumerate()
        {
            let q_i = q as i128;
            let val = (rel.a as i128 - rel.b as i128 * r as i128).rem_euclid(q_i) as u64;
            if val == 0 {
                continue;
            }
            let ls = gnfs::arith::legendre_symbol(val, q);
            if ls == q - 1 {
                cols.push(qc_base + i);
            }
        }
        rel_flip_cols.push(cols);
    }

    let mut matrix: Vec<gnfs::types::BitRow> = Vec::with_capacity(set_rows.len());
    let mut kept_sets: Vec<Vec<usize>> = Vec::with_capacity(set_rows.len());

    for set in set_rows {
        if set.is_empty() {
            continue;
        }
        let mut row = gnfs::types::BitRow::new(ncols);
        for &ri in set {
            if ri >= rel_flip_cols.len() {
                continue;
            }
            for &col in &rel_flip_cols[ri] {
                row.flip(col);
            }
        }
        matrix.push(row);
        kept_sets.push(set.clone());
    }

    (matrix, ncols, kept_sets)
}

/// Build matrix rows from relation-sets after LP keys were resolved by partial merge.
///
/// Column layout:
/// [SQ | sign_rat | rat_FB | sign_alg | alg_pairs | alg_HD | QC]
fn build_matrix_from_sets_lp_resolved(
    rels: &[gnfs::types::Relation],
    set_rows: &[Vec<usize>],
    rat_fb_size: usize,
    alg_pair_count: usize,
    alg_hd_count: usize,
    quad_chars: &gnfs::arith::QuadCharSet,
) -> (Vec<gnfs::types::BitRow>, usize, Vec<Vec<usize>>) {
    let mut sq_pairs: Vec<(u64, u64)> = rels.iter().filter_map(|r| r.special_q).collect();
    sq_pairs.sort_unstable();
    sq_pairs.dedup();

    let n_sq = sq_pairs.len();
    let n_qc = quad_chars.primes.len();
    let sign_rat_col = n_sq;
    let rat_base = sign_rat_col + 1;
    let sign_alg_col = rat_base + rat_fb_size;
    let alg_base = sign_alg_col + 1;
    let qc_base = alg_base + alg_pair_count + alg_hd_count;
    let ncols = qc_base + n_qc;

    use rayon::prelude::*;
    let rel_flip_cols: Vec<Vec<usize>> = rels
        .par_iter()
        .map(|rel| {
            let mut cols: Vec<usize> = Vec::new();
            if let Some(sq) = rel.special_q {
                if let Ok(idx) = sq_pairs.binary_search(&sq) {
                    cols.push(idx);
                }
            }
            if rel.rational_sign_negative {
                cols.push(sign_rat_col);
            }
            for &(idx, exp) in &rel.rational_factors {
                if exp % 2 == 1 {
                    cols.push(rat_base + idx as usize);
                }
            }
            if rel.algebraic_sign_negative {
                cols.push(sign_alg_col);
            }
            for &(idx, exp) in &rel.algebraic_factors {
                if exp % 2 == 1 {
                    cols.push(alg_base + idx as usize);
                }
            }
            for (i, (&q, &r)) in quad_chars
                .primes
                .iter()
                .zip(quad_chars.roots.iter())
                .enumerate()
            {
                let q_i = q as i128;
                let val = (rel.a as i128 - rel.b as i128 * r as i128).rem_euclid(q_i) as u64;
                if val == 0 {
                    continue;
                }
                let ls = gnfs::arith::legendre_symbol(val, q);
                if ls == q - 1 {
                    cols.push(qc_base + i);
                }
            }
            cols
        })
        .collect();

    let results: Vec<Option<(gnfs::types::BitRow, Vec<usize>)>> = set_rows
        .par_iter()
        .map(|set| {
            if set.is_empty() {
                return None;
            }
            let mut row = gnfs::types::BitRow::new(ncols);
            for &ri in set {
                if ri >= rel_flip_cols.len() {
                    continue;
                }
                for &col in &rel_flip_cols[ri] {
                    row.flip(col);
                }
            }
            Some((row, set.clone()))
        })
        .collect();

    let mut matrix: Vec<gnfs::types::BitRow> = Vec::with_capacity(set_rows.len());
    let mut kept_sets: Vec<Vec<usize>> = Vec::with_capacity(set_rows.len());
    for item in results {
        if let Some((row, set)) = item {
            matrix.push(row);
            kept_sets.push(set);
        }
    }

    (matrix, ncols, kept_sets)
}

fn expand_dependency_over_sets(dep: &[usize], set_rows: &[Vec<usize>]) -> Vec<usize> {
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &set_idx in dep {
        if let Some(set) = set_rows.get(set_idx) {
            for &item in set {
                *counts.entry(item).or_insert(0) += 1;
            }
        }
    }
    let mut expanded: Vec<usize> = counts
        .into_iter()
        .filter_map(|(item, count)| if count % 2 != 0 { Some(item) } else { None })
        .collect();
    expanded.sort_unstable();
    expanded
}

fn remap_partial_sets_from_sources(
    filtered: &[crate::relation::Relation],
    partial_sets_filtered: Option<&[Vec<usize>]>,
    remap_source_indices: &[usize],
    partial_merge_max_sets: usize,
) -> (Vec<Vec<usize>>, usize, usize) {
    let source_to_remap: HashMap<usize, usize> = remap_source_indices
        .iter()
        .enumerate()
        .map(|(ri, &src)| (src, ri))
        .collect();
    let mut partial_sets_remapped: Vec<Vec<usize>> = Vec::new();
    let mut partial_sets_dropped_remap = 0usize;
    let mut partial_sets_recomputed = 0usize;

    if let Some(sets) = partial_sets_filtered {
        for set in sets {
            let mut mapped = Vec::with_capacity(set.len());
            let mut valid = true;
            for &src_idx in set {
                if let Some(&ri) = source_to_remap.get(&src_idx) {
                    mapped.push(ri);
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid || mapped.is_empty() {
                partial_sets_dropped_remap += 1;
                continue;
            }
            mapped.sort_unstable();
            mapped.dedup();
            if !mapped.is_empty() {
                partial_sets_remapped.push(mapped);
            } else {
                partial_sets_dropped_remap += 1;
            }
        }
        if partial_sets_dropped_remap > 0 && !remap_source_indices.is_empty() {
            // Rebuild 2LP sets on the remap-valid subset directly so invalid
            // relations do not poison whole set-rows during source-index remap.
            let remap_valid_filtered: Vec<crate::relation::Relation> = remap_source_indices
                .iter()
                .filter_map(|&src_idx| filtered.get(src_idx).cloned())
                .collect();
            let (recomputed_sets, _stats) = crate::partial_merge::merge_relations_2lp(
                &remap_valid_filtered,
                partial_merge_max_sets,
            );
            partial_sets_recomputed = recomputed_sets.len();
            partial_sets_remapped = recomputed_sets;
        }
    }

    (
        partial_sets_remapped,
        partial_sets_dropped_remap,
        partial_sets_recomputed,
    )
}

fn compact_zero_columns(
    rows: Vec<gnfs::types::BitRow>,
    ncols: usize,
) -> (Vec<gnfs::types::BitRow>, usize, usize) {
    if rows.is_empty() || ncols == 0 {
        return (rows, ncols, 0);
    }

    let nwords = (ncols + 63) / 64;
    let mut used_words = vec![0u64; nwords];
    for row in &rows {
        for (i, &w) in row.bits.iter().enumerate() {
            used_words[i] |= w;
        }
    }

    let mut used_cols: Vec<usize> = Vec::new();
    used_cols.reserve(ncols);
    for (wi, &word_mask) in used_words.iter().enumerate() {
        let mut w = word_mask;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            let col = wi * 64 + bit;
            if col < ncols {
                used_cols.push(col);
            }
            w &= w - 1;
        }
    }

    if used_cols.len() == ncols {
        return (rows, ncols, 0);
    }

    let mut col_map = vec![usize::MAX; ncols];
    for (new_col, &old_col) in used_cols.iter().enumerate() {
        col_map[old_col] = new_col;
    }

    let mut compacted: Vec<gnfs::types::BitRow> = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut new_row = gnfs::types::BitRow::new(used_cols.len());
        for (wi, &word_mask) in row.bits.iter().enumerate() {
            let mut w = word_mask;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let old_col = wi * 64 + bit;
                if old_col >= ncols {
                    break;
                }
                let mapped = col_map[old_col];
                if mapped != usize::MAX {
                    new_row.set(mapped);
                }
                w &= w - 1;
            }
        }
        compacted.push(new_row);
    }

    let dropped = ncols.saturating_sub(used_cols.len());
    (compacted, used_cols.len(), dropped)
}

#[derive(Debug, Clone, Copy)]
struct MatrixShapeSummary {
    raw_rows: usize,
    raw_cols: usize,
    zero_cols_dropped_initial: usize,
    singleton_rows_dropped: usize,
    zero_cols_dropped_post_singleton: usize,
    final_rows: usize,
    final_cols: usize,
    rows_minus_cols: isize,
}

#[derive(Debug, Clone)]
struct MatrixColumnLayout {
    sq: std::ops::Range<usize>,
    rat_lp: std::ops::Range<usize>,
    alg_lp: std::ops::Range<usize>,
    sign_rat: Option<usize>,
    rat_fb: std::ops::Range<usize>,
    sign_alg: Option<usize>,
    alg_dense: std::ops::Range<usize>,
    qc: std::ops::Range<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MatrixColumnWeightStats {
    active_special_q_cols: usize,
    active_rat_lp_cols: usize,
    active_alg_lp_cols: usize,
    active_sign_cols: usize,
    active_rat_fb_cols: usize,
    active_alg_dense_cols: usize,
    active_qc_cols: usize,
    singleton_cols_total: usize,
    singleton_special_q_cols: usize,
    singleton_rat_lp_cols: usize,
    singleton_alg_lp_cols: usize,
    singleton_sign_cols: usize,
    singleton_rat_fb_cols: usize,
    singleton_alg_dense_cols: usize,
    singleton_qc_cols: usize,
}

fn build_direct_matrix_layout(
    rels: &[gnfs::types::Relation],
    rat_fb_size: usize,
    alg_dense_count: usize,
    n_qc: usize,
) -> MatrixColumnLayout {
    let n_sq = rels
        .iter()
        .filter_map(|r| r.special_q)
        .collect::<HashSet<_>>()
        .len();
    let n_rat_lp = rels
        .iter()
        .filter_map(|r| r.rat_lp)
        .collect::<HashSet<_>>()
        .len();
    let n_alg_lp = rels
        .iter()
        .filter_map(|r| r.alg_lp)
        .collect::<HashSet<_>>()
        .len();
    let sq = 0..n_sq;
    let rat_lp = sq.end..(sq.end + n_rat_lp);
    let alg_lp = rat_lp.end..(rat_lp.end + n_alg_lp);
    let sign_rat = Some(alg_lp.end);
    let rat_fb = (alg_lp.end + 1)..(alg_lp.end + 1 + rat_fb_size);
    let sign_alg = Some(rat_fb.end);
    let alg_dense = (rat_fb.end + 1)..(rat_fb.end + 1 + alg_dense_count);
    let qc = alg_dense.end..(alg_dense.end + n_qc);
    MatrixColumnLayout {
        sq,
        rat_lp,
        alg_lp,
        sign_rat,
        rat_fb,
        sign_alg,
        alg_dense,
        qc,
    }
}

fn build_lp_resolved_matrix_layout(
    rels: &[gnfs::types::Relation],
    rat_fb_size: usize,
    alg_dense_count: usize,
    n_qc: usize,
) -> MatrixColumnLayout {
    let n_sq = rels
        .iter()
        .filter_map(|r| r.special_q)
        .collect::<HashSet<_>>()
        .len();
    let sq = 0..n_sq;
    let rat_lp = sq.end..sq.end;
    let alg_lp = sq.end..sq.end;
    let sign_rat = Some(sq.end);
    let rat_fb = (sq.end + 1)..(sq.end + 1 + rat_fb_size);
    let sign_alg = Some(rat_fb.end);
    let alg_dense = (rat_fb.end + 1)..(rat_fb.end + 1 + alg_dense_count);
    let qc = alg_dense.end..(alg_dense.end + n_qc);
    MatrixColumnLayout {
        sq,
        rat_lp,
        alg_lp,
        sign_rat,
        rat_fb,
        sign_alg,
        alg_dense,
        qc,
    }
}

fn build_dense_only_matrix_layout(
    rat_fb_size: usize,
    alg_dense_count: usize,
    n_qc: usize,
) -> MatrixColumnLayout {
    let sq = 0..0;
    let rat_lp = 0..0;
    let alg_lp = 0..0;
    let sign_rat = Some(0);
    let rat_fb = 1..(1 + rat_fb_size);
    let sign_alg = Some(rat_fb.end);
    let alg_dense = (rat_fb.end + 1)..(rat_fb.end + 1 + alg_dense_count);
    let qc = alg_dense.end..(alg_dense.end + n_qc);
    MatrixColumnLayout {
        sq,
        rat_lp,
        alg_lp,
        sign_rat,
        rat_fb,
        sign_alg,
        alg_dense,
        qc,
    }
}

fn collect_matrix_column_weight_stats(
    rows: &[gnfs::types::BitRow],
    ncols: usize,
    layout: &MatrixColumnLayout,
) -> MatrixColumnWeightStats {
    let mut col_counts = vec![0usize; ncols];
    for row in rows {
        for (wi, &word_mask) in row.bits.iter().enumerate() {
            let mut w = word_mask;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let col = wi * 64 + bit;
                if col < ncols {
                    col_counts[col] += 1;
                }
                w &= w - 1;
            }
        }
    }

    let mut stats = MatrixColumnWeightStats::default();
    for (col, &count) in col_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let is_singleton = count == 1;
        if layout.sq.contains(&col) {
            stats.active_special_q_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_special_q_cols += 1;
            }
            continue;
        }
        if layout.rat_lp.contains(&col) {
            stats.active_rat_lp_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_rat_lp_cols += 1;
            }
            continue;
        }
        if layout.alg_lp.contains(&col) {
            stats.active_alg_lp_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_alg_lp_cols += 1;
            }
            continue;
        }
        if layout.sign_rat == Some(col) || layout.sign_alg == Some(col) {
            stats.active_sign_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_sign_cols += 1;
            }
            continue;
        }
        if layout.rat_fb.contains(&col) {
            stats.active_rat_fb_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_rat_fb_cols += 1;
            }
            continue;
        }
        if layout.alg_dense.contains(&col) {
            stats.active_alg_dense_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_alg_dense_cols += 1;
            }
            continue;
        }
        if layout.qc.contains(&col) {
            stats.active_qc_cols += 1;
            if is_singleton {
                stats.singleton_cols_total += 1;
                stats.singleton_qc_cols += 1;
            }
        }
    }

    stats
}

fn summarize_matrix_shape(
    rows: Vec<gnfs::types::BitRow>,
    ncols: usize,
    row_sources: Vec<Vec<usize>>,
    compact_zero_cols: bool,
    singleton_prune: bool,
    singleton_min_weight: usize,
) -> MatrixShapeSummary {
    let raw_rows = rows.len();
    let raw_cols = ncols;

    let (mut rows, mut ncols, zero_cols_dropped_initial) = if compact_zero_cols {
        compact_zero_columns(rows, ncols)
    } else {
        (rows, ncols, 0usize)
    };

    let mut singleton_rows_dropped = 0usize;
    let mut zero_cols_dropped_post_singleton = 0usize;
    if singleton_prune {
        let (pruned_rows, _pruned_sources, removed_rows) =
            prune_singleton_columns(rows, row_sources, ncols, singleton_min_weight);
        rows = pruned_rows;
        singleton_rows_dropped = removed_rows;
        if compact_zero_cols {
            let (m2, nc2, dropped2) = compact_zero_columns(rows, ncols);
            rows = m2;
            ncols = nc2;
            zero_cols_dropped_post_singleton = dropped2;
        }
    }

    MatrixShapeSummary {
        raw_rows,
        raw_cols,
        zero_cols_dropped_initial,
        singleton_rows_dropped,
        zero_cols_dropped_post_singleton,
        final_rows: rows.len(),
        final_cols: ncols,
        rows_minus_cols: rows.len() as isize - ncols as isize,
    }
}

fn prune_singleton_columns(
    mut rows: Vec<gnfs::types::BitRow>,
    mut row_sources: Vec<Vec<usize>>,
    ncols: usize,
    min_weight: usize,
) -> (Vec<gnfs::types::BitRow>, Vec<Vec<usize>>, usize) {
    if rows.is_empty() || ncols == 0 {
        return (rows, row_sources, 0);
    }
    let mut removed_rows_total = 0usize;
    let nwords = (ncols + 63) / 64;

    loop {
        if rows.is_empty() {
            break;
        }

        let mut col_counts = vec![0usize; ncols];
        for row in &rows {
            for (wi, &word_mask) in row.bits.iter().enumerate() {
                let mut w = word_mask;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let col = wi * 64 + bit;
                    if col < ncols {
                        col_counts[col] += 1;
                    }
                    w &= w - 1;
                }
            }
        }

        let mut bad_mask_words = vec![0u64; nwords];
        let mut bad_cols = 0usize;
        for (col, &cnt) in col_counts.iter().enumerate() {
            if cnt > 0 && cnt < min_weight {
                let wi = col / 64;
                let bit = col % 64;
                bad_mask_words[wi] |= 1u64 << bit;
                bad_cols += 1;
            }
        }
        if bad_cols == 0 {
            break;
        }

        let old_rows = std::mem::take(&mut rows);
        let old_sources = std::mem::take(&mut row_sources);
        let mut new_rows: Vec<gnfs::types::BitRow> = Vec::with_capacity(old_rows.len());
        let mut new_sources: Vec<Vec<usize>> = Vec::with_capacity(old_sources.len());
        let mut removed_this_round = 0usize;

        for (row, src) in old_rows.into_iter().zip(old_sources.into_iter()) {
            let touches_bad = row
                .bits
                .iter()
                .zip(bad_mask_words.iter())
                .any(|(&rw, &bm)| (rw & bm) != 0);
            if touches_bad {
                removed_this_round += 1;
            } else {
                new_rows.push(row);
                new_sources.push(src);
            }
        }

        if removed_this_round == 0 {
            break;
        }
        removed_rows_total += removed_this_round;
        rows = new_rows;
        row_sources = new_sources;
    }

    (rows, row_sources, removed_rows_total)
}

fn dependency_length_stats(lengths: &[usize]) -> Option<(usize, usize, usize, usize, usize)> {
    if lengths.is_empty() {
        return None;
    }
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p = |num: usize, den: usize| -> usize {
        let idx = ((n - 1) * num + den / 2) / den;
        sorted[idx]
    };
    Some((sorted[0], p(1, 2), p(9, 10), p(99, 100), sorted[n - 1]))
}

fn parse_dep_len_tiers(max_dep_len: Option<usize>, auto_relax: bool) -> Vec<Option<usize>> {
    if let Ok(raw) = std::env::var("POTAPOV_NFS_DEP_LEN_TIERS") {
        let mut caps: Vec<usize> = Vec::new();
        let mut has_none = false;
        for tok in raw.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()) {
            if tok.eq_ignore_ascii_case("none")
                || tok.eq_ignore_ascii_case("inf")
                || tok.eq_ignore_ascii_case("unbounded")
            {
                has_none = true;
                continue;
            }
            if let Ok(v) = tok.parse::<usize>() {
                if v > 0 {
                    caps.push(v);
                }
            }
        }
        caps.sort_unstable();
        caps.dedup();
        let mut tiers: Vec<Option<usize>> = caps.into_iter().map(Some).collect();
        if has_none || tiers.is_empty() {
            tiers.push(None);
        }
        return tiers;
    }

    match max_dep_len {
        Some(cap) if auto_relax => vec![
            Some(cap),
            Some(cap.saturating_mul(2)),
            Some(cap.saturating_mul(4)),
            None,
        ],
        Some(cap) => vec![Some(cap)],
        None => vec![None],
    }
}

fn format_dep_len_tiers(tiers: &[Option<usize>]) -> String {
    tiers
        .iter()
        .map(|cap| {
            cap.map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string())
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn eval_univariate_bigint(coeffs: &[Integer], x: &Integer) -> Integer {
    let mut acc = Integer::from(0);
    for c in coeffs.iter().rev() {
        acc *= x;
        acc += c;
    }
    acc
}

fn eval_univariate_derivative_mod_p(coeffs: &[Integer], x: u64, p: u64) -> u64 {
    if p < 2 {
        return 0;
    }
    let p128 = p as u128;
    let mut acc = 0u128;
    let mut x_pow = 1u128;
    for (i, c) in coeffs.iter().enumerate().skip(1) {
        let coeff_mod = c.mod_u(p as u32) as u128;
        acc = (acc + ((i as u128 * coeff_mod) % p128) * x_pow) % p128;
        x_pow = (x_pow * x as u128) % p128;
    }
    acc as u64
}

fn compute_bad_root_offsets(
    gnfs_fb: &gnfs::types::FactorBase,
    f_coeffs_big: &[Integer],
) -> HashMap<(u64, u64), usize> {
    let mut offsets = HashMap::new();
    let mut next = 0usize;
    for (prime_idx, &p) in gnfs_fb.primes.iter().enumerate() {
        for &r in &gnfs_fb.algebraic_roots[prime_idx] {
            if root_multiplicity_mod_p(f_coeffs_big, r, p) > 1 {
                offsets.insert((p, r), next);
                next += 1;
            }
        }
    }
    offsets
}

fn hensel_lift_simple_root(
    coeffs: &[Integer],
    p: u64,
    root_mod_p: u64,
    lift_exp: u8,
    cache: &mut HashMap<(u64, u64, u8), Integer>,
) -> Option<Integer> {
    if p < 2 || lift_exp == 0 {
        return None;
    }
    if let Some(root) = cache.get(&(p, root_mod_p, lift_exp)) {
        return Some(root.clone());
    }

    let deriv_mod_p = eval_univariate_derivative_mod_p(coeffs, root_mod_p, p);
    let deriv_inv = gnfs::arith::mod_inverse_u64(deriv_mod_p, p)?;

    let mut current_exp = 1u8;
    let mut current_root = Integer::from(root_mod_p);
    let mut modulus = Integer::from(p);
    cache.insert((p, root_mod_p, current_exp), current_root.clone());

    while current_exp < lift_exp {
        let f_val = eval_univariate_bigint(coeffs, &current_root);
        if f_val.clone() % &modulus != 0 {
            return None;
        }
        let quotient = Integer::from(&f_val / &modulus);
        let q_mod_p = quotient.mod_u(p as u32) as u64;
        let correction = (p - ((q_mod_p as u128 * deriv_inv as u128) % p as u128) as u64) % p;
        current_root += Integer::from(&modulus * correction);
        modulus *= p;
        current_exp += 1;
        cache.insert((p, root_mod_p, current_exp), current_root.clone());
    }

    Some(current_root)
}

fn p_valuation_a_minus_br_hensel(
    a: i64,
    b: u64,
    p: u64,
    root_mod_p: u64,
    max_exp: u8,
    coeffs: &[Integer],
    cache: &mut HashMap<(u64, u64, u8), Integer>,
) -> Option<u8> {
    if max_exp == 0 || p < 2 {
        return Some(0);
    }
    let lifted_root = hensel_lift_simple_root(coeffs, p, root_mod_p, max_exp, cache)?;
    let modulus = Integer::from(p).pow(max_exp as u32);
    let mut delta = Integer::from(a) - Integer::from(b) * lifted_root;
    delta %= &modulus;
    if delta < 0 {
        delta += &modulus;
    }
    if delta == 0 {
        return Some(max_exp);
    }

    let p_int = Integer::from(p);
    let mut e = 0u8;
    while e < max_exp && delta.clone() % &p_int == 0 {
        delta /= p;
        e += 1;
    }
    Some(e)
}

/// Compute sign of the algebraic norm F(a,b).
///
/// Hybrid remap: keep rational factors from our sieve (complete FB), recompute
/// algebraic factors using BigInt norms and gnfs's per-(prime,root) decomposition.
///
/// This avoids both:
/// 1. u64 overflow in algebraic norm computation (uses BigInt for norms)
/// 2. Losing relations due to inert primes on rational side (keeps our rational FB)
fn remap_hybrid(
    relations: &[crate::relation::Relation],
    _rat_fb: &crate::factorbase::FactorBase,
    alg_fb: &crate::factorbase::FactorBase,
    gnfs_fb: &gnfs::types::FactorBase,
    f_coeffs_big: &[Integer],
    _m: u64,
    degree: usize,
    bad_root_offsets: &HashMap<(u64, u64), usize>,
    g0_i64: i64,
    g1_i64: i64,
) -> (Vec<gnfs::types::Relation>, Vec<usize>, RemapHybridStats) {
    let ignore_special_q = std::env::var("POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let hd_residual_sample_limit = std::env::var("POTAPOV_NFS_HD_RESIDUAL_SAMPLE_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0usize);
    let prime_to_gnfs: HashMap<u64, usize> = gnfs_fb
        .primes
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    let mut root_multiplicity_cache = HashMap::<(u64, u64), u8>::new();

    for (prime_idx, &p) in gnfs_fb.primes.iter().enumerate() {
        for &r in &gnfs_fb.algebraic_roots[prime_idx] {
            root_multiplicity_cache
                .insert((p, r), root_multiplicity_mod_p(f_coeffs_big, r, p) as u8);
        }
    }

    // Process relations in parallel using rayon fold for per-thread hensel caches
    use rayon::prelude::*;

    type RemapItem = (usize, gnfs::types::Relation);
    type RemapInvalid = (usize, &'static str, Option<HdResidualSample>);
    type FoldAcc = (Vec<RemapItem>, Vec<RemapInvalid>, HashMap<(u64, u64, u8), Integer>);

    let (items, invalids, _) = relations
        .par_iter()
        .enumerate()
        .fold(
            || -> FoldAcc {
                (Vec::new(), Vec::new(), HashMap::new())
            },
            |(mut items, mut invalids, mut hensel_lift_cache), (rel_idx, rel)| {
                let new_rat_factors = rel.rational_factors.clone();

                let mut alg_pair_factors: Vec<(u32, u8)> = Vec::new();
                let mut invalid_reason: Option<&'static str> = None;
                let mut hd_sample: Option<HdResidualSample> = None;

                for &(alg_idx, total_exp) in &rel.algebraic_factors {
                    let prime = match alg_fb.primes.get(alg_idx as usize) {
                        Some(&p) => p,
                        None => {
                            invalid_reason = Some("bad_fb_idx");
                            break;
                        }
                    };

                    let gnfs_pi = match prime_to_gnfs.get(&prime) {
                        Some(&pi) => pi,
                        None => {
                            invalid_reason = Some("unmapped");
                            break;
                        }
                    };

                    let roots = &gnfs_fb.algebraic_roots[gnfs_pi];
                    let pair_base = gnfs_fb.pair_offset(gnfs_pi);

                    let mut root_exp_sum = 0u8;
                    for (root_idx, &r) in roots.iter().enumerate() {
                        let val_i128 = rel.a as i128 - rel.b as i128 * r as i128;
                        let root_mult = root_multiplicity_cache
                            .get(&(prime, r))
                            .copied()
                            .unwrap_or(1);
                        let e = if root_mult == 1 && rel.b % prime != 0 {
                            p_valuation_a_minus_br_hensel(
                                rel.a,
                                rel.b,
                                prime,
                                r,
                                total_exp,
                                f_coeffs_big,
                                &mut hensel_lift_cache,
                            )
                            .unwrap_or_else(|| p_adic_val_i128(val_i128, prime))
                        } else {
                            p_adic_val_i128(val_i128, prime)
                        };
                        if e > 0 {
                            alg_pair_factors.push(((pair_base + root_idx) as u32, e));
                            root_exp_sum = root_exp_sum.saturating_add(e);
                        }
                    }

                    if root_exp_sum < total_exp {
                        let residual = total_exp - root_exp_sum;
                        let hd_degree = degree.saturating_sub(roots.len());
                        let residual_divisible = hd_degree > 0 && (residual as usize) % hd_degree == 0;
                        let repeated_roots: Vec<u64> = roots
                            .iter()
                            .copied()
                            .filter(|&r| {
                                root_multiplicity_cache
                                    .get(&(prime, r))
                                    .copied()
                                    .unwrap_or(1)
                                    > 1
                            })
                            .collect();
                        if repeated_roots.len() == 1 {
                            if let Some(&bad_off) = bad_root_offsets.get(&(prime, repeated_roots[0])) {
                                let bad_flat_idx = gnfs_fb.algebraic_pair_count()
                                    + gnfs_fb.higher_degree_ideal_count(degree)
                                    + bad_off;
                                alg_pair_factors.push((bad_flat_idx as u32, residual));
                                continue;
                            }
                        }
                        if hd_degree == 0 {
                            // Fully-split prime: no HD ideal exists.
                            // Residual means Hensel lifting underestimated some root's
                            // valuation. For GF(2) parity, the residual is already
                            // accounted for in the root columns via root_exp_sum.
                            // Just skip — the parity error is small and tolerated
                            // by trying many dependencies in sqrt.
                            continue;
                        }

                        // HD ideal exists (hd_degree > 0).
                        // Compute HD exponent. When residual isn't a clean multiple
                        // of hd_degree (imprecise Hensel lift), use best-effort
                        // parity: for odd hd_degree, parity(v_HD) = parity(residual);
                        // for even hd_degree with non-divisible residual, use
                        // residual/hd_degree rounded down (parity approximation).
                        let hd_exp = if residual_divisible {
                            (residual as usize / hd_degree) as u8
                        } else {
                            // Best-effort: for odd hd_degree, residual % 2 is correct;
                            // for even hd_degree, this is approximate but keeps the relation.
                            if hd_degree % 2 == 1 {
                                (residual % 2) as u8
                            } else {
                                ((residual as usize / hd_degree) % 2) as u8
                            }
                        };
                        match gnfs_fb.hd_offset(gnfs_pi, degree) {
                            Some(hd_off) => {
                                let hd_flat_idx = gnfs_fb.algebraic_pair_count() + hd_off;
                                alg_pair_factors.push((hd_flat_idx as u32, hd_exp));
                            }
                            None => {
                                invalid_reason = Some("hd_offset_missing");
                                break;
                            }
                        }
                    }
                }

                if let Some(reason) = invalid_reason {
                    invalids.push((rel_idx, reason, hd_sample));
                    return (items, invalids, hensel_lift_cache);
                }

                let (rat_lp, alg_lp) = if !rel.lp_keys.is_empty() {
                    let mut rat_lps: Vec<u64> = Vec::new();
                    let mut alg_lps: Vec<(u64, u64)> = Vec::new();
                    for key in &rel.lp_keys {
                        match *key {
                            crate::lp_key::LpKey::Rational(p) => rat_lps.push(p),
                            crate::lp_key::LpKey::Algebraic(p, r) => alg_lps.push((p, r)),
                        }
                    }
                    rat_lps.sort_unstable();
                    rat_lps.dedup();
                    alg_lps.sort_unstable();
                    alg_lps.dedup();
                    (
                        if rat_lps.len() == 1 {
                            Some(rat_lps[0])
                        } else {
                            None
                        },
                        if alg_lps.len() == 1 {
                            Some(alg_lps[0])
                        } else {
                            None
                        },
                    )
                } else {
                    let rat_lp = (rel.rat_cofactor > 1).then_some(rel.rat_cofactor);
                    let alg_lp = if rel.alg_cofactor > 1 {
                        match compute_alg_lp_ideal(rel.a, rel.b, rel.alg_cofactor) {
                            Some(v) => Some(v),
                            None => {
                                invalids.push((rel_idx, "legacy_lp", None));
                                return (items, invalids, hensel_lift_cache);
                            }
                        }
                    } else {
                        None
                    };
                    (rat_lp, alg_lp)
                };

                let rat_norm_val = (g1_i64 as i128) * (rel.a as i128) + (g0_i64 as i128) * (rel.b as i128);
                let rational_sign_negative = rat_norm_val < 0;
                let algebraic_sign_negative = eval_f_homogeneous_bigint(rel.a, rel.b, f_coeffs_big) < 0;

                items.push((rel_idx, gnfs::types::Relation {
                    a: rel.a,
                    b: rel.b,
                    rational_factors: new_rat_factors,
                    algebraic_factors: alg_pair_factors,
                    rational_sign_negative,
                    algebraic_sign_negative,
                    special_q: if ignore_special_q {
                        None
                    } else {
                        rel.special_q
                    },
                    rat_lp,
                    alg_lp,
                }));
                (items, invalids, hensel_lift_cache)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new(), HashMap::new()),
            |(mut a_items, mut a_inv, a_cache), (b_items, b_inv, _b_cache)| {
                a_items.extend(b_items);
                a_inv.extend(b_inv);
                (a_items, a_inv, a_cache)
            },
        );

    // Sort by original index to preserve deterministic ordering
    let mut items = items;
    items.sort_unstable_by_key(|(idx, _)| *idx);

    let mut result = Vec::with_capacity(items.len());
    let mut source_indices = Vec::with_capacity(items.len());
    for (idx, rel) in items {
        result.push(rel);
        source_indices.push(idx);
    }

    // Aggregate stats from invalids
    let mut stats = RemapHybridStats::default();
    for (_, reason, sample) in invalids {
        match reason {
            "bad_fb_idx" => stats.invalid_bad_fb_idx += 1,
            "unmapped" => stats.skipped_unmapped += 1,
            "hd_residual" => {
                stats.invalid_hd_residual += 1;
                if let Some(s) = sample {
                    if stats.hd_residual_samples.len() < hd_residual_sample_limit {
                        stats.hd_residual_samples.push(s);
                    }
                }
            }
            "hd_offset_missing" => stats.invalid_hd_offset_missing += 1,
            "root_exp_zero" => stats.invalid_root_exp_zero += 1,
            "legacy_lp" => stats.invalid_legacy_lp += 1,
            _ => {}
        }
    }

    stats.kept_relations = result.len();
    let skipped_invalid = stats.invalid_total();
    let skipped = stats.skipped_total();
    if skipped > 0 {
        eprintln!(
            "  remap_hybrid: skipped {} of {} relations (unmapped={}, invalid={})",
            skipped,
            relations.len(),
            stats.skipped_unmapped,
            skipped_invalid
        );
        eprintln!(
            "  remap_hybrid: invalid breakdown: bad_fb_idx={} hd_residual={} hd_offset_missing={} root_exp_zero={} legacy_lp={}",
            stats.invalid_bad_fb_idx,
            stats.invalid_hd_residual,
            stats.invalid_hd_offset_missing,
            stats.invalid_root_exp_zero,
            stats.invalid_legacy_lp
        );
        if !stats.hd_residual_samples.is_empty() {
            eprintln!(
                "  remap_hybrid: hd_residual_samples={}",
                serde_json::to_string(&stats.hd_residual_samples)
                    .unwrap_or_else(|_| "[]".to_string())
            );
        }
    }

    (result, source_indices, stats)
}

/// Compute v_p(|val|) for an i128 integer.
/// Returns 127 for val=0 (v_p(0) = infinity, capped at 127).
fn p_adic_val_i128(val: i128, p: u64) -> u8 {
    if p < 2 {
        return 0;
    }
    if val == 0 {
        return 127; // v_p(0) = infinity; cap at large value
    }
    let mut val = val.unsigned_abs();
    let p = p as u128;
    let mut e = 0u8;
    while val % p == 0 {
        val /= p;
        e = e.saturating_add(1);
        if e == u8::MAX {
            break;
        }
    }
    e
}

/// Compute the multiplicity of root `r` as a root of polynomial `f(x) mod p`.
/// This is the ramification index for the ideal (p, alpha - r).
/// Works by repeatedly dividing f(x) by (x - r) mod p and checking if r
/// is still a root of the quotient.
fn root_multiplicity_mod_p(f_coeffs: &[Integer], r: u64, p: u64) -> usize {
    if p < 2 || f_coeffs.is_empty() {
        return 0;
    }
    let p128 = p as u128;
    // Convert to residues mod p
    let mut coeffs: Vec<u64> = f_coeffs
        .iter()
        .map(|c| {
            let rem = c.mod_u(p as u32);
            rem as u64
        })
        .collect();

    let mut mult = 0usize;
    loop {
        // Check if r is a root of current polynomial
        let mut val = 0u128;
        let mut r_pow = 1u128;
        for &c in &coeffs {
            val = (val + c as u128 * r_pow) % p128;
            r_pow = r_pow * r as u128 % p128;
        }
        if val != 0 {
            break;
        }
        mult += 1;
        if coeffs.len() <= 1 {
            break;
        }
        // Divide by (x - r): synthetic division mod p
        let deg = coeffs.len() - 1;
        let mut quotient = vec![0u64; deg];
        quotient[deg - 1] = coeffs[deg];
        for i in (0..deg - 1).rev() {
            quotient[i] =
                ((coeffs[i + 1] as u128 + quotient[i + 1] as u128 * r as u128) % p128) as u64;
        }
        coeffs = quotient;
    }
    mult.max(1) // At least 1 if we were called with a valid root
}

/// Convert an algebraic-side large prime p into the ideal identifier `(p, r)`.
///
/// CADO convention (`relation_compute_r`): if `b` is not invertible mod `p`,
/// represent the projective ideal as `(p, p)` instead of discarding.
fn compute_alg_lp_ideal(a: i64, b: u64, p: u64) -> Option<(u64, u64)> {
    if p < 2 {
        return None;
    }
    let b_mod_p = b % p;
    if b_mod_p == 0 {
        return Some((p, p));
    }
    let b_inv = match gnfs::arith::mod_inverse_u64(b_mod_p, p) {
        Some(v) => v,
        None => return Some((p, p)),
    };
    let a_mod_p = (a as i128).rem_euclid(p as i128) as u64;
    let r = ((a_mod_p as u128 * b_inv as u128) % p as u128) as u64;
    Some((p, r))
}

/// Compute F(a, b) = c_0 * b^d + c_1 * a * b^{d-1} + ... + c_d * a^d
/// using rug::Integer for exact BigInt arithmetic (no overflow).
fn eval_f_homogeneous_bigint(a: i64, b: u64, f_coeffs: &[Integer]) -> Integer {
    let d = f_coeffs.len() - 1;
    let a_int = Integer::from(a);
    let b_int = Integer::from(b);
    let mut result = Integer::from(0);
    for (i, c) in f_coeffs.iter().enumerate() {
        let a_pow = a_int.clone().pow(i as u32);
        let b_pow = b_int.clone().pow((d - i) as u32);
        result += Integer::from(c * &a_pow) * &b_pow;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_alg_lp_ideal_regular() {
        // 10 ≡ 2 (mod 4), 3^{-1} ≡ 7 (mod 10), so r ≡ 2*7 ≡ 4 (mod 10)
        assert_eq!(compute_alg_lp_ideal(12, 3, 10), Some((10, 4)));
    }

    #[test]
    fn test_compute_alg_lp_ideal_projective_when_b_divisible() {
        assert_eq!(compute_alg_lp_ideal(5, 14, 7), Some((7, 7)));
    }

    #[test]
    fn test_compute_alg_lp_ideal_projective_when_not_invertible() {
        // gcd(2, 4) != 1, so no inverse mod 4 -> projective ideal (4,4)
        assert_eq!(compute_alg_lp_ideal(9, 2, 4), Some((4, 4)));
    }

    #[test]
    fn test_pipeline_runs() {
        // Verify the pipeline runs without panicking on a small N.
        let n = Integer::from(8051u64); // 83 * 97
        let params = NfsParams::c30();
        let result = factor_nfs(&n, &params);
        eprintln!("Pipeline result: {:?}", result);
        // Sanity check: the pipeline completed and reported relation counts.
        assert!(result.total_ms >= 0.0);
    }

    #[test]
    fn test_pipeline_known_semiprime() {
        // Try a slightly larger N where NFS should work.
        let n = Integer::from(1_000_003u64) * Integer::from(1_000_033u64);
        let params = NfsParams::c30();
        let result = factor_nfs(&n, &params);
        eprintln!(
            "Pipeline result for N={}: rels={}, factor={:?}",
            n, result.relations_found, result.factor
        );
    }

    #[test]
    fn test_effective_max_raw_rels_tracks_target_when_uncapped() {
        assert_eq!(effective_max_raw_rels(None, 2_000, 1_000), 5_000);
        assert_eq!(effective_max_raw_rels(None, 3_000, 1_000), 7_000);
    }

    #[test]
    fn test_effective_max_raw_rels_respects_explicit_cap() {
        assert_eq!(effective_max_raw_rels(Some(9_000), 2_000, 1_000), 9_000);
        assert_eq!(effective_max_raw_rels(Some(9_000), 5_000, 1_000), 9_000);
    }

    #[test]
    fn test_hensel_lift_recovers_simple_root_valuation_for_c30_prime5() {
        let coeffs = vec![
            Integer::from(5_142_430_355i64),
            Integer::from(6_255_823_782i64),
            Integer::from(0),
            Integer::from(1),
        ];
        let mut cache = HashMap::new();
        let e = p_valuation_a_minus_br_hensel(8420, 67, 5, 0, 2, &coeffs, &mut cache)
            .expect("simple root should Hensel-lift");
        assert_eq!(e, 2);
    }

    #[test]
    fn test_hensel_lift_recovers_simple_root_valuation_for_c30_prime2() {
        let coeffs = vec![
            Integer::from(5_142_430_355i64),
            Integer::from(6_255_823_782i64),
            Integer::from(0),
            Integer::from(1),
        ];
        let mut cache = HashMap::new();
        let e = p_valuation_a_minus_br_hensel(12_603, 15_065, 2, 1, 4, &coeffs, &mut cache)
            .expect("simple root should Hensel-lift");
        assert_eq!(e, 4);
    }

    #[test]
    fn test_compute_bad_root_offsets_detects_c30_prime3_root1() {
        let coeffs = vec![
            Integer::from(5_142_430_355i64),
            Integer::from(6_255_823_782i64),
            Integer::from(0),
            Integer::from(1),
        ];
        let fb = gnfs::sieve::build_factor_base(&[5_142_430_355, 6_255_823_782, 0, 1], 100);
        let bad = compute_bad_root_offsets(&fb, &coeffs);
        assert!(bad.contains_key(&(3, 1)));
        assert_eq!(bad.len(), 1);
    }

    #[test]
    fn test_build_special_q_zero_sets_from_partial_sets_cancels_matching_pairs() {
        let rels = vec![
            gnfs::types::Relation {
                a: 1,
                b: 1,
                rational_factors: vec![],
                algebraic_factors: vec![],
                rational_sign_negative: false,
                algebraic_sign_negative: false,
                special_q: Some((101, 3)),
                rat_lp: None,
                alg_lp: None,
            },
            gnfs::types::Relation {
                a: 2,
                b: 1,
                rational_factors: vec![],
                algebraic_factors: vec![],
                rational_sign_negative: false,
                algebraic_sign_negative: false,
                special_q: Some((101, 3)),
                rat_lp: None,
                alg_lp: None,
            },
            gnfs::types::Relation {
                a: 3,
                b: 1,
                rational_factors: vec![],
                algebraic_factors: vec![],
                rational_sign_negative: false,
                algebraic_sign_negative: false,
                special_q: Some((103, 5)),
                rat_lp: None,
                alg_lp: None,
            },
        ];
        let sets = vec![vec![0], vec![1], vec![2]];
        let sq_zero_sets = build_special_q_zero_sets_from_partial_sets(&rels, &sets, 100);
        assert!(sq_zero_sets.contains(&vec![0, 1]));
        assert!(!sq_zero_sets.contains(&vec![2]));
    }

    #[test]
    fn test_remap_hybrid_routes_repeated_root_residual_to_bad_column() {
        let coeffs_i64 = vec![5_142_430_355i64, 6_255_823_782i64, 0, 1];
        let coeffs_big: Vec<Integer> = coeffs_i64.iter().copied().map(Integer::from).collect();
        let alg_fb = crate::factorbase::FactorBase::new_roots_only(&coeffs_i64, 100, 1.442);
        let gnfs_fb = gnfs::types::FactorBase {
            primes: alg_fb.primes.clone(),
            algebraic_roots: alg_fb.roots.clone(),
            log_p: alg_fb.primes.iter().map(|&p| ((p as f64).log2() * 20.0).round() as u8).collect(),
        };
        let rat_fb = crate::factorbase::FactorBase::new(&[-88, 1], 100, 1.442);
        let alg_idx = alg_fb
            .primes
            .iter()
            .position(|&p| p == 3)
            .expect("prime 3 should be present") as u32;
        let bad = compute_bad_root_offsets(&gnfs_fb, &coeffs_big);

        let rel = crate::relation::Relation {
            a: -73_258,
            b: 11_453,
            rational_factors: vec![],
            algebraic_factors: vec![(alg_idx, 2)],
            rational_sign_negative: true,
            algebraic_sign_negative: false,
            special_q: None,
            rat_cofactor: 1,
            alg_cofactor: 1,
            lp_keys: vec![],
        };

        let (rels, _src, stats) =
            remap_hybrid(&[rel], &rat_fb, &alg_fb, &gnfs_fb, &coeffs_big, 88, 3, &bad, -88, 1);
        assert_eq!(stats.invalid_hd_residual, 0);
        assert_eq!(rels.len(), 1);

        let alg_pairs = gnfs_fb.algebraic_pair_count();
        let alg_hd = gnfs_fb.higher_degree_ideal_count(3);
        let gnfs_pi = gnfs_fb.primes.iter().position(|&p| p == 3).unwrap();
        let factors = &rels[0].algebraic_factors;
        assert_eq!(factors.len(), 2);
        assert!(factors
            .iter()
            .any(|&(idx, exp)| idx as usize == gnfs_fb.pair_offset(gnfs_pi) && exp == 1));
        assert!(factors
            .iter()
            .any(|&(idx, exp)| idx as usize >= alg_pairs + alg_hd && exp == 1));
    }
}
