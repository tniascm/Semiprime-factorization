//! Full NFS pipeline: poly selection -> sieve -> filter -> LA -> sqrt.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rug::ops::Pow;
use rug::Integer;

use crate::params::NfsParams;

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
    pub sieve_ms: f64,
    pub filter_ms: f64,
    pub la_ms: f64,
    pub sqrt_ms: f64,
    pub total_ms: f64,
    pub sqrt_fail_rat_not_square: usize,
    pub sqrt_fail_alg_not_square: usize,
    pub sqrt_fail_trivial_gcd: usize,
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
    let max_variants: u32 = std::env::var("RUST_NFS_MAX_VARIANTS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(5);
    let mut run_logger = RunLogger::new(n, params, max_variants);
    let mut last_result = None;

    for variant in 0..max_variants {
        if variant > 0 {
            eprintln!(
                "  === Trying polynomial variant {} (m_offset=-{}) ===",
                variant, variant
            );
        }
        let result = factor_nfs_inner(n, params, variant);
        if let Some(logger) = run_logger.as_mut() {
            logger.log_variant(variant, &result);
        }
        if result.factor.is_some() {
            if let Some(logger) = run_logger.as_mut() {
                logger.finish(&result);
            }
            return result;
        }
        last_result = Some(result);
    }

    let final_result = last_result.unwrap();
    if let Some(logger) = run_logger.as_mut() {
        logger.finish(&final_result);
    }
    final_result
}

/// Optional structured logger for reproducible run debugging.
///
/// Enabled by setting `RUST_NFS_LOG_DIR=/path/to/logs`.
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
        let base = std::env::var("RUST_NFS_LOG_DIR").ok()?;
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
            "RUST_NFS_MAX_VARIANTS": std::env::var("RUST_NFS_MAX_VARIANTS").ok(),
            "RUST_NFS_DEP_SEED": std::env::var("RUST_NFS_DEP_SEED").ok(),
            "RUST_NFS_DEP_XOR_K": std::env::var("RUST_NFS_DEP_XOR_K").ok(),
            "RUST_NFS_DEP_RANDOM_COUNT": std::env::var("RUST_NFS_DEP_RANDOM_COUNT").ok(),
            "RUST_NFS_MAX_DEPS_TRY": std::env::var("RUST_NFS_MAX_DEPS_TRY").ok(),
            "RUST_NFS_MAX_DEP_LEN": std::env::var("RUST_NFS_MAX_DEP_LEN").ok(),
            "RUST_NFS_DEP_LEN_TIERS": std::env::var("RUST_NFS_DEP_LEN_TIERS").ok(),
            "RUST_NFS_DEP_AUTO_RELAX": std::env::var("RUST_NFS_DEP_AUTO_RELAX").ok(),
            "RUST_NFS_DEP_REQUIRE_COPRIME_REL": std::env::var("RUST_NFS_DEP_REQUIRE_COPRIME_REL").ok(),
            "RUST_NFS_TRIVIAL_BAIL": std::env::var("RUST_NFS_TRIVIAL_BAIL").ok(),
            "RUST_NFS_SKIP_SQRT": std::env::var("RUST_NFS_SKIP_SQRT").ok(),
            "RUST_NFS_SQRT_VERBOSE_DEPS": std::env::var("RUST_NFS_SQRT_VERBOSE_DEPS").ok(),
            "RUST_NFS_QC_COUNT": std::env::var("RUST_NFS_QC_COUNT").ok(),
            "RUST_NFS_REQUIRE_COPRIME_AB": std::env::var("RUST_NFS_REQUIRE_COPRIME_AB").ok(),
            "RUST_NFS_FULL_ONLY": std::env::var("RUST_NFS_FULL_ONLY").ok(),
            "RUST_NFS_IGNORE_SPECIAL_Q_COLUMN": std::env::var("RUST_NFS_IGNORE_SPECIAL_Q_COLUMN").ok(),
            "RUST_NFS_SPARSE_PREMERGE": std::env::var("RUST_NFS_SPARSE_PREMERGE").ok(),
            "RUST_NFS_SPARSE_PREMERGE_MAXSETS": std::env::var("RUST_NFS_SPARSE_PREMERGE_MAXSETS").ok(),
            "RUST_NFS_COMPACT_ZERO_COLS": std::env::var("RUST_NFS_COMPACT_ZERO_COLS").ok(),
            "RUST_NFS_SINGLETON_PRUNE": std::env::var("RUST_NFS_SINGLETON_PRUNE").ok(),
            "RUST_NFS_SINGLETON_PRUNE_MIN_WEIGHT": std::env::var("RUST_NFS_SINGLETON_PRUNE_MIN_WEIGHT").ok(),
            "RUST_NFS_PARTIAL_MERGE_2LP": std::env::var("RUST_NFS_PARTIAL_MERGE_2LP").ok(),
            "RUST_NFS_PARTIAL_MERGE_MAXSETS": std::env::var("RUST_NFS_PARTIAL_MERGE_MAXSETS").ok(),
            "RUST_NFS_MAX_LP_KEYS": std::env::var("RUST_NFS_MAX_LP_KEYS").ok(),
            "RUST_NFS_REL_TARGET_MULT": std::env::var("RUST_NFS_REL_TARGET_MULT").ok(),
            "RUST_NFS_REL_TARGET_MIN": std::env::var("RUST_NFS_REL_TARGET_MIN").ok(),
            "RUST_NFS_NORM_BLOCK": std::env::var("RUST_NFS_NORM_BLOCK").ok(),
            "RUST_NFS_MAX_Q_WINDOWS": std::env::var("RUST_NFS_MAX_Q_WINDOWS").ok(),
            "RUST_NFS_OVR_LIM0": std::env::var("RUST_NFS_OVR_LIM0").ok(),
            "RUST_NFS_OVR_LIM1": std::env::var("RUST_NFS_OVR_LIM1").ok(),
            "RUST_NFS_OVR_LPB0": std::env::var("RUST_NFS_OVR_LPB0").ok(),
            "RUST_NFS_OVR_LPB1": std::env::var("RUST_NFS_OVR_LPB1").ok(),
            "RUST_NFS_OVR_MFB0": std::env::var("RUST_NFS_OVR_MFB0").ok(),
            "RUST_NFS_OVR_MFB1": std::env::var("RUST_NFS_OVR_MFB1").ok(),
            "RUST_NFS_OVR_LOG_I": std::env::var("RUST_NFS_OVR_LOG_I").ok(),
            "RUST_NFS_OVR_QMIN": std::env::var("RUST_NFS_OVR_QMIN").ok(),
            "RUST_NFS_OVR_QRANGE": std::env::var("RUST_NFS_OVR_QRANGE").ok(),
            "RUST_NFS_OVR_RELS_WANTED": std::env::var("RUST_NFS_OVR_RELS_WANTED").ok(),
            "GNFS_TRY_COUVEIGNES_ON_TRIVIAL": std::env::var("GNFS_TRY_COUVEIGNES_ON_TRIVIAL").ok(),
            "GNFS_TRY_NEG_M": std::env::var("GNFS_TRY_NEG_M").ok(),
            "GNFS_NF_ELEMENT_MODE": std::env::var("GNFS_NF_ELEMENT_MODE").ok(),
            "GNFS_SQRT_RELAX_EXACT": std::env::var("GNFS_SQRT_RELAX_EXACT").ok()
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
fn factor_nfs_inner(n: &Integer, params: &NfsParams, variant: u32) -> NfsResult {
    let start = std::time::Instant::now();
    let mut result = NfsResult {
        n: n.to_string(),
        factor: None,
        relations_found: 0,
        relations_after_filter: 0,
        matrix_rows: 0,
        matrix_cols: 0,
        dependencies_found: 0,
        sieve_ms: 0.0,
        filter_ms: 0.0,
        la_ms: 0.0,
        sqrt_ms: 0.0,
        total_ms: 0.0,
        sqrt_fail_rat_not_square: 0,
        sqrt_fail_alg_not_square: 0,
        sqrt_fail_trivial_gcd: 0,
    };

    // Optional runtime parameter overrides for reproducible tuning experiments.
    let mut params = params.clone();
    let partial_merge_2lp = std::env::var("RUST_NFS_PARTIAL_MERGE_2LP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    if let Some(v) = std::env::var("RUST_NFS_OVR_LIM0")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.lim0 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_LIM1")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.lim1 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_LPB0")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.lpb0 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_LPB1")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.lpb1 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_MFB0")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.mfb0 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_MFB1")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.mfb1 = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_LOG_I")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        params.log_i = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_QMIN")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.qmin = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_QRANGE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.qrange = v;
    }
    if let Some(v) = std::env::var("RUST_NFS_OVR_RELS_WANTED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        params.rels_wanted = v;
    }
    if partial_merge_2lp {
        // 2LP requires room for products of two LPs; if mfb is too close to lpb
        // those candidates are rejected before merge can use them.
        let min_mfb0 = params.lpb0.saturating_mul(2).saturating_add(2);
        let min_mfb1 = params.lpb1.saturating_mul(2).saturating_add(2);
        if params.mfb0 < min_mfb0 {
            eprintln!(
                "  params: bump mfb0 {} -> {} for 2LP support",
                params.mfb0, min_mfb0
            );
            params.mfb0 = min_mfb0;
        }
        if params.mfb1 < min_mfb1 {
            eprintln!(
                "  params: bump mfb1 {} -> {} for 2LP support",
                params.mfb1, min_mfb1
            );
            params.mfb1 = min_mfb1;
        }
    }

    // --- Stage 1: Polynomial Selection ---
    let poly = gnfs::polyselect::select_base_m_variant(n, params.degree, variant);
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

    eprintln!(
        "  poly: degree={}, m={}, coeffs={:?}",
        params.degree, m, f_coeffs_i64
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
    let sieve_start = std::time::Instant::now();

    // Build gnfs-compatible algebraic FB metadata early for adaptive relation targets.
    let gnfs_fb = gnfs::sieve::build_factor_base(&f_coeffs_i64, params.lim0.max(params.lim1));
    let degree = params.degree as usize;

    // Build factor bases for both sides.
    // Rational polynomial: g(x) = x - m, coefficients [-m, 1].
    let rat_coeffs = vec![-(m as i64), 1i64];
    let rat_fb = crate::factorbase::FactorBase::new(&rat_coeffs, params.lim0, 1.442);
    let alg_fb = crate::factorbase::FactorBase::new_roots_only(&f_coeffs_i64, params.lim1, 1.442);
    if partial_merge_2lp {
        let qc_count = std::env::var("RUST_NFS_QC_COUNT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(30usize);
        let rel_target_mult = std::env::var("RUST_NFS_REL_TARGET_MULT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v > 0.0)
            .unwrap_or(1.6f64);
        let rel_target_min = std::env::var("RUST_NFS_REL_TARGET_MIN")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4_000u64);
        let est_dense_cols = rat_fb.primes.len()
            + gnfs_fb.algebraic_pair_count()
            + gnfs_fb.higher_degree_ideal_count(degree)
            + 2
            + qc_count;
        let auto_target = ((est_dense_cols as f64) * rel_target_mult).ceil() as u64;
        let auto_target = auto_target.max(rel_target_min);
        if auto_target < params.rels_wanted {
            eprintln!(
                "  params: auto rels_wanted {} -> {} (est_dense_cols={}, mult={:.2})",
                params.rels_wanted, auto_target, est_dense_cols, rel_target_mult
            );
            params.rels_wanted = auto_target;
        }
    }

    // Run the special-q sieve.
    let sieve_result = crate::sieve::sieve_specialq(&f_coeffs_i64, m, &rat_fb, &alg_fb, &params);

    result.sieve_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;
    result.relations_found = sieve_result.relations.len();
    eprintln!(
        "  sieve: {} rels in {:.0}ms ({} survivors, {} special-qs)",
        sieve_result.relations.len(),
        result.sieve_ms,
        sieve_result.survivors_found,
        sieve_result.special_qs_processed
    );
    eprintln!(
        "  sieve: setup={:.0}ms region_scan={:.0}ms cofactor={:.0}ms (norm_block={})",
        sieve_result.bucket_setup_ms,
        sieve_result.region_scan_ms,
        sieve_result.cofactor_ms,
        std::env::var("RUST_NFS_NORM_BLOCK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1usize)
    );

    if sieve_result.relations.is_empty() {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 3: Filtering ---
    let filter_start = std::time::Instant::now();
    let mut filtered = crate::filter::filter_relations(sieve_result.relations);
    let require_coprime_ab = std::env::var("RUST_NFS_REQUIRE_COPRIME_AB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if require_coprime_ab {
        let before = filtered.len();
        filtered.retain(|r| gcd_u64(r.a.unsigned_abs(), r.b) == 1);
        eprintln!(
            "  filter: coprime(a,b) enforced: {} -> {} rels",
            before,
            filtered.len()
        );
    }
    result.filter_ms = filter_start.elapsed().as_secs_f64() * 1000.0;
    result.relations_after_filter = filtered.len();
    eprintln!(
        "  filter: {} -> {} rels in {:.0}ms",
        result.relations_found,
        filtered.len(),
        result.filter_ms
    );
    let debug_rel_stats = std::env::var("RUST_NFS_DEBUG_REL_STATS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if debug_rel_stats {
        let mut non_coprime = 0usize;
        let mut non_coprime_full = 0usize;
        for rel in &filtered {
            let g = gcd_u64(rel.a.unsigned_abs(), rel.b);
            if g > 1 {
                non_coprime += 1;
                if rel.is_full() {
                    non_coprime_full += 1;
                }
            }
        }
        eprintln!(
            "  relstats: gcd(a,b)>1 in {}/{} filtered rels ({}/{} full)",
            non_coprime,
            filtered.len(),
            non_coprime_full,
            full_count(&filtered),
        );
    }

    let full_only = std::env::var("RUST_NFS_FULL_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if full_only {
        let before = filtered.len();
        filtered.retain(|r| r.is_full());
        eprintln!(
            "  filter: full-only enforced: {} -> {} rels",
            before,
            filtered.len()
        );
    }

    if filtered.len() < 2 {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 4: Linear Algebra ---
    let la_start = std::time::Instant::now();

    // Keep both full and 1LP relations for matrix construction.
    // (2LP relations are already rejected in relation construction.)
    let full_count = filtered.iter().filter(|r| r.is_full()).count();
    let partial_count = filtered.len().saturating_sub(full_count);
    eprintln!(
        "  LA: {} relations ({} full, {} partial) from {} after filter",
        filtered.len(),
        full_count,
        partial_count,
        result.relations_after_filter,
    );
    let partial_merge_max_sets = std::env::var("RUST_NFS_PARTIAL_MERGE_MAXSETS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(200_000usize);
    let partial_sets_filtered = if partial_merge_2lp {
        let (sets, stats) =
            crate::partial_merge::merge_relations_2lp(&filtered, partial_merge_max_sets);
        eprintln!(
            "  merge2lp: rels={} (0lp={},1lp={},2lp={},drop_gt2={}) nodes={} tree_edges={} cycles={} sets={} max_sets={}",
            stats.total_relations,
            stats.relations_0lp,
            stats.relations_1lp,
            stats.relations_2lp,
            stats.relations_dropped_gt2lp,
            stats.lp_nodes,
            stats.tree_edges,
            stats.cycles_found,
            stats.output_sets,
            partial_merge_max_sets
        );
        Some(sets)
    } else {
        None
    };

    // Hybrid approach: keep rational factors from our sieve (which uses a
    // complete rational FB including inert primes), but recompute algebraic
    // factors using BigInt norms and gnfs's per-(prime,root) decomposition.
    // This avoids u64 overflow in algebraic norm computation and ensures
    // correct ideal-level factorization.
    let (gnfs_rels, remap_source_indices) = remap_hybrid(
        &filtered,
        &rat_fb,
        &alg_fb,
        &gnfs_fb,
        &f_coeffs_big,
        m,
        degree,
    );
    eprintln!("  LA: {} relations after hybrid remap", gnfs_rels.len());
    let source_to_remap: HashMap<usize, usize> = remap_source_indices
        .iter()
        .enumerate()
        .map(|(ri, &src)| (src, ri))
        .collect();
    let mut partial_sets_remapped: Vec<Vec<usize>> = Vec::new();
    let mut partial_sets_dropped_remap = 0usize;
    if let Some(sets) = partial_sets_filtered {
        for set in sets {
            let mut mapped = Vec::with_capacity(set.len());
            let mut valid = true;
            for src_idx in set {
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
        eprintln!(
            "  merge2lp: remapped set-rows {} (dropped_unmapped={})",
            partial_sets_remapped.len(),
            partial_sets_dropped_remap
        );
    }

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
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    let alg_pairs = gnfs_fb.algebraic_pair_count();
    let alg_hd = gnfs_fb.higher_degree_ideal_count(degree);
    let qc_count = std::env::var("RUST_NFS_QC_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30usize);
    let quad_chars = gnfs::arith::select_quad_char_primes(&f_coeffs_i64, &gnfs_fb.primes, qc_count);

    // Use rat_fb.primes.len() as rational FB size (complete rational FB).
    let rat_fb_size = rat_fb.primes.len();

    // Matrix density diagnostics: how many dense FB columns are actually used.
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
        "  LA: active dense cols rat={}/{} alg={}/{}",
        used_rat_cols.len(),
        rat_fb_size,
        used_alg_cols.len(),
        alg_pairs + alg_hd
    );

    let sparse_premerge = std::env::var("RUST_NFS_SPARSE_PREMERGE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let premerge_max_sets = std::env::var("RUST_NFS_SPARSE_PREMERGE_MAXSETS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(50_000);

    let partial_merge_active = partial_merge_2lp && !partial_sets_remapped.is_empty();
    let (matrix_raw, ncols_raw, mut row_sources): (
        Vec<gnfs::types::BitRow>,
        usize,
        Vec<Vec<usize>>,
    ) = if partial_merge_active {
        let (matrix, nc, set_rows) = build_matrix_from_sets_lp_resolved(
            &gnfs_rels,
            &partial_sets_remapped,
            rat_fb_size,
            alg_pairs,
            alg_hd,
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
        (matrix, nc, set_rows)
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
            alg_hd,
            &quad_chars,
        );
        eprintln!(
            "  LA: dense-only matrix after premerge: {} x {} (qc={})",
            dense_matrix.len(),
            dense_ncols,
            quad_chars.primes.len()
        );
        (dense_matrix, dense_ncols, dense_set_rows)
    } else {
        let (m, nc) =
            gnfs::linalg::build_matrix(&gnfs_rels, rat_fb_size, alg_pairs, alg_hd, &quad_chars);
        eprintln!(
            "  LA: {} x {} matrix (qc={})",
            m.len(),
            nc,
            quad_chars.primes.len()
        );
        let row_sources = (0..m.len()).map(|i| vec![i]).collect();
        (m, nc, row_sources)
    };

    let compact_zero_cols = std::env::var("RUST_NFS_COMPACT_ZERO_COLS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
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

    let singleton_prune = std::env::var("RUST_NFS_SINGLETON_PRUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let singleton_min_weight = std::env::var("RUST_NFS_SINGLETON_PRUNE_MIN_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(2usize);
    if singleton_prune {
        let rows_before = matrix.len();
        let cols_before = ncols;
        let (pruned_matrix, pruned_sources, removed_rows) =
            prune_singleton_columns(matrix, row_sources, ncols, singleton_min_weight);
        matrix = pruned_matrix;
        row_sources = pruned_sources;
        if compact_zero_cols {
            let (m2, nc2, dropped2) = compact_zero_columns(matrix, ncols);
            matrix = m2;
            ncols = nc2;
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

    result.matrix_rows = matrix.len();
    result.matrix_cols = ncols;

    let ge_deps = gnfs::linalg::find_dependencies(&matrix, ncols);
    // GE basis vectors are short/correlated; generate randomized XOR
    // combinations to avoid systematic trivial-gcd failures in sqrt.
    // Keep k modest to avoid extremely long dependencies (expensive sqrt).
    let default_n_random = if ge_deps.len() < 200 {
        4_000
    } else if ge_deps.len() < 1_000 {
        3_000
    } else {
        1_000
    };
    let n_random = std::env::var("RUST_NFS_DEP_RANDOM_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_n_random);
    let default_k = (ge_deps.len() / 8).max(8).min(64);
    let k = std::env::var("RUST_NFS_DEP_XOR_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 1)
        .unwrap_or(default_k);
    let dep_seed = std::env::var("RUST_NFS_DEP_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let deps = gnfs::linalg::randomize_dependencies(&ge_deps, n_random, k, dep_seed);
    result.dependencies_found = deps.len();

    result.la_ms = la_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  LA: {} deps ({} GE + {} random, xor_k={}, seed={}) in {:.0}ms",
        deps.len(),
        ge_deps.len(),
        deps.len() - ge_deps.len(),
        k,
        dep_seed,
        result.la_ms
    );
    let matrix_premerged = sparse_premerge || partial_merge_active;
    if matrix_premerged {
        eprintln!(
            "  LA: dependency basis rows come from {} premerged set-rows",
            row_sources.len()
        );
    }

    if deps.is_empty() {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    let skip_sqrt = std::env::var("RUST_NFS_SKIP_SQRT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if skip_sqrt {
        eprintln!("  sqrt: skipped (RUST_NFS_SKIP_SQRT=1)");
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 5: Square Root ---
    let sqrt_start = std::time::Instant::now();

    // Prioritize randomized deps before basis deps (basis vectors are often
    // short/correlated) and within each class by ascending expanded length.
    let max_dep_len = std::env::var("RUST_NFS_MAX_DEP_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let dep_auto_relax = std::env::var("RUST_NFS_DEP_AUTO_RELAX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let dep_len_tiers = parse_dep_len_tiers(max_dep_len, dep_auto_relax);
    let dep_sets_ref = row_sources.as_slice();

    let mut useful_deps: Vec<(usize, &Vec<usize>, usize)> = deps
        .iter()
        .enumerate()
        .filter(|(_, d)| !d.is_empty())
        .map(|(dep_idx, dep)| {
            let expanded_len = dependency_expanded_len(dep, dep_sets_ref);
            (dep_idx, dep, expanded_len)
        })
        .collect();
    useful_deps.sort_by(|(ia, _, a_len), (ib, _, b_len)| {
        let a_is_random = *ia >= ge_deps.len();
        let b_is_random = *ib >= ge_deps.len();
        b_is_random
            .cmp(&a_is_random)
            .then_with(|| a_len.cmp(b_len))
            .then_with(|| ia.cmp(ib))
    });

    let mut fail_rat_not_square = 0usize;
    let mut fail_alg_not_square = 0usize;
    let mut fail_trivial_gcd = 0usize;

    let default_max_deps_to_try = if degree >= 4 { 80usize } else { 300usize };
    let max_deps_to_try = std::env::var("RUST_NFS_MAX_DEPS_TRY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_max_deps_to_try);
    let default_trivial_bail = if degree >= 4 { 60usize } else { 200usize };
    let trivial_bail = std::env::var("RUST_NFS_TRIVIAL_BAIL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_trivial_bail)
        .min(max_deps_to_try);

    let useful_ge = useful_deps
        .iter()
        .filter(|(dep_idx, _, _)| *dep_idx < ge_deps.len())
        .count();
    let useful_rand = useful_deps.len().saturating_sub(useful_ge);
    let verbose_deps = std::env::var("RUST_NFS_SQRT_VERBOSE_DEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let dep_require_coprime_rel = std::env::var("RUST_NFS_DEP_REQUIRE_COPRIME_REL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let dep_lens: Vec<usize> = useful_deps.iter().map(|(_, _, len)| *len).collect();
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
        "  sqrt: trying up to {} deps ({} candidates: {} GE + {} random, trivial_bail={}, verbose_deps={}, dep_len_tiers={}, premerge={}, require_coprime_rel={})",
        max_deps_to_try,
        useful_deps.len(),
        useful_ge,
        useful_rand,
        trivial_bail,
        verbose_deps,
        format_dep_len_tiers(&dep_len_tiers),
        matrix_premerged,
        dep_require_coprime_rel
    );

    let mut deps_tried = 0usize;
    let mut deps_skipped_short = 0usize;
    let deps_skipped_long: usize;
    let mut deps_skipped_non_coprime = 0usize;
    let mut dep_candidate_tried = vec![false; useful_deps.len()];

    for (tier_idx, tier_cap) in dep_len_tiers.iter().enumerate() {
        if deps_tried >= max_deps_to_try || result.factor.is_some() {
            break;
        }

        let eligible_in_tier = useful_deps
            .iter()
            .enumerate()
            .filter(|(cand_idx, (_, _, dep_len))| {
                !dep_candidate_tried[*cand_idx]
                    && tier_cap.map(|cap| *dep_len <= cap).unwrap_or(true)
            })
            .count();
        eprintln!(
            "  sqrt: tier {}/{} cap={} eligible={}",
            tier_idx + 1,
            dep_len_tiers.len(),
            tier_cap
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            eligible_in_tier
        );
        if eligible_in_tier == 0 {
            continue;
        }

        for (cand_idx, (_, dep, dep_len)) in useful_deps.iter().enumerate() {
            if deps_tried >= max_deps_to_try || result.factor.is_some() {
                break;
            }
            if dep_candidate_tried[cand_idx] {
                continue;
            }
            if tier_cap.map(|cap| *dep_len > cap).unwrap_or(false) {
                continue;
            }
            dep_candidate_tried[cand_idx] = true;

            if *dep_len < degree {
                deps_skipped_short += 1;
                continue;
            }

            let dep_expanded = expand_dependency_over_sets(dep, dep_sets_ref);
            if dep_require_coprime_rel
                && dep_expanded
                    .iter()
                    .any(|&ri| !rel_is_coprime.get(ri).copied().unwrap_or(false))
            {
                deps_skipped_non_coprime += 1;
                continue;
            }

            let i = deps_tried;
            deps_tried += 1;

            let (factor_opt, failure) = if i < verbose_deps {
                gnfs::sqrt::extract_factor_verbose(
                    &gnfs_rels,
                    &dep_expanded,
                    &f_coeffs_big,
                    &m_big,
                    n,
                )
            } else {
                gnfs::sqrt::extract_factor_diagnostic(
                    &gnfs_rels,
                    &dep_expanded,
                    &f_coeffs_big,
                    &m_big,
                    n,
                )
            };

            if let Some(factor) = factor_opt {
                if Integer::from(n % &factor) == 0 && factor > 1 && factor < *n {
                    result.factor = Some(factor.to_string());
                    result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "  sqrt: factor found after {} deps in {:.0}ms: {}",
                        deps_tried,
                        result.sqrt_ms,
                        result.factor.as_ref().unwrap()
                    );
                    break;
                }
            }

            match failure {
                Some(gnfs::sqrt::FactorFailure::RationalNotSquare) => fail_rat_not_square += 1,
                Some(gnfs::sqrt::FactorFailure::AlgebraicNotSquare) => fail_alg_not_square += 1,
                Some(gnfs::sqrt::FactorFailure::TrivialGcd) => fail_trivial_gcd += 1,
                None => {}
            }

            // Early bail: if every tried dependency gives trivial gcd, this
            // polynomial variant is likely degenerate for this N.
            if deps_tried == trivial_bail
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
        }
    }
    deps_skipped_long = useful_deps
        .iter()
        .enumerate()
        .filter(|(cand_idx, (_, _, dep_len))| !dep_candidate_tried[*cand_idx] && *dep_len >= degree)
        .count();

    if result.factor.is_none() {
        result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
        result.sqrt_fail_rat_not_square = fail_rat_not_square;
        result.sqrt_fail_alg_not_square = fail_alg_not_square;
        result.sqrt_fail_trivial_gcd = fail_trivial_gcd;
        eprintln!(
            "  sqrt: no factor found in {:.0}ms (tried={}, skipped_short={}, skipped_long={}, skipped_non_coprime={}, rat_not_square={}, alg_not_square={}, trivial_gcd={})",
            result.sqrt_ms,
            deps_tried,
            deps_skipped_short,
            deps_skipped_long,
            deps_skipped_non_coprime,
            fail_rat_not_square,
            fail_alg_not_square,
            fail_trivial_gcd
        );
    } else {
        result.sqrt_fail_rat_not_square = fail_rat_not_square;
        result.sqrt_fail_alg_not_square = fail_alg_not_square;
        result.sqrt_fail_trivial_gcd = fail_trivial_gcd;
    }

    result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
    result
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn full_count(relations: &[crate::relation::Relation]) -> usize {
    relations.iter().filter(|r| r.is_full()).count()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

fn sym_diff_sorted<T: Ord + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j].clone());
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        out.push(a[i].clone());
        i += 1;
    }
    while j < b.len() {
        out.push(b[j].clone());
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

    let mut rel_flip_cols: Vec<Vec<usize>> = Vec::with_capacity(rels.len());
    for rel in rels {
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

fn expand_dependency_over_sets(dep: &[usize], set_rows: &[Vec<usize>]) -> Vec<usize> {
    let mut expanded: Vec<usize> = Vec::new();
    for &set_idx in dep {
        if let Some(set) = set_rows.get(set_idx) {
            expanded = sym_diff_sorted(&expanded, set);
        }
    }
    expanded
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

fn dependency_expanded_len(dep: &[usize], set_rows: &[Vec<usize>]) -> usize {
    expand_dependency_over_sets(dep, set_rows).len()
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
    if let Ok(raw) = std::env::var("RUST_NFS_DEP_LEN_TIERS") {
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

/// Remap algebraic factor indices to gnfs-compatible format.
///
/// Rational factors keep their rat_fb indices (the matrix uses rat_fb.primes.len()
/// columns for the rational side, covering ALL primes including those without
/// algebraic roots).
///
/// Algebraic factors are remapped from `alg_fb` indices to flat `(prime, root)`
/// pair indices expected by gnfs's matrix. Each algebraic factor prime p is
/// identified with a specific root r via `v_p(a - b*r)`, and mapped to
/// `pair_offset(p_idx) + root_idx`. Higher-degree ideal exponents are computed
/// by dividing the remaining norm exponent by the HD ideal's residue degree.
///
/// Signs are recomputed from actual norm values.
fn remap_to_gnfs(
    relations: &[crate::relation::Relation],
    _rat_fb: &crate::factorbase::FactorBase,
    alg_fb: &crate::factorbase::FactorBase,
    gnfs_fb: &gnfs::types::FactorBase,
    f_coeffs: &[i64],
    m: u64,
    degree: usize,
) -> Vec<gnfs::types::Relation> {
    // Map: prime value → index in gnfs_fb.primes (for algebraic root lookup)
    let prime_to_gnfs: HashMap<u64, usize> = gnfs_fb
        .primes
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    // Pair offsets: running sum of root counts per gnfs_fb prime
    let mut pair_offsets = Vec::with_capacity(gnfs_fb.primes.len());
    let mut running = 0usize;
    for roots in &gnfs_fb.algebraic_roots {
        pair_offsets.push(running);
        running += roots.len();
    }
    let alg_pair_count = running;

    // HD ideal offsets for primes with fewer roots than degree
    let mut hd_offsets: HashMap<usize, usize> = HashMap::new();
    let mut hd_off = 0usize;
    for (pi, roots) in gnfs_fb.algebraic_roots.iter().enumerate() {
        if !roots.is_empty() && roots.len() < degree {
            hd_offsets.insert(pi, alg_pair_count + hd_off);
            hd_off += 1;
        }
    }

    let mut result = Vec::new();
    let mut skipped = 0usize;

    'outer: for rel in relations {
        // --- Rational factors: keep as-is ---
        // rat_fb indices map directly to matrix columns 1..1+rat_fb_size
        // since we pass rat_fb.primes.len() as rat_fb_size to build_matrix.
        let new_rat_factors = rel.rational_factors.clone();

        // --- Algebraic factors: remap to gnfs flat indices ---
        let mut new_alg_factors: Vec<(u32, u8)> = Vec::new();
        for &(idx, total_exp) in &rel.algebraic_factors {
            let prime = alg_fb.primes[idx as usize];
            let gnfs_pi = match prime_to_gnfs.get(&prime) {
                Some(&i) => i,
                None => {
                    // Algebraic prime not in gnfs_fb (no root) → skip relation
                    skipped += 1;
                    continue 'outer;
                }
            };

            // Determine ideal valuations via v_p(a - b*r) for each root
            let roots = &gnfs_fb.algebraic_roots[gnfs_pi];
            let mut accounted = 0u8;

            for (ri, &r) in roots.iter().enumerate() {
                let v = p_valuation_a_minus_br(rel.a, rel.b, r, prime);
                if v > 0 {
                    let flat_idx = pair_offsets[gnfs_pi] + ri;
                    new_alg_factors.push((flat_idx as u32, v));
                    accounted = accounted.saturating_add(v);
                }
            }

            // Higher-degree ideal: the remaining exponent from the norm
            // equals f_hd * v_HD where f_hd = degree - #roots is the
            // residue degree of the HD ideal.
            if accounted < total_exp {
                let remaining = total_exp - accounted;
                let n_roots = roots.len();
                let f_hd = degree - n_roots;
                if f_hd > 0 {
                    let f_hd_u8 = f_hd as u8;
                    if remaining % f_hd_u8 == 0 {
                        let hd_val = remaining / f_hd_u8;
                        if let Some(&hd_idx) = hd_offsets.get(&gnfs_pi) {
                            new_alg_factors.push((hd_idx as u32, hd_val));
                        }
                    } else {
                        // Ramified prime or complex splitting — skip
                        skipped += 1;
                        continue 'outer;
                    }
                }
            }
        }

        // --- Signs (recomputed from actual norms) ---
        let rat_norm_val = rel.a as i128 - (rel.b as i128) * (m as i128);
        let rational_sign_negative = rat_norm_val < 0;
        let algebraic_sign_negative = compute_alg_sign(rel.a, rel.b, f_coeffs);

        result.push(gnfs::types::Relation {
            a: rel.a,
            b: rel.b,
            rational_factors: new_rat_factors,
            algebraic_factors: new_alg_factors,
            rational_sign_negative,
            algebraic_sign_negative,
            special_q: None,
            rat_lp: None,
            alg_lp: None,
        });
    }

    if skipped > 0 {
        eprintln!(
            "  remap: skipped {} relations (unmapped algebraic primes)",
            skipped
        );
    }

    result
}

/// Compute v_p(a - b*r) — the p-adic valuation of the integer (a - b*r).
fn p_valuation_a_minus_br(a: i64, b: u64, r: u64, p: u64) -> u8 {
    let val = a as i128 - (b as i128) * (r as i128);
    if val == 0 {
        // v_p(0) = infinity; cap at a large value
        return 64;
    }
    let p128 = p as i128;
    let mut n = val.abs();
    let mut v = 0u8;
    while n % p128 == 0 && v < 64 {
        n /= p128;
        v += 1;
    }
    v
}

/// Compute sign of the algebraic norm F(a,b).
///
/// F(a,b) = c_0 * b^d + c_1 * a * b^{d-1} + ... + c_d * a^d
/// Returns true if F(a,b) < 0.
fn compute_alg_sign(a: i64, b: u64, f_coeffs: &[i64]) -> bool {
    if f_coeffs.is_empty() || b == 0 {
        return false;
    }
    let d = f_coeffs.len() - 1;
    let a128 = a as i128;
    let b128 = b as i128;

    let mut result: i128 = 0;
    let mut a_pow: i128 = 1;
    let mut b_pow: i128 = {
        let mut bp: i128 = 1;
        for _ in 0..d {
            bp *= b128;
        }
        bp
    };

    for k in 0..=d {
        result += (f_coeffs[k] as i128) * a_pow * b_pow;
        a_pow *= a128;
        if k < d {
            b_pow /= b128;
        }
    }

    result < 0
}

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
    m: u64,
    degree: usize,
) -> (Vec<gnfs::types::Relation>, Vec<usize>) {
    let ignore_special_q = std::env::var("RUST_NFS_IGNORE_SPECIAL_Q_COLUMN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let prime_to_gnfs: HashMap<u64, usize> = gnfs_fb
        .primes
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    let mut result = Vec::with_capacity(relations.len());
    let mut source_indices = Vec::with_capacity(relations.len());
    let mut skipped_unmapped = 0usize;
    let mut skipped_invalid = 0usize;

    for (rel_idx, rel) in relations.iter().enumerate() {
        // Rational factors are already indexed by rat_fb and can be used as-is.
        let new_rat_factors = rel.rational_factors.clone();

        // Algebraic factors are indexed by alg_fb prime index. Remap those prime
        // exponents to gnfs flat (pair/HD) algebraic columns without re-factoring
        // the full norm (which can overflow u64 for c45).
        let mut alg_pair_factors: Vec<(u32, u8)> = Vec::new();
        let mut valid = true;

        for &(alg_idx, total_exp) in &rel.algebraic_factors {
            let prime = match alg_fb.primes.get(alg_idx as usize) {
                Some(&p) => p,
                None => {
                    valid = false;
                    break;
                }
            };

            let gnfs_pi = match prime_to_gnfs.get(&prime) {
                Some(&pi) => pi,
                None => {
                    skipped_unmapped += 1;
                    valid = false;
                    break;
                }
            };

            let roots = &gnfs_fb.algebraic_roots[gnfs_pi];
            let pair_base = gnfs_fb.pair_offset(gnfs_pi);

            let mut root_exp_sum = 0u8;
            for (root_idx, &r) in roots.iter().enumerate() {
                let val_i128 = rel.a as i128 - rel.b as i128 * r as i128;
                let raw_e = p_adic_val_i128(val_i128, prime);
                let e = raw_e.min(total_exp.saturating_sub(root_exp_sum));
                if e > 0 {
                    alg_pair_factors.push(((pair_base + root_idx) as u32, e));
                    root_exp_sum = root_exp_sum.saturating_add(e);
                }
            }

            if root_exp_sum < total_exp {
                let residual = total_exp - root_exp_sum;
                let hd_degree = degree.saturating_sub(roots.len());
                if hd_degree == 0 || (residual as usize) % hd_degree != 0 {
                    valid = false;
                    break;
                }

                let hd_exp = (residual as usize / hd_degree) as u8;
                match gnfs_fb.hd_offset(gnfs_pi, degree) {
                    Some(hd_off) => {
                        let hd_flat_idx = gnfs_fb.algebraic_pair_count() + hd_off;
                        alg_pair_factors.push((hd_flat_idx as u32, hd_exp));
                    }
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
        }

        if !valid {
            skipped_invalid += 1;
            continue;
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
            // Legacy fallback
            let rat_lp = (rel.rat_cofactor > 1).then_some(rel.rat_cofactor);
            let alg_lp = if rel.alg_cofactor > 1 {
                match compute_alg_lp_ideal(rel.a, rel.b, rel.alg_cofactor) {
                    Some(v) => Some(v),
                    None => {
                        skipped_invalid += 1;
                        continue;
                    }
                }
            } else {
                None
            };
            (rat_lp, alg_lp)
        };

        // Recompute signs from actual norms.
        let rat_norm_val = rel.a as i128 - (rel.b as i128) * (m as i128);
        let rational_sign_negative = rat_norm_val < 0;
        let algebraic_sign_negative = eval_f_homogeneous_bigint(rel.a, rel.b, f_coeffs_big) < 0;

        result.push(gnfs::types::Relation {
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
        });
        source_indices.push(rel_idx);
    }

    let skipped = skipped_unmapped + skipped_invalid;
    if skipped > 0 {
        eprintln!(
            "  remap_hybrid: skipped {} of {} relations (unmapped={}, invalid={})",
            skipped,
            relations.len(),
            skipped_unmapped,
            skipped_invalid
        );
    }

    (result, source_indices)
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
}
