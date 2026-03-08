use rug::Integer;
use std::path::Path;

use crate::arith::select_quad_char_primes;
use crate::linalg::{
    build_matrix, find_dependencies, find_dependencies_with_preelim,
    find_dependencies_with_preelim_bw, randomize_dependencies,
};
use crate::log::StageLogger;
use crate::params::{GnfsParams, SieveMode};
use crate::polyselect::select_base_m_variant;
use crate::relation::collect_smooth_relations;
use crate::sieve::{build_factor_base, lattice_sieve, line_sieve, poly_coeffs_to_i64};
use crate::sqrt::{extract_factor_diagnostic, extract_factor_verbose, FactorFailure};

/// Result of the GNFS pipeline.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GnfsResult {
    pub factor: Option<String>,
    pub n: String,
    pub total_secs: f64,
    pub relations_found: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub dependencies_found: usize,
    pub dependencies_tried: usize,
}

/// Run the full GNFS pipeline to factor N.
///
/// Tries the primary polynomial degree first. If all dependencies produce
/// trivial gcd (indicating a degenerate polynomial/number field), automatically
/// retries with alternate degrees.
pub fn factor_gnfs(n: &Integer, params: &GnfsParams, output_dir: Option<&Path>) -> GnfsResult {
    // Strategy: try polynomial variants (different m at same degree) first,
    // then switch degree as a last resort.
    // Different m values produce different number fields with independent
    // unit structures, breaking the systematic trivial-gcd problem.
    let max_variants = 5;

    let mut last_result = None;
    for variant in 0..max_variants {
        let label = if variant == 0 {
            String::new()
        } else {
            format!(" (poly variant #{}, m_offset=-{})", variant, variant)
        };

        let result = factor_gnfs_inner(n, params, output_dir, params.degree, variant, &label);
        if result.factor.is_some() {
            return result;
        }
        last_result = Some(result);
    }

    last_result.unwrap()
}

/// Inner GNFS pipeline with a specific polynomial variant.
fn factor_gnfs_inner(
    n: &Integer,
    params: &GnfsParams,
    output_dir: Option<&Path>,
    _degree: u32,
    variant: u32,
    label: &str,
) -> GnfsResult {
    let mut result = GnfsResult {
        factor: None,
        n: n.to_string(),
        total_secs: 0.0,
        relations_found: 0,
        matrix_rows: 0,
        matrix_cols: 0,
        dependencies_found: 0,
        dependencies_tried: 0,
    };

    // --- Stage 1: Polynomial Selection ---
    let mut poly_log = StageLogger::new("polyselect", output_dir);
    poly_log.start(&params);

    let degree = params.degree;
    let poly = select_base_m_variant(n, degree, variant);
    let f_coeffs = poly.f_coeffs();
    let m = poly.m();

    poly_log.log(&format!(
        "degree={}, m={} ({} bits), coeffs={}{}",
        degree,
        &m,
        m.significant_bits(),
        f_coeffs.len(),
        label
    ));

    if let Some(dir) = output_dir {
        let poly_json = serde_json::to_string_pretty(&poly).unwrap();
        let stage_dir = dir.join("stage_polyselect");
        std::fs::create_dir_all(&stage_dir).ok();
        std::fs::write(stage_dir.join("poly.json"), poly_json).ok();
    }

    poly_log.finish(&serde_json::json!({"degree": degree, "m_bits": m.significant_bits()}));

    // --- Stage 2: Sieving ---
    let mut sieve_log = StageLogger::new("sieve", output_dir);
    sieve_log.start(&params);

    let f_i64 = match poly_coeffs_to_i64(&f_coeffs) {
        Some(v) => v,
        None => {
            sieve_log.log("ERROR: polynomial coefficients too large for i64 sieve");
            return result;
        }
    };

    let fb = build_factor_base(&f_i64, params.lim0.max(params.lim1));
    sieve_log.log(&format!(
        "Factor base: {} primes, {} (prime,root) pairs (bound={})",
        fb.primes.len(),
        fb.algebraic_pair_count(),
        params.lim0
    ));

    // Need significantly MORE relations than matrix columns for useful dependencies.
    // Dependencies of size < degree are provably trivial (no reduction mod f),
    // and short dependencies are mostly trivial. Targeting 3x excess ensures
    // the null space has enough dimension for long, useful dependencies.
    let n_quad_chars = 64usize;
    let alg_pairs = fb.algebraic_pair_count();
    let alg_hd = fb.higher_degree_ideal_count(degree as usize);
    let ncols_est = fb.primes.len() + alg_pairs + alg_hd + 2 + n_quad_chars;
    // Excess needed: small matrices need high excess (short deps → trivial gcd),
    // larger matrices produce long deps naturally. Randomized dep XOR helps even at
    // moderate excess. Conservative values to avoid excessive sieving.
    let excess_frac = if ncols_est < 200 {
        3.0
    } else if ncols_est < 500 {
        2.0
    } else if ncols_est < 2000 {
        1.0
    } else {
        0.5
    };
    let target_rels = ncols_est + ((ncols_est as f64 * excess_frac) as usize).max(50);
    let mut all_hits = Vec::new();

    if params.sieve_mode == SieveMode::Lattice {
        // --- Lattice sieve with special-q ---
        // Use 1LP for adequate yield. Sparse LP/SQ columns are handled by
        // column reordering (sparse first) + aggressive randomization of
        // GE basis vectors with high k to produce long deps.
        let lpb = params.large_prime_bound_0();
        let qmax = params.qmin + 10_000; // initial q range
        sieve_log.log(&format!(
            "Lattice sieve: log_i={}, qmin={}, qmax={}, lpb={}",
            params.log_i, params.qmin, qmax, lpb
        ));

        let mut qmin = params.qmin;
        let mut qmax_curr = qmax;
        for expansion in 0..10 {
            let hits = lattice_sieve(&poly, &fb, params.log_i, qmin, qmax_curr);
            let (rels, partials) = collect_smooth_relations(&hits, &fb, lpb, degree as usize);

            sieve_log.log(&format!(
                "Expansion {}: q=[{}, {}] → {} hits, {} smooth, {} partial",
                expansion,
                qmin,
                qmax_curr,
                hits.len(),
                rels.len(),
                partials
            ));

            for rel in rels {
                if !all_hits
                    .iter()
                    .any(|r: &crate::types::Relation| r.a == rel.a && r.b == rel.b)
                {
                    all_hits.push(rel);
                }
            }

            if all_hits.len() >= target_rels {
                break;
            }

            // Expand q range for next iteration
            qmin = qmax_curr;
            qmax_curr += 10_000;
        }
    } else {
        // --- Line sieve (original) ---
        // No 1LP for line sieve: fully smooth only. 1LP creates huge sparse
        // matrices that make GE and sqrt slow for small numbers.
        let lpb = 0u64; // remainder must be 1 (fully smooth)
        let mut sieve_a = params.sieve_a;
        let mut max_b = params.max_b;

        for expansion in 0..10 {
            let hits = line_sieve(&poly, &fb, sieve_a, max_b);
            let (rels, partials) = collect_smooth_relations(&hits, &fb, lpb, degree as usize);

            sieve_log.log(&format!(
                "Expansion {}: sieve_a={}, max_b={} → {} hits, {} smooth, {} partial",
                expansion,
                sieve_a,
                max_b,
                hits.len(),
                rels.len(),
                partials
            ));

            for rel in rels {
                if !all_hits
                    .iter()
                    .any(|r: &crate::types::Relation| r.a == rel.a && r.b == rel.b)
                {
                    all_hits.push(rel);
                }
            }

            if all_hits.len() >= target_rels {
                break;
            }

            sieve_a = (sieve_a as f64 * 1.5) as u64;
            max_b = (max_b as f64 * 1.3) as u64;
        }
    }

    result.relations_found = all_hits.len();
    sieve_log.log(&format!(
        "Total relations: {} (target: {})",
        all_hits.len(),
        target_rels
    ));

    if let Some(dir) = output_dir {
        sieve_log.checkpoint(
            &serde_json::json!({"relations": all_hits.len(), "target": target_rels}),
            dir,
            "checkpoint.json",
        );
    }

    sieve_log.finish(&serde_json::json!({"relations": all_hits.len()}));

    if all_hits.len() < 2 {
        return result;
    }

    // --- Stage 3: Filtering ---
    let mut filter_log = StageLogger::new("filter", output_dir);
    filter_log.start(&serde_json::json!({"relations": all_hits.len()}));

    // Singleton filtering: remove relations whose large primes appear only once.
    // A singleton LP can never cancel in a dependency (it's the only source of
    // that column's bit), so including it wastes matrix space.
    let pre_filter = all_hits.len();
    {
        use std::collections::HashMap;
        loop {
            // Count occurrences of each rational LP
            let mut rat_lp_count: HashMap<u64, usize> = HashMap::new();
            for rel in &all_hits {
                if let Some(lp) = rel.rat_lp {
                    *rat_lp_count.entry(lp).or_insert(0) += 1;
                }
            }
            // Count occurrences of each algebraic LP
            let mut alg_lp_count: HashMap<(u64, u64), usize> = HashMap::new();
            for rel in &all_hits {
                if let Some(lp) = rel.alg_lp {
                    *alg_lp_count.entry(lp).or_insert(0) += 1;
                }
            }
            // Count occurrences of each special-q
            let mut sq_count: HashMap<(u64, u64), usize> = HashMap::new();
            for rel in &all_hits {
                if let Some(sq) = rel.special_q {
                    *sq_count.entry(sq).or_insert(0) += 1;
                }
            }

            let before = all_hits.len();
            all_hits.retain(|rel| {
                if let Some(lp) = rel.rat_lp {
                    if rat_lp_count.get(&lp).copied().unwrap_or(0) < 2 {
                        return false;
                    }
                }
                if let Some(lp) = rel.alg_lp {
                    if alg_lp_count.get(&lp).copied().unwrap_or(0) < 2 {
                        return false;
                    }
                }
                if let Some(sq) = rel.special_q {
                    if sq_count.get(&sq).copied().unwrap_or(0) < 2 {
                        return false;
                    }
                }
                true
            });
            if all_hits.len() == before {
                break; // No more singletons
            }
        }
    }
    filter_log.log(&format!(
        "Singleton filtering: {} → {} relations",
        pre_filter,
        all_hits.len()
    ));

    // Cap relations to prevent O(n²) memory explosion in dense GE history matrix.
    let max_relations = ncols_est.saturating_mul(5).max(ncols_est + 2000);
    if all_hits.len() > max_relations {
        // Keep relations with smallest |a| — these produce smaller algebraic
        // products, making Hensel lifting faster and more stable.
        all_hits.sort_by_key(|r| r.a.unsigned_abs());
        all_hits.truncate(max_relations);
    }
    filter_log.finish(&serde_json::json!({"relations_after": all_hits.len()}));

    // --- Stage 4: Linear Algebra ---
    let mut la_log = StageLogger::new("linalg", output_dir);
    la_log.start(&serde_json::json!({"relations": all_hits.len(), "fb_size": fb.primes.len()}));

    // Select quadratic character primes to ensure algebraic product is a square in O_K
    let quad_chars = select_quad_char_primes(&f_i64, &fb.primes, n_quad_chars);
    la_log.log(&format!(
        "Quadratic characters: {} primes",
        quad_chars.primes.len()
    ));

    la_log.log(&format!(
        "HD ideal primes: {} (columns for degree-2+ ideals)",
        alg_hd
    ));
    let (matrix, ncols) = build_matrix(&all_hits, fb.primes.len(), alg_pairs, alg_hd, &quad_chars);
    result.matrix_rows = matrix.len();
    result.matrix_cols = ncols;

    la_log.log(&format!("Matrix: {} rows × {} cols", matrix.len(), ncols));

    let bw_threshold: usize = std::env::var("GNFS_BW_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let ge_deps = if matrix.len() > bw_threshold {
        find_dependencies_with_preelim_bw(&matrix, ncols)
    } else if matrix.len() > 5_000 {
        find_dependencies_with_preelim(&matrix, ncols)
    } else {
        find_dependencies(&matrix, ncols)
    };

    // Randomize: combine GE basis vectors to produce decorrelated dependencies.
    // GE basis vectors are short and correlated → trivial gcd. Random XOR
    // combinations produce longer deps with ~50% nontrivial-factor probability.
    // Very high k ensures deps involve 100+ original relations for the algebraic
    // sqrt to produce sufficiently random values mod p vs mod q.
    let n_random = ge_deps.len().min(10000);
    let k = (ge_deps.len() / 3).max(50).min(500);
    let deps = randomize_dependencies(&ge_deps, n_random, k, 42);
    result.dependencies_found = deps.len();

    la_log.log(&format!(
        "Dependencies: {} GE basis + {} random combinations = {} total",
        ge_deps.len(),
        deps.len() - ge_deps.len(),
        deps.len()
    ));
    la_log.finish(&serde_json::json!({"ge_deps": ge_deps.len(), "random_deps": deps.len() - ge_deps.len(), "total_deps": deps.len()}));

    if deps.is_empty() {
        return result;
    }

    // --- Stage 5: Square Root ---
    let mut sqrt_log = StageLogger::new("sqrt", output_dir);
    sqrt_log.start(&serde_json::json!({"dependencies": deps.len()}));

    // Filter: dependencies smaller than polynomial degree are provably trivial.
    // When the product of k < d linear terms never reduces modulo f(α),
    // P(m) = ∏(aᵢ - bᵢm) exactly, so γ(m) = ±√R = ±x always.
    let min_dep_size = degree as usize;
    let mut useful_deps: Vec<&Vec<usize>> =
        deps.iter().filter(|d| d.len() >= min_dep_size).collect();
    // Sort by size descending: longer deps (random combinations) tried first,
    // as they have higher probability of giving nontrivial factors.
    useful_deps.sort_by(|a, b| b.len().cmp(&a.len()));
    sqrt_log.log(&format!(
        "Filtered: {} → {} useful dependencies (min size {}, largest {})",
        deps.len(),
        useful_deps.len(),
        min_dep_size,
        useful_deps.first().map_or(0, |d| d.len())
    ));

    // Cap dependencies tried per variant. Each sqrt extraction costs O(dep_size × degree²)
    // for the Couveignes Hensel lifting. For small numbers with 100K+ relations, there
    // can be 100K+ deps. Trying all is wasteful — if 5000 deps don't produce a factor,
    // the polynomial is likely problematic.
    let max_deps_to_try = 5000;
    let useful_deps_capped = if useful_deps.len() > max_deps_to_try {
        &useful_deps[..max_deps_to_try]
    } else {
        &useful_deps[..]
    };
    sqrt_log.log(&format!(
        "Will try up to {} dependencies (of {} useful)",
        useful_deps_capped.len(),
        useful_deps.len()
    ));

    let mut fail_rat = 0usize;
    let mut fail_alg = 0usize;
    let mut fail_gcd = 0usize;

    for (i, dep) in useful_deps_capped.iter().enumerate() {
        result.dependencies_tried = i + 1;
        sqrt_log.log(&format!(
            "Trying dependency {}/{} ({} relations)",
            i + 1,
            useful_deps_capped.len(),
            dep.len()
        ));

        let (factor_opt, failure) = if i < 10 {
            extract_factor_verbose(&all_hits, dep, &f_coeffs, &m, n)
        } else {
            extract_factor_diagnostic(&all_hits, dep, &f_coeffs, &m, n)
        };
        if let Some(factor) = factor_opt {
            if Integer::from(n % &factor) == 0 && factor > 1 && factor < *n {
                sqrt_log.log(&format!("Factor found: {}", factor));
                sqrt_log.log(&format!(
                    "Failures up to this point: rat_not_sq={}, alg_not_sq={}, trivial_gcd={}",
                    fail_rat, fail_alg, fail_gcd
                ));
                result.factor = Some(factor.to_string());

                if let Some(dir) = output_dir {
                    let cofactor = Integer::from(n / &factor);
                    sqrt_log.checkpoint(
                        &serde_json::json!({
                            "factor": factor.to_string(),
                            "cofactor": cofactor.to_string(),
                            "n": n.to_string(),
                            "dependency_index": i,
                        }),
                        dir,
                        "factors.json",
                    );
                }

                sqrt_log
                    .finish(&serde_json::json!({"factor": factor.to_string(), "dep_tried": i + 1}));
                result.total_secs = sqrt_log.elapsed_secs();
                return result;
            }
        }
        match failure {
            Some(FactorFailure::RationalNotSquare) => fail_rat += 1,
            Some(FactorFailure::AlgebraicNotSquare) => fail_alg += 1,
            Some(FactorFailure::TrivialGcd) => fail_gcd += 1,
            None => {}
        }

        // Early bail-out: if we've tried many deps with ALL giving trivial gcd
        // (no algebraic sqrt failures), the polynomial/relation-set is likely
        // degenerate. Move to the next polynomial variant quickly.
        let early_bail_threshold = 200;
        if i + 1 == early_bail_threshold
            && fail_gcd == early_bail_threshold
            && fail_rat == 0
            && fail_alg == 0
        {
            sqrt_log.log(&format!(
                "Early bail: all {} deps gave trivial gcd — polynomial likely degenerate",
                early_bail_threshold
            ));
            break;
        }
    }

    sqrt_log.log(&format!(
        "No factor extracted — rat_not_sq={}, alg_not_sq={}, trivial_gcd={}",
        fail_rat, fail_alg, fail_gcd
    ));
    sqrt_log.finish(&serde_json::json!({"factor": serde_json::Value::Null, "deps_tried": result.dependencies_tried}));
    result.total_secs = sqrt_log.elapsed_secs();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::GnfsParams;

    #[test]
    fn test_pipeline_small() {
        let n = Integer::from(8051u64); // 83 * 97
        let params = GnfsParams::test_small();
        let result = factor_gnfs(&n, &params, None);
        if let Some(ref f) = result.factor {
            let factor: Integer = f.parse().unwrap();
            assert!(Integer::from(&n % &factor) == 0);
            assert!(factor > 1);
            eprintln!("Factored 8051: {}", f);
        } else {
            eprintln!("Pipeline did not factor 8051 (may need parameter tuning)");
            eprintln!(
                "  relations: {}, deps: {}, tried: {}",
                result.relations_found, result.dependencies_found, result.dependencies_tried
            );
        }
    }

    #[test]
    fn test_pipeline_15347() {
        let n = Integer::from(15347u64); // 103 * 149
        let params = GnfsParams::test_small();
        let result = factor_gnfs(&n, &params, None);
        eprintln!(
            "Pipeline 15347: factor={:?}, rels={}, deps={}",
            result.factor, result.relations_found, result.dependencies_found
        );
    }
}

#[cfg(test)]
mod tests_5b {
    use super::*;

    #[test]
    fn test_pipeline_5b() {
        // 70001 * 71429 — balanced 33-bit semiprime
        let n = Integer::from(5000101429u64);
        let params = GnfsParams::c20();
        let result = factor_gnfs(&n, &params, None);
        eprintln!(
            "Pipeline 5000101429: factor={:?}, rels={}, deps={}, tried={}",
            result.factor,
            result.relations_found,
            result.dependencies_found,
            result.dependencies_tried
        );
        if let Some(ref f) = result.factor {
            let fint: Integer = f.parse().unwrap();
            assert!(Integer::from(&n % &fint) == 0);
            eprintln!("SUCCESS: {} = {} × {}", n, f, Integer::from(&n / &fint));
        }
    }
}
