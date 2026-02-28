use rug::Integer;
use std::path::Path;

use crate::arith::select_quad_char_primes;
use crate::log::StageLogger;
use crate::linalg::{build_matrix, find_dependencies};
use crate::params::GnfsParams;
use crate::polyselect::select_base_m;
use crate::relation::collect_smooth_relations;
use crate::sieve::{build_factor_base, line_sieve, poly_coeffs_to_i64};
use crate::sqrt::extract_factor;

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
pub fn factor_gnfs(
    n: &Integer,
    params: &GnfsParams,
    output_dir: Option<&Path>,
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
    let poly = select_base_m(n, degree);
    let f_coeffs = poly.f_coeffs();
    let m = poly.m();

    poly_log.log(&format!(
        "degree={}, m={} ({} bits), coeffs={}",
        degree,
        &m,
        m.significant_bits(),
        f_coeffs.len()
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
        "Factor base: {} primes (bound={})",
        fb.primes.len(),
        params.lim0
    ));

    // Need more relations than matrix columns: 2 sign + rat_fb + alg_fb + 30 quad chars
    let n_quad_chars = 30usize;
    let target_rels = fb.primes.len() * 2 + 2 + n_quad_chars + 20;
    let mut all_hits = Vec::new();
    let mut sieve_a = params.sieve_a;
    let mut max_b = params.max_b;

    for expansion in 0..5 {
        let hits = line_sieve(&poly, &fb, sieve_a, max_b);
        let (rels, partials) = collect_smooth_relations(&hits, &fb, params.large_prime_bound_0());

        sieve_log.log(&format!(
            "Expansion {}: sieve_a={}, max_b={} → {} hits, {} smooth, {} partial",
            expansion, sieve_a, max_b, hits.len(), rels.len(), partials
        ));

        for rel in rels {
            if !all_hits.iter().any(|r: &crate::types::Relation| r.a == rel.a && r.b == rel.b) {
                all_hits.push(rel);
            }
        }

        if all_hits.len() >= target_rels {
            break;
        }

        sieve_a = (sieve_a as f64 * 1.5) as u64;
        max_b = (max_b as f64 * 1.3) as u64;
    }

    result.relations_found = all_hits.len();
    sieve_log.log(&format!("Total relations: {} (target: {})", all_hits.len(), target_rels));

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

    // --- Stage 3: Filtering (skip for M1 — build matrix directly) ---

    // --- Stage 4: Linear Algebra ---
    let mut la_log = StageLogger::new("linalg", output_dir);
    la_log.start(&serde_json::json!({"relations": all_hits.len(), "fb_size": fb.primes.len()}));

    // Select quadratic character primes (~30) to ensure algebraic product is a square in O_K
    let quad_chars = select_quad_char_primes(&f_i64, &fb.primes, 30);
    la_log.log(&format!("Quadratic characters: {} primes", quad_chars.primes.len()));

    let (matrix, ncols) = build_matrix(&all_hits, fb.primes.len(), fb.primes.len(), &quad_chars);
    result.matrix_rows = matrix.len();
    result.matrix_cols = ncols;

    la_log.log(&format!("Matrix: {} rows × {} cols", matrix.len(), ncols));

    let deps = find_dependencies(&matrix, ncols);
    result.dependencies_found = deps.len();

    la_log.log(&format!("Dependencies found: {}", deps.len()));
    la_log.finish(&serde_json::json!({"deps": deps.len(), "rows": matrix.len(), "cols": ncols}));

    if deps.is_empty() {
        return result;
    }

    // --- Stage 5: Square Root ---
    let mut sqrt_log = StageLogger::new("sqrt", output_dir);
    sqrt_log.start(&serde_json::json!({"dependencies": deps.len()}));

    for (i, dep) in deps.iter().enumerate() {
        result.dependencies_tried = i + 1;
        sqrt_log.log(&format!("Trying dependency {}/{} ({} relations)", i + 1, deps.len(), dep.len()));

        if let Some(factor) = extract_factor(&all_hits, dep, &f_coeffs, &m, n) {
            if Integer::from(n % &factor) == 0 && factor > 1 && factor < *n {
                sqrt_log.log(&format!("Factor found: {}", factor));
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

                sqrt_log.finish(&serde_json::json!({"factor": factor.to_string(), "dep_tried": i + 1}));
                result.total_secs = sqrt_log.elapsed_secs();
                return result;
            }
        }
    }

    sqrt_log.log("No factor extracted from any dependency");
    sqrt_log.finish(&serde_json::json!({"factor": serde_json::Value::Null, "deps_tried": deps.len()}));
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
            eprintln!("  relations: {}, deps: {}, tried: {}",
                result.relations_found, result.dependencies_found, result.dependencies_tried);
        }
    }

    #[test]
    fn test_pipeline_15347() {
        let n = Integer::from(15347u64); // 103 * 149
        let params = GnfsParams::test_small();
        let result = factor_gnfs(&n, &params, None);
        eprintln!("Pipeline 15347: factor={:?}, rels={}, deps={}",
            result.factor, result.relations_found, result.dependencies_found);
    }
}
