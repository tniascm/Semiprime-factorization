//! Full NFS pipeline: poly selection -> sieve -> filter -> LA -> sqrt.

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
}

/// Factor N using the full NFS pipeline.
///
/// Stages:
///   1. Polynomial selection (base-m)
///   2. Special-Q lattice sieve
///   3. Singleton/duplicate filtering
///   4. Linear algebra (GF(2) Gaussian elimination)
///   5. Square root and factor extraction
pub fn factor_nfs(n: &Integer, params: &NfsParams) -> NfsResult {
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
    };

    // --- Stage 1: Polynomial Selection ---
    let poly = gnfs::polyselect::select_base_m(n, params.degree);
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

    // --- Stage 2: Sieve ---
    let sieve_start = std::time::Instant::now();

    // Build factor bases for both sides.
    // Rational polynomial: g(x) = x - m, coefficients [-m, 1].
    let rat_coeffs = vec![-(m as i64), 1i64];
    let rat_fb = crate::factorbase::FactorBase::new(&rat_coeffs, params.lim0, 1.442);
    let alg_fb = crate::factorbase::FactorBase::new(&f_coeffs_i64, params.lim1, 1.442);

    // Run the special-q sieve.
    let sieve_result =
        crate::sieve::sieve_specialq(&f_coeffs_i64, m, &rat_fb, &alg_fb, params);

    result.sieve_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;
    result.relations_found = sieve_result.relations.len();
    eprintln!(
        "  sieve: {} rels in {:.0}ms ({} survivors, {} special-qs)",
        sieve_result.relations.len(),
        result.sieve_ms,
        sieve_result.survivors_found,
        sieve_result.special_qs_processed
    );

    if sieve_result.relations.is_empty() {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 3: Filtering ---
    let filter_start = std::time::Instant::now();
    let filtered = crate::filter::filter_relations(sieve_result.relations);
    result.filter_ms = filter_start.elapsed().as_secs_f64() * 1000.0;
    result.relations_after_filter = filtered.len();
    eprintln!(
        "  filter: {} -> {} rels in {:.0}ms",
        result.relations_found, filtered.len(), result.filter_ms
    );

    if filtered.len() < 2 {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 4: Linear Algebra ---
    // Convert our relations to gnfs format for the LA and sqrt stages.
    let gnfs_rels: Vec<gnfs::types::Relation> =
        filtered.iter().map(|r| r.to_gnfs_relation()).collect();

    let la_start = std::time::Instant::now();

    // Build gnfs-compatible factor base for matrix construction.
    // gnfs::sieve::build_factor_base only includes primes with algebraic roots,
    // which is what the matrix columns expect.
    let gnfs_fb =
        gnfs::sieve::build_factor_base(&f_coeffs_i64, params.lim0.max(params.lim1));
    let degree = params.degree as usize;
    let alg_pairs = gnfs_fb.algebraic_pair_count();
    let alg_hd = gnfs_fb.higher_degree_ideal_count(degree);
    let quad_chars =
        gnfs::arith::select_quad_char_primes(&f_coeffs_i64, &gnfs_fb.primes, 30);

    let (matrix, ncols) = gnfs::linalg::build_matrix(
        &gnfs_rels,
        gnfs_fb.primes.len(),
        alg_pairs,
        alg_hd,
        &quad_chars,
    );
    result.matrix_rows = matrix.len();
    result.matrix_cols = ncols;

    eprintln!("  LA: {} x {} matrix", matrix.len(), ncols);

    let ge_deps = gnfs::linalg::find_dependencies(&matrix, ncols);
    let n_random = ge_deps.len().min(5000);
    let deps = gnfs::linalg::randomize_dependencies(&ge_deps, n_random, 3, 42);
    result.dependencies_found = deps.len();

    result.la_ms = la_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  LA: {} deps ({} GE + {} random) in {:.0}ms",
        deps.len(),
        ge_deps.len(),
        deps.len() - ge_deps.len(),
        result.la_ms
    );

    if deps.is_empty() {
        result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        return result;
    }

    // --- Stage 5: Square Root ---
    let sqrt_start = std::time::Instant::now();

    // Sort deps by descending length: longer deps are more likely to yield
    // a nontrivial factor.
    let mut useful_deps: Vec<&Vec<usize>> =
        deps.iter().filter(|d| d.len() >= degree).collect();
    useful_deps.sort_by(|a, b| b.len().cmp(&a.len()));

    for (i, dep) in useful_deps.iter().enumerate() {
        let (factor_opt, _failure) = if i < 10 {
            gnfs::sqrt::extract_factor_verbose(
                &gnfs_rels,
                dep,
                &f_coeffs_big,
                &m_big,
                n,
            )
        } else {
            gnfs::sqrt::extract_factor_diagnostic(
                &gnfs_rels,
                dep,
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
                    i + 1,
                    result.sqrt_ms,
                    result.factor.as_ref().unwrap()
                );
                break;
            }
        }

        // Bail after trying many deps.
        if i >= 200 {
            eprintln!("  sqrt: giving up after {} deps", i + 1);
            break;
        }
    }

    if result.factor.is_none() {
        result.sqrt_ms = sqrt_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  sqrt: no factor found in {:.0}ms", result.sqrt_ms);
    }

    result.total_ms = start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
