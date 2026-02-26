//! Matrix construction and eigenvector extraction for the Eisenstein CRT matrix.
//!
//! The matrix M is defined as:
//!   M[p_i][p_j] = (σ_{k−1}(p_i · p_j) mod ℓ) mod 2  ∈ {0, 1}
//! for all balanced-semiprime pairs (p_i, p_j) in the prime set.
//! M is symmetric since σ_{k−1}(N) is symmetric in its prime factors.
//!
//! Eigenvectors are extracted via deflated power iteration with
//! re-orthogonalisation — no external BLAS required.

use rand::rngs::StdRng;
use rand::Rng;

// ---------------------------------------------------------------------------
// Matrix build
// ---------------------------------------------------------------------------

/// Symmetric matrix-vector multiply: w = M · v.
#[inline]
pub fn sym_matvec(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        let mi = &m[i];
        let mut acc = 0.0f64;
        for j in 0..n {
            acc += mi[j] * v[j];
        }
        w[i] = acc;
    }
    w
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Subtract projections onto all basis vectors (Gram-Schmidt orthogonalisation).
pub fn orth_against(v: &mut [f64], basis: &[Vec<f64>]) {
    let n = v.len();
    for b in basis {
        let dot: f64 = v.iter().zip(b[..n].iter()).map(|(a, x)| a * x).sum();
        for j in 0..n {
            v[j] -= dot * b[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Top-k eigenvectors via deflated power iteration
// ---------------------------------------------------------------------------

/// Return the top-`k` (eigenvalue, eigenvector) pairs by |eigenvalue|.
///
/// Uses deflated power iteration: each new eigenvector is orthogonalised
/// against all previously found ones before and after each Matvec step.
/// This correctly extracts eigenvectors even when eigenvalues are negative.
///
/// Eigenvalues are signed (Rayleigh quotient v^T M v).
/// Returned pairs are sorted by |eigenvalue| descending.
pub fn top_k_eigenvectors(
    m: &[Vec<f64>],
    n: usize,
    k: usize,
    n_iters: usize,
    rng: &mut StdRng,
) -> Vec<(f64, Vec<f64>)> {
    if n == 0 {
        return vec![];
    }
    let k = k.min(n);
    let mut results: Vec<(f64, Vec<f64>)> = Vec::with_capacity(k);
    let mut basis: Vec<Vec<f64>>           = Vec::with_capacity(k);

    for eigidx in 0..k {
        // Random initialisation
        let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
        orth_against(&mut v, &basis);
        let vn = vec_norm(&v);
        if vn < 1e-10 {
            eprintln!(
                "Warning: eigenvector {eigidx}/{k} has near-zero norm after \
                 orthogonalisation; stopping extraction (matrix may have rank < {k})"
            );
            break;
        }
        for x in &mut v {
            *x /= vn;
        }

        for _ in 0..n_iters {
            let mut w = sym_matvec(m, &v, n);
            orth_against(&mut w, &basis);
            let wn = vec_norm(&w);
            if wn < 1e-10 {
                break;
            }
            for j in 0..n {
                v[j] = w[j] / wn;
            }
        }

        // Signed Rayleigh quotient: λ = v^T M v
        let mv  = sym_matvec(m, &v, n);
        let ray: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();

        basis.push(v.clone());
        results.push((ray, v));
    }

    results.sort_by(|a, b| {
        b.0.abs()
            .partial_cmp(&a.0.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(0xdeadbeef_cafe1234)
    }

    #[test]
    fn test_sym_matvec_identity() {
        let n = 3;
        let id: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let v = vec![1.0, 2.0, 3.0];
        let w = sym_matvec(&id, &v, n);
        for (a, b) in v.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_top_eigenvector_rank1() {
        // M = u·u^T where u = [1/√2, 1/√2, 0] → eigenvalue = 1
        let n = 3;
        let u = [0.5f64.sqrt(), 0.5f64.sqrt(), 0.0];
        let m: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| u[i] * u[j]).collect())
            .collect();

        let mut rng = make_rng();
        let pairs = top_k_eigenvectors(&m, n, 2, 200, &mut rng);
        assert!(!pairs.is_empty());
        let (lam, v) = &pairs[0];

        // Dominant eigenvalue should be ≈ 1
        assert!((*lam - 1.0).abs() < 1e-6, "lambda={lam}");
        // Eigenvector ≈ ±u
        let dot: f64 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() > 0.999, "dot with expected eigenvector = {dot}");
    }

    #[test]
    fn test_eigenvectors_orthogonal() {
        // Diagonal matrix with distinct eigenvalues
        let n    = 4;
        let diag = [4.0, 3.0, 2.0, 1.0];
        let m: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { diag[i] } else { 0.0 }).collect())
            .collect();

        let mut rng = make_rng();
        let pairs = top_k_eigenvectors(&m, n, 4, 200, &mut rng);
        assert_eq!(pairs.len(), 4);

        // All pairs of eigenvectors should be orthogonal
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dot: f64 = pairs[i]
                    .1
                    .iter()
                    .zip(pairs[j].1.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-8, "eigenvectors {i},{j} not orthogonal: dot={dot}");
            }
        }

        // Eigenvalues should match diagonal (sorted descending)
        for (k, (lam, _)) in pairs.iter().enumerate() {
            assert!(
                (*lam - diag[k]).abs() < 1e-6,
                "eigenvalue[{k}]: expected {} got {lam}",
                diag[k]
            );
        }
    }
}
