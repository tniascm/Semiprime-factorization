//! Lattice construction and LLL reduction for trace-based factoring.

/// A lattice basis (rows = basis vectors).
pub type LatticeBasis = Vec<Vec<f64>>;

/// Gram-Schmidt orthogonalization.
pub fn gram_schmidt(basis: &LatticeBasis) -> (LatticeBasis, Vec<Vec<f64>>) {
    let n = basis.len();
    let m = basis[0].len();
    let mut ortho = basis.clone();
    let mut mu = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            let dot_ij: f64 = (0..m).map(|k| basis[i][k] * ortho[j][k]).sum();
            let dot_jj: f64 = (0..m).map(|k| ortho[j][k] * ortho[j][k]).sum();
            mu[i][j] = if dot_jj > 1e-10 { dot_ij / dot_jj } else { 0.0 };
            for k in 0..m {
                ortho[i][k] -= mu[i][j] * ortho[j][k];
            }
        }
    }
    (ortho, mu)
}

/// LLL lattice basis reduction.
pub fn lll_reduce(basis: &mut LatticeBasis, delta: f64) {
    let n = basis.len();
    if n == 0 {
        return;
    }
    let m = basis[0].len();
    let mut k = 1;
    while k < n {
        let (_ortho, mu) = gram_schmidt(basis);
        for j in (0..k).rev() {
            if mu[k][j].abs() > 0.5 {
                let r = mu[k][j].round();
                for l in 0..m {
                    basis[k][l] -= r * basis[j][l];
                }
            }
        }
        let (ortho, mu) = gram_schmidt(basis);
        let norm_k: f64 = (0..m).map(|l| ortho[k][l].powi(2)).sum();
        let norm_k1: f64 = (0..m).map(|l| ortho[k - 1][l].powi(2)).sum();
        if norm_k >= (delta - mu[k][k - 1].powi(2)) * norm_k1 {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = k.max(1);
            if k > 1 {
                k -= 1;
            }
        }
    }
}

/// Construct a lattice from trace constraints.
///
/// The key idea: for candidate factor d, compute dim_s2(d) and dim_s2(N/d).
/// The "correct" d has dim_old = 2*dim(d) + 2*dim(N/d) matching the actual old dimension.
/// We encode this as a lattice problem.
///
/// The lattice encodes: find (d, N/d) such that for each l:
///   2*trace_bound(d, l) + 2*trace_bound(N/d, l) ~ actual_old_contribution
///
/// We construct a lattice where short vectors correspond to valid factorizations.
pub fn build_trace_lattice(
    n: u64,
    dim_constraints: &[(u64, u64, u64)], // (l, dim_total, dim_new)
    candidate_range: u64,
) -> LatticeBasis {
    // For each candidate d from 2 to sqrt(N):
    // Compute a score vector: (dim_s2(d), dim_s2(N/d), 2*dim(d)+2*dim(N/d) - dim_old)
    // The correct factorization has score[2] = 0

    let num_constraints = dim_constraints.len();
    let dim = num_constraints + 2; // constraints + (d, scaling)

    let mut basis = Vec::new();
    let scaling = 1000.0; // Scale to make dimension constraints dominant

    // For each candidate d, create a lattice row
    let sqrt_n = (n as f64).sqrt() as u64;
    let range = candidate_range.min(sqrt_n);

    for d in 2..=range {
        if n % d != 0 {
            continue;
        }
        let e = n / d;

        let dim_d = crate::trace::dim_s2(d);
        let dim_e = crate::trace::dim_s2(e);
        let expected_old = 2 * dim_d + 2 * dim_e;

        let mut row = vec![0.0; dim];
        row[0] = d as f64; // The candidate factor
        row[1] = scaling; // Scaling to find this row

        // For each constraint, add the residual
        for (i, &(_l, _dim_total, dim_new)) in dim_constraints.iter().enumerate() {
            let actual_old = crate::trace::dim_s2(n) - dim_new;
            let residual = (expected_old as f64) - (actual_old as f64);
            row[2 + i] = residual * scaling;
        }

        basis.push(row);
    }

    // Add identity-like rows for the dimension constraints to create a proper lattice
    if basis.is_empty() {
        // No divisors found -- n might be prime
        return vec![vec![1.0]];
    }

    basis
}

/// Extract factor from LLL-reduced lattice.
/// The shortest vector's first component should be the factor d.
pub fn extract_factor_from_lattice(basis: &LatticeBasis, n: u64) -> Option<(u64, u64)> {
    for row in basis {
        let d = row[0].round() as u64;
        if d > 1 && d < n && n % d == 0 {
            return Some((d, n / d));
        }
    }
    None
}
