//! Block Korkin-Zolotarev (BKZ) Lattice Reduction.
//!
//! BKZ provides much stronger lattice reduction than LLL by combining LLL
//! with exact shortest vector enumeration on projected sublattices (blocks).
//! As block size increases, the quality of the basis approaches the exact
//! shortest vector problem (SVP), at the cost of exponential time in the
//! block size. This is crucial for solving factoring challenges in higher
//! dimensions where LLL leaves vectors too long.

use crate::{LatticeBasis, LllParams};

/// Configuration for BKZ reduction.
#[derive(Debug, Clone)]
pub struct BkzParams {
    /// The block size for exact SVP enumeration. Usually between 10 and 40.
    pub block_size: usize,
    /// LLL parameters used internally.
    pub lll_params: LllParams,
}

impl Default for BkzParams {
    fn default() -> Self {
        Self {
            block_size: 20,
            lll_params: LllParams::default(),
        }
    }
}

/// Perform BKZ lattice reduction on the given basis.

pub fn bkz_reduce(basis: &mut LatticeBasis, params: &BkzParams) {
    let mut big_basis = crate::basis_to_rug(basis);
    bkz_reduce_rug(&mut big_basis, params);
    *basis = crate::basis_from_rug(&big_basis);
}

pub fn bkz_reduce_rug(basis: &mut crate::BigLatticeBasis, params: &BkzParams) {
    use crate::{gram_schmidt_rug, lll_reduce_rug};

    let n = basis.len();
    if n < 2 {
        return;
    }

    let beta = params.block_size.clamp(2, n);
    lll_reduce_rug(basis, &params.lll_params);

    if beta == 2 {
        return;
    }

    let mut z = 0;
    let mut j = 0;

    while z < n - 1 {
        j += 1;
        if j == n {
            j = 1;
        }

        let k = (j + beta - 1).min(n);
        let h = (k + 1).min(n);

        let (ortho, mu) = gram_schmidt_rug(basis);

        let (found_shorter, new_vector) =
            enumerate_projected_svp_rug(basis, &ortho, &mu, j - 1, k - 1);

        if found_shorter {
            basis.insert(j - 1, new_vector);

            let mut sub_basis = basis[0..h].to_vec();
            lll_reduce_rug(&mut sub_basis, &params.lll_params);

            let mut zero_idx = None;
            for i in 0..sub_basis.len() {
                if sub_basis[i].iter().all(|x| x.clone().abs() < 1e-9) {
                    zero_idx = Some(i);
                    break;
                }
            }

            if let Some(idx) = zero_idx {
                sub_basis.remove(idx);
            } else {
                sub_basis.pop();
            }

            for i in 0..(h - 1) {
                basis[i] = sub_basis[i].clone();
            }
            basis.remove(h - 1);

            z = 0;
        } else {
            z += 1;
        }
    }
}

fn enumerate_projected_svp_rug(
    basis: &crate::BigLatticeBasis,
    ortho: &crate::BigLatticeBasis,
    mu: &[Vec<rug::Float>],
    start: usize,
    end: usize,
) -> (bool, Vec<rug::Float>) {
    use rug::Float;

    let dim = end - start + 1;
    if dim < 2 {
        return (false, vec![]);
    }

    let mut radius_sq = Float::with_val(crate::LLL_PRECISION, 0.0);
    for x in &ortho[start] {
        radius_sq += x.clone() * x;
    }
    radius_sq *= 0.999999;

    let mut ortho_norms_sq = Vec::with_capacity(dim);
    for i in start..=end {
        let mut norm_sq = Float::with_val(crate::LLL_PRECISION, 0.0);
        for x in &ortho[i] {
            norm_sq += x.clone() * x;
        }
        ortho_norms_sq.push(norm_sq);
    }

    let mut u = vec![0i64; dim];
    let mut p = vec![Float::with_val(crate::LLL_PRECISION, 0.0); dim + 1];
    let mut center = vec![Float::with_val(crate::LLL_PRECISION, 0.0); dim];
    let mut step = vec![0i64; dim];

    let mut k = dim - 1;
    u[k] = 1;
    center[k] = Float::with_val(crate::LLL_PRECISION, 0.0);
    p[k] = Float::with_val(crate::LLL_PRECISION, 0.0);
    step[k] = 1;

    let mut best_u = vec![0i64; dim];
    let mut found = false;

    loop {
        let offset = Float::with_val(crate::LLL_PRECISION, u[k]) - &center[k];
        let offset_sq = offset.clone() * offset;
        let term = offset_sq * &ortho_norms_sq[k];
        let new_norm_sq = Float::with_val(crate::LLL_PRECISION, &p[k + 1] + term);

        if new_norm_sq <= radius_sq {
            if k == 0 {
                if u.iter().any(|&x| x != 0) {
                    found = true;
                    best_u.copy_from_slice(&u);
                    break;
                }

                step[k] = if step[k] > 0 { -step[k] } else { -step[k] + 1 };
                let center_round = center[k].to_f64().round() as i64;
                u[k] = center_round + step[k];
            } else {
                k -= 1;
                p[k + 1] = new_norm_sq;

                let mut sum_mu_u = Float::with_val(crate::LLL_PRECISION, 0.0);
                for i in (k + 1)..dim {
                    let u_i = Float::with_val(crate::LLL_PRECISION, u[i]);
                    sum_mu_u += u_i * &mu[start + i][start + k];
                }
                center[k] = -sum_mu_u;

                let center_round = center[k].to_f64().round() as i64;
                u[k] = center_round;
                let c_f = center[k].to_f64();
                step[k] = if (c_f - u[k] as f64) >= 0.0 { 1 } else { -1 };
            }
        } else {
            if k == dim - 1 {
                break;
            } else {
                k += 1;
                step[k] = if step[k] > 0 { -step[k] } else { -step[k] + 1 };
                let center_round = center[k].to_f64().round() as i64;
                u[k] = center_round + step[k];
            }
        }
    }

    if found {
        let m_cols = basis[0].len();
        let mut new_vec = vec![Float::with_val(crate::LLL_PRECISION, 0.0); m_cols];
        for i in 0..dim {
            let c = Float::with_val(crate::LLL_PRECISION, best_u[i]);
            for j in 0..m_cols {
                new_vec[j] += &c * &basis[start + i][j];
            }
        }
        (true, new_vec)
    } else {
        (false, vec![])
    }
}
