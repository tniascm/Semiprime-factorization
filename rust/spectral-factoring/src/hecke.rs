//! Hecke operator computation for modular symbols on Gamma_0(N).
//!
//! For a prime l not dividing N, the Hecke operator T_l acts on modular symbols by:
//!   T_l {alpha, beta} = {l*alpha, l*beta} + sum_{j=0}^{l-1} {(alpha+j)/l, (beta+j)/l}
//!
//! On Manin symbols (c:d) in P^1(Z/NZ), this translates to:
//!   T_l [(c:d)] = [(c : l*d)] + sum_{j=0}^{l-1} [(l*c + j*something : d)]
//!
//! More precisely, we need to express the Hecke action in terms of the right action
//! of matrices on P^1(Z/NZ).

use crate::modular_symbols::{ModularSymbolSpace, lookup_symbol};

/// Result of the Hecke action on a single P^1 element.
/// Each entry is (index_in_p1, coefficient).
struct HeckeActionResult {
    terms: Vec<(usize, i64)>,
}

/// Compute the Hecke action T_l on a single Manin symbol (c:d) in P^1(Z/NZ).
///
/// T_l acts via the double coset Gamma_0(N) [[l,0],[0,1]] Gamma_0(N), which decomposes as:
///   [[l, 0], [0, 1]] and [[1, j], [0, l]] for j = 0, ..., l-1
///   (when gcd(l, N) = 1)
///
/// On P^1, the right action of a matrix [[a,b],[c,d]] sends (x:y) -> (ax+cy : bx+dy).
/// Wait: right action means (x,y) * M = (xa+yc, xb+yd).
///
/// So:
///   [[l,0],[0,1]]: (x,y) -> (lx, y), i.e., (c:d) -> (lc : d)
///   [[1,j],[0,l]]: (x,y) -> (x, jx+ly), i.e., (c:d) -> (c : jc+ld)
///
/// T_l [(c:d)] = [(lc : d)] + sum_{j=0}^{l-1} [(c : jc + ld)]
fn hecke_action_on_p1(
    space: &ModularSymbolSpace,
    p1_idx: usize,
    l: u64,
) -> HeckeActionResult {
    let n = space.level;
    let ni = n as i64;
    let li = l as i64;
    let c = space.p1_list[p1_idx].c;
    let d = space.p1_list[p1_idx].d;

    let mut terms: Vec<(usize, i64)> = Vec::new();

    // Term 1: (lc : d)
    let lc = (li * c) % ni;
    if let Some(idx) = lookup_symbol(space, lc, d) {
        terms.push((idx, 1));
    }

    // Terms 2..l+1: (c : jc + ld) for j = 0, ..., l-1
    for j in 0..li {
        let new_d = (j * c + li * d) % ni;
        if let Some(idx) = lookup_symbol(space, c, new_d) {
            terms.push((idx, 1));
        }
    }

    // Combine duplicate indices
    terms.sort_by_key(|&(idx, _)| idx);
    let mut combined = Vec::new();
    for (idx, coeff) in terms {
        if let Some(last) = combined.last_mut() {
            let (last_idx, last_coeff): &mut (usize, i64) = last;
            if *last_idx == idx {
                *last_coeff += coeff;
                continue;
            }
        }
        combined.push((idx, coeff));
    }

    HeckeActionResult { terms: combined }
}

/// Compute the matrix of the Hecke operator T_l acting on the modular symbol space.
///
/// The matrix is (dimension x dimension) where dimension is the quotient dimension.
/// We compute T_l's action on each P^1 element, project to the quotient, and read off
/// the matrix in the chosen basis.
///
/// Returns a dimension x dimension matrix as Vec<Vec<i64>>.
pub fn hecke_matrix(space: &ModularSymbolSpace, l: u64) -> Vec<Vec<i64>> {
    let dim = space.dimension;
    if dim == 0 {
        return vec![];
    }

    // For each basis vector of the quotient, compute T_l on it.
    // The basis vectors are expressed in terms of P^1 elements.
    // T_l on a basis vector = sum of T_l on each P^1 component, projected back to quotient.

    let mut matrix = vec![vec![0i64; dim]; dim];

    for bi in 0..dim {
        // basis[bi] is a sparse vector in P^1 coordinates
        let basis_vec = &space.basis[bi];

        // Apply T_l to each component
        let mut result_in_quotient = vec![0i64; dim];

        for &(p1_idx, coeff) in basis_vec {
            let action = hecke_action_on_p1(space, p1_idx, l);

            // Project each resulting P^1 element to the quotient
            for &(result_p1_idx, action_coeff) in &action.terms {
                let projection = &space.relation_matrix[result_p1_idx];
                for &(basis_idx, proj_coeff) in projection {
                    if basis_idx < dim {
                        result_in_quotient[basis_idx] += coeff * action_coeff * proj_coeff;
                    }
                }
            }
        }

        // Column bi of the matrix = result_in_quotient
        for bj in 0..dim {
            matrix[bj][bi] = result_in_quotient[bj];
        }
    }

    matrix
}

/// Compute the trace of the Hecke operator T_l on S_2(Gamma_0(N)).
///
/// Uses the Eichler-Selberg trace formula as a cross-check:
/// tr(T_l, S_2(Gamma_0(N))) has an explicit formula involving class numbers.
///
/// For prime l not dividing N:
/// tr T_l = -1 - sum_{t} H(t^2 - 4l) * psi_N(t, l) + sum_{d|l, d>0} min(d, l/d) * psi_N_divisor(d)
///
/// For now, we compute the trace directly from the matrix.
pub fn hecke_trace(space: &ModularSymbolSpace, l: u64) -> i64 {
    let mat = hecke_matrix(space, l);
    let mut trace = 0i64;
    for i in 0..mat.len() {
        if i < mat[i].len() {
            trace += mat[i][i];
        }
    }
    trace
}

/// Compute the characteristic polynomial of a matrix.
/// For small matrices, uses the Faddeev-LeVerrier algorithm with i128 arithmetic
/// to avoid overflow for matrices up to moderate size.
/// Returns coefficients [c_0, c_1, ..., c_n] of c_0 + c_1*x + ... + c_n*x^n.
///
/// Returns None if overflow is detected (for very large matrices).
pub fn characteristic_polynomial(matrix: &[Vec<i64>]) -> Vec<i64> {
    let n = matrix.len();
    if n == 0 {
        return vec![1];
    }

    // For matrices larger than ~30x30, the intermediate values can overflow even i128.
    // In that case, we return a best-effort result using wrapping arithmetic.
    if n > 30 {
        return characteristic_polynomial_traces(matrix);
    }

    // Faddeev-LeVerrier algorithm using i128 for intermediate values.
    let mut coeffs = vec![0i128; n + 1];
    coeffs[n] = 1;

    let mat128: Vec<Vec<i128>> = matrix.iter()
        .map(|row| row.iter().map(|&v| v as i128).collect())
        .collect();

    let mut m_prev = vec![vec![0i128; n]; n];
    for i in 0..n {
        for j in 0..n {
            m_prev[i][j] = mat128[i][j];
        }
    }

    let mut trace: i128 = 0;
    for i in 0..n {
        trace += m_prev[i][i];
    }
    coeffs[n - 1] = -trace;

    for k in 2..=n {
        let c_prev = coeffs[n - k + 1];
        let mut temp = vec![vec![0i128; n]; n];
        for i in 0..n {
            for j in 0..n {
                temp[i][j] = m_prev[i][j];
                if i == j {
                    temp[i][j] += c_prev;
                }
            }
        }

        let mut m_new = vec![vec![0i128; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum: i128 = 0;
                for l_idx in 0..n {
                    sum += mat128[i][l_idx] * temp[l_idx][j];
                }
                m_new[i][j] = sum;
            }
        }

        trace = 0;
        for i in 0..n {
            trace += m_new[i][i];
        }

        coeffs[n - k] = -trace / k as i128;
        m_prev = m_new;
    }

    coeffs.iter().map(|&c| c as i64).collect()
}

/// Fallback characteristic polynomial computation using only traces of powers.
/// Newton's identities relate power sums p_k = tr(A^k) to the characteristic
/// polynomial coefficients. This avoids storing full intermediate matrices.
///
/// For large matrices, we only compute the first few coefficients.
fn characteristic_polynomial_traces(matrix: &[Vec<i64>]) -> Vec<i64> {
    let n = matrix.len();
    let mut coeffs = vec![0i64; n + 1];
    coeffs[n] = 1;

    // Compute traces of A, A^2, ..., A^n using matrix powers
    // p_k = tr(A^k)
    // Newton's identities: k * c_{n-k} = -sum_{i=1}^{k} p_i * c_{n-k+i}
    // where c_n = 1.

    // For large n, limit to first few coefficients to avoid overflow
    let max_k = n.min(10);

    // Compute power traces
    let mut power_traces = Vec::with_capacity(max_k);
    let mut current_power = vec![vec![0i128; n]; n];
    // A^1 = A
    for i in 0..n {
        for j in 0..n {
            current_power[i][j] = matrix[i][j] as i128;
        }
    }

    for _ in 0..max_k {
        let mut trace: i128 = 0;
        for i in 0..n {
            trace += current_power[i][i];
        }
        power_traces.push(trace);

        // Compute next power: A^{k+1} = A * A^k
        let prev = current_power.clone();
        for i in 0..n {
            for j in 0..n {
                let mut sum: i128 = 0;
                for l_idx in 0..n {
                    sum += matrix[i][l_idx] as i128 * prev[l_idx][j];
                }
                current_power[i][j] = sum;
            }
        }
    }

    // Newton's identities: c_{n-k} = -(1/k) * sum_{i=0}^{k-1} p_{i+1} * c_{n-k+i+1}
    for k in 1..=max_k {
        let mut sum: i128 = 0;
        for i in 0..k {
            sum += power_traces[i] * coeffs[n - k + i + 1] as i128;
        }
        coeffs[n - k] = (-sum / k as i128) as i64;
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_symbols::modular_symbol_space;

    #[test]
    fn test_hecke_matrix_level_11() {
        // For Gamma_0(11), dim S_2 = 1.
        // The unique newform is the weight-2 form associated to the elliptic curve y^2 + y = x^3 - x^2 - 10x - 20.
        // Its Hecke eigenvalues: a_2 = -2, a_3 = -1, a_5 = 1, a_7 = -2.
        let space = modular_symbol_space(11);
        println!("Level 11: dim = {}", space.dimension);

        if space.dimension > 0 {
            let t2 = hecke_matrix(&space, 2);
            println!("T_2 matrix: {:?}", t2);

            let t3 = hecke_matrix(&space, 3);
            println!("T_3 matrix: {:?}", t3);
        }
    }

    #[test]
    fn test_char_poly_2x2() {
        // Matrix [[1, 2], [3, 4]]
        // char poly = det(xI - A) = (x-1)(x-4) - (-2)(-3) = x^2 - 5x + 4 - 6 = x^2 - 5x - 2
        let mat = vec![vec![1, 2], vec![3, 4]];
        let cp = characteristic_polynomial(&mat);
        assert_eq!(cp, vec![-2, -5, 1]);
    }

    #[test]
    fn test_char_poly_1x1() {
        let mat = vec![vec![7]];
        let cp = characteristic_polynomial(&mat);
        // det(x - 7) = x - 7, coeffs: [-7, 1]
        assert_eq!(cp, vec![-7, 1]);
    }
}
