//! Atkin-Lehner involutions for Gamma_0(N).
//!
//! For each exact divisor Q || N (i.e., Q | N and gcd(Q, N/Q) = 1),
//! the Atkin-Lehner involution W_Q acts on modular forms of level N.
//!
//! W_Q is defined by the matrix [[Qa, b], [Nc, Qd]] where ad*Q - bc*N/Q = Q,
//! i.e., [[Q*a, b], [N*c, Q*d]] with a*d*Q^2 - b*c*N = Q.
//!
//! On P^1(Z/NZ), the right action of W_Q sends (x:y) -> (Qax + Ncy : bx + Qdy).
//!
//! For N = pq, W_p and W_q reveal the factorization since their eigenvalues
//! on newforms are the pseudo-eigenvalues (Atkin-Lehner signs).

use crate::modular_symbols::{ModularSymbolSpace, lookup_symbol};
use num_integer::Integer;

/// Find the Atkin-Lehner matrix W_Q for Q || N.
/// Returns [[a, b], [c, d]] such that the AL matrix is [[Q*a, b], [N*c, Q*d]]
/// with Q^2*a*d - N*b*c = Q.
///
/// We need: Q*a*d*Q - b*c*N = Q, i.e., a*d*Q - b*c*(N/Q) = 1.
/// So we need a*d*Q - b*c*M = 1 where M = N/Q.
/// Since gcd(Q, M) = 1, we can find x, y with Q*x - M*y = 1 (extended gcd).
/// Then set a = x, d = 1, c = y (adjusted), b = 1... but we need a 2x2 matrix.
///
/// Standard construction: find integers a, b with Q*a + M*b = 1 (since gcd(Q,M)=1).
/// Then W_Q = [[Q*a, b], [N*(-b + ...), Q*...]] ... let me use the standard form.
///
/// Actually, W_Q = [[Q*a, b], [-N*c, Q*d]] where a*Q*d + b*c*(N/Q) ... is not standard.
///
/// The correct definition: W_Q is represented by any matrix [[Q*alpha, beta], [N*gamma, Q*delta]]
/// with determinant Q (not Q^2). That means Q^2*alpha*delta - N*beta*gamma = Q,
/// so Q*alpha*delta - (N/Q)*beta*gamma = 1.
///
/// With M = N/Q, gcd(Q, M) = 1, use extended gcd: Q*s + M*t = 1 for some s, t.
/// Set alpha = s, delta = 1, beta = -t, gamma = 1:
/// Check: Q*s*1 - M*(-t)*1 = Qs + Mt = 1. Yes!
///
/// So W_Q = [[Q*s, -t], [N, Q]] (in the sense of the right action matrix,
/// but we need to be careful about the action convention).
fn atkin_lehner_params(q: u64, n: u64) -> (i64, i64, i64, i64) {
    let m = n / q;
    assert_eq!(n, q * m, "Q must divide N");
    assert_eq!(q.gcd(&m), 1, "Q and N/Q must be coprime");

    // Extended gcd: Q*s + M*t = 1
    let (s, t) = extended_gcd(q as i64, m as i64);
    // Verify: q*s + m*t = 1
    debug_assert_eq!(q as i64 * s + m as i64 * t, 1);

    // W_Q matrix = [[Q*s, -t], [N*1, Q*1]]
    // Actually: [[Q*s, -t], [N, Q]] with determinant Q*s*Q - (-t)*N = Q^2*s + t*N = Q(Q*s + t*M) = Q*1 = Q.
    // Wait: Q^2*s + t*N = Q^2*s + t*Q*M = Q(Q*s + t*M) = Q*1 = Q. Correct.
    //
    // But the action on P^1(Z/NZ) depends on the convention:
    // Right action: (x,y) -> (x*Q*s + y*N, x*(-t) + y*Q)
    //             = (Q*s*x + N*y, -t*x + Q*y)
    (q as i64 * s, -(t), n as i64, q as i64)
}

/// Extended GCD: returns (s, t) such that a*s + b*t = gcd(a, b).
fn extended_gcd(a: i64, b: i64) -> (i64, i64) {
    if b == 0 {
        return (1, 0);
    }
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i64, 0i64);
    let (mut old_t, mut t) = (0i64, 1i64);

    while r != 0 {
        let q = old_r / r;
        let temp = r;
        r = old_r - q * r;
        old_r = temp;
        let temp = s;
        s = old_s - q * s;
        old_s = temp;
        let temp = t;
        t = old_t - q * t;
        old_t = temp;
    }

    (old_s, old_t)
}

/// Compute the Atkin-Lehner action W_Q on a single element of P^1(Z/NZ).
///
/// Returns the index of the resulting P^1 element.
pub fn atkin_lehner_action_on_p1(
    space: &ModularSymbolSpace,
    p1_idx: usize,
    q: u64,
) -> Option<usize> {
    let n = space.level;
    let ni = n as i64;
    let (a, b, c, d) = atkin_lehner_params(q, n);

    let x = space.p1_list[p1_idx].c;
    let y = space.p1_list[p1_idx].d;

    // Right action: (x, y) -> (a*x + c*y, b*x + d*y)
    let new_x = ((a as i128 * x as i128 + c as i128 * y as i128).rem_euclid(ni as i128)) as i64;
    let new_y = ((b as i128 * x as i128 + d as i128 * y as i128).rem_euclid(ni as i128)) as i64;

    lookup_symbol(space, new_x, new_y)
}

/// Compute the matrix of the Atkin-Lehner involution W_Q acting on the modular symbol space.
///
/// W_Q is an involution, so W_Q^2 = identity (up to sign).
/// Its eigenvalues are +1 and -1 (the "Atkin-Lehner signs").
///
/// Returns a dimension x dimension matrix.
pub fn atkin_lehner_matrix(space: &ModularSymbolSpace, q: u64) -> Vec<Vec<i64>> {
    let dim = space.dimension;
    if dim == 0 {
        return vec![];
    }

    let mut matrix = vec![vec![0i64; dim]; dim];

    for bi in 0..dim {
        let basis_vec = &space.basis[bi];

        let mut result_in_quotient = vec![0i64; dim];

        for &(p1_idx, coeff) in basis_vec {
            if let Some(result_p1_idx) = atkin_lehner_action_on_p1(space, p1_idx, q) {
                let projection = &space.relation_matrix[result_p1_idx];
                for &(basis_idx, proj_coeff) in projection {
                    if basis_idx < dim {
                        result_in_quotient[basis_idx] += coeff * proj_coeff;
                    }
                }
            }
        }

        for bj in 0..dim {
            matrix[bj][bi] = result_in_quotient[bj];
        }
    }

    matrix
}

/// Check if W_Q^2 = +/- Identity on the quotient space.
/// Returns the sign (+1 or -1) if W_Q^2 is a scalar, or None if not.
pub fn check_involution(matrix: &[Vec<i64>]) -> Option<i64> {
    let n = matrix.len();
    if n == 0 {
        return Some(1);
    }

    // Compute M^2
    let mut m2 = vec![vec![0i64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                m2[i][j] += matrix[i][k] * matrix[k][j];
            }
        }
    }

    // Check if M^2 = c * I
    let c = m2[0][0];
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { c } else { 0 };
            if m2[i][j] != expected {
                return None;
            }
        }
    }

    Some(c)
}

/// For N = pq, compute both W_p and W_q and analyze their eigenvalues.
/// The eigenvalue decomposition of W_p reveals information about the factors.
pub fn analyze_atkin_lehner(space: &ModularSymbolSpace, p: u64, q: u64) -> AtkinLehnerAnalysis {
    let wp_matrix = atkin_lehner_matrix(space, p);
    let wq_matrix = atkin_lehner_matrix(space, q);

    let wp_squared = check_involution(&wp_matrix);
    let wq_squared = check_involution(&wq_matrix);

    // Count +1 and -1 eigenvalues from the trace
    // For an involution M with M^2 = I:
    //   tr(M) = (number of +1 eigenvalues) - (number of -1 eigenvalues)
    //   dim = (number of +1 eigenvalues) + (number of -1 eigenvalues)
    // So: #(+1) = (dim + tr)/2, #(-1) = (dim - tr)/2
    let dim = space.dimension as i64;
    let wp_trace = matrix_trace(&wp_matrix);
    let wq_trace = matrix_trace(&wq_matrix);

    AtkinLehnerAnalysis {
        wp_matrix,
        wq_matrix,
        wp_squared,
        wq_squared,
        wp_trace,
        wq_trace,
        wp_plus_dim: if wp_squared == Some(1) { Some((dim + wp_trace) / 2) } else { None },
        wp_minus_dim: if wp_squared == Some(1) { Some((dim - wp_trace) / 2) } else { None },
        wq_plus_dim: if wq_squared == Some(1) { Some((dim + wq_trace) / 2) } else { None },
        wq_minus_dim: if wq_squared == Some(1) { Some((dim - wq_trace) / 2) } else { None },
    }
}

/// Result of Atkin-Lehner analysis for N = pq.
#[derive(Debug)]
pub struct AtkinLehnerAnalysis {
    pub wp_matrix: Vec<Vec<i64>>,
    pub wq_matrix: Vec<Vec<i64>>,
    pub wp_squared: Option<i64>,
    pub wq_squared: Option<i64>,
    pub wp_trace: i64,
    pub wq_trace: i64,
    pub wp_plus_dim: Option<i64>,
    pub wp_minus_dim: Option<i64>,
    pub wq_plus_dim: Option<i64>,
    pub wq_minus_dim: Option<i64>,
}

fn matrix_trace(matrix: &[Vec<i64>]) -> i64 {
    let mut trace = 0i64;
    for i in 0..matrix.len() {
        if i < matrix[i].len() {
            trace += matrix[i][i];
        }
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_symbols::modular_symbol_space;

    #[test]
    fn test_extended_gcd() {
        let (s, t) = extended_gcd(7, 11);
        assert_eq!(7 * s + 11 * t, 1);

        let (s, t) = extended_gcd(13, 17);
        assert_eq!(13 * s + 17 * t, 1);
    }

    #[test]
    fn test_atkin_lehner_params() {
        // N = 77 = 7 * 11
        let (a, b, c, d) = atkin_lehner_params(7, 77);
        // Verify determinant = Q = 7
        let det = a as i128 * d as i128 - b as i128 * c as i128;
        assert_eq!(det, 7, "W_7 matrix should have determinant 7");

        let (a, b, c, d) = atkin_lehner_params(11, 77);
        let det = a as i128 * d as i128 - b as i128 * c as i128;
        assert_eq!(det, 11, "W_11 matrix should have determinant 11");
    }

    #[test]
    fn test_atkin_lehner_level_77() {
        let space = modular_symbol_space(77);
        println!("Level 77: dim = {}", space.dimension);

        if space.dimension > 0 {
            let analysis = analyze_atkin_lehner(&space, 7, 11);
            println!("W_7 trace: {}, W_7^2 = {:?}", analysis.wp_trace, analysis.wp_squared);
            println!("W_11 trace: {}, W_11^2 = {:?}", analysis.wq_trace, analysis.wq_squared);
            if let (Some(p_plus), Some(p_minus)) = (analysis.wp_plus_dim, analysis.wp_minus_dim) {
                println!("W_7 eigenspace dims: +1 -> {}, -1 -> {}", p_plus, p_minus);
            }
            if let (Some(q_plus), Some(q_minus)) = (analysis.wq_plus_dim, analysis.wq_minus_dim) {
                println!("W_11 eigenspace dims: +1 -> {}, -1 -> {}", q_plus, q_minus);
            }
        }
    }
}
