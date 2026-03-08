use crate::types::BitRow;
use rand::Rng;

/// A sparse GF(2) matrix represented in Compressed Sparse Row (CSR) format.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Indices into `col_indices` where each row starts.
    /// Length is `rows + 1`. `row_ptr[rows]` is the total number of non-zeros.
    pub row_ptr: Vec<usize>,
    /// Column indices of the non-zero elements.
    pub col_indices: Vec<usize>,
}

impl SparseMatrix {
    /// Creates a SparseMatrix from a list of `BitRow`s.
    pub fn from_bitrows(bit_rows: &[BitRow], cols: usize) -> Self {
        let rows = bit_rows.len();
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();

        row_ptr.push(0);
        for row in bit_rows {
            for c in 0..cols {
                if row.get(c) {
                    col_indices.push(c);
                }
            }
            row_ptr.push(col_indices.len());
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_indices,
        }
    }

    /// Multiply this matrix by a block vector of 64 GF(2) elements packed into `u64`.
    /// `x` is an array of size `cols`. Each `x[i]` is a `u64` containing 64 independent bits.
    /// `y` is an array of size `rows` where the result `A * x` is stored.
    pub fn mul_block_packed(&self, x: &[u64], y: &mut [u64]) {
        y.fill(0);
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut dot = 0u64;
            for &c in &self.col_indices[start..end] {
                dot ^= x[c];
            }
            y[r] = dot;
        }
    }

    /// Transpose the sparse matrix.
    pub fn transpose(&self) -> Self {
        let mut row_counts = vec![0; self.cols];
        for &c in &self.col_indices {
            row_counts[c] += 1;
        }

        let mut row_ptr = Vec::with_capacity(self.cols + 1);
        let mut sum = 0;
        row_ptr.push(0);
        for &count in &row_counts {
            sum += count;
            row_ptr.push(sum);
        }

        let mut col_indices = vec![0; self.col_indices.len()];
        let mut current_pos = row_ptr.clone();

        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for &c in &self.col_indices[start..end] {
                let pos = current_pos[c];
                col_indices[pos] = r;
                current_pos[c] += 1;
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptr,
            col_indices,
        }
    }
}

/// Computes the minimal polynomial using the Berlekamp-Massey algorithm over GF(2).
/// Returns the coefficients of the polynomial.
pub fn berlekamp_massey(seq: &[u8]) -> Vec<u8> {
    let mut c = vec![1u8];
    let mut b = vec![1u8];
    let mut l = 0;
    let mut m = 1;

    for i in 0..seq.len() {
        let mut d = 0;
        for j in 0..=l {
            if c.len() > j {
                d ^= c[j] & seq[i - j];
            }
        }

        if d == 1 {
            let t = c.clone();

            let new_len = c.len().max(b.len() + m);
            c.resize(new_len, 0);
            for j in 0..b.len() {
                c[j + m] ^= b[j];
            }

            if 2 * l <= i {
                l = i + 1 - l;
                b = t;
                m = 1;
            } else {
                m += 1;
            }
        } else {
            m += 1;
        }
    }
    c
}

/// Find multiple dependencies (null space vectors) using Parallel Scalar Wiedemann.
/// Evaluates 64 Krylov sequences simultaneously by treating `u64` as 64 independent bits.
pub fn wiedemann_nullspace_multi(matrix: &SparseMatrix) -> Vec<Vec<usize>> {
    let n = matrix.cols;
    let m = matrix.rows;
    let max_dim = n.max(m);

    // Transpose matrix because we want left nullspace (A^T * x = 0)
    let a_t = matrix.transpose();

    let mut rng = rand::thread_rng();

    // The maximum dimension of padded A is `max_dim`.
    // In parallel scalar Wiedemann, each element of the vector is a u64 representing
    // 64 independent random trials. Length of vector is max_dim.

    // Random projection vector u
    let mut u = vec![0u64; max_dim];
    for val in &mut u {
        *val = rng.gen();
    }
    // Random initial vector v (in GF(2)^m)
    let mut v = vec![0u64; max_dim];
    for val in &mut v {
        *val = rng.gen();
    }

    // Zero out padding bits for v and u so they truly reside in max_dim
    for i in a_t.cols..max_dim {
        v[i] = 0;
        u[i] = 0;
    }

    // Sequence generation: s_i = u^T * A^i * v
    let seq_len = 2 * max_dim + 10;
    let mut seqs = vec![vec![0u8; seq_len]; 64];

    let mut current_v = v.clone();
    let mut next_v = vec![0u64; max_dim];

    for k in 0..seq_len {
        let mut dot = 0u64;
        for i in 0..max_dim {
            dot ^= u[i] & current_v[i];
        }

        for bit_idx in 0..64 {
            seqs[bit_idx][k] = ((dot >> bit_idx) & 1) as u8;
        }

        a_t.mul_block_packed(&current_v, &mut next_v);

        // Pad operation implicitly: A^T maps GF(2)^m -> GF(2)^n.
        // We padded A to be max_dim x max_dim.
        current_v.fill(0);
        current_v[..a_t.rows].copy_from_slice(&next_v[..a_t.rows]);
    }

    // Find minimal polynomials for each of the 64 independent sequences
    let mut minpolys = Vec::with_capacity(64);
    for bit_idx in 0..64 {
        minpolys.push(berlekamp_massey(&seqs[bit_idx]));
    }

    // Evaluate polynomials to construct nullspace vectors: w_k = p_k(A)v_k
    let mut result_vec = vec![0u64; max_dim];

    // Since each bit sequence k has its own minimal polynomial, we have to evaluate them.
    // However, A * current_v computes the matrix-vector product for all 64 sequences at once!
    // Instead of doing 64 independent evaluations taking O(64 * N^2), we can compute
    // all A^i * v simultaneously up to the max degree, and accumulate the results
    // for sequence k if its polynomial has a 1 at that degree.

    let max_degree = minpolys.iter().map(|p| p.len()).max().unwrap_or(0);

    // Determine the shifts for each polynomial to ensure we don't evaluate 0 terms
    let mut shifts = vec![0; 64];
    for bit_idx in 0..64 {
        while shifts[bit_idx] < minpolys[bit_idx].len() && minpolys[bit_idx][shifts[bit_idx]] == 0 {
            shifts[bit_idx] += 1;
        }
    }

    let mut term_v = v.clone();

    for i in 0..max_degree {
        for bit_idx in 0..64 {
            let shift = shifts[bit_idx];
            if i >= shift && i < minpolys[bit_idx].len() && minpolys[bit_idx][i] == 1 {
                for j in 0..max_dim {
                    // We only accumulate the bit_idx-th bit of term_v[j]
                    let term_bit = (term_v[j] >> bit_idx) & 1;
                    result_vec[j] ^= term_bit << bit_idx;
                }
            }
        }

        a_t.mul_block_packed(&term_v, &mut next_v);
        term_v.fill(0);
        term_v[..a_t.rows].copy_from_slice(&next_v[..a_t.rows]);
    }

    // Extract non-zero indices as dependencies (length m)
    let mut dependencies = Vec::new();

    for bit_idx in 0..64 {
        let mut dep = Vec::new();
        let mut is_zero = true;
        for i in 0..m {
            if (result_vec[i] >> bit_idx) & 1 == 1 {
                dep.push(i);
                is_zero = false;
            }
        }

        if !is_zero && !dep.is_empty() {
            // Verify
            let mut verify_vec = vec![0u64; a_t.rows];
            let mut test_input = vec![0u64; max_dim];
            for &idx in &dep {
                test_input[idx] = 1;
            }
            a_t.mul_block_packed(&test_input, &mut verify_vec);

            let mut verified = true;
            for val in verify_vec {
                if val != 0 {
                    verified = false;
                    break;
                }
            }

            if verified {
                dependencies.push(dep);
            }
        }
    }

    dependencies
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BitRow;

    #[test]
    fn test_wiedemann_nullspace_multi_simple() {
        let ncols = 3;
        let mut rows: Vec<BitRow> = (0..3).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0); rows[0].set(2);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(0); rows[2].set(1);

        let sparse = SparseMatrix::from_bitrows(&rows, ncols);
        let deps = wiedemann_nullspace_multi(&sparse);

        assert!(!deps.is_empty(), "Failed to find dependency");
        let dep_vec = &deps[0];

        let mut result = vec![0; ncols];
        for &r in dep_vec {
            for c in 0..ncols {
                if rows[r].get(c) {
                    result[c] ^= 1;
                }
            }
        }

        for v in result {
            assert_eq!(v, 0, "Dependency did not XOR to 0 vector");
        }
    }

    #[test]
    fn test_wiedemann_nullspace_multi_rectangular() {
        let m = 5;
        let n = 3;
        let mut rows: Vec<BitRow> = (0..m).map(|_| BitRow::new(n)).collect();
        rows[0].set(0);
        rows[1].set(1);
        rows[2].set(2);
        rows[3].set(0); rows[3].set(1);
        rows[4].set(1); rows[4].set(2);

        let sparse = SparseMatrix::from_bitrows(&rows, n);

        let deps = wiedemann_nullspace_multi(&sparse);
        assert!(!deps.is_empty(), "Failed to find dependency");

        for dep_vec in deps {
            let mut result = vec![0; n];
            for &r in &dep_vec {
                for c in 0..n {
                    if rows[r].get(c) {
                        result[c] ^= 1;
                    }
                }
            }
            for v in result {
                assert_eq!(v, 0, "Dependency did not XOR to 0 vector");
            }
        }
    }
}
