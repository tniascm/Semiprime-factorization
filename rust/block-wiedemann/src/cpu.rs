//! CPU fallback implementation for GF(2) SpMV.
//!
//! Uses Rayon for multi-threading, providing a robust fallback when
//! Metal hardware (M4 GPU) is unavailable.

use crate::matrix::CsrMatrix;
use rayon::prelude::*;

/// Computes y = A * x over GF(2) where elements of x and y are packed u64 blocks.
pub fn spmv_gf2_64_cpu(matrix: &CsrMatrix, x: &[u64], y: &mut [u64]) {
    y.par_iter_mut().enumerate().for_each(|(i, y_val)| {
        let start = matrix.row_ptrs[i] as usize;
        let end = matrix.row_ptrs[i + 1] as usize;

        let mut acc = 0u64;
        for j in start..end {
            let col = matrix.col_indices[j] as usize;
            acc ^= x[col];
        }
        *y_val = acc;
    });
}

/// Run a sequence generation for Block Wiedemann using the CPU.
pub fn compute_sequence_cpu(
    matrix_a: &CsrMatrix,
    matrix_at: &CsrMatrix,
    initial_vector: &[u64],
    iterations: usize,
) -> Vec<Vec<u64>> {
    let n = matrix_a.cols;
    let m = matrix_a.rows;

    let mut v_in = initial_vector.to_vec();
    let mut v_mid = vec![0u64; m];
    let mut v_out = vec![0u64; n];

    let mut results = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        spmv_gf2_64_cpu(matrix_a, &v_in, &mut v_mid);
        spmv_gf2_64_cpu(matrix_at, &v_mid, &mut v_out);

        results.push(v_out.clone());
        std::mem::swap(&mut v_in, &mut v_out);
    }

    results
}
