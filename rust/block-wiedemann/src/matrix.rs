//! Sparse matrix representation over GF(2).

use std::fmt::Debug;

/// Compressed Sparse Row (CSR) matrix over GF(2).
///
/// We only need to store row pointers and column indices since
/// all non-zero values in GF(2) are 1.
#[derive(Clone)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    /// `row_ptrs[i]` to `row_ptrs[i+1]` gives the range in `col_indices` for row `i`.
    pub row_ptrs: Vec<u32>,
    /// The column indices of the non-zero entries (1s).
    pub col_indices: Vec<u32>,
}

impl Debug for CsrMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.col_indices.len())
            .finish()
    }
}

impl CsrMatrix {
    /// Create an empty matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptrs: vec![0; rows + 1],
            col_indices: Vec::new(),
        }
    }

    /// Transpose the matrix (returns a new CSR matrix).
    /// Essential for Block Wiedemann which requires A^T * A.
    pub fn transpose(&self) -> Self {
        let mut row_counts = vec![0; self.cols];
        for &col in &self.col_indices {
            row_counts[col as usize] += 1;
        }

        let mut row_ptrs = vec![0; self.cols + 1];
        for i in 0..self.cols {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }

        let mut current_ptrs = row_ptrs.clone();
        let mut col_indices = vec![0; self.col_indices.len()];

        for i in 0..self.rows {
            let start = self.row_ptrs[i] as usize;
            let end = self.row_ptrs[i + 1] as usize;
            for j in start..end {
                let col = self.col_indices[j] as usize;
                let dest = current_ptrs[col] as usize;
                col_indices[dest] = i as u32;
                current_ptrs[col] += 1;
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptrs,
            col_indices,
        }
    }
}
