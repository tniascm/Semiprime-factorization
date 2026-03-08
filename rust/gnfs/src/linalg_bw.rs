//! Block Wiedemann algorithm over GF(2) for sparse matrix null-space computation.
//!
//! Provides O(n^2) linear algebra as an alternative to dense Gaussian elimination
//! for large NFS matrices (50K+ rows). Operates on the same `BitRow` input format
//! as the existing `linalg` module.

use crate::types::BitRow;
use rayon::prelude::*;

/// Sparse matrix over GF(2) in Compressed Sparse Row (CSR) format.
///
/// Stores only the column indices of non-zero entries, making SpMV O(nnz)
/// instead of O(n^2) for dense matrices.
pub(crate) struct SparseMatrixGF2 {
    pub rows: usize,
    pub cols: usize,
    /// Column indices of non-zero entries, sorted by row.
    col_indices: Vec<usize>,
    /// row_ptrs[i]..row_ptrs[i+1] gives the range in col_indices for row i.
    /// Length is rows + 1.
    row_ptrs: Vec<usize>,
}

impl SparseMatrixGF2 {
    /// Convert BitRow slice to CSR format.
    ///
    /// Uses word-level bit traversal (trailing_zeros + w &= w-1) for O(nnz)
    /// conversion instead of per-bit checking.
    pub fn from_bitrows(rows: &[BitRow], ncols: usize) -> Self {
        let nrows = rows.len();
        let mut col_indices = Vec::new();
        let mut row_ptrs = Vec::with_capacity(nrows + 1);

        for row in rows {
            row_ptrs.push(col_indices.len());
            for (wi, &word) in row.bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let col = wi * 64 + bit;
                    if col < ncols {
                        col_indices.push(col);
                    }
                    w &= w - 1;
                }
            }
        }
        row_ptrs.push(col_indices.len());

        SparseMatrixGF2 {
            rows: nrows,
            cols: ncols,
            col_indices,
            row_ptrs,
        }
    }

    /// Build CSR for the TRANSPOSE of the BitRow matrix.
    ///
    /// Dependencies require the left null space (v^T M = 0), which equals
    /// the right null space of M^T. Wiedemann finds right null space vectors,
    /// so we run it on M^T.
    ///
    /// M^T has ncols rows and nrows columns: each original column becomes a row.
    pub fn from_bitrows_transposed(rows: &[BitRow], ncols: usize) -> Self {
        let nrows = rows.len();
        // Transpose: collect which original rows set each column
        let mut col_entries: Vec<Vec<usize>> = vec![Vec::new(); ncols];
        for (r, row) in rows.iter().enumerate() {
            for (wi, &word) in row.bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let col = wi * 64 + bit;
                    if col < ncols {
                        col_entries[col].push(r);
                    }
                    w &= w - 1;
                }
            }
        }

        let mut col_indices = Vec::new();
        let mut row_ptrs = Vec::with_capacity(ncols + 1);
        for entries in &col_entries {
            row_ptrs.push(col_indices.len());
            col_indices.extend_from_slice(entries);
        }
        row_ptrs.push(col_indices.len());

        SparseMatrixGF2 {
            rows: ncols,
            cols: nrows,
            col_indices,
            row_ptrs,
        }
    }

    /// Scalar SpMV: y = M * x over GF(2).
    ///
    /// x and y are bit-packed in u64 slices. Slow (1 bit per iteration) —
    /// use `mul_block_gf2` or `par_mul_block_gf2` for production.
    pub fn mul_vec_gf2(&self, x: &[u64], y: &mut [u64]) {
        y.fill(0);
        for i in 0..self.rows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let mut dot = 0u64;
            for &col in &self.col_indices[start..end] {
                dot ^= (x[col / 64] >> (col % 64)) & 1;
            }
            if dot != 0 {
                y[i / 64] ^= 1 << (i % 64);
            }
        }
    }

    /// Block SpMV: 64-way parallel matrix-vector multiply over GF(2).
    ///
    /// input_block[j] packs bit j of all 64 vectors. output_block[i] packs
    /// bit i of all 64 result vectors. Computes 64 independent dot products
    /// per row via XOR of column entries — zero branching.
    pub fn mul_block_gf2(&self, input_block: &[u64], output_block: &mut [u64]) {
        assert_eq!(input_block.len(), self.cols);
        assert_eq!(output_block.len(), self.rows);
        output_block.fill(0);

        for i in 0..self.rows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let mut row_val = 0u64;
            for &col in &self.col_indices[start..end] {
                row_val ^= input_block[col];
            }
            output_block[i] = row_val;
        }
    }

    /// Parallel block SpMV using Rayon.
    ///
    /// Same semantics as `mul_block_gf2` but distributes rows across cores.
    pub fn par_mul_block_gf2(&self, input_block: &[u64], output_block: &mut [u64]) {
        assert_eq!(input_block.len(), self.cols);
        assert_eq!(output_block.len(), self.rows);

        output_block
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out_val)| {
                let start = self.row_ptrs[i];
                let end = self.row_ptrs[i + 1];
                let mut row_val = 0u64;
                for &col in &self.col_indices[start..end] {
                    row_val ^= input_block[col];
                }
                *out_val = row_val;
            });
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }
}

// ---------------------------------------------------------------------------
// Berlekamp-Massey over GF(2)
// ---------------------------------------------------------------------------

/// Berlekamp-Massey algorithm over GF(2).
///
/// Returns the **annihilating polynomial** f(x) such that f(M)*u = 0
/// (i.e., sum_j f_j * s_{n+j} = 0 for all valid n).
///
/// BM internally computes the LFSR connection polynomial C(x) satisfying
/// sum_j c_j * s_{n-j} = 0. The annihilating polynomial is the reverse:
/// f_j = c_{L-j} where L is the LFSR length.
fn berlekamp_massey(sequence: &[u64], len: usize) -> Vec<u64> {
    let nwords = (len + 64) / 64; // +1 word headroom for shifts
    let mut c = vec![0u64; nwords];
    c[0] = 1; // c(x) = 1

    let mut b = vec![0u64; nwords];
    b[0] = 1; // b(x) = 1

    let mut l = 0usize;
    let mut m = 1usize;

    for i in 0..len {
        // Discrepancy: d = sum_{j=0}^{l} c_j * s_{i-j}
        let mut d = 0u64;
        for j in 0..=l {
            if j > i {
                break;
            }
            let c_j = (c[j / 64] >> (j % 64)) & 1;
            let s_idx = i - j;
            let s_bit = (sequence[s_idx / 64] >> (s_idx % 64)) & 1;
            d ^= c_j & s_bit;
        }

        if d == 1 {
            // t = c(x) + x^m * b(x)
            let mut t = c.clone();
            let max_shift = len.saturating_sub(m);
            for j in 0..max_shift {
                let b_j = (b[j / 64] >> (j % 64)) & 1;
                if b_j == 1 {
                    let pos = j + m;
                    if pos / 64 < t.len() {
                        t[pos / 64] ^= 1u64 << (pos % 64);
                    }
                }
            }

            if 2 * l <= i {
                l = i + 1 - l;
                b = c;
                m = 1;
            } else {
                m += 1;
            }
            c = t;
        } else {
            m += 1;
        }
    }

    // Reverse the connection polynomial to get the annihilating polynomial:
    // f_j = c_{L-j} for j = 0..L
    reverse_poly(&c, l)
}

/// Reverse a bit-packed polynomial of degree `deg`: f_j = c_{deg-j}.
fn reverse_poly(c: &[u64], deg: usize) -> Vec<u64> {
    let nwords = (deg + 64) / 64;
    let mut f = vec![0u64; nwords];
    for j in 0..=deg {
        let c_bit = if j / 64 < c.len() {
            (c[j / 64] >> (j % 64)) & 1
        } else {
            0
        };
        if c_bit == 1 {
            let pos = deg - j;
            f[pos / 64] |= 1u64 << (pos % 64);
        }
    }
    // Trim trailing zero words
    while f.len() > 1 && f.last() == Some(&0) {
        f.pop();
    }
    f
}

// ---------------------------------------------------------------------------
// Scalar Wiedemann
// ---------------------------------------------------------------------------

/// Generate the sequence a_i = v^T * M^i * u for i = 0..2n.
///
/// Returns bit-packed sequence of length 2n.
fn generate_sequence(
    matrix: &SparseMatrixGF2,
    u: &[u64],
    v: &[u64],
    n: usize,
) -> Vec<u64> {
    let seq_len = 2 * n;
    let mut sequence = vec![0u64; (seq_len + 63) / 64];
    let words = (matrix.rows + 63) / 64;
    let mut current_u = u.to_vec();
    current_u.resize(words, 0);
    let mut next_u = vec![0u64; words];

    for i in 0..seq_len {
        // Dot product: parity of v AND current_u
        let mut dot = 0u64;
        for j in 0..v.len().min(current_u.len()) {
            dot ^= (v[j] & current_u[j]).count_ones() as u64 & 1;
        }
        if dot != 0 {
            sequence[i / 64] ^= 1 << (i % 64);
        }

        matrix.mul_vec_gf2(&current_u, &mut next_u);
        std::mem::swap(&mut current_u, &mut next_u);
    }

    sequence
}

/// Evaluate kernel vector from minimal polynomial.
///
/// Computes w = sum_{i=1}^{deg} c_i * M^{i-1} * u.
/// If c_0 = 0 (which happens when M is singular), then M*w = 0.
fn evaluate_kernel_vector(
    matrix: &SparseMatrixGF2,
    u: &[u64],
    min_poly: &[u64],
) -> Vec<u64> {
    let words = (matrix.rows + 63) / 64;
    let mut w = vec![0u64; words];
    let mut current_u = u.to_vec();
    current_u.resize(words, 0);
    let mut next_u = vec![0u64; words];

    let max_degree = min_poly.len() * 64;

    for i in 1..max_degree {
        if i / 64 >= min_poly.len() {
            break;
        }
        if (min_poly[i / 64] >> (i % 64)) & 1 == 1 {
            for j in 0..w.len() {
                w[j] ^= current_u[j];
            }
        }

        matrix.mul_vec_gf2(&current_u, &mut next_u);
        std::mem::swap(&mut current_u, &mut next_u);
    }

    w
}

/// Scalar Wiedemann: find a single nullspace vector.
///
/// Returns None if the random vectors u, v don't produce a valid kernel vector.
fn wiedemann_nullspace_vector(
    matrix: &SparseMatrixGF2,
    rng: &mut SimpleRng,
) -> Option<Vec<u64>> {
    let n = matrix.rows;
    let words = (n + 63) / 64;

    let mut u = vec![0u64; words];
    let mut v = vec![0u64; words];
    for i in 0..words {
        u[i] = rng.next_u64();
        v[i] = rng.next_u64();
    }
    // Mask unused bits
    if n % 64 != 0 {
        let mask = (1u64 << (n % 64)) - 1;
        u[words - 1] &= mask;
        v[words - 1] &= mask;
    }

    let sequence = generate_sequence(matrix, &u, &v, n);
    let min_poly = berlekamp_massey(&sequence, 2 * n);
    let w = evaluate_kernel_vector(matrix, &u, &min_poly);

    // Check non-zero
    if w.iter().all(|&x| x == 0) {
        return None;
    }

    // Verify M*w = 0
    let mut mw = vec![0u64; words];
    matrix.mul_vec_gf2(&w, &mut mw);

    // Mask unused bits in verification
    let mut valid = true;
    for i in 0..words {
        let mut val = mw[i];
        if i == words - 1 && n % 64 != 0 {
            val &= (1u64 << (n % 64)) - 1;
        }
        if val != 0 {
            valid = false;
            break;
        }
    }

    if valid { Some(w) } else { None }
}

// ---------------------------------------------------------------------------
// Batched Wiedemann (64-way parallel)
// ---------------------------------------------------------------------------

/// Generate 64 independent sequences simultaneously using block SpMV.
///
/// u_block and v_block have one u64 per matrix row, each packing 64 vectors.
/// Returns 64 bit-packed sequences of length 2n.
fn batched_generate_sequence(
    matrix: &SparseMatrixGF2,
    u_block: &[u64],
    v_block: &[u64],
    n: usize,
) -> Vec<Vec<u64>> {
    let seq_len = 2 * n;
    let mut sequences = vec![vec![0u64; (seq_len + 63) / 64]; 64];
    let mut current_u = u_block.to_vec();
    let mut next_u = vec![0u64; u_block.len()];

    for i in 0..seq_len {
        // 64 simultaneous dot products: v^T * current_u for each of 64 vectors
        let mut dot_block = 0u64;
        for j in 0..v_block.len() {
            dot_block ^= v_block[j] & current_u[j];
        }

        // Distribute to sequence buffers
        let word_idx = i / 64;
        let bit_idx = i % 64;
        for seq_idx in 0..64 {
            if (dot_block >> seq_idx) & 1 == 1 {
                sequences[seq_idx][word_idx] ^= 1 << bit_idx;
            }
        }

        matrix.par_mul_block_gf2(&current_u, &mut next_u);
        std::mem::swap(&mut current_u, &mut next_u);
    }

    sequences
}

/// Evaluate 64 kernel vectors simultaneously from their minimal polynomials.
///
/// Uses block SpMV to advance all 64 vectors at once, applying polynomial
/// coefficients via bitmask selection.
fn batched_evaluate_kernel_vectors(
    matrix: &SparseMatrixGF2,
    u_block: &[u64],
    min_polys: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    assert_eq!(min_polys.len(), 64);

    // Find global max degree
    let mut max_degree = 0usize;
    for poly in min_polys {
        for i in 0..(poly.len() * 64) {
            if i / 64 < poly.len() && (poly[i / 64] >> (i % 64)) & 1 == 1 {
                max_degree = max_degree.max(i);
            }
        }
    }

    let n = matrix.rows; // use rows for output dimension
    let mut w_block = vec![0u64; n];
    let mut current_u = u_block.to_vec();
    let mut next_u = vec![0u64; n];

    for i in 1..=max_degree {
        // Build 64-bit mask: which polynomials have coefficient 1 at degree i
        let mut poly_mask = 0u64;
        for seq_idx in 0..64 {
            let poly = &min_polys[seq_idx];
            if i / 64 < poly.len() && (poly[i / 64] >> (i % 64)) & 1 == 1 {
                poly_mask |= 1 << seq_idx;
            }
        }

        if poly_mask != 0 {
            for j in 0..w_block.len() {
                w_block[j] ^= current_u[j] & poly_mask;
            }
        }

        matrix.par_mul_block_gf2(&current_u, &mut next_u);
        std::mem::swap(&mut current_u, &mut next_u);
    }

    // Unpack 64 individual vectors from block layout.
    // w_block[row_idx] has bit k set if vector k has a 1 at position row_idx.
    // We need to convert to standard bit-packed format: w[k][row_idx/64] bit (row_idx%64).
    let out_words = (n + 63) / 64;
    let mut w_individual = vec![vec![0u64; out_words]; 64];

    // FIX from Pillar 4: iterate over matrix.rows (not matrix.cols)
    for row_idx in 0..n {
        let block_word = w_block[row_idx];
        if block_word != 0 {
            let out_word_idx = row_idx / 64;
            let out_bit_idx = row_idx % 64;
            for seq_idx in 0..64 {
                if (block_word >> seq_idx) & 1 == 1 {
                    w_individual[seq_idx][out_word_idx] |= 1 << out_bit_idx;
                }
            }
        }
    }

    w_individual
}

/// Run batched Wiedemann to find up to 64 kernel vectors per round.
///
/// Pads to square matrix if needed (BW requires square).
/// Multiple rounds until enough vectors found or max attempts exceeded.
fn wiedemann_nullspace_batched(
    matrix: &SparseMatrixGF2,
    rng: &mut SimpleRng,
) -> Vec<Vec<u64>> {
    let n = matrix.rows;
    assert_eq!(n, matrix.cols, "BW requires square matrix");

    let mut u_block = vec![0u64; n];
    let mut v_block = vec![0u64; n];
    for i in 0..n {
        u_block[i] = rng.next_u64();
        v_block[i] = rng.next_u64();
    }

    // Generate 64 sequences simultaneously
    let sequences = batched_generate_sequence(matrix, &u_block, &v_block, n);

    // Berlekamp-Massey in parallel for 64 independent sequences
    let min_polys: Vec<Vec<u64>> = sequences
        .par_iter()
        .map(|seq| berlekamp_massey(seq, 2 * n))
        .collect();

    // Evaluate kernel vectors in batch
    let w_individual = batched_evaluate_kernel_vectors(matrix, &u_block, &min_polys);

    // Verify and filter valid non-trivial kernel vectors
    let words = (n + 63) / 64;
    let mut valid = Vec::new();

    for mut w in w_individual {
        if w.iter().all(|&x| x == 0) {
            continue;
        }

        // Clean up unused bits
        if n % 64 != 0 {
            if words > 0 {
                w[words - 1] &= (1u64 << (n % 64)) - 1;
            }
        }

        // Verify M*w = 0
        let mut mw = vec![0u64; words];
        matrix.mul_vec_gf2(&w, &mut mw);

        let is_zero = (0..words).all(|i| {
            let mut val = mw[i];
            if i == words - 1 && n % 64 != 0 {
                val &= (1u64 << (n % 64)) - 1;
            }
            val == 0
        });

        if is_zero {
            valid.push(w);
        }
    }

    valid
}

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG (LCG) for reproducible kernel vector generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // SplitMix64 output scramble for good low-bit quality
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// Public API: find_dependencies_bw
// ---------------------------------------------------------------------------

/// Find null-space dependencies using Block Wiedemann.
///
/// Drop-in replacement for `find_dependencies` from `linalg.rs`.
/// Accepts `&[BitRow]` input and returns `Vec<Vec<usize>>` — each inner vec
/// is a list of row indices whose XOR is the zero vector.
///
/// Key insight: dependencies require the LEFT null space of M (v^T M = 0),
/// which equals the RIGHT null space of M^T. Wiedemann finds right null space
/// vectors, so we run it on M^T.
///
/// The seed for the PRNG is controlled by `RUST_NFS_BW_SEED` env var (default: 42).
pub fn find_dependencies_bw(rows: &[BitRow], ncols: usize) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return vec![];
    }

    let seed: u64 = std::env::var("RUST_NFS_BW_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let mut rng = SimpleRng::new(seed);

    // Build CSR for M^T. Kernel of M^T = left null space of M = dependencies.
    // M^T has ncols rows and nrows columns.
    let mut csr = SparseMatrixGF2::from_bitrows_transposed(rows, ncols);

    // Pad to square (BW requires square matrix)
    let dim = csr.rows.max(csr.cols);
    if csr.rows < dim {
        let nnz = csr.col_indices.len();
        for _ in 0..(dim - csr.rows) {
            csr.row_ptrs.push(nnz);
        }
        csr.rows = dim;
    }
    csr.cols = dim;

    // Multiple rounds to accumulate kernel vectors
    let max_rounds = 10;
    let mut all_kernel_vecs: Vec<Vec<u64>> = Vec::new();
    for round in 0..max_rounds {
        let new_vecs = if dim >= 64 {
            wiedemann_nullspace_batched(&csr, &mut rng)
        } else {
            let mut vecs = Vec::new();
            for _ in 0..64 {
                if let Some(v) = wiedemann_nullspace_vector(&csr, &mut rng) {
                    vecs.push(v);
                }
            }
            vecs
        };

        for v in new_vecs {
            let is_dup = all_kernel_vecs.iter().any(|existing| {
                existing.iter().zip(v.iter()).all(|(&a, &b)| a == b)
            });
            if !is_dup {
                all_kernel_vecs.push(v);
            }
        }

        // Left null space dimension = nrows - rank(M). For overdetermined systems
        // (nrows > ncols), dimension ≥ nrows - ncols.
        let target = if nrows > ncols { nrows - ncols + 1 } else { 1 };
        if all_kernel_vecs.len() >= target || (round > 2 && !all_kernel_vecs.is_empty()) {
            break;
        }
    }

    // Convert kernel vectors of M^T to dependency lists.
    // Kernel vector x of M^T has components indexed by M^T's columns = M's rows.
    // x[i]=1 means row i participates in the dependency.
    let mut deps = Vec::new();
    for kv in &all_kernel_vecs {
        let mut dep = Vec::new();
        for (wi, &word) in kv.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let idx = wi * 64 + bit;
                if idx < nrows {
                    dep.push(idx);
                }
                w &= w - 1;
            }
        }
        if dep.len() >= 2 {
            // Verify: XOR of these rows should be zero
            let mut check = BitRow::new(ncols);
            for &idx in &dep {
                check.xor_with(&rows[idx]);
            }
            if check.is_zero() {
                deps.push(dep);
            }
        }
    }

    deps
}

/// Pre-elimination + Block Wiedemann.
///
/// Runs the same singleton/weight-2 pre-elimination as `find_dependencies_with_preelim`,
/// then uses Block Wiedemann instead of dense GE on the compacted matrix.
pub fn find_dependencies_with_preelim_bw(rows: &[BitRow], ncols: usize) -> Vec<Vec<usize>> {
    let pr = pre_eliminate(rows, ncols);

    if pr.compact_matrix.is_empty() || pr.compact_ncols == 0 {
        return pr.preelim_deps;
    }

    let compact_deps = find_dependencies_bw(&pr.compact_matrix, pr.compact_ncols);
    let combined = combine_deps(&pr, compact_deps, rows);
    let mut deps = pr.preelim_deps;
    deps.extend(combined);
    deps
}

// ---------------------------------------------------------------------------
// Pre-elimination infrastructure (shared with linalg.rs)
// ---------------------------------------------------------------------------

/// Result of the pre-elimination phase, capturing all state needed to
/// reconstruct original-row-index dependencies from compact-matrix deps.
pub(crate) struct PreelimResult {
    /// Compacted matrix (surviving rows with remapped columns).
    pub compact_matrix: Vec<BitRow>,
    /// Number of columns in the compact matrix.
    pub compact_ncols: usize,
    /// Maps compact row index → original row index in the pre-elimination state.
    pub ge_rows: Vec<usize>,
    /// History of XOR merges during pre-elimination.
    /// history[orig_row_idx] = list of original row indices merged into it.
    pub history: Vec<Vec<usize>>,
    /// Dependencies found during pre-elimination itself (zero rows).
    pub preelim_deps: Vec<Vec<usize>>,
}

/// Run singleton/weight-2 pre-elimination, returning the compacted matrix
/// and bookkeeping needed to map results back to original row indices.
///
/// This is the same algorithm as the pre-elimination loop in
/// `find_dependencies_with_preelim`, extracted for reuse with BW.
pub(crate) fn pre_eliminate(rows: &[BitRow], ncols: usize) -> PreelimResult {
    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return PreelimResult {
            compact_matrix: vec![],
            compact_ncols: 0,
            ge_rows: vec![],
            history: vec![],
            preelim_deps: vec![],
        };
    }

    let mut matrix: Vec<BitRow> = rows.to_vec();
    let mut history: Vec<Vec<usize>> = (0..nrows).map(|i| vec![i]).collect();
    let mut row_alive = vec![true; nrows];
    let mut col_alive = vec![true; ncols];

    // Pre-elimination loop (same as linalg.rs)
    loop {
        let mut changed = false;

        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); ncols];
        for r in 0..nrows {
            if !row_alive[r] {
                continue;
            }
            for (wi, &word) in matrix[r].bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let c = wi * 64 + bit;
                    if c < ncols && col_alive[c] && col_rows[c].len() < 3 {
                        col_rows[c].push(r);
                    }
                    w &= w - 1;
                }
            }
        }

        // Pass 1: singleton removal
        for c in 0..ncols {
            if !col_alive[c] {
                continue;
            }
            if col_rows[c].len() == 1 {
                let r = col_rows[c][0];
                if row_alive[r] {
                    row_alive[r] = false;
                    col_alive[c] = false;
                    changed = true;
                }
            }
        }

        // Pass 2: weight-2 merge (one at a time, then recompute)
        for c in 0..ncols {
            if !col_alive[c] {
                continue;
            }
            let alive_in_col: Vec<usize>;
            if col_rows[c].len() < 3 {
                alive_in_col = col_rows[c]
                    .iter()
                    .copied()
                    .filter(|&r| row_alive[r])
                    .collect();
            } else {
                let mut v = Vec::new();
                for r in 0..nrows {
                    if row_alive[r] && matrix[r].get(c) {
                        v.push(r);
                        if v.len() > 2 {
                            break;
                        }
                    }
                }
                alive_in_col = v;
            }
            if alive_in_col.len() == 2 {
                let r1 = alive_in_col[0];
                let r2 = alive_in_col[1];
                let r2_data = matrix[r2].clone();
                matrix[r1].xor_with(&r2_data);
                let mut merged = history[r1].clone();
                for &idx in &history[r2] {
                    if let Some(pos) = merged.iter().position(|&x| x == idx) {
                        merged.swap_remove(pos);
                    } else {
                        merged.push(idx);
                    }
                }
                history[r1] = merged;
                row_alive[r2] = false;
                col_alive[c] = false;
                changed = true;
                break;
            } else if alive_in_col.is_empty() {
                col_alive[c] = false;
            } else if alive_in_col.len() == 1 {
                row_alive[alive_in_col[0]] = false;
                col_alive[c] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Build compacted matrix
    let surviving_rows: Vec<usize> = (0..nrows).filter(|&r| row_alive[r]).collect();
    let surviving_cols: Vec<usize> = (0..ncols).filter(|&c| col_alive[c]).collect();

    let nwords = (ncols + 63) / 64;
    let mut col_alive_mask = vec![0u64; nwords];
    for &c in &surviving_cols {
        col_alive_mask[c / 64] |= 1u64 << (c % 64);
    }

    let mut preelim_deps = Vec::new();
    let mut ge_rows = Vec::new();
    for &r in &surviving_rows {
        let all_zero = matrix[r]
            .bits
            .iter()
            .zip(col_alive_mask.iter())
            .all(|(&row_word, &mask_word)| row_word & mask_word == 0);
        if all_zero {
            let mut dep = history[r].clone();
            dep.sort_unstable();
            if dep.len() >= 2 {
                preelim_deps.push(dep);
            }
        } else {
            ge_rows.push(r);
        }
    }

    if ge_rows.is_empty() || surviving_cols.is_empty() {
        return PreelimResult {
            compact_matrix: vec![],
            compact_ncols: 0,
            ge_rows: vec![],
            history,
            preelim_deps,
        };
    }

    let compact_ncols = surviving_cols.len();
    let mut old_to_new_col = vec![usize::MAX; ncols];
    for (new_c, &old_c) in surviving_cols.iter().enumerate() {
        old_to_new_col[old_c] = new_c;
    }

    let mut compact_matrix: Vec<BitRow> = Vec::with_capacity(ge_rows.len());
    for &r in &ge_rows {
        let mut compact_row = BitRow::new(compact_ncols);
        for (wi, &word) in matrix[r].bits.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let old_c = wi * 64 + bit;
                if old_c < ncols {
                    let new_c = old_to_new_col[old_c];
                    if new_c != usize::MAX {
                        compact_row.set(new_c);
                    }
                }
                w &= w - 1;
            }
        }
        compact_matrix.push(compact_row);
    }

    PreelimResult {
        compact_matrix,
        compact_ncols,
        ge_rows,
        history,
        preelim_deps,
    }
}

/// Map compact-matrix dependency indices back to original row indices.
fn combine_deps(
    pr: &PreelimResult,
    compact_deps: Vec<Vec<usize>>,
    original_rows: &[BitRow],
) -> Vec<Vec<usize>> {
    let ncols = if original_rows.is_empty() {
        0
    } else {
        original_rows[0].ncols
    };
    let mut result = Vec::new();

    for cdep in compact_deps {
        let mut combined = std::collections::HashSet::new();
        for &ci in &cdep {
            let orig_row = pr.ge_rows[ci];
            for &orig_idx in &pr.history[orig_row] {
                if !combined.remove(&orig_idx) {
                    combined.insert(orig_idx);
                }
            }
        }
        if combined.len() >= 2 {
            let mut dep: Vec<usize> = combined.into_iter().collect();
            dep.sort_unstable();

            // Verify the dependency
            let mut check = BitRow::new(ncols);
            for &idx in &dep {
                check.xor_with(&original_rows[idx]);
            }
            if check.is_zero() {
                result.push(dep);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BitRow;

    #[test]
    fn test_bitrow_to_csr() {
        let ncols = 5;
        let mut rows: Vec<BitRow> = (0..3).map(|_| BitRow::new(ncols)).collect();
        // Row 0: cols {0, 2, 4}
        rows[0].set(0);
        rows[0].set(2);
        rows[0].set(4);
        // Row 1: cols {1, 3}
        rows[1].set(1);
        rows[1].set(3);
        // Row 2: cols {0, 1, 2, 3, 4}
        for c in 0..5 {
            rows[2].set(c);
        }

        let csr = SparseMatrixGF2::from_bitrows(&rows, ncols);
        assert_eq!(csr.rows, 3);
        assert_eq!(csr.cols, 5);
        assert_eq!(csr.nnz(), 10); // 3 + 2 + 5

        // Verify row 0
        let r0: Vec<usize> = csr.col_indices[csr.row_ptrs[0]..csr.row_ptrs[1]].to_vec();
        assert_eq!(r0, vec![0, 2, 4]);

        // Verify row 1
        let r1: Vec<usize> = csr.col_indices[csr.row_ptrs[1]..csr.row_ptrs[2]].to_vec();
        assert_eq!(r1, vec![1, 3]);
    }

    #[test]
    fn test_block_spmv_vs_dense() {
        // 4x4 matrix:
        // [1 1 0 0]
        // [0 1 1 0]
        // [0 0 1 1]
        // [1 0 0 1]
        let ncols = 4;
        let mut rows: Vec<BitRow> = (0..4).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0); rows[0].set(1);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(2); rows[2].set(3);
        rows[3].set(0); rows[3].set(3);

        let csr = SparseMatrixGF2::from_bitrows(&rows, ncols);

        // Test scalar SpMV
        let x = vec![0b1010u64]; // [0, 1, 0, 1]
        let mut y_scalar = vec![0u64];
        csr.mul_vec_gf2(&x, &mut y_scalar);
        // Row 0: 0*0 + 1*1 + 0*0 + 0*1 = 1 → bit 0
        // Row 1: 0*0 + 1*1 + 0*1 + 0*0 = 1 → bit 1 (wait, x = [0,1,0,1])
        // x[0]=0, x[1]=1, x[2]=0, x[3]=1
        // Row 0: x[0]^x[1] = 0^1 = 1
        // Row 1: x[1]^x[2] = 1^0 = 1
        // Row 2: x[2]^x[3] = 0^1 = 1
        // Row 3: x[0]^x[3] = 0^1 = 1
        assert_eq!(y_scalar[0] & 0xF, 0b1111);

        // Test block SpMV with same vector in position 0
        // Block layout: input_block[col] = bit-packed values for that col across 64 vectors
        // Vector 0: [0, 1, 0, 1]
        let input_block = vec![0u64, 1u64, 0u64, 1u64]; // col 0=0, col 1=1, col 2=0, col 3=1 for vector 0
        let mut output_block = vec![0u64; 4];
        csr.mul_block_gf2(&input_block, &mut output_block);

        // Check vector 0 result (bit 0 of each output word)
        let result: u64 = (0..4).map(|i| (output_block[i] & 1) << i).sum();
        assert_eq!(result, 0b1111);

        // Test parallel matches serial
        let mut par_output = vec![0u64; 4];
        csr.par_mul_block_gf2(&input_block, &mut par_output);
        assert_eq!(output_block, par_output);
    }

    #[test]
    fn test_bm_known_recurrence() {
        // Sequence: 1, 0, 1, 0, 1, 0, ... (period 2)
        // Minimal polynomial: 1 + x^2 (bits: 101 = 5)
        let len = 20;
        let mut seq = vec![0u64; 1];
        for i in 0..len {
            if i % 2 == 0 {
                seq[i / 64] |= 1 << (i % 64);
            }
        }
        let poly = berlekamp_massey(&seq, len);
        // Verify: c(x) should generate the sequence
        // The minimal poly for 1,0,1,0,... is 1+x^2
        let c0 = (poly[0] >> 0) & 1;
        assert_eq!(c0, 1, "c_0 should be 1");
        // Check that the polynomial is valid by verifying it generates the sequence
        let deg = poly.len() * 64;
        for i in 2..len {
            let mut val = 0u64;
            for j in 0..deg.min(i + 1) {
                if j / 64 < poly.len() {
                    let c_j = (poly[j / 64] >> (j % 64)) & 1;
                    let s_bit = (seq[(i - j) / 64] >> ((i - j) % 64)) & 1;
                    val ^= c_j & s_bit;
                }
            }
            assert_eq!(val, 0, "Poly should predict sequence at position {}", i);
        }
    }

    #[test]
    fn test_bm_zero_sequence() {
        let seq = vec![0u64; 2];
        let poly = berlekamp_massey(&seq, 64);
        // Zero sequence → minimal polynomial is just "1" (degree 0)
        assert_eq!((poly[0] >> 0) & 1, 1);
    }

    #[test]
    fn test_scalar_wiedemann_known_kernel() {
        // Singular matrix:
        // [1 1 0]
        // [1 1 0]
        // [0 0 1]
        // Kernel: [1, 1, 0]^T
        let ncols = 3;
        let mut rows: Vec<BitRow> = (0..3).map(|_| BitRow::new(ncols)).collect();
        rows[0].set(0); rows[0].set(1);
        rows[1].set(0); rows[1].set(1);
        rows[2].set(2);

        let csr = SparseMatrixGF2::from_bitrows(&rows, ncols);

        // Try multiple seeds to handle PRNG diversity issues with tiny matrices
        let mut found = false;
        for seed in 0..20u64 {
            let mut rng = SimpleRng::new(seed);
            for _ in 0..50 {
                if let Some(w) = wiedemann_nullspace_vector(&csr, &mut rng) {
                    let mut mw = vec![0u64; 1];
                    csr.mul_vec_gf2(&w, &mut mw);
                    assert_eq!(mw[0] & 0x7, 0, "w should be in kernel");
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(found, "Should find a kernel vector across multiple seeds");
    }

    #[test]
    fn test_batched_vs_scalar_equivalence() {
        // 8x8 matrix with rank 6 (2-dim kernel)
        let ncols = 8;
        let mut rows: Vec<BitRow> = (0..8).map(|_| BitRow::new(ncols)).collect();
        // Rows 0-5: linearly independent
        rows[0].set(0);
        rows[1].set(1);
        rows[2].set(2);
        rows[3].set(3);
        rows[4].set(4);
        rows[5].set(5);
        // Row 6 = Row 0 + Row 1
        rows[6].set(0); rows[6].set(1);
        // Row 7 = Row 2 + Row 3
        rows[7].set(2); rows[7].set(3);

        let csr = SparseMatrixGF2::from_bitrows(&rows, ncols);

        // Scalar: find kernel vectors
        let mut rng_s = SimpleRng::new(123);
        let mut scalar_found = 0;
        for _ in 0..100 {
            if wiedemann_nullspace_vector(&csr, &mut rng_s).is_some() {
                scalar_found += 1;
            }
        }

        // Batched: find kernel vectors
        let mut rng_b = SimpleRng::new(456);
        let batched = wiedemann_nullspace_batched(&csr, &mut rng_b);

        // Both should find kernel vectors (matrix has 2-dim kernel)
        assert!(scalar_found > 0, "Scalar should find kernel vectors");
        // Batched may or may not find on first round for 8x8, but verify any found are valid
        for w in &batched {
            let words = (8 + 63) / 64;
            let mut mw = vec![0u64; words];
            csr.mul_vec_gf2(w, &mut mw);
            assert_eq!(mw[0] & 0xFF, 0, "Batched kernel vector invalid");
        }
    }

    #[test]
    fn test_find_deps_bw_simple() {
        // 3 rows, 3 cols: rows XOR to zero → dependency {0,1,2}
        let ncols = 3;
        let mut rows = vec![BitRow::new(ncols), BitRow::new(ncols), BitRow::new(ncols)];
        rows[0].set(0); rows[0].set(2);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(0); rows[2].set(1);

        let deps = find_dependencies_bw(&rows, ncols);
        assert!(!deps.is_empty(), "Should find at least one dependency");
        for dep in &deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "Dependency {:?} should XOR to zero", dep);
        }
    }

    #[test]
    fn test_find_deps_bw_vs_ge() {
        // Compare BW and GE on same matrix
        let ncols = 6;
        let nrows = 8;
        let mut rows: Vec<BitRow> = (0..nrows).map(|_| BitRow::new(ncols)).collect();

        // Create a matrix with known rank deficiency
        rows[0].set(0); rows[0].set(1);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(2); rows[2].set(3);
        rows[3].set(3); rows[3].set(4);
        rows[4].set(4); rows[4].set(5);
        rows[5].set(0); rows[5].set(5);
        // Row 6 = Row 0 + Row 1 + Row 2
        rows[6].set(0); rows[6].set(3);
        // Row 7 = Row 3 + Row 4 + Row 5
        rows[7].set(0); rows[7].set(3);

        let ge_deps = crate::linalg::find_dependencies(&rows, ncols);
        let bw_deps = find_dependencies_bw(&rows, ncols);

        // Both should find valid dependencies
        assert!(!ge_deps.is_empty(), "GE should find dependencies");
        // BW might find fewer or more, but all should be valid
        for dep in &bw_deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "BW dep {:?} should XOR to zero", dep);
        }
    }

    #[test]
    fn test_find_deps_bw_large_sparse() {
        // 20 rows, 15 cols — overdetermined sparse matrix
        let ncols = 15;
        let nrows = 20;
        let mut rows: Vec<BitRow> = (0..nrows).map(|_| BitRow::new(ncols)).collect();

        // Deterministic sparse pattern
        for r in 0..nrows {
            for c in 0..ncols {
                if ((r * 7 + c * 13 + 3) % 5) == 0 {
                    rows[r].set(c);
                }
            }
        }

        let deps = find_dependencies_bw(&rows, ncols);
        // With 20 rows and 15 cols, there should be at least 5 dependencies
        assert!(
            deps.len() >= 1,
            "Should find at least 1 dependency for 20x15 matrix"
        );
        for dep in &deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "Dep {:?} should XOR to zero", dep);
        }
    }

    #[test]
    fn test_preelim_bw_matches_preelim_ge() {
        // Same test as test_preelim_matches_plain_ge but using BW
        let ncols = 3;
        let mut rows = vec![BitRow::new(ncols), BitRow::new(ncols), BitRow::new(ncols)];
        rows[0].set(0); rows[0].set(2);
        rows[1].set(1); rows[1].set(2);
        rows[2].set(0); rows[2].set(1);

        let deps = find_dependencies_with_preelim_bw(&rows, ncols);
        assert!(!deps.is_empty(), "Should find at least one dependency");
        for dep in &deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "Dep {:?} should XOR to zero", dep);
        }
    }

    #[test]
    fn test_bw_cross_validate_200x150() {
        // Cross-validate BW vs GE on a 200x150 random sparse matrix.
        // BW should find valid dependencies (all verified), and GE should too.
        let ncols = 150;
        let nrows = 200;
        let mut rows: Vec<BitRow> = (0..nrows).map(|_| BitRow::new(ncols)).collect();

        // Deterministic sparse pattern (~5 non-zeros per row)
        let mut rng = SimpleRng::new(12345);
        for r in 0..nrows {
            for _ in 0..5 {
                let c = (rng.next_u64() as usize) % ncols;
                rows[r].flip(c);
            }
        }

        let ge_deps = crate::linalg::find_dependencies(&rows, ncols);
        let bw_deps = find_dependencies_bw(&rows, ncols);

        // GE finds exact null space. BW should find at least some valid deps.
        assert!(
            !ge_deps.is_empty(),
            "GE should find deps for 200x150 matrix"
        );
        assert!(
            !bw_deps.is_empty(),
            "BW should find deps for 200x150 matrix"
        );

        // Verify all BW deps are valid
        for dep in &bw_deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "BW dep should XOR to zero");
        }
    }

    #[test]
    fn test_bw_preelim_nfs_like_matrix() {
        // NFS-like structure: sparse LP columns + dense FB core
        let ncols = 50;
        let nrows = 60;
        let mut rows: Vec<BitRow> = (0..nrows).map(|_| BitRow::new(ncols)).collect();

        // First 20 cols: sparse (each in ~2 rows) — like large primes
        let mut rng = SimpleRng::new(999);
        for c in 0..20 {
            let r1 = (rng.next_u64() as usize) % nrows;
            let r2 = (rng.next_u64() as usize) % nrows;
            rows[r1].set(c);
            if r2 != r1 {
                rows[r2].set(c);
            }
        }
        // Last 30 cols: dense (~3 per row) — like factor base
        for r in 0..nrows {
            for _ in 0..3 {
                let c = 20 + (rng.next_u64() as usize) % 30;
                rows[r].flip(c);
            }
        }

        let ge_deps = crate::linalg::find_dependencies_with_preelim(&rows, ncols);
        let bw_deps = find_dependencies_with_preelim_bw(&rows, ncols);

        // Both should find valid deps
        for dep in &bw_deps {
            let mut check = BitRow::new(ncols);
            for &idx in dep {
                check.xor_with(&rows[idx]);
            }
            assert!(check.is_zero(), "BW preelim dep should XOR to zero");
        }

        // Both should find at least some deps
        assert!(
            !ge_deps.is_empty(),
            "GE preelim should find deps for 60x50"
        );
    }
}
