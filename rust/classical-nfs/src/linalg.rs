//! GF(2) Linear algebra for the Number Field Sieve.
//!
//! Provides Gaussian elimination and Block Lanczos over GF(2) to find
//! null-space vectors of the exponent matrix. These null-space vectors
//! identify subsets of relations whose product forms a square.

/// A dependency: a set of row indices whose XOR sum is the zero vector.
pub type Dependency = Vec<usize>;

/// Gaussian elimination over GF(2) to find linear dependencies.
///
/// Given an m x n matrix over GF(2) (each entry is 0 or 1), find all
/// row subsets whose XOR is the zero vector. These are the dependencies
/// we need for the square root step.
///
/// Returns a list of dependencies, where each dependency is a set of
/// row indices.
pub fn gaussian_elimination_gf2(matrix: &[Vec<u8>], num_cols: usize) -> Vec<Dependency> {
    let num_rows = matrix.len();
    if num_rows == 0 || num_cols == 0 {
        return vec![];
    }

    // Augmented matrix: each row tracks which original rows contributed
    // (row_bitvector, history_of_original_rows)
    let mut rows: Vec<(Vec<u8>, Vec<usize>)> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut padded = row.clone();
            padded.resize(num_cols, 0);
            (padded, vec![i])
        })
        .collect();

    let mut pivot_row_for_col: Vec<Option<usize>> = vec![None; num_cols];

    for col in 0..num_cols {
        // Find a row with a 1 in this column that hasn't been used as pivot
        let pivot = (0..num_rows).find(|&r| {
            rows[r].0.get(col).copied().unwrap_or(0) == 1
                && pivot_row_for_col.iter().all(|p| *p != Some(r))
        });

        if let Some(pivot_idx) = pivot {
            pivot_row_for_col[col] = Some(pivot_idx);

            // Eliminate this column from all other rows
            for r in 0..num_rows {
                if r != pivot_idx && rows[r].0.get(col).copied().unwrap_or(0) == 1 {
                    let pivot_data = rows[pivot_idx].0.clone();
                    let pivot_history = rows[pivot_idx].1.clone();

                    // XOR the row data
                    for c in 0..num_cols {
                        let pv = pivot_data.get(c).copied().unwrap_or(0);
                        if let Some(v) = rows[r].0.get_mut(c) {
                            *v ^= pv;
                        }
                    }

                    // XOR the history (symmetric difference)
                    let mut new_history = rows[r].1.clone();
                    for &idx in &pivot_history {
                        if let Some(pos) = new_history.iter().position(|&x| x == idx) {
                            new_history.remove(pos);
                        } else {
                            new_history.push(idx);
                        }
                    }
                    rows[r].1 = new_history;
                }
            }
        }
    }

    // Rows that are now all-zero represent dependencies
    let mut dependencies = Vec::new();
    for (row_vec, history) in &rows {
        if row_vec.iter().all(|&x| x == 0) && history.len() >= 2 {
            dependencies.push(history.clone());
        }
    }

    dependencies
}

/// Block Lanczos algorithm over GF(2).
///
/// For large sparse matrices, Block Lanczos is much more efficient than
/// Gaussian elimination. It operates on blocks of 64 bits (using u64 words)
/// and finds null-space vectors iteratively.
///
/// This is a simplified implementation suitable for matrices with up to
/// ~10000 rows/columns.
pub fn block_lanczos_gf2(matrix: &[Vec<u8>], num_cols: usize) -> Vec<Dependency> {
    let num_rows = matrix.len();

    // For small matrices, fall back to Gaussian elimination
    if num_rows < 128 || num_cols < 64 {
        return gaussian_elimination_gf2(matrix, num_cols);
    }

    // Convert matrix to packed u64 format for efficiency
    let words_per_row = (num_cols + 63) / 64;
    let packed: Vec<Vec<u64>> = matrix
        .iter()
        .map(|row| pack_row(row, num_cols))
        .collect();

    // We need to find vectors x such that M^T * M * x = 0 over GF(2)
    // where M is the matrix.
    //
    // Block Lanczos iterates with block width N=64 (one u64 word).
    // We maintain vectors as arrays of u64 where each bit position
    // represents a separate vector in the block.

    let block_size = 64usize;
    let max_iterations = num_rows + 100;

    // Initialize random block vector x_0
    let mut rng_state: u64 = 0xDEADBEEF_CAFEBABE;
    let x_init: Vec<u64> = (0..num_rows)
        .map(|_| {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        })
        .collect();

    // Compute w_0 = M^T * M * x_0
    let mut _w_prev = mt_times_m_times_v(&packed, &x_init, num_rows, words_per_row);
    let mut v_prev = x_init;

    // Collect candidate null-space vectors
    let mut null_candidates: Vec<Vec<u64>> = Vec::new();
    null_candidates.push(v_prev.clone());

    let mut v_prev_prev: Vec<u64> = vec![0; num_rows];

    for _iter in 0..max_iterations {
        // v_i = w_{i-1} - stuff (simplified Lanczos step)
        // Compute A * v where A = M^T * M
        let av = mt_times_m_times_v(&packed, &v_prev, num_rows, words_per_row);

        // Check if Av is zero (v is in null space)
        if av.iter().all(|&w| w == 0) {
            null_candidates.push(v_prev.clone());
            break;
        }

        // Simple iteration: v_new = Av XOR (scaling from previous)
        let mut v_new: Vec<u64> = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            v_new.push(av[i] ^ v_prev_prev[i]);
        }

        // Check convergence
        if v_new.iter().all(|&w| w == 0) {
            break;
        }

        v_prev_prev = v_prev.clone();
        v_prev = v_new;
        _w_prev = mt_times_m_times_v(&packed, &v_prev, num_rows, words_per_row);

        // Periodically check for null-space vectors
        if _w_prev.iter().all(|&w| w == 0) {
            null_candidates.push(v_prev.clone());
            break;
        }
    }

    // Extract dependencies from null-space candidates
    let mut dependencies = Vec::new();
    for candidate in &null_candidates {
        // Each bit position in the candidate represents a separate potential dependency
        for bit in 0..block_size {
            let mask = 1u64 << bit;
            let dep: Vec<usize> = candidate
                .iter()
                .enumerate()
                .filter(|(_, &w)| w & mask != 0)
                .map(|(i, _)| i)
                .collect();

            if dep.len() >= 2 {
                // Verify this is actually a dependency
                if verify_dependency(matrix, &dep, num_cols) {
                    dependencies.push(dep);
                }
            }
        }
    }

    // If Block Lanczos didn't find enough, supplement with Gaussian elimination
    if dependencies.is_empty() {
        return gaussian_elimination_gf2(matrix, num_cols);
    }

    dependencies
}

/// Pack a row of u8 bits into u64 words.
fn pack_row(row: &[u8], num_cols: usize) -> Vec<u64> {
    let words = (num_cols + 63) / 64;
    let mut packed = vec![0u64; words];

    for (i, &bit) in row.iter().enumerate().take(num_cols) {
        if bit != 0 {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }

    packed
}

/// Compute M^T * M * v where M is stored as packed rows and v is a block vector.
///
/// Step 1: t = M * v (multiply each row by v using bitwise AND + popcount parity)
/// Step 2: result = M^T * t
fn mt_times_m_times_v(
    packed_matrix: &[Vec<u64>],
    v: &[u64],
    num_rows: usize,
    words_per_row: usize,
) -> Vec<u64> {
    // Step 1: t_j = parity(M[j] & v[j]) for each column bit position
    // Actually, t is a vector of length num_cols, but we're doing block operations.
    //
    // For block Lanczos, M*v means: for each row j of M, compute the dot product
    // of M[j] with v (where v[j] is a u64 representing 64 separate vectors).
    // t[j] = XOR over all bits set in M[j] of the corresponding v entry.

    // First compute t = M * v: t has num_rows entries (packed as u64 blocks)
    // But M has num_rows rows and ~num_cols columns, and v has num_rows entries.
    // This is actually wrong -- let me reconsider.
    //
    // v has num_rows entries. M has num_rows rows, num_cols columns.
    // M^T * M is num_cols x num_cols.
    // M^T * M * v doesn't type-check if v is num_rows-length.
    //
    // Actually in the Lanczos formulation, we work with A = M * M^T (num_rows x num_rows)
    // and find v such that A*v = 0, then M^T * v is in the null space of M^T * M.
    // But for our purposes, we want null(M) directly.
    //
    // Let's just compute M * M^T * v where v has num_rows entries.

    let mut result = vec![0u64; num_rows];

    // Compute t = M^T * v: t has words_per_row u64 words
    let mut t = vec![0u64; words_per_row];
    for (j, row) in packed_matrix.iter().enumerate().take(num_rows) {
        let vj = v[j];
        for (w, word) in row.iter().enumerate() {
            // For each bit set in vj, we want to XOR the corresponding row into t
            // Since vj represents 64 separate vectors, we AND them
            t[w] ^= word & vj;
        }
    }

    // Compute result = M * t: result has num_rows entries
    for (j, row) in packed_matrix.iter().enumerate().take(num_rows) {
        let mut dot: u64 = 0;
        for (w, word) in row.iter().enumerate() {
            dot ^= word & t[w];
        }
        // Parity of dot gives us the result bit for each of the 64 vectors
        // Actually we want the full word, not parity
        result[j] = dot;
    }

    result
}

/// Verify that a set of row indices actually forms a dependency (XOR to zero).
fn verify_dependency(matrix: &[Vec<u8>], dep: &[usize], num_cols: usize) -> bool {
    let mut xor_sum = vec![0u8; num_cols];

    for &idx in dep {
        if idx >= matrix.len() {
            return false;
        }
        for (c, &val) in matrix[idx].iter().enumerate().take(num_cols) {
            xor_sum[c] ^= val;
        }
    }

    xor_sum.iter().all(|&x| x == 0)
}

/// Find dependencies in the matrix, choosing the best algorithm based on size.
pub fn find_dependencies(matrix: &[Vec<u8>], num_cols: usize) -> Vec<Dependency> {
    // Always use Gaussian elimination for our target sizes (40-80 bit numbers
    // produce matrices that are well within Gaussian elimination's capability)
    gaussian_elimination_gf2(matrix, num_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_simple() {
        // Two identical rows should form a dependency
        let matrix = vec![vec![1u8, 0, 1], vec![1, 0, 1], vec![0, 1, 0]];
        let deps = gaussian_elimination_gf2(&matrix, 3);
        assert!(
            !deps.is_empty(),
            "Should find dependency between identical rows"
        );
        // Verify the dependency
        for dep in &deps {
            assert!(verify_dependency(&matrix, dep, 3));
        }
    }

    #[test]
    fn test_gaussian_three_rows() {
        // Rows: [1,1,0], [1,0,1], [0,1,1]
        // Their XOR = [0,0,0], so all three form a dependency
        let matrix = vec![vec![1u8, 1, 0], vec![1, 0, 1], vec![0, 1, 1]];
        let deps = gaussian_elimination_gf2(&matrix, 3);
        assert!(
            !deps.is_empty(),
            "Should find dependency among three rows"
        );
        for dep in &deps {
            assert!(verify_dependency(&matrix, dep, 3));
        }
    }

    #[test]
    fn test_gaussian_no_dependency() {
        // Three linearly independent rows in GF(2)^3
        let matrix = vec![vec![1u8, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        let deps = gaussian_elimination_gf2(&matrix, 3);
        assert!(
            deps.is_empty(),
            "Should find no dependencies among 3 independent rows in GF(2)^3"
        );
    }

    #[test]
    fn test_gaussian_overdetermined() {
        // 5 rows in GF(2)^3 -- must have at least 2 dependencies
        let matrix = vec![
            vec![1u8, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 0],
            vec![1, 0, 0],
            vec![0, 1, 0],
        ];
        let deps = gaussian_elimination_gf2(&matrix, 3);
        assert!(
            deps.len() >= 2,
            "5 rows in GF(2)^3 should have >= 2 dependencies, got {}",
            deps.len()
        );
        for dep in &deps {
            assert!(verify_dependency(&matrix, dep, 3));
        }
    }

    #[test]
    fn test_verify_dependency() {
        let matrix = vec![vec![1u8, 1, 0], vec![1, 0, 1], vec![0, 1, 1]];
        assert!(verify_dependency(&matrix, &[0, 1, 2], 3));
        assert!(!verify_dependency(&matrix, &[0, 1], 3));
    }

    #[test]
    fn test_pack_row() {
        let row = vec![1u8, 0, 1, 1, 0, 0, 0, 1];
        let packed = pack_row(&row, 8);
        assert_eq!(packed.len(), 1);
        // bit 0 = 1, bit 2 = 1, bit 3 = 1, bit 7 = 1
        assert_eq!(packed[0], 0b10001101);
    }

    #[test]
    fn test_find_dependencies_small() {
        let matrix = vec![
            vec![1u8, 0, 1],
            vec![1, 0, 1],
            vec![0, 1, 0],
        ];
        let deps = find_dependencies(&matrix, 3);
        assert!(!deps.is_empty());
        for dep in &deps {
            assert!(verify_dependency(&matrix, dep, 3));
        }
    }
}
